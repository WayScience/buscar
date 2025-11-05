"""
Functions for assessing cellular heterogeneity via clustering, including Optuna-based
parameter optimization to maximize silhouette score. Uses scanpy for dimensionality
reduction and clustering (Louvain/Leiden), performed per treatment group.
"""

import gc
import pathlib
import shutil
import tempfile
from typing import Any, Literal

import optuna
import polars as pl
import scanpy as sc
from beartype import beartype
from joblib import Parallel, delayed
from sklearn.metrics import silhouette_score

from .checks import check_for_nans
from .io_utils import load_profiles
from .validator import _validate_param_grid


@beartype
def calculate_silhouette_score(
    clustered_profiles: pl.DataFrame,
    morph_features: list[str] | pl.Series,
) -> float:
    """Calculate silhouette score for clustered profiles.

    Parameters
    ----------
    clustered_profiles : pl.DataFrame
        DataFrame with cluster assignments in 'Metadata_cluster_id' column.
    morph_features : list[str] | pl.Series
        Morphological features used for clustering.

    Returns
    -------
    float
        Silhouette score (-1 to 1, higher is better). Returns -1 if calculation fails.
    """
    try:
        # Ensure morph_features is a list
        if isinstance(morph_features, pl.Series):
            morph_features = morph_features.to_list()

        # Filter out insufficient/failed clusters
        # Cast to String first since str methods don't work on Categorical
        valid_mask = ~clustered_profiles["Metadata_cluster_id"].cast(
            pl.String
        ).str.contains("insufficient_cells") & ~clustered_profiles[
            "Metadata_cluster_id"
        ].cast(pl.String).str.contains("cluster_failed")
        valid_profiles = clustered_profiles.filter(valid_mask)

        # Need at least 2 clusters for silhouette score
        n_clusters = valid_profiles["Metadata_cluster_id"].n_unique()
        if n_clusters < 2:
            return -1.0

        X = valid_profiles.select(morph_features).to_numpy()
        labels = valid_profiles["Metadata_cluster_id"].to_numpy()

        return float(silhouette_score(X, labels))
    except Exception as e:
        print(f"Warning: Silhouette score calculation failed: {e}")
        return -1.0


@beartype
def cluster_profiles(
    profiles: pl.DataFrame | str | pathlib.Path,
    meta_features: list[str] | pl.Series,
    morph_features: list[str] | pl.Series,
    treatment_col: str,
    cluster_method: Literal["louvain", "leiden"] = "leiden",
    cluster_resolution: float = 1.0,
    n_neighbors: int = 15,
    neighbor_distance_metric: Literal["cosine", "euclidean", "manhattan"] = "euclidean",
    min_cells_per_treatment: int = 10,
    seed: int = 0,
    convert_to_f32: bool = False,
) -> pl.DataFrame:
    """Cluster single-cell morphological profiles to identify cellular
    subpopulations.

    This function performs graph-based clustering on single-cell profiles to
    identify distinct cellular subpopulations based on morphological similarity.
    This is useful for quantifying cellular heterogeneity within a treatment
    condition, where cells may respond differently to the same perturbation due
    to cell cycle state, spatial context, or stochastic variation in cellular
    responses.

    The clustering workflow:
    1. Constructs a k-nearest neighbor graph based on morphological feature similarity
    2. Applies community detection (Louvain or Leiden) to identify cell clusters
    3. Labels each cell with its cluster assignment
    4. Calculates cluster statistics (size, proportion of total population)

    Clusters represent morphologically similar cells that may share similar
    biological states or responses. Multiple clusters within a single treatment
    suggest heterogeneous cellular responses, while a single dominant cluster
    indicates a more uniform population response.

    Parameters
    ----------
    profiles : pl.DataFrame
        Single-cell profiles containing morphological measurements. Must contain
        cells from exactly one treatment or experimental condition (raises error
        if multiple treatments found).
    meta_features : list[str]
        Column names for metadata fields (e.g., 'Metadata_Well', 'Metadata_Plate')
        to preserve in the output.
    morph_features : list[str]
        Column names for morphological features (e.g., cell area, intensity, texture)
        used to compute cellular similarity and perform clustering.
    treatment_col : str
        Column name identifying the treatment/condition. Used to label clusters and
        verify single-treatment input.
    cluster_method : Literal["louvain", "leiden"], default "leiden"
        Graph-based clustering algorithm. Leiden is generally preferred for better
        cluster quality and speed.
    cluster_resolution : float, default 1.0
        Controls cluster granularity. Higher values yield more clusters (finer
        subpopulations), lower values yield fewer clusters (broader groupings).
    n_neighbors : int, default 15
        Number of nearest neighbors for graph construction. Smaller values emphasize
        local structure; larger values capture broader patterns. Will be automatically
        adjusted if greater than (n_cells - 1).
    neighbor_distance_metric : Literal["cosine", "euclidean", "manhattan"], default "euclidean"
        Distance metric for computing cell-cell similarity. Euclidean is standard;
        cosine emphasizes correlation over magnitude.
    min_cells_per_treatment : int, default 10
        Minimum cells required for clustering. Samples with fewer cells are labeled
        as "insufficient_cells" without clustering.
    seed : int, default 0
        Random seed for reproducible clustering results.
    convert_to_f32 : bool, default False
        If True, converts Float64 columns to Float32 when loading profiles to
        save memory. Only used when profiles is a file path.

    Returns
    -------
    pl.DataFrame
        Original profiles with four additional columns:
        - **Metadata_cluster_id**: Cluster assignment as "{treatment}_{method}_{id}",
          or "{treatment}_insufficient_cells" for small samples, or
          "{treatment}_cluster_failed" if clustering encounters errors.
        - **Metadata_cluster_n_cells**: Number of cells in each cluster.
        - **Metadata_treatment_n_cells**: Total cells in the treatment.
        - **Metadata_cluster_ratio**: Proportion of treatment cells in each cluster
          (0-1 scale).

    Raises
    ------
    ValueError
        If treatment_col is not found in profiles, if morphological features
        contain NaN or infinite values, or if profiles contain multiple treatments.

    Notes
    -----
    - Input profiles must contain exactly one treatment/condition
    - Features should be properly normalized before clustering (e.g., z-scored)
    - Consider PCA dimensionality reduction before clustering for high-dimensional data
    - Cluster resolution tuning is recommended; use `optimized_clustering()` for
      automated parameter selection
    """
    # If a path is provided, load the profiles
    if isinstance(profiles, (str, pathlib.Path)):
        profiles = load_profiles(profiles, convert_to_f32=convert_to_f32)
    if treatment_col not in profiles.columns:
        raise ValueError(f"treatment_col '{treatment_col}' not found in profiles.")

    # Check for NaN/Inf in morph_features
    # this will raise ValueError if any are found
    check_for_nans(profiles, morph_features)

    # Convert to AnnData and add treatment info to .obs
    obs_df = profiles.select(meta_features).to_pandas()
    obs_df.index = obs_df.index.astype(str)
    adata = sc.AnnData(
        X=profiles.select(morph_features).to_numpy(),
        obs=obs_df,
    )

    # Ensure the treatment column is categorical
    if adata.obs[treatment_col].dtype != "category":
        adata.obs[treatment_col] = adata.obs[treatment_col].astype("category")

    # Initialize cluster labels
    all_cluster_labels = [""] * len(profiles)

    # Get the treatment name (require single treatment in input)
    unique_treatments = profiles[treatment_col].unique()
    if len(unique_treatments) > 1:
        raise ValueError(
            f"Expected single treatment per file, found "
            f"{len(unique_treatments)}: {unique_treatments.to_list()}"
        )
    treatment = profiles[treatment_col][0]

    # Check if there are enough cells
    n_cells = len(profiles)
    if n_cells < min_cells_per_treatment:
        # Label all as insufficient
        all_cluster_labels = [f"{treatment}_insufficient_cells"] * n_cells
    else:
        # Compute neighbors
        sc.pp.neighbors(
            adata=adata,
            n_neighbors=min(n_neighbors, n_cells - 1),  # Ensure n_neighbors < n_cells
            use_rep="X",
            random_state=seed,
            metric=neighbor_distance_metric,
        )

        try:
            # Apply clustering
            if cluster_method == "louvain":
                sc.tl.louvain(
                    adata=adata,
                    resolution=cluster_resolution,
                    random_state=seed,
                    key_added="cluster",
                )
            elif cluster_method == "leiden":
                sc.tl.leiden(
                    adata=adata,
                    resolution=cluster_resolution,
                    random_state=seed,
                    key_added="cluster",
                )

            # Extract cluster labels
            cluster_labels = adata.obs["cluster"].values

            # Create cluster IDs with treatment prefix
            all_cluster_labels = [
                f"{treatment}_{cluster_method}_{label}" for label in cluster_labels
            ]

        except Exception as e:
            # Handle clustering failures gracefully
            all_cluster_labels = [f"{treatment}_cluster_failed"] * n_cells
            print(f"Warning: Clustering failed for treatment {treatment}: {e}")

    # Add cluster labels and statistics to original DataFrame
    result_df = profiles.with_columns(
        pl.Series(name="Metadata_cluster_id", values=all_cluster_labels).cast(
            pl.Categorical
        )
    )

    # Add cluster cell counts
    result_df = result_df.with_columns(
        pl.count().over("Metadata_cluster_id").alias("Metadata_cluster_n_cells")
    )

    # Add total cells per treatment
    result_df = result_df.with_columns(
        pl.count().over(treatment_col).alias("Metadata_treatment_n_cells")
    )

    # Calculate cluster ratio as a proportion
    result_df = result_df.with_columns(
        (
            pl.col("Metadata_cluster_n_cells") / pl.col("Metadata_treatment_n_cells")
        ).alias("Metadata_cluster_ratio")
    )

    return result_df


# ---------------------
# Optuna optimization functions
# ---------------------
@beartype
def _optimize_single_profile(
    profile_path: str | pathlib.Path,
    meta_features: list[str],
    morph_features: list[str],
    treatment_col: str,
    param_grid: dict[str, Any],
    n_trials: int,
    seed: int,
    study_name: str,
) -> tuple[pl.DataFrame, dict[str, Any], float]:
    """Optimize clustering for a single treatment profile.

    Returns
    -------
    tuple[pl.DataFrame, dict[str, Any], float]
        Best clustered profile, best parameters, and best silhouette score.
    """

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function to maximize silhouette score."""
        params = {}

        for param_name, param_config in param_grid.items():
            if param_config["type"] == "float":
                log_scale = param_config.get("log", False)
                params[param_name] = trial.suggest_float(
                    param_name, param_config["low"], param_config["high"], log=log_scale
                )
            elif param_config["type"] == "int":
                log_scale = param_config.get("log", False)
                params[param_name] = trial.suggest_int(
                    param_name, param_config["low"], param_config["high"], log=log_scale
                )
            elif param_config["type"] == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config["choices"]
                )

        # Add fixed parameters
        params["seed"] = seed

        try:
            # Run clustering with current parameters
            clustered = cluster_profiles(
                profiles=profile_path,
                meta_features=meta_features,
                morph_features=morph_features,
                treatment_col=treatment_col,
                **params,
            )

            # Calculate silhouette score
            score = calculate_silhouette_score(
                clustered_profiles=clustered,
                morph_features=morph_features,
            )

            return score

        except Exception as e:
            print(f"Trial failed for {profile_path}: {e}")
            return -1.0

    # Create and run study
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
    )

    # Run final clustering with best parameters
    best_clustered = cluster_profiles(
        profiles=profile_path,
        meta_features=meta_features,
        morph_features=morph_features,
        treatment_col=treatment_col,
        **study.best_params,
        seed=seed,
    )

    return best_clustered, study.best_params, study.best_value


@beartype
def optimized_clustering(
    profiles: pl.DataFrame | list[str | pathlib.Path],
    meta_features: list[str],
    morph_features: list[str],
    treatment_col: str,
    param_grid: dict[str, Any],
    n_trials: int = 100,
    seed: int = 0,
    n_jobs: int = 1,
    study_name: str | None = None,
) -> tuple[pl.DataFrame, dict[str, dict[str, Any]]]:
    """Optimize clustering parameters across treatments using parallel processing.

    This function performs hyperparameter optimization for clustering using Optuna,
    with parallel processing across multiple treatment groups. Each treatment is
    optimized independently to find parameters that maximize the silhouette score
    for that specific treatment.

    If a DataFrame is provided, it will be automatically split by treatment, with
    each treatment processed in parallel. If a list of file paths is provided,
    each file should contain profiles for a single treatment.

    The silhouette score quantifies how well-separated the clusters are, with higher
    values indicating better-defined clusters. By optimizing clustering parameters
    for each treatment, we can better capture cellular heterogeneity specific to
    each condition.

    Parameters
    ----------
    profiles : pl.DataFrame | list[str | pathlib.Path]
        Either a DataFrame containing multiple treatments (will be split automatically),
        or a list of file paths where each file contains a single treatment's profiles.
    meta_features : list[str]
        Column names for metadata fields to preserve in output.
    morph_features : list[str]
        Column names for morphological features used for clustering.
    treatment_col : str
        Column name identifying treatments. Used to split DataFrame if provided.
    param_grid : dict[str, Any]
        Parameter search space. Each key is a parameter name, each value is a dict
        with 'type' and range info (e.g., {"type": "float", "low": 0.1, "high": 2.0}).
    n_trials : int, default 100
        Number of optimization trials per treatment.
    seed : int, default 0
        Random seed for reproducibility.
    n_jobs : int, default 1
        Number of parallel jobs. Use -1 for all available cores.
    study_name : str | None, default None
        Base name for Optuna studies. Treatment names will be appended.

    Returns
    -------
    tuple[pl.DataFrame, dict[str, dict[str, Any]]]
        - Concatenated DataFrame with optimized clustering for all treatments
        - Dictionary mapping treatment names to their best parameters and scores

    Raises
    ------
    ValueError
        If param_grid contains invalid parameter specifications.

    Notes
    -----
    - Temporary files are created when splitting a DataFrame, then cleaned up automatically
    - Each treatment is optimized independently with its own Optuna study
    - Progress is printed for each treatment showing best silhouette score and parameters
    """
    # Validate parameter grid
    _validate_param_grid(param_grid)

    # Track if we created a temporary directory that needs cleanup
    temp_dir_path = None

    # If DataFrame is passed, split by treatment and save to temp files
    # replace profiles with list of file paths to those temp files
    if isinstance(profiles, pl.DataFrame):
        if "Metadata_cell_id" not in meta_features:
            raise ValueError(
                "meta_features must include 'Metadata_cell_id' when passing a "
                "DataFrame."
            )
        temp_dir_path = pathlib.Path(tempfile.mkdtemp())
        treatment_groups = profiles.group_by(treatment_col)
        profile_paths = []

        for treatment, group_df in treatment_groups:
            temp_file = (temp_dir_path / f"{treatment}_profiles.parquet").resolve()
            group_df.write_parquet(temp_file)
            profile_paths.append(temp_file)

        profiles = profile_paths

        # free up memory generated by the grouping
        del treatment_groups, group_df
        gc.collect()

    try:
        # setting study name
        if study_name is None:
            study_name = f"cluster_optimization_{seed}"

        # Prepare parallel optimization tasks for each treatment profile
        # Each task is a delayed call to optimize clustering for a single treatment
        tasks = []
        for profile_path in profiles:
            treatment_name = pathlib.Path(profile_path).stem.replace("_profiles", "")
            task_study_name = f"{study_name}_{treatment_name}"

            # generate the delayed function call
            tasks.append(
                delayed(_optimize_single_profile)(
                    profile_path=profile_path,
                    meta_features=meta_features,
                    morph_features=morph_features,
                    treatment_col=treatment_col,
                    param_grid=param_grid,
                    n_trials=n_trials,
                    seed=seed,
                    study_name=task_study_name,
                )
            )

        # Run optimization in parallel for each treatment group.
        # This executes the delayed optimization tasks concurrently using joblib's Parallel,
        # which distributes the work across multiple CPU cores for efficiency.
        print(
            f"Optimizing clustering for {len(tasks)} treatment(s) with {n_jobs} job(s)..."
        )
        results = Parallel(n_jobs=n_jobs, backend="loky")(tasks)

        # Collect results
        all_clustered = []
        best_params_per_treatment = {}

        for profile_path, (clustered_df, best_params, best_score) in zip(
            profiles, results
        ):
            treatment_name = pathlib.Path(profile_path).stem.replace("_profiles", "")

            all_clustered.append(clustered_df)
            best_params_per_treatment[treatment_name] = {
                "params": best_params,
                "silhouette_score": best_score,
            }

            print(
                f"  {treatment_name}: silhouette={best_score:.3f}, params={best_params}"
            )

        # Concatenate all results
        final_df = pl.concat(all_clustered, how="vertical")

        return final_df, best_params_per_treatment

    finally:
        # once the clustering is finished, we clean the temp directory if it was
        # created
        if temp_dir_path is not None and temp_dir_path.exists():
            shutil.rmtree(temp_dir_path)
