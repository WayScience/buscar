"""
Functions for assessing cellular heterogeneity via clustering, including Optuna-based
parameter optimization to maximize silhouette score. Uses scanpy for dimensionality
reduction and clustering (Louvain/Leiden), performed per treatment group.
"""

from typing import Any, Literal

import numpy as np
import optuna
import polars as pl
import scanpy as sc
from beartype import beartype
from sklearn.metrics import silhouette_score

from .validator import _validate_param_grid


def calculate_mean_silhouette_score(
    clustered_profiles: pl.DataFrame,
    morph_features: list[str] | pl.Series,
    treatment_col: str,
) -> float:
    """Calculate mean silhouette score across all treatments in clustered profiles.

    This function computes the silhouette score for each treatment group separately
    and returns the mean score across all treatments. Treatments with too few cells
    or only one cluster are skipped. If no valid scores can be computed, returns -1.0.

    Parameters
    ----------
    clustered_profiles : pl.DataFrame
        DataFrame containing clustered single-cell profiles with a "Metadata_cluster_id"
        column indicating cluster assignments.
    morph_features : list[str] | pl.Series
        List or Series of column names representing morphological features to use for
        silhouette score calculation.
    treatment_col : str
        Column name indicating treatment groups.

    Returns
    -------
    float
        Mean silhouette score across all valid treatments. Returns -1.0 if no valid
        scores can be computed (e.g., all treatments have only one cluster or too
        few cells).

    Notes
    -----
    The silhouette score measures how similar a cell is to its own cluster compared
    to other clusters. Scores range from -1 to 1:
    - Near +1: Cell is well-matched to its cluster
    - Near 0: Cell is on the border between clusters
    - Near -1: Cell may be assigned to the wrong cluster

    Examples
    --------
    >>> clustered_df = cluster_profiles(profiles, meta_features, morph_features, "treatment")
    >>> score = calculate_mean_silhouette_score(clustered_df, morph_features, "treatment")
    >>> print(f"Mean silhouette score: {score:.3f}")
    Mean silhouette score: 0.342
    """
    silhouette_scores = []

    for treatment in clustered_profiles.get_column(treatment_col).unique().to_list():
        treatment_mask = clustered_profiles.get_column(treatment_col) == treatment
        treatment_data = clustered_profiles.filter(treatment_mask)

        # Skip treatments with too few cells or only one cluster
        if len(treatment_data) < 2:
            continue

        cluster_labels = treatment_data.get_column("Metadata_cluster_id").to_numpy()
        unique_clusters = np.unique(cluster_labels)

        # Skip if only one cluster (silhouette score undefined)
        if len(unique_clusters) < 2:
            continue

        # Get the feature matrix for this treatment
        features_matrix = treatment_data.select(morph_features).to_numpy()

        # Calculate silhouette score
        score = silhouette_score(features_matrix, cluster_labels)
        silhouette_scores.append(score)

    # Return mean silhouette score across all treatments
    if silhouette_scores:
        return np.mean(silhouette_scores)
    else:
        # If no valid scores, return a very low score to penalize this configuration
        return -1.0


@beartype
def cluster_profiles(
    profiles: pl.DataFrame,
    meta_features: list[str] | pl.Series,
    morph_features: list[str] | pl.Series,
    treatment_col: str,
    cluster_method: Literal["louvain", "leiden"] = "leiden",
    cluster_resolution: float = 1.0,
    dim_reduction: Literal["PCA", "raw"] = "PCA",
    n_neighbors: int = 15,
    neighbor_distance_metric: Literal["cosine", "euclidean", "manhattan"] = "euclidean",
    pca_variance_explained: float = 0.95,
    pca_n_components_to_capture_variance: int = 200,
    pca_svd_solver: Literal["arpack", "randomized"] = "randomized",
    seed: int = 0,
) -> pl.DataFrame:
    """Cluster single-cell profiles using a dimensionality reduction and clustering pipeline.

    This function performs clustering on single-cell morphological profiles by first applying
    dimensionality reduction (PCA or raw data) and then clustering using
    Louvain or Leiden algorithms. Clustering is performed per treatment group defined by
    meta_features, with dynamic adjustment of neighbors based on the number of cells in each group.

    Keep in mind that this function assumes that you have normalized your data prior to
    using this approach.


    Pipeline for dim_reduction="PCA":
    1. Run PCA with up to 100 components (or fewer based on data constraints).
    2. Determine the number of PCs that explain at least pca_variance_explained of the
    variance.
    4. Compute neighbors in PCA space.
    5. Apply clustering (Louvain or Leiden) in UMAP space per treatment group.

    For dim_reduction="raw", neighbors are computed directly on raw data, and clustering
    is applied per treatment group.

    Parameters
    ----------
    profiles : pl.DataFrame
        DataFrame containing single-cell profiles with morphological features and metadata.
    meta_features : list[str] | pl.Series
        List or Series of column names used to group profiles into treatment groups for
        per-group clustering.
    morph_features : list[str] | pl.Series
        List or Series of column names representing morphological features to use for
        clustering.
    treatment_col : str, default
        Column name in profiles indicating treatment (used for labeling clusters).
    cluster_method : Literal["louvain", "leiden"], default "louvain"
        Clustering algorithm to use: "louvain" or "leiden".
    cluster_resolution : float, default 1.0
        Resolution parameter for clustering (higher values lead to more clusters).
    dim_reduction : Literal["PCA", "raw"], default "PCA"
        Dimensionality reduction method: "PCA" for PCA->UMAP pipeline, "raw" for direct
        use of raw data.
    n_neighbors : int, default 15
        Maximum number of neighbors for neighbor graph construction.
    neighbor_distance_metric : Literal["cosine", "euclidean", "manhattan"], default
    "euclidean"
        Distance metric to use when constructing the neighbor graph.
        - For PCA or UMAP-reduced spaces, "euclidean" or "manhattan" are recommended.
        - For clustering directly on raw feature space, "cosine" is often preferred.
    pca_variance_explained : float, default 0.95
        Fraction of variance to be explained by selected PCs (must be between 0 and 1).
    pca_n_components_to_capture_variance : int, default 200
        Maximum number of PCA components to compute when capturing variance.
    pca_svd_solver : Literal["arpack", "randomized"], default "randomized"
        SVD solver is the underlying algorithm used to compute the principal components
    seed : int, default 0
        Random seed for reproducibility.

    Returns
    -------
    pl.DataFrame
        Original profiles DataFrame with an additional column "Metadata_cluster_id"
        containing cluster labels as categorical values, prefixed by treatment
        (e.g., "treatment_0").

    Raises
    ------
    ValueError
        If pca_variance_explained is not between 0 and 1.
    """

    # Validation
    if not (0 < pca_variance_explained <= 1):
        raise ValueError("pca_variance_explained must be between 0 and 1")

    # 1. Convert to AnnData and add treatment info to .obs
    obs_df = profiles.select(meta_features).to_pandas()
    obs_df.index = obs_df.index.astype(str)

    adata = sc.AnnData(
        X=profiles.select(morph_features).to_numpy(),
        obs=obs_df,
    )

    # Ensure the treatment column is categorical
    if adata.obs[treatment_col].dtype != "category":
        adata.obs[treatment_col] = adata.obs[treatment_col].astype("category")

    if dim_reduction == "PCA":
        # 1. Run PCA with enough components to capture variance
        sc.pp.pca(
            adata,
            n_comps=pca_n_components_to_capture_variance,
            svd_solver=pca_svd_solver,
            random_state=seed,
        )

        # 2. Find the number of PCs that explains specified variance
        variance_ratio = np.cumsum(adata.uns["pca"]["variance_ratio"])
        n_pcs_95 = np.min(np.where(variance_ratio >= pca_variance_explained)[0]) + 1

        # 3. Use the PCA space to compute neighbors
        sc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            n_pcs=n_pcs_95,
            metric=neighbor_distance_metric,
            random_state=seed,
        )

    elif dim_reduction == "raw":
        # Compute neighbors directly on raw data
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X", random_state=seed)

    # Prepare a list to hold cluster labels for all cells
    all_cluster_labels = [""] * len(profiles)

    # Iterate over unique treatments
    for treatment in profiles.get_column(treatment_col).unique().to_list():
        # Get indices for the current treatment
        treatment_mask = profiles.get_column(treatment_col) == treatment
        treatment_indices = np.where(treatment_mask.to_numpy())[0]

        # For treatments with too few cells, assign "no_cluster"
        if len(treatment_indices) < 2:
            for idx in treatment_indices:
                all_cluster_labels[idx] = "no_cluster"
            continue

        # Apply clustering with restrict_to for this treatment
        if cluster_method == "louvain":
            sc.tl.louvain(
                adata,
                restrict_to=(treatment_col, [treatment]),  # Only cluster these cells
                resolution=cluster_resolution,
                random_state=seed,
                key_added=f"louvain_{treatment}",
            )
            cluster_key = f"louvain_{treatment}"
        elif cluster_method == "leiden":
            sc.tl.leiden(
                adata,
                restrict_to=(treatment_col, [treatment]),  # Only cluster these cells
                resolution=cluster_resolution,
                random_state=seed,
                key_added=f"leiden_{treatment}",
            )
            cluster_key = f"leiden_{treatment}"

        for _, idx in enumerate(treatment_indices):
            # Get the cluster label for this cell using iloc
            cluster_label = adata.obs[cluster_key].iloc[idx]

            # Extract cluster id if label is a tuple or string with comma
            if isinstance(cluster_label, (tuple, list)):
                cluster_id = cluster_label[-1]
            elif isinstance(cluster_label, str) and "," in cluster_label:
                cluster_id = cluster_label.split(",")[-1].strip()
            else:
                cluster_id = cluster_label
            all_cluster_labels[idx] = f"{treatment}_{cluster_method}_{cluster_id}"

    # Add the cluster labels to the original Polars DataFrame
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
    # Calculate the ratio as a percentage
    result_df = result_df.with_columns(
        (
            pl.col("Metadata_cluster_n_cells")
            / pl.col("Metadata_treatment_n_cells")
            * 100
        ).alias("Metadata_cluster_ratio")
    )

    return result_df


@beartype
def optimized_clustering(
    profiles: pl.DataFrame,
    meta_features: list[str] | pl.Series,
    morph_features: list[str] | pl.Series,
    treatment_col: str,
    param_grid: dict[str, Any],
    n_trials: int = 100,
    seed: int = 0,
    n_jobs: int = 1,
    study_name: str | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Optimize clustering parameters using Optuna to maximize silhouette score.

    This function uses Optuna to find the best parameters for the cluster_profiles function
    by maximizing the silhouette score across all treatments. It performs hyperparameter
    optimization and returns the clustered profiles using the best parameters found.

    Parameters
    ----------
    profiles : pl.DataFrame
        DataFrame containing single-cell profiles with morphological features and
        metadata.
    meta_features : list[str] | pl.Series
        List or Series of column names used to group profiles into treatment groups for
        per-group clustering.
    morph_features : list[str] | pl.Series
        List or Series of column names representing morphological features to use for
        clustering.
    treatment_col : str
        Column name in profiles indicating treatment (used for labeling clusters).
    param_grid : Dict[str, Any]
        Dictionary defining the parameter search space. Each key should be a parameter
        name from cluster_profiles, and each value should be a dictionary with 'type'
        and range info.
    n_trials : int, default 100
        Number of optimization trials to run.
    seed : int, default 0
        Random seed for reproducibility.
    n_jobs : int, default 1
        Number of parallel jobs for Optuna optimization.
    study_name : str | None, default None
        Name for the Optuna study. If None, a default name will be generated.

    Returns
    -------
    tuple(pl.DataFrame, dict[str, Any])
        Original profiles DataFrame with optimized clustering results, same format as
        cluster_profiles output. And a dictionary of the best parameters found.

    Raises
    ------
    ValueError
        If param_grid contains unsupported parameter types or invalid parameter names.
    """

    # first check if the param_grid is valid and contains valid parameter names
    _validate_param_grid(param_grid)

    # generate the objective function for Optuna
    # this function will be called by Optuna to evaluate each set of parameters
    def objective(trial: optuna.Trial):
        """Optuna objective function to maximize silhouette score."""
        # Sample parameters from the parameter grid
        params = {}

        for param_name, param_config in param_grid.items():
            if param_config["type"] == "float":
                # Support log-scale sampling for float parameters
                log_scale = param_config.get("log", False)
                params[param_name] = trial.suggest_float(
                    param_name, param_config["low"], param_config["high"], log=log_scale
                )
            elif param_config["type"] == "int":
                # Support log-scale sampling for int parameters
                log_scale = param_config.get("log", False)
                params[param_name] = trial.suggest_int(
                    param_name, param_config["low"], param_config["high"], log=log_scale
                )
            elif param_config["type"] == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config["choices"]
                )
            else:
                raise ValueError(f"Unsupported parameter type: {param_config['type']}")

        # Add seed to params (not optimized but needed for cluster_profiles)
        params["seed"] = seed

        try:
            # Run clustering with current parameters
            clustered_profiles = cluster_profiles(
                profiles=profiles,
                meta_features=meta_features,
                morph_features=morph_features,
                treatment_col=treatment_col,
                **params,
            )

            # Calculate and return mean silhouette score
            return calculate_mean_silhouette_score(
                clustered_profiles=clustered_profiles,
                morph_features=morph_features,
                treatment_col=treatment_col,
            )

        except Exception as e:
            print(f"Exception in Optuna objective: {e}")
            return -1.0

    # Create Optuna study
    if study_name is None:
        study_name = f"cluster_optimization_{seed}"

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    # Run optimization with or without progress bar
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
    )

    # Run final clustering with best parameters
    optimized_result = cluster_profiles(
        profiles=profiles,
        meta_features=meta_features,
        morph_features=morph_features,
        treatment_col=treatment_col,
        **study.best_params,
        seed=seed,
    )

    return optimized_result, study.best_params
