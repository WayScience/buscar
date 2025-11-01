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
from scipy.stats import hmean
from sklearn.metrics import silhouette_score

from .checks import check_for_nans
from .validator import _validate_param_grid


def calculate_hmean_silhouette_score(
    clustered_profiles: pl.DataFrame,
    morph_features: list[str] | pl.Series,
    treatment_col: str,
) -> float:
    """Calculate harmonic mean silhouette score across all treatments in clustered profiles.

    This function computes the silhouette score for each treatment group separately
    and returns the harmonic mean score across all treatments. Treatments with too few cells
    or only one cluster are skipped. If no valid scores can be computed, returns -1.0.

    These calculations are done within the optimized_clustering function to evaluate
    clustering quality during hyperparameter optimization using Optuna.

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
        Harmonic mean silhouette score across all valid treatments. Returns -1.0 if no valid
        scores can be computed (e.g., all treatments have only one cluster or too
        few cells).

    Notes
    -----
    The silhouette score measures how similar a cell is to its own cluster compared
    to other clusters. Scores range from -1 to 1:
    - Near +1: Cell is well-matched to its cluster
    - Near 0: Cell is on the border between clusters
    - Near -1: Cell may be assigned to the wrong cluster
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

    # Return harmonic mean silhouette score across all treatments
    if silhouette_scores:
        return hmean(silhouette_scores)
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
    n_neighbors: int = 15,
    neighbor_distance_metric: Literal["cosine", "euclidean", "manhattan"] = "euclidean",
    min_cells_per_treatment: int = 10,
    seed: int = 0,
) -> pl.DataFrame:
    """Cluster single-cell profiles using graph-based clustering algorithms.

    Performs per-treatment clustering where each treatment is analyzed independently:
    1. Subset profiles to each treatment
    2. Compute neighbors within that treatment only
    3. Cluster the treatment subset
    4. Combine results with unique cluster IDs per treatment

    This ensures clustering reflects within-treatment heterogeneity without
    cross-treatment influence.

    Parameters
    ----------
    profiles : pl.DataFrame
        DataFrame containing single-cell profiles with morphological features
        and metadata.
    meta_features : list[str] | pl.Series
        List or Series of column names representing metadata features to retain
        in output.
    morph_features : list[str] | pl.Series
        List or Series of column names representing morphological features to
        use for
        neighbor graph construction and clustering.
    treatment_col : str
        Column name indicating treatment groups. Each treatment will be
        clustered
        independently.
    cluster_method : Literal["louvain", "leiden"], default "leiden"
        Clustering algorithm to use.
    cluster_resolution : float, default 1.0
        Resolution parameter controlling cluster granularity.
    n_neighbors : int, default 15
        Number of nearest neighbors for graph construction. Must be less than the
        number of cells in the smallest treatment.
    neighbor_distance_metric : Literal["cosine", "euclidean", "manhattan"], default "euclidean"
        Distance metric for neighbor graph.
    min_cells_per_treatment : int, default 10
        Minimum cells required per treatment for clustering. Treatments with
        fewer cells are labeled as "{treatment}_insufficient_cells".
    seed : int, default 0
        Random seed for reproducibility.

    Returns
    -------
    pl.DataFrame
        Original profiles with four additional columns:
        - "Metadata_cluster_id": "{treatment}_{method}_{cluster_id}" or
        "{treatment}_insufficient_cells" for small treatments
        - "Metadata_cluster_n_cells": Number of cells in each cluster
        - "Metadata_treatment_n_cells": Total cells per treatment
        - "Metadata_cluster_ratio": Percentage of treatment cells in each cluster

    Raises
    ------
    ValueError
        If treatment_col not in profiles, or if morph_features contain NaN/Inf values.

    """
    # Check if treatment_col exists
    if treatment_col not in profiles.columns:
        raise ValueError(f"treatment_col '{treatment_col}' not found in profiles.")

    # Check for NaN/Inf in morph_features
    # this will raise ValueError if any are found
    check_for_nans(profiles, morph_features)

    # Convert to AnnData and add treatment info to .obs
    # this can either be PCA-reduced data or raw data
    obs_df = profiles.select(meta_features).to_pandas()
    obs_df.index = obs_df.index.astype(str)

    adata = sc.AnnData(
        X=profiles.select(morph_features).to_numpy(),
        obs=obs_df,
    )

    # Ensure the treatment column is categorical
    if adata.obs[treatment_col].dtype != "category":
        adata.obs[treatment_col] = adata.obs[treatment_col].astype("category")

    # Cluster each treatment independently
    all_cluster_labels = [""] * len(profiles)
    treatments = profiles.get_column(treatment_col).unique().to_list()

    for treatment in treatments:
        # Get indices for current treatment
        treatment_mask = profiles.get_column(treatment_col) == treatment
        treatment_indices = np.where(treatment_mask.to_numpy())[0]

        # Check if treatment has enough cells
        n_cells = len(treatment_indices)
        if n_cells < min_cells_per_treatment:
            # Label as insufficient
            for idx in treatment_indices:
                all_cluster_labels[idx] = f"{treatment}_insufficient_cells"
            continue

        # Subset AnnData to this treatment only
        treatment_adata = adata[treatment_indices].copy()

        # compute neighbors within selected treatment
        sc.pp.neighbors(
            treatment_adata,
            n_neighbors=n_neighbors,
            use_rep="X",
            random_state=seed,
            metric=neighbor_distance_metric,
        )

        try:
            # Apply clustering to the subset
            if cluster_method == "louvain":
                sc.tl.louvain(
                    treatment_adata,
                    resolution=cluster_resolution,
                    random_state=seed,
                    key_added="cluster",
                )
            elif cluster_method == "leiden":
                sc.tl.leiden(
                    treatment_adata,
                    resolution=cluster_resolution,
                    random_state=seed,
                    key_added="cluster",
                )

            # Extract cluster labels (they're simple strings like "0", "1", etc.)
            cluster_labels = treatment_adata.obs["cluster"].values

            # Assign back to full list with treatment prefix
            for i, idx in enumerate(treatment_indices):
                all_cluster_labels[idx] = (
                    f"{treatment}_{cluster_method}_{cluster_labels[i]}"
                )

        except Exception as e:
            # Handle clustering failures gracefully
            for idx in treatment_indices:
                all_cluster_labels[idx] = f"{treatment}_cluster_failed"
            print(f"Warning: Clustering failed for treatment {treatment}: {e}")

    # 4. Add cluster labels and statistics to original DataFrame
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

    # Calculate cluster ratio as percentage
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

            # Calculate and return harmonic mean silhouette score
            return calculate_hmean_silhouette_score(
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
