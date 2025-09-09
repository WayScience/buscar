from typing import Any

import numpy as np
import optuna
import polars as pl
import scanpy as sc
from sklearn.metrics import silhouette_score

from .params.clustering import (
    ClusteringImplementation,
    ClusteringMethod,
    ClusteringParams,
    DimensionalityReduction,
    DistanceMetric,
    PcaSolver,
)


def _validate_inputs(
    profiles: pl.DataFrame,
    meta: list[str] | pl.Series,
    features: list[str] | pl.Series,
) -> None:
    """Validate inputs for clustering.

    Parameters
    ----------
    profiles : pl.DataFrame
        Single-cell profiles.
    meta : Union[list[str], pl.Series]
        Metadata columns.
    features : Union[list[str], pl.Series]
        Features to use for clustering.

    Raises
    ------
    TypeError
        If any input is of the wrong type.
    ValueError
        If any input is invalid.
    """
    if not isinstance(profiles, pl.DataFrame):
        raise TypeError("profiles must be a polars DataFrame")
    if not isinstance(meta, (list, pl.Series)):
        raise TypeError("meta must be a list of strings or a polars Series")
    if isinstance(meta, list):
        if not all(isinstance(m, str) for m in meta):
            raise TypeError("All meta columns must be strings")
    elif isinstance(meta, pl.Series):
        if not meta.dtype == pl.Utf8:
            raise TypeError("meta Series must contain strings")
    if not isinstance(features, (list, pl.Series)):
        raise TypeError("features must be a list of strings or a polars Series")
    if isinstance(features, list):
        if not all(isinstance(f, str) for f in features):
            raise TypeError("All features must be strings")
    elif isinstance(features, pl.Series):
        if not features.dtype == pl.Utf8:
            raise TypeError("features Series must contain strings")
    if profiles.is_empty():
        raise ValueError("profiles DataFrame cannot be empty")
    if len(features) == 0:
        raise ValueError("features list cannot be empty")


def _prepare_anndata(
    profiles: pl.DataFrame,
    features: list[str] | pl.Series,
    params: ClusteringParams,
) -> tuple[sc.AnnData, str]:
    """Prepare AnnData object with optional PCA transformation.

    Parameters
    ----------
    profiles : pl.DataFrame
        Single-cell profiles.
    features : Union[list[str], pl.Series]
        Features to use for clustering.
    params : ClusteringParams
        Clustering parameters.

    Returns
    -------
    Tuple[sc.AnnData, str]
        AnnData object and the representation used for clustering.
    features : Union[list[str], pl.Series]
        Features to use for clustering.
    params : ClusteringParams
        Clustering parameters.

    Returns
    -------
    Tuple[sc.AnnData, str]
        AnnData object and the representation used for clustering (X or X_pca). if PCA
        is used, the representation will be "X_pca", otherwise it will be "X"
        representing the raw data.
    """

    # type check
    if not isinstance(profiles, pl.DataFrame):
        raise TypeError("profiles must be a polars DataFrame")
    if not isinstance(features, (list, pl.Series)):
        raise TypeError("features must be a list of strings or a polars Series")
    if not isinstance(params, ClusteringParams):
        raise TypeError("params must be an instance of ClusteringParams")

    # Extract feature data and create AnnData object
    feature_data = profiles.select(features)
    adata = sc.AnnData(X=feature_data.to_numpy())

    # if PCA is specified, compute PCA
    # 'use_rep' is how scanpy will refer the representation of the data
    # X = raw data, X_pca = PCA transformed data
    if params.dim_reduction == DimensionalityReduction.PCA:
        sc.pp.pca(
            adata,
            n_comps=params.pca_components,
            svd_solver=params.pca_solver.value,
            zero_center=params.pca_zero_center,
            random_state=params.seed,
        )
        use_rep = "X_pca"
    else:  # RAW
        use_rep = "X"

    return adata, use_rep


def _compute_neighbors(
    adata: sc.AnnData,
    params: ClusteringParams,
    use_rep: str,
) -> None:
    """Compute nearest neighbors for clustering.

    This function computes the nearest neighbors for the given AnnData object
    using the specified parameters. It updates the AnnData object in-place with
    the neighbors information, which will be used for clustering.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object with single-cell profiles.
    params : ClusteringParams
        Clustering parameters.
    use_rep : str
        Representation to use for neighbors computation. if X uses all features,
        if X_pca uses PCA components.

    Returns
    -------
    None
        Updates adata object in-place with neighbors information.
        The neighbors information will be stored in `adata.obsp['distances']`
        and `adata.obsp['connectivities']`.
    """
    # setting number of pcs to None if PCA is not used
    # else set it to the number of PCA components
    n_pcs = (
        params.pca_components
        if params.dim_reduction == DimensionalityReduction.PCA
        else None
    )

    # compute neighbors using scanpy
    # this will create a kNN graph based on the specified parameters
    sc.pp.neighbors(
        adata=adata,
        n_neighbors=params.n_neighbors,
        n_pcs=n_pcs,
        use_rep=use_rep,
        metric=params.dist_metric.value,
        random_state=params.seed,
    )


def _perform_clustering(adata: sc.AnnData, params: ClusteringParams) -> str:
    """Perform clustering on the AnnData object containing single-cell profiles.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object with single-cell profiles and computed neighbors.
    params : ClusteringParams
        Clustering parameters including method, resolution, and implementation.

    Returns
    -------
    str
        The key used for the clustering labels in `adata.obs`.
        Returns either "louvain" or "leiden" based on the clustering method used.
        The AnnData object is updated in-place with clustering results.

    Raises
    ------
    ValueError
        If the clustering method is not supported.
    """

    # type check
    if not isinstance(adata, sc.AnnData):
        raise TypeError("adata must be an instance of sc.AnnData")
    if not isinstance(params, ClusteringParams):
        raise TypeError("params must be an instance of ClusteringParams")

    # perform louvain or leiden clustering based on the method specified
    if params.method == ClusteringMethod.LOUVAIN:
        sc.tl.louvain(
            adata=adata,
            resolution=params.cluster_resolution,
            flavor=params.louv_clustering_imp.value,
            random_state=params.seed,
        )
        return "louvain"
    elif params.method == ClusteringMethod.LEIDEN:
        sc.tl.leiden(
            adata=adata,
            resolution=params.cluster_resolution,
            flavor=params.leid_clustering_imp.value,
            random_state=params.seed,
        )
        return "leiden"
    else:
        # raise error if method is not supported
        supported_methods = [m.value for m in ClusteringMethod]
        raise ValueError(
            f"Unsupported clustering method: {params.method}."
            f"Supported methods are: {supported_methods}"
        )


def cluster_single_cells(
    profiles: pl.DataFrame,
    meta: list[str] | pl.Series,
    features: list[str] | pl.Series,
    params: ClusteringParams | None = None,
) -> pl.DataFrame:
    """Cluster single cells using graph-based clustering methods.

    This function performs clustering on single-cell profiles using either Louvain or Leiden
    algorithms with optional PCA dimensionality reduction and customizable nearest neighbor
    computation parameters.

    Parameters
    ----------
    profiles : pl.DataFrame
        DataFrame containing single-cell profiles.
    meta : Union[list[str], pl.Series]
        Metadata columns (currently unused but kept for API consistency).
    features : Union[list[str], pl.Series]
        Feature columns to use for clustering.
    params : Optional[ClusteringParams], optional
        Clustering parameters. If None, uses defaults.

    Returns
    -------
    pl.DataFrame
        DataFrame with added 'Metadata_cluster' column containing cluster labels.

    Raises
    ------
    TypeError
        If input types are invalid.
    ValueError
        If input values are invalid.
    """
    # validate inputs
    _validate_inputs(profiles, meta, features)

    # use default parameters if none provided
    if params is None:
        params = ClusteringParams()

    # prepare AnnData object
    adata, use_rep = _prepare_anndata(profiles, features, params)

    # compute neighbors
    _compute_neighbors(adata, params, use_rep)

    # perform clustering
    cluster_key = _perform_clustering(adata, params)

    # extract cluster labels and add to profiles
    labels = adata.obs[cluster_key].to_numpy(dtype=str)
    profiles_with_clusters = profiles.with_columns(
        pl.Series("Metadata_cluster", labels)
    )

    return profiles_with_clusters


def _create_optimization_objective(
    profiles: pl.DataFrame,
    meta: list[str] | pl.Series,
    features: list[str] | pl.Series,
    seed: int,
) -> callable:
    """Create an objective function for Optuna hyperparameter optimization.

    This function creates and returns a callable objective function that Optuna uses to
    evaluate clustering hyperparameters. The objective function samples hyperparameters,
    performs clustering using `cluster_single_cells`, and computes the silhouette score
    as the optimization metric. Returns a poor score (-1.0) if clustering fails or
    results in fewer than two clusters.

    Parameters
    ----------
    profiles : pl.DataFrame
        DataFrame containing single-cell profiles.
    meta : Union[list[str], pl.Series]
        Metadata columns (currently unused but kept for API consistency).
    features : Union[list[str], pl.Series]
        Feature columns to use for clustering.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    callable
        Objective function that takes an optuna.trial.Trial and returns a float score.
    """

    # creating the objective function for Optuna
    # this function will be called by Optuna to evaluate the hyperparameters
    def objective(trial: optuna.trial.Trial) -> float:
        # sample hyperparameters using Optuna trial
        # create ClusteringParams object with trial-suggested values
        # these parameters will be evaluated by the clustering pipeline
        params = ClusteringParams(
            n_neighbors=trial.suggest_int("n_neighbors", 5, 200),
            dim_reduction=DimensionalityReduction(
                trial.suggest_categorical("dim_reduction", ["pca", "raw"])
            ),
            pca_solver=PcaSolver(
                trial.suggest_categorical(
                    "pca_solver", ["arpack", "randomized", "auto"]
                )
            ),
            pca_components=trial.suggest_int("pca_components", 10, 100),
            cluster_resolution=trial.suggest_float("resolution", 0.1, 2.0),
            method=ClusteringMethod(
                trial.suggest_categorical("method", ["louvain", "leiden"])
            ),
            dist_metric=DistanceMetric(
                trial.suggest_categorical("dist_metric", ["euclidean", "cosine"])
            ),
            louv_clustering_imp=ClusteringImplementation(
                trial.suggest_categorical("louv_clustering_imp", ["vtraag", "igraph"])
            ),
            leid_clustering_imp=ClusteringImplementation(
                trial.suggest_categorical(
                    "leid_clustering_imp", ["leidenalg", "igraph"]
                )
            ),
            seed=seed,
        )

        try:
            # perform clustering
            clustered = cluster_single_cells(
                profiles=profiles,
                meta=meta,
                features=features,
                params=params,
            )

            # extract labels and feature matrix
            labels = clustered["Metadata_cluster"].to_numpy().astype(int)
            X = profiles.select(features).to_numpy()

            # compute silhouette score (only if >1 cluster)
            n_clusters = len(set(labels))
            if n_clusters < 2:
                return -1.0

            # calculate silhouette score
            score = silhouette_score(X, labels)
            return score

        except Exception:
            # Return poor score for failed trials
            return -1.0

    return objective


def optimized_clustering(
    profiles: pl.DataFrame,
    meta: list[str] | pl.Series,
    features: list[str] | pl.Series,
    n_trials: int = 20,
    n_jobs: int = 1,
    seed: int = 0,
    study_name: str | None = None,
) -> tuple[optuna.Study, np.ndarray]:
    """Perform hyperparameter optimization for clustering.

    This function wraps the `cluster_single_cells` function with an Optuna study
    to find the best clustering parameters based on silhouette score.

    The `_create_optimization_objective` function is used to create the objective, which
    contains the `cluster_single_cells` call and computes the silhouette score for the
    clustering results. The study is then optimized over a specified number of trials.

    Parameters
    ----------
    profiles : pl.DataFrame
        DataFrame containing single-cell profiles
    meta : Union[list[str], pl.Series]
        Metadata columns
    features : Union[list[str], pl.Series]
        Feature columns to use for clustering
    n_trials : int, optional
        Number of optimization trials, by default 20
    n_jobs : int, optional
        Number of parallel jobs, by default 1
    seed : int, optional
        Random seed, by default 0
    study_name : Optional[str], optional
        Name for the Optuna study, by default None

    Returns
    -------
    Tuple[optuna.Study, np.ndarray]
        Tuple of (Optuna study object, best cluster labels)
    """
    # Validate inputs
    _validate_inputs(profiles, meta, features)

    # Create objective function
    objective = _create_optimization_objective(profiles, meta, features, seed)

    # Create and run study
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    # Get best parameters and perform final clustering
    best_params = study.best_trial.params

    final_params = ClusteringParams(
        method=ClusteringMethod(best_params["method"]),
        n_neighbors=best_params["n_neighbors"],
        dist_metric=DistanceMetric(best_params["dist_metric"]),
        dim_reduction=DimensionalityReduction(best_params["dim_reduction"]),
        pca_components=best_params["pca_components"],
        pca_solver=PcaSolver(best_params["pca_solver"]),
        louv_clustering_imp=ClusteringImplementation(
            best_params["louv_clustering_imp"]
        ),
        leid_clustering_imp=ClusteringImplementation(
            best_params["leid_clustering_imp"]
        ),
        cluster_resolution=best_params["resolution"],
        seed=seed,
    )

    # perform final clustering with best parameters
    # this will use the best parameters found during optimization
    clustered = cluster_single_cells(
        profiles=profiles,
        meta=meta,
        features=features,
        params=final_params,
    )

    cluster_labels = clustered["Metadata_cluster"].to_numpy()

    return study, cluster_labels


def assess_heterogeneity(
    profiles: pl.DataFrame,
    meta: list[str] | pl.Series,
    features: list[str] | pl.Series,
    n_trials: int = 20,
    n_jobs: int = 1,
    seed: int = 0,
    study_name: str | None = None,
) -> dict[str, Any]:
    """Assess cellular heterogeneity through optimized clustering.

    Parameters
    ----------
    profiles : pl.DataFrame
        DataFrame containing single-cell profiles
    meta : Union[list[str], pl.Series]
        Metadata columns
    features : Union[list[str], pl.Series]
        Feature columns to use for clustering
    n_trials : int, optional
        Number of optimization trials, by default 20
    n_jobs : int, optional
        Number of parallel jobs, by default 1
    seed : int, optional
        Random seed, by default 0
    study_name : Optional[str], optional
        Name for the Optuna study, by default None

    Returns
    -------
    Dict[str, Any]
        Dictionary containing study results and cluster labels
    """
    # Validate inputs
    _validate_inputs(profiles, meta, features)

    # Perform optimization
    study, cluster_labels = optimized_clustering(
        profiles=profiles,
        meta=meta,
        features=features,
        n_trials=n_trials,
        n_jobs=n_jobs,
        seed=seed,
        study_name=study_name,
    )

    # Compile results
    results = {
        "study": study,
        "cluster_labels": cluster_labels,
        "best_score": study.best_value,
        "best_params": study.best_trial.params,
        "n_clusters": np.unique(cluster_labels).size,
        "n_trials": len(study.trials),
    }

    return results
