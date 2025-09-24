from typing import Literal

import numpy as np
import polars as pl
import scanpy as sc
from beartype import beartype


@beartype
def cluster_profiles(
    profiles: pl.DataFrame,
    meta_features: list[str] | pl.Series,
    morph_features: list[str] | pl.Series,
    treatment_col: str, 
    cluster_method: Literal["louvain", "leiden"] = "leiden",
    cluster_resolution: float = 1.0,
    dim_reduction: Literal["PCA_UMAP", "raw"] = "PCA_UMAP",
    umap_n_components: int = 15,
    n_neighbors: int = 15,
    neighbor_distance_metric: Literal["cosine", "euclidean", "manhattan"] = "euclidean",
    pca_variance_explained: float = 0.95,
    pca_n_components_to_capture_variance: int = 200,
    pca_svd_solver: Literal["arpack", "randomized"] = "randomized",
    seed: int = 0,
) -> pl.DataFrame:
    """Cluster single-cell profiles using a dimensionality reduction and clustering pipeline.

    This function performs clustering on single-cell morphological profiles by first applying
    dimensionality reduction (PCA followed by UMAP, or raw data) and then clustering using
    Louvain or Leiden algorithms. Clustering is performed per treatment group defined by
    meta_features, with dynamic adjustment of neighbors based on the number of cells in each group.

    Keep in mind that this function assums that you have normalized your data prior to
    using this approach.


    Pipeline for dim_reduction="PCA":
    1. Run PCA with up to 100 components (or fewer based on data constraints).
    2. Determine the number of PCs that explain at least pca_variance_explained of the variance.
    3. Run UMAP on the selected PCs to reduce to umap_n_components dimensions.
    4. Compute neighbors in UMAP space.
    5. Apply clustering (Louvain or Leiden) in UMAP space per treatment group.

    For dim_reduction="raw", neighbors are computed directly on raw data, and clustering is applied
    per treatment group.

    Parameters
    ----------
    profiles : pl.DataFrame
        DataFrame containing single-cell profiles with morphological features and metadata.
    meta_features : list[str] | pl.Series
        List or Series of column names used to group profiles into treatment groups for per-group clustering.
    morph_features : list[str] | pl.Series
        List or Series of column names representing morphological features to use for clustering.
    treatment_col : str, default 
        Column name in profiles indicating treatment (used for labeling clusters).
    cluster_method : Literal["louvain", "leiden"], default "louvain"
        Clustering algorithm to use: "louvain" or "leiden".
    cluster_resolution : float, default 1.0
        Resolution parameter for clustering (higher values lead to more clusters).
    dim_reduction : Literal["PCA", "raw"], default "PCA"
        Dimensionality reduction method: "PCA" for PCA->UMAP pipeline, "raw" for direct use of raw data.
    umap_n_components : int, default 15
        Number of components for UMAP embedding (only used if dim_reduction="PCA").
    n_neighbors : int, default 15
        Maximum number of neighbors for neighbor graph construction.
    neighbor_distance_metric : Literal["cosine", "euclidean", "manhattan"], default "euclidean"
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
    pd.DataFrame
        Original profiles DataFrame with an additional column "Metadata_cluster_id" containing
        cluster labels as categorical values, prefixed by treatment (e.g., "treatment_0").

    Raises
    ------
    ValueError
        If pca_variance_explained is not between 0 and 1.
    """

    # Validation
    if not (0 < pca_variance_explained <= 1):
        raise ValueError("pca_variance_explained must be between 0 and 1")

    # 1. Convert to AnnData and add treatment info to .obs
    adata = sc.AnnData(
        X=profiles.select(morph_features).to_numpy(),
        obs=profiles.select(meta_features).to_pandas(),
    )

    # Ensure the treatment column is categorical
    if adata.obs[treatment_col].dtype != "category":
        adata.obs[treatment_col] = adata.obs[treatment_col].astype("category")

    if dim_reduction == "PCA_UMAP":
        # 2. Run PCA with enough components to capture variance
        sc.pp.pca(
            adata,
            n_comps=pca_n_components_to_capture_variance,
            svd_solver=pca_svd_solver,
            random_state=seed,
        )

        # 3. Find the number of PCs that explains specified variance
        variance_ratio = np.cumsum(adata.uns["pca"]["variance_ratio"])
        n_pcs_95 = np.min(np.where(variance_ratio >= pca_variance_explained)[0]) + 1

        # 4. Run UMAP on the PCs that explain 95% variance
        # - calculate neighbors in PCA space
        # - apply UMAP 95% PCA space
        # - calculate neighbors in UMAP space
        sc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            n_pcs=n_pcs_95,
            metric=neighbor_distance_metric,
            random_state=seed,
        )
        sc.tl.umap(
            adata,
            n_components=umap_n_components,
            random_state=seed,
        )

        # 5. Calculate neighbors in UMAP space for clustering
        sc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            metric=neighbor_distance_metric,
            use_rep="X_umap",
            random_state=seed,
        )
    elif dim_reduction == "raw":
        # Use raw data directly
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
            # Get the cluster label for this cell
            cluster_label = adata.obs[cluster_key][idx]

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
        pl.count().over("Metadata_treatment").alias("Metadata_treatment_n_cells")
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
