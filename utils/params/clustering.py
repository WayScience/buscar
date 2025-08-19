"""
This module defines the parameters and configurations for clustering methods
used in data analysis. It includes various enums for different clustering methods,
dimensionality reduction techniques, distance metrics, and PCA solvers, as well as
a dataclass to encapsulate the clustering parameters.
"""

from dataclasses import dataclass
from enum import Enum


class ClusteringMethod(Enum):
    """Supported clustering methods."""

    LOUVAIN: str = "louvain"
    LEIDEN: str = "leiden"


class DimensionalityReduction(Enum):
    """Supported dimensionality reduction methods."""

    PCA: str = "pca"
    RAW: str = "raw"


class DistanceMetric(Enum):
    """Supported distance metrics."""

    EUCLIDEAN: str = "euclidean"
    COSINE: str = "cosine"


class PcaSolver(Enum):
    """Supported PCA solvers."""

    ARPACK: str = "arpack"
    RANDOMIZED: str = "randomized"
    AUTO: str = "auto"


class ClusteringImplementation(Enum):
    """Supported clustering implementations."""

    VTRAAG: str = "vtraag"
    IGRAPH: str = "igraph"
    LEIDENALG: str = "leidenalg"


@dataclass
class ClusteringParams:
    """
    Parameters for clustering configuration.

    This dataclass contains all the necessary parameters to configure clustering
    algorithms and their preprocessing steps.

    Parameters
    ----------
    method : ClusteringMethod, default=ClusteringMethod.LOUVAIN
        The clustering algorithm to use for grouping data points.
    n_neighbors : int, default=15
        Number of nearest neighbors to consider when building the neighborhood graph.
    dist_metric : DistanceMetric, default=DistanceMetric.COSINE
        Distance metric used to compute similarities between data points.
    dim_reduction : DimensionalityReduction, default=DimensionalityReduction.PCA
        Dimensionality reduction technique to apply before clustering.
    cluster_resolution : float, optional, default=None
        Resolution parameter for community detection algorithms. Higher values
        lead to smaller clusters. If None, uses algorithm default.
    louv_clustering_imp : ClusteringImplementation, default=ClusteringImplementation.VTRAAG
        Implementation library to use for Louvain clustering algorithm.
    leid_clustering_imp : ClusteringImplementation, default=ClusteringImplementation.LEIDENALG
        Implementation library to use for Leiden clustering algorithm.
    pca_components : int, default=50
        Number of principal components to retain when using PCA for dimensionality
        reduction.
    pca_solver : PcaSolver, default=PcaSolver.ARPACK
        Solver algorithm to use for PCA computation.
    pca_zero_center : bool, default=False
        Whether to center the data (subtract mean) before applying PCA.
    seed : int, default=0
        Random seed for reproducible clustering results.
    k_neighbors: int = 15"""

    method: ClusteringMethod = ClusteringMethod.LOUVAIN
    n_neighbors: int = 15
    dist_metric: DistanceMetric = DistanceMetric.COSINE
    dim_reduction: DimensionalityReduction = DimensionalityReduction.PCA
    cluster_resolution: float | None = None
    louv_clustering_imp: ClusteringImplementation = ClusteringImplementation.VTRAAG
    leid_clustering_imp: ClusteringImplementation = ClusteringImplementation.LEIDENALG
    pca_components: int = 50
    pca_solver: PcaSolver = PcaSolver.ARPACK
    pca_zero_center: bool = False
    seed: int = 0
