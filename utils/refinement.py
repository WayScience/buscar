"""
Profile refinement utilities for filtering out poor quality clusters.

This module provides functions to refine cell profile data by removing treatment-cluster
combinations that have insufficient data quality or representation. The primary method
implemented filters based on cell count thresholds using percentile cutoffs to exclude
clusters with too few cells for reliable analysis.
"""

from typing import Literal

import numpy as np
import polars as pl


def get_cell_counts_per_cluster(
    profile: pl.DataFrame, treatment_col: str, cluster_col: str = "Metadata_cluster"
) -> list[int]:
    """
    Get cell counts for each treatment-cluster combination.

    Series of chained methods that groups the DataFrame by treatment and cluster columns,
    counts the number of cells in each group, and returns a list of these counts.

    Parameters
    ----------
    profile : pl.DataFrame
        Input DataFrame containing cell profiles with treatment and cluster metadata.
    treatment_col : str
        Column name containing treatment identifiers.
    cluster_col : str, default "Metadata_cluster"
        Column name containing cluster identifiers.

    Returns
    -------
    list[int]
        List of cell counts for each unique treatment-cluster combination.
    """
    # Group by treatment and cluster label, count cells per group
    return (
        profile.group_by(
            [treatment_col, cluster_col]
        )  # grouping by treatment and cluster label
        .agg(pl.len().alias("cell_count"))  # counting number of cells within group
        .select("cell_count")  # selecting the "cell_count" column
        .to_series()  # converting to a series
        .to_list()  # converting to a list of all counts per group
    )


def refined_profiles_by_cluster_cell_counts(
    profile: pl.DataFrame,
    treatment_col: str,
    cluster_col: str = "Metadata_cluster",
    percentile_cutoff: int | float = 20,
) -> pl.DataFrame:
    # group by treatment  and cluster label, count cells per group
    cell_counts_per_cluster = get_cell_counts_per_cluster(
        profile, treatment_col, cluster_col
    )

    # using list of counts, calculate the  percentile
    threshold = np.percentile(cell_counts_per_cluster, percentile_cutoff)

    # Add cell counts to original DataFrame using window function
    # over() performs a window operation that:
    # 1. Groups rows by [treatment_col, cluster_col] combinations
    # 2. Calculates the count (pl.len()) for each group
    # 3. Broadcasts that count value to every row within the same group
    # 4. Preserves all original rows and columns while adding the new "cell_count"
    # column
    # This allows us to filter based on the provided threshold
    return (
        profile.with_columns(
            pl.len().over([treatment_col, cluster_col]).alias("cell_count")
        )
        .filter(pl.col("cell_count") >= threshold)
        .drop("cell_count")
    )


def refine_profiles(
    profile: pl.DataFrame,
    treatment_col: str,
    cluster_col: str,
    method: Literal["cluster_cellcounts"] = "cluster_cellcounts",
    percentile_cutoff: int | float = 20,
) -> pl.DataFrame:
    """
    Refine cell profiles by filtering out treatment-cluster combinations with
    insufficient data.

    This function serves as a dispatcher for different refinement methods.
    Currently supports filtering based on cell count thresholds using percentile
    cutoffs.

    Parameters
    ----------
    profile : pl.DataFrame
        Input DataFrame containing cell profiles with treatment and cluster
        metadata.
    treatment_col : str
        Column name containing treatment identifiers.
    cluster_col : str
        Column name containing cluster identifiers.
    method : Literal["cluster_cellcounts"], default "cluster_cellcounts"
        Refinement method to use. Currently only supports "cluster_cellcounts".
    percentile_cutoff : Union[int, float], default 20
        Percentile threshold (0-100) for filtering out clusters with low cell
        counts.

    Returns
    -------
    pl.DataFrame
        Refined DataFrame with treatment-cluster combinations that have
        sufficient cell counts.

    Raises
    ------
    ValueError
        If an unsupported method is specified.

    """
    if method == "cluster_cellcounts":
        return refined_profiles_by_cluster_cell_counts(
            profile=profile,
            treatment_col=treatment_col,
            cluster_col=cluster_col,
            percentile_cutoff=percentile_cutoff,
        )
    else:
        raise ValueError(
            f"Unknown method: {method}. Supported methods: ['cluster_cellcounts']"
        )
