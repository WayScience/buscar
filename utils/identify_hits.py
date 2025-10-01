"""
This module provides functions to identify and rank compound (drug) performance
by analyzing cluster-level scores from treatment and control samples. The main
goal is to determine which compounds exhibit the most desirable effects, as
measured by various scoring methods that aggregate cluster-level metrics into a
single compound score.
"""

from typing import Literal

import polars as pl


# creating a function that calculates the weighted sum
def calculate_weighted_sum(scores_df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes compound scores for each treatment based on the weighted sum of
    both on_score and off_score, weighted by the ratio of the cluster (cluster
    single-cells / total treatment single-cells).

    Parameters
    ----------
    scores_df : pl.DataFrame
        DataFrame containing distance scores between treatment and control

    Returns
    -------
    pl.DataFrame
        DataFrame with one row per treatment, containing the computed

    """

    # calculated weighted sum
    compound_scores = (
        scores_df.lazy()
        .group_by("treatment")
        .agg(
            (
                (pl.col("on_score") * pl.col("ratio")).sum()
                + (pl.col("off_score") * pl.col("ratio")).sum()
            ).alias("compound_score")
        )
        .sort("compound_score")
        .collect()
    )

    return compound_scores


def identify_compound_hit(
    distance_df: pl.DataFrame, method: Literal["weighted_sum"]
) -> pl.DataFrame:
    """
    Identify and rank compound (drug) performance based on optimal cluster
    pairing and scoring.

    For each treatment (drug), this function pairs each treatment cluster with
    the best matching control cluster (based on lowest on_score and off_score),
    then computes a compound score for the drug. The compound score is
    calculated as the weighted sum of on_score and off_score, weighted by the
    ratio (cluster-pop/treatment-pop) for each cluster pair.

    Parameters
    ----------
    distance_df : pl.DataFrame
        DataFrame containing distance scores between treatment and control
        clusters. Must include columns: 'control_cluster_id',
        'treatment_cluster_id', 'treatment', 'on_score', 'off_score', 'ratio'.

    method : Literal["weighted_sum"]
        Method to compute compound score. Currently, only "weighted_sum" is
        implemented.  This method computes the sum of (on_score * ratio) and
        (off_score * ratio) for each treatment.  Treatments with lower compound
        scores are ranked higher (better performance).

    Returns
    -------
    pl.DataFrame
        DataFrame with one row per treatment, containing the computed
        compound_score for each drug, sorted in ascending order (best-performing
        drugs first).
    """

    # Select best control cluster for each treatment cluster
    # forming control treatment cluster pairs
    paired_scores_df = (
        distance_df.lazy()
        .sort(["treatment", "treatment_cluster_id", "on_score", "off_score"])
        .group_by(["treatment", "treatment_cluster_id"])
        .agg([pl.all().first()])
        .collect()
    )

    # Compute compound score for each treatment (drug)
    if method == "weighted_sum":
        compound_scores = calculate_weighted_sum(paired_scores_df)

        # rank the compounds where 1 is the highest rank(lowest score)
        compound_scores = compound_scores.with_columns(
            (pl.col("compound_score").rank("ordinal")).alias("rank")
        )

    return compound_scores
