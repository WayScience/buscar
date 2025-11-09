"""
This module provides metrics for quantifying phenotypic activity, particularly by comparing
two morphological signatures ("on" and "off"). These signatures represent distinct sets of
features within morphological profiles, enabling the measurement of differences between
reference and experimental conditions.
"""

from typing import Literal

import numpy as np
import ot
import polars as pl
from beartype import beartype


def _generate_on_off_profiles(
    profiles: pl.DataFrame, on_signature: list[str], off_signature: list[str]
):
    """Generate on and off profiles from the given profiles.
    This function generates two DataFrames: one for on-morphology profiles, containing
    morphological features that are significantly different between cellular states
    (e.g., healthy vs. diseased), and another for off-morphology profiles, containing
    features that are not significant for these states. Both off and on-morphology profiles
    are then used to compute phenotypic activity and interpret cellular dynamics. This
    will generate two scores: on and off scores for the morphological profiles.

    Parameters
    ----------
    profiles : pl.DataFrame
        The input profiles DataFrame.
    on_signature : list[str]
        Morphological profiles that are in the on-morphology signature.
    off_signature : list[str]
        The list of features to include in the off profile.

    Returns
    -------
    tuple
        A tuple containing two DataFrames: the on profile and the off profile.
    """
    on_profiles = profiles[on_signature]
    off_profiles = profiles[off_signature]
    return on_profiles, off_profiles


def compute_earth_movers_distance(
    ref_profiles: pl.DataFrame,
    exp_profiles: pl.DataFrame,
    on_signature: list[str],
    off_signature: list[str],
    distance_metric: Literal["euclidean", "cosine", "sqeuclidean"] = "euclidean",
):
    """Compute the Earth Mover's Distance (EMD) between reference and
    experimental profiles.

    Takes in the reference and experimental profiles, along with their on and off
    signatures, and computes the EMD. Two scores will be returned: the EMD for the
    on-morphology profiles and the EMD for the off-morphology profiles.

    Parameters
    ----------
    ref_profiles : pl.DataFrame
        The reference profiles DataFrame.
    exp_profiles : pl.DataFrame
        The experimental profiles DataFrame.
    on_signature : list[str]
        Morphological profiles that are in the on-morphology signature.
    off_signature : list[str]
        The list of features to include in the off profile.
    distance_metric : Literal, optional
        Distance metric to use when generating the distance matrices.
        Must be one of: "euclidean", "cosine", "sqeuclidean".
    Returns
    -------
    tuple
        A tuple containing the EMD for the on-morphology and off-morphology profiles.

    Notes
    -----
        Earth mover's distance citation: https://doi.org/10.1023/A:1026543900054
    """

    # Check for empty DataFrames to avoid division by zero
    if ref_profiles.shape[0] == 0 or exp_profiles.shape[0] == 0:
        raise ValueError("ref_profiles and exp_profiles must not be empty.")
    if exp_profiles.shape[0] == 0:
        raise ValueError("exp_profiles is empty, cannot compute weights.")

    # Compute a uniform distribution of weights for each point
    # This allows for each cell to be weighted equally in the EMD calculation.
    weights_ref = np.ones(ref_profiles.shape[0]) / ref_profiles.shape[0]
    weights_exp = np.ones(exp_profiles.shape[0]) / exp_profiles.shape[0]

    # creating on and off profiles for both the reference and experimental profiles
    on_ref_profiles, off_ref_profiles = _generate_on_off_profiles(
        ref_profiles, on_signature, off_signature
    )
    on_exp_profiles, off_exp_profiles = _generate_on_off_profiles(
        exp_profiles, on_signature, off_signature
    )

    # Create distance matrices between reference and experimental profiles.
    # These matrices quantify the cost of moving mass between distributions
    # in the Earth Mover's Distance calculation.
    off_M = ot.dist(
        x1=off_ref_profiles.to_numpy(),
        x2=off_exp_profiles.to_numpy(),
        metric=distance_metric,
    )
    on_M = ot.dist(
        x1=on_ref_profiles.to_numpy(),
        x2=on_exp_profiles.to_numpy(),
        metric=distance_metric,
    )

    # compute on and off emd scores
    on_emd = ot.emd2(weights_ref, weights_exp, on_M)
    off_emd = ot.emd2(weights_ref, weights_exp, off_M)

    return on_emd, off_emd


@beartype
def measure_phenotypic_activity(
    profiles: pl.DataFrame,
    on_signature: list[str],
    off_signature: list[str],
    ref_treatment: str = "DMSO",
    cluster_col: str = "Metadata_cluster_id",
    treatment_col: str = "Metadata_treatment",
    method: Literal["emd"] = "emd",
    emd_dist_matrix_method: Literal["euclidean", "cosine", "sqeuclidean"] = "euclidean",
) -> pl.DataFrame:
    """Measure how different treatment clusters are from reference (control) clusters.

    This function compares cell populations between a reference treatment (e.g., DMSO control)
    and experimental treatments by calculating distance scores. For each treatment cluster,
    it computes distances to all reference clusters using two sets of morphological features:

    - **On-signature**: Features that should change with treatment (biologically relevant)
    - **Off-signature**: Features that should remain stable (used as baseline)

    The function returns pairwise comparisons with distance scores and cluster ratios,
    which can be used to identify which treatment clusters show meaningful phenotypic changes.

    Parameters
    ----------
    profiles : pl.DataFrame
        Combined DataFrame containing both reference and experimental profiles with
        morphological features, cluster assignments, and treatment labels.
    on_signature : list[str]
        Morphological feature columns expected to change with treatment (target features).
    off_signature : list[str]
        Morphological feature columns expected to remain stable (baseline features).
    ref_treatment : str, optional
        Name of the reference/control treatment to compare against, by default "DMSO".
    cluster_col : str, optional
        Column name containing cluster identifiers, by default "Metadata_cluster".
    treatment_col : str, optional
        Column name containing treatment labels, by default "Metadata_treatment".
    method : Literal["emd"], optional
        Distance calculation method. Currently only "emd" (Earth Mover's Distance)
        is supported, by default "emd".
    emd_dist_matrix_method : Literal["euclidean", "cosine", "sqeuclidean"], optional
        Distance metric for computing pairwise distances in EMD, by default "euclidean".

    Returns
    -------
    pl.DataFrame
        DataFrame with one row per reference-treatment cluster pair, containing:
        - ref_cluster: Reference cluster ID
        - treatment: Treatment name
        - exp_cluster: Experimental treatment cluster ID
        - on_dist: Distance score for on-signature features (lower = more similar)
        - off_dist: Distance score for off-signature features (lower = more similar)
        - exp_cluster_ratio: Proportion of treatment cells in this cluster

        Returns empty DataFrame if no valid comparisons can be made.

    Raises
    ------
    KeyError
        If cluster_col is not found in the profiles DataFrame.
    """
    # Validate required columns exist
    meta_feats = profiles.drop(on_signature + off_signature).columns

    if cluster_col not in meta_feats:
        raise KeyError(f"Column '{cluster_col}' not found in ref_profile")

    # create a cluster ratio dataframe
    if (
        "Metadata_cluster_ratio" not in meta_feats
        and "Metadata_cluster_id" not in meta_feats
    ):
        raise KeyError(
            "Cluster ratio columns 'Metadata_cluster_ratio' and"
            "'Metadata_cluster_id' not found in profiles DataFrame. This"
            "indicates that your profiles have not been clustered. Please run"
            "clustering before measuring phenotypic activity."
        )
    cluster_ratio_dict = dict(
        profiles[["Metadata_cluster_id", "Metadata_cluster_ratio"]].unique().rows()
    )

    # separating ref and exp profiles
    ref_profiles = profiles.filter(pl.col(treatment_col) == ref_treatment)
    exp_profiles = profiles.filter(pl.col(treatment_col) != ref_treatment)

    # get all unique combinations by using group by
    ref_clusters = (
        ref_profiles.group_by(cluster_col)
        .len()
        .select(cluster_col)
        .to_series()
        .to_list()
    )

    exp_combinations = (
        exp_profiles.group_by([treatment_col, cluster_col])
        .len()
        .select([treatment_col, cluster_col])
        .rows()
    )

    # generate all treatment-cluster combinations
    treatment_cluster_combinations = [
        (treatment, ref_cluster, exp_cluster)
        for treatment, exp_cluster in exp_combinations
        for ref_cluster in ref_clusters
    ]

    # Calculate distances for each combination
    dist_scores = []
    for treatment, ref_cluster, exp_cluster in treatment_cluster_combinations:
        try:
            # Get pre-filtered data
            ref_cluster_population_df = ref_profiles.filter(
                pl.col(cluster_col) == ref_cluster
            )

            exp_cluster_population_df = exp_profiles.filter(
                (pl.col(treatment_col) == treatment)
                & (pl.col(cluster_col) == exp_cluster)
            )

            # Skip if either population is empty
            if (
                ref_cluster_population_df.height == 0
                or exp_cluster_population_df.height == 0
            ):
                continue

            # Calculate EMD distances
            on_dist, off_dist = compute_earth_movers_distance(
                ref_cluster_population_df,
                exp_cluster_population_df,
                on_signature,
                off_signature,
                distance_metric=emd_dist_matrix_method,
            )

            # Store results
            dist_scores.append(
                {
                    "ref_cluster": ref_cluster,
                    "treatment": treatment,
                    "trt_cluster": exp_cluster,
                    "on_dist": on_dist,
                    "off_dist": off_dist,
                    "exp_cluster_ratio": cluster_ratio_dict[exp_cluster],
                }
            )

        except Exception:
            continue

    # if no valid scores were computed,
    # return an empty DataFrame with the correct schema
    if not dist_scores:
        # Return empty DataFrame with correct schema
        return pl.DataFrame(
            {
                "ref_cluster": [],
                "treatment": [],
                "exp_cluster": [],
                "on_dist": [],
                "off_dist": [],
            }
        )

    return pl.DataFrame(dist_scores)
