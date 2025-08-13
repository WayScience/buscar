import itertools

import numpy as np
import ot
import polars as pl
from scipy.spatial.distance import cdist


def _generate_on_off_profiles(
    profiles: pl.DataFrame, on_signature: list[str], off_signature: list[str]
):
    on_profiles = profiles[on_signature]
    off_profiles = profiles[off_signature]
    return on_profiles, off_profiles


def earths_movers_distance(
    ref_profiles: pl.DataFrame,
    exp_profiles: pl.DataFrame,
    on_signature: list[str],
    off_signature: list[str],
    distance_metric: str = "euclidean",
):
    # compute weights for each point
    weights_ref = np.ones(ref_profiles.shape[0]) / ref_profiles.shape[0]
    weights_exp = np.ones(exp_profiles.shape[0]) / exp_profiles.shape[0]

    # create two dataframes one with only off and other with only on
    on_ref_profiles, off_ref_profiles = _generate_on_off_profiles(
        ref_profiles, on_signature, off_signature
    )
    on_exp_profiles, off_exp_profiles = _generate_on_off_profiles(
        exp_profiles, on_signature, off_signature
    )

    # create a distance matrix between on and off
    # this measures the dissimilarity between the two distributions at single-cell level
    # provides like a "cost list" determining how much work is required for shifting mass
    off_M = cdist(off_ref_profiles, off_exp_profiles, metric=distance_metric)
    on_M = cdist(on_ref_profiles, on_exp_profiles, metric=distance_metric)

    # compute on and off emd
    on_emd = ot.emd2(weights_ref, weights_exp, on_M)
    off_emd = ot.emd2(weights_ref, weights_exp, off_M)

    return on_emd, off_emd

def measure_phenotypic_activity(
    ref_profile: pl.DataFrame,
    exp_profiles: pl.DataFrame,
    on_signature: list[str],
    off_signature: list[str],
    method: str = "emd",
    cluster_col: str = "Metadata_cluster",
    treatment_col: str = "Metadata_treatment",
    emd_dist_matrix_method: str = "euclidean",
):
    # type check
    if not isinstance(ref_profile, pl.DataFrame):
        raise TypeError("ref_profile must be a polars DataFrame")
    if not isinstance(exp_profiles, pl.DataFrame):
        raise TypeError("exp_profiles must be a polars DataFrame")
    if not isinstance(method, str):
        raise TypeError("method must be a string")

    # generate all the posible combiations of cluster between these two profiles
    cluster_combinations = list(
        itertools.product(
            ref_profile[cluster_col].unique().to_list(),
            exp_profiles[cluster_col].unique().to_list(),
        )
    )

    # iterate over cluster combinations and apply distance metric
    dist_scores = []
    for treatment in exp_profiles[treatment_col].unique().to_list():
        for ref_cluster, exp_cluster in cluster_combinations:

            # filter single-cells based on selected cluster
            ref_cluster_population_df = ref_profile.filter(
                pl.col(cluster_col).is_in([ref_cluster])
            )

            # filter single-cells based on treatment and selected cluster
            exp_cluster_population_df = exp_profiles.filter(
                pl.col(treatment_col).is_in([treatment])
            ).filter(pl.col(cluster_col).is_in([exp_cluster]))

            # calculate distances between on and off
            if method == "emd":
                on_dist, off_dist = earths_movers_distance(
                    ref_cluster_population_df,
                    exp_cluster_population_df,
                    on_signature,
                    off_signature,
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            # append the results
            dist_scores.append(
                {
                    "ref_cluster": ref_cluster,
                    "treatment": treatment,
                    "exp_cluster": exp_cluster,
                    "on_dist": on_dist,
                    "off_dist": off_dist,
                }
            )

    # convert the results to a DataFrame
    dist_scores_df = pl.DataFrame(dist_scores)
    return dist_scores_df
