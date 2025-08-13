import itertools

import numpy as np
import ot
import polars as pl
from scipy.spatial.distance import cdist


def _generate_on_off_profiles(
    profiles: pl.DataFrame, on_signature: list[str], off_signature: list[str]
):
    """Generate on and off profiles from the given profiles.

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


def earths_movers_distance(
    ref_profiles: pl.DataFrame,
    exp_profiles: pl.DataFrame,
    on_signature: list[str],
    off_signature: list[str],
    distance_metric: str = "euclidean",
):
    """Compute the Earth Mover's Distance (EMD) between reference and experimental profiles.

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
    distance_metric : str, optional
        Distance metric to use when generating the distance matrices.
    Returns
    -------
    tuple
        A tuple containing the EMD for the on-morphology and off-morphology profiles.
    """

    # compute a uniform distribution of weights for each point
    weights_ref = np.ones(ref_profiles.shape[0]) / ref_profiles.shape[0]
    weights_exp = np.ones(exp_profiles.shape[0]) / exp_profiles.shape[0]

    # creating on and off profiles for both the reference and experimental profiles
    on_ref_profiles, off_ref_profiles = _generate_on_off_profiles(
        ref_profiles, on_signature, off_signature
    )
    on_exp_profiles, off_exp_profiles = _generate_on_off_profiles(
        exp_profiles, on_signature, off_signature
    )

    # create distance matrices between reference and experimental profiles.
    # these matrices quantify the cost of moving mass between distributions
    # in the Earth Mover's Distance calculation.
    off_M = cdist(off_ref_profiles, off_exp_profiles, metric=distance_metric)
    on_M = cdist(on_ref_profiles, on_exp_profiles, metric=distance_metric)

    # compute on and off emd scores
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
) -> pl.DataFrame:
    """Measure phenotypic activity between reference and experimental profiles using
    on- and off- morphology signatures.

    Parameters
    ----------
    ref_profile : pl.DataFrame
        Reference profile DataFrame.
    exp_profiles : pl.DataFrame
        Experimental profiles DataFrame.
    on_signature : list[str]
        Morphological profiles that are in the on-morphology signature.
    off_signature : list[str]
        The list of features to include in the off profile.
    method : str, optional
        Method to use for measuring phenotypic activity, by default "emd"
    cluster_col : str, optional
        Column name for clustering information, by default "Metadata_cluster"
    treatment_col : str, optional
        Column name for treatment information, by default "Metadata_treatment"
    emd_dist_matrix_method : str, optional
        Distance metric to use when generating the distance matrices, by default "euclidean"

    Returns
    -------
    pl.DataFrame
        DataFrame containing the phenotypic activity measurements.

    Raises
    ------
    TypeError
        If any of the input parameters are of the wrong type.
    TypeError
        If any of the input parameters are of the wrong type.
    TypeError
        If any of the input parameters are of the wrong type.
    ValueError
        If the method is not recognized.
    """
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
                    distance_metric=emd_dist_matrix_method,
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
