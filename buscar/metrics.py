"""This module provides metrics for quantifying phenotypic efficacy and specificity.

In typical Buscar usage:
- ``target`` is the desired phenotype (for example, healthy cells).
- ``ref_state`` is the starting phenotype (for example, diseased cells).

On-morphology signature distances are normalized by the distance between the selected
``target`` and ``ref_state`` so scores are interpretable across perturbations.
"""

from typing import Literal

import numpy as np
import ot
import polars as pl
from beartype import beartype

from .signatures import get_signatures


@beartype
def _normalize_on_buscar_scores(
    scores_df: pl.DataFrame,
) -> pl.DataFrame:
    """Normalize on-Buscar_scores by the reference-distance.

    The row marked with ``is_reference_distance == True`` is used as the
    denominator so the target-reference on distance becomes ``1.0``.

    Parameters
    ----------
    scores_df : pl.DataFrame
        DataFrame containing, at minimum, the columns:
        - ``on_buscar_scores``: raw on-signature distances
        - ``is_reference_distance``: boolean marker for the reference row

    Returns
    -------
    pl.DataFrame
        The same dataframe with ``on_buscar_scores`` normalized.

    Raises
    ------
    ValueError
        If required columns are missing.
        If zero or multiple rows are marked as reference distance.
        If the reference on-score is zero.
    """

    if "is_reference_distance" not in scores_df.columns:
        raise ValueError(
            "The scores DataFrame must contain an 'is_reference_distance' column "
            "indicating which row corresponds to the reference state."
        )

    # get the reference distance from the row where is_reference_distance is True
    reference_rows = scores_df.filter(pl.col("is_reference_distance"))
    ref_dist = reference_rows.select("on_buscar_scores").item()
    if ref_dist == 0:
        raise ValueError(
            "Reference distance is zero and cannot be used for normalization."
        )

    # dividing all on_scores by the reference distance to normalize
    normalized_scores_df = scores_df.with_columns(
        (pl.col("on_buscar_scores") / ref_dist).alias("on_buscar_scores")
    )

    return normalized_scores_df


def compute_earth_movers_distance(
    profile1: pl.DataFrame,
    profile2: pl.DataFrame,
    subsample_size: int | None = None,
    seed: int | None = 0,
    n_threads: int = 1,
) -> float:
    """Computing the earth mover's distance between two profiles

    Parameters
    ----------
    profile1 : pl.DataFrame
        First morphological profile containing feature measurements
    profile2 : pl.DataFrame
        Second morphological profile containing feature measurements
    subsample_size : Optional[int], optional
        If provided, the number of samples to subsample from each profile for distance
        computation, by default None (use all samples)

    Returns
    -------
    float
        Earth Mover's Distance (Wasserstein distance) between the two profiles
    """

    # if n_threads is -1, change varaible to "max" (use all available threads)
    # docs: https://pythonot.github.io/all.html#ot.emd2
    if n_threads == -1:
        n_threads = "max"
    elif n_threads < 1:
        raise ValueError("n_threads must be a positive integer or -1 for max threads.")

    # check if either profile is empty and raise an error if so
    # this avoid division by zero errors when computing the EMD
    if profile1.is_empty() or profile2.is_empty():
        raise ValueError("Both profiles must contain at least one row.")

    # Convert the profiles to numpy arrays
    p1 = profile1.to_numpy()
    p2 = profile2.to_numpy()

    # Subsample if requested
    if subsample_size is not None:
        rng = np.random.default_rng(seed)  # set random seed for reproducibility
        if subsample_size < p1.shape[0]:
            p1 = p1[rng.choice(p1.shape[0], subsample_size, replace=False)]
        if subsample_size < p2.shape[0]:
            p2 = p2[rng.choice(p2.shape[0], subsample_size, replace=False)]

    # Compute the sample-sample distance matrix (using Euclidean distance)
    M = ot.dist(p1, p2)

    # Create uniform distributions over samples for optimal transport for both ref and
    # target profiles
    ref_weights = np.ones(p1.shape[0]) / p1.shape[0]
    target_weights = np.ones(p2.shape[0]) / p2.shape[0]

    # Compute the Earth Mover's Distance (EMD)
    emd_value = ot.emd2(ref_weights, target_weights, M, numThreads=n_threads)

    return emd_value


def affected_off_features_ratio(
    ref_profiles: pl.DataFrame,
    treated_profiles: pl.DataFrame,
    off_morphology_signature: list[str],
    method: str = "ks_test",
) -> float:
    """Calculate the ratio of affected off features

    This metric calculates the ratio of features within the off-morphological signature
    that have become significant in the treated profiles compared to the reference
    profiles. A higher ratio indicates that more off features have been affected by
    the perturbation or condition being evaluated.

    Parameters
    ----------
    ref_profiles : pl.DataFrame
        DataFrame containing the reference morphological profiles.
    treated_profiles : pl.DataFrame
        DataFrame containing the treated morphological profiles.
    off_morphology_signature : list[str]
        List of feature names that constitute the off-morphological signature.
    method : str, optional
        Statistical test method to use for determining significance, by default "ks_test"

    Returns
    -------
    float
        Ratio of affected off features (number of affected off features / total number
        of off features).
    """

    # generate signatures for the off features and count how many are affected
    affected_off_sig, _, _ = get_signatures(
        ref_profiles,
        treated_profiles,
        morph_feats=off_morphology_signature,
        test_method=method,
    )

    return len(affected_off_sig) / len(off_morphology_signature)


@beartype
def calculate_score(
    target_profile: pl.DataFrame,
    treated_profile: pl.DataFrame,
    signature: list[str],
    signature_type: Literal["on", "off"],
    on_calculation: Literal["emd"] = "emd",
    off_calculation: Literal["affected_ratio"] = "affected_ratio",
    ratio_stats_method: str = "ks_test",
    n_threads: int = 1,
    seed: int = 0,
) -> float:
    """Calculate on or off score for a given morphological signature.

    Depending on ``signature_type``, this function measures either the magnitude of
    change in expected features ("on") or the unintended effects on features that should
    remain unchanged ("off").

    Parameters
    ----------
    target_profile : pl.DataFrame
        DataFrame containing the desired phenotype profile used as the comparison
        anchor (for example, healthy cells).
    treated_profile : pl.DataFrame
        DataFrame containing the perturbation profile being scored, typically starting
        from the reference phenotype (for example, diseased cells).
    signature : list[str]
        List of feature names that constitute the morphological signature.
    signature_type : Literal["on", "off"]
        Whether to compute an on-score ("on") or off-score ("off").
    on_calculation : Literal["emd"], optional
        Method used to compute the on-score. Only Earth Mover's Distance ("emd") is
        currently supported, by default "emd".
    off_calculation : Literal["affected_ratio"]
        Method used to compute the off-score:
        - "affected_ratio": proportion of off features that became significant.
        By default "affected_ratio".
    ratio_stats_method : str, optional
        Statistical test used when ``off_calculation`` is ``"affected_ratio"`` to assess
        significance of changes in off-signature features, by default "ks_test".
    seed : int, optional
        Random seed for reproducibility in stochastic methods, by default 0.

    Returns
    -------
    float
        Computed score for the given signature type and calculation method.
    """

    if signature_type == "on":
        if on_calculation == "emd":
            return compute_earth_movers_distance(
                target_profile.select(pl.col(signature)),
                treated_profile.select(pl.col(signature)),
                n_threads=n_threads,
            )
        else:
            raise ValueError(
                f"Invalid on_calculation '{on_calculation}'. Must be 'emd'."
            )

    elif signature_type == "off":
        if off_calculation == "affected_ratio":
            return affected_off_features_ratio(
                target_profile, treated_profile, signature, method=ratio_stats_method
            )
        else:
            raise ValueError(
                f"Invalid off_calculation '{off_calculation}'. Must be 'affected_ratio'"
            )

    else:
        raise ValueError(
            f"Invalid signature_type '{signature_type}'. Must be 'on' or 'off'."
        )


@beartype
def calculate_buscar_scores(
    profiles: pl.DataFrame,
    meta_cols: list[str],
    on_morphology_signature: list[str],
    off_morphology_signature: list[str],
    ref_state: str,
    target: str,
    perturbation_col: str,
    state_col: str | None = None,
    on_method: Literal["emd"] = "emd",
    off_method: Literal["affected_ratio", "emd"] = "affected_ratio",
    raw_emd_scores: bool = False,
    ratio_stats_method: str = "ks_test",
    seed: int = 0,
    n_threads: int = 1,
) -> pl.DataFrame:
    """Calculate BUSCAR scores for perturbations relative to a desired phenotype.

    This function quantifies phenotypic movement between a desired phenotype
    (``target``) and perturbation conditions that begin from another phenotype
    (``ref_state``), using two complementary metrics:

    1. ``on_buscar_scores``: distance in on-signature features
    2. ``off_buscar_scores``: off-target effect score in off-signature features

    ``on_buscar_scores`` are normalized by the on-signature distance between the
    selected ``target`` and ``ref_state``. This normalization sets the
    target-reference distance to ``1.0``.

    Parameters
    ----------
    profiles : pl.DataFrame
        Morphological profiles containing feature measurements and metadata for all
        experimental conditions.
    meta_cols : list[str]
        Column names containing metadata (e.g., perturbation, well, plate). These
        columns
        will be excluded from distance calculations.
    on_morphology_signature : list[str]
        Feature names expected to change between reference and target states. These
        define the desired phenotypic response.
    off_morphology_signature : list[str]
        Feature names expected to remain unchanged. These serve as controls to detect
        off-target or non-specific effects.
    ref_state : str
        Starting phenotype used as the perturbation baseline (for example, diseased
        cells). The row where ``perturbation == ref_state`` is used as the
        reference-distance row
        for on-score normalization.
    target : str
        Desired phenotype used as the comparison anchor (for example, healthy cells).
    perturbation_col : str
        Column name containing perturbation identifiers.
    state_col : str, optional
        Column containing cell state or perturbation identifier. If None, defaults to
        perturbation_col, indicating the state of interest is in perturbation_col.
    on_method : Literal["emd"], optional
        Method for computing on-scores. Currently only Earth Mover's Distance (EMD)
        is supported, by default "emd"
    off_method : Literal["affected_ratio", "emd"], optional
        Method for computing off-scores:
        - "affected_ratio": proportion of off features that became significant
        - "emd": Earth Mover's Distance in off-feature space
        by default "affected_ratio"
    ratio_stats_method : str, optional
        Statistical test used when ``off_method`` is set to ``"affected_ratio"`` to
        assess significance of changes in off-signature features.
    seed : int, optional
        Random seed for reproducibility in stochastic methods, by default 0

    Returns
    -------
    pl.DataFrame
        Results with columns:
        - target: target phenotype used for comparison
        - perturbation: perturbation condition
        - on_buscar_scores: on-signature score (normalized unless ``raw_emd_scores``)
        - off_buscar_scores: off-signature score
        - is_reference_distance: row containing the reference distance

    Raises
    ------
    ValueError
        If the profiles DataFrame is empty.
        If perturbation_col is not in the profiles DataFrame.
        If perturbation_col contains null values.
        If on_morphology_signature or off_morphology_signature features are missing
        from profiles.

    Notes
    -----
    ``target`` is excluded from the scored perturbation list because its distance
    to itself is not informative for ranking.
    """

    # validate input data integrity
    if profiles.is_empty():
        raise ValueError("The profiles DataFrame is empty.")
    if perturbation_col not in profiles.columns:
        raise ValueError(
            f"The perturbation column '{perturbation_col}' is not in the profiles DataFrame"
        )
    if profiles[perturbation_col].is_null().any():
        raise ValueError(
            f"The perturbation column '{perturbation_col}' contains null values."
        )
    if not set(on_morphology_signature).issubset(profiles.columns):
        raise ValueError(
            "Some features in the on_morphology_signature are not present in the "
            "profiles DataFrame."
        )
    if not set(off_morphology_signature).issubset(profiles.columns):
        raise ValueError(
            "Some features in the off_morphology_signature are not present in the "
            "profiles DataFrame."
        )

    # extract all unique perturbation conditions excluding the target phenotype
    perturbations = (
        profiles.filter(pl.col(perturbation_col) != target)
        .select(perturbation_col)
        .unique()
        .to_series()
        .to_list()
    )

    # generate desired target profile (e.g. healthy state) to compare against
    comparison_state_col = state_col if state_col is not None else perturbation_col
    target_profile = profiles.filter(pl.col(comparison_state_col) == target)
    if target_profile.is_empty():
        raise ValueError(
            f"No profiles found for target state '{target}' in column "
            f"'{comparison_state_col}'."
        )

    # initialize storage for computed scores
    scores = []

    # iterate through each perturbation condition
    for perturbation in perturbations:
        # extract morphological features for current perturbation condition
        trt_profile = profiles.filter(pl.col(perturbation_col) == perturbation).drop(
            meta_cols
        )

        # raise error if the perturbation profile is empty
        if trt_profile.height == 0 or trt_profile.is_empty():
            raise ValueError(
                f"Empty perturbation profile detected: {trt_profile.shape}"
            )

        # compute on-Buscar score between target and current treated profile
        on_buscar_scores = calculate_score(
            target_profile,
            trt_profile,
            on_morphology_signature,
            signature_type="on",
            on_calculation=on_method,
            n_threads=n_threads,
            seed=seed,
        )

        # compute off-Buscar score between target and current treated profile
        off_buscar_scores = calculate_score(
            target_profile,
            trt_profile,
            off_morphology_signature,
            signature_type="off",
            ratio_stats_method=ratio_stats_method,
            off_calculation=off_method,
            seed=seed,
        )

        # store computed scores for this perturbation
        scores.append([target, perturbation, on_buscar_scores, off_buscar_scores])

    # construct dataframe from collected scores
    scores_df = pl.DataFrame(
        scores,
        schema=["target", "perturbation", "on_buscar_scores", "off_buscar_scores"],
        orient="row",
    )

    # add a column indicating the reference distance
    scores_df = scores_df.with_columns(
        pl.when(pl.col("perturbation") == ref_state)
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias("is_reference_distance")
    )

    # normalize scores if EMD method was used to enable comparison across different
    # feature sets
    if not raw_emd_scores:
        return _normalize_on_buscar_scores(scores_df)

    return scores_df
