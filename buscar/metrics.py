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

from .signatures import get_signatures


@beartype
def _normalize_scores(
    scores_df: pl.DataFrame,
    target_state: str,
    on_method: bool = False,
    off_method: bool = False,
) -> pl.DataFrame:
    """Normalize EMD scores relative to the target state.

    This function normalizes scores by dividing all values by the target state's score,
    making the target state equal to 1.0 and all other scores relative to it.

    This enables interpretation as:
    - score < 1.0: closer to reference than target
    - score = 1.0: equivalent to target state
    - score > 1.0: further from reference than target

    Parameters
    ----------
    scores_df : pl.DataFrame
        DataFrame containing computed scores with columns:
        - "treatment": treatment identifiers
        - "on_score": EMD scores in on-feature space
        - "off_score": EMD scores in off-feature space
    target_state : str
        Treatment identifier representing the desired phenotypic state.
        Used as the normalization reference (its score becomes 1.0).
    on_method : bool, optional
        If True, normalize the "on_score" column, by default False.
    off_method : bool, optional
        If True, normalize the "off_score" column, by default False.

    Returns
    -------
    pl.DataFrame
        DataFrame with normalized scores. Columns are unchanged if their
        corresponding method flag is False.

    Raises
    ------
    ValueError
        If target_state is not found in the scores DataFrame.
        If the target state's score is 0 (division by zero).
    """

    # normalize on_scores if set to true
    if on_method:
        target_rows = scores_df.filter(pl.col("treatment") == target_state)
        if target_rows.height == 0:
            raise ValueError(
                f"Target state '{target_state}' not found in scores DataFrame. "
                "Cannot normalize EMD scores."
            )
        ref_on_score = target_rows.select("on_score").item()
        if ref_on_score == 0:
            raise ValueError("Target state on_score is 0, cannot normalize.")
        scores_df = scores_df.with_columns(
            (pl.col("on_score") / ref_on_score).alias("on_score")
        )

    # normalize off_scores if set to true
    if off_method:
        target_rows = scores_df.filter(pl.col("treatment") == target_state)
        if target_rows.height == 0:
            raise ValueError(
                f"Target state '{target_state}' not found in scores DataFrame. "
                "Cannot normalize EMD scores."
            )
        ref_off_score = target_rows.select("off_score").item()
        if ref_off_score == 0:
            raise ValueError("Target state off_score is 0, cannot normalize.")
        scores_df = scores_df.with_columns(
            (pl.col("off_score") / ref_off_score).alias("off_score")
        )

    return scores_df


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

    # Convert the profiles to numpy arrays
    p1 = profile1.to_numpy()
    p2 = profile2.to_numpy()

    # check if either profile is empty and raise an error if so
    # this avoid division by zero errors when computing the EMD
    if profile1.is_empty() or profile2.is_empty():
        raise ValueError("Both profiles must contain at least one row.")

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
    target_profiles: pl.DataFrame,
    off_signature: list[str],
    method: str = "ks_test",
) -> float:
    """Calculate the ratio of affected off features

    This metric calculates the ratio of features within the off-morphological signature
    that have become significant in the target profiles compared to the reference
    profiles. A higher ratio indicates that more off features have been affected by
    the treatment or condition being evaluated.

    Parameters
    ----------
    ref_profiles : pl.DataFrame
        DataFrame containing the reference morphological profiles.
    target_profiles : pl.DataFrame
        DataFrame containing the target morphological profiles.
    off_signature : list[str]
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
        target_profiles,
        morph_feats=off_signature,
        test_method=method,
    )

    return len(affected_off_sig) / len(off_signature)


@beartype
def calculate_score(
    ref_profile: pl.DataFrame,
    target_profile: pl.DataFrame,
    signature: list[str],
    signature_type: Literal["on", "off"],
    on_calculation: Literal["emd"] = "emd",
    off_calculation: Literal["ratio_affected", "emd"] = "ratio_affected",
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
    ref_profile : pl.DataFrame
        DataFrame containing the reference morphological profile.
    target_profile : pl.DataFrame
        DataFrame containing the target morphological profile.
    signature : list[str]
        List of feature names that constitute the morphological signature.
    signature_type : Literal["on", "off"]
        Whether to compute an on-score ("on") or off-score ("off").
    on_calculation : Literal["emd"], optional
        Method used to compute the on-score. Only Earth Mover's Distance ("emd") is
        currently supported, by default "emd".
    off_calculation : Literal["ratio_affected", "emd"], optional
        Method used to compute the off-score:
        - "ratio_affected": proportion of off features that became significant.
        - "emd": Earth Mover's Distance in off-feature space.
        By default "ratio_affected".
    ratio_stats_method : str, optional
        Statistical test used when ``off_calculation`` is ``"ratio_affected"`` to assess
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
                ref_profile.select(pl.col(signature)),
                target_profile.select(pl.col(signature)),
                n_threads=n_threads,
            )
        else:
            raise ValueError(
                f"Invalid on_calculation '{on_calculation}'. Must be 'emd'."
            )

    elif signature_type == "off":
        if off_calculation == "ratio_affected":
            return affected_off_features_ratio(
                ref_profile, target_profile, signature, method=ratio_stats_method
            )
        else:
            raise ValueError(
                f"Invalid off_calculation '{off_calculation}'. Must be 'ratio_affected'"
                " or 'emd'."
            )

    else:
        raise ValueError(
            f"Invalid signature_type '{signature_type}'. Must be 'on' or 'off'."
        )


@beartype
def measure_phenotypic_activity(
    profiles: pl.DataFrame,
    meta_cols: list[str],
    on_signature: list[str],
    off_signature: list[str],
    ref_state: str,
    target_state: str,
    treatment_col: str,
    state_col: str | None = None,
    on_method: Literal["emd"] = "emd",
    off_method: Literal["ratio_affected", "emd"] = "ratio_affected",
    raw_emd_scores: bool = False,
    ratio_stats_method: str = "ks_test",
    seed: int = 0,
    n_threads: int = 1,
) -> pl.DataFrame:
    """Measure phenotypic activity by comparing morphological profiles across
    conditions.

    This function quantifies phenotypic changes between a reference state and multiple
    treatment conditions using two complementary metrics:

    1. On-score: measures the magnitude of change in features expected to be affected
    2. Off-score: measures unintended effects on features expected to remain unchanged

    Lower on-scores indicate profiles more similar to the target phenotype, while lower
    off-scores indicate higher specificity (fewer off-target effects).

    Parameters
    ----------
    profiles : pl.DataFrame
        Morphological profiles containing feature measurements and metadata for all
        experimental conditions.
    meta_cols : list[str]
        Column names containing metadata (e.g., treatment, well, plate). These columns
        will be excluded from distance calculations.
    on_signature : list[str]
        Feature names expected to change between reference and target states. These
        define the desired phenotypic response.
    off_signature : list[str]
        Feature names expected to remain unchanged. These serve as controls to detect
        off-target or non-specific effects.
    ref_state : str
        Value in treatment_col representing the baseline/control condition.
    target_state : str
        Value in treatment_col representing the desired phenotypic state.
    treatment_col : str, optional
        Column name containing treatment identifiers, by default "Metadata_treatment"
    state_col : str, optional
        Column containing cell state or treatment identifier. If None, defaults to
        treatment_col indicating the state of intrest is within the treatment_col.
    on_method : Literal["emd"], optional
        Method for computing on-scores. Currently only Earth Mover's Distance (EMD)
        is supported, by default "emd"
    off_method : Literal["ratio_affected", "emd"], optional
        Method for computing off-scores:
        - "ratio_affected": proportion of off features that became significant
        - "emd": Earth Mover's Distance in off-feature space
        by default "ratio_affected"
    ratio_stats_method : str, optional
        Statistical test used when ``off_method`` is set to ``"ratio_affected"`` to
        assess significance of changes in off-signature features.
    seed : int, optional
        Random seed for reproducibility in stochastic methods, by default 0

    Returns
    -------
    pl.DataFrame
        Ranked results with columns:
        - rank: integer ranking (1 = best match to target)
        - ref_profile: reference state identifier
        - treatment: treatment condition identifier
        - on_score: normalized distance in on-feature space (lower is better)
        - off_score: measure of off-target effects (lower is more specific)

    Raises
    ------
    ValueError
        If the profiles DataFrame is empty.
        If treatment_col is not in the profiles DataFrame.
        If treatment_col contains null values.
        If on_signature or off_signature features are missing from profiles.

    Notes
    -----
    On-scores are normalized relative to the reference state's self-distance to enable
    comparison across different feature sets and experimental conditions.
    """

    # validate input data integrity
    if profiles.is_empty():
        raise ValueError("The profiles DataFrame is empty.")
    if treatment_col not in profiles.columns:
        raise ValueError(
            f"The treatment column '{treatment_col}' is not in the profiles DataFrame"
        )
    if profiles[treatment_col].is_null().any():
        raise ValueError(
            f"The treatment column '{treatment_col}' contains null values."
        )
    if not set(on_signature).issubset(profiles.columns):
        raise ValueError(
            "Some features in the on_signature are not present in the "
            "profiles DataFrame."
        )
    if not set(off_signature).issubset(profiles.columns):
        raise ValueError(
            "Some features in the off_signature are not present in the "
            "profiles DataFrame."
        )

    # extract all unique treatment conditions excluding the reference
    treatments = (
        profiles.filter(pl.col(treatment_col) != ref_state)
        .select(treatment_col)
        .unique()
        .to_series()
        .to_list()
    )

    # initialize storage for computed scores
    scores = []

    # iterate through each treatment condition
    for treatment in treatments:
        # skipping the reference state itself it will be comparing to itself
        if treatment == ref_state:
            continue

        # extract morphological features for reference condition (excluding metadata)
        ref_profile = profiles.filter(pl.col(state_col) == ref_state).drop(meta_cols)

        # extract morphological features for current treatment condition
        target_profile = profiles.filter(pl.col(treatment_col) == treatment).drop(
            meta_cols
        )

        # raise error if the shape of both target and ref profiles are 0
        if ref_profile.height == 0 or target_profile.height == 0:
            raise ValueError(
                f"Empty profile detected: target {target_profile.height} "
                f"rows, reference {ref_profile.height} rows."
            )

        # compute distance in on-feature space (expected changes)
        on_score = calculate_score(
            ref_profile,
            target_profile,
            on_signature,
            signature_type="on",
            on_calculation=on_method,
            n_threads=n_threads,
            seed=seed,
        )

        # compute distance in off-feature space (unintended changes)
        off_score = calculate_score(
            ref_profile,
            target_profile,
            off_signature,
            signature_type="off",
            ratio_stats_method=ratio_stats_method,
            off_calculation=off_method,
            seed=seed,
        )

        # store computed scores for this treatment
        scores.append([ref_state, treatment, on_score, off_score])

    # construct dataframe from collected scores
    scores_df = pl.DataFrame(
        scores,
        schema=["ref_profile", "treatment", "on_score", "off_score"],
        orient="row",
    )

    # rank treatments: prioritize low on-scores, then low off-scores
    scores_df = scores_df.sort(
        ["on_score", "off_score"], descending=[False, False]
    ).with_row_index(name="rank", offset=1)

    # normalize scores if EMD method was used to enable comparison across different
    # feature sets
    if not raw_emd_scores:
        return _normalize_scores(
            scores_df,
            target_state,
            on_method=(on_method == "emd"),
            off_method=(off_method == "emd"),
        )

    return scores_df
