"""
This module provides statistical tests to identify significant differences in
morphology features between two profiles (reference and experimental). It supports
Mann-Whitney U test, Welch’s t-test, Kolmogorov–Smirnov test, and permutation test,
using scipy and statsmodels. The core function, get_signatures, compares the two
profiles using a specified test and a list of morphology features.

It returns two lists of features: significant (on-morphology) and non-significant
(off-morphology) signatures.

- On-morphology signatures: significant features associated with the cellular state.
- Off-morphology signatures: non-significant features not associated with the cellular
state.
"""

from typing import Literal

import numpy as np
import polars as pl
from beartype import beartype
from scipy.stats import ks_2samp, mannwhitneyu, permutation_test
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import ttest_ind


@beartype
def apply_mann_whitney_u_test(
    ref_profiles: pl.DataFrame, exp_profiles: pl.DataFrame, morph_feats: list[str]
) -> pl.DataFrame:
    """Perform Mann-Whitney U test for each feature in the provided profiles and return a
    DataFrame with p-values.

    The Mann-Whitney U test is a non-parametric statistical test that compares two
    independent samples to determine if they come from the same distribution. The test
    works by:
    1. Pooling all observations from both groups
    2. Ranking all values from smallest to largest (handling ties by averaging ranks)
    3. Calculating the sum of ranks for each group
    4. Computing the U statistic based on rank sums
    5. Testing the null hypothesis that the distributions are identical against the
       alternative that one distribution tends to have larger values than the other

    This test makes no assumptions about the underlying distribution shape and is
    particularly useful when data is not normally distributed or when sample sizes
    are small.

    Parameters
    ----------
    ref_profiles : polars.DataFrame
        Reference profile containing features to be tested.
    exp_profiles : polars.DataFrame
        Experimental profile containing features to be tested.
    morph_feats : list[str]
        List of feature names to perform the statistical test on.

    Returns
    -------
    polars.DataFrame
        DataFrame with the following columns:
        - "features": Feature names.
        - "pval": Raw p-values.
    """
    pvals = {}

    for morph_feat in morph_feats:
        try:
            _, p_value = mannwhitneyu(
                ref_profiles[morph_feat].to_numpy(),
                exp_profiles[morph_feat].to_numpy(),
                alternative="two-sided",
            )
        except ValueError as e:
            print(f"Error in Mann-Whitney U test for {morph_feat}: {e}")
            pvals[morph_feat] = np.nan
            continue

        pvals[morph_feat] = p_value

    return pl.DataFrame(
        {
            "features": morph_feats,
            "pval": [pvals[morph_feat] for morph_feat in morph_feats],
        }
    )


@beartype
def apply_welchs_ttest(
    ref_profiles: pl.DataFrame,
    exp_profiles: pl.DataFrame,
    morph_feats: list[str],
) -> pl.DataFrame:
    """Perform Welch's t-test for each feature in the provided profiles and
    return a DataFrame with p-values.

    Welch's t-test is a statistical method that compares the average values of a
    feature between two groups to determine if they are significantly different.
    Unlike other t-tests, Welch's version is more flexible because it doesn't
    assume that both groups have the same amount of variation (variance).

    How it works:
    1. Calculates the average value for each feature in both groups
    2. Measures how much the values vary within each group
    3. Compares the difference between group averages relative to the variation
    4. Produces a p-value indicating the likelihood that any observed difference
       is due to random chance rather than a true difference


    Parameters
    ----------
    ref_profiles : polars.DataFrame
        Reference profile containing features to be tested.
    exp_profiles : polars.DataFrame
        Experimental profile containing features to be tested.
    morph_feats : list[str]
        List of feature names to perform the statistical test on.
    sig_threshold : float, optional
        Significance threshold for labeling features. Default is 0.05.

    Returns
    -------
    polars.DataFrame
        DataFrame with the following columns:
        - "features": Feature names.
        - "pval": Raw p-values.
    """
    # Initialize dictionary to store p-values
    pvals = {}

    # Iterate through each morphology feature
    for morph_feat in morph_feats:
        try:
            # Perform Welch's t-test (two-sided, unequal variance)
            _, p_value, _ = ttest_ind(
                ref_profiles[morph_feat].to_numpy(),
                exp_profiles[morph_feat].to_numpy(),
                alternative="two-sided",
                usevar="unequal",
                value=0,
            )
        except ValueError as e:
            # Handle errors (e.g., insufficient data) and assign NaN for the feature
            print(f"Error in t-test for {morph_feat}: {e}")
            pvals[morph_feat] = np.nan
            continue

        # Store the computed p-value
        pvals[morph_feat] = p_value

    # Create a DataFrame to store features and their corresponding p-values
    return pl.DataFrame(
        {
            "features": morph_feats,
            "pval": [pvals[morph_feat] for morph_feat in morph_feats],
        }
    )


@beartype
def apply_perm_test(
    ref_profiles: pl.DataFrame,
    exp_profiles: pl.DataFrame,
    morph_feats: list[str],
    n_resamples: int | None = 1000,
    statistic: Literal["mean", "median"] = "mean",
    seed: int | None = 0,
) -> pl.DataFrame:
    """Perform a permutation test for each morphological feature in image-based
    profiles and identifies significant differences between experimental
    conditions.

    A permutation test is a non-parametric statistical method that determines
    if observed differences in cellular morphology between two conditions are
    statistically significant by comparing them to what would be expected by
    random chance alone.

    In the context of image-based profiling:
    1. Calculates the actual difference in morphological features (mean or median)
       between reference and experimental cell populations
    2. Creates thousands of "fake" comparisons by randomly shuffling cells between
       groups while keeping group sizes the same
    3. Computes the same statistic for each random shuffle to build a distribution
       of what differences would look like due to chance alone
    4. Compares the real observed difference to this null distribution to determine
       if the treatment effect is statistically significant

    Parameters
    ----------
    ref_profiles : pl.DataFrame
        Reference DataFrame containing morphology features.
    exp_profiles : pl.DataFrame
        Experimental DataFrame containing morphology features.
    morph_feats : list[str]
        List of morphology feature names to test.
    n_resamples : int, optional
        Number of resamples for the permutation test. Default is 1000.
    sig_threshold : float, optional
        Significance threshold for labeling features. Default is 0.05.
    statistic : Literal["mean", "median"], optional
        Statistic to use for the permutation test. Default is "mean".
    seed : int, optional
        Random seed for reproducibility. Default is 0.

    Returns
    -------
    pl.DataFrame
        DataFrame with the following columns:
        - "features": Feature names.
        - "pval": Raw p-values.
    """

    # Define statistic function based on specified statistic type
    if statistic == "mean":

        def _compute_mean_difference(
            ref_vals: np.ndarray, exp_vals: np.ndarray
        ) -> float:
            return np.mean(exp_vals) - np.mean(ref_vals)

        statistic_func = _compute_mean_difference
    elif statistic == "median":

        def _compute_median_difference(
            ref_vals: np.ndarray, exp_vals: np.ndarray
        ) -> float:
            return np.median(exp_vals) - np.median(ref_vals)

        statistic_func = _compute_median_difference

    # setting up dictionary to store p-values
    pvals = {}

    # iterate through each feature and perform permutation test
    for morph_feat in morph_feats:
        # Perform permutation test to compare distributions of the current feature
        # between reference and experimental profiles using the specified statistic.
        # If the test fails due to insufficient data or other issues, assign NaN
        # to the p-value and continue with the next feature.
        try:
            result = permutation_test(
                data=(
                    ref_profiles[morph_feat].to_numpy(),
                    exp_profiles[morph_feat].to_numpy(),
                ),
                statistic=statistic_func,
                alternative="two-sided",
                n_resamples=n_resamples,
                random_state=seed,
            )
        except Exception:
            # handle the exception
            pvals[morph_feat] = np.nan
            continue

        # store p-value in dictionary
        pvals[morph_feat] = result.pvalue

    # convert p-values dictionary to a polars dataframe and add significance label
    pval_list = [pvals[morph_feat] for morph_feat in morph_feats]

    return pl.DataFrame(
        {
            "features": morph_feats,
            "pval": pval_list,
        }
    )


@beartype
def apply_ks_test(
    ref_profiles: pl.DataFrame,
    exp_profiles: pl.DataFrame,
    morph_feats: list[str],
) -> pl.DataFrame:
    """Perform KS-test for each feature in the morphology profiles and return p-values.

    This function performs a Kolmogorov-Smirnov test for each feature in the morphology profiles
    and returns a DataFrame containing feature names and raw p-values. P-value correction and
    significance thresholding are not handled in this function and should be applied externally,
    for example in the `get_signatures` function.

    Parameters
    ----------
    ref_profiles : pl.DataFrame
        Reference DataFrame.
    exp_profiles : pl.DataFrame
        Experimental DataFrame.
    morph_feats : list[str]
        List of morphology feature names.

    Returns
    -------
    pl.DataFrame
        DataFrame with the following columns:
        - "features": Feature names.
        - "pval": Raw p-values.
    """

    # Perform KS-test for each column and directly create a DataFrame.
    # using a list comprehension to iterate over the morphology features
    # the list comprehension creates a list of dictionaries, each containing
    # the feature name and its corresponding p-value
    # Perform KS-test for each feature and store results in a list
    pvals = {}
    for morph_feat in morph_feats:
        # calculate the p-value using the KS-test
        # if the KS-test fails, catch the exception and continue
        # sets pval to nan
        try:
            p_value = ks_2samp(
                ref_profiles[morph_feat].to_numpy(),
                exp_profiles[morph_feat].to_numpy(),
                method="auto",
                nan_policy="omit",
            )[1]
        except Exception as e:
            # handle the exception
            print(f"Error occurred for feature {morph_feat}: {e}")
            pvals[morph_feat] = np.nan
            continue

        # store the p-value in the dictionary
        pvals[morph_feat] = p_value

    # Prepare results for correction
    features = list(pvals.keys())
    pval_list = list(pvals.values())

    # Create a DataFrame from the results
    return pl.DataFrame(
        {
            "features": features,
            "pval": pval_list,
        }
    )


@beartype  # handles type checking
def get_signatures(
    ref_profiles: pl.DataFrame,
    exp_profiles: pl.DataFrame,
    morph_feats: list[str],
    test_method: Literal[
        "ks_test", "permutation_test", "welchs_ttest", "mann_whitney_u"
    ] = "ks_test",
    fdr_method: Literal["fdr_bh"] = "fdr_bh",
    p_threshold: float | None = 0.05,
    p_value_padding: float = 0.0,
    permutation_resamples: int | None = 1000,
    permutation_statistic: Literal["mean", "median"] = "mean",
    seed: int | None = 0,
) -> tuple[list[str], list[str], list[str]]:
    """Identifies significant, non-significant, and ambiguous features between two
    profiles.

    This function compares cellular morphology profiles using one of the statistical
    methods, applies multiple testing correction, and categorizes features based on
    their statistical significance. Features are classified into three groups: those
    clearly associated with the cell state (significant), those are not
    associated (non-significant), and those with uncertain significance (ambiguous).
    Ambiguous features have corrected p-values within a buffer zone around the
    significance threshold, defined by p_threshold ± p_value_padding, indicating
    uncertain statistical evidence for their association with the cell state.

    P-value correction for multiple testing is always applied to the results,
    regardless of the test method chosen, using the method specified by the
    `fdr_method` parameter.

    The function applies p-value correction and labels features as significant
    or non-significant based on a given significance threshold.

    Parameters
    ----------
    ref_profiles : pl.DataFrame
        Reference profile as a Polars DataFrame.
    exp_profiles : pl.DataFrame
        Experimental profile as a Polars DataFrame.
    morph_feats : list[str]
        List of morphology feature names to compare.
    test_method : Literal["ks_test", "permutation_test", "welchs_ttest",
        "mann_whitney_u"], optional
        Statistical method to use for comparison. Default is "ks_test".
    fdr_method : str | None, optional
        Method for p-value correction. Default is "fdr_bh".
    p_threshold : float | None, optional
        Significance threshold for p-values. Default is 0.05.
    p_value_padding : float, optional
        Padding around the p-value threshold to create a buffer zone. Default is 0.0.
    permutation_resamples : int | None, optional
        Number of resamples for permutation test. Default is 1000.
    permutation_statistic : Literal["mean", "median"], optional
        Statistic to use for permutation test. Default is "mean".
    seed : int | None, optional
        Random seed for reproducibility. Default is 0.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        A tuple containing three lists:
        - Significant features (on-morphology).
        - Non-significant features (off-morphology).
        - Ambiguous features (features with p-values in the buffer zone around the
        threshold).

    Raises
    ------
    TypeError
        If input types are not as expected (handled by @beartype decorator).
    """
    if seed is not None:
        np.random.seed(seed)

    # Apply statistical test to determine significance of morphology features
    if test_method == "ks_test":
        pvals_df = apply_ks_test(
            ref_profiles=ref_profiles,
            exp_profiles=exp_profiles,
            morph_feats=morph_feats,
        )
    elif test_method == "permutation_test":
        pvals_df = apply_perm_test(
            ref_profiles=ref_profiles,
            exp_profiles=exp_profiles,
            morph_feats=morph_feats,
            n_resamples=permutation_resamples,
            statistic=permutation_statistic,
            seed=seed,
        )
    elif test_method == "welchs_ttest":
        pvals_df = apply_welchs_ttest(
            ref_profiles=ref_profiles,
            exp_profiles=exp_profiles,
            morph_feats=morph_feats,
        )
    elif test_method == "mann_whitney_u":
        pvals_df = apply_mann_whitney_u_test(
            ref_profiles=ref_profiles,
            exp_profiles=exp_profiles,
            morph_feats=morph_feats,
        )

    # calculate corrected pvalue
    corrected_pvals = multipletests(pvals_df["pval"].to_numpy(), method=fdr_method)[1]
    pvals_df = pvals_df.with_columns(pl.Series("corrected_p_value", corrected_pvals))

    # Determine significance using p_threshold
    # Create a buffer zone around the p-value threshold
    # Label features as significant, non-significant, or ambiguous based on the buffer
    # zone
    pvals_df = pvals_df.with_columns(
        pl.when(pl.col("corrected_p_value") < (p_threshold - p_value_padding))
        .then(pl.lit("significant"))
        .when(pl.col("corrected_p_value") > (p_threshold + p_value_padding))
        .then(pl.lit("non_significant"))
        .otherwise(pl.lit("ambiguous"))
        .alias("significance_category")
    ).with_columns(
        (pl.col("significance_category") == "significant").alias("is_significant")
    )

    # returns significant, non-significant, and variant features as lists
    return (
        pvals_df.filter(pl.col("significance_category") == "significant")[
            "features"
        ].to_list(),
        pvals_df.filter(pl.col("significance_category") == "non_significant")[
            "features"
        ].to_list(),
        pvals_df.filter(pl.col("significance_category") == "ambiguous")[
            "features"
        ].to_list(),
    )
