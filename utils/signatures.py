"""This module provides statistical tests to identify significant differences in morphology features
between two profiles (reference and experimental). It supports Welch’s t-test, Kolmogorov–Smirnov
test, and permutation test, using scipy and statsmodels.

The core function, get_signatures, compares the two profiles using a specified test and a list
of morphology features. It returns two lists of features: significant (on-morphology) and non-significant
- On-morphology signatures: significant features associated with the cellular state.
- Off-morphology signatures: non-significant features not associated with the cellular state
"""

from typing import Literal

import numpy as np
import polars as pl
from beartype import beartype
from scipy.stats import ks_2samp, permutation_test
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import ttest_ind


@beartype
def apply_welchs_ttest(
    ref_profiles: pl.DataFrame,
    exp_profiles: pl.DataFrame,
    morph_feats: list[str],
) -> pl.DataFrame:
    """Perform Welch's t-test for each feature in the provided profiles and return a
    DataFrame with p-values.

    reference: https://doi.org/10.2307/2332510

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
    """Perform a permutation test for each feature in the morphology profiles and
    identifies significant features based on a specified significance threshold.
    Returns a DataFrame containing feature names, p-values, and significance labels.

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
        - "is_significant": Boolean indicating if the feature is significant based on
        the threshold.
    """

    # Define statistic function based on specified statistic type
    if statistic == "mean":
        statistic_func = lambda ref_vals, exp_vals: np.mean(exp_vals) - np.mean(  # noqa: E731
            ref_vals
        )  # noqa: E731
    elif statistic == "median":
        statistic_func = lambda ref_vals, exp_vals: np.median(exp_vals) - np.median(  # noqa: E731
            ref_vals
        )  # noqa: E731

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
    """Perform KS-test for each feature in the morphology profiles and identifies
    significant features.

    This function performs a Kolmogorov-Smirnov test for each feature in the morphology profiles
    and identifies significant features based on a specified p-value correction method and
    significance threshold. Returns a DataFrame containing feature names, p-values,
    corrected p-values,

    Parameters
    ----------
    ref_profiles : pl.DataFrame
        Reference DataFrame.
    exp_profiles : pl.DataFrame
        Experimental DataFrame.
    morph_feats : list[str]
        List of morphology feature names.
    correction_method : str, optional
        Method for p-value correction. Default is "fdr_bh".
    sig_threshold : float, optional
        Significance threshold for labeling features. Default is 0.05.

    Returns
    -------
    pl.DataFrame
        DataFrame with the following columns:
        - "features": Feature names.
        - "pval": Raw p-values.
        - "corrected_p_value": Corrected p-values after multiple testing correction.
        - "is_significant": Boolean indicating if the feature is significant based on
        the threshold.
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
    test_method: Literal["ks_test", "permutation_test", "welchs_ttest"] = "ks_test",
    fdr_method: str = "fdr_bh",
    p_threshold: float | None = 0.05,
    permutation_resamples: int | None = 1000,
    permutation_statistic: Literal["mean", "median"] = "mean",
    seed: int | None = 0,
) -> tuple[list[str], list[str]]:
    """Identifies significant and non-significant features between two profiles.

    This function performs statistical tests to compare two profiles (reference and experimental)
    based on specified morphology features. It identifies significant features using the
    Kolmogorov-Smirnov (KS) test or other specified methods. The function applies p-value
    correction and labels features as significant or non-significant based on a given
    significance threshold.

    Parameters
    ----------
    ref_profiles : pl.DataFrame
        Reference profile as a Polars DataFrame.
    exp_profiles : pl.DataFrame
        Experimental profile as a Polars DataFrame.
    morph_feats : list[str]
        List of morphology feature names to compare.
    test_method : Literal["ks_test", "permutation_test", "welchs_ttest"], optional
        Statistical method to use for comparison. Default is "ks_test".
    fdr_method : str | None, optional
        Method for p-value correction. Default is "fdr_bh".
    p_threshold : float | None, optional
        Significance threshold for p-values. Default is 0.05.
    permutation_resamples : int | None, optional
        Number of resamples for permutation test. Default is 1000.
    seed : int | None, optional
        Random seed for reproducibility. Default is 0.
    Returns
    -------
    tuple
        A tuple containing two lists:
        - Significant features (on-morphology).
        - Non-significant features (off-morphology).

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

    # calculate corrected pvalue
    corrected_pvals = multipletests(pvals_df["pval"].to_numpy(), method=fdr_method)[1]
    pvals_df = pvals_df.with_columns(pl.Series("corrected_p_value", corrected_pvals))

    # Determine significance using p_threshold
    pvals_df = pvals_df.with_columns(
        (pl.col("corrected_p_value") < p_threshold).alias("is_significant")
    )

    # returns significant and non-significant features as lists
    return (
        pvals_df.filter(pl.col("is_significant"))["features"].to_list(),
        pvals_df.filter(~pl.col("is_significant"))["features"].to_list(),
    )
