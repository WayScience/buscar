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
def _split_morphology_features(pval_df: pl.DataFrame) -> tuple[list[str], list[str]]:
    """Split features into two groups based on their significance.

    This function separates features into two categories: those that are
    significant (based on the "is_significant" column) and those that are not.

    Parameters
    ----------
    pval_df : pl.DataFrame
        A DataFrame containing feature names, and its p-values


    Returns
    -------
    tuple
        A tuple containing two lists:
        - The first list contains the names of significant features. (on-morphology
        signature)
        - The second list contains the names of non-significant features.
        (off-morphology signature)

    Raises
    ------
    TypeError
        If the input is not a polars DataFrame.
    """
    # now separate the morphology features that are significant and non-significant
    feats_dict = pval_df.select(
        pl.col("features").filter(pl.col("is_significant")).alias("on_morph"),
        pl.col("features").filter(~pl.col("is_significant")).alias("off_morph"),
    ).to_dict(as_series=False)

    on_morph_feats = feats_dict["on_morph"]
    off_morph_feats = feats_dict["off_morph"]

    return on_morph_feats, off_morph_feats


def apply_welchs_ttest(
    ref_profiles: pl.DataFrame,
    exp_profiles: pl.DataFrame,
    morph_feats: list[str],
    correction_method: str | None = "fdr_bh",
    sig_threshold: float | None = 0.05,
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
    correction_method : str, optional
        Method for multiple testing correction (e.g., "fdr_bh", "bonferroni"). Default
        is "fdr_bh".
    sig_threshold : float, optional
        Significance threshold for corrected p-values. Default is 0.05.

    Returns
    -------
    polars.DataFrame
        DataFrame with the following columns:
        - "features": Feature names.
        - "pval": Raw p-values.
        - "corrected_p_value": Corrected p-values after multiple testing correction.
        - "is_significant": Boolean indicating if the feature is significant based on
        the threshold.
    """
    # Dictionary to store p-values for each feature
    pvals = {}
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


def apply_perm_test(
    ref_profiles: pl.DataFrame,
    exp_profiles: pl.DataFrame,
    morph_feats: list[str],
    n_resamples: int | None = 1000,
    correction_method: str | None = "fdr_bh",
    statistic: Literal["mean", "median"] = "mean",
    sig_threshold: float | None = 0.05,
    seed: int | None = 0,
) -> pl.DataFrame:
    """Perform a permutation test for each feature in the morphology profiles and
    identify significant features.

    Performs a permutation test for each feature in the morphology profiles
    and identifies significant features based on a specified p-value correction method
    and significance threshold. Returns a DataFrame containing feature names, p-values,
    corrected p-values, and significance labels.

    Parameters
    ----------
    ref_profiles : pl.DataFrame
        Reference DataFrame containing morphology features.
    exp_profiles : pl.DataFrame
        Experimental DataFrame containing morphology features.
    morph_feats : list[str]
        List of morphology feature names to perform the permutation test on.
    n_resamples : int, optional
        Number of resamples for the permutation test. Default is 1000.
    correction_method : str, optional
        Method for p-value correction (e.g., "fdr_bh"). Default is "fdr_bh".
    statistic : Literal["mean", "median"], optional
        Statistic to use for the permutation test. Default is "mean".
    sig_threshold : float, optional
        Significance threshold for labeling features. Default is 0.05.
    seed : int, optional
        Random seed for reproducibility. Default is 0.

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

    # if the statistic is mean, use diff_of_means, if median, use diff_of_medians
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

    # convert p-values dictionary to a polars dataframe
    return pl.DataFrame(
        {
            "features": list(pvals.keys()),
            "pval": list(pvals.values()),
        }
    )


def apply_ks_test(
    ref_profiles: pl.DataFrame,
    exp_profiles: pl.DataFrame,
    morph_feats: list[str],
    correction_method: str | None = "fdr_bh",
    sig_threshold: float | None = 0.05,
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

    # Create a DataFrame from the results
    return pl.DataFrame(
        {
            "features": morph_feats,
            "pval": [pvals[morph_feat] for morph_feat in morph_feats],
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

    # selecting statistical test to determine the significance of the morphology features
    # and to create the on-morphology and off-morphology signatures
    if test_method == "ks_test":
        pvals_df = apply_ks_test(
            ref_profiles=ref_profiles,
            exp_profiles=exp_profiles,
            morph_feats=morph_feats,
            sig_threshold=p_threshold,
        )
    elif test_method == "permutation_test":
        pvals_df = apply_perm_test(
            ref_profiles=ref_profiles,
            exp_profiles=exp_profiles,
            morph_feats=morph_feats,
            n_resamples=permutation_resamples,
            sig_threshold=p_threshold,
            statistic=permutation_statistic,
            seed=seed,
        )
    elif test_method == "welchs_ttest":
        pvals_df = apply_welchs_ttest(
            ref_profiles=ref_profiles,
            exp_profiles=exp_profiles,
            morph_feats=morph_feats,
            sig_threshold=p_threshold,
        )

    # correct for multiple testing and add significance labels
    pvals_df = pvals_df.with_columns(
        {
            "corrected_p_value": pl.Series(
                multipletests(pvals_df["pval"].to_numpy(), method=fdr_method)[1]
            ),
            "is_significant": pl.when(pl.col("corrected_p_value") < p_threshold)
            .then(True)
            .otherwise(False),
        }
    )

    # Split the features into significant and non-significant based on the significance label
    # this returns a tuple (on_morphology_feats, off_morphology_feats)
    return _split_morphology_features(pvals_df=pvals_df)
