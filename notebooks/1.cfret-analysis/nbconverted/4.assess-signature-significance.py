#!/usr/bin/env python

# # Assess Morphology Signature Significance
#
# This notebook measures the statistical significance and effect size of each morphology feature in the CFReT dataset. We compare healthy and failing cardiomyocyte cells (both treated with DMSO) to identify which features differ between the two conditions.
#
# For each morphology feature, we run a two-sample Kolmogorov-Smirnov (KS) test and apply Benjamini-Hochberg FDR correction. Features that pass the significance threshold are labeled **"on"** (part of the morphology signature), while the rest are labeled **"off"**.
#
# We also run the same analysis on shuffled data as a null control to confirm that the features identified in the observed data are not due to chance.

# In[1]:


import pathlib
import sys

import polars as pl

sys.path.append("../../")
from scipy.stats import ks_2samp
from statsmodels.stats.multitest import fdrcorrection

from utils.data_utils import shuffle_feature_profiles, split_meta_and_features
from utils.io_utils import load_configs, load_profiles

# ## KS Test Function
#
# `compute_ks_signature` runs a two-sample KS test for each morphology feature between a reference and a target population. It applies FDR correction to control for multiple testing, then labels each feature as **"on"** (statistically significant) or **"off"** (not significant). Results are saved to a CSV file.

# In[2]:


def compute_ks_signature(
    ref_df: pl.DataFrame,
    target_df: pl.DataFrame,
    features: list[str],
    output_path: pathlib.Path,
    alpha: float = 0.05,
) -> pl.DataFrame:
    """Run KS test on each feature between ref and target, apply FDR correction,
    and save results to a CSV.

    Parameters
    ----------
    ref_df : pl.DataFrame
        Reference population DataFrame.
    target_df : pl.DataFrame
        Target population DataFrame.
    features : list[str]
        Feature column names to test.
    output_path : pathlib.Path
        Path to write the resulting CSV.
    alpha : float
        Significance threshold for the "on"/"off" signature label.

    Returns
    -------
    pl.DataFrame
        KS test results with FDR-corrected p-values, -log10 transform, signature
        label, and channel.
    """
    # known channels in the dataset
    KNOWN_CHANNELS = {"Actin", "ER", "Hoechst", "Mitochondria", "PM"}
    channel_pattern = "|".join(KNOWN_CHANNELS)
    ks_stats, p_values = zip(
        *[ks_2samp(ref_df[feat], target_df[feat]) for feat in features]
    )

    _, p_values_fdr = fdrcorrection(list(p_values))

    results_df = (
        pl.DataFrame(
            {
                "feature": features,
                "p_value": list(p_values),
                "ks_stat": list(ks_stats),
                "p_value_fdr_corrected": p_values_fdr,
            }
        )
        .with_columns(
            (-pl.col("p_value_fdr_corrected").log10()).alias("neg_log10_p_value")
        )
        .with_columns(
            pl.when(pl.col("p_value_fdr_corrected") < alpha)
            .then(pl.lit("on"))
            .otherwise(pl.lit("off"))
            .alias("signature")
        )
        .with_columns(pl.col("feature").str.split("_").list.get(0).alias("compartment"))
    )

    # add channel extraction logic here if needed, e.g. using regex to extract known channels
    results_df = results_df.with_columns(
        [
            pl.col("feature")
            .str.extract(rf"_({channel_pattern})(?:_|$)", group_index=1)
            .fill_null("no-channel")
            .alias("channel")
        ]
    )

    results_df.write_csv(output_path)
    return results_df


# ## Setup
#
# Define file paths for the input data and create output directories for saving the signature results.

# In[3]:


# load raw data paths
cfret_data_dir = pathlib.Path("../0.download-data/data/sc-profiles/cfret/").resolve(
    strict=True
)
cfret_profiles_path = (
    cfret_data_dir / "localhost230405150001_sc_feature_selected.parquet"
).resolve(strict=True)
cfret_feature_space_path = (
    cfret_data_dir / "cfret_feature_space_configs.json"
).resolve(strict=True)

# create results directory
results_dir = pathlib.Path("./results").resolve()
results_dir.mkdir(parents=True, exist_ok=True)

# create subdirectory for signature outputs
signatures_results_dir = pathlib.Path(results_dir / "signatures")
signatures_results_dir.mkdir(exist_ok=True)


# ## Parameters
#
# Set the column names and labels used to identify the healthy and failing cell populations, and the method used to compute the on/off signature labels.

# In[4]:


# column that holds the combined cell type and treatment label
treatment_col = "Metadata_cell_type_and_treatment"

# population labels used to split cells into reference and target groups
healthy_label = "healthy_DMSO"  # target: healthy cardiomyocytes treated with DMSO
failing_label = "failing_DMSO"  # reference: failing cardiomyocytes treated with DMSO

# statistical method used to compute signature feature importance
on_off_signatures_method = "ks_test"


# ## Load Data
#
# Load the CFReT single-cell profiles and restrict columns to the relevant metadata and morphology features. A combined cell-type-and-treatment label is added for easy population filtering.

# In[5]:


# loading profiles
cfret_df = load_profiles(cfret_profiles_path)

# load cfret_df feature space and update cfret_df
cfret_feature_space = load_configs(cfret_feature_space_path)
cfret_meta_features = cfret_feature_space["metadata-features"]
cfret_features = cfret_feature_space["morphology-features"]
cfret_df = cfret_df.select(pl.col(cfret_meta_features + cfret_features))

# add another metadata column that combins both Metadata_heart_number and Metadata_treatment
cfret_df = cfret_df.with_columns(
    (
        pl.col("Metadata_treatment").cast(pl.Utf8)
        + "_heart_"
        + pl.col("Metadata_heart_number").cast(pl.Utf8)
    ).alias("Metadata_treatment_and_heart")
)

# renaming Metadata_treatment to Metadata_cell_type + Metadata_treatment
cfret_df = cfret_df.with_columns(
    (
        pl.col("Metadata_cell_type").cast(pl.Utf8)
        + "_"
        + pl.col("Metadata_treatment").cast(pl.Utf8)
    ).alias(treatment_col)
)

# split features
cfret_meta, cfret_feats = split_meta_and_features(cfret_df)

# Display data
print(f"Dataframe shape: {cfret_df.shape}")
cfret_df.head()


# ## Split Populations
#
# Separate cells into the reference (failing DMSO) and target (healthy DMSO) populations. The KS test will compare these two groups feature by feature.

# In[6]:


# reference population: failing cardiomyocytes (DMSO-treated)
ref_df = cfret_df.filter(pl.col("Metadata_cell_type_and_treatment") == failing_label)

# target population: healthy cardiomyocytes (DMSO-treated)
target_df = cfret_df.filter(pl.col("Metadata_cell_type_and_treatment") == healthy_label)


# ## Run KS Test on Observed Data
#
# Compute the KS statistic and FDR-corrected p-value for each morphology feature. Features with a corrected p-value below the significance threshold (alpha = 0.05) are labeled **"on"** and make up the morphology signature. All results are saved to a CSV file.

# In[7]:


ks_results_df = compute_ks_signature(
    ref_df=ref_df,
    target_df=target_df,
    features=cfret_feats,
    output_path=signatures_results_dir / "signature_importance.csv",
)

print(ks_results_df.shape)
ks_results_df.head()


# ## Null Control: Shuffled Data
#
# To verify that the observed signatures are not due to chance, we repeat the KS test on shuffled data. Shuffling randomly permutes feature values across all cells, breaking any real biological signal. We expect very few (or no) features to be labeled **"on"** in the shuffled results.

# In[8]:


# combine reference and target into one dataframe before shuffling
concat_df = pl.concat([ref_df, target_df])

# shuffle feature values across all cells to remove biological signal (null control)
shuffled_concat_df = shuffle_feature_profiles(
    concat_df, cfret_feats, method="column", seed=0
)

# re-split the shuffled data into reference and target populations
shuffled_ref_df = shuffled_concat_df.filter(
    pl.col("Metadata_cell_type_and_treatment") == failing_label
)
shuffled_target_df = shuffled_concat_df.filter(
    pl.col("Metadata_cell_type_and_treatment") == healthy_label
)


# In[9]:


ks_results_df = compute_ks_signature(
    ref_df=shuffled_ref_df,
    target_df=shuffled_target_df,
    features=cfret_feats,
    output_path=signatures_results_dir / "shuffle_signature_importance.csv",
)

print(ks_results_df.shape)
ks_results_df.head()
