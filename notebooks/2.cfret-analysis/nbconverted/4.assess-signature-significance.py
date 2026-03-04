#!/usr/bin/env python

# # 4. Assessing Morphological Significance
#
# In this notebook, we evaluate the morphological significance of the "on" and "off" signatures to determine how informative these features are when comparing healthy and diseased (failing) cardiac fibroblasts.
#
# We calculate statistical significance using the Kolmogorov-Smirnov (KS) test for each feature and apply False Discovery Rate (FDR) correction. The resulting table will be used in subsequent steps to visualize the significance of these feature spaces.

# In[1]:


import pathlib
import sys

import polars as pl

sys.path.append("../../")
from scipy.stats import ks_2samp
from statsmodels.stats.multitest import fdrcorrection

from utils.data_utils import split_meta_and_features
from utils.io_utils import load_configs, load_profiles

# Setting input and output paths

# In[2]:


# load in raw data from
cfret_data_dir = pathlib.Path("../0.download-data/data/sc-profiles/cfret/").resolve(
    strict=True
)
cfret_profiles_path = (
    cfret_data_dir / "localhost230405150001_sc_feature_selected.parquet"
).resolve(strict=True)
cfret_feature_space_path = (
    cfret_data_dir / "cfret_feature_space_configs.json"
).resolve(strict=True)

# make results dir
results_dir = pathlib.Path("./results").resolve()
results_dir.mkdir(parents=True, exist_ok=True)

# signatures effect
signatures_results_dir = pathlib.Path(results_dir / "signatures")
signatures_results_dir.mkdir(exist_ok=True)


# Setting notebook parameters

# In[3]:


# setting parameters
treatment_col = "Metadata_cell_type_and_treatment"

# buscar parameters
healthy_label = "healthy_DMSO"
failing_label = "failing_DMSO"
on_off_signatures_method = "ks_test"


# Loading profiles

# In[4]:


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


# Separating profiles

# In[5]:


ref_df = cfret_df.filter(pl.col(treatment_col) == failing_label)
target_df = cfret_df.filter(pl.col(treatment_col) == healthy_label)


# We apply a statistical test (Kolmogorov-Smirnov) to each feature, comparing the distributions between the two profiles. Following this, we correct the p-values using the False Discovery Rate (FDR) method. Finally, we store the results for downstream plotting and analysis.

# In[6]:


# apply ks test to each feature (single loop)
ks_stats = []
p_values = []
for feature in cfret_feats:
    stat, p_value = ks_2samp(ref_df[feature], target_df[feature])
    ks_stats.append(stat)
    p_values.append(p_value)

# FDR correction (vectorized)
_, p_values_fdr = fdrcorrection(p_values)

# Create DataFrame
ks_results_df = pl.DataFrame(
    {
        "feature": cfret_feats,
        "p_value": p_values,
        "ks_stat": ks_stats,
        "p_value_fdr_corrected": p_values_fdr,
    }
)

# Vectorized log transformation
ks_results_df = ks_results_df.with_columns(
    (-pl.col("p_value_fdr_corrected").log10()).alias("neg_log10_p_value")
)

# add a column for significance based on a threshold (e.g., 0.05)
# if lower than 0.05 then there should be a label known as "on" and higher than 0.05
# should be "off"
ks_results_df = ks_results_df.with_columns(
    pl.when(pl.col("p_value_fdr_corrected") < 0.05)
    .then(pl.lit("on"))
    .otherwise(pl.lit("off"))
    .alias("signature")
)

# add a "channel" column where we split the feature name and take the first split
ks_results_df = ks_results_df.with_columns(
    pl.col("feature").str.split("_").list.get(0).alias("channel")
)

# save dataframe as csv
ks_results_df.write_csv(signatures_results_dir / "signature_importance.csv")

# display
print(ks_results_df.shape)
ks_results_df.head()
