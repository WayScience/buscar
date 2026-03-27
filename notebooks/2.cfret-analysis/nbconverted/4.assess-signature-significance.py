#!/usr/bin/env python

# In[11]:


import pathlib
import sys

import polars as pl

sys.path.append("../../")
from scipy.stats import ks_2samp
from statsmodels.stats.multitest import fdrcorrection

from utils.data_utils import shuffle_feature_profiles, split_meta_and_features
from utils.io_utils import load_configs, load_profiles

# In[12]:


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
        .with_columns(pl.col("feature").str.split("_").list.get(0).alias("channel"))
    )

    results_df.write_csv(output_path)
    return results_df


# In[13]:


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


# In[14]:


# setting parameters
treatment_col = "Metadata_cell_type_and_treatment"

# buscar parameters
healthy_label = "healthy_DMSO"
failing_label = "failing_DMSO"
on_off_signatures_method = "ks_test"


# In[15]:


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


# In[16]:


ref_df = cfret_df.filter(pl.col("Metadata_cell_type_and_treatment") == failing_label)
target_df = cfret_df.filter(pl.col("Metadata_cell_type_and_treatment") == healthy_label)


# In[17]:


ks_results_df = compute_ks_signature(
    ref_df=ref_df,
    target_df=target_df,
    features=cfret_feats,
    output_path=signatures_results_dir / "signature_importance.csv",
)

print(ks_results_df.shape)
ks_results_df.head()


# Now apply on shuffled data

# In[18]:


# concat
concat_df = pl.concat([ref_df, target_df])

# shuffle
shuffled_concat_df = shuffle_feature_profiles(
    concat_df, cfret_feats, method="column", seed=0
)

# filter shuffled_concat_df to get shuffled_ref_df and shuffled_target_df
shuffled_ref_df = shuffled_concat_df.filter(
    pl.col("Metadata_cell_type_and_treatment") == failing_label
)
shuffled_target_df = shuffled_concat_df.filter(
    pl.col("Metadata_cell_type_and_treatment") == healthy_label
)


# In[19]:


ks_results_df = compute_ks_signature(
    ref_df=shuffled_ref_df,
    target_df=shuffled_target_df,
    features=cfret_feats,
    output_path=signatures_results_dir / "shuffle_signature_importance.csv",
)

print(ks_results_df.shape)
ks_results_df.head()


# In[ ]:
