#!/usr/bin/env python

# # 2. Generating Aggregate Profiles
#
# This notebook transforms single-cell morphological profiles from the CFReT pilot dataset into summary representations for downstream analysis. Aggregation reduces noise and enables robust comparisons between experimental conditions by collapsing hundreds or thousands of single-cell measurements into representative profiles.
#
# Two levels of aggregation are generated:
# 1. **Replicate-level profiles**: Aggregate cells by well position, heart number, cell type, and treatment to create technical replicate profiles
# 2. **Consensus profiles**: Further aggregate replicates by heart type and treatment to generate condition-level consensus signatures
#
# Here we used `pycytominer.aggregate()` to apply median aggregation to generate two profiles explained above. Then output profiles are saved as parquet files.

# In[1]:


import pathlib
import sys

import polars as pl
from pycytominer import aggregate

sys.path.append("../../")
from utils.data_utils import split_meta_and_features

# Setting input and output paths

# In[2]:


# setting data path for cfret-pilot dataset
cfret_profiles_path = pathlib.Path(
    "../0.download-data/data/sc-profiles/cfret/localhost230405150001_sc_feature_selected.parquet"
).resolve(strict=True)

# set results directory path
results_dir = pathlib.Path("./results").resolve()
results_dir.mkdir(exist_ok=True)

# make aggregate profile directory
aggregate_profiles_dir = results_dir / "aggregate_profiles"
aggregate_profiles_dir.mkdir(exist_ok=True)


# In[3]:


# load in the cfret-pilot dataset
cfret_df = pl.read_parquet(cfret_profiles_path)

# add a column that indicates the heart and treatment added
cfret_df = cfret_df.with_columns(
    pl.concat_str(
        [
            pl.col("Metadata_cell_type"),
            pl.col("Metadata_treatment"),
        ],
        separator="_",
    ).alias("Metadata_heart_treatment")
)

# split feature space
cfret_meta, cfret_feats = split_meta_and_features(cfret_df)

# display
print(cfret_df.shape)
cfret_df.head()


# Generating aggregate profiles at the replicate level

# In[4]:


aggregate(
    population_df=cfret_df.to_pandas(),
    strata=[
        "Metadata_heart_treatment",
        "Metadata_WellRow",
        "Metadata_WellCol",
        "Metadata_heart_number",
        "Metadata_cell_type",
        "Metadata_treatment",
    ],
    features=cfret_feats,
    operation="median",
    output_type="parquet",
    output_file=(aggregate_profiles_dir / "cfret_replicate_profiles.parquet").resolve(),
)


# Generating consensus profiles of of the treatment and heart type

# In[5]:


# aggregating profiles by heart and treatment
aggregate(
    population_df=cfret_df.to_pandas(),
    strata=["Metadata_heart_treatment", "Metadata_cell_type", "Metadata_treatment"],
    features=cfret_feats,
    operation="median",
    output_type="parquet",
    output_file=(aggregate_profiles_dir / "cfret_consensus_profiles.parquet").resolve(),
)
