#!/usr/bin/env python

# # 3. Subsetting CPJUMP1 controls
#
# In this notebook, we subset control samples from the CPJUMP1 CRISPR dataset using stratified sampling. We generate 10 different random seeds to create multiple subsets, each containing 15% of the original control data stratified by plate and well metadata. This approach ensures reproducible sampling while maintaining the distribution of controls across experimental conditions.
#
# The subsampled datasets are saved as individual parquet files for downstream analysis and model training purposes.
#

# In[1]:


import pathlib
import sys

import polars as pl

sys.path.append("../../")
from utils.data_utils import load_group_stratified_data

# Setting input and output paths

#

# In[ ]:


# setting data path
data_dir = pathlib.Path("../0.download-data/data").resolve(strict=True)
download_module_results_dir = pathlib.Path("../0.download-data/results").resolve(
    strict=True
)

# setting cpjump1 data dir
cpjump_crispr_data_dir = (data_dir / "cpjump1-crispr").resolve(strict=True)

# set CPJUMP1 CRISPR dataset
cpjump_crispr_data_path = (
    download_module_results_dir / "concat_crispr_profiles.parquet"
).resolve(strict=True)


# Loading data

# In[ ]:


# only loading controls
controls_df = pl.read_parquet(cpjump_crispr_data_path).filter(
    pl.col("Metadata_pert_type") == "control"
)


# generating 10 seeds of randomly sampled controls

# In[ ]:


for seed_val in range(10):
    # load the dataset with group stratified sub sampling
    subsampled_df = load_group_stratified_data(
        profiles=controls_df,
        group_columns=["Metadata_Plate", "Metadata_Well"],
        sample_percentage=0.15,
        seed=seed_val,
    )

    # save the file
    subsampled_df.write_parquet(
        cpjump_crispr_data_dir / f"cpjump1_crispr_negcon_seed{seed_val}.parquet"
    )
