#!/usr/bin/env python

# # 2.assess-heterogeneity
#
# This section of the notebook uses buscar's clustering module to assess single-cell heterogeneity. We'll focus on three specific datasets: **CFReT**, **MitoCheck**, and **CPJUMP (crispir)**. The goal is to use our clustering algorithms to identify cellular heterogeneity at the single-cell level.
#
# A key advantage of using these datasets is that they include ground-truth labels. This allows us to evaluate whether our clustering algorithms are identifying biologically meaningful groups in a data-driven way, and to assess the accuracy of our approach.

# In[1]:


import pathlib
import pickle
import sys

import polars as pl

sys.path.append("../../")
from utils.heterogeneity import assess_heterogeneity
from utils.io_utils import load_profiles

# Setting paths

# In[2]:


# set module and data directory paths
download_module_path = pathlib.Path("../0.download-data/").resolve(strict=True)
sc_profiles_path = (download_module_path / "data" / "sc-profiles").resolve(strict=True)


# setting profiles paths
cfret_profiles_path = (sc_profiles_path / "cfret" / "localhost230405150001_sc_feature_selected.parquet").resolve(strict=True)
cpjump1_trt_crispr_profiles_path = (sc_profiles_path / "cpjump1" / "trt-profiles" / "cpjump1_crispr_trt_profiles.parquet").resolve(strict=True)
mitocheck_trt_profiles_path = (sc_profiles_path / "mitocheck" / "treated_mitocheck_cp_profiles.parquet").resolve(strict=True)

# create signature output paths
results_dir = pathlib.Path("./results/cluster-labels").resolve()
results_dir.mkdir(exist_ok=True, parents=True)


# Loading datasets

# In[3]:


# load all profiles
mitocheck_trt_profile_df = load_profiles(mitocheck_trt_profiles_path)
cfret_profile_df = load_profiles(cfret_profiles_path)
cpjump1_trt_crispr_df = load_profiles(cpjump1_trt_crispr_profiles_path)


# ## Clustering profiles

# ### Assessing heterogeneity for MitoCheck data

# In[4]:


# separate metadata based on phenotypic class
# split metadata and features
mito_meta = mitocheck_trt_profile_df.columns[:13]
mito_features = mitocheck_trt_profile_df.columns[13:]


# In[ ]:


mitocheck_cluster_results = assess_heterogeneity(profiles=mitocheck_trt_profile_df, meta=mito_meta, features=mito_features, n_trials=500, n_jobs=1, study_name="mitocheck_heterogeneity", seed=0)
with open(results_dir / "mitocheck_cluster_results.pkl", "wb") as f:
    pickle.dump(mitocheck_cluster_results, f)


# ### Assessing heterogeneity for CFReT data

# In[5]:


# only selected treatment profiles from cfret
cfret_trt = cfret_profile_df.filter(pl.col("Metadata_treatment") != "DMSO")

# split metadata and features for cfret
cfret_meta = cfret_trt.columns[:19]
cfret_feats = cfret_trt.columns[19:]


# In[ ]:


cfret_cluster_results = assess_heterogeneity(profiles=cfret_trt, meta=cfret_meta, features=cfret_feats, n_trials=500, n_jobs=1, study_name="cfret_heterogeneity", seed=0)
with open(results_dir / "cfret_cluster_results.pkl", "wb") as f:
    pickle.dump(cfret_cluster_results, f)


# ### Assessing heterogeneity for CPJUMP1 CRISPR data

# In[6]:


# split metadata and features for cpjump1
cpjump1_meta = cpjump1_trt_crispr_df.columns[:18]
cpjump1_feats = cpjump1_trt_crispr_df.columns[18:]


# In[7]:


cpjump1_cluster_results = assess_heterogeneity(profiles=cpjump1_trt_crispr_df, meta=cpjump1_meta, features=cpjump1_feats, n_trials=10, n_jobs=1, study_name="cpjump1_heterogeneity", seed=0)
with open(results_dir / "cpjump1_cluster_results.pkl", "wb") as f:
    pickle.dump(cpjump1_cluster_results, f)
