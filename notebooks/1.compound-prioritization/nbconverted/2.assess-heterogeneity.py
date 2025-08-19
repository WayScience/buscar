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

# Setting paths

# In[2]:


# set module and data directory paths
download_module_path = pathlib.Path("../0.download-data/").resolve(strict=True)
sc_profiles_path = (download_module_path / "data" / "sc-profiles").resolve(strict=True)

# set paths for the profiles
mitocheck_profile_path = (sc_profiles_path / "mitocheck" / "concat_mitocheck_cp_profiles_shared_feats.parquet").resolve(strict=True)
cfret_profile_path = (sc_profiles_path / "cfret" / "localhost230405150001_sc_feature_selected.parquet").resolve(strict=True)
cpjump1_crispir_path = (download_module_path / "results" / "concat_crispr_profiles.parquet").resolve(strict=True)

# create signature output paths
results_dir = pathlib.Path("./results").resolve()
results_dir.mkdir(exist_ok=True)


# Loading datasets

# In[3]:


# load all profiles
mitocheck_profile_df = pl.read_parquet(mitocheck_profile_path)
cfret_profile_df = pl.read_parquet(cfret_profile_path)
cpjump1_crispir_df = pl.read_parquet(cpjump1_crispir_path)


# Splitting datasets, only selected the treated profiles

# In[4]:


# separate metadata based on phenotypic class
mito_trt = mitocheck_profile_df.filter(pl.col("Mitocheck_Phenotypic_Class") != "negcon")

# split metadata and features
mito_meta = mito_trt.columns[:12]
mito_features = mito_trt.columns[12:]


# In[5]:


# only selected treatment profiles from cfret
cfret_trt = cfret_profile_df.filter(pl.col("Metadata_treatment") != "DMSO")

# split metadata and features for cfret
cfret_meta = cfret_trt.columns[:19]
cfret_feats = cfret_trt.columns[19:]


# In[6]:


# selecting only treated profiles from cpjump1
cpjump1_trt = cpjump1_crispir_df.filter(pl.col("Metadata_pert_type") == "trt")

# split metadata and features for cpjump1
cpjump1_meta = cpjump1_trt.columns[:18]
cpjump1_feats = cpjump1_trt.columns[18:]


# ## Clustering profiles

# In[7]:


mitocheck_cluster_results = assess_heterogeneity(profiles=mito_trt, meta=mito_meta, features=mito_features, n_trials=500, n_jobs=1, study_name="mitocheck_heterogeneity", seed=0)
with open(results_dir / "mitocheck_cluster_results.pkl", "wb") as f:
    pickle.dump(mitocheck_cluster_results, f)


# In[8]:


cfret_cluster_results = assess_heterogeneity(profiles=cfret_trt, meta=cfret_meta, features=cfret_feats, n_trials=500, n_jobs=1, study_name="cfret_heterogeneity", seed=0)
with open(results_dir / "cfret_cluster_results.pkl", "wb") as f:
    pickle.dump(cfret_cluster_results, f)


# In[ ]:


cpjump1_cluster_results = assess_heterogeneity(profiles=cpjump1_trt, meta=cpjump1_meta, features=cpjump1_feats, n_trials=500, n_jobs=1, study_name="cpjump1_heterogeneity", seed=0)
with open(results_dir / "cpjump1_cluster_results.pkl", "wb") as f:
    pickle.dump(cpjump1_cluster_results, f)
