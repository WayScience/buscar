#!/usr/bin/env python

# ## 4. Measuring Phenotypic Activity
#
# In this notebook, we measure the **phenotypic activity** of compounds by comparing them against the negative control. To do this, we focus on the treatment-specific clusters identified earlier, which allow us to capture the **heterogeneous effects** that each treatment produces across different subpopulations of cells. We also make use of the *on* and *off* signatures.
#
# Our approach is based on the **Earth Mover’s Distance (EMD)**, a distance metric that is interpretable in the sense that it reflects how much “work” is needed to transform one distribution (cells treated with a compound) into another (e.g., the healthy cell state as a control).
#
# We calculate EMD on two sets of features: those defining the *on-signature* and those defining the *off-signature*. This can be interpreted as follows:
#
# * **On-signature scores:**
#
#   * High EMD → the treatment/perturbation produces strong differences in morphology features associated with the targeted cell state.
#   * Low EMD → the targeted features resemble the controls, suggesting potential evidence of reversal.
#
# * **Off-signature scores:**
#
#   * High EMD → strong off-target effects, since these features are unrelated to the specific cell state.
#   * Low EMD → minimal off-target effects, indicating the compound acts more selectively.

# In[1]:


import pathlib
import sys

import polars as pl

sys.path.append("../../")
from utils import io_utils
from utils.metrics import measure_phenotypic_activity

# helper functions

# setting import and output paths

# In[ ]:


# setting directories
data_dir = pathlib.Path("../0.download-data/data/sc-profiles").resolve(strict=True)
results_dir = pathlib.Path("./results").resolve(strict=True)
signatures_dir = (results_dir / "signature_results").resolve(strict=True)
cluster_labels_dir = (results_dir / "cluster-labels").resolve(strict=True)

# setting cpjump1 profile paths, signatures, cluster labels
cpjump1_negcon_profiles_path = (data_dir / "cpjump1" / "negcon").resolve(strict=True).glob("*.parquet")
cpjump1_trt_profiles_path = (data_dir / "cpjump1" / "trt-profiles"/ "cpjump1_crispr_trt_profiles.parquet").resolve(strict=True)
cpjump1_signatures_path = (results_dir / "signature_results" /"ks_test_cpjump1_consensus_signatures.json").resolve(strict=True)
#cpjump1_cluster_labels_path = (results_dir / "cluster-labels" / "cpjump1_cluster_labels.json").resolve(strict=True)

# setting cfret1 profile paths, signatures, cluster labels
cfret_profiles_path = (data_dir / "cfret" / "localhost230405150001_sc_feature_selected.parquet").resolve(strict=True)
cfret_signatures_path = (signatures_dir / "ks_test_cfret_signatures.json").resolve(strict=True)
cfret_cluster_labels_path = (cluster_labels_dir / "cfret_cluster_results.pkl").resolve(strict=True)

# setting mitocheck profile paths, signatures, cluster labels
mitocheck_negcon_profiles_path = (data_dir / "mitocheck" / "negcon_mitocheck_cp_profiles.parquet").resolve(strict=True)
mitocheck_trt_profiles_path = (data_dir / "mitocheck" / "treated_mitocheck_cp_profiles.parquet").resolve(strict=True)
mitocheck_signatures_path = (signatures_dir / "ks_test_mitocheck_signatures.json").resolve(strict=True)
mitocheck_cluster_labels_path = (cluster_labels_dir / "mitocheck_cluster_results.pkl").resolve(strict=True)

# setting output paths
metric_out_dir = (results_dir / "metrics").resolve()
metric_out_dir.mkdir(exist_ok=True)


# In[ ]:


# load cfret profiles
cfret_profiles = pl.read_parquet(cfret_profiles_path)
negcon_cfret_profiles = cfret_profiles.filter(pl.col("Metadata_treatment") == "DMSO")
treated_cfret_profiles = cfret_profiles.filter(pl.col("Metadata_treatment") != "DMSO")

# load mitocheck
mitocheck_negcon_profiles = pl.read_parquet(mitocheck_negcon_profiles_path)
mitocheck_trt_profiles = pl.read_parquet(mitocheck_trt_profiles_path)

# load signatures (pickle files)
mitocheck_sigs = io_utils.load_configs(mitocheck_signatures_path)
cfret_sigs = io_utils.load_configs(cfret_signatures_path)

# load cluster labels (pickle files)
mitocheck_cluster_labels = io_utils.load_configs(mitocheck_cluster_labels_path)
cfret_cluster_labels = io_utils.load_configs(cfret_cluster_labels_path)


# In[ ]:


# adding cluster labels to the profiles
mitocheck_negcon_profiles = mitocheck_negcon_profiles.with_columns(
    pl.lit(0).alias("Metadata_cluster")
)
mitocheck_trt_profiles = mitocheck_trt_profiles.with_columns(
    pl.Series("Metadata_cluster", mitocheck_cluster_labels["cluster_labels"])
)


negcon_cfret_profiles = negcon_cfret_profiles.with_columns(
    pl.lit(0).alias("Metadata_cluster")
)
treated_cfret_profiles = treated_cfret_profiles.with_columns(
    pl.Series("Metadata_cluster", cfret_cluster_labels["cluster_labels"]))


# ## Measuring phenotypic activity

# In[6]:


mitocheck_phenotypic_activity = measure_phenotypic_activity(ref_profile=mitocheck_negcon_profiles,
                            exp_profiles=mitocheck_trt_profiles,
                            on_signature= mitocheck_sigs["mitocheck_negcon_ENSG00000149503_poscon"]["signatures"]["on"],
                            off_signature= mitocheck_sigs["mitocheck_negcon_ENSG00000149503_poscon"]["signatures"]["off"],
                            method="emd",
                            treatment_col="Metadata_Gene",
                            emd_dist_matrix_method="euclidean",
                            )


# In[7]:


cfret_phenotypic_activity = measure_phenotypic_activity(ref_profile=negcon_cfret_profiles,
                            exp_profiles=treated_cfret_profiles,
                            on_signature= cfret_sigs["cfret_negcon_TGFRi_poscon"]["signatures"]["on"],
                            off_signature= cfret_sigs["cfret_negcon_TGFRi_poscon"]["signatures"]["off"],
                            method="emd",
                            treatment_col="Metadata_treatment",
                            emd_dist_matrix_method="euclidean",
                            )


# In[10]:


# save phenotypic activity scores
mitocheck_phenotypic_activity.write_csv(metric_out_dir / "mitocheck_phenotypic_activity_scores.csv")
cfret_phenotypic_activity.write_csv(metric_out_dir / "cfret_phenotypic_activity_scores.csv")
