#!/usr/bin/env python

# # CFReT Buscar analysis
#
# This notebook demonstrates applying the BUSCAR pipeline to the CFReT pilot Cell Painting dataset.
# It walks through data loading, signature extraction, clustering, measuring phenotypic activity
# (between a reference control and experimental treatments) and ranking treatments by effect.
#
# Data & references
# - Data source: CFReT pilot dataset (see paper: https://www.ahajournals.org/doi/full/10.1161/CIRCULATIONAHA.124.071956)
# - Original data repo: https://github.com/WayScience/cellpainting_predicts_cardiac_fibrosis

# In[1]:


import json
import pathlib
import sys

import polars as pl

sys.path.append("../../")
from buscar.metrics import measure_phenotypic_activity
from buscar.signatures import get_signatures

# from utils.metrics import measure_phenotypic_activity
from utils.data_utils import split_meta_and_features
from utils.io_utils import load_profiles

# Setting paramters

# In[2]:


# setting parameters
treatment_col = "Metadata_cell_type_and_treatment"

# buscar parameters
healthy_label = "healthy_DMSO"
failing_label = "failing_DMSO"
on_off_signatures_method = "ks_test"


# Setting input and output paths

# In[3]:


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

# set signatures results dir
signatures_results_dir = (results_dir / "signatures").resolve()
signatures_results_dir.mkdir(parents=True, exist_ok=True)

# set phenotypic scores results dir
phenotypic_scores_results_dir = (results_dir / "phenotypic_scores").resolve()
phenotypic_scores_results_dir.mkdir(parents=True, exist_ok=True)


# Data preprocessing
# -

# In[4]:


# loading profiles
cfret_df = load_profiles(cfret_profiles_path)

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
cfret_df.head()


# Display the treatments and number of cells per heart-treatment combination

# In[5]:


# show how many cells per treatment
# shows the number of cells per treatment that will be clustered.
cells_per_treatment_counts = cfret_df.group_by(treatment_col).len().sort(treatment_col)
cells_per_treatment_counts


# ## BUSCAR pipeline

# ### Creating on and off morphology signatures
#
# Here we generate the **on** and **off** morphological signatures to distinguish how phenotypes respond to treatments.
#
# On-Target Signatures (`on_sigs`):
#
# These represent the features that show a **significant statistical shift** from both the control group and the phenotype of interest. We use these to identify the specific morphological changes driven by the experimental conditions.
#
# Off-Target Features (`off_sigs`):
#
# These are the features that remain **unaffected** when comparing the control against the phenotype. This provides a critical baseline, helping us understand which biological markers should stay stable even when a compound is applied.

# In[6]:


# setting output paths
signatures_outpath = (signatures_results_dir / "cfret_pilot_signatures.json").resolve()

if signatures_outpath.exists():
    print("Signatures already exist, skipping this step.")
    with open(signatures_outpath) as f:
        sigs = json.load(f)
        on_sigs = sigs["on"]
        off_sigs = sigs["off"]
else:
    # once the data is loaded, separate the controls
    negcon_df = cfret_df.filter(pl.col(treatment_col) == failing_label)
    poscon_df = cfret_df.filter(pl.col(treatment_col) == healthy_label)

    # creating signatures
    on_sigs, off_sigs, _ = get_signatures(
        ref_profiles=negcon_df,
        exp_profiles=poscon_df,
        morph_feats=cfret_feats,
        test_method=on_off_signatures_method,
    )

    # Save signatures as json file
    with open(signatures_outpath, "w") as f:
        json.dump({"on": on_sigs, "off": off_sigs}, f, indent=4)


# ### Measuring phenotypic activity
#
# This section quantifies how each treatment affects cell morphology relative to the healthy reference state (`healthy_label`) and the failing target state (`failing_label`), using the previously defined on and off signatures. The resulting phenotypic scores are used to rank treatments and highlight the most active compounds.
#
# **How scores are calculated:**
#
# - **On-signature features:** We use the Earth Mover's Distance (EMD) to measure how much the on-features for each treatment differ from the healthy reference. The EMD is normalized so that the failing target state has an on-score of 1.0; treatments closer to the healthy reference have lower on-scores, and treatments more similar to the failing state have higher on-scores.
# - **Off-signature features:** We use the affected ratio, which detects if features that should remain stable (off-features) are altered by a treatment. A higher affected ratio suggests more off-target or unintended effects, while a lower ratio indicates better preservation of off-features.
#
# In summary, low on-signature scores indicate profiles that remain close to the healthy reference (desired effect), and low off-signature scores indicate minimal off-target or broad effects. Treatments are therefore ranked so that lower on- and off-signature scores are preferred.

# In[7]:


treatment_scores = measure_phenotypic_activity(
    profiles=cfret_df,
    meta_cols=cfret_meta,
    on_signature=on_sigs,
    off_signature=off_sigs,
    ref_state=healthy_label,
    target_state=failing_label,
    on_method="emd",
    off_method="affected_ratio",
    ratio_stats_method=on_off_signatures_method,
    treatment_col=treatment_col,
)

# save phenotypic scores
treatment_scores.write_csv(
    phenotypic_scores_results_dir / "cfret_pilot_phenotypic_scores.csv"
)


# In[8]:


treatment_scores
