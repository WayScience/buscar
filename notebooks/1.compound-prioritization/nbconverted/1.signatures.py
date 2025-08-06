#!/usr/bin/env python

# # Extracting Morphological Signatures
#
# In this notebook, we extract morphological signatures associated with two distinct cellular states:
# - **On-morphology features**: Features that significantly change with the cellular state
# - **Off-morphology features**: Features that do not show significant changes
#
# We identify and categorize features as either on- or off-morphology signatures using a systematic workflow.
# This approach is applied to three datasets: Pilot-CFReT, MitoCheck, and CPJUMP1 (CRISPR only).

# In[ ]:


import json
import pathlib
import pprint
import sys
from collections import Counter

import polars as pl

sys.path.append("../../")
from utils.data_utils import split_meta_and_features
from utils.io_utils import load_profiles
from utils.signatures import get_signatures

# In[ ]:


# parameters
method = "ks_test"


# Setting input and output paths

# In[ ]:


# setting data directory path
data_dir = pathlib.Path("../0.download-data/data/").resolve(strict=True)
data_sc_profiles_path = (data_dir / "sc-profiles").resolve(strict=True)
data_results_dir = pathlib.Path("../0.download-data/results/").resolve(strict=True)

# setting dataset paths
jump_crispr_data_path = (data_results_dir / "concat_crispr_profiles.parquet").resolve(
    strict=True
)

# setting mitocheck profile path
mitocheck_profiles_path = (
    data_sc_profiles_path
    / "mitocheck"
    / "concat_mitocheck_cp_profiles_shared_feats.parquet"
).resolve(strict=True)

# setting cfret profile path
cfret_plate_plat = (
    data_sc_profiles_path
    / "cfret"
    / "localhost230405150001_sc_feature_selected.parquet"
).resolve(strict=True)

# setting making a results directory and creating it
results_dir = pathlib.Path("results").resolve()
results_dir.mkdir(exist_ok=True)

# now making a "signature_results" within the results directory
signature_results_dir = (results_dir / "signature_results").resolve()
signature_results_dir.mkdir(exist_ok=True)


# ## Loading profiles

# ### Loading CPJUMP CRISPR profiles

# In[ ]:


# loading CPJUMP CRISPR profiles
sc_jump_crispr_profiles = load_profiles(jump_crispr_data_path)

# splitting metadata and features
sc_jump_crispr_meta, sc_jump_crispr_feats = split_meta_and_features(
    sc_jump_crispr_profiles
)

# displaying the first few rows of the profiles
sc_jump_crispr_profiles.head()


# ## Generating on and off morpholgy signatures

# ### Get signatures from CPJUMP1 dataset

# In[ ]:


# generating metadata dataframe
cp_jump_meta_df = sc_jump_crispr_profiles[sc_jump_crispr_meta]

# creating positive control genes list
negcon_profiles_df = sc_jump_crispr_profiles.filter(
    pl.col("Metadata_control_type") == "negcon"
)

# treatment profiles
trt_profiles_df = sc_jump_crispr_profiles.filter(pl.col("Metadata_pert_type") == "trt")

# selecting positive control profiles
# poscon_cp = known chemical probs that module specific genes
poscon_profiles_df = sc_jump_crispr_profiles.filter(
    pl.col("Metadata_control_type") == "poscon_cp"
)
poscon_genes = (
    cp_jump_meta_df.filter(pl.col("Metadata_control_type") == "poscon_cp")[
        "Metadata_gene"
    ]
    .unique()
    .sort()
    .to_list()
)

# displaying the number of positive control genes and their names
pprint.pprint(f"Number of positive control genes: {len(poscon_genes)}")
pprint.pprint(f"These are poscon genes: {poscon_genes}")

# display dataframe of positive control profiles
poscon_profiles_df.head()


# In[ ]:


# group by plate first
results = {}
for gene_name, gene_group_df in trt_profiles_df.group_by("Metadata_gene"):
    gene_name = gene_name[0]  # extract string value from tuple

    # getting signatures for the current gene
    on_morph_sig, off_morph_sig = get_signatures(
        ref_profiles=negcon_profiles_df,
        exp_profiles=gene_group_df,
        morph_feats=sc_jump_crispr_feats,
        method=method,
    )
    # Counting compartment signatures
    on_morph_compartments_counts = dict(
        Counter([feat.split("_")[0] for feat in on_morph_sig])
    )

    # store in dict
    results[f"negcon_{gene_name}"] = {
        "on_morph_sig": on_morph_sig,
        "off_morph_sig": off_morph_sig,
        "on_morph_compartments_counts": on_morph_compartments_counts,
    }

# writing results to a json file
with open(
    signature_results_dir / f"{method}_negcon_trt_signature_results.json", "w"
) as f:
    json.dump(results, f, indent=4)


# In[ ]:


# create a dataframe from the results
counts = []
for key, value in results.items():
    gene_name = key.replace("negcon_", "")  # remove prefix to get just gene name
    compartment_counts = value["on_morph_compartments_counts"]

    # get counts for each compartment, defaulting to 0 if not present
    nuclei_count = compartment_counts.get("Nuclei", 0)
    cytoplasm_count = compartment_counts.get("Cytoplasm", 0)
    cells_count = compartment_counts.get("Cells", 0)

    counts.append([gene_name, nuclei_count, cytoplasm_count, cells_count])

# creating a dataframe with the counts
cols = ["Gene", "Nuclei", "Cytoplasm", "Cells"]
on_sig_compartment_counts = pl.DataFrame(counts, schema=cols)

# save
on_sig_compartment_counts.write_csv(
    signature_results_dir / f"{method}_negcon_trt_on_sig_compartment_counts.csv",
)


# ### Generating signatures from Mitocheck data
#

# Loading in mitocheck data

# In[4]:


# loading MitoCheck profiles
mitocheck_profiles = load_profiles(mitocheck_profiles_path)

# splitting metadata and features columns
mito_meta = mitocheck_profiles.columns[:12]
mito_feats = mitocheck_profiles.drop(mito_meta).columns


# Splitting them into control and treated profiles

# In[ ]:


# selecting column that contains the phenotype class
phenotype_class = "Mitocheck_Phenotypic_Class"

# splitting control and treated profiles
mitocheck_negcontrol_profiles = mitocheck_profiles.filter(
    pl.col(phenotype_class) == "negcon"
)
mitocheck_trt_profiles = mitocheck_profiles.filter(pl.col(phenotype_class) != "negcon")


# In[ ]:


# group by plate first
results = {}
for class_name, class_group_df in mitocheck_trt_profiles.group_by(phenotype_class):
    class_name = class_name[0]  # extract string value from tuple

    # getting signatures for the current class
    on_morph_sig, off_morph_sig = get_signatures(
        ref_profiles=mitocheck_negcontrol_profiles,
        exp_profiles=class_group_df,
        morph_feats=mito_feats,
        method=method,
    )
    # Counting compartment signatures
    off_morph_compartments_counts = dict(
        Counter([feat.split("_")[0] for feat in off_morph_sig])
    )
    on_morph_compartments_counts = dict(
        Counter([feat.split("_")[0] for feat in on_morph_sig])
    )

    # store in dict
    results[f"negcon_{class_name}"] = {
        "off_morph_compartments_counts": off_morph_compartments_counts,
        "on_morph_compartments_counts": on_morph_compartments_counts,
        "on_morph_sig": on_morph_sig,
        "off_morph_sig": off_morph_sig,
    }


# writing results to a json file
with open(
    signature_results_dir / f"{method}_mitocheck_trt_signature_results.json", "w"
) as f:
    json.dump(results, f, indent=4)


# ### Generating signatures from CFReT data

# In[10]:


# loading cfret profiles
phenotype_class = "Metadata_treatment"
cfret_profiles = load_profiles(cfret_plate_plat)

# splitting metadata and features columns
cfret_meta, cfret_feats = split_meta_and_features(cfret_profiles)

# splitting negative control and treatment profiles
negcon_cfret_profiles = cfret_profiles.filter(pl.col("Metadata_treatment") == "DMSO")
trt_cfret_profiles = cfret_profiles.filter(pl.col("Metadata_treatment") != "DMSO")


# In[14]:


results = {}
for trt_name, trt_group_df in trt_cfret_profiles.group_by(phenotype_class):
    trt_name = trt_name[0]

    # getting signatures for the current class
    on_morph_sig, off_morph_sig = get_signatures(
        ref_profiles=negcon_cfret_profiles,
        exp_profiles=trt_group_df,
        morph_feats=cfret_feats,
        method=method,
    )
    # counting compartment signatures
    off_morph_compartments_counts = dict(
        Counter([feat.split("_")[0] for feat in off_morph_sig])
    )
    on_morph_compartments_counts = dict(
        Counter([feat.split("_")[0] for feat in on_morph_sig])
    )

    # store in dict
    results[f"negcon_{trt_name}"] = {
        "off_morph_compartments_counts": off_morph_compartments_counts,
        "on_morph_compartments_counts": on_morph_compartments_counts,
        "on_morph_sig": on_morph_sig,
        "off_morph_sig": off_morph_sig,
    }

# writing results to a json file
with open(
    signature_results_dir / f"{method}_cfret_trt_signature_results.json", "w"
) as f:
    json.dump(results, f, indent=4)
