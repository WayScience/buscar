#!/usr/bin/env python

# # Extracting Morphological Signatures
#
# In this notebook, we extract morphological signatures associated with two distinct cell states:
# - **On-morphology features**: Features that significantly change with the cell state
# - **Off-morphology features**: Features that do not show significant changes with the cell state
#
# We identify and categorize features as either on- or off-morphology signatures using a systematic workflow.
# This approach is applied to three datasets: Pilot-CFReT, MitoCheck, and CPJUMP1 (CRISPR only).
#
# The analysis workflow consists of:
# 1. Loading morphological profiles from negative and positive controls
# 2. Comparing profiles to identify differentially affected features
# 3. Categorizing features into on-morphology (significantly changed) and off-morphology (unchanged) signatures
# 4. Generating consensus signatures across multiple control combinations

# In[1]:


import itertools
import json
import pathlib

# Import required libraries
import sys
from collections import defaultdict

import polars as pl
import tqdm

# Import custom utility functions
sys.path.append("../../")
from utils.data_utils import generate_consensus_signatures, split_meta_and_features
from utils.io_utils import load_profiles
from utils.signatures import get_signatures

# In[2]:


# Analysis parameters
seed = 0  # Random seed for reproducibility
method = "ks_test"  # Statistical test method for signature identification


# ## Setting Input and Output Paths
#
# Configure file paths for data loading and results storage.

# In[3]:


# setting data directory paths
data_dir = pathlib.Path("../0.download-data/data/").resolve(strict=True)
data_results_dir = pathlib.Path("../0.download-data/results/").resolve(strict=True)

# setting directory path were all the single-cell profiles are
data_sc_profiles_path = (data_dir / "sc-profiles").resolve(strict=True)

# setting all profile directory paths
cpjump1_profiles_dir_path = (data_sc_profiles_path / "cpjump1").resolve(strict=True)
cfret_profiles_dir_path = (data_sc_profiles_path / "cfret").resolve(strict=True)
mitocheck_profiles_dir_path = (data_sc_profiles_path / "mitocheck").resolve(strict=True)

# setting CPJUMP1 data paths and configs
cpjump1_negcon_profile_path = list(
    (cpjump1_profiles_dir_path / "negcon").resolve(strict=True).glob("*.parquet")
)
cpjump1_poscon_profile_path = (
    cpjump1_profiles_dir_path / "poscon" / "poscon_cp_df.parquet"
).resolve(strict=True)
cpjump1_shared_features_config_path = (
    data_sc_profiles_path / "cpjump1" / "feature_selected_sc_qc_features.json"
).resolve(strict=True)

# setting Mitocheck data paths
poscon_mitocheck_profile_path = (
    mitocheck_profiles_dir_path / "poscon_mitocheck_cp_profiles.parquet"
).resolve(strict=True)
negcon_mitocheck_profile_path = (
    mitocheck_profiles_dir_path / "negcon_mitocheck_cp_profiles.parquet"
).resolve(strict=True)
mitocheck_feature_space_config = (
    mitocheck_profiles_dir_path / "mitocheck_feature_space_configs.json"
).resolve(strict=True)

# setting CFReT profile path
cfret_plate_path = (
    cfret_profiles_dir_path / "localhost230405150001_sc_feature_selected.parquet"
).resolve(strict=True)

# Creating results directory structure
results_dir = pathlib.Path("results").resolve()
results_dir.mkdir(exist_ok=True)

# Creating "signature_results" subdirectory within the results directory
signature_results_dir = (results_dir / "signature_results").resolve()
signature_results_dir.mkdir(exist_ok=True)


# ## Loading Morphological Profiles
#
# Load the morphological profiles for negative and positive controls from the CPJUMP1 dataset.

# ### Loading CPJUMP1 CRISPR Profiles
#
# Load the negative and positive control profiles from the CPJUMP1 dataset.

# In[4]:


# Loading shared features configuration
with open(cpjump1_shared_features_config_path) as f:
    feature_space = json.load(f)
shared_features = feature_space["shared-features"]

# Loading the 10 randomly subsampled negative control profiles
cpjump1_negcon_df = []
for negcon_df_path in cpjump1_negcon_profile_path:
    # Extract seed ID from file path
    seed_id = negcon_df_path.stem.split("_")[-1]

    # Load profile data
    loaded_negcon_df = load_profiles(negcon_df_path)

    # Insert seed ID at the beginning of the dataframe
    loaded_negcon_df = loaded_negcon_df.with_columns(
        pl.lit(seed_id).alias("Metadata_seed_id")
    ).select(["Metadata_seed_id"] + loaded_negcon_df.columns)

    # Append to list
    cpjump1_negcon_df.append(loaded_negcon_df)

# Concatenate all negative control dataframes
cpjump1_negcon_df = pl.concat(cpjump1_negcon_df)

# Load positive control profiles
cpjump1_poscon_df = load_profiles(cpjump1_poscon_profile_path)

# Display the unique positive control types
print("Unique positive control types:")
print(cpjump1_poscon_df["Metadata_gene"].unique().to_list())


# ### Loading in MitoCheck data

# In[5]:


mitocheck_poscon_df = pl.read_parquet(poscon_mitocheck_profile_path)
mitocheck_negcon_df = pl.read_parquet(negcon_mitocheck_profile_path)

# loading in feature space
with open(mitocheck_feature_space_config) as f:
    mitocheck_feature_space = json.load(f)

# displaying the poscon genes
print(f"These are the positive control gene: {mitocheck_poscon_df["Metadata_Gene"].unique().to_list()}")
print(f"Dataframe shape for poscon: {mitocheck_poscon_df.shape}")
print(f"Dataframe shape for negcon: {mitocheck_negcon_df.shape}")


# ### Loading CFReT Dataset

# In[6]:


# More efficient loading and filtering with lazy evaluation
cfret_df = load_profiles(cfret_plate_path)

# split features
cfret_meta, cfret_feats = split_meta_and_features(cfret_df)

# Filter and collect only what you need
cfret_negcon_df = cfret_df.filter(pl.col("Metadata_treatment") == "DMSO")
cfret_poscon_df = cfret_df.filter(pl.col("Metadata_treatment") == "TGFRi")


# ## Generating On and Off Morphology Signatures

# ### Signature Generation Process for CPJUMP CRISPR data
#
# Create on and off morphological signatures using different positive controls and randomly sampled negative control profiles. This process compares each negative control sample against each positive control to identify consistent morphological changes.

# In[7]:


# Setting save path
cpjump1_save_path = (
    signature_results_dir / f"{method}_cpjump1_negcon_poscon_signatures.json"
).resolve()

# If the file does not exist, iterate all combinations and identify signatures
# If the file exists, skip the whole process and just load the saved file
if not cpjump1_save_path.exists():
    # Creating all possible combinations between the randomly selected negative
    # control profiles and positive controls
    negcon_and_poscon_combinations = list(
        itertools.product(
            cpjump1_negcon_df["Metadata_seed_id"].unique().to_list(),
            cpjump1_poscon_df["Metadata_gene"].unique().to_list(),
        )
    )

    # Iterate through each randomly sampled negative control and positive control combination
    cpjump1_signature_results = defaultdict(
        lambda: None
    )  # Used for storing comparisons and signatures
    for negcon_seed, poscon_gene in tqdm.tqdm(
        negcon_and_poscon_combinations, desc="Processing combinations"
    ):
        # Select negative control profile for current seed
        selected_negcon_df = cpjump1_negcon_df.filter(
            pl.col("Metadata_seed_id") == negcon_seed
        )

        # Select positive control profile for current gene
        selected_poscon_df = cpjump1_poscon_df.filter(
            pl.col("Metadata_gene") == poscon_gene
        )

        # Find the morphological signatures
        on_sig, off_sig = get_signatures(
            ref_profiles=selected_negcon_df,
            exp_profiles=selected_poscon_df,
            morph_feats=shared_features,
            test_method=method,
            seed=seed,
        )

        # Process signatures and store in dictionary
        cpjump1_signature_results[f"{negcon_seed}_negcon_{poscon_gene}_poscon"] = {
            "controls": {"negative": negcon_seed, "positive": poscon_gene},
            "signatures": {"on": on_sig, "off": off_sig},
            "meta": {
                "total-on-signatures": len(on_sig),
                "total-off-signatures": len(off_sig),
            },
        }

    # Save results to file
    with open(cpjump1_save_path, "w") as f:
        json.dump(dict(cpjump1_signature_results), f, indent=4)

else:
    with open(cpjump1_save_path) as f:
        cpjump1_signature_results = json.load(f)


# After generating the signatures for all combinations, the next step is to create consensus on and off morphological signatures. Due to the multiple randomly sampled negative controls present, we need to find the average morphological features affected per positive control. This consensus approach will help us analyze and compare across different positive controls to determine if we are capturing known biological effects.

# In[ ]:


# generate consensus signatures
consensus_siagnatures = generate_consensus_signatures(
    cpjump1_signature_results, features=shared_features, min_consensus_threshold=0.5
)

# save to json file
with open(
    (signature_results_dir / f"{method}_cpjump1_consensus_signatures.json").resolve(),
    "w",
) as f:
    json.dump(consensus_siagnatures, f, indent=4)


# ### Signature Generation Process for MitoCheck data

# In[9]:


# set path
mitocheck_save_path = (
    signature_results_dir / f"{method}_mitocheck_signatures.json"
).resolve()

if not mitocheck_save_path.exists():
    mitocheck_signature_results = defaultdict(lambda: None)
    for poscon_gene in tqdm.tqdm(mitocheck_poscon_df["Metadata_Gene"].unique().to_list(), desc="Processing MitoCheck genes"):
        # create poscon dataframe based on positive control gene
        selected_poscon_df = mitocheck_poscon_df.filter(
            pl.col("Metadata_Gene") == poscon_gene
        )

        # Find the morphological signatures
        on_sig, off_sig = get_signatures(
            ref_profiles=mitocheck_negcon_df,
            exp_profiles=selected_poscon_df,
            morph_feats=mitocheck_feature_space["shared-features"],
            test_method=method,
            seed=seed,
        )

        # Process signatures and store in dictionary
        mitocheck_signature_results[f"mitocheck_negcon_{poscon_gene}_poscon"] = {
            "controls": {"negative": "DMSO", "positive": poscon_gene},
            "signatures": {"on": on_sig, "off": off_sig},
            "meta": {
                "total-on-signatures": len(on_sig),
                "total-off-signatures": len(off_sig),
            },
        }

    # save signatures
    with open(mitocheck_save_path, "w") as f:
        json.dump(dict(mitocheck_signature_results), f, indent=4)
else:
    # read from json file
    with open(mitocheck_save_path) as f:
        mitocheck_signature_results = json.load(f)


# ### Signature Generation Process for CFReT data

# In[10]:


# save path
cfret_save_path = (signature_results_dir / f"{method}_cfret_signatures.json").resolve()

if not cfret_save_path.exists():

    # set up results dictionary
    cfret_signature_results = defaultdict(lambda: None)

    # Find the morphological signatures
    on_sig, off_sig = get_signatures(
        ref_profiles=cfret_negcon_df,
        exp_profiles=cfret_poscon_df,
        morph_feats=cfret_feats,
        test_method=method,
        seed=seed,
    )

    # save results
    cfret_signature_results["cfret_negcon_TGFRi_poscon"] = {
        "controls": {"negative": "DMSO", "positive": "TGFRi"},
        "signatures": {"on": on_sig, "off": off_sig},
        "meta": {
            "total-on-signatures": len(on_sig),
            "total-off-signatures": len(off_sig),
        },
    }

    # save results in json path
    with open(cfret_save_path, "w") as f:
        json.dump(dict(cfret_signature_results), f, indent=4)

else:
    with open(cfret_save_path) as f:
        cfret_signature_results = json.load(f)
