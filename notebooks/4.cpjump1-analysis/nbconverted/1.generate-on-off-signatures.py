#!/usr/bin/env python

# # 1. generating on and off signatures
#
# In this notebook, we generate on and off signatures for the compound CPJUMP1 [dataset](https://github.com/carpenter-singh-lab/2024_Chandrasekaran_NatureMethods). These signatures are generated separately for each positive control, meaning that every signature derived from the JUMP dataset is specific to an individual positive control.
#
# The dataset is split by cell type, specifically U2OS and A549 cells, and signatures are generated independently for each cell type.
#

# In[1]:


import itertools
import json
import pathlib
import sys
from collections import defaultdict

import polars as pl
import tqdm

# add buscar function
sys.path.append("../../")
from utils.data_utils import generate_consensus_signatures
from utils.io_utils import load_configs, load_profiles
from utils.signatures import get_signatures

# Parameters
#
# The following parameters are used throughout this notebook to ensure consistency and reproducibility:
#
# - **`seed`** (int): Random seed value used for reproducibility in statistical tests and data sampling.
# - **`method`** (str): The statistical test method (e.g., `ks_test`) employed to identify significant "on" and "off" morphological signatures.
#

# In[2]:


seed = 0
method = "ks_test"


# Setting input and output paths

# In[3]:


# set data data
data_dir = pathlib.Path("../0.download-data/data/sc-profiles/").resolve(strict=True)

# create a results dir
results_dir = pathlib.Path("./results")
results_dir.mkdir(exist_ok=True)

# set an output directory for signatures
signature_results_dir = (results_dir / "signatures").resolve()
signature_results_dir.mkdir(exist_ok=True)

# setting cpjump1 experimental data
cpjump1_experimental_data_path = (
    data_dir / "cpjump1/CPJUMP1-experimental-metadata.csv"
).resolve(strict=True)

# set paths to cpjump1 poscon and negcon profiles
cpjump1_negcon_profile_path = list(
    (data_dir / "cpjump1/negcon").resolve(strict=True).glob("*.parquet")
)
cpjump1_poscon_profile_path = (
    data_dir / "cpjump1/poscon" / "poscon_cp_df.parquet"
).resolve(strict=True)

# set path to config file that has the shared cpjump1 features
cpjump1_shared_features_path = (
    data_dir / "cpjump1/feature_selected_sc_qc_features.json"
).resolve(strict=True)

# set cpjump path
crispr_cpjump1_path = (
    data_dir / "cpjump1/cpjump1_compound_concat_profiles.parquet"
).resolve(strict=True)


# In[4]:


cpjump1_experimental_data = pl.read_csv(cpjump1_experimental_data_path)


# In[5]:


cpjump1_experimental_data.filter(pl.col("Cell_type") == "U2OS")


# Identify the plates containing U2OS and A549 cells treated for 144 hours.

# In[6]:


# Split the dataset by cell type and treatment duration
# Filter U2OS cells (all records)
cpjump1_u2os_exp_metadata = cpjump1_experimental_data.filter(
    pl.col("Cell_type") == "U2OS"
)

# Filter A549 cells with density of 100 for consistency
cpjump1_a549_exp_metadata = cpjump1_experimental_data.filter(
    (pl.col("Cell_type") == "A549") & (pl.col("Density") == 100)
)

# Extract plate identifiers for each cell type
u2os_plates = cpjump1_u2os_exp_metadata["Assay_Plate_Barcode"].unique().to_list()
a549_plates = cpjump1_a549_exp_metadata["Assay_Plate_Barcode"].unique().to_list()

# Display the extracted plates for verification
print(f"U2OS plates: {u2os_plates}")
print(f"A549 plates: {a549_plates}")


# loading in dataset

# In[ ]:


# load shared features
shared_features = load_configs(cpjump1_shared_features_path)["shared-features"]

# Loading the 10 randomly subsampled negative control profiles
cpjump1_negcon_df = []
for negcon_df_path in cpjump1_negcon_profile_path:
    # Extract seed ID from file path
    seed_id = negcon_df_path.stem.split("_")[-1]

    # Load profile data
    # filter to the selected plates
    loaded_negcon_df = load_profiles(negcon_df_path).filter(
        pl.col("Metadata_Plate").is_in(u2os_plates + a549_plates)
    )

    # Insert seed ID at the beginning of the dataframe
    loaded_negcon_df = loaded_negcon_df.with_columns(
        pl.lit(seed_id).alias("Metadata_seed_id")
    ).select(["Metadata_seed_id"] + loaded_negcon_df.columns)

    # Append to list
    cpjump1_negcon_df.append(loaded_negcon_df)

# Concatenate all negative control dataframes but set to the select plates
cpjump1_negcon_df = pl.concat(cpjump1_negcon_df)

# load positive controls and filter to selected plates
cpjump1_poscon_df = load_profiles(cpjump1_poscon_profile_path).filter(
    pl.col("Metadata_Plate").is_in(u2os_plates + a549_plates)
)

# Display the unique positive control types
all_poscon_trts = cpjump1_poscon_df["Metadata_pert_iname"].unique().to_list()
print("Unique positive control types:")
print(f"Number of unique positive control treatments: {len(all_poscon_trts)}")
print(all_poscon_trts)


# For each positive control compound, we generate "on" and "off" morphological signatures by comparing profiles from all 10 randomly selected DMSO (negative control) seeds to the corresponding positive control profiles.

# On and off signatures for the u2os cells

# In[8]:


# Setting save path
cpjump1_u2os_sig_save_path = (
    signature_results_dir / f"{method}_cpjump1_u2os_negcon_poscon_signatures.json"
).resolve()

# set negative controls for U2OS plates
cpjump1_u2os_negcon_df = cpjump1_negcon_df.filter(
    pl.col("Metadata_Plate").is_in(u2os_plates)
)
cpjump1_u2os_poscon_df = cpjump1_poscon_df.filter(
    pl.col("Metadata_Plate").is_in(u2os_plates)
    & (pl.col("Metadata_pert_iname").is_in(all_poscon_trts))
)

# If the file does not exist, iterate all combinations and identify signatures
# If the file exists, skip the whole process and just load the saved file
if not cpjump1_u2os_sig_save_path.exists():
    # Creating all possible combinations between the randomly selected negative
    # control profiles and positive controls
    negcon_and_poscon_combinations = list(
        itertools.product(
            cpjump1_u2os_negcon_df["Metadata_seed_id"].unique().to_list(),
            cpjump1_u2os_poscon_df["Metadata_pert_iname"].unique().to_list(),
        )
    )

    # Iterate through each randomly sampled negative control and positive control combination
    cpjump1_u2os_signature_results = defaultdict(
        lambda: None
    )  # Used for storing comparisons and signatures
    for negcon_seed, poscon_trt in tqdm.tqdm(
        negcon_and_poscon_combinations, desc="Processing combinations"
    ):
        # Select negative control profile for current seed
        selected_negcon_df = cpjump1_u2os_negcon_df.filter(
            pl.col("Metadata_seed_id") == negcon_seed
        )

        # Select positive control profile for current gene
        selected_poscon_df = cpjump1_u2os_poscon_df.filter(
            pl.col("Metadata_pert_iname") == poscon_trt
        )

        # Find the morphological signatures
        on_sig, off_sig, _ = get_signatures(
            ref_profiles=selected_negcon_df,
            exp_profiles=selected_poscon_df,
            morph_feats=shared_features,
            test_method=method,
            seed=seed,
        )

        # Process signatures and store in dictionary
        cpjump1_u2os_signature_results[f"{negcon_seed}_negcon_{poscon_trt}_poscon"] = {
            "controls": {"negative": negcon_seed, "positive": poscon_trt},
            "signatures": {"on": on_sig, "off": off_sig},
            "meta": {
                "total-on-signatures": len(on_sig),
                "total-off-signatures": len(off_sig),
            },
        }

    # Save results to file
    with open(cpjump1_u2os_sig_save_path, "w") as f:
        json.dump(dict(cpjump1_u2os_signature_results), f, indent=4)

else:
    with open(cpjump1_u2os_sig_save_path) as f:
        cpjump1_u2os_signature_results = json.load(f)


# On and Off signatures for the a549 cells

# In[9]:


# Setting save path
cpjump1_a549_sig_save_path = (
    signature_results_dir / f"{method}_cpjump1_a549_negcon_poscon_signatures.json"
).resolve()

# set negative controls for A549 plates
cpjump1_a549_negcon_df = cpjump1_negcon_df.filter(
    pl.col("Metadata_Plate").is_in(a549_plates)
)
cpjump1_a549_poscon_df = cpjump1_poscon_df.filter(
    pl.col("Metadata_Plate").is_in(a549_plates)
    & (pl.col("Metadata_pert_iname").is_in(all_poscon_trts))
)
# If the file does not exist, iterate all combinations and identify signatures
# If the file exists, skip the whole process and just load the saved file
if not cpjump1_a549_sig_save_path.exists():
    # Creating all possible combinations between the randomly selected negative
    # control profiles and positive controls
    negcon_and_poscon_combinations = list(
        itertools.product(
            cpjump1_a549_negcon_df["Metadata_seed_id"].unique().to_list(),
            cpjump1_a549_poscon_df["Metadata_pert_iname"].unique().to_list(),
        )
    )

    # Iterate through each randomly sampled negative control and positive control combination
    cpjump1_a549_signature_results = defaultdict(
        lambda: None
    )  # Used for storing comparisons and signatures
    for negcon_seed, poscon_trt in tqdm.tqdm(
        negcon_and_poscon_combinations, desc="Processing combinations"
    ):
        # Select negative control profile for current seed
        selected_negcon_df = cpjump1_a549_negcon_df.filter(
            pl.col("Metadata_seed_id") == negcon_seed
        )

        # Select positive control profile for current gene
        selected_poscon_df = cpjump1_a549_poscon_df.filter(
            pl.col("Metadata_pert_iname") == poscon_trt
        )

        # Find the morphological signatures
        on_sig, off_sig, _ = get_signatures(
            ref_profiles=selected_negcon_df,
            exp_profiles=selected_poscon_df,
            morph_feats=shared_features,
            test_method=method,
            seed=seed,
        )

        # Process signatures and store in dictionary
        cpjump1_a549_signature_results[f"{negcon_seed}_negcon_{poscon_trt}_poscon"] = {
            "controls": {"negative": negcon_seed, "positive": poscon_trt},
            "signatures": {"on": on_sig, "off": off_sig},
            "meta": {
                "total-on-signatures": len(on_sig),
                "total-off-signatures": len(off_sig),
            },
        }

    # Save results to file
    with open(cpjump1_a549_sig_save_path, "w") as f:
        json.dump(dict(cpjump1_a549_signature_results), f, indent=4)

else:
    with open(cpjump1_a549_sig_save_path) as f:
        cpjump1_a549_signature_results = json.load(f)


# Next we generate consensus on and off signatures for each positive control (poscon) gene by aggregating results from 10 independent negative control (DMSO) subsamples. Features included in the consensus signature are those identified as significant in at least 80% of comparisons (≥8 out of 10), highlighting robust and reproducible morphological differences.
#

# In[10]:


# Generate consensus on and off signatures of U2OS cpjump1 results
u2os_cpjump1_consensus_signatures = generate_consensus_signatures(
    cpjump1_u2os_signature_results,
    features=shared_features,
    min_consensus_threshold=0.80,
)
u2os_save_path = (
    signature_results_dir
    / f"{method}_u2os_cpjump1_consensus_signatures_per_poscon.json"
)

with open(u2os_save_path, mode="w") as f:
    json.dump(u2os_cpjump1_consensus_signatures, f, indent=4)

# Generate consensus on and off signatures of A549 cpjump1 results
a549_cpjump1_consensus_signatures = generate_consensus_signatures(
    cpjump1_a549_signature_results,
    features=shared_features,
    min_consensus_threshold=0.80,
)
save_path = (
    signature_results_dir
    / f"{method}_a549_cpjump1_consensus_signatures_per_poscon.json"
)

with open(save_path, mode="w") as f:
    json.dump(a549_cpjump1_consensus_signatures, f, indent=4)
