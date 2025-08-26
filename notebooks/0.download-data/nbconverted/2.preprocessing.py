#!/usr/bin/env python
# coding: utf-8

# # 2. Preprocessing Data
# 
# This notebook demonstrates how to preprocess single-cell profile data for downstream analysis. It covers the following steps:
# 
# **Overview**
# 
# - **Data Exploration**: Examining the structure and contents of the downloaded datasets
# - **Metadata Handling**: Loading experimental metadata to guide data selection and organization
# - **Feature Selection**: Applying a shared feature space for consistency across datasets
# - **Profile Concatenation**: Merging profiles from multiple experimental plates into a unified DataFrame
# - **Format Conversion**: Converting raw CSV files to Parquet format for efficient storage and access
# - **Metadata and Feature Documentation**: Saving metadata and feature information to ensure reproducibility
# 
# These preprocessing steps ensure that all datasets are standardized, well-documented, and ready for comparative and integrative analyses.

# In[1]:


import sys
import json
import pathlib
from typing import Optional

import polars as pl

sys.path.append("../../")
from utils.data_utils import split_meta_and_features


# ## Helper functions 
# 
# Contains helper function that pertains to this notebook.

# In[2]:


def load_and_concat_profiles(
    profile_dir: str | pathlib.Path,
    shared_features: Optional[list[str]] = None,
    specific_plates: Optional[list[pathlib.Path]] = None,
) -> pl.DataFrame:
    """
    Load all profile files from a directory and concatenate them into a single Polars DataFrame.

    Parameters
    ----------
    profile_dir : str or pathlib.Path
        Directory containing the profile files (.parquet).
    shared_features : Optional[list[str]], optional
        List of shared feature names to filter the profiles. If None, all features are loaded.
    specific_plates : Optional[list[pathlib.Path]], optional
        List of specific plate file paths to load. If None, all profiles in the directory are loaded.

    Returns
    -------
    pl.DataFrame
        Concatenated Polars DataFrame containing all loaded profiles.
    """
    # Ensure profile_dir is a pathlib.Path
    if isinstance(profile_dir, str):
        profile_dir = pathlib.Path(profile_dir)
    elif not isinstance(profile_dir, pathlib.Path):
        raise TypeError("profile_dir must be a string or a pathlib.Path object")

    # Validate specific_plates
    if specific_plates is not None:
        if not isinstance(specific_plates, list):
            raise TypeError("specific_plates must be a list of pathlib.Path objects")
        if not all(isinstance(path, pathlib.Path) for path in specific_plates):
            raise TypeError(
                "All elements in specific_plates must be pathlib.Path objects"
            )

    def load_profile(file: pathlib.Path) -> pl.DataFrame:
        """internal function to load a single profile file.
        """
        profile_df = pl.read_parquet(file)
        meta_cols, _ = split_meta_and_features(profile_df)
        if shared_features is not None:
            # Only select metadata and shared features
            return profile_df.select(meta_cols + shared_features)
        return profile_df

    # Use specific_plates if provided, otherwise gather all .parquet files
    if specific_plates is not None:
        # Validate that all specific plate files exist
        for plate_path in specific_plates:
            if not plate_path.exists():
                raise FileNotFoundError(f"Profile file not found: {plate_path}")
        files_to_load = specific_plates
    else:
        files_to_load = list(profile_dir.glob("*.parquet"))
        if not files_to_load:
            raise FileNotFoundError(f"No profile files found in {profile_dir}")

    # Load and concatenate profiles
    loaded_profiles = [load_profile(f) for f in files_to_load]

    # Concatenate all loaded profiles
    return pl.concat(loaded_profiles, rechunk=True)

    
def split_data(pycytominer_output: pl.DataFrame, dataset: str = "CP_and_DP") -> pl.DataFrame:
    """
    Split pycytominer output to metadata dataframe and feature values using Polars.

    Parameters
    ----------
    pycytominer_output : pl.DataFrame
        Polars DataFrame with pycytominer output
    dataset : str, optional
        Which dataset features to split,
        can be "CP" or "DP" or by default "CP_and_DP"

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with metadata and selected features
    """
    all_cols = pycytominer_output.columns

    # Get DP, CP, or both features from all columns depending on desired dataset
    if dataset == "CP":
        feature_cols = [col for col in all_cols if "CP__" in col]
    elif dataset == "DP":
        feature_cols = [col for col in all_cols if "DP__" in col]
    elif dataset == "CP_and_DP":
        feature_cols = [col for col in all_cols if "P__" in col]
    else:
        raise ValueError(
            f"Invalid dataset '{dataset}'. Choose from 'CP', 'DP', or 'CP_and_DP'."
        )

    # Metadata columns is all columns except feature columns
    metadata_cols = [col for col in all_cols if "P__" not in col]

    # Select metadata and feature columns
    selected_cols = metadata_cols + feature_cols
    
    return pycytominer_output.select(selected_cols)


# Defining the input and output directories used throughout the notebook.
# 
# > **Note:** The shared profiles utilized here are sourced from the [JUMP-single-cell](https://github.com/WayScience/JUMP-single-cell) repository. All preprocessing and profile generation steps are performed in that repository, and this notebook focuses on downstream analysis using the generated profiles.

# In[3]:


# Setting data directory
data_dir = pathlib.Path("./data").resolve(strict=True)

# Setting profiles directory
profiles_dir = (data_dir / "sc-profiles").resolve(strict=True)

# Experimental metadata
exp_metadata_path = (profiles_dir / "cpjump1" / "CPJUMP1-experimental-metadata.csv").resolve(strict=True)

# Setting feature selection path
shared_features_config_path = (
    profiles_dir / "cpjump1" / "feature_selected_sc_qc_features.json"
).resolve(strict=True)

# setting mitocheck profiles directory
mitocheck_profiles_dir = (profiles_dir / "mitocheck").resolve(strict=True)
mitocheck_norm_profiles_dir = (mitocheck_profiles_dir / "normalized_data").resolve(strict=True)

# output directories
cpjump1_output_dir = (profiles_dir / "cpjump1" / "trt-profiles").resolve()
cpjump1_output_dir.mkdir(exist_ok=True)

# Make a results folder
results_dir = pathlib.Path("./results").resolve()
results_dir.mkdir(exist_ok=True)


# Create a list of paths that only points crispr treated plates and load the shared features config file that can be found in this [repo](https://github.com/WayScience/JUMP-single-cell)

# In[4]:


# Load experimental metadata
# selecting plates that pertains to the cpjump1 CRISPR dataset
exp_metadata = pl.read_csv(exp_metadata_path)
crispr_plate_names = exp_metadata.select("Assay_Plate_Barcode").unique().to_series().to_list()
crispr_plate_paths = [
        (profiles_dir / "cpjump1" / f"{plate}_feature_selected_sc_qc.parquet").resolve(strict=True) for plate in crispr_plate_names
    ]
# Load shared features
with open(shared_features_config_path) as f:
    loaded_shared_features = json.load(f)

shared_features = loaded_shared_features["shared-features"]


# ## Preprocessing CPJUMP1 CRISPR data
# 
# Using the filtered CRISPR plate file paths and shared features configuration, we load all individual profile files and concatenate them into a single comprehensive DataFrame. This step combines data from multiple experimental plates while maintaining the consistent feature space defined by the shared features list.
# 
# The concatenation process ensures:
# - All profiles use the same feature set for downstream compatibility
# - Metadata columns are preserved across all plates
# - Data integrity is maintained during the merge operation

# In[5]:


# Loading crispr profiles with shared features and concat into a single DataFrame
concat_output_path = cpjump1_output_dir / "cpjump1_crispr_trt_profiles.parquet"

if concat_output_path.exists():
    print("concat profiles already exists, loading from file")
else:
    loaded_profiles = load_and_concat_profiles(
        profile_dir=profiles_dir,
        specific_plates=crispr_plate_paths,
        shared_features=shared_features
    )

    # Add index column 
    loaded_profiles = loaded_profiles.with_row_index("index")

    # Split meta and features
    meta_cols, features_cols = split_meta_and_features(loaded_profiles)

    # Saving metadata and features of the concat profile into a json file
    meta_features_dict = {
        "concat-profiles": {
            "meta-features": meta_cols,
            "shared-features": features_cols
        }
    }
    with open(cpjump1_output_dir / "concat_profiles_meta_features.json", "w") as f:
        json.dump(meta_features_dict, f, indent=4)

    # filter profiles that contains treatment data
    loaded_profiles = loaded_profiles.filter(pl.col("Metadata_pert_type") == "trt")

    # save as parquet
    loaded_profiles.write_parquet(concat_output_path)


# ## Preprocessing MitoCheck
# 
# This section processes the MitoCheck dataset by loading the training and negative control data from compressed CSV files. The original CSV format is converted to Parquet for consistency with other processed data and improved performance.
# 
# Key preprocessing steps include:
# - **Loading training data**: Reading the main MitoCheck profiles containing various phenotypic classes
# - **Processing negative controls**: Loading control samples and adding phenotypic class labels
# - **Feature filtering**: Extracting only Cell Profiler (CP) features to match the CPJUMP1 dataset structure
# - **Standardization**: Ensuring consistent column naming and metadata structure across datasets
# - **Feature alignment**: Identifying shared features between training and control data for unified analysis
# 
# The processed data is saved in Parquet format with optimized storage and maintained metadata integrity, enabling efficient downstream comparative analysis between MitoCheck and CPJUMP1 datasets.

# In[6]:


output_path = (cpjump1_output_dir / "cpjump1_crispr_trt_profiles.parquet").resolve()
if output_path.exists():
    print("Output path already exists.")
else:

    # load in mitocheck profiles and save as parquet
    mitocheck_profile = pl.read_csv(
        mitocheck_norm_profiles_dir / "training_data.csv.gz",
    )

    # drop first column by index
    # This is done to remove the index column that is not needed
    mitocheck_profile = mitocheck_profile.select(mitocheck_profile.columns[1:])

    # loading in negative control profiles
    mitocheck_neg_control_profiles = pl.read_csv(
        mitocheck_norm_profiles_dir / "negative_control_data.csv.gz",
        )

    # insert new column "Mitocheck_Phenotypic_Class"
    mitocheck_neg_control_profiles = mitocheck_neg_control_profiles.with_columns(
        pl.lit("negcon").alias("Mitocheck_Phenotypic_Class")
    ).select(["Mitocheck_Phenotypic_Class"] + mitocheck_neg_control_profiles.columns)


# Filter Cell Profiler (CP) features and preprocess columns by removing the "CP__" prefix to standardize feature names for downstream analysis.

# In[7]:


# split profiles 
cp_mitocheck_profile = split_data(mitocheck_profile, dataset="CP")
cp_mitocheck_neg_control_profiles = split_data(mitocheck_neg_control_profiles, dataset="CP")

# rename columns to remove "CP__" prefix and replace it with "NoCompartment_"
cp_mitocheck_profile = cp_mitocheck_profile.rename(
    lambda x: x.replace("CP__", "") if "CP__" in x else x
)
cp_mitocheck_neg_control_profiles = cp_mitocheck_neg_control_profiles.rename(
    lambda x: x.replace("CP__", "") if "CP__" in x else x
)


# Splitting the metadata and feature columns for each dataset to enable targeted downstream analysis and ensure consistent data structure across all profiles.

# In[8]:


# get metadata features
cp_mitocheck_profile_meta = cp_mitocheck_profile.columns[:13]
cp_mitocheck_neg_control_profiles_meta = cp_mitocheck_neg_control_profiles.columns[:12]

# morphology features 
cp_mitocheck_profile_features = cp_mitocheck_profile.drop(cp_mitocheck_profile_meta).columns
cp_mitocheck_neg_control_profiles_features = cp_mitocheck_neg_control_profiles.drop(cp_mitocheck_neg_control_profiles_meta).columns


# Identify the shared metadata and feature columns between the two datasets, concatenate them into a unified DataFrame containing only these shared columns, and save the result as a Parquet file for downstream analysis.

# In[9]:


# find shared metadata columns
cp_mitocheck_profile_meta_cols = set(cp_mitocheck_profile_meta)
cp_mitocheck_neg_control_profiles_meta_cols = set(cp_mitocheck_neg_control_profiles_meta)
shared_meta = cp_mitocheck_profile_meta_cols.intersection(cp_mitocheck_neg_control_profiles_meta_cols)

# find shared feature columns
cp_mitocheck_profile_features_cols = set(cp_mitocheck_profile_features)
cp_mitocheck_neg_control_profiles_features_cols = set(cp_mitocheck_neg_control_profiles_features)
shared_features = cp_mitocheck_profile_features_cols.intersection(cp_mitocheck_neg_control_profiles_features_cols)
    
# concat both shared metadata and features
final_shared_features = list(shared_meta) + list(shared_features)

# select only shared metadata and features from both profiles and concat
cp_mitocheck_profiles_path = mitocheck_profiles_dir / "concat_mitocheck_cp_profiles_shared_feats.parquet"
pl.concat([
    cp_mitocheck_profile.select(final_shared_features),
    cp_mitocheck_neg_control_profiles.select(final_shared_features)
], rechunk=True
).write_parquet(cp_mitocheck_profiles_path)

