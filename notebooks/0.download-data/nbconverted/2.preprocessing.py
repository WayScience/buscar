#!/usr/bin/env python

# # 2. Preprocessing Data
#
# In this notebook, we explore the contents of the downloaded dataset and perform preprocessing steps to prepare the data for downstream analysis.
# **Overview**
#
# We focus on concatenating profiles from plates containing CRISPR knockdown experiments. The workflow includes:
#
# 1. **Plate Selection**: Loading only plates with CRISPR knockdown wells from the experimental metadata
# 2. **Feature Space Reduction**: Using the shared feature space defined in the [JUMP-single-cell repository](https://github.com/WayScience/JUMP-single-cell)
# 3. **Data Concatenation**: Combining all selected plates into a single DataFrame with consistent features
# 4. **Metadata Preservation**: Generating a JSON record containing metadata and shared feature information for reproducibility
#
# This preprocessing ensures all profiles share the same feature space and are ready for comparative analysis across different experimental conditions.

# In[1]:


import json
import pathlib
import sys

import polars as pl

sys.path.append("../../")
from utils.data_utils import split_meta_and_features

# ## Helper functions
#
# Contains helper function that pertains to this notebook.

# In[2]:


def load_and_concat_profiles(
    profile_dir: str | pathlib.Path,
    shared_features: list[str] | None = None,
    specific_plates: list[pathlib.Path] | None = None,
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


# Defining the input and output directories used throughout the notebook.
#
# > **Note:** The shared profiles utilized here are sourced from the [JUMP-single-cell](https://github.com/WayScience/JUMP-single-cell) repository. All preprocessing and profile generation steps are performed in that repository, and this notebook focuses on downstream analysis using the generated profiles.

# In[3]:


# Setting data directory
data_dir = pathlib.Path("./data").resolve(strict=True)

# Setting profiles directory
profiles_dir = (data_dir / "sc-profiles").resolve(strict=True)

# Experimental metadata
exp_metadata_path = (data_dir / "CPJUMP1-experimental-metadata.csv").resolve(strict=True)
exp_platemaps_path = (data_dir / "CPJUMP1-plate-maps.csv").resolve(strict=True)

# Setting feature selection path
shared_features_config_path = (data_dir / "feature_selected_sc_qc_features.json").resolve(strict=True)

# Make a results folder
results_dir = pathlib.Path("./results").resolve()
results_dir.mkdir(exist_ok=True)


# Create a list of paths that only points crispr treated plates and load the shared features config file that can be found in this [repo](https://github.com/WayScience/JUMP-single-cell)

# In[4]:


# Load experimental metadata
exp_metadata = pl.read_csv(exp_metadata_path)
platemap_df = pl.read_csv(exp_platemaps_path)
crispr_plate_names = exp_metadata.select("Assay_Plate_Barcode").unique().to_series().to_list()
crispr_plate_paths = [
        (profiles_dir / f"{plate}_feature_selected_sc_qc.parquet").resolve(strict=True) for plate in crispr_plate_names
    ]
# Load shared features
with open(shared_features_config_path) as f:
    loaded_shared_features = json.load(f)

shared_features = loaded_shared_features["shared-features"]


# Using the filtered CRISPR plate file paths and shared features configuration, we load all individual profile files and concatenate them into a single comprehensive DataFrame. This step combines data from multiple experimental plates while maintaining the consistent feature space defined by the shared features list.
#
# The concatenation process ensures:
# - All profiles use the same feature set for downstream compatibility
# - Metadata columns are preserved across all plates
# - Data integrity is maintained during the merge operation

# In[5]:


# Loading crispr profiles with shared features and concat into a single DataFrame
loaded_profiles = load_and_concat_profiles(
    profile_dir=profiles_dir,
    specific_plates=crispr_plate_paths,
    shared_features=shared_features
)

# Add index column
loaded_profiles = loaded_profiles.with_row_index("index")

# Split meta and features
meta_cols, features_cols = split_meta_and_features(loaded_profiles)


# Generating a complete platemap containing well positions and all relevant metadata.

# In[6]:


# select columns that only start with Metadata
meta_cols = [col for col in loaded_profiles.columns if col.startswith("Metadata")]
features_cols = [col for col in loaded_profiles.columns if col.startswith("Feature")]

treatment_metadata_df = loaded_profiles[["Metadata_broad_sample", "Metadata_Well", "Metadata_gene", "Metadata_pert_type", "Metadata_control_type"]]

# group these duplicates based on Metadata_broad_sample and Metadata_Well
complete_crispr_platemap_df = (
	treatment_metadata_df
	.group_by(["Metadata_broad_sample", "Metadata_Well"])
	.agg([
		pl.col("Metadata_gene").first().alias("Metadata_gene"),
		pl.col("Metadata_pert_type").first().alias("Metadata_pert_type"),
		pl.col("Metadata_control_type").first().alias("Metadata_control_type"),
	])
)[["Metadata_Well", "Metadata_broad_sample", "Metadata_gene", "Metadata_pert_type", "Metadata_control_type"]].sort(by="Metadata_Well")


# Saving the concatenated CRISPR profiles and feature space information

# In[7]:


# Saving metadata and features of the concat profile into a json file
meta_features_dict = {
    "concat-profiles": {
        "meta-features": meta_cols,
        "shared-features": features_cols
    }
}
with open(results_dir / "concat_profiles_meta_features.json", "w") as f:
    json.dump(meta_features_dict, f, indent=4)

# save complete plate map
complete_crispr_platemap_df.write_csv(
    results_dir / "complete_crispr_platemap.csv")

# Save the concated profiles
loaded_profiles.write_parquet(
    results_dir / "concat_crispr_profiles.parquet")
