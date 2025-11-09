#!/usr/bin/env python

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

# In[2]:


import json
import pathlib
import sys

import polars as pl

sys.path.append("../../")
from utils.data_utils import add_cell_id_hash, split_meta_and_features

# ## Helper functions
#
# Contains helper function that pertains to this notebook.

# In[3]:


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
        """internal function to load a single profile file."""
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


def split_data(
    pycytominer_output: pl.DataFrame, dataset: str = "CP_and_DP"
) -> pl.DataFrame:
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


def remove_feature_prefixes(df: pl.DataFrame, prefix: str = "CP__") -> pl.DataFrame:
    """
    Remove feature prefixes from column names in a DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with prefixed column names
    prefix : str, default "CP__"
        Prefix to remove from column names

    Returns
    -------
    pl.DataFrame
        DataFrame with cleaned column names
    """
    return df.rename(lambda x: x.replace(prefix, "") if prefix in x else x)


# Defining the input and output directories used throughout the notebook.
#
# > **Note:** The shared profiles utilized here are sourced from the [JUMP-single-cell](https://github.com/WayScience/JUMP-single-cell) repository. All preprocessing and profile generation steps are performed in that repository, and this notebook focuses on downstream analysis using the generated profiles.

# In[ ]:


# Setting data directory
data_dir = pathlib.Path("./data").resolve(strict=True)

# Setting profiles directory
profiles_dir = (data_dir / "sc-profiles").resolve(strict=True)

# Experimental metadata
exp_metadata_path = (
    profiles_dir / "cpjump1" / "CPJUMP1-experimental-metadata.csv"
).resolve(strict=True)

# Setting CFReT profiles directory
cfret_profiles_dir = (profiles_dir / "cfret").resolve(strict=True)
cfret_profiles_path = (
    cfret_profiles_dir / "localhost230405150001_sc_feature_selected.parquet"
).resolve(strict=True)

# Setting feature selection path
shared_features_config_path = (
    profiles_dir / "cpjump1" / "feature_selected_sc_qc_features.json"
).resolve(strict=True)

# setting mitocheck profiles directory
mitocheck_profiles_dir = (profiles_dir / "mitocheck").resolve(strict=True)
mitocheck_norm_profiles_dir = (mitocheck_profiles_dir / "normalized_data").resolve(
    strict=True
)

# output directories
cpjump1_output_dir = (profiles_dir / "cpjump1").resolve()
cpjump1_output_dir.mkdir(exist_ok=True)

# Make a results folder
results_dir = pathlib.Path("./results").resolve()
results_dir.mkdir(exist_ok=True)


# Create a list of paths that only points crispr treated plates and load the shared features config file that can be found in this [repo](https://github.com/WayScience/JUMP-single-cell)

# In[6]:


# Load experimental metadata
# selecting plates that pertains to the cpjump1 CRISPR dataset
exp_metadata = pl.read_csv(exp_metadata_path)
crispr_plate_names = (
    exp_metadata.select("Assay_Plate_Barcode").unique().to_series().to_list()
)
crispr_plate_paths = [
    (profiles_dir / "cpjump1" / f"{plate}_feature_selected_sc_qc.parquet").resolve(
        strict=True
    )
    for plate in crispr_plate_names
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
# - Adding a unique cell id has column `Metadata_cell_id`

# In[7]:


# Loading crispr profiles with shared features and concat into a single DataFrame
concat_output_path = (
    cpjump1_output_dir / "cpjump1_crispr_concat_profiles.parquet"
).resolve()

cjump_profiles = load_and_concat_profiles(
    profile_dir=profiles_dir,
    specific_plates=crispr_plate_paths,
    shared_features=shared_features,
)

# create an index columm and unique cell ID based on features of a single profiles
jump_profiles = add_cell_id_hash(cjump_profiles)

# Split meta and features
meta_cols, features_cols = split_meta_and_features(jump_profiles)

# Saving metadata and features of the concat profile into a json file
meta_features_dict = {
    "concat-profiles": {
        "meta-features": meta_cols,
        "shared-features": features_cols,
    }
}
with open(cpjump1_output_dir / "concat_profiles_meta_features.json", "w") as f:
    json.dump(meta_features_dict, f, indent=4)

# save as parquet
cjump_profiles.write_parquet(concat_output_path)


# ## Preprocessing MitoCheck Dataset
#
# This section processes the MitoCheck dataset by loading training data, positive controls, and negative controls from compressed CSV files. The data is standardized and converted to Parquet format for consistency with other datasets and improved performance.
#
# **Key preprocessing steps:**
#
# - **Loading datasets**: Reading training data, positive controls, and negative controls from compressed CSV files
# - **Control labeling**: Adding phenotypic class labels ("poscon" and "negcon") to distinguish control types
# - **Feature filtering**: Extracting only Cell Profiler (CP) features to match the CPJUMP1 dataset structure
# - **Column standardization**: Removing "CP__" prefixes and ensuring consistent naming conventions
# - **Feature alignment**: Identifying shared features across all three datasets (training, positive controls, negative controls)
# - **Metadata preservation**: Maintaining consistent metadata structure across all profile types
# - **Format conversion**: Saving processed data in optimized Parquet format for efficient downstream analysis
# - **adding cell id**: adding a cell id column `Metadata_cell_id`
#
# The preprocessing ensures that all MitoCheck datasets share a common feature space and are ready for comparative analysis with CPJUMP1 profiles.

# In[ ]:


# load in mitocheck profiles and save as parquet
# drop first column which is an additional index column
mitocheck_profile = pl.read_csv(
    mitocheck_norm_profiles_dir / "training_data.csv.gz",
)
mitocheck_profile = mitocheck_profile.select(mitocheck_profile.columns[1:])

# load in the mitocheck positive controls
mitocheck_pos_control_profiles = pl.read_csv(
    mitocheck_norm_profiles_dir / "positive_control_data.csv.gz",
)

# loading in negative control profiles
mitocheck_neg_control_profiles = pl.read_csv(
    mitocheck_norm_profiles_dir / "negative_control_data.csv.gz",
)

# insert new column "Mitocheck_Phenotypic_Class" for both positive and negative controls
mitocheck_neg_control_profiles = mitocheck_neg_control_profiles.with_columns(
    pl.lit("negcon").alias("Mitocheck_Phenotypic_Class")
).select(["Mitocheck_Phenotypic_Class"] + mitocheck_neg_control_profiles.columns)

mitocheck_pos_control_profiles = mitocheck_pos_control_profiles.with_columns(
    pl.lit("poscon").alias("Mitocheck_Phenotypic_Class")
).select(["Mitocheck_Phenotypic_Class"] + mitocheck_pos_control_profiles.columns)


# insert new column "Metadata_treatment_type" for mitocheck profiles
mitocheck_profile = mitocheck_profile.with_columns(
    pl.lit("trt").alias("Metadata_treatment_type")
).select(["Metadata_treatment_type"] + mitocheck_profile.columns)
mitocheck_neg_control_profiles = mitocheck_neg_control_profiles.with_columns(
    pl.lit("negcon").alias("Metadata_treatment_type")
).select(["Metadata_treatment_type"] + mitocheck_neg_control_profiles.columns)
mitocheck_pos_control_profiles = mitocheck_pos_control_profiles.with_columns(
    pl.lit("poscon").alias("Metadata_treatment_type")
).select(["Metadata_treatment_type"] + mitocheck_pos_control_profiles.columns)


# Filter Cell Profiler (CP) features and preprocess columns by removing the "CP__" prefix to standardize feature names for downstream analysis.

# In[ ]:


# Split profiles to only retain cell profiler features
cp_mitocheck_profile = split_data(mitocheck_profile, dataset="CP")
cp_mitocheck_neg_control_profiles = split_data(
    mitocheck_neg_control_profiles, dataset="CP"
)
cp_mitocheck_pos_control_profiles = split_data(
    mitocheck_pos_control_profiles, dataset="CP"
)
# Remove "CP__" prefix from all datasets for standardized feature names
cp_mitocheck_profile = remove_feature_prefixes(cp_mitocheck_profile)
cp_mitocheck_neg_control_profiles = remove_feature_prefixes(
    cp_mitocheck_neg_control_profiles
)
cp_mitocheck_pos_control_profiles = remove_feature_prefixes(
    cp_mitocheck_pos_control_profiles
)


# Splitting the metadata and feature columns for each dataset to enable targeted downstream analysis and ensure consistent data structure across all profiles.

# In[ ]:


# select # naming the metadata of mitocheck profiles
mitocheck_meta_data = [
    "Mitocheck_Phenotypic_Class",
    "Cell_UUID",
    "Location_Center_X",
    "Location_Center_Y",
    "Metadata_Plate",
    "Metadata_Well",
    "Metadata_Frame",
    "Metadata_Site",
    "Metadata_Plate_Map_Name",
    "Metadata_DNA",
    "Metadata_Gene",
    "Metadata_Gene_Replicate",
]

# select morphology features by dropping the metadata features and getting only the column names
cp_mitocheck_profile_features = cp_mitocheck_profile.drop(mitocheck_meta_data).columns
cp_mitocheck_neg_control_profiles_features = cp_mitocheck_neg_control_profiles.drop(
    mitocheck_meta_data
).columns
cp_mitocheck_pos_control_profiles_features = cp_mitocheck_pos_control_profiles.drop(
    mitocheck_meta_data
).columns

# now find shared profiles between all feature columns
shared_features = list(
    set(cp_mitocheck_profile_features)
    & set(cp_mitocheck_neg_control_profiles_features)
    & set(cp_mitocheck_pos_control_profiles_features)
)

# now create a json file that contains the feature space configs
with open(mitocheck_profiles_dir / "mitocheck_feature_space_configs.json", "w") as f:
    json.dump(
        {
            "metadata-features": mitocheck_meta_data,
            "morphology-features": shared_features,
        },
        f,
        indent=4,
    )


# In[ ]:


# create concatenated mitocheck profiles
concat_mitocheck_profiles = (
    # concat all mitocheck profiles with only shared features and metadata
    pl.concat(
        [
            cp_mitocheck_profile.select(mitocheck_meta_data + shared_features),
            cp_mitocheck_neg_control_profiles.select(
                mitocheck_meta_data + shared_features
            ),
            cp_mitocheck_pos_control_profiles.select(
                mitocheck_meta_data + shared_features
            ),
        ],
        rechunk=True,
    )
    # add index and unique cell ID
    .with_row_index("index")
)

# add unique cell ID based on features of a single profiles
concat_mitocheck_profiles = add_cell_id_hash(concat_mitocheck_profiles)

# save concatenated mitocheck profiles
concat_mitocheck_profiles.write_parquet(
    mitocheck_profiles_dir / "mitocheck_concat_profiles.parquet"
)


# ## Preprocessing CFReT Dataset
#
# This section preprocesses the CFReT (CRISPR Fluorescent Reporter of Transcription) dataset to ensure compatibility with downstream analysis workflows.
#
# - **Unique cell identification**: Adding `Metadata_cell_id` column with unique hash values based on all profile features to enable precise cell tracking and deduplication
#

# In[8]:


# load in cfret profiles and add a unique cell ID
cfret_profiles = pl.read_parquet(cfret_profiles_path)

# adding a unique cell ID based on all features
cfret_profiles = add_cell_id_hash(cfret_profiles, force=True)

# split features
meta_cols, features_cols = split_meta_and_features(cfret_profiles)

# save feature space config to json file
with open(cfret_profiles_dir / "cfret_feature_space_configs.json", "w") as f:
    json.dump(
        {
            "metadata-features": meta_cols,
            "morphology-features": features_cols,
        },
        f,
        indent=4,
    )

# overwrite dataset with cell
cfret_profiles.select(meta_cols + features_cols).write_parquet(cfret_profiles_path)
