#!/usr/bin/env python

# # 3. Subsetting CPJUMP1 controls
#
# In this notebook, we subset control samples from the CPJUMP1 CRISPR dataset using stratified sampling. We generate 10 different random seeds to create multiple subsets, each containing 15% of the original control data stratified by plate and well metadata. This approach ensures reproducible sampling while maintaining the distribution of controls across experimental conditions.
#
# The subsampled datasets are saved as individual parquet files for downstream analysis and model training purposes.
#

# In[1]:


import json
import pathlib
import sys

import polars as pl

sys.path.append("../../")
from utils.data_utils import split_meta_and_features

# Load helper functions

# In[2]:


def load_group_stratified_data(
    profiles: str | pathlib.Path | pl.DataFrame,
    group_columns: list[str] = ["Metadata_Plate", "Metadata_Well"],
    sample_percentage: float = 0.2,
    seed: int = 0,
) -> pl.DataFrame:
    """Memory-efficiently sample a percentage of rows from each group in a dataset.

    This function performs stratified sampling by loading only the grouping columns first
    to determine group memberships and sizes, then samples indices from each group, and
    finally loads the full dataset filtered to only the sampled rows. This approach
    minimizes memory usage compared to loading the entire dataset upfront.

    Parameters
    ----------
    dataset_path : str or pathlib.Path
        Path to the parquet dataset file to sample from
    group_columns : list[str], default ["Metadata_Plate", "Metadata_Well"]
        Column names to use for grouping. Sampling will be performed independently
        within each unique combination of these columns
    sample_percentage : float, default 0.2
        Fraction of rows to sample from each group (must be between 0.0 and 1.0)

    Returns
    -------
    pl.DataFrame
        Subsampled dataframe containing the sampled rows from each group,
        preserving all original columns

    Raises
    ------
    ValueError
        If sample_percentage is not between 0 and 1
    FileNotFoundError
        If dataset_path does not exist
    """
    # validate inputs
    if not 0 <= sample_percentage <= 1:
        raise ValueError("sample_percentage must be between 0 and 1")

    # convert str types to pathlib types
    if isinstance(profiles, str):
        profiles = pathlib.Path(profiles).resolve(strict=True)

    # load only the grouping columns to determine groups
    if isinstance(profiles, pl.DataFrame):
        # if a polars DataFrame is provided, use it directly
        metadata_df = profiles.select(group_columns).with_row_index("original_idx")
    else:
        metadata_df = pl.read_parquet(profiles, columns=group_columns).with_row_index(
            "original_idx"
        )

    # sample indices for each group based on the group_columns
    sampled_indices = (
        metadata_df
        # group rows by the specified columns (e.g., Plate and Well combinations)
        .group_by(group_columns)
        # for each group, randomly sample a fraction of the original row indices
        .agg(
            pl.col("original_idx")
            .sample(
                fraction=sample_percentage, seed=seed
            )  # sample specified percentage from each group
            .alias("sampled_idx")  # rename the sampled indices column
        )
        # extract only the sampled indices column, discarding group identifiers
        .select("sampled_idx")
        # convert list of indices per group into individual rows (flatten the structure)
        .explode("sampled_idx")
        # extract the sampled indices as a single column series
        .get_column("sampled_idx")
        .sort()
    )

    # load the entire dataset and filter to sampled indices
    sampled_df = (
        profiles.with_row_index("idx")
        .filter(pl.col("idx").is_in(sampled_indices.implode()))
        .drop("idx")
    )

    return sampled_df


# Setting input and output paths

# In[3]:


# setting data path
data_dir = pathlib.Path("../0.download-data/data").resolve(strict=True)
download_module_results_dir = pathlib.Path("../0.download-data/results").resolve(
    strict=True
)

# setting directory where all the single-cell profiles are stored
profiles_dir = (data_dir / "sc-profiles").resolve(strict=True)

exp_metadata_path = (
    profiles_dir / "cpjump1" / "CPJUMP1-experimental-metadata.csv"
).resolve(strict=True)

# Setting feature selection path
shared_features_config_path = (
    profiles_dir / "cpjump1" / "feature_selected_sc_qc_features.json"
).resolve(strict=True)

# setting cpjump1 data dir
cpjump_crispr_data_dir = (data_dir / "sc-profiles" / "cpjump1-crispr-negcon").resolve()
cpjump_crispr_data_dir.mkdir(exist_ok=True)


# setting negative control
negcon_data_dir = (profiles_dir / "cpjump1" / "negcon").resolve()
negcon_data_dir.mkdir(exist_ok=True)
poscon_data_dir = (profiles_dir / "cpjump1" / "poscon").resolve()
poscon_data_dir.mkdir(exist_ok=True)


# Loading data

# In[4]:


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


# In[5]:


control_df = []
for plate_path in crispr_plate_paths:
    # load plate data and filter to controls
    plate_controls_df = pl.read_parquet(plate_path).filter(
        pl.col("Metadata_pert_type") == "control"
    )

    # split features
    controls_meta, _ = split_meta_and_features(plate_controls_df)

    # select metadata and shared features together
    controls_df = plate_controls_df.select(controls_meta + shared_features)

    # then append to list
    control_df.append(controls_df)

# concatenate dataframes
controls_df = pl.concat(control_df)


# In[6]:


negcon_df = controls_df.filter(pl.col("Metadata_control_type") == "negcon")
negcon_df


# generating 10 seeds of randomly sampled negative controls

# In[7]:


for seed_val in range(10):
    # load the dataset with group stratified sub sampling
    subsampled_df = load_group_stratified_data(
        profiles=negcon_df,
        group_columns=["Metadata_Plate", "Metadata_Well"],
        sample_percentage=0.15,
        seed=seed_val,
    )

    # save the file
    subsampled_df.write_parquet(
        negcon_data_dir / f"cpjump1_crispr_negcon_seed{seed_val}.parquet"
    )


# Selecting only positive controls and saving it

# In[8]:


# write as parquet file
poscon_cp_df = controls_df.filter(pl.col("Metadata_control_type") == "poscon_cp")
poscon_cp_df.write_parquet(poscon_data_dir / "poscon_cp_df.parquet")
