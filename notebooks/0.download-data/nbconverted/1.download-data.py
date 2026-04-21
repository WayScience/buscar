#!/usr/bin/env python

# # Downloading Single-Cell Profiles
#
# This notebook downloads metadata and single-cell profiles from three key datasets:
#
# 1. **CPJUMP1 Pilot Dataset** ([link](https://github.com/jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1)): Experimental metadata is downloaded and processed to identify and organize plates containing wells treated with either **compound** or **CRISPR** perturbations for downstream analysis.
# 2. **MitoCheck Dataset**: Normalized and feature-selected single-cell profiles are downloaded for further analysis.
# 3. **CFReT Dataset**: Normalized and feature-selected single-cell profiles from the CFReT plate are downloaded for downstream analysis.

# In[1]:


import pathlib
import sys

import polars as pl

sys.path.append("../../")
from utils.io_utils import download_compressed_file, load_configs

# ## Downloading data

# Parameters used in this notebook

# In[2]:


# setting perturbation type
# other options are "compound", "crispr",
pert_type = "compound"


# setting input and output paths

# In[3]:


# setting config path
config_path = pathlib.Path("dl-configs.yaml").resolve(strict=True)

# setting results setting a data directory
data_dir = pathlib.Path("./data").resolve()
data_dir.mkdir(exist_ok=True)

# setting a path to save the experimental metadata
exp_metadata_path = (data_dir / "CPJUMP1-experimental-metadata.csv").resolve()

# setting profile directory
profiles_dir = (data_dir / "sc-profiles").resolve()
profiles_dir.mkdir(exist_ok=True)

# create cpjump1 directory
cpjump1_dir = (profiles_dir / "cpjump1").resolve()
cpjump1_dir.mkdir(exist_ok=True)

# create mitocheck directory
mitocheck_dir = (profiles_dir / "mitocheck").resolve()
mitocheck_dir.mkdir(exist_ok=True)

# create cfret directory
cfret_dir = (profiles_dir / "cfret").resolve()
cfret_dir.mkdir(exist_ok=True)


# ## Downloading CPJUMP1 Metadata
#
# In this section, we download the [experimental metadata](https://github.com/carpenter-singh-lab/2024_Chandrasekaran_NatureMethods/blob/main/benchmark/output/experiment-metadata.tsv) for the CPJUMP1 dataset. This metadata contains detailed information about each experimental batch, including plate barcodes, cell lines, perturbation types, and incubation times. More information about the batch and plate metadata can be found in the [CPJUMP1 documentation](https://github.com/carpenter-singh-lab/2024_Chandrasekaran_NatureMethods/blob/main/README.md#batch-and-plate-metadata).
#
# We apply perturbation-specific filters to select plates from the `2020_11_04_CPJUMP1` batch:
#
# - **Compound-treated plates**: Plates where U2OS or A549 parental cell lines were treated with compound perturbations for 48 hours, with no antibiotics.
# - **CRISPR-treated plates**: Plates where U2OS or A549 cell lines were treated with CRISPR perturbations for 144 hours (long time point), with antibiotics absent.
#
# Note: Both datasets contain anomalies. In the compound plates, two U2OS plates show anomalies in the MitoTracker stain. In the CRISPR plates, all four U2OS plates exhibit anomalies in the WGA stain. Documentation states there shouldn't be an impact.

# In[4]:


# loading config file and setting experimental metadata URL
nb_configs = load_configs(config_path)
CPJUMP1_exp_metadata_url = nb_configs["links"]["CPJUMP1-experimental-metadata-source"]

# read in the experimental metadata CSV file and only filter down to plays that
# have an CRISPR perturbation
exp_metadata = pl.read_csv(
    CPJUMP1_exp_metadata_url, separator="\t", has_header=True, encoding="utf-8"
)

# apply a single filter to select only rows matching all criteria
if pert_type == "compound":
    exp_metadata = exp_metadata.filter(
        (pl.col("Perturbation").str.contains(pert_type))  # selecting based on pert type
        & (
            pl.col("Time") == 48
        )  # time of incubation with compound (select long time point)
        & (pl.col("Cell_type").is_in(["U2OS", "A549"]))  # selecting based on cell type
        & (
            pl.col("Antibiotics") == "absent"
        )  # selecting only the plates without antibiotics
        & (pl.col("Cell_line") == "Parental")  # selecting only the parental cell line
        & (
            pl.col("Batch") == "2020_11_04_CPJUMP1"
        )  # selecting only CellProfiler features
        & (pl.col("Density") == 100)  # selecting only the baseline cell density
    )
if pert_type == "crispr":
    exp_metadata = exp_metadata.filter(
        (pl.col("Perturbation").str.contains(pert_type))  # selecting based on pert type
        & (pl.col("Time") == 144)  # selecting the long time point
        & (pl.col("Cell_type").is_in(["U2OS", "A549"]))  # selecting based on cell type
        & (
            pl.col("Antibiotics") == "absent"
        )  # selecting only the plates without antibiotics
        & (
            pl.col("Batch") == "2020_11_04_CPJUMP1"
        )  # selecting only CellProfiler features
    )

# save the experimental metadata as a csv file
exp_metadata.write_csv(cpjump1_dir / f"cpjump1_{pert_type}_experimental-metadata.csv")

# display
print(
    "plates that will be downloaded are: ", exp_metadata["Assay_Plate_Barcode"].unique()
)
print("shape: ", exp_metadata.shape)
exp_metadata


#
# In this section, we download:
#
# 1. **Compound metadata** from the CPJUMP1 repository
# 2. **Mechanism of action (MOA) metadata** from the Broad Repurposing Hub
#
# We then merge both datasets into a single compound metadata table.
#
# If a compound has missing MOA information, the value in `Metadata_moa` is replaced with `"unknown"`. This indicates that no MOA annotation is currently available for that compound.

# In[5]:


if pert_type == "compound":
    # downloading compound metadata from cpjump1 repo
    CPJUMP_compound_metadata = pl.read_csv(
        nb_configs["links"]["CPJUMP1-compound-metadata-source"],
        separator="\t",
        has_header=True,
        encoding="utf-8",
    )

    # downloading compound moa metadata from broad institute drug repurposing hub
    broad_compound_moa_metadata = pl.read_csv(
        nb_configs["links"]["Broad-compounds-moa-source"],
        separator="\t",
        skip_rows=9,
        encoding="utf8-lossy",
    )

    # for both dataframes make sure that all columns have "Metadata_" in the column name
    CPJUMP_compound_metadata = CPJUMP_compound_metadata.rename(
        {col: f"Metadata_{col}" for col in CPJUMP_compound_metadata.columns}
    )
    broad_compound_moa_metadata = broad_compound_moa_metadata.rename(
        {col: f"Metadata_{col}" for col in broad_compound_moa_metadata.columns}
    )

    # replace null values in the broad compound moa to "unknown"
    broad_compound_moa_metadata = broad_compound_moa_metadata.with_columns(
        pl.col("Metadata_moa").fill_null("unknown")
    )
    complete_compound_metadata = CPJUMP_compound_metadata.join(
        broad_compound_moa_metadata,
        left_on="Metadata_pert_iname",
        right_on="Metadata_pert_iname",
        how="left",
    )

    # now merge moa metadata to the cpjump1 compound metadata
    complete_compound_metadata = CPJUMP_compound_metadata.join(
        broad_compound_moa_metadata,
        left_on="Metadata_pert_iname",
        right_on="Metadata_pert_iname",
        how="left",
    )

    # save the complete compound metadata as a tsv file
    complete_compound_metadata.write_csv(
        cpjump1_dir / "cpjump1_compound_compound-metadata.tsv", separator="\t"
    )
else:
    print(
        "Skipping this step:"
        f"This is a {pert_type} perturbation type, there's no moa info available."
    )


# ## Downloading MitoCheck Data
#
# In this section, we download the MitoCheck data generated in [this study](https://pmc.ncbi.nlm.nih.gov/articles/PMC3108885/).
#
# Specifically, we are downloading data that has already been normalized and feature-selected. The normalization and feature selection pipeline is available [here](https://github.com/WayScience/mitocheck_data/tree/main/3.normalize_data).

# In[6]:


# url source for the MitoCheck data
mitocheck_url = nb_configs["links"]["MitoCheck-profiles-source"]
save_path = (mitocheck_dir / "normalized_data").resolve()
if save_path.exists():
    print(f"File {save_path} already exists. Skipping download.")
else:
    download_compressed_file(mitocheck_url, save_path)


# ## Downloading CFReT Data
#
# This section downloads and saves feature-selected single-cell profiles from the CFReT plate `localhost230405150001`.
#
# - Only processed single-cell profiles are downloaded (no raw data).
# - Data is saved as a Parquet file for fast access.
# - Used in published cardiac fibrosis research ([study link](https://doi.org/10.1161/CIRCULATIONAHA.124.071956)).

# In[7]:


# setting the source for the CFReT data
cfret_source = nb_configs["links"]["CFReT-profiles-source"]

# use the correct filename from the source URL
output_path = (
    cfret_dir / "localhost230405150001_sc_feature_selected.parquet"
).resolve()

# check if it exists
if output_path.exists():
    print(f"File {output_path} already exists. Skipping download.")
else:
    # download cfret data from github and convert to parquet
    cfret_df = pl.read_parquet(cfret_source)
    cfret_df.write_parquet(output_path)

    # display
    print("shape: ", cfret_df.shape)
    cfret_df.head()
