#!/usr/bin/env python

# # Download CPJUMP1 Data
#
# This notebook documents the workflow for downloading and processing data from the JUMP Cell Painting dataset, available in the Cell Painting Gallery.
#
# We focus on datasets that have been perturbations with CRISPR knockdowns.
#
# Key steps in this workflow:
# - Load configuration and metadata from a YAML file.
# - Filter experimental metadata to include only plates with CRISPR perturbations.
# - Download each plate's data as a CSV file, convert it to Parquet format, and save it in the `./data/profiles` directory.
#
# Each individual plate data file is saved as a `parquet` file in the `./data/profiles` folder. If a file with the same name already exists, it will be replaced with the newly downloaded data.

# In[1]:


import pathlib
import sys
import time

import polars
import tqdm

sys.path.append("../../")
from utils import io_utils

# Parameters used in this notebook

# In[2]:


# setting perturbation type
pert_type = "crispr"


# setting input and output paths

# In[3]:


# setting config path
config_path = pathlib.Path("../nb-configs.yaml").resolve(strict=True)

# setting results setting a data directory
data_dir = pathlib.Path("./data").resolve()
data_dir.mkdir(exist_ok=True)

# setting profile directory
profiles_dir = (data_dir / "profiles").resolve(strict=True)
profiles_dir.mkdir(exist_ok=True)

# setting a path to save the experimental metadata
exp_metadata_path = (data_dir / "CPJUMP1-experimental-metadata.csv").resolve()


# Loading in the notebook configurations and downloading the experimental metadata

# In[4]:


# loading config file and setting experimental metadata URL
nb_configs = io_utils.load_configs(config_path)
CPJUMP1_exp_metadata_url = nb_configs["links"]["CPJUMP1-experimental-metadata-source"]

# read in the experimental metadata CSV file and only filter down to plays that
# have an ORF perturbation
exp_metadata = polars.read_csv(
    CPJUMP1_exp_metadata_url, separator="\t", has_header=True, encoding="utf-8"
)

# filtering the metadata to only includes plates that their perturbation types are crispr
exp_metadata = exp_metadata.filter(exp_metadata["Perturbation"].str.contains(pert_type))

# save the experimental metadata as a csv file
exp_metadata.write_csv(exp_metadata_path)

# display
exp_metadata


# Creating a dictionary to group plates by their corresponding experimental batch
#
# This step organizes the plate barcodes from the experimental metadata into groups based on their batch. Grouping plates by batch is useful for batch-wise data processing and downstream analyses.

# In[5]:


# creating a dictionary for the batch and the associated plates iwthi nthe a batch
batch_plates_dict = {}
exp_metadata_batches = exp_metadata["Batch"].unique().to_list()

for batch in exp_metadata_batches:
    # getting the plates in the batch
    plates_in_batch = exp_metadata.filter(exp_metadata["Batch"] == batch)["Assay_Plate_Barcode"].to_list()

    # adding the plates to the dictionary
    batch_plates_dict[batch] = plates_in_batch

batch_plates_dict


# The profiles for each plate are downloaded and stored in the `./data/profiles` directory. Each file is prefixed with its corresponding batch name, making it easy to identify the experimental batch for each plate.

# In[6]:


# setting CPJUMP1 source link, this points to the main directory where all the plate data
# is stored
header_link = nb_configs["links"]["CPJUMP1-source"]

# create a for loop with progress bar for downloading plate data
for batch, plates in batch_plates_dict.items():

    loaded_profiles_in_batch = []
    for plate in tqdm.tqdm(plates, desc="Downloading plates"):

        # constructing the plate data source URL
        plate_data_source = f"{header_link}/{plate}/{plate}_normalized_negcon.csv.gz"

        # reading the plate data from the source URL
        # if the plate cannot be downloaded and read, it will skip to the next plate
        try:
            # load and save the plate data as a parquet file
            orf_plate_df = polars.read_csv(plate_data_source, separator=",", has_header=True)
            orf_plate_df.write_parquet(profiles_dir / f"{batch}_{plate}_normalized_negcon.parquet")
        except Exception as e:
            raise FileNotFoundError(f"Failed to download and read plate data for {plate}. Error: {e}")

        # store the loaded plate data in a list
        loaded_profiles_in_batch.append(orf_plate_df)

        # sleep to avoid overwhelming the AWS hosting the data
        time.sleep(0.7)
