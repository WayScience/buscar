#!/usr/bin/env python

# # Downloading and Processing CPJUMP1 Experimental Metadata
#
# This notebook focuses on downloading and processing the metadata associated with the [CPJUMP1 pilot dataset](https://github.com/jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1). The primary goal is to identify and organize plates that contain wells treated with CRISPR perturbations for downstream analysis.
#
# **Key Points:**
# - Only metadata is downloaded and processed in this notebook. The full CPJUMP1 dataset is not downloaded here.
# - The metadata provides essential information about which plates and wells are relevant for CRISPR-based experiments.
# - The processed dataset used in this notebook has already undergone quality control and feature selection. For access to the full processed dataset, refer to [repo](https://github.com/WayScience/JUMP-single-cell)
#
#

# In[1]:


import pathlib
import pprint
import sys

import polars as pl

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
profiles_dir = (data_dir / "sc-profiles").resolve()
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
exp_metadata = pl.read_csv(
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


# creating a dictionary for the batch and the associated plates with the a batch
batch_plates_dict = {}
exp_metadata_batches = exp_metadata["Batch"].unique().to_list()

for batch in exp_metadata_batches:
    # getting the plates in the batch
    plates_in_batch = exp_metadata.filter(exp_metadata["Batch"] == batch)["Assay_Plate_Barcode"].to_list()

    # adding the plates to the dictionary
    batch_plates_dict[batch] = plates_in_batch

# display batch (Keys) and plates (values) within each batch
pprint.pprint(batch_plates_dict)
