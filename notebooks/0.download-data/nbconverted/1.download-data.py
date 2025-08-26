#!/usr/bin/env python
# coding: utf-8

# # Downloading Single-Cell Profiles
# 
# This notebook focuses on downloading metadata and single-cell profiles from three key datasets:
# 
# 1. **CPJUMP1 Pilot Dataset** ([link](https://github.com/jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1)): Metadata is downloaded and processed to identify and organize plates containing wells treated with CRISPR perturbations for downstream analysis.
# 2. **MitoCheck Dataset**: Normalized and feature-selected single-cell profiles are downloaded for further analysis.
# 3. **CFReT Dataset**: Normalized and feature-selected single-cell profiles from the CFReT plate are downloaded for downstream analysis.

# In[ ]:


import sys
import pprint
import pathlib

import requests
import polars as pl
from tqdm import tqdm

sys.path.append("../../")
from utils import io_utils


# ## Helper functions

# In[ ]:


import gzip
import zipfile
import tarfile

def download_compressed_file(
    source_url: str, output_path: pathlib.Path | str, chunk_size: int = 8192, extract: bool = True
):
    """ Downloads a compressed file from a URL with progress tracking.

    Downloads a file from the specified URL and saves it to the given output path.
    The download is performed in chunks to handle large files efficiently, and the progress is displayed using
    the `tqdm` library. The function raises exceptions for various error conditions, including
    invalid input types, file system errors, and issues during the download process.

    Parameters
    ----------
    source_url : str
        URL to download the file from.
    output_path : pathlib.Path
        Full path where the file should be saved.
    chunk_size : int, optional
        Size of chunks to download in bytes. Defaults to 8192.
    extract : bool, optional
        Whether to extract the compressed file after download. Defaults to True.

    Raises
    ------
    requests.exceptions.RequestException
        If there is an error during the download request.
    Exception
        For any unexpected error during file writing or progress tracking.
    """

    # type checking
    if not isinstance(source_url, str):
        raise TypeError(f"source_url must be a string, got {type(source_url)}")
    if not isinstance(output_path, (pathlib.Path, str)):
        raise TypeError(
            f"output_path must be a pathlib.Path or str, got {type(output_path)}"
        )
    if isinstance(output_path, str):
        output_path = pathlib.Path(output_path)
    if not output_path.parent.exists():
        raise FileNotFoundError(
            f"Output directory {output_path.parent} does not exist."
        )
    if output_path.exists() and not output_path.is_file():
        raise FileExistsError(f"Output path {output_path} exists and is not a file.")

    # starting downloading process
    try:
        # sending GET request to the source URL
        with requests.get(source_url, stream=True) as response:
            # raise an error if the request was unsuccessful
            response.raise_for_status()

            # get the total size of the file from the response headers
            total_size = int(response.headers.get("content-length", 0))

            # using tqdm to track the download progress
            with (
                open(output_path, "wb") as file,
                tqdm(
                    desc="Downloading",
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar,
            ):
                # iterating over the response content in chunks
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        file.write(chunk)
                        
                        # this updates the progress bar
                        pbar.update(len(chunk))

        # extract the file if requested
        if extract:

            # ensring that the path is a directory if the output path is a file
            # this is necessary for extraction
            extract_dir = output_path
            if extract_dir.is_file():
                extract_dir = output_path.parent
            
            if output_path.suffix == '.gz':
                # handle gzip files
                extracted_path = output_path.with_suffix('')
                with gzip.open(output_path, 'rb') as f_in:
                    with open(extracted_path, 'wb') as f_out:
                        f_out.write(f_in.read())
                print(f"Extracted to: {extracted_path}")
                
            elif output_path.suffix == '.zip':
                # handle zip files
                with zipfile.ZipFile(output_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                print(f"Extracted to: {extract_dir}")
                
            elif output_path.suffix in ['.tar', '.tgz'] or '.tar.' in output_path.name:
                # handle tar files
                with tarfile.open(output_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_dir)
                print(f"Extracted to: {extract_dir}")

    # handling exceptions
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"Error downloading file: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error: {e}")


# ## Downloading data

# Parameters used in this notebook

# In[ ]:


# setting perturbation type
pert_type = "crispr"


# setting input and output paths

# In[ ]:


# setting config path
config_path = pathlib.Path("../nb-configs.yaml").resolve(strict=True)

# setting results setting a data directory
data_dir = pathlib.Path("./data").resolve()
data_dir.mkdir(exist_ok=True)

# setting a path to save the experimental metadata
exp_metadata_path = (data_dir / "CPJUMP1-experimental-metadata.csv").resolve()

# setting profile directory
profiles_dir = (data_dir / "sc-profiles").resolve()
profiles_dir.mkdir(exist_ok=True)

# create mitocheck directory
mitocheck_dir = (profiles_dir / "mitocheck").resolve()
mitocheck_dir.mkdir(exist_ok=True)

# create cfret directory
cfret_dir = (profiles_dir / "cfret").resolve()
cfret_dir.mkdir(exist_ok=True)


# ## Downloading CPJUMP1 Metadata
# 
# In this section, we download and process the CPJUMP1 experimental metadata. This metadata contains information about assay plates, batches, and perturbation types, which is essential for organizing and analyzing single-cell profiles. Only plates treated with CRISPR perturbations are selected for downstream analysis.

# In[ ]:


# loading config file and setting experimental metadata URL
nb_configs = io_utils.load_configs(config_path)
CPJUMP1_exp_metadata_url = nb_configs["links"]["CPJUMP1-experimental-metadata-source"]

# read in the experimental metadata CSV file and only filter down to plays that
# have an CRISPR perturbation
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

# In[ ]:


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


# ## Downloading MitoCheck Data
# 
# In this section, we download the MitoCheck data generated in [this study](https://pmc.ncbi.nlm.nih.gov/articles/PMC3108885/).
# 
# Specifically, we are downloading data that has already been normalized and feature-selected. The normalization and feature selection pipeline is available [here](https://github.com/WayScience/mitocheck_data/tree/main/3.normalize_data).

# In[ ]:


# url source for the MitoCheck data
mitocheck_url = nb_configs["links"]["MitoCheck-profiles-source"]
output_path = mitocheck_dir / "mitocheck_profile.zip"

# checking if the downloaded file already exists
if output_path.exists():
    print(f"File {output_path} already exists. Skipping download.")
else:
    # downloading mitocheck profiles
    download_compressed_file(
        source_url=mitocheck_url,
        output_path=output_path,
        chunk_size=8192,
        extract=True,
    )


# ## Downloading CFReT Data
# 
# In this section, we download feature-selected single-cell profiles from the CFReT plate `localhost230405150001`. This plate contains three treatments: DMSO (control), drug_x, and TGFRi. The dataset consists of high-content imaging data that has already undergone feature selection, making it suitable for downstream analysis.
# 
# **Key Points:**
# - Only the processed single-cell profiles are downloaded [here](https://github.com/WayScience/cellpainting_predicts_cardiac_fibrosis/tree/main/3.process_cfret_features/data/single_cell_profiles)
# - The CFReT dataset was used and published in [this study](https://doi.org/10.1161/CIRCULATIONAHA.124.071956).

# In[ ]:


# setting the source for the CFReT data
cfret_source = nb_configs["links"]["CFReT-profiles-source"]

# use the correct filename from the source URL
output_path = (cfret_dir / "localhost230405150001_sc_feature_selected.parquet").resolve()

# checking if the download already exists if it does not exist 
# download the file
if output_path.exists():
    print(f"File {output_path} already exists. Skipping download.")
else:
    download_compressed_file(
        source_url=cfret_source, 
        output_path=output_path, 
    )

