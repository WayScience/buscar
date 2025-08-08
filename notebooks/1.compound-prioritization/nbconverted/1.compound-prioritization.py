#!/usr/bin/env python

# # Compound Prioritization Analysis
#
# In this notebook, we conduct a compound prioritization analysis to identify potential candidate compounds from high-content drug screens. This process enables us to select compounds with the most promising biological effects based on their morphological impact.

# In[1]:


import pathlib
import sys

import pandas as pd

# using analysis module imports
sys.path.append("../../")
import src.utils as utils

# In[2]:


# creating results directory
result_dir = utils.create_results_dir()

# loading in cell-injury data
cell_injury_data_path = pathlib.Path("../data/labeled_cell_injury_df.parquet").resolve(
    strict=True
)


# In[10]:


# loading data files
cell_injury_df = pd.read_parquet(cell_injury_data_path)

# split metadata and morphology feature column names
meta_colnames, feat_colnames = utils.split_meta_and_features(profile=cell_injury_df)


# In[17]:


# only selecting wells that have been treated with DMSO and CytoSkeletal
control_df = cell_injury_df.loc[cell_injury_df["injury_type"] == "Control"][
    feat_colnames
]
cyto_injury_df = cell_injury_df.loc[cell_injury_df["injury_type"] == "Cytoskeletal"][
    feat_colnames
]

# display sizes
print(f"Total number of control wells: {control_df.shape[0]}")
print(f"Total number of wells cytoskeletal injury: {cyto_injury_df.shape[0]}")


# In[49]:


target = control_df
reference = cyto_injury_df


# In[ ]:
