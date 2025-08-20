#!/usr/bin/env python

# # 3. Subsetting CPJUMP1 controls

# In[6]:


import pathlib

import pandas as pd

# In[3]:


# setting data path
data_path = pathlib.Path("./data/CPJUMP1-experimental-metadata.csv")


# In[7]:


pd.read_csv(data_path).head(100)
