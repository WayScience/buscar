#!/usr/bin/env python

# In[15]:


import pathlib
import sys

import polars as pl

sys.path.append("../../")
from utils.data_utils import split_meta_and_features
from utils.io_utils import load_configs, load_profiles
from utils.preprocess import apply_pca, apply_umap

# Setting input and output paths

# In[16]:


# setting results path
results_dir = pathlib.Path("./results").resolve(strict=True)

# setting on- and off mophlgoical signature path
signatures_path = (results_dir / "signatures/cfret_pilot_signatures.json").resolve()

# setting cfret-pilot data path
data_dir = pathlib.Path("../0.download-data/data/sc-profiles").resolve()

# setting cfret-screen data path
cfret_pilot_profiles_path = (
    data_dir / "cfret/localhost230405150001_sc_feature_selected.parquet"
).resolve(strict=True)

# setting results path
results_dir = pathlib.Path("./results").resolve()
results_dir.mkdir(exist_ok=True)

# setting pca subdir
pca_dir = (results_dir / "pca").resolve()
pca_dir.mkdir(exist_ok=True)
umap_dir = (results_dir / "umap").resolve()
umap_dir.mkdir(exist_ok=True)

# setting plots folder and pca
plots_dir = (results_dir / "plots").resolve()
plots_dir.mkdir(exist_ok=True)


# In[17]:


# loading profiles and signatures
signatures = load_configs(signatures_path)
cfret_profiles_df = load_profiles(cfret_pilot_profiles_path)

# filter to only DMSO and TGFRi treated (since these was the ref and targeted conditions)
cfret_profiles_df = cfret_profiles_df.filter(
    (pl.col("Metadata_treatment") == "DMSO") | (pl.col("Metadata_treatment") == "TGFRi")
)

cfret_meta, cfret_feats = split_meta_and_features(cfret_profiles_df)


# In[20]:


# count cells within Metadata_treatment
cell_count_df = cfret_profiles_df.group_by("Metadata_treatment").len()
cell_count_df


# In[24]:


# separate on and off morpholgoical profiles
on_sigsnature_feats = signatures["on"]
off_signature_feats = signatures["off"]
on_profiles_df = cfret_profiles_df.select(cfret_meta + on_sigsnature_feats)
off_profiles_df = cfret_profiles_df.select(cfret_meta + off_signature_feats)


# Apply PCA to both on and off profiles

# In[25]:


# apply pca on single-cell profiles with on morphological signatures
pca_on_result = apply_pca(
    profiles=on_profiles_df,
    meta_features=cfret_meta,
    morph_features=on_sigsnature_feats,
    var_explained=0.95,
    random_state=0,
)

# apply pca on single-cell profiles with off morphological signatures
pca_off_results = apply_pca(
    profiles=off_profiles_df,
    meta_features=cfret_meta,
    morph_features=off_signature_feats,
    var_explained=0.95,
    random_state=0,
)

# save pca compounents
pca_on_result.write_parquet(pca_dir / "cfret_pilot_on_morph_pca.parquet")
pca_off_results.write_parquet(pca_dir / "cfret_pilot_off_morph_pca.parquet")

# print shapes
print("PCA on morph shape:", pca_on_result.shape)
print("PCA off morph shape:", pca_off_results.shape)


# Apply UMAP to both on and off profiles

# In[26]:


# apply umap on single-cell profiles with off morphological signatures
umap_on_result = apply_umap(
    profiles=on_profiles_df,
    meta_features=cfret_meta,
    morph_features=on_sigsnature_feats,
    n_components=2,
    random_state=0,
)

# apply umap on single-cell profiles with off morphological signatures
umap_off_result = apply_umap(
    profiles=off_profiles_df,
    meta_features=cfret_meta,
    morph_features=off_signature_feats,
    n_components=2,
    random_state=0,
)

# save umap compounents
umap_on_result.write_parquet(umap_dir / "cfret_pilot_on_morph_umap.parquet")
umap_off_result.write_parquet(umap_dir / "cfret_pilot_off_morph_umap.parquet")

# print shapes
print("UMAP on morph shape:", umap_on_result.shape)
print("UMAP off morph shape:", umap_off_result.shape)


# In[ ]:
