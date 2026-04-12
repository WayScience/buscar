#!/usr/bin/env python

# # 4.Generate PCA and UMAP embeddings for CFReT single-cell data
#
# In this notebook we are create PCA and UMAP embeddings for the single-cell CFReT dataset.

# In[1]:


import pathlib
import sys

import polars as pl

sys.path.append("../../")
from buscar.preprocess import apply_pca, apply_umap

from utils.data_utils import split_meta_and_features
from utils.io_utils import load_configs, load_profiles

# Setting input and output paths

# In[2]:


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


# In[3]:


# loading profiles and signatures
signatures = load_configs(signatures_path)
cfret_profiles_df = load_profiles(cfret_pilot_profiles_path)

# filter to only DMSO and TGFRi treated (since these was the ref and targeted conditions)
cfret_profiles_df = cfret_profiles_df.filter(
    (pl.col("Metadata_treatment") == "DMSO") | (pl.col("Metadata_treatment") == "TGFRi")
)

cfret_meta, cfret_feats = split_meta_and_features(cfret_profiles_df)


# In[4]:


# count number of cell per cell and treatment condition
cell_count_df = cfret_profiles_df.group_by(
    "Metadata_cell_type", "Metadata_treatment"
).len()

print("Number of cells per cell type and treatment condition:")
cell_count_df


# In[5]:


# separate on and off morpholgoical profiles
on_sigsnature_feats = signatures["on"]
off_signature_feats = signatures["off"]
on_profiles_df = cfret_profiles_df.select(cfret_meta + on_sigsnature_feats)
off_profiles_df = cfret_profiles_df.select(cfret_meta + off_signature_feats)


# In this section, we compute PCA embeddings separately for two distinct morphological signature sets:
# - **On-morphology signatures**: Features associated with TGFRi-induced morphological changes
# - **Off-morphology signatures**: Features that are not associated with the morpholgical changes
#
# Each PCA model is fit independently using only the respective feature set, resulting in 2 principal components for visualization in 2D space.

# In[6]:


# apply pca on single-cell profiles with on morphological signatures
pca_on_result, explained_variance_df = apply_pca(
    profiles=on_profiles_df,
    meta_features=cfret_meta,
    morph_features=on_sigsnature_feats,
    var_explained=2,  # getting two components to visualize in 2D space
    random_state=0,
)

# apply pca on single-cell profiles with off morphological signatures
pca_off_results, explained_variance_off_df = apply_pca(
    profiles=off_profiles_df,
    meta_features=cfret_meta,
    morph_features=off_signature_feats,
    var_explained=2,  # getting two components to visualize in 2D space
    random_state=0,
)

# save pca compounents
pca_on_result.write_parquet(pca_dir / "cfret_pilot_on_morph_pca.parquet")
explained_variance_df.write_parquet(
    pca_dir / "cfret_pilot_on_morph_pca_var_explained.parquet"
)
pca_off_results.write_parquet(pca_dir / "cfret_pilot_off_morph_pca.parquet")
explained_variance_off_df.write_parquet(
    pca_dir / "cfret_pilot_off_morph_pca_var_explained.parquet"
)

# print shapes
print("PCA on morph shape:", pca_on_result.shape)
print("PCA off morph shape:", pca_off_results.shape)


# Apply UMAP with 2 components independently on three feature sets:
# 1. **All morphological features** - using the complete feature set from `cfret_feats`
# 2. **On-morphology signatures only** - using features in the "on" signature
# 3. **Off-morphology signatures only** - using features in the "off" signature
#
# Each UMAP embedding is saved separately for downstream analysis and visualization.

# In[7]:


# apply a global umap on single-cell profiles with on and off morphological signatures
umap_all_result = apply_umap(
    profiles=cfret_profiles_df,
    meta_features=cfret_meta,
    morph_features=cfret_feats,
    n_components=2,
    random_state=0,
    metric="cosine",
)
# apply umap on single-cell profiles with off morphological signatures
umap_on_result = apply_umap(
    profiles=on_profiles_df,
    meta_features=cfret_meta,
    morph_features=on_sigsnature_feats,
    n_components=2,
    random_state=0,
    metric="cosine",
)

# apply umap on single-cell profiles with off morphological signatures
umap_off_result = apply_umap(
    profiles=off_profiles_df,
    meta_features=cfret_meta,
    morph_features=off_signature_feats,
    n_components=2,
    random_state=0,
    metric="cosine",
)

# save umap compounents
umap_all_result.write_parquet(umap_dir / "cfret_pilot_all_morph_umap.parquet")
umap_on_result.write_parquet(umap_dir / "cfret_pilot_on_morph_umap.parquet")
umap_off_result.write_parquet(umap_dir / "cfret_pilot_off_morph_umap.parquet")

# print shapes
print("UMAP on all morph shape:", umap_all_result.shape)
print("UMAP on morph shape:", umap_on_result.shape)
print("UMAP off morph shape:", umap_off_result.shape)
