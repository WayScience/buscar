#!/usr/bin/env python

# ## 3 Generating centroid profiles
# In this notebook, we identify the centroid for each cluster found in the single-cell profiles after running the buscar clustering module.
#
# The centroid is the representative cell that best captures the distribution of cells within a cluster.

# In[1]:


import pathlib
import sys

import polars as pl
from pycytominer import aggregate

sys.path.append("../../")
from utils.data_utils import split_meta_and_features

# Setting input and output paths

# In[2]:


# setting data path for cfret-pilot dataset
cfret_profiles_path = pathlib.Path(
    "../0.download-data/data/sc-profiles/cfret/localhost230405150001_sc_feature_selected.parquet"
).resolve(strict=True)

# setting cluster labels path
cluster_labels_path = pathlib.Path(
    "./results/clusters/cfret_pilot_cluster_labels.parquet"
).resolve(strict=True)

# setting outpaths for results
results_dir = pathlib.Path("./results").resolve()
results_dir.mkdir(exist_ok=True)

# setting outpath for centroids
centroids_dir = (results_dir / "centroids").resolve()
centroids_dir.mkdir(exist_ok=True)


# Loading profiles

# In[3]:


# loading in profiles and add cluster labels to profiles dataframe
cfret_df = pl.read_parquet(cfret_profiles_path).join(
    pl.read_parquet(cluster_labels_path), on="Metadata_cell_id", how="inner"
)

# add a column that indicates the heart and treatment added
cfret_df = cfret_df.with_columns(
    pl.concat_str(
        [
            pl.col("Metadata_cell_type"),
            pl.col("Metadata_treatment"),
        ],
        separator="_",
    ).alias("Metadata_heart_treatment")
)

# split feature space
cfret_meta, cfret_feats = split_meta_and_features(cfret_df)

# display
print(cfret_df.shape)


cfret_df.select(cfret_meta + cfret_feats).head()


# We use **median aggregation** to generate centroid profiles for each cluster. For each cluster, we calculate the component-wise median across all cells to create a synthetic representative profile that captures the central tendency. This approach is robust to outliers, consistent with replicate and consensus profile generation workflow, and works well for high-dimensional morphological features.

# In[4]:


# split metadata and features
cfret_meta, cfret_feats = split_meta_and_features(cfret_df)

print(f"Total cells: {len(cfret_df)}")
print(f"Number of features: {len(cfret_feats)}")
print(f"Unique clusters: {cfret_df['Metadata_cluster_id'].n_unique()}")


# Save centroid profiles

# In[5]:


# aggregate by cluster using median to generate centroid profiles
centroids_df = aggregate(
    population_df=cfret_df.to_pandas(),
    strata=["Metadata_cluster_id"],
    features=cfret_feats,
    operation="median",
)

# convert back to polars
centroids_df = pl.from_pandas(centroids_df)

print(f"Total centroids generated: {len(centroids_df)}")
print(f"Centroid shape: {centroids_df.shape}")
centroids_df


# In[6]:


# create a mapping of cluster_id to heart_treatment (unique per cluster)
cluster_treatment_mapping = cfret_df.select(
    ["Metadata_cluster_id", "Metadata_heart_treatment"]
).unique()

# join centroids with the treatment metadata
centroids_df = centroids_df.join(
    cluster_treatment_mapping, on="Metadata_cluster_id", how="left"
)

# save centroids to parquet file
centroids_output_path = centroids_dir / "cfret_pilot_centroids.parquet"
centroids_df.write_parquet(centroids_output_path)

print(f"Centroids saved to: {centroids_output_path}")
print(f"Final centroid shape: {centroids_df.shape}")
