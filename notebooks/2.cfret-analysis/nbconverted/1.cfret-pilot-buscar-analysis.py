#!/usr/bin/env python

# # CFReT Buscar analysis
#
#
# This notebook demonstrates applying the BUSCAR pipeline to the CFReT pilot Cell Painting dataset.
# It walks through data loading, signature extraction, clustering, measuring phenotypic activity
# (between a reference control and experimental treatments) and ranking treatments by effect.
#
# Data & references
# - Data source: CFReT pilot dataset (see paper: https://www.ahajournals.org/doi/full/10.1161/CIRCULATIONAHA.124.071956)
# - Original data repo: https://github.com/WayScience/cellpainting_predicts_cardiac_fibrosis

# In[1]:


import json
import pathlib
import sys

import polars as pl

sys.path.append("../../")
# from utils.metrics import measure_phenotypic_activity
from utils.data_utils import split_meta_and_features
from utils.heterogeneity import optimized_clustering
from utils.identify_hits import identify_compound_hit
from utils.io_utils import load_profiles
from utils.metrics import measure_phenotypic_activity
from utils.preprocess import apply_pca
from utils.signatures import get_signatures

# Setting paramters

# In[2]:


# setting parameters
treatment_col = "Metadata_treatment"
treatment_heart_col = "Metadata_treatment_and_heart"

# parameters used for clustering optimization
cfret_pilot_cluster_param_grid = {
    # Clustering resolution: how granular the clusters should be
    "cluster_resolution": {"type": "float", "low": 0.1, "high": 2.2},
    # Number of neighbors for graph construction
    "n_neighbors": {"type": "int", "low": 5, "high": 100},
    # Clustering algorithm
    "cluster_method": {"type": "categorical", "choices": ["leiden"]},
    # Distance metric for neighbor computation
    "neighbor_distance_metric": {
        "type": "categorical",
        "choices": ["euclidean", "cosine", "manhattan"],
    },
}


# Setting input and output paths

# In[3]:


# load in raw data from
cfret_data_dir = pathlib.Path("../0.download-data/data/sc-profiles/cfret/").resolve(
    strict=True
)
cfret_profiles_path = (
    cfret_data_dir / "localhost230405150001_sc_feature_selected.parquet"
).resolve(strict=True)
cfret_feature_space_path = (
    cfret_data_dir / "cfret_feature_space_configs.json"
).resolve(strict=True)

# make results dir
results_dir = pathlib.Path("./results/cfret-pilot").resolve()
results_dir.mkdir(parents=True, exist_ok=True)


# Data preprocessing
# -

# In[4]:


# loading profiles
cfret_df = load_profiles(cfret_profiles_path)

# add another metadata column that combins both Metadata_heart_number and Metadata_treatment
cfret_df = cfret_df.with_columns(
    (
        pl.col("Metadata_treatment").cast(pl.Utf8)
        + "_heart_"
        + pl.col("Metadata_heart_number").cast(pl.Utf8)
    ).alias("Metadata_treatment_and_heart")
)

# Update the Metadata_treatment column that distinguishes what is the reference
# among other treatments in this example we are using heart 11 + DMSO as our
# reference (healthy heart + DMSO)
cfret_df = cfret_df.with_columns(
    pl.when(
        (pl.col("Metadata_treatment") == "DMSO")
        & (pl.col("Metadata_heart_number") == 11)
    )
    .then(pl.lit("DMSO_heart_11"))
    .otherwise(pl.col("Metadata_treatment"))
    .alias("Metadata_treatment")
)
cfret_df = cfret_df.with_columns(
    pl.when(
        (pl.col("Metadata_treatment") == "DMSO")
        & (pl.col("Metadata_heart_number") == 9)
    )
    .then(pl.lit("DMSO_heart_9"))
    .otherwise(pl.col("Metadata_treatment"))
    .alias("Metadata_treatment")
)

# split features
cfret_meta, cfret_feats = split_meta_and_features(cfret_df)


# Display data
cfret_df.head()


# In[5]:


# show how many cells per treatment
# shows the number of cells per treatment that will be clustered.
cells_per_treatment_counts = (
    cfret_df.group_by(treatment_heart_col).count().sort(treatment_heart_col)
)
cells_per_treatment_counts


# ## BUSCAR pipeline

# Creating on and off morphology signatures

# In[6]:


# setting output paths
signatures_outpath = (results_dir / "cfret_pilot_signatures.json").resolve()

if signatures_outpath.exists():
    print("Signatures already exist, skipping this step.")
    with open(signatures_outpath) as f:
        sigs = json.load(f)
        on_sigs = sigs["on"]
        off_sigs = sigs["off"]
else:
    # once the data is loaded, separate the controls
    negcon_df = cfret_df.filter(pl.col("Metadata_treatment") == "DMSO_heart_9")
    poscon_df = cfret_df.filter(pl.col("Metadata_treatment") == "DMSO_heart_11")

    # creating signatures
    on_sigs, off_sigs, _ = get_signatures(
        ref_profiles=negcon_df,
        exp_profiles=poscon_df,
        morph_feats=cfret_feats,
        test_method="mann_whitney_u",
    )

    # Save signatures as json file
    with open(signatures_outpath, "w") as f:
        json.dump({"on": on_sigs, "off": off_sigs}, f, indent=4)


# Search for heterogenous effects for each treatment

# In[7]:


# Apply PCA to cfret_data
pca_cfret_df = apply_pca(
    profiles=cfret_df,
    meta_features=cfret_meta,
    morph_features=cfret_feats,
    var_explained=0.95,
    seed=0,
)

# update cfret_feats because PCA was applied
cfret_pca_feats = pca_cfret_df.drop(cfret_meta).columns


# In[ ]:


# number of cores required for optuna jobs
# each optuna jobs is distrubred per treament
cfret_n_jobs = cfret_df["Metadata_treatment_and_heart"].n_unique()
print(f"Number of unique treatments (hearts + treatment): {cfret_n_jobs}")

# check if the cluster labels already exist; if so just load the labels and skip optimization
# if not run optimization
cluster_labels_output = (results_dir / "cfret_pilot_cluster_labels.parquet").resolve()
if cluster_labels_output.exists():
    print("Cluster labels already exist, skipping clustering optimization.")
    cfret_cluster_labels_df = pl.read_parquet(cluster_labels_output)
    cfret_best_params = (
        None  # You may want to load best params from a saved file if needed
    )
else:
    # optimizing clustering
    cfret_cluster_labels_df, cfret_best_params = optimized_clustering(
        profiles=pca_cfret_df,
        meta_features=cfret_meta,
        morph_features=cfret_pca_feats,
        treatment_col=treatment_heart_col,
        param_grid=cfret_pilot_cluster_param_grid,
        n_trials=500,
        n_jobs=cfret_n_jobs,
        seed=0,
        study_name="cfret_pilot_pca",
    )

    # write out cluster labels
    cfret_cluster_labels_df.write_parquet(cluster_labels_output)

    # write best params as a json file
    with open(results_dir / "cfret_pilot_best_clustering_params.json", "w") as f:
        json.dump(cfret_best_params, f, indent=4)


# In[9]:


# merge cfret_df with the cluster labels and make sure to drop duplicate Metadata_cell_id columns
labeled_cfret_df = cfret_df.join(
    cfret_cluster_labels_df,
    on=["Metadata_cell_id"],
    how="left",
)

# check if the no rows added
assert cfret_df.height == labeled_cfret_df.height, (
    "Merged DataFrame has different number of rows!"
)


# In[10]:


# setting output paths
treatment_dist_scores_outpath = (
    results_dir / "treatment_phenotypic_scores.csv"
).resolve()
if treatment_dist_scores_outpath.exists():
    print("Treatment phenotypic distance scores already exist, skipping this step.")
    treatment_dist_scores = pl.read_csv(treatment_dist_scores_outpath)

else:
    treatment_heart_dist_scores = measure_phenotypic_activity(
        profiles=labeled_cfret_df,
        on_signature=on_sigs,
        off_signature=off_sigs,
        ref_treatment="DMSO_heart_11",
        treatment_col=treatment_heart_col,
    )

    # save those as csv files
    treatment_heart_dist_scores.write_csv(treatment_dist_scores_outpath)


# In[11]:


treatment_heart_dist_scores


# In[12]:


# setting outptut paths
treatment_heart_rankings_outpath = (
    results_dir / "treatment_heart_rankings.csv"
).resolve()
if treatment_heart_rankings_outpath.exists():
    print("Treatment heart rankings already exist, skipping this step.")
    treatment_heart_rankings = pl.read_csv(treatment_heart_rankings_outpath)
else:
    treatment_heart_rankings = identify_compound_hit(
        distance_df=treatment_heart_dist_scores, method="weighted_sum"
    )

    # save as csv files
    treatment_heart_rankings.write_csv(treatment_heart_rankings_outpath)


# In[13]:


treatment_heart_rankings
