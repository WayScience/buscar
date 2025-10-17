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
from utils.signatures import get_signatures

# Setting paramters

# In[2]:


# setting parameters
treatment_col = "Metadata_treatment"
treatment_heart_col = "Metadata_treatment_and_heart"


# parameters used for clustering optimization
cfret_cluster_param_grid = {
    # Clustering resolution: how granular the clusters should be
    "cluster_resolution": {"type": "float", "low": 0.1, "high": 2.2},
    # Number of neighbors for graph construction
    "n_neighbors": {"type": "int", "low": 5, "high": 100},
    # Clustering algorithm
    "cluster_method": {"type": "categorical", "choices": ["leiden", "louvain"]},
    # Distance metric for neighbor computation
    "neighbor_distance_metric": {
        "type": "categorical",
        "choices": ["euclidean", "cosine", "manhattan"],
    },
    # Dimensionality reduction approach
    "dim_reduction": {"type": "categorical", "choices": ["PCA", "raw"]},
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


# ## BUSCAR pipeline

# Creating on and off morphology signatures

# In[5]:


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


# assess heterogeneity

# In[6]:


# setting best params outputs
treatment_best_params_outpath = (
    results_dir / "cfret_treatment_clustering_params.json"
).resolve()
cfret_treatment_cluster_df_outpath = (
    results_dir / "cfret_treatment_clustered.parquet"
).resolve()
cfret_treatment_heart_cluster_df_outpath = (
    results_dir / "cfret_treatment_heart_clustered.parquet"
).resolve()
treatment_heart_best_params_outpath = (
    results_dir / "cfret_treatment_heart_clustering_params.json"
).resolve()

# check if the files exist, if they do skip this step
if all(
    path.exists()
    for path in [
        treatment_best_params_outpath,
        cfret_treatment_cluster_df_outpath,
    ]
):
    # load the profiles
    cfret_treatment_clustered_df = pl.read_parquet(cfret_treatment_cluster_df_outpath)
else:
    # here we are clustering each treatment regardless of heart
    # this will allow us to see how each treatment affects the population as a whole
    cfret_treatment_clustered_df, cfret_treatment_clustered_best_params = (
        optimized_clustering(
            profiles=cfret_df,
            meta_features=cfret_meta,
            morph_features=cfret_feats,
            treatment_col=treatment_col,
            param_grid=cfret_cluster_param_grid,
            n_trials=200,
            n_jobs=1,
        )
    )

    # save best params as json and dataframe as parquet
    cfret_treatment_clustered_df.write_parquet(cfret_treatment_cluster_df_outpath)
    with open(treatment_best_params_outpath, "w") as f:
        json.dump(
            cfret_treatment_clustered_best_params,
            f,
            indent=4,
        )


# check if the files exist, if they do skip this step aswell
if all(
    path.exists()
    for path in [
        cfret_treatment_heart_cluster_df_outpath,
        treatment_heart_best_params_outpath,
    ]
):
    # load the profiles
    cfret_treatment_heart_clustered_df = pl.read_parquet(
        cfret_treatment_heart_cluster_df_outpath
    )
else:
    # here we are clustering each treatment-heart combination
    # this will allow us to see how each heart responds to each treatment
    cfret_treatment_heart_clustered_df, cfret_treatment_heart_clustered_best_params = (
        optimized_clustering(
            profiles=cfret_df,
            meta_features=cfret_meta,
            morph_features=cfret_feats,
            treatment_col=treatment_heart_col,
            param_grid=cfret_cluster_param_grid,
            n_trials=200,
            n_jobs=1,
        )
    )

    # save best params as json and dataframe as parquet
    cfret_treatment_heart_clustered_df.write_parquet(
        cfret_treatment_heart_cluster_df_outpath
    )
    with open(treatment_heart_best_params_outpath, "w") as f:
        json.dump(
            cfret_treatment_heart_clustered_best_params,
            f,
            indent=4,
        )


# Measure phenotypic activity between clusters

# In[7]:


# setting output paths
treatment_dist_scores_outpath = (
    results_dir / "treatment_phenotypic_dist_scores.csv"
).resolve()
treatment_heart_dist_scores_outpath = (
    results_dir / "treatment_heart_dist_scores.csv"
).resolve()

if all(
    path.exists()
    for path in [
        treatment_dist_scores_outpath,
        treatment_heart_dist_scores_outpath,
    ]
):
    print("Distance scores already exist, skipping this step.")

    # load the distance scores
    treatment_dist_scores = pl.read_csv(treatment_dist_scores_outpath)
    treatment_heart_dist_scores = pl.read_csv(treatment_heart_dist_scores_outpath)

else:
    # measuring phenotypic activity
    treatment_dist_scores = measure_phenotypic_activity(
        profiles=cfret_treatment_clustered_df,
        on_signature=on_sigs,
        off_signature=off_sigs,
        ref_treatment="DMSO_heart_11",
    )

    treatment_heart_dist_scores = measure_phenotypic_activity(
        profiles=cfret_treatment_heart_clustered_df,
        on_signature=on_sigs,
        off_signature=off_sigs,
        ref_treatment="DMSO_heart_11",
        treatment_col=treatment_heart_col,
    )

    # save those as csv files
    treatment_dist_scores.write_csv(treatment_dist_scores_outpath)
    treatment_heart_dist_scores.write_csv(treatment_heart_dist_scores_outpath)


# Rank treatments

# In[8]:


# setting outptut paths
treatment_rankings_outpath = (results_dir / "treatment_rankings.csv").resolve()
treatment_heart_rankings_outpath = (
    results_dir / "treatment_heart_rankings.csv"
).resolve()

if all(
    path.exists()
    for path in [
        treatment_rankings_outpath,
        treatment_heart_rankings_outpath,
    ]
):
    print("Rankings already exist, skipping this step.")

    # load the rankings
    treatment_rankings = pl.read_csv(treatment_rankings_outpath)
    treatment_heart_rankings = pl.read_csv(treatment_heart_rankings_outpath)
else:
    treatment_rankings = identify_compound_hit(
        distance_df=treatment_dist_scores, method="weighted_sum"
    )

    treatment_heart_rankings = identify_compound_hit(
        distance_df=treatment_heart_dist_scores, method="weighted_sum"
    )

    # save as csv files
    treatment_rankings.write_csv(treatment_rankings_outpath)
    treatment_heart_rankings.write_csv(treatment_heart_rankings_outpath)
