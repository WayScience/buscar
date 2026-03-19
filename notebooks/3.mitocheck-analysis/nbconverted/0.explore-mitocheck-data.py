#!/usr/bin/env python

# # 0. MitoCheck Dataset: Exploratory Data Analysis
#
# ## Overview
# This notebook performs an exploratory data analysis (EDA) of the [MitoCheck](https://www.mitocheck.org/) single-cell morphology dataset.
# The MitoCheck project systematically knocked down ~21,000 human genes via siRNA and classified the resulting cell phenotypes using time-lapse microscopy.
# Each cell in this dataset is assigned a **phenotypic class** (e.g., "Mitosis", "Apoptosis", "Shape") based on its observed morphology after gene knockdown.
#
# We use this dataset as a benchmark for the **buscar** compound-prioritization pipeline, since it provides ground-truth phenotypic labels that let us evaluate whether morphological signatures capture biologically meaningful variation.
#
# **Sections in this notebook:**
# 1. **Setup** — Imports, paths, and configuration loading
# 2. **Data Loading & Cleaning** — Read profiles, remove QC failures, standardize control labels
# 3. **Dataset Overview** — Dimensions, feature space, and unique genes/classes
# 4. **Phenotypic Class Distribution** — Cell counts per class and control vs. treatment breakdown
# 5. **Gene-Level Phenotype Composition** — Per-gene phenotype proportions and cell-count distributions
# 6. **Data Quality Checks** — Missing values, duplicate rows
# 7. **Feature Correlations** — Identify redundant features via correlation clustermap

# In[1]:


import itertools
import pathlib
import sys

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

# allow imports from the project root (e.g., utils/ and buscar/)
sys.path.append("../../")
from utils.io_utils import load_configs, load_profiles

# ## 1. Setup: Input & Output Paths
# Define all file paths for input data, configuration files, and output directories.
# All results (tables, plots) are saved under `results/eda/` to keep EDA outputs separate from downstream analysis artifacts.

# In[2]:


# ---------------------
# Input paths
# ---------------------
# Root directory containing downloaded single-cell morphology profiles
data_dir = pathlib.Path("../0.download-data/data/sc-profiles/").resolve(strict=True)
mitocheck_data = (data_dir / "mitocheck").resolve(strict=True)

# Concatenated MitoCheck single-cell profiles (one row per cell)
mitocheck_profile_path = (mitocheck_data / "mitocheck_concat_profiles.parquet").resolve(
    strict=True
)

# ENSG → gene-symbol mapping (MitoCheck uses Ensembl gene IDs internally)
# ensg_genes_config_path = (
#     mitocheck_data / "mitocheck_ensg_to_gene_symbol_mapping.json"
# ).resolve(strict=True)

# Feature-space config: defines which columns are metadata vs. morphology features
mitocheck_feature_space_config = (
    mitocheck_data / "mitocheck_feature_space_configs.json"
).resolve(strict=True)

# ---------------------
# Output paths
# ---------------------
# All EDA outputs (CSV tables + plots) are stored under results/eda/
results_dir = pathlib.Path("results/").resolve()
results_dir.mkdir(exist_ok=True)

eda_results_dir = (results_dir / "eda").resolve()
eda_results_dir.mkdir(exist_ok=True)

plots_dir = (eda_results_dir / "plots/").resolve()
plots_dir.mkdir(exist_ok=True)


# ## 2. Data Loading & Cleaning
# Load the concatenated MitoCheck profiles and apply two cleaning steps:
# 1. **Remove QC failures** — Rows where `Metadata_Gene == "failed QC"` did not pass upstream quality control and are excluded.
# 2. **Standardize control labels** — Rename `"negative control"` → `"negcon"` and `"positive control"` → `"poscon"` so control labels are consistent with the buscar pipeline conventions.

# In[3]:


# Load configuration files:
#   - ensg_genes_decoder: maps Ensembl gene IDs (ENSG*) to human-readable gene symbols
#   - feature_space_configs: lists metadata columns vs. morphology feature columns
# ensg_genes_decoder = load_configs(ensg_genes_config_path)
feature_space_configs = load_configs(mitocheck_feature_space_config)


# In[4]:


# Load the parquet file into a Polars DataFrame (one row = one single cell)
mitocheck_df = load_profiles(mitocheck_profile_path)

# Remove rows that failed upstream QC — these cells have unreliable measurements
mitocheck_df = mitocheck_df.filter(pl.col("Metadata_Gene") != "failed QC")

# Standardize control labels to match buscar pipeline conventions:
#   "negative control" → "negcon"  (untreated / non-targeting siRNA)
#   "positive control" → "poscon"  (known phenotype-inducing siRNA)
mitocheck_df = mitocheck_df.with_columns(
    pl.col("Metadata_Gene").map_elements(
        lambda x: (
            "negcon"
            if x == "negative control"
            else ("poscon" if x == "positive control" else x)
        ),
        return_dtype=pl.String,
    )
)

print(f"Loaded profiles shape: {mitocheck_df.shape}")
mitocheck_df.head()


# ## 3. Dataset Overview
# Summarize the dataset dimensions: total cells, number of metadata vs. morphology columns, unique genes tested, and unique phenotypic classes.
# This gives a high-level sense of the dataset's size and complexity before diving deeper.

# In[5]:


# Split columns into metadata (annotations) and morphology (numeric features)
meta_cols = [
    c
    for c in mitocheck_df.columns
    if c.startswith("Metadata_") or c in feature_space_configs["metadata-features"]
]
morph_cols = [c for c in mitocheck_df.columns if c not in meta_cols]

# Compute high-level dataset statistics
n_cells, n_total_cols = mitocheck_df.shape
n_genes = mitocheck_df.filter(~pl.col("Metadata_Gene").is_in(["negcon", "poscon"]))[
    "Metadata_Gene"
].n_unique()
n_phenotypes = mitocheck_df["Mitocheck_Phenotypic_Class"].n_unique()

print(f"Total cells (rows):            {n_cells:,}")
print(f"Total columns:                 {n_total_cols}")
print(f"  - Metadata columns:          {len(meta_cols)}")
print(f"  - Morphology features:       {len(morph_cols)}")
print(f"Unique genes (excl. controls): {n_genes}")
print(f"Unique phenotypic classes:     {n_phenotypes}")
print(
    f"\nPhenotypic classes: {sorted(mitocheck_df['Mitocheck_Phenotypic_Class'].unique().to_list())}"
)


# ## 4. Phenotypic Class Distribution
#
# ### 4a. Control vs. Treatment Breakdown
# Count how many cells belong to negative controls (`negcon`), positive controls (`poscon`), and gene-knockdown treatments.
# This breakdown is important because downstream buscar analysis compares treatment signatures against the negative-control distribution.

# In[6]:


# Separate cells into control groups and gene-knockdown treatments
negcon_df = mitocheck_df.filter(pl.col("Mitocheck_Phenotypic_Class") == "negcon")
poscon_df = mitocheck_df.filter(pl.col("Mitocheck_Phenotypic_Class") == "poscon")
treatment_df = mitocheck_df.filter(
    ~pl.col("Mitocheck_Phenotypic_Class").is_in(["negcon", "poscon"])
)

print(f"Negative control cells: {negcon_df.shape[0]:,}")
print(f"Positive control cells: {poscon_df.shape[0]:,}")
print(f"  - Positive control genes: {poscon_df['Metadata_Gene'].unique().to_list()}")
print(f"Treatment cells:        {treatment_df.shape[0]:,}")


# ### 4b. Cell Counts per Phenotypic Class
# Visualize how many treatment cells fall into each MitoCheck phenotypic class.
# A log-scale y-axis is used because some classes (e.g., "Shape") dominate while others are very rare — understanding this imbalance is critical for interpreting downstream results.

# In[7]:


# Count cells per phenotypic class (excluding controls)
cell_counts = (
    treatment_df["Mitocheck_Phenotypic_Class"]
    .value_counts()
    .sort("count", descending=True)
)

# Bar plot with linear y-axis to visualize class imbalance
cell_counts_pd = cell_counts.to_pandas()
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(
    cell_counts_pd["Mitocheck_Phenotypic_Class"],
    cell_counts_pd["count"],
    color="steelblue",
)
ax.set_title("Cell counts per phenotypic class (treatment cells only)")
ax.set_xlabel("Phenotypic class")
ax.set_ylabel("Cell count")
# Set y-axis limit to 500
ax.set_ylim(0, 500)
# Add the number of cells on top of each bar
for bar, count in zip(bars, cell_counts_pd["count"]):
    height = bar.get_height()
    ax.annotate(
        f"{int(count):,}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
        color="black",
    )
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(plots_dir / "phenotypic_class_distribution.png", dpi=300)
plt.show()


# In[8]:


# Persist the cell-count table for reference in later notebooks
cell_counts.write_csv(eda_results_dir / "cell_counts_per_phenotypic_class.csv")

# Display the full table
cell_counts


# ## 5. Gene-Level Phenotype Composition
#
# For each gene knockdown, what proportion of its cells belong to each phenotypic class?
# A gene whose cells are dominated by a single class has a "pure" phenotype, whereas a gene with a roughly uniform distribution across many classes may have a weak or noisy phenotypic effect.
# This per-gene breakdown helps us understand phenotype purity and will inform how we aggregate signatures in downstream buscar analysis.

# In[9]:


# For each gene, compute the proportion of cells in every phenotypic class.
# Steps:
#   1. Group by (gene, phenotypic class) and count cells
#   2. Compute total cells per gene using a window function
#   3. Derive proportion = count / total_count
phenotype_proportions = (
    treatment_df.group_by(["Metadata_Gene", "Mitocheck_Phenotypic_Class"])
    .agg(pl.len().alias("count"))
    .with_columns(pl.col("count").sum().over("Metadata_Gene").alias("total_count"))
    .with_columns((pl.col("count") / pl.col("total_count")).alias("proportion"))
    .select(
        [
            "Metadata_Gene",
            "Mitocheck_Phenotypic_Class",
            "count",
            "total_count",
            "proportion",
        ]
    )
    .sort(["Metadata_Gene", "proportion"], descending=[False, True])
)

# Save the full table for downstream use
phenotype_proportions.write_csv(eda_results_dir / "phenotype_proportions_per_gene.csv")

# Preview the first few rows
phenotype_proportions.head()


# In[10]:


# Spot-check: inspect phenotype composition for a single gene (PAPPA)
# to verify the table looks reasonable and proportions sum to 1.0
phenotype_proportions.filter(pl.col("Metadata_Gene") == "PAPPA")


# In[11]:


# How many cells does each gene knockdown contribute?
# Genes with very few cells may produce unreliable signatures.
cells_per_gene = (
    treatment_df.group_by("Metadata_Gene")
    .agg(pl.len().alias("n_cells"))
    .sort("n_cells", descending=True)
)

# Histogram of cells-per-gene with a median reference line
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(
    cells_per_gene["n_cells"].to_list(), bins=50, color="steelblue", edgecolor="white"
)
ax.set_title("Distribution of cell counts per gene knockdown")
ax.set_xlabel("Number of cells")
ax.set_ylabel("Number of genes")
median_val = cells_per_gene["n_cells"].median()
ax.axvline(median_val, color="red", linestyle="--", label=f"Median = {median_val:.0f}")
ax.legend()
plt.tight_layout()
plt.savefig(plots_dir / "cells_per_gene_distribution.png", dpi=150)
plt.show()


# ## 6. Data Quality Checks
# Before using this data in downstream analyses, verify it is free from common quality issues:
# missing/null values and duplicate rows. Problems here could silently bias morphological signatures.

# ### 6a. Missing Values
# Check every column for null entries. Missing morphology features could distort mean/median signatures,
# and missing metadata would break downstream grouping operations.

# In[12]:


# Count null values in every column and report any that are non-zero
null_counts = mitocheck_df.null_count()
cols_with_nulls = {
    col: null_counts[col][0] for col in null_counts.columns if null_counts[col][0] > 0
}

if cols_with_nulls:
    print(f"Columns with missing values ({len(cols_with_nulls)}):")
    for col, cnt in sorted(cols_with_nulls.items(), key=lambda x: -x[1]):
        pct = cnt / mitocheck_df.shape[0] * 100
        print(f"  {col}: {cnt:,} ({pct:.2f}%)")
else:
    print("No missing values found in any column.")


# ### 6b. Duplicate Rows
# Each cell should have a unique `Cell_UUID`. Duplicates would inflate sample sizes and distort aggregate statistics.

# In[13]:


# Verify that every cell has a unique identifier (no duplicate rows)
if "Cell_UUID" in mitocheck_df.columns:
    n_unique_uuids = mitocheck_df["Cell_UUID"].n_unique()
    n_total = mitocheck_df.shape[0]
    n_dupes = n_total - n_unique_uuids
    print(f"Total rows:        {n_total:,}")
    print(f"Unique Cell_UUIDs: {n_unique_uuids:,}")
    print(f"Duplicate UUIDs:   {n_dupes:,}")
else:
    print("No Cell_UUID column found; skipping duplicate check.")


# ## 7. Feature Correlations
# Highly correlated features carry redundant information and can inflate variance in downstream analyses (e.g., PCA, signature generation).
# We compute the full pairwise Pearson correlation matrix across all morphology features and visualize it as a **clustermap** (with Ward hierarchical clustering on both axes) to reveal groups of co-varying features.
#
# We also count the number of feature pairs with |r| > 0.9 as a rough measure of redundancy.

# In[14]:


# Compute pairwise Pearson correlation matrix for all morphology features
corr_matrix = mitocheck_df.select(morph_cols).to_pandas().corr()

# Clustermap: hierarchical clustering (Ward linkage) groups correlated features together,
# making blocks of high correlation easy to spot visually.
# cbar_pos=(left, bottom, width, height) — width/height set to 1/4 of original
g = sns.clustermap(
    corr_matrix,
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    figsize=(12, 10),
    xticklabels=False,
    yticklabels=False,
    dendrogram_ratio=(0.1, 0.1),
    cbar_pos=(0.02, 0.8, 0.008, 0.04),
    method="ward",
    metric="euclidean",
)

# Place the title at the very top of the figure (not on the heatmap axes)
g.figure.suptitle("Morphology feature correlation clustermap", fontsize=14, y=1.02)

# Adjust layout so the heatmap sits snugly against the dendrograms
g.figure.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

plt.savefig(
    plots_dir / "feature_correlation_clustermap.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()

# Count feature pairs with |correlation| > 0.9 as a measure of redundancy
high_corr_pairs = [
    (f1, f2, corr_matrix.loc[f1, f2])
    for f1, f2 in itertools.combinations(morph_cols, 2)
    if abs(corr_matrix.loc[f1, f2]) > 0.9
]
print(f"Feature pairs with |correlation| > 0.9: {len(high_corr_pairs)}")
