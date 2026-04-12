#!/usr/bin/env python

# # Leave-One-Gene-Out Analysis
#
# In this analysis, we perform a leave-one-gene-out (LOGO) evaluation to assess whether data leakage from pooling single-cell profiles inflates phenotypic activity scores.
#
# For each gene known to be associated with a target phenotype (e.g., **Prometaphase**):
# 1. Its associated cells are **excluded** from building the on/off signatures.
# 2. The on/off signatures are computed from the remaining cells in the target phenotype against the negative controls.
# 3. The **excluded gene's cells** are then scored against those signatures.
#
# Here, **the phenotype (e.g., Prometaphase) is set as the target** and **negative controls (negcon) are used as the reference baseline**. The scores reflect how close the held-out gene's cells are to the target phenotype. This means:
# - **Lower scores = good** — the held-out gene's cells are morphologically similar to the target phenotype, indicating a genuine phenotypic signal.
# - If data leakage were present (i.e., the gene's own cells contributed to the signature), scores would be artificially low. Under the LOGO design, **scores that remain low confirm the signal is real** — those cells genuinely resemble the target phenotype even when they played no role in building the signature.
#
# To create a null distribution or negative control baseline, we shuffle the feature profiles to break the biological relationships while preserving the data structure.

# ## Importing packages

# In[1]:


import pathlib
import sys

import numpy as np
import polars as pl
from tqdm import tqdm

sys.path.append("../../")
from buscar.metrics import calculate_buscar_scores
from buscar.signatures import get_signatures

from utils.data_utils import shuffle_feature_profiles
from utils.io_utils import load_configs, load_profiles

# ## Setting helper functions

# In[2]:


def shuffle_signatures(
    on_sig: list[str], off_sig: list[str], all_features: list[str], seed: int = 0
) -> tuple[list[str], list[str]]:
    """
    Breaks biological meaning of on/off signatures by randomly sampling
    features from the full feature space, while preserving the original
    on/off size ratio.

    Preserves:
      - len(on_sig) and len(off_sig)  ← ratio intact
      - Features drawn from same pool as real signatures

    Breaks:
      - Which specific features are "on" vs "off"
      - Any biological grouping derived from KS test
    """
    rng = np.random.default_rng(seed)

    n_on = len(on_sig)
    n_off = len(off_sig)

    # guard: need enough features to fill both without overlap
    if n_on + n_off > len(all_features):
        raise ValueError(
            f"Not enough features ({len(all_features)}) to fill "
            f"on ({n_on}) + off ({n_off}) without replacement"
        )

    # sample without replacement so on and off don't overlap
    sampled = rng.choice(all_features, size=n_on + n_off, replace=False)

    shuffled_on = sampled[:n_on].tolist()
    shuffled_off = sampled[n_on:].tolist()

    return shuffled_on, shuffled_off


# ## setting input and output paths

# In[3]:


# set data path
data_dir = pathlib.Path("../0.download-data/data/sc-profiles/").resolve(strict=True)
mitocheck_data = (data_dir / "mitocheck").resolve(strict=True)

# sertting mitocheck paths
mitocheck_profile_path = (mitocheck_data / "mitocheck_concat_profiles.parquet").resolve(
    strict=True
)

# setting config paths
# ensg_genes_config_path = (
#     mitocheck_data / "mitocheck_ensg_to_gene_symbol_mapping.json"
# ).resolve(strict=True)
mitocheck_feature_space_config = (
    mitocheck_data / "mitocheck_feature_space_configs.json"
).resolve(strict=True)

# set results output path
results_dir = pathlib.Path("./results/").resolve()
results_dir.mkdir(exist_ok=True)

moa_analysis_output = (results_dir / "logo_analysis").resolve()
moa_analysis_output.mkdir(exist_ok=True)


# ## Loading data

# In[4]:


# load in configs
# ensg_genes_decoder = load_configs(ensg_genes_config_path)
feature_space_configs = load_configs(mitocheck_feature_space_config)
meta_feats = feature_space_configs["metadata-features"]
morph_feats = feature_space_configs["morphology-features"]


# In[5]:


# load in mitocheck profiles
mitocheck_df = load_profiles(mitocheck_profile_path)
mitocheck_df = mitocheck_df.select(pl.col(meta_feats + morph_feats))

# removing failed qc
mitocheck_df = mitocheck_df.filter(pl.col("Metadata_Gene") != "failed QC")

# replace "negative_control" and "positive_control" values in Metadata_Gene with
# "negcon" and "poscon" respectively
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


# In[6]:


labeled_mitocheck_df = mitocheck_df.filter(
    (pl.col("Mitocheck_Phenotypic_Class") != "negcon")
    & (pl.col("Mitocheck_Phenotypic_Class") != "poscon")
)

print("Shape of the labeled mitocheck profiles:", labeled_mitocheck_df.shape)
labeled_mitocheck_df.head()


# In[7]:


# Creating a proportion dataframe for all genes and phenotypic classes
cell_proportion_df = (
    mitocheck_df.filter(
        (pl.col("Mitocheck_Phenotypic_Class") != "negcon")
        & (pl.col("Mitocheck_Phenotypic_Class") != "poscon")
    )
    .group_by(["Metadata_Gene", "Mitocheck_Phenotypic_Class"])
    .agg(pl.len().alias("count"))
    .with_columns(pl.col("count").sum().over("Metadata_Gene").alias("total_count"))
    .with_columns((pl.col("count") / pl.col("total_count")).alias("proportion"))
)


# Get cell state information

# In[8]:


cell_states = (
    # remove negcon and poscon since they do not have cell state information
    mitocheck_df.filter(
        (pl.col("Mitocheck_Phenotypic_Class") != "negcon")
        & (pl.col("Mitocheck_Phenotypic_Class") != "poscon")
    )
    .select("Mitocheck_Phenotypic_Class")
    .unique()
    .to_series()
    .to_list()
)


# ## LOGO analysis

# In[9]:


# parameters for the analysis
shuffle_flag = True
seed = 0


# In[10]:


if shuffle_flag:
    print("Shuffling the mitocheck profiles...")
    shuffled_mitocheck_df = shuffle_feature_profiles(
        profiles=labeled_mitocheck_df,
        feature_cols=morph_feats,
        method="column",
        label_col="Mitocheck_Phenotypic_Class",
        seed=seed,
    )


# In[11]:


# select data based on shuffle_flag
profiles = shuffled_mitocheck_df if shuffle_flag else labeled_mitocheck_df

negcon_profiles = mitocheck_df.filter(
    pl.col("Mitocheck_Phenotypic_Class") == "negcon"
).sample(fraction=0.01)

on_off_sigs = []
min_cells = 2

results_df = []
for cell_state in tqdm(cell_states, desc="Processing cell states"):
    # state of interest for this cell state
    selected_state = profiles.filter(pl.col("Mitocheck_Phenotypic_Class") == cell_state)

    # genes that are associated with this cell state
    genes_associated_with_state = (
        selected_state.select("Metadata_Gene").unique().to_series().to_list()
    )

    # genes that are not associated with this cell state
    genes_not_associated_with_state = (
        profiles.filter(~pl.col("Metadata_Gene").is_in(genes_associated_with_state))
        .select("Metadata_Gene")
        .unique()
        .to_series()
        .to_list()
    )

    associated_gene_scores = []
    for gene in tqdm(
        genes_associated_with_state,
        desc=f"  Processing genes for {cell_state}",
        leave=False,
    ):
        # filter the target profiles to only include cells treated with the current
        # gene of interest
        heldout_df = selected_state.filter(pl.col("Metadata_Gene") == gene)

        # skip genes with too few cells (EMD requires >= 2 samples)
        if heldout_df.height < min_cells:
            print(
                f"Skipping gene '{gene}': only {heldout_df.height} cell(s), need >= "
                f"{min_cells}"
            )
            # create an empty dataframe with the same structure as the
            # associated_gene_score to maintain consistency
            associated_gene_score = pl.DataFrame(
                {
                    "target": pl.Series([cell_state], dtype=pl.String),
                    "perturbation": pl.Series([gene], dtype=pl.String),
                    "on_buscar_scores": pl.Series([None], dtype=pl.Float64),
                    "off_buscar_scores": pl.Series([None], dtype=pl.Float64),
                    "is_reference_distance": pl.Series([None], dtype=pl.Boolean),
                    "proportion": pl.Series([None], dtype=pl.Float64),
                    "is_associated": pl.Series([None], dtype=pl.Boolean),
                }
            )
            associated_gene_scores.append(associated_gene_score)
            continue

        # remove the current gene's cells from the positive control pool
        # to prevent data leakage: the gene being ranked must not influence its own
        # signature
        state_pool = selected_state.filter(pl.col("Metadata_Gene") != gene)

        # generate on and off signatures (leave-one-out: current gene's cells excluded)
        morph_feats = feature_space_configs["morphology-features"]
        on_sig, off_sig, _ = get_signatures(
            state_pool,
            negcon_profiles,
            morph_feats=morph_feats,
            test_method="ks_test",
            p_threshold=0.05,
            seed=seed,
        )

        # concatenating negcon, the gene that has been held out, and the state_pool
        test_df = pl.concat([negcon_profiles, heldout_df, state_pool])

        test_df = test_df.with_columns(
            pl.when(pl.col("Metadata_Gene") == "negcon")
            .then(pl.lit("negcon"))
            .when(pl.col("Metadata_Gene") == gene)
            .then(pl.col("Metadata_Gene"))
            .otherwise(pl.lit(cell_state))  # label pooled target as cell state
            .alias("_labeled_references")
        )

        if shuffle_flag:
            # shuffle the on and off signatures and shuffle
            on_sig, off_sig = shuffle_signatures(
                on_sig, off_sig, morph_feats, seed=seed
            )
            test_df = shuffle_feature_profiles(
                profiles=test_df,
                feature_cols=morph_feats,
                method="column",
                seed=seed,
            )

        # if no signature was found, skip the gene
        if len(on_sig) == 0 or len(off_sig) == 0:
            print(f"skipping {gene}")
            continue

        # rank the gene using the generated signatures
        associated_gene_score = calculate_buscar_scores(
            profiles=test_df,
            meta_cols=feature_space_configs["metadata-features"],
            on_morphology_signature=on_sig,
            off_morphology_signature=off_sig,
            target=cell_state,
            ref_state="negcon",
            perturbation_col="_labeled_references",
            n_threads=1,
            raw_emd_scores=False,
        )

        # calculate the proportion of cells that make up this phenotype with the
        # current gene perturbation
        try:
            cell_state_proportion = cell_proportion_df.filter(
                (pl.col("Metadata_Gene") == gene)
                & (pl.col("Mitocheck_Phenotypic_Class") == cell_state)
            )["proportion"][0]
        except IndexError:
            cell_state_proportion = 0.0

        # remove negcon scores; we are only interested in the scores of the gene
        associated_gene_score = associated_gene_score.filter(
            pl.col("perturbation") != "negcon"
        )

        # add cell state proportion to the associated gene scores df
        associated_gene_score = associated_gene_score.with_columns(
            pl.lit(cell_state_proportion).alias("proportion"),
            pl.lit(True).alias("is_associated"),
        )

        # store on and off signatures
        on_off_sigs.append((cell_state, on_sig, off_sig))
        associated_gene_scores.append(associated_gene_score)

    if len(associated_gene_scores) > 0:
        associated_gene_scores = pl.concat(associated_gene_scores)
    else:
        associated_gene_scores = pl.DataFrame(
            schema={
                "target": pl.String,
                "perturbation": pl.String,
                "on_buscar_scores": pl.Float64,
                "off_buscar_scores": pl.Float64,
                "is_reference_distance": pl.Boolean,
                "proportion": pl.Float64,
                "is_associated": pl.Boolean,
            }
        )

    # Step 2: rank genes that are not associated with this cell state

    # create on and off sigs with pooled poscon cell state
    on_sig, off_sig, _ = get_signatures(
        ref_profiles=selected_state,
        target_profiles=negcon_profiles,
        morph_feats=morph_feats,
        test_method="ks_test",
        p_threshold=0.05,
        seed=seed,
    )

    test_non_associated_df = pl.concat(
        [
            selected_state,
            profiles.filter(
                pl.col("Metadata_Gene").is_in(genes_not_associated_with_state)
            ),
            negcon_profiles,
        ]
    )

    # the genes not associated with the cell state, and the target phenotype pool
    test_non_associated_df = test_non_associated_df.with_columns(
        pl.when(pl.col("Metadata_Gene") == "negcon")
        .then(pl.lit("negcon"))
        .when(pl.col("Metadata_Gene").is_in(genes_associated_with_state))
        .then(pl.lit(cell_state))  # label pooled target as cell state
        .otherwise(pl.col("Metadata_Gene"))  # keep non-associated as gene names
        .alias("_labeled_references")
    )

    if shuffle_flag:
        on_sig, off_sig = shuffle_signatures(on_sig, off_sig, morph_feats, seed=seed)
        test_non_associated_df = shuffle_feature_profiles(
            profiles=test_non_associated_df,
            feature_cols=morph_feats,
            method="column",
            seed=seed,
        )

    # rank all treatments that are not associated with this cell state using the pooled
    # poscon signatures
    not_associated_gene_scores = calculate_buscar_scores(
        profiles=test_non_associated_df,
        meta_cols=meta_feats,
        on_morphology_signature=on_sig,
        off_morphology_signature=off_sig,
        target=cell_state,
        ref_state="negcon",
        perturbation_col="_labeled_references",
        n_threads=1,
        seed=seed,
        raw_emd_scores=False,
    )

    # remove scores of genes that are associated with the cell state
    not_associated_gene_scores = not_associated_gene_scores.filter(
        pl.col("perturbation").is_in(genes_not_associated_with_state)
    )

    # add proportion of cells; if a gene has no cells in this state, assign 0
    not_associated_gene_scores = not_associated_gene_scores.join(
        cell_proportion_df.select(
            ["Metadata_Gene", "Mitocheck_Phenotypic_Class", "proportion"]
        ),
        left_on=["perturbation", "target"],
        right_on=["Metadata_Gene", "Mitocheck_Phenotypic_Class"],
        how="left",
    ).with_columns(
        pl.col("proportion").fill_null(0.0), pl.lit(False).alias("is_associated")
    )

    # final result for this cell state
    results_df.append(
        pl.concat([associated_gene_scores, not_associated_gene_scores], how="vertical")
    )

# step 3: store results
if len(results_df) > 0:
    results_df = pl.concat(results_df)
    output_filename = f"{'shuffled' if shuffle_flag else 'original'}_mitocheck_logo_analysis_results.parquet"
    results_df.write_parquet(moa_analysis_output / output_filename)
else:
    print("No results generated to save.")
