#!/usr/bin/env python

# # Leave on gene out analysis

# In[1]:


import pathlib
import sys

import polars as pl
from tqdm import tqdm

sys.path.append("../../")
from buscar.metrics import calculate_buscar_scores
from buscar.signatures import get_signatures

from utils.data_utils import shuffle_feature_profiles, shuffle_signatures
from utils.io_utils import load_configs, load_profiles

# setting input and output paths

# In[2]:


# set data path
data_dir = pathlib.Path("../0.download-data/data/sc-profiles/").resolve(strict=True)
mitocheck_data = (data_dir / "mitocheck").resolve(strict=True)

# sertting mitocheck paths
mitocheck_profile_path = (mitocheck_data / "mitocheck_concat_profiles.parquet").resolve(
    strict=True
)

mitocheck_feature_space_config = (
    mitocheck_data / "mitocheck_feature_space_configs.json"
).resolve(strict=True)

# set results output path
results_dir = pathlib.Path("./results/").resolve()
results_dir.mkdir(exist_ok=True)

logo_analysis_output = (results_dir / "logo_analysis").resolve()
logo_analysis_output.mkdir(exist_ok=True)


# In[3]:


# load in configs
feature_space_configs = load_configs(mitocheck_feature_space_config)
meta_feats = feature_space_configs["metadata-features"]
morph_feats = feature_space_configs["morphology-features"]


# In[4]:


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


# In[5]:


labeled_mitocheck_df = mitocheck_df.filter(
    (pl.col("Mitocheck_Phenotypic_Class") != "negcon")
    & (pl.col("Mitocheck_Phenotypic_Class") != "poscon")
)

print("Shape of the labeled mitocheck profiles:", labeled_mitocheck_df.shape)
labeled_mitocheck_df.head()


# In[6]:


cell_proportion_df = (
    labeled_mitocheck_df.group_by(["Metadata_Gene", "Mitocheck_Phenotypic_Class"])
    .agg(pl.len().alias("cell_count"))
    .with_columns(
        (pl.col("cell_count") / pl.col("cell_count").sum().over("Metadata_Gene")).alias(
            "proportion"
        )
    )
).sort(["Metadata_Gene", "proportion"])


# Here, **Prometaphase is used as the reference baseline**, so scores reflect how close the held-out gene's cells are to the Prometaphase phenotype. This means:
# - **Lower scores = good** — the held-out gene's cells are morphologically similar to Prometaphase, indicating genuine phenotypic signal.
# - If data leakage were present (i.e., the gene's own cells contributed to the signature), scores would be artificially low. Under the LOGO design, **scores that remain low confirm the signal is real** — those cells genuinely resemble Prometaphase even when they played no role in building the signature.
#
# To make a negative control baseline, we shuffled the lablels and the on and off signature scores. For the on and off signature scores we retained the same s

# Get all cell phenotypes

# In[7]:


all_phenotypes = (
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


# Caclulate the proportion of cell states that makes up a specific gene

# In[ ]:


# parameters for the analysis
shuffle_flag = False
seed = 0
min_cells = 5
n_iterations = 5
control_fraction = 0.01


# In[9]:


if shuffle_flag:
    print("Shuffling the mitocheck profiles...")
    shuffled_mitocheck_df = shuffle_feature_profiles(
        profiles=labeled_mitocheck_df,
        feature_cols=morph_feats,
        method="column",
        seed=seed,
    )


# In[ ]:


# select data based on shuffle_flag
profiles = shuffled_mitocheck_df if shuffle_flag else labeled_mitocheck_df


on_off_sigs = []

all_iter_results = []
for iter_idx in range(n_iterations):
    iter_seed = seed + iter_idx

    # resample negative control profiles for this iteration
    negcon_profiles = mitocheck_df.filter(
        pl.col("Mitocheck_Phenotypic_Class") == "negcon"
    ).sample(fraction=control_fraction, seed=iter_seed, with_replacement=False)

    results_df = []
    for phenotype in tqdm(
        all_phenotypes, desc=f"Iteration {iter_idx + 1}/{n_iterations}: cell states"
    ):
        # poscon phenotype of interest for this cell state
        target_df = profiles.filter(pl.col("Mitocheck_Phenotypic_Class") == phenotype)

        # genes that are associated with this cell state
        genes_associated_with_state = (
            target_df.select("Metadata_Gene").unique().to_series().to_list()
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
            desc=f"  Processing genes for {phenotype}",
            leave=False,
        ):
            # filter the target profiles to only include cells treated with the current
            # gene of interest
            heldout_df = target_df.filter(pl.col("Metadata_Gene") == gene)

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
                        "target": pl.Series([phenotype], dtype=pl.String),
                        "perturbation": pl.Series([gene], dtype=pl.String),
                        "on_buscar_scores": pl.Series([None], dtype=pl.Float64),
                        "off_buscar_scores": pl.Series([None], dtype=pl.Float64),
                        "is_reference_distance": pl.Series([None], dtype=pl.Boolean),
                        "proportion": pl.Series([None], dtype=pl.Float64),
                        "associated_with_phenotype": pl.Series(
                            [True], dtype=pl.Boolean
                        ),
                    }
                )
                associated_gene_scores.append(associated_gene_score)
                continue

            # remove the current gene's cells from target_df to create
            # to prevent data leakage: the gene being ranked must not influence its own
            # signature
            phenotype_pool = target_df.filter(pl.col("Metadata_Gene") != gene)

            # generate on and off signatures (leave-one-out: current gene's cells excluded)
            morph_feats = feature_space_configs["morphology-features"]
            on_sig, off_sig, _ = get_signatures(
                ref_profiles=negcon_profiles,
                target_profiles=phenotype_pool,
                morph_feats=morph_feats,
                test_method="ks_test",
                p_threshold=0.05,
                seed=iter_seed,
            )

            # concatenating negcon and the gene that has been held out
            test_df = pl.concat([negcon_profiles, heldout_df, phenotype_pool])

            # add a column that differentiates the heldout gene from the rest of the phenotype pool
            test_df = test_df.with_columns(
                pl.when(pl.col("Metadata_Gene") == "negcon")  # reference
                .then(pl.lit("negcon"))
                .when(pl.col("Metadata_Gene") == gene)  # held out gene that is scored
                .then(pl.lit(gene))
                .when(
                    pl.col("Metadata_Gene") != gene
                )  # target pool excluding held out gene
                .then(pl.lit(f"{phenotype}_gene_pooled"))
                .alias("_labeled_references")
            )

            if shuffle_flag:
                # shuffle the on and off signatures and shuffle
                on_sig, off_sig = shuffle_signatures(
                    on_sig=on_sig,
                    off_sig=off_sig,
                    all_features=morph_feats,
                    seed=iter_seed,
                )
                test_df = shuffle_feature_profiles(
                    profiles=test_df,
                    feature_cols=morph_feats,
                    method="column",
                    seed=iter_seed,
                )

            # if no signature was found, skip the gene
            if len(on_sig) == 0 or len(off_sig) == 0:
                print(f"skipping {gene}")
                continue

            # score the held out gene using the generated signatures
            associated_gene_score = calculate_buscar_scores(
                profiles=test_df,
                meta_cols=feature_space_configs["metadata-features"],
                on_morphology_signature=on_sig,
                off_morphology_signature=off_sig,
                ref_state="negcon",
                target=f"{phenotype}_gene_pooled",
                perturbation_col="_labeled_references",
                n_threads=1,
                seed=iter_seed,
            )

            # calculate the proportion of cells that make up this phenotype with the
            # current gene perturbation
            try:
                cell_state_proportion = cell_proportion_df.filter(
                    (pl.col("Metadata_Gene") == gene)
                    & (pl.col("Mitocheck_Phenotypic_Class") == phenotype)
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
            )

            # add column indicating that the gene is associated with the cell phenotype
            associated_gene_score = associated_gene_score.with_columns(
                pl.lit(True).alias("associated_with_phenotype")
            )

            # store on and off signatures
            on_off_sigs.append((iter_idx, phenotype, on_sig, off_sig))
            associated_gene_scores.append(associated_gene_score)

        associated_gene_scores = pl.concat(associated_gene_scores)

        # Step 2: rank genes that are not associated with this cell state

        # create on and off sigs with pooled poscon cell state
        on_sig, off_sig, _ = get_signatures(
            ref_profiles=negcon_profiles,
            target_profiles=target_df,
            morph_feats=morph_feats,
            test_method="ks_test",
            p_threshold=0.05,
            seed=iter_seed,
        )
        if len(on_sig) == 0 or len(off_sig) == 0:
            raise ValueError(
                f"No signature found for {phenotype}, skipping ranking of non-associated genes"
            )

        # create a test dataframe that includes the negcon profiles,
        # the profiles of genes that are not associated with this cell state,
        # and the target_df (which contains the profiles of the genes associated with this cell state)
        test_non_associated_df = pl.concat(
            [
                negcon_profiles,
                profiles.filter(
                    pl.col("Metadata_Gene").is_in(genes_not_associated_with_state)
                ),
                target_df,
            ]
        )
        if shuffle_flag:
            on_sig, off_sig = shuffle_signatures(
                on_sig=on_sig, off_sig=off_sig, all_features=morph_feats, seed=iter_seed
            )
            test_non_associated_df = shuffle_feature_profiles(
                profiles=test_non_associated_df,
                feature_cols=morph_feats,
                method="column",
                seed=iter_seed,
            )

        # label the test_non_associated_df to differentiate between negcon,
        # the genes not associated with the cell state, and the target phenotype pool
        test_non_associated_df = test_non_associated_df.with_columns(
            pl.when(pl.col("Metadata_Gene") == "negcon")
            .then(pl.lit("negcon"))
            .when(pl.col("Metadata_Gene").is_in(genes_associated_with_state))
            .then(pl.lit(phenotype))  # label pooled target as phenotypic state
            .otherwise(pl.col("Metadata_Gene"))  # keep non-associated as gene names
            .alias("_labeled_references")
        )

        # rank all treatments not associated with this cell state using pooled signatures
        not_associated_gene_scores = calculate_buscar_scores(
            profiles=test_non_associated_df,
            meta_cols=meta_feats,
            on_morphology_signature=on_sig,
            off_morphology_signature=off_sig,
            ref_state="negcon",
            target=phenotype,
            perturbation_col="_labeled_references",
            n_threads=1,
            seed=iter_seed,
        )

        # remove scores of genes that are associated with the cell state
        not_associated_gene_scores = not_associated_gene_scores.filter(
            pl.col("perturbation").is_in(genes_not_associated_with_state)
        )

        # add column indicating that the gene is not associated with the cell phenotype
        not_associated_gene_scores = not_associated_gene_scores.with_columns(
            pl.lit(False).alias("associated_with_phenotype")
        )

        # add proportion of cells; if a gene has no cells in this state, assign 0
        not_associated_gene_scores = not_associated_gene_scores.join(
            cell_proportion_df.select(
                ["Metadata_Gene", "Mitocheck_Phenotypic_Class", "proportion"]
            ),
            left_on=["perturbation", "target"],
            right_on=["Metadata_Gene", "Mitocheck_Phenotypic_Class"],
            how="left",
        ).with_columns(pl.col("proportion").fill_null(0.0))

        # enforce matching schema before vertical concat
        expected_cols = [
            "target",
            "perturbation",
            "on_buscar_scores",
            "off_buscar_scores",
            "is_reference_distance",
            "proportion",
            "associated_with_phenotype",
        ]
        associated_gene_scores = associated_gene_scores.select(expected_cols)
        not_associated_gene_scores = not_associated_gene_scores.select(expected_cols)

        # final result for this cell state
        state_scores = pl.concat(
            [associated_gene_scores, not_associated_gene_scores], how="vertical"
        ).with_columns(pl.lit(iter_idx).alias("iteration"))
        results_df.append(state_scores)

    # collect one dataframe per iteration
    iter_results_df = pl.concat(results_df)

    # keep only real treatment scores (drop reference-distance and placeholder rows)
    iter_results_df = iter_results_df.filter(
        (~pl.col("is_reference_distance").fill_null(False))
        & pl.col("on_buscar_scores").is_not_null()
        & pl.col("off_buscar_scores").is_not_null()
    )

    all_iter_results.append(iter_results_df)

# step 3: store results from all iterations
results_df = pl.concat(all_iter_results)

# ensure one score row per iteration-target-treatment pair
results_df = results_df.unique(
    subset=["iteration", "target", "perturbation"], keep="first"
)

output_filename = f"{'shuffled' if shuffle_flag else 'original'}_mitocheck_moa_analysis_results_iter{n_iterations}.parquet"
results_df.write_parquet(logo_analysis_output / output_filename)


# In[12]:


def summarize_iteration_scores(results: pl.DataFrame) -> pl.DataFrame:
    """Aggregate target-perturbation scores across iterations."""
    return (
        results.group_by(["target", "perturbation", "associated_with_phenotype"])
        .agg(
            [
                pl.col("on_buscar_scores").mean().alias("on_buscar_scores_mean"),
                pl.col("on_buscar_scores").std().alias("on_buscar_scores_std"),
                pl.col("off_buscar_scores").mean().alias("off_buscar_scores_mean"),
                pl.col("off_buscar_scores").std().alias("off_buscar_scores_std"),
                pl.col("proportion").mean().alias("proportion_mean"),
                pl.col("iteration").n_unique().alias("n_iterations_seen"),
            ]
        )
        .sort(
            ["target", "associated_with_phenotype", "on_buscar_scores_mean"],
            descending=[False, True, False],
        )
    )


mean_scores_df = summarize_iteration_scores(results_df)

# save the summarized scores dataframe
output_summary_filename = f"{'shuffled' if shuffle_flag else 'original'}_mitocheck_moa_analysis_summary_iter_{n_iterations}.parquet"
mean_scores_df.write_parquet(logo_analysis_output / output_summary_filename)


# In[13]:


mean_scores_df
