#!/usr/bin/env python

# ## Replicate Consistency
#
# Here, we will assess whether buscar can be used in scenarios where treatment replicates are distributed across multiple plates. This means that replicates are not located within the same plate, but instead are spread across different plates, possibly occupying the same well positions.
#
# Since we are developing a metric at the single-cell level, it is important to understand how measuring treatments across different plates affects the scoring produced by buscar.

# In[1]:


import json
import pathlib
import sys

import numpy as np
import polars as pl
from tqdm.auto import tqdm

sys.path.append("../../")
from buscar.metrics import measure_phenotypic_activity
from buscar.signatures import get_signatures
from utils.io_utils import load_sc_profiles

# Setting input and out paths

# In[2]:


pert_type = "crispr"


# In[3]:


data_dir = pathlib.Path("../0.download-data/data/sc-profiles/").resolve(strict=True)

# setting results dir
results_dir = pathlib.Path("./results")
results_dir.mkdir(exist_ok=True)


# experimental metadata
cpjump1_experimental_data_path = (
    data_dir / f"cpjump1/cpjump1_{pert_type}_experimental-metadata.csv"
).resolve(strict=True)

# negcon and poscon profiles
cpjump1_negcon_profile_path = list(
    (data_dir / "cpjump1/negcon").resolve(strict=True).glob("*.parquet")
)

# shared feature set
cpjump1_shared_features_path = (
    data_dir / "cpjump1/feature_selected_sc_qc_features.json"
).resolve(strict=True)

# generate output dir
outdir = (results_dir / "replicate_analysis").resolve()
outdir.mkdir(exist_ok=True)


# In[4]:


cpjump1_experimental_data = pl.read_csv(cpjump1_experimental_data_path)
meta_features, morph_features, cpjump1_df = load_sc_profiles(
    data_name="cpjump1", datatype=pert_type
)

# change null values in Metadata_negcon_control_type to "na"
cpjump1_df = cpjump1_df.with_columns(
    pl.when(pl.col("Metadata_negcon_control_type").is_null())
    .then(pl.lit("none"))
    .otherwise(pl.col("Metadata_negcon_control_type"))
    .alias("Metadata_negcon_control_type")
)

# replace metadata_control_type null values with "negcon" when Metadata_negcon_control_type is not null
cpjump1_df = cpjump1_df.with_columns(
    pl.when(
        (pl.col("Metadata_control_type").is_null())
        & (pl.col("Metadata_negcon_control_type") != "none")
    )
    .then(pl.lit("negcon"))
    .otherwise(pl.col("Metadata_control_type"))
    .alias("Metadata_control_type")
)

#  next raplace null values in Metadata_control_type with "trt"
cpjump1_df = cpjump1_df.with_columns(
    pl.when(pl.col("Metadata_control_type").is_null())
    .then(pl.lit("trt"))
    .otherwise(pl.col("Metadata_control_type"))
    .alias("Metadata_control_type")
)

# convert nulls in metdata_gene to "negcon", if they are nulls
cpjump1_df = cpjump1_df.with_columns(
    pl.when(pl.col("Metadata_gene").is_null())
    .then(pl.lit("negcon"))
    .otherwise(pl.col("Metadata_gene"))
    .alias("Metadata_gene")
)


# next it to filter out cells where Metadata_negcon_control_type is ONE_INTERGENIC_SITE
cpjump1_df = cpjump1_df.filter(
    pl.col("Metadata_negcon_control_type") != "ONE_INTERGENIC_SITE"
)

print(f"Shape: {cpjump1_df.shape}")
print(
    f"Metadata columns: {len(meta_features)}, Morphological features: {len(morph_features)}"
)


# In[5]:


# Split the dataset by cell type and treatment duration
# Filter U2OS cells (all records)
cpjump1_u2os_exp_metadata = cpjump1_experimental_data.filter(
    pl.col("Cell_type") == "U2OS"
)

# Filter A549 cells with density of 100 for consistency
cpjump1_a549_exp_metadata = cpjump1_experimental_data.filter(
    pl.col("Cell_type") == "A549"
)

# Extract plate identifiers for each cell type
u2os_plates = cpjump1_u2os_exp_metadata["Assay_Plate_Barcode"].unique().to_list()
a549_plates = cpjump1_a549_exp_metadata["Assay_Plate_Barcode"].unique().to_list()


# Display the extracted plates for verification
print(f"U2OS plates: {u2os_plates}")
print(f"A549 plates: {a549_plates}")


# In[6]:


u2os_df = cpjump1_df.filter(pl.col("Metadata_Plate").is_in(u2os_plates))
a549_df = cpjump1_df.filter(pl.col("Metadata_Plate").is_in(a549_plates))

# summary of the filtered dataframes
print(f"U2OS DataFrame shape: {u2os_df.shape}")
print(f"A549 DataFrame shape: {a549_df.shape}")


# In[8]:


cpjump1_df["Metadata_gene"].n_unique()


# In[ ]:


# groups = ["Metadata_Plate", "Metadata_Well"]

# u2os_sampled_df = u2os_df.filter(
#     pl.int_range(0, pl.len()).shuffle(seed=42).over(groups) < (pl.len() * 0.1).over(groups)
# )
# u2os_sampled_df


# ## Replicate Analysis
#
# Here we assess whether buscar can reliably score treatments when replicates are spread across multiple plates (cross-plate replicates), rather than co-located on a single plate.
#
# aproach:
# For each treatment and iteration:
# 1. Select a **reference plate** — use its negcon and treatment cells to generate on/off signatures via buscar.
# 2. Combine the reference plate's negcon cells with all treatment cells from **all plates**, labeling each cell with a `_plate_label` column.
# 3. Run `measure_phenotypic_activity` using the reference plate as the `target_state`, so on-scores are normalized to ~1.0 relative to the reference plate.
# 4. Scores from all other plates are interpreted relative to the reference: a value near **1.0** means the compared plate replicates the reference phenotype consistently.
#
#
# - **on-score ~1.0**: the compared plate's treatment cells occupy a similar morphological space to the reference plate's treatment cells in the on-feature subspace — indicating consistent replication.
# - **off-score ~0.0**: features expected to be unaffected by the treatment remain stable across plates — indicating low plate-to-plate technical variation.
#
# Note that a high off-score does not conclusively distinguish between technical batch effects, weak compound labeling, or genuine multi-target activity. It signals that something changed in the off-feature space, but the source of that change requires further investigation.
#
# ### Controls
#
# The analysis is run twice — once with real signatures and once with **shuffled signatures** (on/off feature assignments randomized). The shuffled condition serves as a negative control: scores should regress toward chance, confirming that observed consistency in the real condition is driven by the treatment's phenotypic signal rather than experimental artifacts.
#
# This procedure is repeated for every plate as the reference and across `n_iterations` iterations with different random seeds.

# In[ ]:


u2os_plate_name_dict = {
    original_name: f"plate_{idx + 1}"
    for idx, original_name in enumerate(u2os_df["Metadata_Plate"].unique().to_list())
}
a549_plate_name_dict = {
    original_name: f"plate_{idx + 1}"
    for idx, original_name in enumerate(a549_df["Metadata_Plate"].unique().to_list())
}


# In[ ]:


def _get_plate_treatment_cell_counts(
    profiles: pl.DataFrame,
    plate_col: str,
    trt_col: str,
    unique_plates: list,
    treatment: str,
    plate_name_dict: dict,
) -> dict:
    """
    Compute the number of cells for each plate and treatment.

    Args:
        profiles (pl.DataFrame): The dataframe containing all profiles.
        plate_col (str): The column name for plate IDs.
        trt_col (str): The column name for treatment names.
        unique_plates (list): List of unique plate IDs.
        treatment (str): The treatment name to count.
        plate_name_dict (dict): Mapping from plate ID to human-readable name.

    Returns:
        dict: Mapping from plate name to cell count for the given treatment.
    """
    return {
        plate_name_dict[p]: profiles.filter(
            (pl.col(plate_col) == p) & (pl.col(trt_col) == treatment)
        ).height
        for p in unique_plates
    }


def _generate_null_result(
    cell_type=None,
    negcon=None,
    treatment=None,
    ref_plate_rep=None,
    compared_plate_rep=None,
    on_score=None,
    off_score=None,
    n_negcon_cells=None,
    n_ref_trt_cells=None,
    n_compared_trt_cells=None,
    iteration=None,
    random_trt_comparison=None,
    ref_treatment=None,
    compared_treatment=None,
):
    return {
        "cell_type": cell_type,
        "negcon": negcon,
        "treatment": treatment,
        "ref_treatment": ref_treatment,
        "compared_treatment": compared_treatment,
        "ref_plate_rep": ref_plate_rep,
        "compared_plate_rep": compared_plate_rep,
        "on_score": on_score,
        "off_score": off_score,
        "n_negcon_cells": n_negcon_cells,
        "n_ref_trt_cells": n_ref_trt_cells,
        "n_compared_trt_cells": n_compared_trt_cells,
        "iteration": iteration,
        "random_trts": random_trt_comparison,
    }


def run_replicate_consistency_analysis(
    profiles: pl.DataFrame,
    plate_name_dict: dict[str, str],
    meta_cols: list[str],
    morph_feats: list[str],
    treatment_col: str,
    cell_type: str,
    negcon_subsample_frac: float = 0.01,
    n_iterations: int = 5,
    save_dir: pathlib.Path | None = None,
    random_trt_comparison: bool = False,
    checked_trts: list[str] | None = None,
    trt_to_ignore: list[str] | None = None,
) -> pl.DataFrame:
    """
    Assess how consistently buscar scores the same treatment phenotype across different plates.


    For each treatment, every plate takes a turn as the "reference plate":

    1. Negative control and treatment cells from the reference plate are used
    to build on/off signatures -- these describe which morphological features
    are most changed (on-features) and least changed (off-features) by the treatment.
    2. Those signatures are then used to score treatment cells from every other
    plate.
    3. Each comparison yields two scores:

       - on-score: how similar the other plate's treatment cells are to the
         reference phenotype. A value near 1.0 means the treatment produces a
         consistent morphological effect across plates.
       - off-score: how stable the features expected to be unaffected remain
         across plates. A value near 0.0 indicates low plate-to-plate technical
         variation in those features.

    This loop is repeated across ``n_iterations`` random seeds to account for
    sampling variability in the negative control.

    When ``random_trt_comparison=True``, each reference plate is scored against
    a randomly selected treatment from another plate instead of the same
    treatment. This breaks the treatment-identity link and serves as a baseline.

    If ``save_dir`` is provided, each result is incrementally appended as a JSON line
    (JSONL) to a file named ``{cell_type}_original_replicate-tracking.jsonl`` (normal
    mode) or ``{cell_type}_shuffled_replicate-tracking.jsonl`` (random_trt_comparison).

    Parameters
    ----------
    profiles : pl.DataFrame
        Single-cell morphological profiles for all plates, including both metadata
        and feature columns.
    plate_name_dict : dict[str, str]
        Mapping from raw plate barcode to a human-readable label
        (e.g., ``{"BR00116991": "plate_1"}``).
    meta_cols : list[str]
        Column names that contain metadata (not morphological features).
    morph_feats : list[str]
        Column names that contain morphological features used for signature
        generation and scoring.
    cell_type : str
        Label for the cell line being analyzed (e.g., ``'U2OS'``, ``'A549'``).
        Used for labeling output files and result rows.
    negcon_subsample_frac : float, optional
        Fraction of negative control cells to sample per reference plate. Lower
        values speed up computation. Default is ``0.01``.
    n_iterations : int, optional
        Number of times to repeat the analysis with different random seeds. More
        iterations yield more stable average scores. Default is ``5``.
    save_dir : pathlib.Path or None, optional
        Directory to write incremental results as JSONL. If ``None``, results are
        only returned as a DataFrame.
    random_trt_comparison : bool, optional
        If ``True``, compare each reference plate against a randomly chosen
        treatment as a negative control. Default is ``False``.
    checked_trts : list[str] or None, optional
        If provided, only these treatments will be analyzed. If ``None``, all
        treatments in the dataset will be included. Default is ``None``.
    trt_to_ignore : list[str] or None, optional
        If provided, these treatments will be skipped during analysis. Default is ``None``.
    Returns
    -------
    pl.DataFrame
        One row per (treatment, reference plate, compared plate, iteration) with
        columns for ``on_score``, ``off_score``, cell counts, and metadata labels.
    """
    # selecting treatments to analyze based on checked_trts
    if checked_trts is not None:
        treatments = [
            t for t in profiles[treatment_col].unique().to_list() if t in checked_trts
        ]
    else:
        print("DEBUGG: filtering to only unprocessed treatments")
        treatments = profiles[treatment_col].unique().to_list()
        print(f"DEBUGG: treatments to analyze ({treatment_col}): {len(treatments)}")

    unique_plates = list(plate_name_dict.keys())
    meta_cols_with_label = meta_cols + ["_plate_label"]

    all_scores = []
    treatment_pbar = tqdm(treatments, desc="Treatments", unit="treatment")
    for treatment in treatment_pbar:
        treatment_pbar.set_postfix(treatment=treatment)

        # debugging message
        print(f"DEBUGG: Analyzing treatment: {treatment} (cell type: {cell_type})")
        if treatment is not None and treatment in trt_to_ignore:
            print(
                f"DEBUGG: Skipping treatment {treatment} as it is in the ignore list."
            )
            continue

        for iteration in tqdm(
            range(n_iterations),
            desc=f"  [{treatment}] Iterations",
            unit="iter",
            leave=False,
        ):
            iter_id = iteration + 1

            # Iterate over all plates, treating each as the reference plate
            for ref_plate in tqdm(
                unique_plates,
                desc=f"  [{treatment}] iter={iter_id} Ref plates",
                unit="plate",
                leave=False,
            ):
                ref_plate_name = plate_name_dict[ref_plate]

                # Select negative control and treatment cells from the reference plate
                ref_negcon = profiles.filter(
                    (pl.col("Metadata_Plate") == ref_plate)
                    & (pl.col("Metadata_control_type") == "negcon")
                ).sample(
                    fraction=negcon_subsample_frac, seed=iter_id, with_replacement=False
                )
                ref_trt = profiles.filter(
                    (pl.col("Metadata_Plate") == ref_plate)
                    & (pl.col(treatment_col) == treatment)
                )

                # Skip if either group is empty
                if ref_trt.height == 0 or ref_negcon.height == 0:
                    tqdm.write(
                        f"  [SKIP] treatment={treatment} | ref={ref_plate_name} | "
                        f"iter={iter_id} — no cells found (negcon={ref_negcon.height}, "
                        f"trt={ref_trt.height})"
                    )

                    all_scores.append(
                        _generate_null_result(
                            cell_type=cell_type,
                            treatment=treatment,
                            ref_plate_rep=ref_plate_name,
                            n_negcon_cells=ref_negcon.height,
                            iteration=iter_id,
                            random_trt_comparison=random_trt_comparison,
                            ref_treatment=treatment,
                        )
                    )
                    continue

                plate_n_cells = _get_plate_treatment_cell_counts(
                    profiles=profiles,
                    plate_col="Metadata_Plate",
                    trt_col=treatment_col,
                    unique_plates=unique_plates,
                    treatment=treatment,
                    plate_name_dict=plate_name_dict,
                )

                # Generate on and off signatures from the reference plate
                on_sig, off_sig, _ = get_signatures(
                    ref_profiles=ref_negcon.select(morph_feats),
                    exp_profiles=ref_trt.select(morph_feats),
                    morph_feats=morph_feats,
                    seed=iter_id,
                )

                # Maps each compared plate name to the treatment used for that plate
                # - not shuffled: all plates use the same treatment as the reference
                # - shuffled: each plate is assigned a different random treatment
                plate_trt_map: dict[str, str] = {}

                # Select a single treatment from each plate (except the reference)
                if random_trt_comparison:
                    # set seed for this radnomly seletion
                    np.random.seed(iter_id)

                    # List of plates to compare (excluding the reference plate)
                    plates_to_compare = [p for p in unique_plates if p != ref_plate]
                    # Select random treatments for each non-reference plate.
                    # Exclude the reference treatment from the sampling pool so no
                    # compared plate is assigned the same treatment as the reference.
                    # This fully breaks the treatment-identity link in shuffled mode.
                    random_trts = list(
                        np.random.choice(
                            [t for t in treatments if t != treatment],
                            size=len(plates_to_compare),
                            replace=True,
                        )
                    )

                    # randomly select a single treatment from each plate
                    sel_plates_trt_to_compare = []
                    for idx, sel_plate in enumerate(plates_to_compare):
                        plate_trt_map[plate_name_dict[sel_plate]] = random_trts[idx]
                        sel_plates_trt_to_compare.append(
                            profiles.filter(
                                (pl.col("Metadata_Plate") == sel_plate)
                                & (pl.col(treatment_col) == random_trts[idx])
                            )
                        )
                    sel_plates_trt_to_compare = pl.concat(sel_plates_trt_to_compare)

                    # Build evaluation set: negcon from ref plate + treatment cells from chosen plate
                    plate_to_test = pl.concat(
                        [ref_negcon, ref_trt, sel_plates_trt_to_compare]
                    )

                else:
                    # Default: compare all plates' treatment cells to the reference
                    # All plates use the same treatment as the reference
                    for p in unique_plates:
                        plate_trt_map[plate_name_dict[p]] = treatment

                    plate_to_test = pl.concat(
                        [
                            ref_negcon,
                            profiles.filter(pl.col(treatment_col) == treatment),
                        ]
                    )

                # return plate_to_test

                # Add "_plate_label" column to distinguish reference, negcon, and other plates
                # - negcon cells → "negcon" (used as ref_state)
                # - ref plate treatment cells → ref_plate_name (used as target_state)
                # - other plate treatment cells → their mapped plate name
                combined = plate_to_test.with_columns(
                    pl.when(pl.col("Metadata_control_type") == "negcon")
                    .then(pl.lit("negcon"))  # negcon cells are labeled as "negcon"
                    .when(pl.col("Metadata_Plate") == ref_plate)
                    .then(
                        pl.lit(ref_plate_name)
                    )  # ref plate treatment cells are labeled as ref_plate_name
                    .otherwise(  # other plate treatment cells are labeled as their mapped plate name
                        pl.col("Metadata_Plate").map_elements(
                            lambda x: plate_name_dict[x], return_dtype=pl.String
                        )
                    )
                    .alias("_plate_label")  # all under the _plate_label_col
                )

                # Score phenotypic activity: on-scores are normalized to the reference plate
                scores_df = measure_phenotypic_activity(
                    profiles=combined,
                    meta_cols=meta_cols_with_label,
                    on_signature=on_sig,
                    off_signature=off_sig,
                    ref_state="negcon",
                    target_state=ref_plate_name,
                    treatment_col="_plate_label",
                    state_col="_plate_label",
                    seed=iter_id,
                    n_threads=1,
                )

                # Collect results with cell counts for each comparison
                for row in scores_df.iter_rows(named=True):
                    compared_plate_name = row["treatment"]
                    # ref treatment is always the treatment used for the reference plate
                    # compared treatment is the same as ref if not shuffled,
                    # or the randomly assigned treatment for that plate if shuffled
                    ref_treatment = treatment
                    compared_treatment = plate_trt_map.get(
                        compared_plate_name, treatment
                    )
                    result = {
                        "cell_type": cell_type,
                        "negcon": "negcon",
                        "treatment": treatment,
                        "ref_treatment": ref_treatment,
                        "compared_treatment": compared_treatment,
                        "ref_plate_rep": ref_plate_name,
                        "compared_plate_rep": compared_plate_name,
                        "on_score": row["on_score"],
                        "off_score": row["off_score"],
                        "n_negcon_cells": ref_negcon.height,
                        "n_ref_trt_cells": ref_trt.height,
                        "n_compared_trt_cells": plate_n_cells.get(
                            compared_plate_name, 0
                        ),
                        "iteration": iter_id,
                        "random_trts": random_trt_comparison,
                    }
                    all_scores.append(result)

                    # Optionally save results to JSONL if save_dir is provided
                    if save_dir is not None:
                        save_path = (
                            save_dir
                            / f"{cell_type}_{'shuffled' if random_trt_comparison else 'original'}_crispr-replicate-tracking.jsonl"
                        ).resolve()
                        with open(save_path, "a") as f:
                            f.write(json.dumps(result) + "\n")

    return pl.DataFrame(all_scores)


# Applying replicate analysis with both shuffled and not shuffle data with both A549 and U2OS cells

# In[ ]:


# do a check. if the file exists load it and set unporcessed trt if not set unprocessed trt toNone
# if (results_dir / "replicate_analysis/U2OS_shuffled_replicate-tracking.jsonl").exists():
#     df = pl.read_ndjson(
#         "./results/replicate_analysis/U2OS_shuffled_replicate-tracking.jsonl"
#     )
#     completed_trt = df["ref_treatment"].unique().to_list()

#     # get all the trt from the original dataset
#     all_trt = u2os_df["Metadata_pert_iname"].unique().to_list()

#     # now find the treatments that wer not processed in the original analysis
#     unprocessed_trt = [trt for trt in all_trt if trt not in completed_trt]
# else:
#     unprocessed_trt = None

# unprocessed_trt = None


# In[ ]:


n_iterations = 10


# In[ ]:


# run analysis
print(
    f"DEBUGG: Running replicate consistency analysis for U2OS cells (n_iterations={10})..."
)
u2os_results_df = run_replicate_consistency_analysis(
    profiles=u2os_df,
    plate_name_dict=u2os_plate_name_dict,
    meta_cols=meta_features,
    morph_feats=morph_features,
    treatment_col="Metadata_gene",
    cell_type="U2OS",
    n_iterations=10,
    save_dir=outdir,
    negcon_subsample_frac=0.02,
    random_trt_comparison=False,
    trt_to_ignore=["negcon"],
)
print(
    f"DEBUGG: Running replicate consistency analysis for U2OS cells (n_iterations={10})..."
)

u2os_rnd_trt_results_df = run_replicate_consistency_analysis(
    profiles=u2os_df,
    plate_name_dict=u2os_plate_name_dict,
    meta_cols=meta_features,
    morph_feats=morph_features,
    treatment_col="Metadata_gene",
    cell_type="U2OS",
    n_iterations=10,
    save_dir=outdir,
    negcon_subsample_frac=0.02,
    random_trt_comparison=True,
    trt_to_ignore=["negcon"],
)


# save results as parquet
u2os_results_df.write_parquet(
    outdir / "u2os_crispr_replicate_consistency_results.parquet"
)
u2os_rnd_trt_results_df.write_parquet(
    outdir / "u2os_random_crispr_replicate_consistency_results.parquet"
)


# In[ ]:


print(
    f"DEBUGG: Running replicate consistency analysis for U2OS cells (n_iterations={n_iterations})..."
)
a549_results_df = run_replicate_consistency_analysis(
    profiles=a549_df,
    plate_name_dict=a549_plate_name_dict,
    meta_cols=meta_features,
    morph_feats=morph_features,
    treatment_col="Metadata_gene",
    cell_type="A549",
    n_iterations=10,
    save_dir=outdir,
    negcon_subsample_frac=0.02,
    random_trt_comparison=False,
    trt_to_ignore=["negcon"],
)
print(
    f"DEBUGG: Running replicate consistency analysis for A549 cells (n_iterations={n_iterations})..."
)

a549_rnd_trt_results_df = run_replicate_consistency_analysis(
    profiles=a549_df,
    plate_name_dict=a549_plate_name_dict,
    meta_cols=meta_features,
    morph_feats=morph_features,
    treatment_col="Metadata_gene",
    cell_type="A549",
    n_iterations=10,
    save_dir=outdir,
    negcon_subsample_frac=0.02,
    random_trt_comparison=True,
    trt_to_ignore=["negcon"],
)


# save results as parquet
a549_results_df.write_parquet(
    outdir / "a549_crispr_replicate_consistency_results.parquet"
)
a549_rnd_trt_results_df.write_parquet(
    outdir / "a549_random_crispr_replicate_consistency_results.parquet"
)
