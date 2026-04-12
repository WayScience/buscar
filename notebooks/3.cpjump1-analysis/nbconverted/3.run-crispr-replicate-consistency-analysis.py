#!/usr/bin/env python

# ## Replicate Consistency
#
# Here, we will assess whether buscar can be used in scenarios where treatment replicates are distributed across multiple plates. This means that replicates are not located within the same plate, but instead are spread across different plates, possibly occupying the same well positions.
#
# Since we are developing a metric at the single-cell level, it is important to understand how measuring treatments across different plates affects the scoring produced by buscar.

# In[ ]:


import json
import pathlib
import sys

import numpy as np
import polars as pl
from tqdm.auto import tqdm

sys.path.append("../../")
from buscar.metrics import calculate_buscar_scores
from buscar.signatures import get_signatures

from utils.io_utils import load_sc_profiles

# Setting input and out paths

# In[ ]:


pert_type = "crispr"


# In[ ]:


data_dir = pathlib.Path("../0.download-data/data/sc-profiles/").resolve(strict=True)

# setting results dir
results_dir = pathlib.Path("./results")
results_dir.mkdir(exist_ok=True)


# experimental metadata
cpjump1_experimental_data_path = (
    data_dir / f"cpjump1/cpjump1_{pert_type}_experimental-metadata.csv"
).resolve(strict=True)

# shared feature set
cpjump1_shared_features_path = (
    data_dir / "cpjump1/feature_selected_sc_qc_features.json"
).resolve(strict=True)

# generate output dir
outdir = (results_dir / "replicate_analysis").resolve()
outdir.mkdir(exist_ok=True)


# In[ ]:


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


# In[ ]:


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


# In[ ]:


u2os_df = cpjump1_df.filter(pl.col("Metadata_Plate").is_in(u2os_plates))
a549_df = cpjump1_df.filter(pl.col("Metadata_Plate").is_in(a549_plates))

# summary of the filtered dataframes
print(f"U2OS DataFrame shape: {u2os_df.shape}")
print(f"A549 DataFrame shape: {a549_df.shape}")


# In[ ]:


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


def _get_plate_perturbation_cell_counts(
    profiles: pl.DataFrame,
    plate_col: str,
    perturbation_col: str,
    unique_plates: list,
    perturbation: str,
    plate_name_dict: dict,
) -> dict:
    """
    Compute the number of cells for each plate and perturbation.

    Args:
        profiles (pl.DataFrame): The dataframe containing all profiles.
        plate_col (str): The column name for plate IDs.
        perturbation_col (str): The column name for perturbation names.
        unique_plates (list): List of unique plate IDs.
        perturbation (str): The perturbation name to count.
        plate_name_dict (dict): Mapping from plate ID to human-readable name.

    Returns:
        dict: Mapping from plate name to cell count for the given perturbation.
    """
    return {
        plate_name_dict[p]: profiles.filter(
            (pl.col(plate_col) == p) & (pl.col(perturbation_col) == perturbation)
        ).height
        for p in unique_plates
    }


def _generate_null_result(
    cell_type=None,
    negcon=None,
    perturbation=None,
    ref_plate_rep=None,
    compared_plate_rep=None,
    on_score=None,
    off_score=None,
    n_negcon_cells=None,
    n_ref_perturbation_cells=None,
    n_compared_perturbation_cells=None,
    iteration=None,
    random_perturbation_comparison=None,
    ref_perturbation=None,
    compared_perturbation=None,
):
    return {
        "cell_type": cell_type,
        "negcon": negcon,
        "perturbation": perturbation,
        "ref_perturbation": ref_perturbation,
        "compared_perturbation": compared_perturbation,
        "ref_plate_rep": ref_plate_rep,
        "compared_plate_rep": compared_plate_rep,
        "on_score": on_score,
        "off_score": off_score,
        "n_negcon_cells": n_negcon_cells,
        "n_ref_perturbation_cells": n_ref_perturbation_cells,
        "n_compared_perturbation_cells": n_compared_perturbation_cells,
        "iteration": iteration,
        "random_perturbations": random_perturbation_comparison,
    }


def run_replicate_consistency_analysis(
    profiles: pl.DataFrame,
    plate_name_dict: dict[str, str],
    meta_cols: list[str],
    morph_feats: list[str],
    perturbation_col: str,
    cell_type: str,
    negcon_subsample_frac: float = 0.01,
    n_iterations: int = 5,
    save_dir: pathlib.Path | None = None,
    random_perturbation_comparison: bool = False,
    checked_perturbations: list[str] | None = None,
    perturbation_to_ignore: list[str] | None = None,
) -> pl.DataFrame:
    """
    Assess how consistently buscar scores the same perturbation phenotype across different plates.


    For each perturbation, every plate takes a turn as the "reference plate":

    1. Negative control and perturbation cells from the reference plate are used
    to build on/off signatures -- these describe which morphological features
    are most changed (on-features) and least changed (off-features) by the perturbation.
    2. Those signatures are then used to score perturbation cells from every other
    plate.
    3. Each comparison yields two scores:

       - on-score: how similar the other plate's perturbation cells are to the
         reference phenotype. A value near 1.0 means the perturbation produces a
         consistent morphological effect across plates.
       - off-score: how stable the features expected to be unaffected remain
         across plates. A value near 0.0 indicates low plate-to-plate technical
         variation in those features.

    This loop is repeated across ``n_iterations`` random seeds to account for
    sampling variability in the negative control.

    When ``random_perturbation_comparison=True``, each reference plate is scored against
    a randomly selected perturbation from another plate instead of the same
    perturbation. This breaks the perturbation-identity link and serves as a baseline.

    If ``save_dir`` is provided, each result is incrementally appended as a JSON line
    (JSONL) to a file named ``{cell_type}_original_replicate-tracking.jsonl`` (normal
    mode) or ``{cell_type}_shuffled_replicate-tracking.jsonl`` (random_perturbation_comparison).

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
    perturbation_col : str
        The column name for perturbation identifiers.
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
    random_perturbation_comparison : bool, optional
        If ``True``, compare each reference plate against a randomly chosen
        perturbation as a negative control. Default is ``False``.
    checked_perturbations : list[str] or None, optional
        If provided, only these perturbations will be analyzed. If ``None``, all
        perturbations in the dataset will be included. Default is ``None``.
    perturbation_to_ignore : list[str] or None, optional
        If provided, these perturbations will be skipped during analysis. Default is ``None``.
    Returns
    -------
    pl.DataFrame
        One row per (perturbation, reference plate, compared plate, iteration) with
        columns for ``on_score``, ``off_score``, cell counts, and metadata labels.
    """
    # selecting perturbations to analyze based on checked_perturbations
    if checked_perturbations is not None:
        perturbations = [
            t
            for t in profiles[perturbation_col].unique().to_list()
            if t in checked_perturbations
        ]
    else:
        print("DEBUGG: filtering to only unprocessed perturbations")
        perturbations = profiles[perturbation_col].unique().to_list()
        print(
            f"DEBUGG: perturbations to analyze ({perturbation_col}): {len(perturbations)}"
        )

    unique_plates = list(plate_name_dict.keys())
    meta_cols_with_label = meta_cols + ["_plate_label"]

    all_scores = []
    perturbation_pbar = tqdm(perturbations, desc="Perturbations", unit="perturbation")
    for perturbation in perturbation_pbar:
        perturbation_pbar.set_postfix(perturbation=perturbation)

        # debugging message
        print(
            f"DEBUGG: Analyzing perturbation: {perturbation} (cell type: {cell_type})"
        )
        if perturbation is not None and perturbation in perturbation_to_ignore:
            print(
                f"DEBUGG: Skipping perturbation {perturbation} as it is in the ignore list."
            )
            continue

        for iteration in tqdm(
            range(n_iterations),
            desc=f"  [{perturbation}] Iterations",
            unit="iter",
            leave=False,
        ):
            iter_id = iteration + 1

            # Iterate over all plates, treating each as the reference plate
            for ref_plate in tqdm(
                unique_plates,
                desc=f"  [{perturbation}] iter={iter_id} Ref plates",
                unit="plate",
                leave=False,
            ):
                ref_plate_name = plate_name_dict[ref_plate]

                # Select negative control and perturbation cells from the reference plate
                ref_negcon = profiles.filter(
                    (pl.col("Metadata_Plate") == ref_plate)
                    & (pl.col("Metadata_control_type") == "negcon")
                ).sample(
                    fraction=negcon_subsample_frac, seed=iter_id, with_replacement=False
                )
                ref_perturbation_cells = profiles.filter(
                    (pl.col("Metadata_Plate") == ref_plate)
                    & (pl.col(perturbation_col) == perturbation)
                )

                # Skip if either group is empty
                if ref_perturbation_cells.height == 0 or ref_negcon.height == 0:
                    tqdm.write(
                        f"  [SKIP] perturbation={perturbation} | ref={ref_plate_name} | "
                        f"iter={iter_id} — no cells found (negcon={ref_negcon.height}, "
                        f"perturbation={ref_perturbation_cells.height})"
                    )

                    all_scores.append(
                        _generate_null_result(
                            cell_type=cell_type,
                            perturbation=perturbation,
                            ref_plate_rep=ref_plate_name,
                            n_negcon_cells=ref_negcon.height,
                            iteration=iter_id,
                            random_perturbation_comparison=random_perturbation_comparison,
                            ref_perturbation=perturbation,
                        )
                    )
                    continue

                plate_perturbation_n_cells = _get_plate_perturbation_cell_counts(
                    profiles=profiles,
                    plate_col="Metadata_Plate",
                    perturbation_col=perturbation_col,
                    unique_plates=unique_plates,
                    perturbation=perturbation,
                    plate_name_dict=plate_name_dict,
                )

                # Generate on and off signatures from the reference plate
                on_sig, off_sig, _ = get_signatures(
                    ref_profiles=ref_negcon.select(morph_feats),
                    target_profiles=ref_perturbation_cells.select(morph_feats),
                    morph_feats=morph_feats,
                    seed=iter_id,
                )

                # Maps each compared plate name to the perturbation used for that plate
                # - not shuffled: all plates use the same perturbation as the reference
                # - shuffled: each plate is assigned a different random perturbation
                plate_perturbation_map: dict[str, str] = {}

                # Select a single perturbation from each plate (except the reference)
                if random_perturbation_comparison:
                    # set seed for this randomly selection
                    np.random.seed(iter_id)

                    # List of plates to compare (excluding the reference plate)
                    plates_to_compare = [p for p in unique_plates if p != ref_plate]
                    # Select random perturbations for each non-reference plate.
                    # Exclude the reference perturbation from the sampling pool so no
                    # compared plate is assigned the same perturbation as the reference.
                    # This fully breaks the perturbation-identity link in shuffled mode.
                    random_perturbations = list(
                        np.random.choice(
                            [t for t in perturbations if t != perturbation],
                            size=len(plates_to_compare),
                            replace=True,
                        )
                    )

                    # randomly select a single perturbation from each plate
                    sel_plates_perturbation_to_compare = []
                    for idx, sel_plate in enumerate(plates_to_compare):
                        plate_perturbation_map[plate_name_dict[sel_plate]] = (
                            random_perturbations[idx]
                        )
                        sel_plates_perturbation_to_compare.append(
                            profiles.filter(
                                (pl.col("Metadata_Plate") == sel_plate)
                                & (
                                    pl.col(perturbation_col)
                                    == random_perturbations[idx]
                                )
                            )
                        )
                    sel_plates_perturbation_to_compare = pl.concat(
                        sel_plates_perturbation_to_compare
                    )

                    # Build evaluation set: negcon from ref plate + perturbation cells from chosen plate
                    plate_to_test = pl.concat(
                        [
                            ref_negcon,
                            ref_perturbation_cells,
                            sel_plates_perturbation_to_compare,
                        ]
                    )

                else:
                    # Default: compare all plates' perturbation cells to the reference
                    # All plates use the same perturbation as the reference
                    for p in unique_plates:
                        plate_perturbation_map[plate_name_dict[p]] = perturbation

                    plate_to_test = pl.concat(
                        [
                            ref_negcon,
                            profiles.filter(pl.col(perturbation_col) == perturbation),
                        ]
                    )

                # Add "_plate_label" column to distinguish reference, negcon, and other plates
                # - negcon cells → "negcon" (used as ref_state)
                # - ref plate perturbation cells → ref_plate_name (used as target_state)
                # - other plate perturbation cells → their mapped plate name
                combined = plate_to_test.with_columns(
                    pl.when(pl.col("Metadata_control_type") == "negcon")
                    .then(pl.lit("negcon"))  # negcon cells are labeled as "negcon"
                    .when(pl.col("Metadata_Plate") == ref_plate)
                    .then(
                        pl.lit(ref_plate_name)
                    )  # ref plate perturbation cells are labeled as ref_plate_name
                    .otherwise(  # other plate perturbation cells are labeled as their mapped plate name
                        pl.col("Metadata_Plate").map_elements(
                            lambda x: plate_name_dict[x], return_dtype=pl.String
                        )
                    )
                    .alias("_plate_label")  # all under the _plate_label_col
                )

                # Score phenotypic activity: on-scores are normalized to the reference plate
                scores_df = calculate_buscar_scores(
                    profiles=combined,
                    meta_cols=meta_cols_with_label,
                    on_morphology_signature=on_sig,
                    off_morphology_signature=off_sig,
                    target=ref_plate_name,
                    ref_state="negcon",
                    perturbation_col="_plate_label",
                    state_col="_plate_label",
                    seed=iter_id,
                    n_threads=1,
                )

                # Collect results with cell counts for each comparison
                for row in scores_df.iter_rows(named=True):
                    # In replicate consistency analysis, "perturbation" from scores_df is the plate we are comparing
                    compared_plate_name = row["perturbation"]

                    # Skip reference distance row and the target itself from results
                    if row["is_reference_distance"] or compared_plate_name == "negcon":
                        continue

                    # ref perturbation is always the perturbation used for the reference plate
                    # compared perturbation is the same as ref if not shuffled,
                    # or the randomly assigned perturbation for that plate if shuffled
                    ref_perturbation = perturbation
                    compared_perturbation = plate_perturbation_map.get(
                        compared_plate_name, perturbation
                    )

                    result = {
                        "cell_type": cell_type,
                        "negcon": "negcon",
                        "perturbation": perturbation,
                        "ref_perturbation": ref_perturbation,
                        "compared_perturbation": compared_perturbation,
                        "ref_plate_rep": ref_plate_name,
                        "compared_plate_rep": compared_plate_name,
                        "on_score": row["on_buscar_scores"],
                        "off_score": row["off_buscar_scores"],
                        "n_negcon_cells": ref_negcon.height,
                        "n_ref_perturbation_cells": ref_perturbation_cells.height,
                        "n_compared_perturbation_cells": plate_perturbation_n_cells.get(
                            compared_plate_name, 0
                        ),
                        "iteration": iter_id,
                        "random_perturbations": random_perturbation_comparison,
                    }
                    all_scores.append(result)

                    # Optionally save results to JSONL if save_dir is provided
                    if save_dir is not None:
                        save_path = (
                            save_dir
                            / f"{cell_type}_{'shuffled' if random_perturbation_comparison else 'original'}_crispr-replicate-tracking.jsonl"
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
    f"DEBUGG: Running replicate consistency analysis for U2OS cells (n_iterations={n_iterations})..."
)
u2os_results_df = run_replicate_consistency_analysis(
    profiles=u2os_df,
    plate_name_dict=u2os_plate_name_dict,
    meta_cols=meta_features,
    morph_feats=morph_features,
    perturbation_col="Metadata_gene",
    cell_type="U2OS",
    n_iterations=n_iterations,
    save_dir=outdir,
    negcon_subsample_frac=0.02,
    random_perturbation_comparison=False,
    perturbation_to_ignore=["negcon"],
)
print(
    f"DEBUGG: Running replicate consistency analysis for U2OS cells (n_iterations={n_iterations})..."
)

u2os_rnd_trt_results_df = run_replicate_consistency_analysis(
    profiles=u2os_df,
    plate_name_dict=u2os_plate_name_dict,
    meta_cols=meta_features,
    morph_feats=morph_features,
    perturbation_col="Metadata_gene",
    cell_type="U2OS",
    n_iterations=n_iterations,
    save_dir=outdir,
    negcon_subsample_frac=0.02,
    random_perturbation_comparison=True,
    perturbation_to_ignore=["negcon"],
)


# In[ ]:


# print(
#     f"DEBUGG: Running replicate consistency analysis for A549 cells (n_iterations={n_iterations})..."
# )
# a549_results_df = run_replicate_consistency_analysis(
#     profiles=a549_df,
#     plate_name_dict=a549_plate_name_dict,
#     meta_cols=meta_features,
#     morph_feats=morph_features,
#     perturbation_col="Metadata_gene",
#     cell_type="A549",
#     n_iterations=n_iterations,
#     save_dir=outdir,
#     negcon_subsample_frac=0.02,
#     random_perturbation_comparison=False,
#     perturbation_to_ignore=["negcon"],
# )
# print(
#     f"DEBUGG: Running replicate consistency analysis for A549 cells (n_iterations={n_iterations})..."
# )

# a549_rnd_trt_results_df = run_replicate_consistency_analysis(
#     profiles=a549_df,
#     plate_name_dict=a549_plate_name_dict,
#     meta_cols=meta_features,
#     morph_feats=morph_features,
#     perturbation_col="Metadata_gene",
#     cell_type="A549",
#     n_iterations=n_iterations,
#     save_dir=outdir,
#     negcon_subsample_frac=0.02,
#     random_perturbation_comparison=True,
#     perturbation_to_ignore=["negcon"],
# )


# # save results as parquet
# a549_results_df.write_parquet(
#     outdir / f"a549_{pert_type}_replicate_consistency_results.parquet"
# )
# a549_rnd_trt_results_df.write_parquet(
#     outdir / f"a549_random_{pert_type}_replicate_consistency_results.parquet"
# )
