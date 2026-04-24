#!/usr/bin/env python

# # CFReT BUSCAR Analysis
#
# This notebook applies the BUSCAR pipeline to the CFReT pilot Cell Painting dataset to identify compounds that rescue failing cardiac cells toward a healthy phenotype.
#
# **Experimental setup:**
# - **Reference (failing):** Failing cardiomyocytes + DMSO — the diseased baseline we want to move *away* from
# - **Target (healthy):** Healthy cardiomyocytes + DMSO — the phenotypic state we want treatments to move *toward*
# - **Treatments evaluated:** TGFRi applied to both failing and healthy cells
#
# BUSCAR quantifies how much each treatment shifts the morphological profile of failing cells toward the healthy state.
#
# **Data & references:**
# - Dataset: CFReT pilot (Cell Painting of cardiac fibrosis model)
# - Paper: https://www.ahajournals.org/doi/full/10.1161/CIRCULATIONAHA.124.071956
# - Data repo: https://github.com/WayScience/cellpainting_predicts_cardiac_fibrosis

# In[1]:


import json
import pathlib
import sys

import polars as pl

sys.path.append("../../")
from buscar.metrics import calculate_buscar_scores
from buscar.signatures import get_signatures

from utils.data_utils import split_meta_and_features
from utils.io_utils import load_profiles

# ## Parameters
#
# - `treatment_col`: column grouping cells by cell type + treatment (e.g. `failing_DMSO`, `healthy_TGFRi`)
# - `failing_label`: the diseased reference state (failing cells + DMSO)
# - `healthy_label`: the healthy target state (healthy cells + DMSO)
# - `on_off_signatures_method`: statistical test used to identify on/off morphological features

# In[2]:


# setting parameters
treatment_col = "Metadata_cell_type_and_treatment"

# buscar parameters
healthy_label = "healthy_DMSO"
failing_label = "failing_DMSO"
on_off_signatures_method = "ks_test"


# ## Paths
#
# Input: single-cell morphology profiles from the CFReT pilot dataset.
# Output: signature files and phenotypic score CSVs written to `./results/`.

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
results_dir = pathlib.Path("./results").resolve()
results_dir.mkdir(parents=True, exist_ok=True)

# set signatures results dir
signatures_results_dir = (results_dir / "signatures").resolve()
signatures_results_dir.mkdir(parents=True, exist_ok=True)

# set phenotypic scores results dir
phenotypic_scores_results_dir = (results_dir / "phenotypic_scores").resolve()
phenotypic_scores_results_dir.mkdir(parents=True, exist_ok=True)


# ## Data Preprocessing
#
# Load single-cell profiles and add two derived metadata columns:
# - `Metadata_treatment_and_heart`: links a treatment to its biological replicate (heart number)
# - `Metadata_cell_type_and_treatment`: combined group label used throughout the analysis (e.g. `failing_DMSO`, `healthy_TGFRi`)

# In[4]:


# Load raw single-cell morphology profiles
cfret_df = load_profiles(cfret_profiles_path)

# Track which heart (biological replicate) each cell came from
cfret_df = cfret_df.with_columns(
    (
        pl.col("Metadata_treatment").cast(pl.Utf8)
        + "_heart_"
        + pl.col("Metadata_heart_number").cast(pl.Utf8)
    ).alias("Metadata_treatment_and_heart")
)

# Create combined group label: "<cell_type>_<treatment>" (e.g. "failing_DMSO", "healthy_TGFRi")
cfret_df = cfret_df.with_columns(
    (
        pl.col("Metadata_cell_type").cast(pl.Utf8)
        + "_"
        + pl.col("Metadata_treatment").cast(pl.Utf8)
    ).alias(treatment_col)
)

# Separate metadata columns from morphology feature columns
cfret_meta, cfret_feats = split_meta_and_features(cfret_df)

cfret_df.head()


# ## Cell Counts per Treatment Group
#
# Verify how many single cells are available per treatment group before running BUSCAR.

# In[5]:


# show how many cells per treatment
# shows the number of cells per treatment that will be clustered.
cells_per_treatment_counts = cfret_df.group_by(treatment_col).len().sort(treatment_col)
cells_per_treatment_counts


# ## BUSCAR pipeline

# ### Step 1: Extract Morphological Signatures
#
# Signatures define which features distinguish the failing (reference) from the healthy (target) DMSO cells. They are derived once from the DMSO controls and reused when scoring all treatments.
#
# - **On_morphology signatures (`on_sigs`):** Features that shift significantly between failing and healthy cells — the morphological hallmarks of the disease-to-health transition.
# - **Off_morpholgy signatures (`off_sigs`):** Features that remain stable between failing and healthy cells. A treatment that perturbs these is causing non-specific morphological changes.

# In[6]:


# Reference: failing DMSO cells — the diseased baseline
ref_df = cfret_df.filter(pl.col(treatment_col) == failing_label)

# Target: healthy DMSO cells — the phenotypic state we want treatments to approach
target_df = cfret_df.filter(pl.col(treatment_col) == healthy_label)


# In[7]:


signatures_outpath = (signatures_results_dir / "cfret_pilot_signatures.json").resolve()

# Load cached signatures if available, otherwise compute from failing vs healthy DMSO cells
if signatures_outpath.exists():
    print("Signatures already exist, skipping this step.")
    with open(signatures_outpath) as f:
        sigs = json.load(f)
        on_sigs = sigs["on"]
        off_sigs = sigs["off"]
else:
    on_sigs, off_sigs, _ = get_signatures(
        ref_profiles=ref_df,
        target_profiles=target_df,
        morph_feats=cfret_feats,
        test_method=on_off_signatures_method,
    )

    # Save signatures for reuse across analyses
    with open(signatures_outpath, "w") as f:
        json.dump({"on": on_sigs, "off": off_sigs}, f, indent=4)


# ### Calculating pertrubations scores
#
# This section quantifies how each treatment affects cell morphology compared to the reference control (DMSO_heart_11), using the previously defined on and off signatures. The resulting buscar scores are used to rank treatments and highlight the most active compounds.
#
# **How scores are calculated:**
#
# - **On-signature features:** We use the Earth Mover's Distance (EMD) to measure how much the on-features for each treatment differ from the reference. A higher EMD means a greater morphological change.
# - **Off_Buscar features:** We use the affected ratio, which detects if features that should remain stable (off-features) are altered by a treatment. A higher affected ratio suggests more off-target or unintended effects.

# In[8]:


scores_output = (
    phenotypic_scores_results_dir / "cellpainting_cardiac_fibrosis_buscar_scores.csv"
).resolve()
if scores_output.exists():
    print("Phenotypic scores already exist, skipping this step.")
    treatment_scores = pl.read_csv(scores_output)
else:
    treatment_scores = calculate_buscar_scores(
        profiles=cfret_df,
        meta_cols=cfret_meta,
        on_morphology_signature=on_sigs,
        off_morphology_signature=off_sigs,
        ref_state=failing_label,
        target=healthy_label,
        on_method="emd",
        off_method="affected_ratio",
        ratio_stats_method=on_off_signatures_method,
        perturbation_col=treatment_col,
    )

    # save phenotypic scores
    treatment_scores.write_csv(scores_output)

# display scores
treatment_scores


# ## Caclulating buscar scores at replicate level
#
# The analysis before pools all replicates into a populations of cells to generate a score. however, for this analysis, we would like to see how each replicate scores seeing a more finner score

# In[9]:


# update Metadata_treatment column to have the "Metadata_treatment + Metadata_Well"
# this allows buscar to see each celltype-treatment-well as a different treatment,
# which allows us to see the variability between replicates and how that affects the scores.
replicate_df = cfret_df.filter(pl.col("Metadata_treatment") == "TGFRi").with_columns(
    (
        pl.col("Metadata_cell_type").cast(pl.Utf8)
        + "-"
        + pl.col("Metadata_treatment").cast(pl.Utf8)
        + "-"
        + pl.col("Metadata_Well").cast(pl.Utf8)
    ).alias("Metadata_treatment")
)

# concatenate the new replicate_df with the original ref_df and target_df to create
# a new cfret_df that has the updated Metadata_treatment column for the TGFRi treatment.
replicate_df = pl.concat([ref_df, target_df, replicate_df])

# next we want to update the "DMSO" value in the replicate_df to be "failing_DMSO" and "healthy_DMSO"
# if Metadata_cell_type is "healthy" then we want to change the Metadata_treatment to "healthy_DMSO"
# if Metadata_cell_type is "failing" then we want to change the Metadata_treatment to "failing_DMSO"
replicate_df = replicate_df.with_columns(
    pl.when(pl.col("Metadata_treatment") == "DMSO")
    .then(pl.col("Metadata_cell_type") + "_DMSO")
    .otherwise(pl.col("Metadata_treatment"))
    .alias("Metadata_treatment")
)


# In[10]:


output_path = (
    phenotypic_scores_results_dir
    / "cellpainting_cardiac_fibrosis_buscar_scores_replicates.csv"
).resolve()
if output_path.exists():
    print("Phenotypic scores with replicates already exist, skipping this step.")
    buscar_scores = pl.read_csv(output_path)
else:
    buscar_scores = calculate_buscar_scores(
        profiles=replicate_df,
        meta_cols=cfret_meta,
        on_morphology_signature=on_sigs,
        off_morphology_signature=off_sigs,
        ref_state=failing_label,
        target=healthy_label,
        on_method="emd",
        off_method="affected_ratio",
        ratio_stats_method=on_off_signatures_method,
        perturbation_col="Metadata_treatment",
    )

    # add cell counts per treatment
    buscar_scores = buscar_scores.join(
        replicate_df.group_by("Metadata_treatment")
        .len()
        .rename({"len": "cell_counts"}),
        left_on="perturbation",
        right_on="Metadata_treatment",
        how="left",
    )

    # Split perturbation into treatment and well
    # Extracts the well (last element after '-') and the treatment (everything before the last '-')
    buscar_scores = (
        buscar_scores.with_columns(
            pl.col("perturbation")
            .str.split_exact("-", 2)
            .struct.rename_fields(["cell_type", "treatment", "well"])
        )
        .unnest("perturbation")
        .with_columns(
            (pl.col("cell_type") + "-" + pl.col("treatment")).alias("treatment")
        )
        .drop("cell_type")
    )

    # save scores
    buscar_scores.write_csv(output_path)

buscar_scores
