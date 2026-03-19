#!/bin/bash
set -euo pipefail

# Ensure conda functions are available in this shell session
eval "$(conda shell.bash hook)"

# --- Step 1: convert notebooks to R scripts ---
conda activate buscar
jupyter nbconvert --output-dir=nbconverted --to script ./*.ipynb

# --- Step 2: run the R scripts ---
conda activate r_buscar
Rscript nbconverted/1.heat-map-on-off-scores.r
Rscript nbconverted/2.gene-ranking-relationship.r
Rscript nbconverted/3.linear-modeling-ranking-and-proportion.r

# --- Step 3: restore the default environment ---
conda activate buscar
