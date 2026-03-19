#!/bin/bash
set -euo pipefail

# Ensure conda functions are available in this shell session
eval "$(conda shell.bash hook)"

# --- Step 1: convert notebooks to python scripts ---
conda activate buscar
jupyter nbconvert --output-dir=nbconverted --to script ./*.ipynb

# --- Step 2: run the python scripts ---
python nbconverted/0.explore-mitocheck-data.py
python nbconverted/1.buscar-analysis.py
