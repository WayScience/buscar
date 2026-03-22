#!/bin/bash
set -e

# Activate your environment (adjust as needed)
conda activate buscar

# convert all
jupyter nbconvert --to script --output-dir=./nbconverted ./*.ipynb

# execute all scripts
python ./nbconverted/1.generate-on-off-signatures.py
python ./nbconverted/2.assess-heterogeneity.py
python ./nbconverted/3.calculate-on-off-scores.py
python ./nbconverted/4.run_buscar_rankings_base_on_moa.py
python ./nbconverted/5.cpjump_u2os_MoA_analysis.py
