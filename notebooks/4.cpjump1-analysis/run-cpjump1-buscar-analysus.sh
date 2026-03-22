#!/bin/bash
set -e

# Activate your environment (adjust as needed)
conda activate buscar

# convert all
jupyter nbconvert --to script --output-dir=./nbconverted ./*.ipynb

# execute all scripts
python ./nbconverted/1.generate-on-off-signatures.py
python ./nbconverted/2.run-compound-replicate-analysis.py
python ./nbconverted/3.plot-replicate-consistency-analysis.py
