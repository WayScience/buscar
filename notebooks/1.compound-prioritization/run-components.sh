#!/bin/bash
# This script is used to run the components of buscar in the notebook

# activate the conda environment
conda activate buscar

# convert the notebook to a script
jupyter nbconvert --output-dir=nbconverted --to script ./*.ipynb

# execute the script
python nbconverted/1.signatures.py
