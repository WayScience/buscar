#!/bin/bash
# This script is used to run the components of buscar in the notebook

# activate the conda environment
conda activate buscar

# convert the notebook to a script
jupyter nbconvert --output-dir=nbconverted --to script ./*.ipynb

# execute the script
python nbconverted/1.signatures.py
python nbconverted/2.assess-heterogeneity.py
python nbconverted/3.refinement.py
python nbconverted/4.measure-phenotypic-activity.py
