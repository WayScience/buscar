#!/bin/bash
# This script executes R plotting notebooks for CFReT analysis

# Activate the R conda environment
conda activate r_buscar

# Convert R notebooks to R scripts
jupyter nbconvert --output-dir=nbconverted --to script ./*.ipynb

# Execute each R script
for script in nbconverted/*.r; do
    if [ -f "$script" ]; then
        echo "Executing: $script"
        Rscript "$script"
    fi
done

echo "All R plotting scripts executed successfully"
