
# This script executes the download and preprocessing of the data


# activate the conda environment
source activate buscar

# convert the notebook to a script
jupyter nbconvert --output-dir=nbconverted --to script *.ipynb

# execute the script
python nbconverted/1.download-data.py
python nbconverted/2.preprocessing.py
