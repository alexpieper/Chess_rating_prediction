# Chess Rating Prediction with machine learning

This project is designed to be the final project for the Machine Learning 2 course of the University of Potsdam in  the wintersemester 2022/23.



## Dataset
the raw dataset for this project can be downloaded at [lichess.org](https://database.lichess.org).\
They save all the games played on the site and make it openly available in the common chess notation PGN. This is basically a list of games, as describe on the above linked page.\
Even compressed, the data of one month is ~33 GB (~100.000.000 games), which is why a lot of optimization in the processing of this data takes place.\


## Repository Structure

There are three main folders, to this repository
1. `src/`, containing the python scripts for preprocessing, analysing and modelling the data
2. `data/`, containing under `data/raw/` the raw data from lichess and under `data/processed/` the cleaned data exports as `.csv`, which can be used for machine learning.
3. `trainded_models/` containing the various models, that were trained, as well as potentially coefficients, and the evaluations on the test data

## Modelling Approach

## Code explanations

The folder `src/` contains the following three python scripts
### `src/data_preprocessing.py`
This script contains several functions that in total read the original file of games and process them into 8 `.csv` files. One training and one testing dataset per timecontrol. 
### `src/data_analysis.py`
### `src/modelling.py`

## Used Libraries

## Further notes
