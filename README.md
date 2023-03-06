# Chess Rating Prediction with machine learning

This project is designed to be the final project for the Machine Learning 2 course of the University of Potsdam in  the wintersemester 2022/23.\
The goal is to use open source data to create a model, that can predict the elo/rating of a chess game.


___
## Dataset
the raw dataset for this project can be downloaded at [lichess.org](https://database.lichess.org).\
They save all the games played on the site and make it openly available in the common chess notation PGN. This is basically a list of games, as describe on the above linked page.\
Even compressed, the data of one month is ~33 GB (~100.000.000 games), which is why a lot of optimization in the processing of this data takes place.\


___
## Repository Structure

There are three main folders, to this repository
1. `src/`, containing the python scripts for preprocessing, analysing and modelling the data
2. `data/`, containing under `data/raw/` the raw data from lichess and under `data/processed/` the cleaned data exports as `.csv`, which can be used for machine learning.
3. `trainded_models/` containing the various models, that were trained, as well as potentially coefficients, and the evaluations on the test data


___
## Code explanations

The folder `src/` contains the following three python scripts
___
### `src/data_preprocessing.py`
The data is grouped into 4 different categories, that are based on the timecontrol of the game. The hypothesis here is, that a game of bullet chess, where each player only has one minute to play, looks very different than a classical game, where each player has at least 30 minutes.\
This script contains several functions that in total read the original file of games and process them into 8 `.csv` files: One training and one testing dataset per timecontrol.\
There are several metadata information, i extract from the list of moves, like 'number of king moves' and 'number of checks', as well as the first moves played and the name of the opening, as one-hot encoded features.\
Furthermore, I added the timeseries of the 'evaluation of the position', as well as the 'time spent per move' to the file.\
For the sake of simplicity, i created the columns `average_elo`, which is the mean of the elo of the two players. This will represent the regression target of the model. Furthermore, all game with a rating difference of >200, will be disregarded. (this was less than 5% of the games)

The input data, expected for these functions are way to large to be stored on git, but can be downloaded at the above mentioned link. In this script, the files from october and november 2022 were used.

These functions are written in a batched manner, to save memory. The last function is designed to concat all the batch exports and also perform a train/test split (60%/40%), before export the data to a .csv file.
___
### `src/data_analysis.py`
This script uses the previously exported datafiles and performs some simple data analysis on it, as well as exports some interesting graphics.

Please note that some of these functions were written, before the encoding of some features, was redesigned in the preprocessing script, and are therefore not expected to work without a rework.

___
### `src/modelling.py`
This script contains the functions for the actual modelling of the `average_elo` of the games. There are several models in this script, each with its own class (Linear Regression, XGBoost, LSTM, Combined Model). Each of these classes contain the following functions:
* `train()`: loads the training data and fits the model 
* `save()`: exports the model, so it can be loaded later for evaluation
* `load()`: loads the exported model
* `evaluate()`: loads the testing data and evaluates the model w.r.t. differen metrices
* `save_and_print_evaluation_outcome()`: prints theevaluations and saves them to a .csv for later analysis.

The combined model is an ensemble model, combining the prediction from the linear Regression and the XGBoost/LSTM. Since they are trained on different subsets of the data the hypothesis is that the Ensemble might improve the quality of either single model. Here, these function names might change.\
Also some models have even more functions, depending on the type of the model.

The Linear Reegression model uses only the metadata columns as input, while the XGBoost and LSTM models use only the timeseries features (evaluation, clocks). That is beacuse i tried the other combinations, but Linear regression performed worse with the timeseries information and LSTM performed worse with the metadata.

There are some hyperparameters in the LSTM Model, that were determined, using manual Bandit based optimization. That means some of the training data was hold back as a validation set. The performance on these sets was measured during training, and when the perfromance on a certain parameter set was very bad after some interations, that set was ruled out.

The hyperparaeters, that were considered were: batch_size, number of hidden layers, number of LSTM units and the length of the evaluation and clock timeseries.

___
## Used Libraries
All the used libraries can be found in `requirements.txt` 
The main libraries used for the Modelling, are [keras](https://keras.io/api/layers/recurrent_layers/lstm/), with its LSTM implementations and [xgboost](https://xgboost.readthedocs.io/en/stable/python/python_api.html), with its gradinent boosting library.

___
## Further notes
Please not that in the encompassing presentation, not all features, evaluations and models will be talked about in detail. Also more analysis was done, than will be presented, mostly due to timeconstraints.

Possible improvements to this repository could be:
* an automization of the hyperparameteroptimization, that was here performed manually.
* additional models, to be tested
* different/more metadata, to be extreacted from the raw data