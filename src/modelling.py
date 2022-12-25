import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import pickle

import xgboost as xgb
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.models import load_model
import tensorflow as tf
import tensorflow_probability as tfp


class LinearRegressionModel:
    '''
    This class represent the Linear Regression model.
    Parameter:
     - It only uses the non-timeseries columns (i.e. metadata like number_moves, opening, first_move etc.)
     - It only uses the features, whose absolute sum is larger than 10. (this filters the very rare opening, as they tend to overfit to the trainin data. coefficient goes up to 10000)
    '''
    def __init__(self, game_type, n_rows):
        self.train_file = os.path.join('data', 'processed', 'all_games_clean', f'{game_type}_train.csv')
        self.test_file = os.path.join('data', 'processed', 'all_games_clean', f'{game_type}_test.csv')
        base_dir = os.path.join('trained_models', 'linear_regression', str(n_rows))
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        self.model_path = os.path.join(base_dir, f'{game_type}_model.pkl')
        self.evaluation_path = os.path.join(base_dir, f'{game_type}_eval.csv')
        self.coefficients_path = os.path.join(base_dir, f'{game_type}_coefs.csv')
        self.valid_columns_file = os.path.join(base_dir, f'{game_type}_valid_cols.txt')
        self.nrows = n_rows
        self.model = LinearRegression()

    def clean_dataset(self, df, load_new):
        '''
         applies the above mentioned filtering of sparse columns and save these columns in a txt file, for the testing
        :param df: training of testing data
        :param load_new: whether to calculate the columns new. usually (training: True, testing: False)
        :return: cleaned df
        '''
        if load_new:
            sums = df.sum(axis=0)
            self.valid_columns = [i for i in df.columns if (abs(sums[i]) > 10) and not (('eval_' in i or 'clock_' in i))]
            df = df[self.valid_columns]
            with open(self.valid_columns_file, 'w') as f:
                for line in self.valid_columns:
                    f.write(f"{line}\n")
        else:
            with open(self.valid_columns_file) as file:
                self.valid_columns  = [line.rstrip() for line in file]
            df = df[self.valid_columns]
        return df

    def train(self):
        '''
        fits the model to the training data
        :return:
        '''
        print('Start: reading training data')
        self.train_df = pd.read_csv(self.train_file, index_col=0, nrows=self.nrows)
        self.train_df = self.clean_dataset(self.train_df, True)
        X_train = self.train_df.drop(columns=['average_elo'])
        y_train = self.train_df['average_elo']
        print('Start: training')
        self.model.fit(X_train, y_train)
        # this is just to clear these variables from the memory, as they are not used anymore
        self.train_df = pd.DataFrame()
        X_train = pd.DataFrame()
        y_train = pd.DataFrame()

    def save(self):
        '''
        exports the trained model, so it can be used later for evaluation
        :return:
        '''
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self):
        '''
        loads the exported model, so it can be used for evaluation
        :return:
        '''
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

    def evaluate(self):
        '''
        evaluates the model on the testing data w.r.t. different evaluation metrics
        :return:
        '''
        print('Start: reading testing data')
        self.test_df = pd.read_csv(self.test_file, index_col=0, nrows=self.nrows)
        self.test_df = self.clean_dataset(self.test_df, False)
        print('Start: evaluation')
        self.test_df = self.test_df[self.valid_columns]
        X_test = self.test_df.drop(columns=['average_elo'])
        y_test = self.test_df['average_elo']
        y_pred = self.model.predict(X_test)

        # calculate the errors
        median_abs_error = np.median([abs(i - j) for i,j in zip(y_pred, y_test.tolist())])
        mean_abs_error = np.mean([abs(i - j) for i,j in zip(y_pred, y_test.tolist())])
        median_squared_error = np.median([(i - j)**2 for i,j in zip(y_pred, y_test.tolist())])
        mean_squared_error = np.mean([(i - j)**2 for i,j in zip(y_pred, y_test.tolist())])
        dummy_guess = np.mean(y_test.tolist())
        median_absolute_dummy_error = np.median([abs(i - dummy_guess) for i in y_test.tolist()])
        mean_absolute_dummy_error = np.mean([abs(i - dummy_guess) for i in y_test.tolist()])
        median_squared_dummy_error = np.median([(i - dummy_guess)**2 for i in y_test.tolist()])
        mean_squared_dummy_error = np.mean([(i - dummy_guess)**2 for i in y_test.tolist()])

        self.eval_measures = {'Mean Squared Error': mean_squared_error, 'Median Squared Error': median_squared_error,
                              'Mean Absolute Error': mean_abs_error, 'Median Absolute Error': median_abs_error,
                              'Mean Squared Dummy Error': mean_squared_dummy_error, 'Median Squared Dummy Error': median_squared_dummy_error,
                              'Mean Absolute Dummy Error': mean_absolute_dummy_error, 'Median Absolute Dummy Error': median_absolute_dummy_error
                              }

    def save_and_print_evaluation_outcome(self):
        '''
        exports the evaluation metrics and prints them to the console
        :return:
        '''
        measure_df = pd.DataFrame(self.eval_measures.items(), columns=['Measure', 'Value'])
        measure_df.to_csv(self.evaluation_path)
        for measure, value in self.eval_measures.items():
            print(f'{measure}: {value:.2f}')

    def save_and_print_coefficients(self, verbose = False):
        '''
        exports the coefficients and prints them to the console. Interesting for later analysis
        :param verbose: whether or not to print the coefficients, since it can be a long list
        :return:
        '''
        coefficients = self.model.coef_
        column_names = self.test_df.drop(columns=['average_elo']).columns
        coef_df = pd.DataFrame(
            {'Feature': column_names.tolist(),
             'value': coefficients,
             })
        coef_df.to_csv(self.coefficients_path)
        if verbose:
            for i, coef in enumerate(coefficients):
                print(f'{column_names[i]}: {"".join([" " for i in range(100 - len(column_names[i]))])} {coef:.2f}')


class LSTMRegression:
    '''
    This class represent the LSTM Neural Network.
    Parameter:
     - It only uses the timeseries columns (i.e. evaluations and clocks up to 60) (this (60) parameter was fitted via gridsearch)
    '''
    def __init__(self, game_type, n_rows):
        self.train_file = os.path.join('data', 'processed', 'all_games_clean', f'{game_type}_train.csv')
        self.test_file = os.path.join('data', 'processed', 'all_games_clean', f'{game_type}_test.csv')
        base_dir = os.path.join('trained_models', 'lstm_only_ts', str(n_rows))
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        self.model_path = os.path.join(base_dir, f'{game_type}_model.pkl')
        self.evaluation_path = os.path.join(base_dir, f'{game_type}_eval.csv')
        self.coefficients_path = os.path.join(base_dir, f'{game_type}_coefs.csv')
        self.valid_columns_file = os.path.join(base_dir, f'{game_type}_valid_cols.txt')
        self.nrows = n_rows
        self.usefuls_cols = [f'eval_{i}' for i in range(60)] + [f'clock_{i}' for i in range(60)] + ['average_elo']
        # self.usefuls_cols = [f'eval_{i}' for i in range(60)] + ['average_elo']

    def median_abs_error(self, y_true, y_pred):
        '''
        A custom evaluation metric for the validation set
        :param y_true: true value
        :param y_pred: predicted value
        :return:
        '''
        abs_diff = abs(y_true - y_pred)
        # this is a tensorflow representation for the median
        return tfp.stats.percentile(abs_diff, 50.0, interpolation='midpoint')

    def train(self):
        '''
        fits the model to the training data. also takes 33% of the training data as validation set
        :return:
        '''
        epochs = 250
        batch_size = 256

        print('Start: reading training data')
        self.train_df = pd.read_csv(self.train_file, nrows=self.nrows, usecols=self.usefuls_cols)
        self.validation_df = self.train_df[int(self.nrows/1.5):]
        self.train_df = self.train_df[:int(self.nrows/1.5)]
        X_train = self.train_df.drop(columns=['average_elo'])
        X_validation = self.validation_df.drop(columns=['average_elo'])
        y_train = self.train_df['average_elo']
        y_validation = self.validation_df['average_elo']
        self.train_mean = np.mean(y_train)
        # centering the data around 0 helped the NN to converge faster. maybe standard normalizing might be even better (std=1)
        y_train = y_train - self.train_mean
        y_validation = y_validation - self.train_mean

        print('Start: training')
        input_shape = (X_train.shape[1],1)
        output_shape = 1

        self.model = Sequential()
        self.model.add(LSTM(units=32, input_shape=input_shape))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(output_shape))
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=[self.median_abs_error])
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_validation, y_validation))
        # this is just to clear these variables from the memory, as they are not used anymore
        self.train_df = pd.DataFrame()
        X_train = pd.DataFrame()
        y_train = pd.DataFrame()

    def save(self):
        '''
        exports the trained model, so it can be used later for evaluation
        :return:
        '''
        self.model.save(self.model_path)

    def load(self):
        '''
        loads the exported model, so it can be used for evaluation
        :return:
        '''
        self.model = load_model(self.model_path, custom_objects = {'median_abs_error': self.median_abs_error})


    def evaluate(self):
        '''
        evaluates the model on the testing data w.r.t. different evaluation metrics
        :return:
        '''
        print('Start: reading testing data')
        self.test_df = pd.read_csv(self.test_file, nrows=self.nrows, usecols=self.usefuls_cols)
        print('Start: evaluation')
        X_test = self.test_df.drop(columns=['average_elo'])
        y_test = self.test_df['average_elo']
        y_pred = self.model.predict(X_test)
        y_pred = [i + self.train_mean for i in y_pred.flatten().tolist()]

        # calculate the errors
        median_abs_error = np.median([abs(i - j) for i, j in zip(y_pred, y_test.tolist())])
        mean_abs_error = np.mean([abs(i - j) for i, j in zip(y_pred, y_test.tolist())])
        median_squared_error = np.median([(i - j) ** 2 for i, j in zip(y_pred, y_test.tolist())])
        mean_squared_error = np.mean([(i - j) ** 2 for i, j in zip(y_pred, y_test.tolist())])
        dummy_guess = np.mean(y_test.tolist())
        median_absolute_dummy_error = np.median([abs(i - dummy_guess) for i in y_test.tolist()])
        mean_absolute_dummy_error = np.mean([abs(i - dummy_guess) for i in y_test.tolist()])
        median_squared_dummy_error = np.median([(i - dummy_guess) ** 2 for i in y_test.tolist()])
        mean_squared_dummy_error = np.mean([(i - dummy_guess) ** 2 for i in y_test.tolist()])

        self.eval_measures = {'Mean Squared Error': mean_squared_error, 'Median Squared Error': median_squared_error,
                              'Mean Absolute Error': mean_abs_error, 'Median Absolute Error': median_abs_error,
                              'Mean Squared Dummy Error': mean_squared_dummy_error,
                              'Median Squared Dummy Error': median_squared_dummy_error,
                              'Mean Absolute Dummy Error': mean_absolute_dummy_error,
                              'Median Absolute Dummy Error': median_absolute_dummy_error
                              }

    def save_and_print_evaluation_outcome(self):
        '''
        exports the evaluation metrics and prints them to the console
        :return:
        '''
        measure_df = pd.DataFrame(self.eval_measures.items(), columns=['Measure', 'Value'])
        measure_df.to_csv(self.evaluation_path)
        for measure, value in self.eval_measures.items():
            print(f'{measure}: {value:.2f}')



class XGBoostRegression:
    '''
    This class represent the XGBoost Regrssion Model.
    Parameter:
     - It only uses the timeseries columns (i.e. evaluations and clocks up to 60) (this (60) parameter was fitted via gridsearch)
    '''
    def __init__(self, game_type, n_rows, limit):
        self.train_file = os.path.join('data', 'processed', 'all_games_clean', f'{game_type}_train.csv')
        self.test_file = os.path.join('data', 'processed', 'all_games_clean', f'{game_type}_test.csv')
        base_dir = os.path.join('trained_models', 'xgboost_only_ts', str(n_rows), str(limit))
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        self.model_path = os.path.join(base_dir, f'{game_type}_model.pkl')
        self.evaluation_path = os.path.join(base_dir, f'{game_type}_eval.csv')
        self.coefficients_path = os.path.join(base_dir, f'{game_type}_coefs.csv')
        self.valid_columns_file = os.path.join(base_dir, f'{game_type}_valid_cols.txt')
        self.nrows = n_rows
        self.usefuls_cols = [f'eval_{i}' for i in range(limit)] + [f'clock_{i}' for i in range(limit)] + ['average_elo']
        self.model = xgb.XGBRegressor()


    def train(self):
        '''
        fits the model to the training data. also takes 33% of the training data as validation set
        :return:
        '''
        print('Start: reading training data')
        self.train_df = pd.read_csv(self.train_file, nrows=self.nrows, usecols = self.usefuls_cols)
        X_train = self.train_df.drop(columns=['average_elo'])
        y_train = self.train_df['average_elo']

        print('Start: training')
        self.model.fit(X_train, y_train)
        # this is just to clear these variables from the memory, as they are not used anymore
        self.train_df = pd.DataFrame()
        X_train = pd.DataFrame()
        y_train = pd.DataFrame()

    def save(self):
        '''
        exports the trained model, so it can be used later for evaluation
        :return:
        '''
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self):
        '''
        loads the exported model, so it can be used for evaluation
        :return:
        '''
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

    def evaluate(self):
        '''
        evaluates the model on the testing data w.r.t. different evaluation metrics
        :return:
        '''
        print('Start: reading testing data')
        self.test_df = pd.read_csv(self.test_file, nrows=self.nrows, usecols = self.usefuls_cols)
        self.test_df = self.clean_dataset(self.test_df, False)
        print('Start: evaluation')
        self.test_df = self.test_df[self.valid_columns]
        X_test = self.test_df.drop(columns=['average_elo'])
        y_test = self.test_df['average_elo']
        y_pred = self.model.predict(X_test)

        # calculate the errors
        median_abs_error = np.median([abs(i - j) for i,j in zip(y_pred, y_test.tolist())])
        mean_abs_error = np.mean([abs(i - j) for i,j in zip(y_pred, y_test.tolist())])
        median_squared_error = np.median([(i - j) ** 2 for i, j in zip(y_pred, y_test.tolist())])
        mean_squared_error = np.mean([(i - j) ** 2 for i, j in zip(y_pred, y_test.tolist())])
        dummy_guess = np.mean(y_test.tolist())
        median_absolute_dummy_error = np.median([abs(i - dummy_guess) for i in y_test.tolist()])
        mean_absolute_dummy_error = np.mean([abs(i - dummy_guess) for i in y_test.tolist()])
        median_squared_dummy_error = np.median([(i - dummy_guess) ** 2 for i in y_test.tolist()])
        mean_squared_dummy_error = np.mean([(i - dummy_guess) ** 2 for i in y_test.tolist()])

        # todo: add benchmark: always to guess the median.
        self.eval_measures = {'Mean Squared Error': mean_squared_error, 'Median Squared Error': median_squared_error,
                              'Mean Absolute Error': mean_abs_error, 'Median Absolute Error': median_abs_error,
                              'Mean Squared Dummy Error': mean_squared_dummy_error, 'Median Squared Dummy Error': median_squared_dummy_error,
                              'Mean Absolute Dummy Error': mean_absolute_dummy_error, 'Median Absolute Dummy Error': median_absolute_dummy_error
                              }

    def save_and_print_evaluation_outcome(self):
        '''
        exports the evaluation metrics and prints them to the console
        :return:
        '''
        measure_df = pd.DataFrame(self.eval_measures.items(), columns=['Measure', 'Value'])
        measure_df.to_csv(self.evaluation_path)
        for measure, value in self.eval_measures.items():
            print(f'{measure}: {value:.2f}')




class CombinedModel():
    '''
    Here we load two trained models (one linear regression, one Advanced), average their predicted values for the test set and see if that is better than either model.
    '''
    def __init__(self, game_type, n_rows, model_type):
        self.test_file = os.path.join('data', 'processed', 'all_games_clean', f'{game_type}_test.csv')

        # initialize all the directories to the models etc.
        linear_base_dir = os.path.join('trained_models', 'linear_regression', str(n_rows))
        self.linear_model_path = os.path.join(linear_base_dir, f'{game_type}_model.pkl')
        self.valid_columns_file = os.path.join(linear_base_dir, f'{game_type}_valid_cols.txt')
        self.load_linear_model()
        with open(self.valid_columns_file) as file:
            self.valid_columns = [line.rstrip() for line in file]
        self.model_type = model_type
        if model_type == 'xgboost':
            xgb_base_dir = os.path.join('trained_models', 'xgboost_only_ts', str(n_rows), '60')
            self.xbg_model_path = os.path.join(xgb_base_dir, f'{game_type}_model.pkl')
            self.valid_columns_file = os.path.join(xgb_base_dir, f'{game_type}_valid_cols.txt')
            self.nrows = n_rows
            self.usefuls_cols = [f'eval_{i}' for i in range(60)] + [f'clock_{i}' for i in range(60)] + ['average_elo']
            self.load_xgb()
        elif model_type == 'lstm':
            self.train_means = {
                'bullet': 1502.622895,
                'blitz': 1613.1635275,
                'rapid': 1562.077505,
                'classical': 1577.9704829927161,
            }
            self.train_mean = self.train_means[game_type]
            lstm_base_dir = os.path.join('trained_models', 'lstm_only_ts', '300000')
            self.lstm_model_path = os.path.join(lstm_base_dir, f'{game_type}_model.pkl')
            self.valid_columns_file = os.path.join(lstm_base_dir, f'{game_type}_valid_cols.txt')
            self.nrows = n_rows
            self.usefuls_cols = [f'eval_{i}' for i in range(60)] + [f'clock_{i}' for i in range(60)] + ['average_elo']
            self.load_lstm()

        base_dir = os.path.join('trained_models', 'combined_model', model_type, str(n_rows))
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        self.evaluation_path = os.path.join(base_dir, f'{game_type}_eval.csv')

    def load_xgb(self):
        '''
        loads the exported xgboost model, so it can be used for evaluation
        :return:
        '''
        with open(self.xbg_model_path, 'rb') as f:
            self.advanced_model = pickle.load(f)

    def median_abs_error(self, y_true, y_pred):
        '''
        A custom evaluation metric for the validation set
        :param y_true: true value
        :param y_pred: predicted value
        :return:
        '''
        abs_diff = abs(y_true - y_pred)
        # this is for the median
        return tfp.stats.percentile(abs_diff, 50.0, interpolation='midpoint')

    def load_lstm(self):
        '''
        loads the exported LSTM model, so it can be used for evaluation
        :return:
        '''
        self.advanced_model = load_model(self.lstm_model_path, custom_objects={'median_abs_error': self.median_abs_error})

    def load_linear_model(self):
        '''
        loads the exported linear model, so it can be used for evaluation
        :return:
        '''
        with open(self.linear_model_path, 'rb') as f:
            self.linear_model = pickle.load(f)

    def evaluate(self):
        '''
        evaluates the models on the testing data w.r.t. different evaluation metrics
        :return:
        '''
        # predict advanced model
        test_df = pd.read_csv(self.test_file, nrows=self.nrows, usecols=self.usefuls_cols)
        X_test = test_df.drop(columns=['average_elo'])
        y_test = test_df['average_elo']
        y_pred_advanced = self.advanced_model.predict(X_test)
        if self.model_type == 'lstm':
            y_pred_advanced = [i + self.train_mean for i in y_pred_advanced.flatten().tolist()]
        advanced_median_abs_error = np.median([abs(i - j) for i, j in zip(y_pred_advanced, y_test.tolist())])
        advanced_mean_abs_error = np.mean([abs(i - j) for i, j in zip(y_pred_advanced, y_test.tolist())])

        # predict Linear
        test_df = pd.read_csv(self.test_file, index_col=0, nrows=self.nrows)
        test_df = test_df[self.valid_columns]
        X_test = test_df.drop(columns=['average_elo'])
        y_test = test_df['average_elo']
        y_pred_linear = self.linear_model.predict(X_test)
        linear_median_abs_error = np.median([abs(i - j) for i, j in zip(y_pred_linear, y_test.tolist())])
        linear_mean_abs_error = np.mean([abs(i - j) for i, j in zip(y_pred_linear, y_test.tolist())])

        # combine both
        y_pred_combined = [0.5 * (i+j) for i,j in zip(y_pred_advanced, y_pred_linear)]
        combined_median_abs_error = np.median([abs(i - j) for i, j in zip(y_pred_combined, y_test.tolist())])
        combined_mean_abs_error = np.mean([abs(i - j) for i, j in zip(y_pred_combined, y_test.tolist())])

        # dummy guess
        dummy_guess = np.mean(y_test.tolist())
        median_absolute_dummy_error = np.median([abs(i - dummy_guess) for i in y_test.tolist()])
        mean_absolute_dummy_error = np.mean([abs(i - dummy_guess) for i in y_test.tolist()])

        self.eval_measures = {'Linear Median Absolute Error': linear_median_abs_error,
                              'Advanced Median Absolute Error': advanced_median_abs_error,
                              'Combined Median Absolute Error': combined_median_abs_error,
                              'Linear Mean Absolute Error': linear_mean_abs_error,
                              'Advanced Mean Absolute Error': advanced_mean_abs_error,
                              'Combined Mean Absolute Error': combined_mean_abs_error,
                              'Dummy Mean Absolute Error': mean_absolute_dummy_error,
                              'Dummy Median Absolute Error': median_absolute_dummy_error,
                              }

    def save_and_print_evaluation_outcome(self):
        '''
        exports the evaluation metrics and prints them to the console
        :return:
        '''
        measure_df = pd.DataFrame(self.eval_measures.items(), columns=['Measure', 'Value'])
        measure_df.to_csv(self.evaluation_path)
        for measure, value in self.eval_measures.items():
            print(f'{measure}: : {"".join([" " for i in range(40 - len(measure))])}  {value:.2f}')



if __name__ == '__main__':
    ##############################################
    ############## Train the models ##############
    ##############################################
    # Linear Regression:
    game_types = ['classical', 'bullet', 'rapid', 'blitz']
    for game_type in game_types:
        model = LinearRegressionModel(game_type, 1000000)
        model.train()
        model.save()
        model.load()
        model.evaluate()
        model.save_and_print_coefficients(verbose=False)
        print(f'Gametype: {game_type}, n_rows = {1000000}')
        model.save_and_print_evaluation_outcome()

    # XGBoost:
    for game_type in game_types:
        model = XGBoostRegression(game_type, 1000000, 60)
        model.train()
        model.save()
        model.load()
        model.evaluate()
        print(f'Gametype: {game_type}, n_rows = {1000000}')
        model.save_and_print_evaluation_outcome()

    # LSTM
    for game_type in game_types:
        model = LSTMRegression(game_type, 300000)
        model.train()
        model.save()
        model.load()
        model.evaluate()
        print(f'Gametype: {game_type}, n_rows: {300000}')
        model.save_and_print_evaluation_outcome()



    ########################################################
    ############## Evaluations on 1 Mio Games ##############
    ########################################################

    # Combined with XGB
    for game_type in game_types:
        model = CombinedModel(game_type, 1000000, 'xgboost')
        model.evaluate()
        print(f'Gametype: {game_type}, n_rows: {1000000}')
        model.save_and_print_evaluation_outcome()


    # Combined with LSTM
    for game_type in game_types:
        print(game_type)
        model = CombinedModel(game_type, 1000000, 'lstm')
        model.evaluate()
        print(f'Gametype: {game_type}, n_rows: {1000000}')
        model.save_and_print_evaluation_outcome()
