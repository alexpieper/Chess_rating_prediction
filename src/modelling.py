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
        try:
            with open(self.valid_columns_file) as file:
                self.valid_columns  = [line.rstrip() for line in file]
        except:
            self.valid_columns = []

        if load_new:
            sums = df.sum(axis=0)
            self.valid_columns = [i for i in df.columns if (abs(sums[i]) > 10) and not (('eval_' in i or 'clock_' in i))]
            df = df[self.valid_columns]
            with open(self.valid_columns_file, 'w') as f:
                for line in self.valid_columns:
                    f.write(f"{line}\n")
        else:
            df = df[self.valid_columns]
        return df

    def train(self):
        print('reading train')
        self.train_df = pd.read_csv(self.train_file, index_col=0, nrows=self.nrows)
        self.train_df = self.clean_dataset(self.train_df, True)
        X_train = self.train_df.drop(columns=['average_elo'])
        y_train = self.train_df['average_elo']
        print('start training')
        self.model.fit(X_train, y_train)
        self.train_df = pd.DataFrame()
        X_train = pd.DataFrame()
        y_train = pd.DataFrame()
        print('finished training')

    def save(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self):
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

    def evaluate(self):
        print('reading test')
        self.test_df = pd.read_csv(self.test_file, index_col=0, nrows=self.nrows)
        self.test_df = self.clean_dataset(self.test_df, False)
        print('starting eval')
        # self.valid_columns = [i for i in self.test_df.columns if not ('eval_' in i or 'clock_' in i)]
        self.test_df = self.test_df[self.valid_columns]
        X_test = self.test_df.drop(columns=['average_elo'])
        y_test = self.test_df['average_elo']
        y_pred = self.model.predict(X_test)

        median_abs_error = np.median([abs(i - j) for i,j in zip(y_pred, y_test.tolist())])
        mean_abs_error = np.mean([abs(i - j) for i,j in zip(y_pred, y_test.tolist())])
        median_squared_error = np.median([(i - j)**2 for i,j in zip(y_pred, y_test.tolist())])
        mean_squared_error = np.mean([(i - j)**2 for i,j in zip(y_pred, y_test.tolist())])
        dummy_guess = np.mean(y_test.tolist())
        median_absolute_dummy_error = np.median([abs(i - dummy_guess) for i in y_test.tolist()])
        mean_absolute_dummy_error = np.mean([abs(i - dummy_guess) for i in y_test.tolist()])
        median_squared_dummy_error = np.median([(i - dummy_guess)**2 for i in y_test.tolist()])
        mean_squared_dummy_error = np.mean([(i - dummy_guess)**2 for i in y_test.tolist()])

        # todo: add benchmark: always to guess the median.
        self.eval_measures = {'Mean Squared Error': mean_squared_error, 'Median Squared Error': median_squared_error,
                              'Mean Absolute Error': mean_abs_error, 'Median Absolute Error': median_abs_error,
                              'Mean Squared Dummy Error': mean_squared_dummy_error, 'Median Squared Dummy Error': median_squared_dummy_error,
                              'Mean Absolute Dummy Error': mean_absolute_dummy_error, 'Median Absolute Dummy Error': median_absolute_dummy_error
                              }

    def save_and_print_evaluation_outcome(self):
        measure_df = pd.DataFrame(self.eval_measures.items(), columns=['Measure', 'Value'])
        measure_df.to_csv(self.evaluation_path)
        for measure, value in self.eval_measures.items():
            print(f'{measure}: {value:.2f}')

    def save_and_print_coefficients(self, verbose = False):
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
    def __init__(self, game_type, n_rows):
        print(game_type)
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
        abs_diff = abs(y_true - y_pred)
        # this is for the median
        return tfp.stats.percentile(abs_diff, 50.0, interpolation='midpoint')

    def train(self):
        epochs = 250
        batch_size = 256

        print('reading train')
        self.train_df = pd.read_csv(self.train_file, nrows=self.nrows, usecols=self.usefuls_cols)
        self.validation_df = self.train_df[int(self.nrows/1.5):]
        self.train_df = self.train_df[:int(self.nrows/1.5)]
        X_train = self.train_df.drop(columns=['average_elo'])
        X_validation = self.validation_df.drop(columns=['average_elo'])
        y_train = self.train_df['average_elo']
        y_validation = self.validation_df['average_elo']
        self.train_mean = np.mean(y_train)
        print(self.train_mean)

        err
        y_train = y_train - self.train_mean
        y_validation = y_validation - self.train_mean
        print('start training')
        input_shape = (X_train.shape[1],1)
        output_shape = 1

        self.model = Sequential()
        self.model.add(LSTM(units=32, input_shape=input_shape))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(output_shape))
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=[self.median_abs_error])
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_validation, y_validation))
        self.train_df = pd.DataFrame()
        X_train = pd.DataFrame()
        y_train = pd.DataFrame()

    def evaluate(self):
        print('reading test')
        self.test_df = pd.read_csv(self.test_file, nrows=self.nrows, usecols=self.usefuls_cols)
        print('starting eval')
        X_test = self.test_df.drop(columns=['average_elo'])
        y_test = self.test_df['average_elo']
        y_pred = self.model.predict(X_test)
        y_pred = [i + self.train_mean for i in y_pred.flatten().tolist()]
        median_abs_error = np.median([abs(i - j) for i, j in zip(y_pred, y_test.tolist())])
        mean_abs_error = np.mean([abs(i - j) for i, j in zip(y_pred, y_test.tolist())])
        dummy_guess = np.mean(y_test.tolist())
        median_absolute_dummy_error = np.median([abs(i - dummy_guess) for i in y_test.tolist()])
        mean_absolute_dummy_error = np.mean([abs(i - dummy_guess) for i in y_test.tolist()])

        # todo: add benchmark: always to guess the median.
        self.eval_measures = {'Mean Absolute Error': mean_abs_error, 'Median Absolute Error': median_abs_error,
                              'Mean Absolute Dummy Error': mean_absolute_dummy_error,
                              'Median Absolute Dummy Error': median_absolute_dummy_error
                              }

    def save_and_print_evaluation_outcome(self):
        measure_df = pd.DataFrame(self.eval_measures.items(), columns=['Measure', 'Value'])
        measure_df.to_csv(self.evaluation_path)
        for measure, value in self.eval_measures.items():
            print(f'{measure}: {value:.2f}')

    def save(self):
        self.model.save(self.model_path)

    def load(self):
        self.model = load_model(self.model_path, custom_objects = {'median_abs_error': self.median_abs_error})




class XGBoostRegression:
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
        print('reading train')
        self.train_df = pd.read_csv(self.train_file, nrows=self.nrows, usecols = self.usefuls_cols)
        X_train = self.train_df.drop(columns=['average_elo'])
        y_train = self.train_df['average_elo']
        print('start training')
        self.model.fit(X_train, y_train)
        self.train_df = pd.DataFrame()
        X_train = pd.DataFrame()
        y_train = pd.DataFrame()
        print('finished training')

    def save(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self):
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

    def evaluate(self):
        print('reading test')
        self.test_df = pd.read_csv(self.test_file, nrows=self.nrows, usecols = self.usefuls_cols)
        self.test_df = self.clean_dataset(self.test_df, False)
        print('starting eval')
        self.test_df = self.test_df[self.valid_columns]
        X_test = self.test_df.drop(columns=['average_elo'])
        y_test = self.test_df['average_elo']
        y_pred = self.model.predict(X_test)

        median_abs_error = np.median([abs(i - j) for i,j in zip(y_pred, y_test.tolist())])
        mean_abs_error = np.mean([abs(i - j) for i,j in zip(y_pred, y_test.tolist())])
        dummy_guess = np.mean(y_test.tolist())
        median_absolute_dummy_error = np.median([abs(i - dummy_guess) for i in y_test.tolist()])
        mean_absolute_dummy_error = np.mean([abs(i - dummy_guess) for i in y_test.tolist()])

        # todo: add benchmark: always to guess the median.
        self.eval_measures = {'Mean Absolute Error': mean_abs_error, 'Median Absolute Error': median_abs_error,
                              'Mean Absolute Dummy Error': mean_absolute_dummy_error, 'Median Absolute Dummy Error': median_absolute_dummy_error
                              }

    def save_and_print_evaluation_outcome(self):
        measure_df = pd.DataFrame(self.eval_measures.items(), columns=['Measure', 'Value'])
        measure_df.to_csv(self.evaluation_path)
        for measure, value in self.eval_measures.items():
            print(f'{measure}: {value:.2f}')




class CombinedModel():
    '''
    Here we want to load two trained models (one linear regression, one XGBoost), average their predicted values for the test set and see if that is better than either model.
    limit 60 is good
    '''
    def __init__(self, game_type, n_rows, model_type):


        self.test_file = os.path.join('data', 'processed', 'all_games_clean', f'{game_type}_test.csv')

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
        with open(self.xbg_model_path, 'rb') as f:
            self.advanced_model = pickle.load(f)

    def median_abs_error(self, y_true, y_pred):
        abs_diff = abs(y_true - y_pred)
        # this is for the median
        return tfp.stats.percentile(abs_diff, 50.0, interpolation='midpoint')

    def load_lstm(self):
        self.advanced_model = load_model(self.lstm_model_path, custom_objects={'median_abs_error': self.median_abs_error})

    def load_linear_model(self):
        with open(self.linear_model_path, 'rb') as f:
            self.linear_model = pickle.load(f)

    def evaluate(self):
        # predict advanced model
        test_df = pd.read_csv(self.test_file, nrows=self.nrows, usecols=self.usefuls_cols)
        print(test_df.shape)
        errr
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
        measure_df = pd.DataFrame(self.eval_measures.items(), columns=['Measure', 'Value'])
        measure_df.to_csv(self.evaluation_path)
        for measure, value in self.eval_measures.items():
            print(f'{measure}: : {"".join([" " for i in range(40 - len(measure))])}  {value:.2f}')



if __name__ == '__main__':
    '''
    Ideas:
    maybe drop the columns, wiht the opening that are super rare (like <10 in the whole dataset)
    #TODOs:
     - add mechanism to save() and load() models, and save and load their evaluations.
     - add excel export for the coefficients
     - add LSTM Model (with everything)
     - add LSTM Model (only with the timeseries)
     - make combination model of lstm + Linear regression
     
    '''
    # Linear Regression:
    # game_types = ['classical', 'bullet', 'rapid', 'blitz']
    # for game_type in game_types:
    #     model = LinearRegressionModel(game_type, 1000000)
    #     model.train()
    #     model.save()
    #     model.load()
    #     model.evaluate()
    #     model.save_and_print_coefficients(verbose=False)
    #     print(f'Gametype: {game_type}')
    #     model.save_and_print_evaluation_outcome()

    # XGBoost:
    # game_types = ['classical', 'bullet', 'rapid', 'blitz']
    # for game_type in game_types:
    #     model = XGBoostRegression(game_type, 1000000, 60)
    #     model.train()
    #     model.save()
    #     model.load()
    #     model.evaluate()
    #     print(f'Gametype: {game_type}, limit = {limit}')
#         model.save_and_print_evaluation_outcome()

    # LSTM
    # game_types = ['classical', 'bullet', 'rapid', 'blitz']
    # n_rows = 300000
    # for game_type in game_types:
    #     model = LSTMRegression(game_type, n_rows)
    #     model.train()
    #     model.save()
    #     model.load()
    #     model.evaluate()
    #     print(f'Gametype: {game_type}, n_rows: {n_rows}')
    #     model.save_and_print_evaluation_outcome()
    #     # thes are f


    # classical, 300000 nrows, 200 epochs
    # bullet, 300000 nrows, 250 epochs
    # rapid, 300000 nrows, 200 epochs
    # blitz, 300000 nrows, 250 epochs

    ########################################################
    ############## Evaluations on 1 Mio Games ##############
    ########################################################

    #
    # Combined with XGB
    game_types = ['classical', 'bullet', 'rapid', 'blitz']
    for game_type in game_types:
        model = CombinedModel(game_type, 1000000, 'xgboost')
        model.evaluate()
        model.save_and_print_evaluation_outcome()


    # Combined with LSTM

    game_types = ['classical', 'bullet', 'rapid', 'blitz']
    for game_type in game_types:
        print(game_type)
        model = CombinedModel(game_type, 1000000, 'lstm')
        model.evaluate()
        model.save_and_print_evaluation_outcome()
