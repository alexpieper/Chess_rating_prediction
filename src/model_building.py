import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os


class LinearRegressionModel:
    def __init__(self):
        # train_file = os.path.join('data', 'processed', 'all_games_clean', 'classical_train_exp.csv')
        # test_file = os.path.join('data', 'processed', 'all_games_clean', 'classical_test_exp.csv')
        train_file = os.path.join('data', 'processed', 'all_games_clean', 'classical_train.csv')
        test_file = os.path.join('data', 'processed', 'all_games_clean', 'classical_test.csv')

        self.train_df = pd.read_csv(train_file, index_col = 0)
        self.test_df = pd.read_csv(test_file, index_col = 0)
        valid_columns = [i for i in self.train_df.columns if not ('eval_' in i or 'clock_' in i)]
        self.train_df = self.train_df[valid_columns]
        self.test_df = self.test_df[valid_columns]

        self.model = LinearRegression()

    def train(self):
        X_train = self.train_df.drop(columns=['average_elo'])
        y_train = self.train_df['average_elo']
        self.model.fit(X_train, y_train)

    def evaluate(self):
        X_test = self.test_df.drop(columns=['average_elo'])
        y_test = self.test_df['average_elo']
        y_pred = self.model.predict(X_test)

        median_abs_error = np.median([abs(i - j) for i,j in zip(y_pred, y_test.tolist())])
        mean_abs_error = np.mean([abs(i - j) for i,j in zip(y_pred, y_test.tolist())])
        median_squared_error = np.median([(i - j)**2 for i,j in zip(y_pred, y_test.tolist())])
        mean_squared_error = np.mean([(i - j)**2 for i,j in zip(y_pred, y_test.tolist())])
        self.eval_measures = {'Mean Squared Error': mean_squared_error, 'Median Squared Error': median_squared_error, 'Mean Absolute Error': mean_abs_error, 'Median Absolute Error': median_abs_error}

    def print_evaluation_outcome(self):
        for measure, value in self.eval_measures.items():
            print(f'{measure}: {value:.2f}')

    def print_coefficients(self):
        coefficients = self.model.coef_
        column_names = self.train_df.drop(columns=['average_elo']).columns

        for i, coef in enumerate(coefficients):
            print(f'{column_names[i]}: \t \t {coef:.2f}')


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
    model = LinearRegressionModel()
    model.train()
    model.evaluate()
    model.print_evaluation_outcome()
    model.print_coefficients()