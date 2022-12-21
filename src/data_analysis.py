import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
from scipy import stats
import ast



from matplotlib.colors import LinearSegmentedColormap

class DataAnalysis:
    '''
    This class is supposed to create a bunch of descriptive statistics and
    TODO: add an overview of the number of games we have
    TODO: add a viz of all the datapoints we have per game
    '''
    def __init__(self):
        # self.data_file = os.path.join('data', 'processed', 'blitz.csv')
        self.data_file = os.path.join('data', 'processed', 'all_games_clean', 'classical_train.csv')
        self.plot_export_folder = os.path.join('plots')
        self.stats_export_folder = os.path.join('evaluations')
        self.data = pd.read_csv(self.data_file, index_col = 0)


        # some more feature engineering and filtering
        print(self.data.shape)
        self.data['blunder_per_move'] = self.data['number_of_blunders'] /self.data['number_of_moves']
        print(self.data.shape)





    def run_everything(self):
        self.histogram_of_rating()
        self.plot_numerical_variables()
        self.plot_rating_vs_all()
        self.plot_timeseries_vs_rating()

    def histogram_of_rating(self):
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        ax.hist(self.data['average_elo'], edgecolor = 'black', bins = 15)
        plt.xlabel('Rating')
        plt.ylabel('Absolute Frequency')
        ax.set_title('Histogram of the Rating')
        fig.tight_layout()
        export_file = os.path.join(self.plot_export_folder, 'rating_histogram.png')
        plt.savefig(export_file)


    def plot_numerical_variables(self):
        numerical_data = self.data[['number_of_checks', 'white_castling', 'black_castling', 'number_of_captures', 'knight_moves', 'bishop_moves', 'rook_moves', 'queen_moves', 'king_moves', 'pawn_moves', 'number_of_blunders', 'number_of_bad_moves', 'number_of_dubious_moves', 'evaluation_variance', 'evaluation_iqr', 'evaluation_range', 'average_elo']]
        numerical_data = self.data[['number_of_blunders', 'average_elo', 'number_of_moves','evaluation_variance']]
        print(numerical_data)
        g = sns.PairGrid(numerical_data)
        g.map_diag(sns.histplot)
        g.map_offdiag(sns.scatterplot)
        export_file = os.path.join(self.plot_export_folder, 'numerical_values.png')
        plt.savefig(export_file)


    def plot_rating_vs_all(self):
        # this was using the data pre one-hot encoding, therefore not working anymore
        return
        fig, ax = plt.subplots(5, 4, figsize=(20, 15))

        columns_of_interest = ['opening',
                           'opening_high_level', 'first_white_move', 'first_black_move',
                           'number_of_moves', 'number_of_checks','black_castling',
                            'number_of_captures', 'knight_moves', 'bishop_moves',
                           'rook_moves', 'queen_moves', 'king_moves', 'pawn_moves',
                           'number_of_blunders', 'number_of_bad_moves', 'number_of_dubious_moves',
                           'evaluation_variance', 'evaluation_iqr', 'evaluation_range']
        row = 0
        col = 0
        for column in columns_of_interest:
            ax[row, col].scatter(self.data['average_elo'], self.data[column], alpha = 0.1, s = 5)
            if column == 'opening' or column == 'opening_high_level':
                ax[row, col].set_yticklabels([])
            ax[row,col].set_ylabel(column)
            ax[row,col].set_xlabel('Rating')
            col += 1
            if col > 3:
                col = 0
                row += 1

        fig.tight_layout()
        export_file = os.path.join(self.plot_export_folder, 'all_vs_rating.png')
        plt.savefig(export_file)


    def plot_timeseries_vs_rating(self):
        colors = ['#fa6e6e', '#e76e57', '#d36f43', '#bd7033', '#a67027', '#8f6e1f', '#796c1b', '#64681d', '#506421', '#3d5e26', '#2a582b']

        base = 1
        self.data['rating_bucket'] = self.data['average_elo'].apply(lambda x: int(base * round(stats.percentileofscore(self.data['average_elo'], x, 'rank')/base)/ 1))
        print(self.data['rating_bucket'].unique().tolist())


        # print(self.data)
        fig, ax = plt.subplots(2,1, figsize=(12, 9), sharex=True)
        for index, row in self.data.iloc[:3000].iterrows():
            if (row['rating_bucket'] > 1) and (row['rating_bucket'] < 99):
                continue
            # evals = [float(i) for i in row[[f'eval_{j}']) if not '#' in i]
            evals = row[[f'eval_{j}' for j in range(100)]]
            # delete the padded 0'es
            for i in range(len(evals) - 1, -1, -1):
                if evals[i] != 0:
                    break
            evals = evals[:i+1]
            # TODO: remove once other script ran through
            # evals = [40 if int(i) == 100 else (-40 if int(i) == -100 else i) for i in evals]
            # here we want to strip the padded 0's away
            if row['rating_bucket'] <= 1:
                ax[0].plot(range(len(evals)), evals, alpha = 0.4, linewidth=1, color = 'tab:red')
            else:
                ax[1].plot(range(len(evals)), evals, alpha = 0.4, linewidth=1, color = 'tab:green')
        ax[0].set_ylim([-50,50])
        ax[0].set_xlim([0, 60])
        ax[1].set_ylim([-50, 50])
        ax[1].set_xlim([0, 60])
        ax[0].set_ylabel('Evaluation')
        ax[1].set_ylabel('Evaluation')
        ax[1].set_xlabel('Move Number')
        ax[0].set_title('Evaluation Timeseries over lowest rated 1% of games')
        ax[1].set_title('Evaluation Timeseries over highest rated 1% of games')


        fig.tight_layout()
        export_file = os.path.join(self.plot_export_folder, 'eval_timeseries.png')
        plt.savefig(export_file)

        fig, ax = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        for index, row in self.data.iloc[:20000].iterrows():
            # only use longer games, to make the difference more obvious
            if row['number_of_moves'] < 80:
                continue
            if (row['rating_bucket'] > 1) and (row['rating_bucket'] < 99):
                continue
            # evals = [float(i) for i in row[[f'eval_{j}']) if not '#' in i]
            evals = row[[f'clock_{j}' for j in range(100)]]
            # delete the padded 0'es
            for i in range(len(evals) - 1, -1, -1):
                if evals[i] != 0:
                    break
            evals = evals[:i + 1]
            # here we want to strip the padded 0's away
            if row['rating_bucket'] <= 1:
                ax[0].plot(range(len(evals)), list(np.cumsum(evals)), alpha=0.4, linewidth=1, color='tab:red')
            else:
                ax[1].plot(range(len(evals)), list(np.cumsum(evals)), alpha=0.4, linewidth=1, color='tab:green')
        ax[0].set_ylim([0, 5000])
        ax[0].set_xlim([0, 100])
        ax[1].set_ylim([0, 5000])
        ax[1].set_xlim([0, 100])
        ax[0].set_ylabel('Cumulative time spent')
        ax[1].set_ylabel('Cumulative time spent')
        ax[1].set_xlabel('Move Number')
        ax[0].set_title('Cumulative time spent over lowest rated 1% of games')
        ax[1].set_title('Cumulative time spent over highest rated 1% of games')

        fig.tight_layout()
        export_file = os.path.join(self.plot_export_folder, 'clock_timeseries.png')
        plt.savefig(export_file)


if __name__ == '__main__':
    data_analysis = DataAnalysis()
    data_analysis.run_everything()
    print('Here')