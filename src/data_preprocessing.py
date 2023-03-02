import os
import chess.pgn
import re
import pandas as pd
import numpy as np
import tqdm
import multiprocessing
import matplotlib.pyplot as plt
from ast import literal_eval
import json
import time
import datetime
import random


def filter_and_preprocess_data():
    '''
    This function reads the list of PGN Games from Lichess.org (https://database.lichess.org) and preprocesses them
    after that we store them in a pandas dataframe, which we export in a batched manner, to save memory.
    Parameter:
        - we are only interested in the ones, that have clock AND Evaluation information and are not terminated
    Notes:
    current speed of the loop is ~ 600 iterations/second
    this runs for ~ 40 hrs per 90.000.000 games (all games played in November 2022 on lichess.org)
    raw data per month >200 GB
    after this processing function: ~15 GB per month

    this function has to be called once with the chess games from october, and once with the games from november.
    just uncomment the part behind # november: for the second run
    '''
    raw_data_folder = os.path.join('data', 'raw')
    
    # october:
    filename = os.path.join(raw_data_folder, 'lichess_db_standard_rated_2022-10.pgn')
    processed_data_folder = os.path.join('data', 'processed', 'batched_exports_2022_10')
    
    # november:
    # filename = os.path.join(raw_data_folder, 'lichess_db_standard_rated_2022-11.pgn')
    # processed_data_folder = os.path.join('data', 'processed', 'batched_exports_2022_11')

    game_metadata = []
    n = 92629656
    # for a smaller example:
    # n = 3000

    man_counter = 0
    export_counter = 0

    with open(filename, "r") as f:
        for i in tqdm.tqdm(range(n)):
            # reads the game and skip certain games
            game = chess.pgn.read_game(f)
            if game.headers['Termination'] == 'Abandoned':
                continue
            if len(game.variations) == 0:
                continue
            if game.variations[0].clock() is None or game.variations[0].eval() is None:
                continue


            # Split the PGN formatted game into lines
            TimeControl = game.headers['TimeControl']
            TimeControl_seconds = int(game.headers['TimeControl'].split('+')[0])
            TimeControl_increment = int(game.headers['TimeControl'].split('+')[1])
            BlackElo = game.headers['BlackElo']
            WhiteElo = game.headers['WhiteElo']
            Opening = game.headers['Opening']
            Opening_high_level = Opening.split(':')[0]
            Termination = game.headers['Termination']
            result = game.headers['Result']


            exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=True, columns=2)
            pgn_string = game.mainline_moves().accept(exporter)


            # extract the moves, evaluations, and clocks
            parsed_list = [i for i in pgn_string.split('\n') if not '$' in i]
            white_moves = parsed_list[1::6]
            black_moves = parsed_list[4::6]
            all_moves = parsed_list[1::3]
            try:
                all_clocks = [i.split('%clk ')[1].split(']')[0] for i in parsed_list[2::3]]
                all_evaluations = [i.split('%eval ')[1].split(']')[0] for i in parsed_list[2:-1:3]]
            except:
                continue

            numeric_evaluations = [i for i in all_evaluations if not '#' in i]
            numeric_evaluations = [float(i) for i in numeric_evaluations]
            evaluation_variance = np.var(numeric_evaluations)
            evaluation_iqr = np.percentile(numeric_evaluations, 75) - np.percentile(numeric_evaluations, 25)
            evaluation_range = np.max(numeric_evaluations) - np.min(numeric_evaluations)


            first_white_move = white_moves[0]
            first_black_move = black_moves[0]
            number_of_moves = len(all_moves)
            number_of_checks = len([i for i in all_moves if '+' in i])
            number_of_captures = len([i for i in all_moves if 'x' in i])
            white_castling = (1 if 'O-O' in white_moves else (2 if 'O-O-O' in white_moves else 0))
            black_castling = (1 if 'O-O' in black_moves else (2 if 'O-O-O' in black_moves else 0))


            knight_moves = len([i for i in all_moves if 'N' in i])
            bishop_moves = len([i for i in all_moves if 'B' in i])
            rook_moves = len([i for i in all_moves if 'R' in i])
            queen_moves = len([i for i in all_moves if 'Q' in i])
            king_moves = len([i for i in all_moves if 'K' in i])
            pawn_moves = len([i for i in all_moves if not any(string in i for string in ['N', 'B', 'R', 'Q', 'K', 'O'])])


            nags = [i for i in pgn_string.split('\n') if '$' in i]
            number_of_bad_moves = len([i for i in nags if '2' in i])
            number_of_blunders = len([i for i in nags if '4' in i])
            number_of_dubious_moves = len([i for i in nags if '6' in i])


            if Termination == 'Normal':
                if (result == '1-0') or (result == '0-1'):
                    if '#' in all_moves[-1]:
                        Termination = 'Checkmate'
                    else:
                        Termination = 'Resignation'
                if result == '1/2-1/2':
                    Termination = 'Stalemate'

            # add the extracted data to the list of games
            game_metadata.append({
                'black_elo': BlackElo,
                'white_elo': WhiteElo,
                'time_control': TimeControl,
                'time_control_seconds': TimeControl_seconds,
                'time_control_increment': TimeControl_increment,
                'termination': Termination,
                'result':result,
                'opening': Opening,
                'opening_high_level': Opening_high_level,
                'first_white_move': first_white_move,
                'first_black_move': first_black_move,
                'number_of_moves': number_of_moves,
                'number_of_checks': number_of_checks,
                'white_castling': white_castling,
                'black_castling': black_castling,
                'number_of_captures': number_of_captures,
                'knight_moves': knight_moves,
                'bishop_moves': bishop_moves,
                'rook_moves': rook_moves,
                'queen_moves': queen_moves,
                'king_moves': king_moves,
                'pawn_moves': pawn_moves,
                'number_of_blunders': number_of_blunders,
                'number_of_bad_moves': number_of_bad_moves,
                'number_of_dubious_moves': number_of_dubious_moves,
                'evaluation_variance': evaluation_variance,
                'evaluation_iqr': evaluation_iqr,
                'evaluation_range': evaluation_range,
                'all_moves': all_moves,
                'all_evaluations': all_evaluations,
                'all_clocks':all_clocks
            })

            # export the data every 100000 th successfully parsed game, to save some memory
            man_counter += 1
            if man_counter in [j for j in range(0,n,50000)]:
                export_counter += 1
                df = pd.DataFrame(game_metadata)

                # split into the 4 categories, based on the time
                bullet = df[df['time_control_seconds'] < 180]
                blitz = df[(df['time_control_seconds'] >= 180) & (df['time_control_seconds'] < 600)]
                rapid = df[(df['time_control_seconds'] >= 600) & (df['time_control_seconds'] < 1800)]
                classical = df[df['time_control_seconds'] >= 1800]

                bullet_file = os.path.join(processed_data_folder, f'bullet_{export_counter}.csv')
                blitz_file = os.path.join(processed_data_folder, f'blitz_{export_counter}.csv')
                rapid_file = os.path.join(processed_data_folder, f'rapid_{export_counter}.csv')
                classical_file = os.path.join(processed_data_folder, f'classical_{export_counter}.csv')

                bullet.to_csv(bullet_file)
                blitz.to_csv(blitz_file)
                rapid.to_csv(rapid_file)
                classical.to_csv(classical_file)

                game_metadata = []
                df = pd.DataFrame()


def clean_batched_files():
    '''
    This function gets rid of useless data and transforms the existing data.
    It encodes the categroical features (i.e. with one-hot-encoding) (termination, opening, opneing_high_level, black_first_move, white_first_move)
    and encode the lists into numerical features, using padding with 0
    and translates from evaluation from #3 into a number
    also encode the clock from 'HH:MM:SS:' into 'seconds_used_for_move'
    :return:
    '''
    processed_data_folder_november = os.path.join('data', 'processed', 'batched_exports_2022_11')
    processed_data_folder_october = os.path.join('data', 'processed', 'batched_exports_2022_10')
    batched_cleaned_folder = os.path.join('data', 'processed', 'clean_batched_exports')
    
    # this limits the amount of moves we take into consideration, based on the game category
    # this is fitted to a histogram of the number of moves
    limit_map = {
        'classical': {'upper': 100,
                      'lower': 30},
        'bullet': {'upper': 80,
                   'lower': 30},
        'rapid': {'upper': 100,
                  'lower': 30},
        'blitz': {'upper': 100,
                  'lower': 30},
    }

    # this is the number of batch exports
    n_november = 150
    n_october = 160

    # this is for smaller export
    # n_november = 10
    # n_october = 10

    # to save memory, we make first all the classicla, then rapid etc.
    for category in ['classical', 'bullet', 'rapid', 'blitz']:
        for export_counter in tqdm.tqdm(range(1,n_november + n_october)):
            if export_counter > n_november:
                classical_file = os.path.join(processed_data_folder_october, f'{category}_{export_counter - n_november + 1}.csv')
            else:
                classical_file = os.path.join(processed_data_folder_november, f'{category}_{export_counter}.csv')
            all_games = pd.read_csv(classical_file, index_col = 0, converters = {'all_evaluations': literal_eval, 'all_clocks': literal_eval})


            #############################
            # Target variable and cleaning of unnessecary data
            # we only want games, that have a maximum elo difference of 200
            all_games['elo_diff'] = abs(all_games['black_elo'] - all_games['white_elo'])
            all_games = all_games[all_games['elo_diff'] <= 200]
            
            # this here was used to get the histograms, to get the limit_map
            # histogram of the number of moves, to get a good sense
            # plt.hist(all_games['number_of_moves'], bins = 20)
            # plt.savefig(f'{category}.png')
            # plt.close()

            # this filters the number of games, based on the moves
            all_games = all_games[(all_games['number_of_moves'] <= limit_map[category]['upper']) & (all_games['number_of_moves'] <= limit_map[category]['upper'])]
            # this "average_elo" is going to be the target variable
            all_games['average_elo'] = 0.5 * (all_games['black_elo'] + all_games['white_elo'])
            all_games = all_games[['average_elo', 'termination', 'opening',
                       'opening_high_level', 'time_control_increment','first_white_move', 'first_black_move',
                       'number_of_moves', 'number_of_checks', 'white_castling',
                       'black_castling', 'number_of_captures', 'knight_moves', 'bishop_moves',
                       'rook_moves', 'queen_moves', 'king_moves', 'pawn_moves',
                       'number_of_blunders', 'number_of_bad_moves', 'number_of_dubious_moves',
                       'evaluation_variance', 'evaluation_iqr', 'evaluation_range',
                       'all_moves', 'all_evaluations', 'all_clocks']]


            #############################
            # ONE HOT:
            for categorical in ['first_black_move', 'first_white_move', 'opening_high_level', 'termination', 'opening']:
                one_hot_df = pd.get_dummies(all_games[categorical])
                if categorical == 'opening_high_level':
                    all_games = all_games.join(one_hot_df.add_suffix('_high_level'))
                else:
                    all_games = all_games.join(one_hot_df)


            # FREQUENCY Encoding, (not used):
            # this is def. wrong as the mapping in each batch will be different
            # for categorical in ['first_black_move', 'first_white_move', 'termination', 'opening']:
            #     freq = all_games[categorical].value_counts()
            #     # map the categories to their frequencies
            #     all_games[categorical] = all_games[categorical].map(freq)

            #############################
            eval_columns = [f'eval_{i}' for i in range(limit_map[category]['upper'])]
            clock_columns = [f'clock_{i}' for i in range(limit_map[category]['upper'])]
            eval_lists = []
            clock_lists = []

            # here we add padding to the timeseries
            for evals in all_games['all_evaluations']:
                parsed_eval = [-40 if '#-' in i else (40 if '#' in i else (40 if float(i) > 40 else (-40 if float(i) < -40 else float(i)))) for i in evals]
                eval_lists += [parsed_eval + [0.0 for i in range(limit_map[category]['upper'] - len(parsed_eval))]]
            for (clocks, increment) in zip(all_games['all_clocks'], all_games['time_control_increment']):
                # the datetime module was too slow
                timestamps_dt = [int(ts[5:7]) + 60*int(ts[2:4]) + 3600*int(ts[0]) for ts in clocks]
                parsed_clock = [int(t1 - t2 + int(increment)) for t1, t2 in zip(timestamps_dt[:-1], timestamps_dt[2:])]
                clock_lists += [parsed_clock + [0 for i in range(limit_map[category]['upper'] - len(parsed_clock))]]

            eval_df = pd.DataFrame(eval_lists, columns = eval_columns)
            clock_df = pd.DataFrame(clock_lists, columns = clock_columns)

            # drop unnessecary columns
            all_games = all_games.drop(['time_control_increment', 'first_black_move', 'first_white_move', 'all_moves', 'all_evaluations', 'all_clocks', 'opening_high_level', 'termination', 'opening'], axis = 1)
            all_games = pd.concat([all_games.reset_index(drop=True), eval_df, clock_df], axis = 1)

            # here we export it into the cleanest file
            all_games.to_csv(os.path.join(batched_cleaned_folder, f'{category}_{export_counter}.csv'))


def concat_clean_batched_files():
    '''
    this function concatenates the batched exports, splits into training and testing and export those as large .csv files.
    also the conversion from int64 to int8/16 was applied to save ~ 50% of storage
    '''
    batched_cleaned_folder = os.path.join('data', 'processed', 'clean_batched_exports')
    all_games_cleaned_folder = os.path.join('data', 'processed', 'all_games_clean')
    
    # there are 309 files per category
    n_total = 309
    for category in ['classical', 'bullet', 'rapid', 'blitz']:
        all_games_list = []
        for export_counter in tqdm.tqdm(range(1,n_total)):
            classical_file = os.path.join(batched_cleaned_folder, f'{category}_{export_counter}.csv')
            game_tmp = pd.read_csv(classical_file, index_col=0)
            # efficiency gain thorugh the following conversions: 1.45 GB = 787 MB
            int_16_columns = [i for i in game_tmp.columns if 'clock_' in i] + ['number_of_moves', 'number_of_checks', 'white_castling', 'black_castling', 'number_of_captures', 'knight_moves', 'bishop_moves','rook_moves', 'queen_moves', 'king_moves', 'pawn_moves','number_of_blunders','number_of_bad_moves', 'number_of_dubious_moves']
            int_8_columns = [i for i in game_tmp.columns if ((i not in ['average_elo', 'evaluation_variance', 'evaluation_iqr', 'evaluation_range']) and (not i in int_16_columns) and (not 'eval_' in i))]
            game_tmp[int_16_columns] = game_tmp[int_16_columns].astype('Int16')
            game_tmp[int_8_columns] = game_tmp[int_8_columns].astype('Int8')
            all_games_list += [game_tmp]
        all_games = pd.concat(all_games_list)

        # some opening_columns are not present in other batches, therefore filling them with 0
        print(all_games.shape)
        all_games = all_games.fillna(0)
        all_games.index = range(all_games.shape[0])

        # make the train test split
        print('randomized indices')
        indices = all_games.index.tolist()
        random.seed(42)
        random.shuffle(indices)

        # i made it 60%, because the data is already soo large, the smaller the train set, the better for computational effort
        print('applying indices')
        split_index = int(len(indices) * (0.6))
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]
        train = all_games.loc[train_indices,:]
        test = all_games.loc[test_indices,:]

        print('start exporting train')
        # the exportin consumed 150 GB of RAM, therefore we export in batches
        train.to_csv(os.path.join(all_games_cleaned_folder, f'{category}_train.csv'), chunksize = 50000)
        print('start exporting test')
        test.to_csv(os.path.join(all_games_cleaned_folder, f'{category}_test.csv'), chunksize = 50000)



if __name__ == '__main__':
    # remember: do this once with october games and once with the november games
    filter_and_preprocess_data()
    clean_batched_files()
    concat_clean_batched_files()
