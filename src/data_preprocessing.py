import os
import chess.pgn
import re
import pandas as pd
import numpy as np
import tqdm
import multiprocessing
import matplotlib.pyplot as plt
from ast import literal_eval
import time
import datetime


def filter_and_preprocess_data():
    '''
    current speed of the loop is ~ 600 iterations/second
    this runs for ~ 40 hrs per 90.000.000 games (all games played in November 2022 on lichess.org)
    :return:
    raw data per month >200 GB
    after processing (this function): 15 GB per month
    '''
    raw_data_folder = os.path.join('data', 'raw')
    processed_data_folder = os.path.join('data', 'processed')
    filename = os.path.join(raw_data_folder, 'lichess_db_standard_rated_2022-10.pgn')

    game_metadata = []
    n = 92629656
    man_counter = 0
    export_counter = 0
    # n = 3000

    with open(filename, "r") as f:
        for i in tqdm.tqdm(range(n)):
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

            # export the data every 100000 th successfully parsed game
            # after that, i should have cleared the game_metadata list and export with an index and later on concat the stuff
            man_counter += 1
            if man_counter in [j for j in range(0,n,50000)]:
                export_counter += 1
                df = pd.DataFrame(game_metadata)
                bullet = df[df['time_control_seconds'] < 180]
                blitz = df[(df['time_control_seconds'] >= 180) & (df['time_control_seconds'] < 600)]
                rapid = df[(df['time_control_seconds'] >= 600) & (df['time_control_seconds'] < 1800)]
                classical = df[df['time_control_seconds'] >= 1800]

                bullet_file = os.path.join(processed_data_folder, 'batched_exports_2022_10', f'bullet_{export_counter}.csv')
                blitz_file = os.path.join(processed_data_folder, 'batched_exports_2022_10', f'blitz_{export_counter}.csv')
                rapid_file = os.path.join(processed_data_folder, 'batched_exports_2022_10', f'rapid_{export_counter}.csv')
                classical_file = os.path.join(processed_data_folder, 'batched_exports_2022_10', f'classical_{export_counter}.csv')

                bullet.to_csv(bullet_file)
                blitz.to_csv(blitz_file)
                rapid.to_csv(rapid_file)
                classical.to_csv(classical_file)

                game_metadata = []
                df = pd.DataFrame()


def concat_batched_files():
    '''
    This function should get rid of useless data,
    first concatenate the batched exports.
    Then encode the categroical features (i.e. with one-hot-encoding) (termination, opening, opneing_high_level, black_first_move, white_first_move)
    and encode the lists into numerical features, using padding and translations from #3 into a number
    also encode the clock as in 'seconds_used'
    :return:
    '''
    processed_data_folder_november = os.path.join('data', 'processed', 'batched_exports_2022_11')
    processed_data_folder_october = os.path.join('data', 'processed', 'batched_exports_2022_10')
    all_games_cleaned_folder = os.path.join('data', 'processed', 'all_games_cleaned')
    # TODO: adjust these values, based on histograms of the umber of moves
    limit_map = {
        'classical': {'upper': 100,
                      'lower': 32},
        'bullet': {'upper': 100,
                   'lower': 32},
        'rapid': {'upper': 100,
                  'lower': 32},
        'blitz': {'upper': 100,
                  'lower': 32},
    }
    # to save memory, we make first all the classicla, then rapid etc.
    n_november = 151
    n_october = 161
    n_november = 10
    n_october = 10


    # for category in ['classical', 'bullet', 'rapid', 'blitz']:
    for category in ['classical']:
        all_games = pd.DataFrame()
        for export_counter in range(1,n_november):
            classical_file = os.path.join(processed_data_folder_november, f'{category}_{export_counter}.csv')
            if all_games.empty:
                all_games = pd.read_csv(classical_file, index_col = 0)
            else:
                all_games = pd.concat([all_games, pd.read_csv(classical_file, index_col = 0)], axis=0)
            print(all_games.shape)

        for export_counter in range(1, n_october):
            classical_file = os.path.join(processed_data_folder_october, f'{category}_{export_counter}.csv')
            all_games = pd.concat([all_games, pd.read_csv(classical_file, index_col=0)], axis=0)
            print(all_games.shape)

        # all_games.to_csv(os.path.join(all_games_raw_folder, f'{category}.csv'))




        # all_games = pd.read_csv(os.path.join(all_games_raw_folder, f'{category}.csv'), index_col = 0)
        print(all_games.columns)
        print(all_games.iloc[0])
        #############################
        # Target variable and cleaning of unnessecary data
        all_games['elo_diff'] = abs(all_games['black_elo'] - all_games['white_elo'])
        print(all_games.shape)
        # plt.hist(all_games['number_of_moves'],bins = 20 )
        # plt.show()
        # this number should change from category to category
        all_games = all_games[(all_games['number_of_moves'] <= limit_map[category]['upper']) & (all_games['number_of_moves'] <= limit_map[category]['upper'])]
        all_games = all_games[all_games['elo_diff'] <= 200]
        all_games['average_elo'] = 0.5 * (all_games['black_elo'] + all_games['white_elo'])
        # we only want games, that have a maximum elo difference of 200
        all_games = all_games[['average_elo', 'termination', 'opening',
                   'opening_high_level', 'first_white_move', 'first_black_move',
                   'number_of_moves', 'number_of_checks', 'white_castling',
                   'black_castling', 'number_of_captures', 'knight_moves', 'bishop_moves',
                   'rook_moves', 'queen_moves', 'king_moves', 'pawn_moves',
                   'number_of_blunders', 'number_of_bad_moves', 'number_of_dubious_moves',
                   'evaluation_variance', 'evaluation_iqr', 'evaluation_range',
                   'all_moves', 'all_evaluations', 'all_clocks']]


        #############################
        # encoding of categories
        # problem: dropping the low level opening, results in a massive loss of explainability, but that would add 1100 columns of data
        # one hot encoding is not an option
        # try binary encoding for these, or frequency encoding

        # ONE HOT:
        # for categorical in ['opening', 'first_white_move', 'first_black_move']:
        start = time.time()
        for categorical in ['first_black_move', 'first_white_move']:
            one_hot_df = pd.get_dummies(all_games[categorical])
            all_games = all_games.join(one_hot_df)
        end = time.time()
        print(start - end)

        # FREQUENCY:
        # for categorical in ['first_black_move', 'first_white_move', 'termination', 'opening']:
        start = time.time()
        for categorical in ['termination', 'opening', 'opening_high_level']:
            freq = all_games[categorical].value_counts()
            # Map the categories to their frequencies
            all_games[categorical] = all_games[categorical].map(freq)
        end = time.time()
        print(start - end)

        #############################
        # encoding of the lists (clock, and eval)
        # do it with iterrows()
        eval_columns = [f'eval_{i}' for i in range(limit_map[category]['upper'])]
        clock_columns = [f'clock_{i}' for i in range(limit_map[category]['upper'])]
        eval_lists = []
        clock_lists = []
        start = time.time()
        for index, row in all_games.iterrows():
            parsed_eval = [-100 if '#-' in i else (100 if '#' in i else float(i)) for i in literal_eval(row['all_evaluations'])]
            # here we add padding to the timeseries
            eval_lists += [parsed_eval + [0.0 for i in range(limit_map[category]['upper'] - len(parsed_eval))]]
            timestamps_dt = [datetime.datetime.strptime(ts, '%H:%M:%S') for ts in literal_eval(row['all_clocks'])]
            parsed_clock = [abs(int((t2 - t1).total_seconds())) for t1, t2 in zip(timestamps_dt[:-1], timestamps_dt[2:])]
            clock_lists += [parsed_clock + [0.0 for i in range(limit_map[category]['upper'] - len(parsed_clock))]]

        eval_df = pd.DataFrame(eval_lists, columns = eval_columns)
        clock_df = pd.DataFrame(clock_lists, columns = clock_columns)

        all_games = pd.concat([all_games.reset_index(drop=True), eval_df, clock_df], axis = 1)
        end = time.time()
        print(start - end)
        all_games = all_games.drop(['first_black_move', 'first_white_move', 'all_moves', 'all_evaluations', 'all_clocks'], axis = 1)


        # here we export it into the cleanest file
        all_games.to_csv(os.path.join(all_games_cleaned_folder, f'{category}.csv'))




if __name__ == '__main__':
    # filter_and_preprocess_data()
    concat_batched_files()
