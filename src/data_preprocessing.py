import os
import chess.pgn
import re
import pandas as pd
import numpy as np
import tqdm
import multiprocessing


def filter_and_preprocess_data():
    '''
    current speed of the loop is ~ 300 iterations/second
    :return:
    '''
    raw_data_folder = os.path.join('data', 'raw')
    processed_data_folder = os.path.join('data', 'processed')
    filename = os.path.join(raw_data_folder, 'lichess_db_standard_rated_2022-11.pgn')

    game_metadata = []
    n = 90000000
    n = 10000

    with open(filename, "r") as f:
        for i in tqdm.tqdm(range(n)):
            game = chess.pgn.read_game(f)
            if game is None:
                break

            # Iterate over the games in the list
            # print(game.mainline_moves())
            # print('%clk' in str(game.mainline_moves()))
            # if (not '%eval' in str(game.mainline_moves())):
            if (not '%eval' in str(game.mainline_moves())) or (not '%clk' in str(game.mainline_moves())):
                continue
            # else:
            #     print(game)



            # Create an empty dictionary to store the metadata for the current game


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

            # $ is an NAG, we can ignore this (found: https://www.enpassant.dk/chess/palview/manual/pgn.htm)
            # these numbers might change, when we also have clock etc.
            try:
                parsed_list = [i for i in pgn_string.split('\n') if not '$' in i]
                white_moves = parsed_list[1::6]
                black_moves = parsed_list[4::6]
                all_moves = parsed_list[1::3]
                white_clocks = [i.split('%clk ')[1].split(']')[0] for i in parsed_list[2::6]]
                black_clocks = [i.split('%clk ')[1].split(']')[0] for i in parsed_list[5::6]]
                all_clocks = [i.split('%clk ')[1].split(']')[0] for i in parsed_list[2::3]]

                white_evaluations = [i.split('%eval ')[1].split(']')[0] for i in parsed_list[2:-1:6]]
                black_evaluations = [i.split('%eval ')[1].split(']')[0] for i in parsed_list[5:-1:6]]
                all_evaluations = [i.split('%eval ')[1].split(']')[0] for i in parsed_list[2:-1:3]]

                numeric_evaluations = [i for i in all_evaluations if not '#' in i]
                numeric_evaluations_with_mate = ['100' if '#' in i and not '-' in i else i for i in all_evaluations]
                numeric_evaluations_with_mate = ['-100' if '#-' in i else i for i in numeric_evaluations_with_mate]
                numeric_evaluations = [float(i) for i in numeric_evaluations]
                evaluation_variance = np.var(numeric_evaluations)
                evaluation_iqr = np.percentile(numeric_evaluations, 75) - np.percentile(numeric_evaluations, 25)
                evaluation_range = np.max(numeric_evaluations) - np.min(numeric_evaluations)
            except:
                continue



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


            # number_of_blunders = len([i for i in all_moves if '??' in i])
            # number_of_excellent_moves = len([i for i in all_moves if '!!' in i])
            # number_of_bad_moves = len([i for i in all_moves if '?' in i and not '??' in i])
            # number_of_good_moves = len([i for i in all_moves if '!' in i and not '!!' in i])
            # number_of_interesting_moves = len([i for i in all_moves if '!?' in i])
            # number_of_dubious_moves = len([i for i in all_moves if '?!' in i])

            numeric_evaluations_with_mate = ['100' if '#' in i and not '-' in i else i for i in all_evaluations]
            numeric_evaluations_with_mate = ['-100' if '#-' in i else i for i in numeric_evaluations_with_mate]
            numeric_evaluations_with_mate = [float(i) for i in numeric_evaluations_with_mate]
            numeric_evaluations_with_mate = [j-i for i, j in zip(numeric_evaluations_with_mate[:-1], numeric_evaluations_with_mate[1:])]
            number_of_blunders = len([i for i in numeric_evaluations_with_mate if abs(i) > 3])
            number_of_bad_moves = len([i for i in numeric_evaluations_with_mate if abs(i) > 1.5 and abs(i) <= 3])


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
                'number_of_bad_moves': number_of_bad_moves,
                'number_of_blunders': number_of_blunders,
                'evaluation_variance': evaluation_variance,
                'evaluation_iqr': evaluation_iqr,
                'evaluation_range': evaluation_range,
                'all_moves': all_moves,
                'all_evaluations': all_evaluations,
                'all_clocks':all_clocks
            })

            if i in [j for j in range(0,n,1000000)]:
                df = pd.DataFrame(game_metadata)
                print(df)
                bullet = df[df['time_control_seconds'] < 180]
                blitz = df[(df['time_control_seconds'] >= 180) & (df['time_control_seconds'] < 600)]
                rapid = df[(df['time_control_seconds'] >= 600) & (df['time_control_seconds'] < 1800)]
                classical = df[df['time_control_seconds'] >= 1800]

                bullet_file = os.path.join(processed_data_folder, 'bullet.csv')
                blitz_file = os.path.join(processed_data_folder, 'blitz.csv')
                rapid_file = os.path.join(processed_data_folder, 'rapid.csv')
                classical_file = os.path.join(processed_data_folder, 'classical.csv')

                bullet.to_csv(bullet_file)
                blitz.to_csv(blitz_file)
                rapid.to_csv(rapid_file)
                classical.to_csv(classical_file)

    df = pd.DataFrame(game_metadata)
    print(df)
    bullet = df[df['time_control_seconds'] < 180]
    blitz = df[(df['time_control_seconds'] >= 180) & (df['time_control_seconds'] < 600)]
    rapid = df[(df['time_control_seconds'] >= 600) & (df['time_control_seconds'] < 1800)]
    classical = df[df['time_control_seconds'] >= 1800]

    bullet_file = os.path.join(processed_data_folder, 'bullet.csv')
    blitz_file = os.path.join(processed_data_folder, 'blitz.csv')
    rapid_file = os.path.join(processed_data_folder, 'rapid.csv')
    classical_file = os.path.join(processed_data_folder, 'classical.csv')

    bullet.to_csv(bullet_file)
    blitz.to_csv(blitz_file)
    rapid.to_csv(rapid_file)
    classical.to_csv(classical_file)

if __name__ == '__main__':
    filter_and_preprocess_data()