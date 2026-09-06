import io
import pytest
import numpy as np
import pandas as pd

from src.data_preprocessing import (
    PreprocessorConfig,
    ChessGameParser,
    ChessDataPreprocessor,
    filter_and_preprocess_data,
)


SAMPLE_PGN_VALID = """[Event "Rated Rapid game"]
[Site "https://lichess.org/test1"]
[Date "2023.01.01"]
[Round "-"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "1600"]
[BlackElo "1550"]
[Opening "Italian Game: Giuoco Piano"]
[TimeControl "600+0"]
[Termination "Normal"]

1. e4 { [%eval 0.15] [%clk 0:10:00] } 1... e5 { [%eval 0.20] [%clk 0:10:00] } 2. Nf3 $2 { [%eval 0.10] [%clk 0:09:55] } 2... Nc6 { [%eval 0.25] [%clk 0:09:58] } 3. Bc4 { [%eval 0.30] [%clk 0:09:50] } 3... Bc5 { [%eval 0.35] [%clk 0:09:55] } 4. O-O { [%eval 0.40] [%clk 0:09:45] } 4... Nf6 { [%eval 0.45] [%clk 0:09:50] } 5. d3 { [%eval 0.50] [%clk 0:09:40] } 5... O-O { [%eval 0.55] [%clk 0:09:45] } 6. Bxf7+?! { [%eval 1.50] [%clk 0:09:30] } 6... Rxf7?? { [%eval #-3] [%clk 0:09:40] } 7. Qh5# 1-0
"""

SAMPLE_PGN_DRAW = """[Event "Rated Classical game"]
[Site "https://lichess.org/test2"]
[White "Player1"]
[Black "Player2"]
[Result "1/2-1/2"]
[WhiteElo "1800"]
[BlackElo "1750"]
[Opening "French Defense: Winawer"]
[TimeControl "1800+0"]
[Termination "Normal"]

1. e4 { [%eval 0.10] [%clk 0:30:00] } 1... e6 { [%eval 0.12] [%clk 0:30:00] } 2. d4 { [%eval 0.15] [%clk 0:29:50] } 2... d5 { [%eval 0.15] [%clk 0:29:55] } 1/2-1/2
"""

SAMPLE_PGN_ABANDONED = """[Event "Rated Blitz game"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "1600"]
[BlackElo "1550"]
[TimeControl "180+0"]
[Termination "Abandoned"]

1. e4 { [%eval 0.15] [%clk 0:03:00] } 1... e5 { [%eval 0.20] [%clk 0:03:00] } 1-0
"""

SAMPLE_PGN_HIGH_ELO_DIFF = """[Event "Rated Blitz game"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "2000"]
[BlackElo "1600"]
[TimeControl "180+0"]
[Termination "Normal"]

1. e4 { [%eval 0.15] [%clk 0:03:00] } 1... e5 { [%eval 0.20] [%clk 0:03:00] } 1-0
"""

SAMPLE_PGN_NO_EVAL = """[Event "Rated Blitz game"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "1600"]
[BlackElo "1550"]
[TimeControl "180+0"]
[Termination "Normal"]

1. e4 { [%clk 0:03:00] } 1... e5 { [%clk 0:03:00] } 1-0
"""

SAMPLE_PGN_INVALID_TC = """[Event "Rated Blitz game"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "1600"]
[BlackElo "1550"]
[TimeControl "-"]
[Termination "Normal"]

1. e4 { [%eval 0.15] [%clk 0:03:00] } 1... e5 { [%eval 0.20] [%clk 0:03:00] } 1-0
"""


def test_parse_clock():
    assert ChessGameParser.parse_clock("0:10:00") == 600
    assert ChessGameParser.parse_clock("1:00:00") == 3600
    assert ChessGameParser.parse_clock("0:00:45") == 45
    assert ChessGameParser.parse_clock("05:30") == 330
    assert ChessGameParser.parse_clock("45") == 45


def test_parse_eval():
    assert ChessGameParser.parse_eval("0.35") == 0.35
    assert ChessGameParser.parse_eval("-1.75") == -1.75
    assert ChessGameParser.parse_eval("#3") == 100.0
    assert ChessGameParser.parse_eval("#-2") == -100.0
    assert ChessGameParser.parse_eval("invalid") == 0.0


def test_classify_time_control():
    config = PreprocessorConfig()
    thresholds = config.time_control_thresholds
    assert ChessGameParser.classify_time_control(60, thresholds) == "bullet"
    assert ChessGameParser.classify_time_control(179, thresholds) == "bullet"
    assert ChessGameParser.classify_time_control(180, thresholds) == "blitz"
    assert ChessGameParser.classify_time_control(300, thresholds) == "blitz"
    assert ChessGameParser.classify_time_control(599, thresholds) == "blitz"
    assert ChessGameParser.classify_time_control(600, thresholds) == "rapid"
    assert ChessGameParser.classify_time_control(1200, thresholds) == "rapid"
    assert ChessGameParser.classify_time_control(1799, thresholds) == "rapid"
    assert ChessGameParser.classify_time_control(1800, thresholds) == "classical"
    assert ChessGameParser.classify_time_control(3600, thresholds) == "classical"


def test_parse_valid_game():
    config = PreprocessorConfig()
    result = ChessGameParser.parse_game(SAMPLE_PGN_VALID, config)
    assert result is not None

    category, record = result
    assert category == "rapid"
    assert record["average_elo"] == 1575.0
    assert record["first_white_move"] == "e4"
    assert record["first_black_move"] == "e5"
    assert record["opening_high_level"] == "Italian Game"
    assert record["white_castling"] == 1
    assert record["black_castling"] == 1
    assert record["termination"] == "Checkmate"
    assert record["number_of_bad_moves"] == 1
    assert record["number_of_dubious_moves"] == 1
    assert record["number_of_blunders"] == 1
    assert record["number_of_checks"] == 1
    assert "eval_0" in record
    assert "clock_0" in record
    assert record["eval_0"] == 0.15


def test_parse_draw_game():
    config = PreprocessorConfig()
    result = ChessGameParser.parse_game(SAMPLE_PGN_DRAW, config)
    assert result is not None

    category, record = result
    assert category == "classical"
    assert record["average_elo"] == 1775.0
    assert record["termination"] == "Stalemate"
    assert record["opening_high_level"] == "French Defense"


def test_filtered_games():
    config = PreprocessorConfig()
    assert ChessGameParser.parse_game(SAMPLE_PGN_ABANDONED, config) is None
    assert ChessGameParser.parse_game(SAMPLE_PGN_HIGH_ELO_DIFF, config) is None
    assert ChessGameParser.parse_game(SAMPLE_PGN_NO_EVAL, config) is None
    assert ChessGameParser.parse_game(SAMPLE_PGN_INVALID_TC, config) is None


def test_move_limit_filtering():
    config = PreprocessorConfig(
        move_limits={
            "rapid": {"upper": 5, "lower": 30}
        }
    )
    # SAMPLE_PGN_VALID has 13 plies (> 5), should be filtered
    assert ChessGameParser.parse_game(SAMPLE_PGN_VALID, config) is None


def test_encode_and_split():
    config = PreprocessorConfig()
    preprocessor = ChessDataPreprocessor(config)

    records = [
        {
            "average_elo": 1500,
            "number_of_moves": 40,
            "first_black_move": "e5",
            "first_white_move": "e4",
            "opening_high_level": "Italian Game",
            "termination": "Resignation",
            "opening": "Italian Game: Giuoco Piano",
            "eval_0": 0.1,
            "clock_0": 5,
        },
        {
            "average_elo": 1600,
            "number_of_moves": 50,
            "first_black_move": "c5",
            "first_white_move": "d4",
            "opening_high_level": "Sicilian Defense",
            "termination": "Checkmate",
            "opening": "Sicilian Defense: Closed",
            "eval_0": 0.2,
            "clock_0": 10,
        },
    ]
    df = pd.DataFrame(records)
    encoded = preprocessor.encode_features(df)

    assert "first_black_move" not in encoded.columns
    assert "e5" in encoded.columns
    assert "c5" in encoded.columns
    assert "Italian Game_high_level" in encoded.columns

    train_df, test_df = preprocessor.split_train_test(encoded)
    assert len(train_df) + len(test_df) == len(df)


def test_process_and_save_pipeline(tmp_path):
    config = PreprocessorConfig(
        processed_data_dir=str(tmp_path),
        train_ratio=0.5,
    )
    preprocessor = ChessDataPreprocessor(config)

    combined_pgn = SAMPLE_PGN_VALID + "\n\n" + SAMPLE_PGN_DRAW
    stream = io.StringIO(combined_pgn)
    results = preprocessor.process_and_save(stream, output_dir=tmp_path)

    assert "rapid" in results
    assert "classical" in results
    assert (tmp_path / "rapid_train.csv").exists()
    assert (tmp_path / "rapid_test.csv").exists()
    assert (tmp_path / "classical_train.csv").exists()
    assert (tmp_path / "classical_test.csv").exists()


def test_filter_and_preprocess_data_wrapper(tmp_path):
    pgn_file = tmp_path / "test.pgn"
    pgn_file.write_text(SAMPLE_PGN_VALID)

    out_dir = tmp_path / "processed"
    results = filter_and_preprocess_data(input_file=str(pgn_file), output_dir=str(out_dir))
    assert "rapid" in results
    assert (out_dir / "rapid_train.csv").exists()
