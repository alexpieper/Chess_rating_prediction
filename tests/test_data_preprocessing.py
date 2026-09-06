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

SAMPLE_PGN_LONG_GAME = """[Event "Rated Rapid game"]
[Site "https://lichess.org/test_long"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "1608"]
[BlackElo "1610"]
[Opening "Queen's Gambit Declined: Exchange Variation"]
[TimeControl "600+0"]
[Termination "Normal"]

1. d4 { [%eval 0.08] [%clk 0:10:00] } 1... d5 { [%eval 0.07] [%clk 0:10:00] } 2. c4 { [%eval 0.18] [%clk 0:09:56] } 2... Nf6 { [%eval 0.65] [%clk 0:10:00] } 3. Nc3?! { [%eval -0.08] [%clk 0:09:54] } 3... e6 { [%eval 0.35] [%clk 0:09:59] } 4. cxd5 { [%eval 0.23] [%clk 0:09:49] } 4... Nxd5 { [%eval 0.2] [%clk 0:09:45] } 5. Nf3 { [%eval 0.08] [%clk 0:09:39] } 5... Nxc3 { [%eval 0.47] [%clk 0:09:43] } 6. bxc3 { [%eval 0.36] [%clk 0:09:37] } 6... Be7 { [%eval 0.5] [%clk 0:09:39] } 7. e4 { [%eval 0.47] [%clk 0:09:36] } 7... O-O { [%eval 0.61] [%clk 0:09:01] } 8. Bd3 { [%eval 0.45] [%clk 0:09:35] } 8... Nc6? { [%eval 1.82] [%clk 0:08:59] } 9. a3? { [%eval 0.19] [%clk 0:09:32] } 9... Bd7? { [%eval 1.23] [%clk 0:08:58] } 10. Be3?! { [%eval 0.55] [%clk 0:09:27] } 10... a6?! { [%eval 1.21] [%clk 0:08:55] } 11. Qc2?! { [%eval 0.58] [%clk 0:09:25] } 11... b5? { [%eval 1.81] [%clk 0:08:50] } 12. e5 { [%eval 1.57] [%clk 0:09:18] } 12... Bc8?? { [%eval 5.57] [%clk 0:08:48] } 13. Bxh7+ { [%eval 5.15] [%clk 0:09:15] } 13... Kh8 { [%eval 5.55] [%clk 0:08:46] } 14. Be4? { [%eval 3.83] [%clk 0:09:11] } 14... Bb7?! { [%eval 4.74] [%clk 0:08:46] } 15. h4 { [%eval 4.41] [%clk 0:08:27] } 15... Rb8? { [%eval 7.1] [%clk 0:08:29] } 16. Ng5 { [%eval 6.88] [%clk 0:08:25] } 16... Bxg5?? { [%eval #4] [%clk 0:08:26] } 17. hxg5+ { [%eval #3] [%clk 0:08:22] } 17... Kg8 { [%eval #3] [%clk 0:08:24] } 18. Bxc6?! { [%eval 10.58] [%clk 0:08:18] } 18... Bxc6?! { [%eval #1] [%clk 0:08:22] } 19. Qh7# { [%clk 0:08:16] } 1-0
"""

SAMPLE_PGN_STALEMATE = """[Event "Rated Classical game"]
[White "Player1"]
[Black "Player2"]
[Result "1/2-1/2"]
[WhiteElo "1800"]
[BlackElo "1750"]
[TimeControl "1800+0"]
[Termination "Normal"]

1. e3 { [%eval 0.0] [%clk 0:30:00] } 1... a5 { [%eval 0.0] [%clk 0:30:00] } 2. Qh5 { [%eval 0.0] [%clk 0:29:50] } 2... Ra6 { [%eval 0.0] [%clk 0:29:55] } 3. Qxa5 { [%eval 0.0] [%clk 0:29:40] } 3... h5 { [%eval 0.0] [%clk 0:29:50] } 4. Qxc7 { [%eval 0.0] [%clk 0:29:30] } 4... Rah6 { [%eval 0.0] [%clk 0:29:45] } 5. h4 { [%eval 0.0] [%clk 0:29:20] } 5... f6 { [%eval 0.0] [%clk 0:29:40] } 6. Qxd7+ { [%eval 0.0] [%clk 0:29:10] } 6... Kf7 { [%eval 0.0] [%clk 0:29:35] } 7. Qxb7 { [%eval 0.0] [%clk 0:29:00] } 7... Qd3 { [%eval 0.0] [%clk 0:29:30] } 8. Qxb8 { [%eval 0.0] [%clk 0:28:50] } 8... Qh7 { [%eval 0.0] [%clk 0:29:25] } 9. Qxc8 { [%eval 0.0] [%clk 0:28:40] } 9... Kg6 { [%eval 0.0] [%clk 0:29:20] } 10. Qe6 { [%eval 0.0] [%clk 0:28:30] } 1/2-1/2
"""

SAMPLE_PGN_TIMEOUT_DRAW = """[Event "Rated Blitz game"]
[White "Player1"]
[Black "Player2"]
[Result "1/2-1/2"]
[WhiteElo "1600"]
[BlackElo "1550"]
[TimeControl "180+0"]
[Termination "Time forfeit"]

1. e4 { [%eval 0.10] [%clk 0:03:00] } 1... e5 { [%eval 0.12] [%clk 0:03:00] } 2. d4 { [%eval 0.15] [%clk 0:02:50] } 2... exd4 { [%eval 0.15] [%clk 0:02:55] } 1/2-1/2
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

SAMPLE_PGN_PROMOTION_CASTLE = """[Event "Rated Rapid game"]
[Site "https://lichess.org/test_promo"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "1700"]
[BlackElo "1680"]
[Opening "King's Pawn Game"]
[TimeControl "900+0"]
[Termination "Normal"]

1. e4 { [%eval 0.1] [%clk 0:15:00] } 1... e5 { [%eval 0.1] [%clk 0:15:00] } 2. f4 { [%eval 0.2] [%clk 0:14:50] } 2... exf4 { [%eval 0.2] [%clk 0:14:55] } 3. d4 { [%eval 0.3] [%clk 0:14:40] } 3... d5 { [%eval 0.3] [%clk 0:14:45] } 4. e5 { [%eval 0.4] [%clk 0:14:30] } 4... f3 { [%eval 0.4] [%clk 0:14:35] } 5. e6 { [%eval 0.5] [%clk 0:14:20] } 5... f2+ { [%eval 0.5] [%clk 0:14:25] } 6. Ke2 { [%eval 0.6] [%clk 0:14:10] } 6... fxg1=N+ { [%eval 1.0] [%clk 0:14:15] } 7. Kd3 { [%eval 1.1] [%clk 0:14:00] } 7... O-O-O { [%eval 1.1] [%clk 0:14:05] } 8. e7 { [%eval 1.2] [%clk 0:13:50] } 8... Bxe7 { [%eval 1.2] [%clk 0:13:55] } 9. exd8=Q+ { [%eval 5.0] [%clk 0:13:40] } 9... Kxd8 { [%eval 5.0] [%clk 0:13:45] } 10. O-O-O 1-0
"""


def make_config(**kwargs) -> PreprocessorConfig:
    """Config with the lower move limit disabled, so the short samples survive."""
    config = PreprocessorConfig(**kwargs)
    for limits in config.move_limits.values():
        limits["lower"] = 0
    return config


def test_parse_clock():
    assert ChessGameParser.parse_clock("0:10:00") == 600
    assert ChessGameParser.parse_clock("1:00:00") == 3600
    assert ChessGameParser.parse_clock("0:00:45") == 45
    assert ChessGameParser.parse_clock("05:30") == 330
    assert ChessGameParser.parse_clock("45") == 45
    assert ChessGameParser.parse_clock("invalid") == 0


def test_parse_eval():
    assert ChessGameParser.parse_eval("0.35") == 0.35
    assert ChessGameParser.parse_eval("-1.75") == -1.75
    assert ChessGameParser.parse_eval("#3") == 100.0
    assert ChessGameParser.parse_eval("#-2") == -100.0
    assert ChessGameParser.parse_eval("invalid") == 0.0


def test_classify_time_control():
    config = make_config()
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
    config = make_config()
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
    assert record["king_moves"] == 2
    assert record["queen_moves"] == 1
    assert record["rook_moves"] == 1
    assert record["bishop_moves"] == 3
    assert record["knight_moves"] == 3
    assert record["pawn_moves"] == 3
    assert record["number_of_moves"] == 13
    assert (
        record["king_moves"]
        + record["queen_moves"]
        + record["rook_moves"]
        + record["bishop_moves"]
        + record["knight_moves"]
        + record["pawn_moves"]
        == record["number_of_moves"]
    )
    assert "eval_0" in record
    assert "clock_0" in record
    assert record["eval_0"] == 0.15


def test_parse_promotion_and_queenside_castling():
    config = make_config()
    result = ChessGameParser.parse_game(SAMPLE_PGN_PROMOTION_CASTLE, config)
    assert result is not None

    category, record = result
    assert category == "rapid"
    assert record["white_castling"] == 2  # White played O-O-O
    assert record["black_castling"] == 2  # Black played O-O-O
    # White played Ke2 (king), Kd3 (king), O-O-O (king) -> 3 king moves
    # Black played O-O-O (king), Kxd8 (king) -> 2 king moves
    assert record["king_moves"] == 5
    # Promotion fxg1=N+ and exd8=Q+ are PAWN moves, NOT knight/queen moves
    assert record["queen_moves"] == 0
    assert record["knight_moves"] == 0
    # Bishop moves: Bxe7 (1)
    assert record["bishop_moves"] == 1
    assert record["rook_moves"] == 0
    # Pawn moves: e4, e5, f4, exf4, d4, d5, e5, f3, e6, f2+, fxg1=N+, e7, exd8=Q+ -> 13
    assert record["pawn_moves"] == 13
    assert record["number_of_moves"] == 19
    assert (
        record["king_moves"]
        + record["queen_moves"]
        + record["rook_moves"]
        + record["bishop_moves"]
        + record["knight_moves"]
        + record["pawn_moves"]
        == record["number_of_moves"]
    )


def test_parse_draw_game():
    config = make_config()
    result = ChessGameParser.parse_game(SAMPLE_PGN_DRAW, config)
    assert result is not None

    category, record = result
    assert category == "classical"
    assert record["average_elo"] == 1775.0
    assert record["termination"] == "Draw by agreement"
    assert record["opening_high_level"] == "French Defense"


def test_parse_draw_reasons():
    config = make_config()

    _, stalemate_record = ChessGameParser.parse_game(SAMPLE_PGN_STALEMATE, config)
    assert stalemate_record["termination"] == "Stalemate"

    _, timeout_record = ChessGameParser.parse_game(SAMPLE_PGN_TIMEOUT_DRAW, config)
    assert timeout_record["termination"] == "Draw by timeout"


def test_classify_draw():
    # 3. Qe6 leaves Black without a legal move
    stalemate_moves = [
        "e3", "a5", "Qh5", "Ra6", "Qxa5", "h5", "Qxc7", "Rah6", "h4", "f6",
        "Qxd7+", "Kf7", "Qxb7", "Qd3", "Qxb8", "Qh7", "Qxc8", "Kg6", "Qe6",
    ]
    assert ChessGameParser.classify_draw(stalemate_moves) == "Stalemate"

    # Knights shuffling back and forth repeats the starting position three times
    threefold_moves = ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"]
    assert ChessGameParser.classify_draw(threefold_moves) == "Threefold repetition"

    assert ChessGameParser.classify_draw(["e4", "e5"]) == "Draw by agreement"
    assert ChessGameParser.classify_draw(["this-is-not-a-move"]) == "Draw by agreement"


def test_filtered_games():
    config = make_config()
    assert ChessGameParser.parse_game(SAMPLE_PGN_ABANDONED, config) is None
    assert ChessGameParser.parse_game(SAMPLE_PGN_HIGH_ELO_DIFF, config) is None
    assert ChessGameParser.parse_game(SAMPLE_PGN_NO_EVAL, config) is None
    assert ChessGameParser.parse_game(SAMPLE_PGN_INVALID_TC, config) is None


def test_move_limit_filtering():
    # SAMPLE_PGN_VALID has 13 plies
    upper_filtered = PreprocessorConfig(move_limits={"rapid": {"upper": 5, "lower": 0}})
    assert ChessGameParser.parse_game(SAMPLE_PGN_VALID, upper_filtered) is None

    lower_filtered = PreprocessorConfig(move_limits={"rapid": {"upper": 100, "lower": 30}})
    assert ChessGameParser.parse_game(SAMPLE_PGN_VALID, lower_filtered) is None

    within_limits = PreprocessorConfig(move_limits={"rapid": {"upper": 100, "lower": 13}})
    assert ChessGameParser.parse_game(SAMPLE_PGN_VALID, within_limits) is not None


def test_encode_and_split():
    config = make_config()
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
    encoded = preprocessor.encode_features(df, category="rapid")

    assert "first_black_move" not in encoded.columns
    assert "e5" in encoded.columns
    assert "c5" in encoded.columns
    assert "Italian Game_high_level" in encoded.columns

    train_df, test_df = preprocessor.split_train_test(encoded)
    assert len(train_df) + len(test_df) == len(df)


def test_locked_columns_consistency():
    config = make_config()
    preprocessor = ChessDataPreprocessor(config)

    expected_rapid_cols = preprocessor.get_feature_columns("rapid")
    expected_bullet_cols = preprocessor.get_feature_columns("bullet")

    # Verify column count is deterministic
    n_terminations = len(config.fixed_terminations)
    assert len(expected_rapid_cols) == 18 + 20 + 20 + n_terminations + len(config.fixed_high_level_openings) + 100 + 100
    assert len(expected_bullet_cols) == 18 + 20 + 20 + n_terminations + len(config.fixed_high_level_openings) + 80 + 80

    # Dataset with 1 game
    df1 = pd.DataFrame([{
        "average_elo": 1500,
        "first_black_move": "e5",
        "first_white_move": "e4",
        "opening_high_level": "Italian Game",
        "termination": "Resignation",
    }])
    encoded1 = preprocessor.encode_features(df1, category="rapid")

    # Dataset with different game and different opening
    df2 = pd.DataFrame([{
        "average_elo": 1800,
        "first_black_move": "c6",
        "first_white_move": "d4",
        "opening_high_level": "Caro-Kann Defense",
        "termination": "Checkmate",
    }])
    encoded2 = preprocessor.encode_features(df2, category="rapid")

    # Both DataFrames must have identical columns in identical order
    assert list(encoded1.columns) == expected_rapid_cols
    assert list(encoded2.columns) == expected_rapid_cols
    assert list(encoded1.columns) == list(encoded2.columns)


def test_process_and_save_pipeline(tmp_path):
    config = make_config(
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

    rapid_train = pd.read_csv(tmp_path / "rapid_train.csv", index_col=0)
    rapid_test = pd.read_csv(tmp_path / "rapid_test.csv", index_col=0)
    assert list(rapid_train.columns) == preprocessor.get_feature_columns("rapid")
    assert list(rapid_test.columns) == preprocessor.get_feature_columns("rapid")


def test_filter_and_preprocess_data_wrapper(tmp_path):
    pgn_file = tmp_path / "test.pgn"
    # 37 plies, so it clears the configured lower move limit
    pgn_file.write_text(SAMPLE_PGN_LONG_GAME)

    out_dir = tmp_path / "processed"
    results = filter_and_preprocess_data(input_file=str(pgn_file), output_dir=str(out_dir))
    assert "rapid" in results
    assert (out_dir / "rapid_train.csv").exists()
