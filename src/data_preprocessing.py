import os
import re
import io
import argparse
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterator, Union
from pathlib import Path

import chess
import numpy as np
import pandas as pd
import tqdm
import zstandard as zstd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


NUMERICAL_COLUMNS: List[str] = [
    "average_elo",
    "number_of_moves",
    "number_of_checks",
    "white_castling",
    "black_castling",
    "number_of_captures",
    "knight_moves",
    "bishop_moves",
    "rook_moves",
    "queen_moves",
    "king_moves",
    "pawn_moves",
    "number_of_blunders",
    "number_of_bad_moves",
    "number_of_dubious_moves",
    "evaluation_variance",
    "evaluation_iqr",
    "evaluation_range",
]

DEFAULT_WHITE_MOVES: List[str] = [
    "a3", "a4", "b3", "b4", "c3", "c4", "d3", "d4", "e3", "e4",
    "f3", "f4", "g3", "g4", "h3", "h4", "Na3", "Nc3", "Nf3", "Nh3"
]

DEFAULT_BLACK_MOVES: List[str] = [
    "a6", "a5", "b6", "b5", "c6", "c5", "d6", "d5", "e6", "e5",
    "f6", "f5", "g6", "g5", "h6", "h5", "Na6", "Nc6", "Nf6", "Nh6"
]

DEFAULT_TERMINATIONS: List[str] = [
    "Checkmate", "Resignation", "Time forfeit", "Rules infraction", "Normal", "Abandoned",
    "Stalemate", "Insufficient material", "Threefold repetition", "Fifty-move rule",
    "Draw by agreement", "Draw by timeout",
]

DEFAULT_HIGH_LEVEL_OPENINGS: List[str] = [
    "Alekhine Defense",
    "Amar Opening",
    "Amazon Attack",
    "Barnes Opening",
    "Benko Gambit",
    "Benko Gambit Accepted",
    "Benko Gambit Declined",
    "Benoni Defense",
    "Bird Opening",
    "Bishop's Opening",
    "Blackmar-Diemer Gambit",
    "Bogo-Indian Defense",
    "Borg Opening",
    "Canard Opening",
    "Caro-Kann Defense",
    "Carr Opening",
    "Catalan Opening",
    "Center Game",
    "Center Game Accepted",
    "Clemenz Opening",
    "Colle System",
    "Crab Opening",
    "Czech Defense",
    "Danish Gambit",
    "Danish Gambit Accepted",
    "Doery Defense",
    "Dresden Opening",
    "Duck Opening",
    "Dunst Opening",
    "Dutch Defense",
    "East Indian Defense",
    "English Defense",
    "English Opening",
    "Four Knights Game",
    "Franco-Indian Defense",
    "French Defense",
    "Giuoco Piano",
    "Goldsmith Defense",
    "Grob Opening",
    "Grünfeld Defense",
    "Guatemala Defense",
    "Gunderam Defense",
    "Horwitz Defense",
    "Hungarian Opening",
    "Indian Defense",
    "Irish Gambit",
    "Italian Game",
    "Kadas Opening",
    "Kangaroo Defense",
    "King's Gambit",
    "King's Gambit Accepted",
    "King's Gambit Declined",
    "King's Indian Attack",
    "King's Indian Defense",
    "King's Knight Opening",
    "King's Pawn Game",
    "Lasker Simul",
    "Lemming Defense",
    "Lion Defense",
    "London System",
    "Marienbad System",
    "Marrow Opening",
    "Meadow Hay Opening",
    "Meitner-Mieses Gambit",
    "Mieses Opening",
    "Modern Defense",
    "Mongoose Defense",
    "Myers Opening",
    "Nimzo-Indian Defense",
    "Nimzo-Larsen Attack",
    "Nimzowitsch Defense",
    "Old Indian Defense",
    "Owen Defense",
    "Paleface Attack",
    "Petrov's Defense",
    "Philidor Defense",
    "Pirc Defense",
    "Polish Defense",
    "Polish Opening",
    "Ponziani Opening",
    "Queen's Gambit",
    "Queen's Gambit Accepted",
    "Queen's Gambit Declined",
    "Queen's Indian Defense",
    "Queen's Knight Defense",
    "Queen's Pawn Game",
    "Rat Defense",
    "Reti Opening",
    "Richter-Veresov Attack",
    "Robatsch Defense",
    "Rook Opening",
    "Russian Game",
    "Ruy Lopez",
    "Saragossa Opening",
    "Scandinavian Defense",
    "Scotch Game",
    "Semi-Slav Defense",
    "Sicilian Defense",
    "Slav Defense",
    "Slav Indian",
    "Sodium Attack",
    "St. George Defense",
    "System",
    "Tarrasch Defense",
    "Three Knights Opening",
    "Torre Attack",
    "Trompowsky Attack",
    "Unknown",
    "Van Geet Opening",
    "Van't Kruijs Opening",
    "Vienna Game",
    "Vulture Defense",
    "Wade Defense",
    "Ware Defense",
    "Ware Opening",
    "Zukertort Opening",
]


@dataclass
class PreprocessorConfig:
    raw_data_dir: str = os.path.join("data", "raw")
    processed_data_dir: str = os.path.join("data", "processed", "all_games_clean")
    max_elo_diff: int = 200
    train_ratio: float = 0.6
    random_seed: int = 42
    move_limits: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {
            "classical": {"upper": 100, "lower": 30},
            "bullet": {"upper": 80, "lower": 30},
            "rapid": {"upper": 100, "lower": 30},
            "blitz": {"upper": 100, "lower": 30},
        }
    )
    time_control_thresholds: Dict[str, Tuple[int, Optional[int]]] = field(
        default_factory=lambda: {
            "bullet": (0, 180),
            "blitz": (180, 600),
            "rapid": (600, 1800),
            "classical": (1800, None),
        }
    )
    categorical_columns: List[str] = field(
        default_factory=lambda: [
            "first_black_move",
            "first_white_move",
            "opening_high_level",
            "termination",
            "opening",
        ]
    )
    fixed_white_moves: List[str] = field(default_factory=lambda: list(DEFAULT_WHITE_MOVES))
    fixed_black_moves: List[str] = field(default_factory=lambda: list(DEFAULT_BLACK_MOVES))
    fixed_terminations: List[str] = field(default_factory=lambda: list(DEFAULT_TERMINATIONS))
    fixed_high_level_openings: List[str] = field(default_factory=lambda: list(DEFAULT_HIGH_LEVEL_OPENINGS))
    numerical_columns: List[str] = field(default_factory=lambda: list(NUMERICAL_COLUMNS))


class ChessGameParser:
    TOKEN_PATTERN = re.compile(
        r'(?:[0-9]+(?:\.+|\s*\.\.\.)\s*)?([a-zA-Z0-9+=#\/-]+)([\?!]*)\s*(?:\$(\d+))?(?:\s*\{\s*([^}]*)\s*\})?'
    )
    EVAL_PATTERN = re.compile(r"\[%eval\s+([^\]]+)\]")
    CLK_PATTERN = re.compile(r"\[%clk\s+([^\]]+)\]")
    HEADER_PATTERN = re.compile(r'\[([a-zA-Z0-9_]+)\s+\"([^\"]*)\"\]')

    @staticmethod
    def parse_clock(clock_str: str) -> int:
        parts = clock_str.split(":")
        try:
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            elif len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            return int(parts[0])
        except (ValueError, IndexError):
            return 0

    @staticmethod
    def parse_eval(eval_str: str) -> float:
        if "#-" in eval_str:
            return -100.0
        if "#" in eval_str:
            return 100.0
        try:
            return float(eval_str)
        except ValueError:
            return 0.0

    @staticmethod
    def extract_castling(moves: List[str]) -> int:
        for m in moves:
            if m.startswith(("O-O-O", "0-0-0")):
                return 2
            elif m.startswith(("O-O", "0-0")):
                return 1
        return 0

    @classmethod
    def parse_game(cls, game_text: str, config: PreprocessorConfig) -> Optional[Tuple[str, Dict]]:
        if "[%eval" not in game_text or "[%clk" not in game_text:
            return None

        headers_text, _, move_text = game_text.partition("\n\n")
        headers = dict(cls.HEADER_PATTERN.findall(headers_text))

        if headers.get("Termination") == "Abandoned":
            return None

        try:
            white_elo = int(headers.get("WhiteElo", "0"))
            black_elo = int(headers.get("BlackElo", "0"))
        except ValueError:
            return None

        if abs(white_elo - black_elo) > config.max_elo_diff:
            return None

        time_control = headers.get("TimeControl", "")
        if "+" not in time_control:
            return None

        try:
            tc_seconds = int(time_control.split("+")[0])
        except ValueError:
            return None

        category = cls.classify_time_control(tc_seconds, config.time_control_thresholds)
        if category not in config.move_limits:
            return None

        upper_limit = config.move_limits[category]["upper"]
        lower_limit = config.move_limits[category]["lower"]

        raw_tokens = cls.TOKEN_PATTERN.findall(move_text)
        all_moves: List[str] = []
        all_evals: List[float] = []
        all_clocks: List[int] = []
        num_bad = 0
        num_blunder = 0
        num_dubious = 0

        for san, annot, nag, comment in raw_tokens:
            if not san or san in ("1-0", "0-1", "1/2-1/2", "*"):
                continue
            all_moves.append(san)

            ev_match = cls.EVAL_PATTERN.search(comment)
            clk_match = cls.CLK_PATTERN.search(comment)
            all_evals.append(cls.parse_eval(ev_match.group(1)) if ev_match else 0.0)
            all_clocks.append(cls.parse_clock(clk_match.group(1)) if clk_match else 0)

            if "??" in annot or nag == "4":
                num_blunder += 1
            elif "?!" in annot or nag == "6":
                num_dubious += 1
            elif "?" in annot or nag == "2":
                num_bad += 1

        num_moves = len(all_moves)
        if num_moves < lower_limit or num_moves > upper_limit or len(all_evals) == 0:
            return None

        white_moves = all_moves[0::2]
        black_moves = all_moves[1::2]
        first_white = white_moves[0] if white_moves else ""
        first_black = black_moves[0] if black_moves else ""

        termination = headers.get("Termination", "Normal")
        result = headers.get("Result", "*")
        if result == "1/2-1/2":
            if termination == "Normal":
                termination = cls.classify_draw(all_moves)
            elif termination == "Time forfeit":
                termination = "Draw by timeout"
        elif termination == "Normal" and result in ("1-0", "0-1"):
            termination = "Checkmate" if all_moves and "#" in all_moves[-1] else "Resignation"

        opening = headers.get("Opening", "Unknown")
        opening_hl = opening.split(":")[0]

        numeric_evals = [e for e in all_evals if abs(e) != 100.0]
        ev_var = float(np.var(numeric_evals)) if numeric_evals else 0.0
        ev_iqr = float(np.percentile(numeric_evals, 75) - np.percentile(numeric_evals, 25)) if numeric_evals else 0.0
        ev_range = float(np.max(numeric_evals) - np.min(numeric_evals)) if numeric_evals else 0.0

        parsed_clock = [abs(int(all_clocks[i + 2] - all_clocks[i])) for i in range(len(all_clocks) - 2)]
        eval_padded = all_evals[:upper_limit] + [0.0] * max(0, upper_limit - len(all_evals))
        clock_padded = parsed_clock[:upper_limit] + [0] * max(0, upper_limit - len(parsed_clock))

        knight_moves = sum(1 for m in all_moves if m.startswith("N"))
        bishop_moves = sum(1 for m in all_moves if m.startswith("B"))
        rook_moves = sum(1 for m in all_moves if m.startswith("R"))
        queen_moves = sum(1 for m in all_moves if m.startswith("Q"))
        king_moves = sum(1 for m in all_moves if m.startswith(("K", "O-O", "0-0")))
        pawn_moves = sum(1 for m in all_moves if m and m[0] in "abcdefgh")

        white_castling = cls.extract_castling(white_moves)
        black_castling = cls.extract_castling(black_moves)

        record = {
            "average_elo": 0.5 * (white_elo + black_elo),
            "number_of_moves": num_moves,
            "number_of_checks": sum(1 for m in all_moves if "+" in m or "#" in m),
            "white_castling": white_castling,
            "black_castling": black_castling,
            "number_of_captures": sum(1 for m in all_moves if "x" in m),
            "knight_moves": knight_moves,
            "bishop_moves": bishop_moves,
            "rook_moves": rook_moves,
            "queen_moves": queen_moves,
            "king_moves": king_moves,
            "pawn_moves": pawn_moves,
            "number_of_blunders": num_blunder,
            "number_of_bad_moves": num_bad,
            "number_of_dubious_moves": num_dubious,
            "evaluation_variance": ev_var,
            "evaluation_iqr": ev_iqr,
            "evaluation_range": ev_range,
            "first_black_move": first_black,
            "first_white_move": first_white,
            "opening_high_level": opening_hl,
            "termination": termination,
            "opening": opening,
        }

        for idx, val in enumerate(eval_padded):
            record[f"eval_{idx}"] = val
        for idx, val in enumerate(clock_padded):
            record[f"clock_{idx}"] = val

        return category, record

    @staticmethod
    def classify_draw(moves: List[str]) -> str:
        """Replay a drawn game to recover why it was drawn.

        The PGN headers only say "Normal", so the final position has to be
        reconstructed. Only drawn games are replayed, so the cost stays small.
        """
        board = chess.Board()
        try:
            for san in moves:
                board.push_san(san)
        except ValueError:
            return "Draw by agreement"

        if board.is_stalemate():
            return "Stalemate"
        if board.is_insufficient_material():
            return "Insufficient material"
        if board.is_repetition(3):
            return "Threefold repetition"
        if board.halfmove_clock >= 100:
            return "Fifty-move rule"
        return "Draw by agreement"

    @staticmethod
    def classify_time_control(tc_seconds: int, thresholds: Dict[str, Tuple[int, Optional[int]]]) -> str:
        for cat, (low, high) in thresholds.items():
            if high is None:
                if tc_seconds >= low:
                    return cat
            else:
                if low <= tc_seconds < high:
                    return cat
        return "classical"


class ChessDataPreprocessor:
    def __init__(self, config: Optional[PreprocessorConfig] = None):
        self.config = config or PreprocessorConfig()

    @staticmethod
    def stream_games(source: Union[str, Path, io.TextIOBase]) -> Iterator[str]:
        if isinstance(source, (str, Path)):
            source_path = Path(source)
            if source_path.suffix == ".zst":
                with open(source_path, "rb") as compressed:
                    dctx = zstd.ZstdDecompressor()
                    with dctx.stream_reader(compressed) as reader:
                        text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                        yield from ChessDataPreprocessor._read_game_blocks(text_stream)
            else:
                with open(source_path, "r", encoding="utf-8", errors="replace") as f:
                    yield from ChessDataPreprocessor._read_game_blocks(f)
        else:
            yield from ChessDataPreprocessor._read_game_blocks(source)

    @staticmethod
    def _read_game_blocks(stream: io.TextIOBase) -> Iterator[str]:
        current_lines: List[str] = []
        for line in stream:
            if line.startswith("[Event "):
                if current_lines:
                    yield "".join(current_lines)
                    current_lines = []
            current_lines.append(line)
        if current_lines:
            yield "".join(current_lines)

    def extract_from_stream(
        self,
        stream_or_path: Union[str, Path, io.TextIOBase],
        max_games: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, List[Dict]]:
        categorized_records: Dict[str, List[Dict]] = {
            cat: [] for cat in self.config.time_control_thresholds.keys()
        }
        game_iterator = self.stream_games(stream_or_path)
        if show_progress:
            game_iterator = tqdm.tqdm(game_iterator, desc="Processing PGN games")

        parsed_count = 0
        for game_text in game_iterator:
            if max_games is not None and parsed_count >= max_games:
                break
            parsed_count += 1
            result = ChessGameParser.parse_game(game_text, self.config)
            if result is not None:
                category, record = result
                categorized_records[category].append(record)

        return categorized_records

    def get_feature_columns(self, category: str = "classical") -> List[str]:
        upper_limit = self.config.move_limits.get(category, {}).get("upper", 100)
        cols = list(self.config.numerical_columns)
        cols.extend(self.config.fixed_white_moves)
        cols.extend(self.config.fixed_black_moves)
        cols.extend(self.config.fixed_terminations)
        cols.extend([f"{op}_high_level" for op in self.config.fixed_high_level_openings])
        cols.extend([f"eval_{i}" for i in range(upper_limit)])
        cols.extend([f"clock_{i}" for i in range(upper_limit)])
        return cols

    def encode_features(self, df: pd.DataFrame, category: Optional[str] = None) -> pd.DataFrame:
        if df.empty:
            target_cols = self.get_feature_columns(category or "classical")
            return pd.DataFrame(columns=target_cols)

        encoded_df = df.copy()

        if "first_white_move" in encoded_df.columns:
            cat_white = pd.Categorical(encoded_df["first_white_move"], categories=self.config.fixed_white_moves)
            white_dummies = pd.get_dummies(cat_white, dtype=np.int8)
            encoded_df = encoded_df.join(white_dummies)

        if "first_black_move" in encoded_df.columns:
            cat_black = pd.Categorical(encoded_df["first_black_move"], categories=self.config.fixed_black_moves)
            black_dummies = pd.get_dummies(cat_black, dtype=np.int8)
            encoded_df = encoded_df.join(black_dummies)

        if "termination" in encoded_df.columns:
            cat_term = pd.Categorical(encoded_df["termination"], categories=self.config.fixed_terminations)
            term_dummies = pd.get_dummies(cat_term, dtype=np.int8)
            encoded_df = encoded_df.join(term_dummies)

        if "opening_high_level" in encoded_df.columns:
            cat_hl = pd.Categorical(encoded_df["opening_high_level"], categories=self.config.fixed_high_level_openings)
            hl_dummies = pd.get_dummies(cat_hl, dtype=np.int8).add_suffix("_high_level")
            encoded_df = encoded_df.join(hl_dummies)

        cols_to_drop = [col for col in self.config.categorical_columns if col in encoded_df.columns]
        encoded_df = encoded_df.drop(columns=cols_to_drop)

        if category is None:
            clock_eval_cols = [c for c in encoded_df.columns if c.startswith("eval_") or c.startswith("clock_")]
            if clock_eval_cols:
                max_idx = max(int(c.split("_")[1]) for c in clock_eval_cols)
                matched_cat = "classical"
                for cat, lims in self.config.move_limits.items():
                    if lims["upper"] - 1 == max_idx:
                        matched_cat = cat
                        break
                target_cols = self.get_feature_columns(matched_cat)
            else:
                target_cols = self.get_feature_columns("classical")
        else:
            target_cols = self.get_feature_columns(category)

        encoded_df = encoded_df.reindex(columns=target_cols, fill_value=0)
        return encoded_df

    def process_and_save(
        self,
        input_source: Union[str, Path, io.TextIOBase],
        output_dir: Optional[Union[str, Path]] = None,
        max_games: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        target_dir = Path(output_dir or self.config.processed_data_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        categorized_records = self.extract_from_stream(
            input_source, max_games=max_games, show_progress=show_progress
        )

        results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
        for category, records in categorized_records.items():
            if not records:
                logger.info(f"No records found for category: {category}")
                continue

            df = pd.DataFrame(records)
            df_encoded = self.encode_features(df, category=category)
            train_df, test_df = self.split_train_test(df_encoded)

            train_path = target_dir / f"{category}_train.csv"
            test_path = target_dir / f"{category}_test.csv"

            train_df.to_csv(train_path, index=True)
            test_df.to_csv(test_path, index=True)

            logger.info(
                f"Saved {category} datasets -> train: {train_df.shape} to {train_path}, test: {test_df.shape} to {test_path}"
            )
            results[category] = (train_df, test_df)

        return results

    def split_train_test(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        np.random.seed(self.config.random_seed)
        shuffled_indices = np.random.permutation(len(df))
        split_point = int(len(shuffled_indices) * self.config.train_ratio)

        train_indices = shuffled_indices[:split_point]
        test_indices = shuffled_indices[split_point:]

        train_df = df.iloc[train_indices].copy()
        test_df = df.iloc[test_indices].copy()

        train_df.index = pd.RangeIndex(len(train_df))
        test_df.index = pd.RangeIndex(len(test_df))

        return train_df, test_df


def filter_and_preprocess_data(
    input_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_games: Optional[int] = None,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    config = PreprocessorConfig()
    if output_dir:
        config.processed_data_dir = output_dir

    if input_file is None:
        raw_folder = Path(config.raw_data_dir)
        zst_files = sorted(list(raw_folder.glob("*.pgn.zst")) + list(raw_folder.glob("*.pgn")))
        if not zst_files:
            raise FileNotFoundError(f"No .pgn or .pgn.zst files found in {raw_folder}")
        input_file = str(zst_files[0])

    preprocessor = ChessDataPreprocessor(config)
    return preprocessor.process_and_save(input_file, max_games=max_games)


def clean_batched_files():
    pass


def concat_clean_batched_files():
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Clean and preprocess chess PGN game databases.")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to input PGN or PGN.zst file. Defaults to first file found in data/raw.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Path to directory where cleaned train/test CSV files are stored.",
    )
    parser.add_argument(
        "--max-games",
        "-n",
        type=int,
        default=None,
        help="Maximum number of games to parse from the input file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    filter_and_preprocess_data(
        input_file=args.input,
        output_dir=args.output_dir,
        max_games=args.max_games,
    )
