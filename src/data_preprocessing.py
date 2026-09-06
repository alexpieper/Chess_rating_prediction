import os
import re
import io
import argparse
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterator, Union
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
import zstandard as zstd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return int(parts[0])

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

            if "?" in annot or nag == "2":
                if "??" in annot or nag == "4":
                    num_blunder += 1
                elif "?!" in annot or nag == "6":
                    num_dubious += 1
                else:
                    num_bad += 1
            elif nag == "4":
                num_blunder += 1
            elif nag == "6":
                num_dubious += 1

        num_moves = len(all_moves)
        if num_moves == 0 or num_moves > upper_limit or len(all_evals) == 0:
            return None

        white_moves = all_moves[0::2]
        black_moves = all_moves[1::2]
        first_white = white_moves[0] if white_moves else ""
        first_black = black_moves[0] if black_moves else ""

        termination = headers.get("Termination", "Normal")
        result = headers.get("Result", "*")
        if termination == "Normal":
            if result in ("1-0", "0-1"):
                termination = "Checkmate" if all_moves and "#" in all_moves[-1] else "Resignation"
            elif result == "1/2-1/2":
                termination = "Stalemate"

        opening = headers.get("Opening", "Unknown")
        opening_hl = opening.split(":")[0]

        numeric_evals = [e for e in all_evals if abs(e) != 100.0]
        ev_var = float(np.var(numeric_evals)) if numeric_evals else 0.0
        ev_iqr = float(np.percentile(numeric_evals, 75) - np.percentile(numeric_evals, 25)) if numeric_evals else 0.0
        ev_range = float(np.max(numeric_evals) - np.min(numeric_evals)) if numeric_evals else 0.0

        parsed_clock = [abs(int(all_clocks[i + 2] - all_clocks[i])) for i in range(len(all_clocks) - 2)]
        eval_padded = all_evals[:upper_limit] + [0.0] * max(0, upper_limit - len(all_evals))
        clock_padded = parsed_clock[:upper_limit] + [0] * max(0, upper_limit - len(parsed_clock))

        record = {
            "average_elo": 0.5 * (white_elo + black_elo),
            "number_of_moves": num_moves,
            "number_of_checks": sum(1 for m in all_moves if "+" in m),
            "white_castling": 1 if "O-O" in white_moves else (2 if "O-O-O" in white_moves else 0),
            "black_castling": 1 if "O-O" in black_moves else (2 if "O-O-O" in black_moves else 0),
            "number_of_captures": sum(1 for m in all_moves if "x" in m),
            "knight_moves": sum(1 for m in all_moves if "N" in m),
            "bishop_moves": sum(1 for m in all_moves if "B" in m),
            "rook_moves": sum(1 for m in all_moves if "R" in m),
            "queen_moves": sum(1 for m in all_moves if "Q" in m),
            "king_moves": sum(1 for m in all_moves if "K" in m),
            "pawn_moves": sum(1 for m in all_moves if not any(c in m for c in ("N", "B", "R", "Q", "K", "O"))),
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
            df_encoded = self.encode_features(df)
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

    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        encoded_df = df.copy()
        for col in self.config.categorical_columns:
            if col not in encoded_df.columns:
                continue
            dummies = pd.get_dummies(encoded_df[col], dtype=np.int8)
            if col == "opening_high_level":
                dummies = dummies.add_suffix("_high_level")
            encoded_df = encoded_df.join(dummies)

        encoded_df = encoded_df.drop(
            columns=[col for col in self.config.categorical_columns if col in encoded_df.columns]
        )
        return encoded_df

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
