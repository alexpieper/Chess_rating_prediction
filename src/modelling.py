import os
import pickle
import argparse
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BaseChessModel:
    def __init__(
        self,
        game_type: str,
        n_rows: int,
        base_dir: str,
        processed_data_dir: str = os.path.join("data", "processed", "all_games_clean"),
    ):
        self.game_type = game_type
        self.nrows = n_rows
        self.base_dir = base_dir
        self.train_file = os.path.join(processed_data_dir, f"{game_type}_train.csv")
        self.test_file = os.path.join(processed_data_dir, f"{game_type}_test.csv")

        os.makedirs(self.base_dir, exist_ok=True)
        self.model_path = os.path.join(self.base_dir, f"{game_type}_model.pkl")
        self.evaluation_path = os.path.join(self.base_dir, f"{game_type}_eval.csv")
        self.coefficients_path = os.path.join(self.base_dir, f"{game_type}_coefs.csv")
        self.valid_columns_file = os.path.join(self.base_dir, f"{game_type}_valid_cols.txt")

        self.model = None
        self.eval_measures: Dict[str, float] = {}

    def save(self):
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {self.model_path}")

    def load(self):
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info(f"Model loaded from {self.model_path}")

    def calculate_metrics(self, y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        errors = y_pred - y_true
        abs_errors = np.abs(errors)
        sq_errors = errors ** 2

        dummy_guess = float(np.mean(y_true))
        dummy_errors = y_true - dummy_guess
        abs_dummy = np.abs(dummy_errors)
        sq_dummy = dummy_errors ** 2

        self.eval_measures = {
            "Mean Squared Error": float(np.mean(sq_errors)),
            "Median Squared Error": float(np.median(sq_errors)),
            "Mean Absolute Error": float(np.mean(abs_errors)),
            "Median Absolute Error": float(np.median(abs_errors)),
            "Mean Squared Dummy Error": float(np.mean(sq_dummy)),
            "Median Squared Dummy Error": float(np.median(sq_dummy)),
            "Mean Absolute Dummy Error": float(np.mean(abs_dummy)),
            "Median Absolute Dummy Error": float(np.median(abs_dummy)),
        }
        return self.eval_measures

    def save_and_print_evaluation_outcome(self):
        measure_df = pd.DataFrame(self.eval_measures.items(), columns=["Measure", "Value"])
        measure_df.to_csv(self.evaluation_path, index=False)
        for measure, value in self.eval_measures.items():
            print(f"{measure}: {value:.2f}")


class LinearRegressionModel(BaseChessModel):
    def __init__(
        self,
        game_type: str,
        n_rows: int = 1_000_000,
        processed_data_dir: str = os.path.join("data", "processed", "all_games_clean"),
    ):
        base_dir = os.path.join("trained_models", "linear_regression", str(n_rows))
        super().__init__(game_type, n_rows, base_dir, processed_data_dir)
        self.model = LinearRegression()
        self.valid_columns: List[str] = []
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

    def clean_dataset(self, df: pd.DataFrame, load_new: bool) -> pd.DataFrame:
        if load_new:
            sums = df.sum(axis=0)
            self.valid_columns = [
                col
                for col in df.columns
                if (abs(sums[col]) > 10 or col == "average_elo")
                and not ("eval_" in col or "clock_" in col)
            ]
            df = df[self.valid_columns]
            with open(self.valid_columns_file, "w") as f:
                for line in self.valid_columns:
                    f.write(f"{line}\n")
        else:
            with open(self.valid_columns_file, "r") as file:
                self.valid_columns = [line.rstrip() for line in file]
            existing_cols = [c for c in self.valid_columns if c in df.columns]
            df = df.reindex(columns=self.valid_columns, fill_value=0)
        return df

    def train(self):
        print("Start: reading training data")
        self.train_df = pd.read_csv(self.train_file, index_col=0, nrows=self.nrows)
        self.train_df = self.clean_dataset(self.train_df, load_new=True)

        x_train = self.train_df.drop(columns=["average_elo"])
        y_train = self.train_df["average_elo"]

        print("Start: training")
        self.model.fit(x_train, y_train)

        self.train_df = pd.DataFrame()

    def evaluate(self):
        print("Start: reading testing data")
        self.test_df = pd.read_csv(self.test_file, index_col=0, nrows=self.nrows)
        self.test_df = self.clean_dataset(self.test_df, load_new=False)

        print("Start: evaluation")
        x_test = self.test_df.drop(columns=["average_elo"])
        y_test = self.test_df["average_elo"]

        y_pred = self.model.predict(x_test)
        self.calculate_metrics(y_test, y_pred)

    def save_and_print_coefficients(self, verbose: bool = False):
        feature_cols = [c for c in self.valid_columns if c != "average_elo"]
        coefficients = self.model.coef_
        coef_df = pd.DataFrame({"Feature": feature_cols, "value": coefficients})
        coef_df.to_csv(self.coefficients_path, index=False)

        if verbose:
            for feat, coef in zip(feature_cols, coefficients):
                print(f"{feat:<50} {coef:.2f}")


class XGBoostRegression(BaseChessModel):
    def __init__(
        self,
        game_type: str,
        n_rows: int = 1_000_000,
        limit: int = 60,
        processed_data_dir: str = os.path.join("data", "processed", "all_games_clean"),
    ):
        self.limit = limit
        base_dir = os.path.join("trained_models", "xgboost_only_ts", str(n_rows), str(limit))
        super().__init__(game_type, n_rows, base_dir, processed_data_dir)

        self.useful_cols = (
            [f"eval_{i}" for i in range(limit)]
            + [f"clock_{i}" for i in range(limit)]
            + ["average_elo"]
        )
        self.usefuls_cols = self.useful_cols
        self.model = xgb.XGBRegressor()
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

    def train(self):
        print("Start: reading training data")
        self.train_df = pd.read_csv(self.train_file, nrows=self.nrows, usecols=self.useful_cols)
        x_train = self.train_df.drop(columns=["average_elo"])
        y_train = self.train_df["average_elo"]

        print("Start: training")
        self.model.fit(x_train, y_train)

        self.train_df = pd.DataFrame()

    def evaluate(self):
        print("Start: reading testing data")
        self.test_df = pd.read_csv(self.test_file, nrows=self.nrows, usecols=self.useful_cols)

        print("Start: evaluation")
        x_test = self.test_df.drop(columns=["average_elo"])
        y_test = self.test_df["average_elo"]

        y_pred = self.model.predict(x_test)
        self.calculate_metrics(y_test, y_pred)

    def save_and_print_coefficients(self, verbose: bool = False):
        feature_cols = [c for c in self.useful_cols if c != "average_elo"]
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            coef_df = pd.DataFrame({"Feature": feature_cols, "value": importances})
            coef_df.to_csv(self.coefficients_path, index=False)
            if verbose:
                for feat, imp in zip(feature_cols, importances):
                    print(f"{feat:<50} {imp:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate chess elo prediction models.")
    parser.add_argument(
        "--game-types",
        "-g",
        nargs="+",
        default=["classical", "bullet", "rapid", "blitz"],
        help="List of game types to train on (classical, bullet, rapid, blitz)",
    )
    parser.add_argument(
        "--n-rows",
        "-n",
        type=int,
        default=1_000_000,
        help="Maximum rows to load for training and testing.",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=60,
        help="Timeseries length limit for XGBoost.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    game_types = args.game_types
    n_rows = args.n_rows

    print("================ Training Linear Regression Models ================")
    for game_type in game_types:
        print(f"\n--- Gametype: {game_type}, n_rows = {n_rows} ---")
        lr_model = LinearRegressionModel(game_type, n_rows)
        try:
            lr_model.train()
            lr_model.save()
            lr_model.load()
            lr_model.evaluate()
            lr_model.save_and_print_coefficients(verbose=False)
            lr_model.save_and_print_evaluation_outcome()
        except Exception as e:
            logger.error(f"Error training LinearRegression for {game_type}: {e}")

    print("\n================ Training XGBoost Models ================")
    for game_type in game_types:
        print(f"\n--- Gametype: {game_type}, n_rows = {n_rows} ---")
        xgb_model = XGBoostRegression(game_type, n_rows, args.limit)
        try:
            xgb_model.train()
            xgb_model.save()
            xgb_model.load()
            xgb_model.evaluate()
            xgb_model.save_and_print_coefficients(verbose=False)
            xgb_model.save_and_print_evaluation_outcome()
        except Exception as e:
            logger.error(f"Error training XGBoost for {game_type}: {e}")
