import os
import pytest
import numpy as np
import pandas as pd

from src.modelling import BaseChessModel, LinearRegressionModel, XGBoostRegression


@pytest.fixture
def dummy_chess_data(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    n_samples = 40
    records = []
    for i in range(n_samples):
        rec = {
            "average_elo": 1500.0 + np.random.randn() * 100,
            "number_of_moves": int(np.random.randint(20, 60)),
            "number_of_checks": int(np.random.randint(0, 5)),
            "white_castling": int(np.random.randint(0, 3)),
            "black_castling": int(np.random.randint(0, 3)),
            "number_of_captures": int(np.random.randint(2, 15)),
            "knight_moves": int(np.random.randint(2, 10)),
            "bishop_moves": int(np.random.randint(2, 10)),
            "rook_moves": int(np.random.randint(0, 5)),
            "queen_moves": int(np.random.randint(1, 8)),
            "king_moves": int(np.random.randint(0, 5)),
            "pawn_moves": int(np.random.randint(10, 25)),
            "number_of_blunders": int(np.random.randint(0, 4)),
            "number_of_bad_moves": int(np.random.randint(0, 6)),
            "number_of_dubious_moves": int(np.random.randint(0, 5)),
            "evaluation_variance": float(np.random.rand() * 2),
            "evaluation_iqr": float(np.random.rand() * 2),
            "evaluation_range": float(np.random.rand() * 5),
            "e4": 1 if i % 2 == 0 else 0,
            "d4": 1 if i % 2 == 1 else 0,
        }
        for j in range(60):
            rec[f"eval_{j}"] = float(np.random.randn() * 0.5)
            rec[f"clock_{j}"] = int(np.random.randint(1, 30))
        records.append(rec)

    df = pd.DataFrame(records)
    train_df = df.iloc[:24]
    test_df = df.iloc[24:]

    train_file = data_dir / "rapid_train.csv"
    test_file = data_dir / "rapid_test.csv"
    train_df.to_csv(train_file, index=True)
    test_df.to_csv(test_file, index=True)

    return str(data_dir)


def test_calculate_metrics():
    model = BaseChessModel("dummy", 100, "/tmp")
    y_true = np.array([1500, 1600, 1700])
    y_pred = np.array([1510, 1590, 1710])

    metrics = model.calculate_metrics(y_true, y_pred)
    assert metrics["Mean Squared Error"] == pytest.approx(100.0)
    assert metrics["Median Squared Error"] == pytest.approx(100.0)
    assert metrics["Mean Absolute Error"] == pytest.approx(10.0)
    assert metrics["Median Absolute Error"] == pytest.approx(10.0)
    assert "Mean Squared Dummy Error" in metrics


def test_linear_regression_workflow(dummy_chess_data, tmp_path):
    model = LinearRegressionModel(
        game_type="rapid",
        n_rows=100,
        processed_data_dir=dummy_chess_data,
    )
    model.base_dir = str(tmp_path / "lr_output")
    os.makedirs(model.base_dir, exist_ok=True)
    model.model_path = os.path.join(model.base_dir, "rapid_model.pkl")
    model.evaluation_path = os.path.join(model.base_dir, "rapid_eval.csv")
    model.coefficients_path = os.path.join(model.base_dir, "rapid_coefs.csv")
    model.valid_columns_file = os.path.join(model.base_dir, "rapid_valid_cols.txt")

    model.train()
    model.save()
    assert os.path.exists(model.model_path)

    model.load()
    model.evaluate()
    model.save_and_print_coefficients(verbose=False)
    model.save_and_print_evaluation_outcome()

    assert os.path.exists(model.evaluation_path)
    assert os.path.exists(model.coefficients_path)
    assert os.path.exists(model.valid_columns_file)
    assert "Mean Squared Error" in model.eval_measures


def test_xgboost_workflow(dummy_chess_data, tmp_path):
    model = XGBoostRegression(
        game_type="rapid",
        n_rows=100,
        limit=10,
        processed_data_dir=dummy_chess_data,
    )
    model.base_dir = str(tmp_path / "xgb_output")
    os.makedirs(model.base_dir, exist_ok=True)
    model.model_path = os.path.join(model.base_dir, "rapid_model.pkl")
    model.evaluation_path = os.path.join(model.base_dir, "rapid_eval.csv")
    model.coefficients_path = os.path.join(model.base_dir, "rapid_coefs.csv")

    model.train()
    model.save()
    assert os.path.exists(model.model_path)

    model.load()
    model.evaluate()
    model.save_and_print_coefficients(verbose=False)
    model.save_and_print_evaluation_outcome()

    assert os.path.exists(model.evaluation_path)
    assert os.path.exists(model.coefficients_path)
    assert "Mean Squared Error" in model.eval_measures
