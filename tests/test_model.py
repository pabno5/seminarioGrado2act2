"""Etapa 4 — Validación del modelo con pytest."""
from pathlib import Path

import joblib
import pandas as pd
import pytest
import yaml
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "model.pkl"
TARGET = "species"


@pytest.fixture(scope="module")
def params():
    return yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def test_data():
    return pd.read_csv(DATA_DIR / "test.csv")


def test_data_loading():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    assert not train.empty and not test.empty
    assert TARGET in train.columns and TARGET in test.columns


def test_model_exists(model):
    assert model is not None
    assert hasattr(model, "predict")


def test_model_accuracy(model, test_data, params):
    X_test, y_test = test_data.drop(TARGET, axis=1), test_data[TARGET]
    accuracy = accuracy_score(y_test, model.predict(X_test))
    min_accuracy = params["evaluate"]["min_accuracy"]
    assert accuracy >= min_accuracy, f"accuracy {accuracy:.3f} < umbral {min_accuracy}"


def test_predictions_are_valid_classes(model, test_data):
    predictions = model.predict(test_data.drop(TARGET, axis=1))
    assert len(predictions) == len(test_data)
    assert set(predictions) <= {0, 1, 2}


def test_known_setosa_sample(model, test_data):
    # Una flor con pétalos pequeños es inequívocamente setosa (clase 0)
    sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=test_data.drop(TARGET, axis=1).columns)
    assert model.predict(sample)[0] == 0
