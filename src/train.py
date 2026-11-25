"""Etapa 3 — Entrenamiento: Random Forest registrado en MLflow."""
import inspect
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "model.pkl"
METRICS_PATH = ROOT / "reports" / "metrics.json"
TARGET = "species"


def main() -> None:
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))["train"]

    mlflow.set_experiment("Proyecto_Final_Grado")

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.set_tag("orquestador", "dvc")

        # Cargar datos procesados
        train = pd.read_csv(DATA_DIR / "train.csv")
        X_train, y_train = train.drop(TARGET, axis=1), train[TARGET]

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"], random_state=params["random_state"]
        )
        model.fit(X_train, y_train)

        # Evaluar con test
        test = pd.read_csv(DATA_DIR / "test.csv")
        X_test, y_test = test.drop(TARGET, axis=1), test[TARGET]
        accuracy = float(model.score(X_test, y_test))

        mlflow.log_metric("accuracy", accuracy)
        # MLflow >= 3.x guarda los modelos sklearn con "skops" (formato seguro) y exige
        # declarar como confiables los tipos internos del árbol; si no, log_model falla.
        extra = {}
        if "skops_trusted_types" in inspect.signature(mlflow.sklearn.log_model).parameters:
            extra["skops_trusted_types"] = ["sklearn.tree._tree.Tree"]
        mlflow.sklearn.log_model(model, name="iris_model", **extra)

        # Guardar modelo localmente para el pipeline y métricas para DVC
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        METRICS_PATH.write_text(
            json.dumps({"accuracy": accuracy, "mlflow_run_id": run.info.run_id}, indent=2),
            encoding="utf-8",
        )

        print(f"Modelo entrenado con accuracy: {accuracy:.4f}")
        print(f"Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
