"""Etapa 1 — Preparación de datos: carga Iris y la divide en train/test."""
from pathlib import Path

import pandas as pd
import yaml
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed"


def main() -> None:
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))["prepare"]

    # Cargar datos de Iris
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data["species"] = iris.target

    # Dividir en train/test
    train, test = train_test_split(
        data, test_size=params["test_size"], random_state=params["random_state"]
    )

    # Guardar (se crea la carpeta si no existe)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)

    print(f"Datos preparados: train={len(train)} filas, test={len(test)} filas -> {OUT_DIR}")


if __name__ == "__main__":
    main()
