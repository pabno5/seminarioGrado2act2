"""Etapa 2 — Validación de calidad de datos con Great Expectations (API 1.x).

Valida train.csv y test.csv contra la misma suite de expectativas.
Escribe un reporte en reports/data_validation.json y termina con código 1
si alguna expectativa falla, lo que detiene el pipeline antes de entrenar.
"""
import json
import os
import sys
from pathlib import Path

# Desactiva el envío de estadísticas de uso de GX (debe ir antes del import)
os.environ.setdefault("GX_ANALYTICS_ENABLED", "False")

import great_expectations as gx  # noqa: E402
import pandas as pd  # noqa: E402
from great_expectations.data_context.types.base import ProgressBarsConfig  # noqa: E402
from input_validation import FEATURE_SCHEMA, TARGET_SCHEMA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_PATH = ROOT / "reports" / "data_validation.json"

# Columnas, rangos y clases salen del mismo esquema que usa la validación
# sin librerías (input_validation.py), así ambas nunca se contradicen.
FEATURES = list(FEATURE_SCHEMA)
TARGET = "species"
RANGES = {column: (rule.min_value, rule.max_value) for column, rule in FEATURE_SCHEMA.items()}
CLASSES = sorted(TARGET_SCHEMA[TARGET].allowed)


def build_suite() -> gx.ExpectationSuite:
    """Define las expectativas que deben cumplir los datos."""
    suite = gx.ExpectationSuite(name="iris_suite")
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchSet(column_set=FEATURES + [TARGET], exact_match=True)
    )
    suite.add_expectation(gx.expectations.ExpectTableRowCountToBeBetween(min_value=1))
    for column in FEATURES + [TARGET]:
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=column))
    for column, (low, high) in RANGES.items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(column=column, min_value=low, max_value=high)
        )
    suite.add_expectation(
        gx.expectations.ExpectColumnDistinctValuesToBeInSet(column=TARGET, value_set=CLASSES)
    )
    return suite


def main() -> int:
    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = ProgressBarsConfig(globally=False)
    source = context.data_sources.add_pandas(name="iris")
    suite = build_suite()

    report = {"success": True, "datasets": {}}
    for name in ("train", "test"):
        df = pd.read_csv(DATA_DIR / f"{name}.csv")
        asset = source.add_dataframe_asset(name=name)
        batch = asset.add_batch_definition_whole_dataframe(f"{name}_batch").get_batch(
            batch_parameters={"dataframe": df}
        )
        result = batch.validate(suite)

        failed = [
            {
                "expectation": r.expectation_config.type,
                "column": r.expectation_config.kwargs.get("column"),
            }
            for r in result.results
            if not r.success
        ]
        report["datasets"][name] = {
            "success": bool(result.success),
            "rows": len(df),
            "evaluated": len(result.results),
            "failed": failed,
        }
        report["success"] = report["success"] and bool(result.success)

        status = "OK" if result.success else "FALLÓ"
        print(f"[{status}] {name}.csv: {len(result.results) - len(failed)}/{len(result.results)} expectativas cumplidas")
        for f in failed:
            print(f"    - {f['expectation']} (columna: {f['column']})")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if not report["success"]:
        print("Validación de datos fallida: el pipeline se detiene antes de entrenar.")
        return 1
    print(f"Validación de datos superada -> {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
