"""El contrato de datos de inferencia se cumple de verdad.

Comprueba que:
  - el contrato es coherente con el esquema que aplica el validador,
  - el ejemplo válido pasa y obtiene la predicción prometida,
  - el ejemplo inválido falla exactamente con los errores declarados.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from input_validation import FEATURE_SCHEMA, validate_record

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "contracts" / "iris_inference_contract.json"
MODEL_PATH = ROOT / "models" / "model.pkl"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


@pytest.fixture(scope="module")
def contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def field_to_column(contract):
    """Campo de la API (sepal_length) -> columna del modelo (sepal length (cm))."""
    return {name: spec["x-model-column"] for name, spec in contract["input_schema"]["properties"].items()}


def to_model_record(payload, field_to_column):
    # Los campos desconocidos se conservan para que el validador los reporte
    return {field_to_column.get(k, k): v for k, v in payload.items()}


def check_pairs(issues, field_to_column):
    column_to_field = {v: k for k, v in field_to_column.items()}
    return sorted((column_to_field.get(i.column, i.column), i.check) for i in issues)


# ----------------------------------------------------------------- coherencia

def test_contract_has_required_sections(contract):
    assert {"contract", "input_schema", "validation_rules", "output_schema", "examples"} <= contract.keys()
    assert {"valid", "invalid"} <= contract["examples"].keys()


def test_contract_matches_validator_schema(contract, field_to_column):
    schema = contract["input_schema"]
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False
    assert set(field_to_column.values()) == set(FEATURE_SCHEMA)
    for name, spec in schema["properties"].items():
        rule = FEATURE_SCHEMA[spec["x-model-column"]]
        assert spec["type"] == "number" and rule.dtype is float, name
        assert (spec["minimum"], spec["maximum"]) == (rule.min_value, rule.max_value), name


# ----------------------------------------------------------------- ejemplos

def test_valid_example_passes(contract, field_to_column):
    payload = contract["examples"]["valid"]["input"]
    issues = validate_record(to_model_record(payload, field_to_column), FEATURE_SCHEMA, strict_types=True)
    assert issues == []


def test_invalid_example_fails_with_declared_errors(contract, field_to_column):
    example = contract["examples"]["invalid"]
    issues = validate_record(to_model_record(example["input"], field_to_column), FEATURE_SCHEMA,
                             strict_types=True)
    declared = sorted((e["field"], e["check"]) for e in example["expected_errors"])
    assert check_pairs(issues, field_to_column) == declared


def test_valid_example_gets_expected_prediction(contract, field_to_column):
    example = contract["examples"]["valid"]
    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([to_model_record(example["input"], field_to_column)])[model.feature_names_in_]
    class_id = int(model.predict(row)[0])
    assert {"prediction": CLASS_NAMES[class_id], "class_id": class_id} == example["expected_output"]


def test_examples_against_json_schema(contract):
    """El contrato también es un JSON Schema estándar (lo usan FastAPI y otras herramientas)."""
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator
    validator.check_schema(contract["input_schema"])
    validator.check_schema(contract["output_schema"])

    examples = contract["examples"]
    validator(contract["input_schema"]).validate(examples["valid"]["input"])
    validator(contract["output_schema"]).validate(examples["valid"]["expected_output"])
    errors = list(validator(contract["input_schema"]).iter_errors(examples["invalid"]["input"]))
    assert len(errors) >= len({e["check"] for e in examples["invalid"]["expected_errors"]})
