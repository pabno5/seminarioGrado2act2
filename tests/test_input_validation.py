"""Pruebas del validador de entrada escrito sin librerías externas."""
import pytest

from input_validation import (
    DATASET_SCHEMA,
    FEATURE_SCHEMA,
    main,
    validate_csv,
    validate_record,
    validate_rows,
)

VALID = {
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2,
    "species": 0,
}
HEADER = ",".join(VALID)


def with_value(column, value):
    record = dict(VALID)
    record[column] = value
    return record


def write_csv(tmp_path, *lines, name="datos.csv"):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ----------------------------------------------------------------- registros válidos

def test_valid_record_has_no_issues():
    assert validate_record(VALID, DATASET_SCHEMA) == []


def test_numeric_text_is_accepted_like_in_a_csv():
    record = {k: str(v) for k, v in VALID.items()}
    assert validate_record(record, DATASET_SCHEMA) == []


def test_range_limits_are_inclusive():
    assert validate_record(with_value("sepal length (cm)", 4.0), DATASET_SCHEMA) == []
    assert validate_record(with_value("sepal length (cm)", 8.0), DATASET_SCHEMA) == []


def test_feature_schema_validates_inference_input_without_label():
    features = {k: v for k, v in VALID.items() if k != "species"}
    assert validate_record(features, FEATURE_SCHEMA) == []


# ----------------------------------------------------------------- nulos

@pytest.mark.parametrize("value", [None, "", "   ", "NA", "n/a", "NaN", "null", "None", "?", float("nan")])
def test_null_values_are_detected(value):
    issues = validate_record(with_value("petal length (cm)", value), DATASET_SCHEMA)
    assert [(i.check, i.column) for i in issues] == [("nulo", "petal length (cm)")]


# ----------------------------------------------------------------- tipos

@pytest.mark.parametrize(
    "column, value",
    [
        ("sepal length (cm)", "abc"),      # texto no numérico
        ("sepal length (cm)", "5,1"),      # coma decimal
        ("sepal length (cm)", True),       # booleano (en Python es subclase de int)
        ("sepal length (cm)", [5.1]),      # lista
        ("sepal width (cm)", "inf"),       # no finito
        ("species", 1.5),                  # decimal donde se espera entero
        ("species", "1.0"),                # texto decimal donde se espera entero
        ("species", "setosa"),             # etiqueta textual en vez de código
    ],
)
def test_wrong_types_are_detected(column, value):
    issues = validate_record(with_value(column, value), DATASET_SCHEMA)
    assert [(i.check, i.column) for i in issues] == [("tipo", column)]


def test_strict_mode_rejects_numbers_sent_as_text():
    # En un CSV todo es texto; en JSON (contrato de inferencia) un número va sin comillas
    record = with_value("petal length (cm)", "1.4")
    assert validate_record(record, DATASET_SCHEMA) == []
    issues = validate_record(record, DATASET_SCHEMA, strict_types=True)
    assert [(i.check, i.column) for i in issues] == [("tipo", "petal length (cm)")]


# ----------------------------------------------------------------- rangos

@pytest.mark.parametrize(
    "column, value",
    [
        ("sepal length (cm)", 9.5),        # demasiado grande
        ("sepal length (cm)", 3.0),        # demasiado pequeño
        ("sepal length (cm)", 51.0),       # medido en mm en vez de cm
        ("petal width (cm)", -0.2),        # negativo
        ("petal width (cm)", 0.0),         # una medida física no puede ser 0
        ("species", 3),                    # clase inexistente
        ("species", -1),
    ],
)
def test_out_of_range_values_are_detected(column, value):
    issues = validate_record(with_value(column, value), DATASET_SCHEMA)
    assert [(i.check, i.column) for i in issues] == [("rango", column)]


# ----------------------------------------------------------------- estructura y reporte

def test_missing_and_unexpected_columns():
    record = dict(VALID)
    del record["species"]
    record["color"] = "azul"
    issues = validate_record(record, DATASET_SCHEMA)
    assert sorted((i.check, i.column) for i in issues) == [("estructura", "color"), ("estructura", "species")]


def test_all_problems_are_reported_not_only_the_first():
    rows = [
        VALID,
        with_value("sepal length (cm)", None),
        with_value("species", "x"),
        with_value("petal width (cm)", 9.0),
    ]
    report = validate_rows(rows, DATASET_SCHEMA)
    assert not report.is_valid
    assert report.rows_checked == 4
    assert report.invalid_rows == 3
    assert report.counts() == {"estructura": 0, "nulo": 1, "tipo": 1, "rango": 1}


# ----------------------------------------------------------------- archivos CSV

def test_valid_csv(tmp_path):
    path = write_csv(tmp_path, HEADER, "5.1,3.5,1.4,0.2,0", "6.7,3.0,5.2,2.3,2")
    report = validate_csv(path)
    assert report.is_valid and report.rows_checked == 2


def test_csv_issues_point_to_the_file_line(tmp_path):
    path = write_csv(
        tmp_path,
        HEADER,
        "5.1,3.5,1.4,0.2,0",        # línea 2: correcta
        "5.1,,1.4,0.2,0",           # línea 3: nulo
        "5.1,3.5,abc,0.2,1",        # línea 4: tipo
        "5.1,3.5,1.4,0.2,7",        # línea 5: rango
    )
    report = validate_csv(path)
    assert [(i.row, i.check) for i in report.issues] == [(3, "nulo"), (4, "tipo"), (5, "rango")]


def test_csv_with_wrong_header_stops_before_rows(tmp_path):
    path = write_csv(tmp_path, "a,b,c", "1,2,3")
    report = validate_csv(path)
    assert report.rows_checked == 0
    assert {i.check for i in report.issues} == {"estructura"}


def test_csv_rows_with_too_many_or_too_few_values(tmp_path):
    path = write_csv(tmp_path, HEADER, "5.1,3.5,1.4,0.2,0,99", "5.1,3.5,1.4")
    report = validate_csv(path)
    assert [(i.row, i.check) for i in report.issues] == [(2, "estructura"), (3, "nulo"), (3, "nulo")]


def test_empty_and_missing_files(tmp_path):
    assert not validate_csv(write_csv(tmp_path, "")).is_valid
    assert not validate_csv(write_csv(tmp_path, HEADER, name="solo_cabecera.csv")).is_valid
    assert not validate_csv(tmp_path / "no_existe.csv").is_valid


def test_cli_returns_error_code_and_writes_report(tmp_path):
    good = write_csv(tmp_path, HEADER, "5.1,3.5,1.4,0.2,0", name="ok.csv")
    bad = write_csv(tmp_path, HEADER, "5.1,3.5,1.4,0.2,9", name="mal.csv")
    report = tmp_path / "reporte.json"
    assert main([str(good), "--report", str(report)]) == 0
    assert main([str(good), str(bad), "--report", str(report)]) == 1
    assert '"success": false' in report.read_text(encoding="utf-8")
