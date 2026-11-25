"""
Validación de datos de entrada SIN librerías externas (solo biblioteca estándar).

Para cada registro se aplican, en este orden, cuatro comprobaciones:

  1. Estructura : están todas las columnas del esquema y ninguna desconocida.
  2. Nulos      : el valor no es None, NaN, una cadena vacía ni un marcador de
                  nulo ("NA", "N/A", "NaN", "null", "None", "?").
  3. Tipo       : el valor es (o se convierte sin ambigüedad a) el tipo declarado.
                  Los booleanos se rechazan aunque en Python True sea un int, y
                  "1.0" no se acepta como entero.
  4. Rango      : el valor está dentro de los límites lógicos de la variable o,
                  si es categórica, en el conjunto de valores permitidos.

Las comprobaciones se encadenan: si un valor es nulo no se evalúa su tipo, y si
el tipo es incorrecto no se evalúa su rango. Así cada problema aparece una sola
vez y con su causa raíz. Se recorren TODOS los registros y se informan todos los
problemas (no se detiene en el primero).

Uso como script (etapa del pipeline):
    python src/input_validation.py                       # valida train.csv y test.csv
    python src/input_validation.py ruta/a/otro.csv

Uso como módulo (por ejemplo, en la API de la Semana 3):
    from input_validation import FEATURE_SCHEMA, validate_record
    problemas = validate_record({"sepal length (cm)": 5.1, ...}, FEATURE_SCHEMA,
                                strict_types=True)   # JSON: números sin comillas

El contrato oficial de inferencia está en contracts/iris_inference_contract.json;
tests/test_data_contract.py comprueba que este esquema y el contrato coinciden.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import numbers
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES = [ROOT / "data" / "processed" / "train.csv", ROOT / "data" / "processed" / "test.csv"]
DEFAULT_REPORT = ROOT / "reports" / "input_validation.json"

CHECKS = ("estructura", "nulo", "tipo", "rango")
NULL_TOKENS = frozenset({"", "na", "n/a", "nan", "null", "none", "?"})


# --------------------------------------------------------------------------- esquema

@dataclass(frozen=True)
class Rule:
    """Regla de una variable: tipo esperado y valores lógicos admitidos."""

    dtype: type                               # float o int
    min_value: float | None = None            # límite inferior (inclusivo)
    max_value: float | None = None            # límite superior (inclusivo)
    allowed: frozenset | None = None          # valores permitidos (variables categóricas)
    description: str = ""


# Rangos lógicos: todas las medidas son longitudes positivas en cm. Los límites
# cubren lo observado en Iris con margen, de modo que un valor fuera de ellos
# indica un error de captura o de unidades (p. ej. mm en vez de cm).
FEATURE_SCHEMA: dict[str, Rule] = {
    "sepal length (cm)": Rule(float, 4.0, 8.0, description="Longitud del sépalo (observado 4.3–7.9)"),
    "sepal width (cm)": Rule(float, 1.5, 5.0, description="Ancho del sépalo (observado 2.0–4.4)"),
    "petal length (cm)": Rule(float, 0.5, 7.5, description="Longitud del pétalo (observado 1.0–6.9)"),
    "petal width (cm)": Rule(float, 0.1, 3.0, description="Ancho del pétalo (observado 0.1–2.5)"),
}
TARGET_SCHEMA: dict[str, Rule] = {
    "species": Rule(int, allowed=frozenset({0, 1, 2}), description="0=setosa, 1=versicolor, 2=virginica"),
}
# Datos de entrenamiento = variables + etiqueta. Para inferencia usar FEATURE_SCHEMA.
DATASET_SCHEMA: dict[str, Rule] = {**FEATURE_SCHEMA, **TARGET_SCHEMA}


# --------------------------------------------------------------------------- resultados

@dataclass(frozen=True)
class Issue:
    """Un problema encontrado en los datos."""

    check: str                 # "estructura" | "nulo" | "tipo" | "rango"
    column: str | None
    message: str
    row: int | None = None     # línea del archivo (la cabecera es la línea 1)
    value: object = None

    def __str__(self) -> str:
        where = f"línea {self.row}" if self.row is not None else "registro"
        col = f", columna '{self.column}'" if self.column else ""
        return f"[{self.check}] {where}{col}: {self.message}"


@dataclass
class ValidationReport:
    source: str
    rows_checked: int = 0
    issues: list[Issue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.issues

    @property
    def invalid_rows(self) -> int:
        return len({i.row for i in self.issues if i.row is not None})

    def counts(self) -> dict[str, int]:
        found = Counter(i.check for i in self.issues)
        return {check: found.get(check, 0) for check in CHECKS}

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "success": self.is_valid,
            "rows_checked": self.rows_checked,
            "invalid_rows": self.invalid_rows,
            "issues_by_check": self.counts(),
            "issues": [
                {"check": i.check, "row": i.row, "column": i.column,
                 "value": repr(i.value), "message": i.message}
                for i in self.issues
            ],
        }

    def summary(self, max_issues: int = 10) -> str:
        status = "OK" if self.is_valid else "FALLÓ"
        counts = ", ".join(f"{k}={v}" for k, v in self.counts().items())
        lines = [f"[{status}] {self.source}: {self.rows_checked} filas revisadas, "
                 f"{self.invalid_rows} con errores ({counts})"]
        lines += [f"    - {issue}" for issue in self.issues[:max_issues]]
        if len(self.issues) > max_issues:
            lines.append(f"    … y {len(self.issues) - max_issues} problemas más (ver reporte JSON)")
        return "\n".join(lines)


# --------------------------------------------------------------------------- comprobaciones

def is_null(value: object) -> bool:
    """True si el valor representa un dato ausente."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in NULL_TOKENS:
        return True
    return False


def parse_value(
    value: object, rule: Rule, strict_types: bool = False
) -> tuple[float | int | None, str | None]:
    """Convierte el valor al tipo de la regla.

    Devuelve (valor_convertido, None) si el tipo es correcto,
    o (None, mensaje) si no lo es. Con strict_types=True (entradas JSON,
    como la API de inferencia) los números enviados como texto se rechazan.
    
    """
    expected = "número decimal" if rule.dtype is float else "número entero"

    if isinstance(value, bool):  # bool hereda de int: hay que excluirlo explícitamente
        return None, f"se esperaba {expected}, llegó un booleano ({value})"

    if isinstance(value, str) and strict_types:
        return None, f"se esperaba {expected}, llegó el texto {value!r} (envíalo sin comillas)"
    if isinstance(value, str):  # los CSV y formularios entregan texto
        try:
            number = rule.dtype(value.strip())  # int("1.0") falla a propósito
        except ValueError:
            return None, f"se esperaba {expected}, llegó el texto {value!r}"
    elif isinstance(value, numbers.Integral) or (
        isinstance(value, numbers.Real) and rule.dtype is float
    ):  # numbers.* también reconoce los tipos numéricos de numpy/pandas
        number = rule.dtype(value)
    elif isinstance(value, numbers.Real):
        return None, f"se esperaba {expected}, llegó el decimal {value}"
    else:
        return None, f"se esperaba {expected}, llegó un valor de tipo {type(value).__name__}"

    if isinstance(number, float) and not math.isfinite(number):
        return None, f"valor no finito ({value!r})"
    return number, None


def check_range(number: float | int, rule: Rule) -> str | None:
    """Devuelve un mensaje si el valor está fuera de su rango lógico."""
    if rule.allowed is not None and number not in rule.allowed:
        return f"{number} no es un valor permitido {sorted(rule.allowed)}"
    if rule.min_value is not None and number < rule.min_value:
        return f"{number} es menor que el mínimo lógico {rule.min_value}"
    if rule.max_value is not None and number > rule.max_value:
        return f"{number} es mayor que el máximo lógico {rule.max_value}"
    return None


def validate_record(
    record: Mapping[str | None, object],
    schema: Mapping[str, Rule],
    row: int | None = None,
    *,
    strict_types: bool = False,
) -> list[Issue]:
    """Valida un registro (dict columna → valor) contra el esquema.

    strict_types=False acepta números escritos como texto (lo normal en un CSV).
    strict_types=True exige números de verdad, como fija el contrato de inferencia.
    """
    issues: list[Issue] = []

    for column in record:
        if column is None:  # csv.DictReader agrupa aquí los valores sobrantes
            issues.append(Issue("estructura", None, "la fila tiene más valores que columnas", row,
                                record[column]))
        elif column not in schema:
            issues.append(Issue("estructura", column, "columna no esperada", row))

    for column, rule in schema.items():
        if column not in record:
            issues.append(Issue("estructura", column, "falta la columna requerida", row))
            continue
        value = record[column]
        if is_null(value):
            issues.append(Issue("nulo", column, "valor nulo o vacío", row, value))
            continue
        number, type_error = parse_value(value, rule, strict_types)
        if type_error:
            issues.append(Issue("tipo", column, type_error, row, value))
            continue
        range_error = check_range(number, rule)
        if range_error:
            issues.append(Issue("rango", column, range_error, row, value))

    return issues


def validate_rows(
    rows: Iterable[Mapping], schema: Mapping[str, Rule], source: str = "registros"
) -> ValidationReport:
    """Valida una colección de registros (por ejemplo, un lote de predicciones)."""
    report = ValidationReport(source)
    for number, record in enumerate(rows, start=1):
        report.issues += validate_record(record, schema, row=number)
        report.rows_checked += 1
    return report


def validate_csv(path: str | Path, schema: Mapping[str, Rule] = DATASET_SCHEMA) -> ValidationReport:
    """Valida un archivo CSV completo: primero la cabecera, después cada fila."""
    path = Path(path)
    report = ValidationReport(path.name)

    if not path.is_file():
        report.issues.append(Issue("estructura", None, f"no existe el archivo {path}"))
        return report

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        if not header:
            report.issues.append(Issue("estructura", None, "el archivo está vacío", row=1))
            return report

        # Cabecera: si está mal, validar las filas no tendría sentido
        duplicated = sorted({c for c in header if header.count(c) > 1})
        missing = [c for c in schema if c not in header]
        unexpected = [c for c in header if c not in schema]
        for c in duplicated:
            report.issues.append(Issue("estructura", c, "columna duplicada en la cabecera", row=1))
        for c in missing:
            report.issues.append(Issue("estructura", c, "falta la columna requerida", row=1))
        for c in unexpected:
            report.issues.append(Issue("estructura", c, "columna no esperada", row=1))
        if report.issues:
            return report

        for record in reader:
            report.issues += validate_record(record, schema, row=reader.line_num)
            report.rows_checked += 1

    if report.rows_checked == 0:
        report.issues.append(Issue("estructura", None, "el archivo no tiene filas de datos"))
    return report


# --------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Valida tipos, nulos y rangos de archivos CSV de Iris.")
    parser.add_argument("files", nargs="*", type=Path, default=DEFAULT_FILES,
                        help="archivos CSV a validar (por defecto train.csv y test.csv)")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT,
                        help="ruta del reporte JSON (por defecto reports/input_validation.json)")
    args = parser.parse_args(argv)

    reports = [validate_csv(path) for path in args.files]
    for report in reports:
        print(report.summary())

    success = all(r.is_valid for r in reports)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps({"success": success, "files": [r.to_dict() for r in reports]},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not success:
        print("Validación de entrada fallida: corrige los datos antes de continuar.")
        return 1
    shown = args.report.relative_to(ROOT) if args.report.is_relative_to(ROOT) else args.report
    print(f"Validación de entrada superada -> {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
