#!/usr/bin/env python
"""
Orquestador del pipeline MLOps (Iris) — Semana 2.

Un solo comando ejecuta todo el flujo definido en dvc.yaml:
    prepare → validate_input (Python puro) → validate (Great Expectations)
            → train (MLflow) → test (pytest)

Uso:
    python run_pipeline.py                 # pipeline completo (igual que "all")
    python run_pipeline.py install         # instala las dependencias
    python run_pipeline.py train           # hasta "train" (y las etapas previas si hace falta)
    python run_pipeline.py all --force     # re-ejecuta todo aunque nada haya cambiado
    python run_pipeline.py metrics | dag | ui | clean

Funciona igual en Windows, macOS y Linux. El Makefile solo llama a este script.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = ["prepare", "validate_input", "validate", "train", "test"]
GENERATED_DIRS = ["data/processed", "models", "reports"]

# módulo a importar -> paquete de pip
REQUIRED = {
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "joblib": "joblib",
    "yaml": "pyyaml",
    "mlflow": "mlflow",
    "dvc": "dvc",
    "pytest": "pytest",
    "great_expectations": "great-expectations",
}


class PipelineError(Exception):
    """Error controlado: se muestra el mensaje y se sale con código 1."""


# --------------------------------------------------------------------------- utilidades

def banner(text: str) -> None:
    print(f"\n{'=' * 64}\n  {text}\n{'=' * 64}", flush=True)


def subprocess_env() -> dict:
    """Entorno para los subprocesos.

    DVC ejecuta cada etapa con el comando `python`. Anteponemos al PATH la
    carpeta del intérprete actual para que ese `python` sea el mismo entorno
    (venv/conda) desde el que se lanzó este script.
    """
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env.setdefault("PYTHONIOENCODING", "utf-8")  # evita errores de acentos en la consola de Windows
    return env


def run(cmd: list[str], description: str) -> None:
    shown = " ".join("python" if c == sys.executable else c for c in cmd)
    print(f"\n>>> {description}\n    $ {shown}\n", flush=True)
    result = subprocess.run(cmd, cwd=ROOT, env=subprocess_env())
    if result.returncode != 0:
        raise PipelineError(f"Falló: {description} (código de salida {result.returncode}).")


def dvc_cmd(*args: str) -> list[str]:
    return [sys.executable, "-m", "dvc", *args]


# --------------------------------------------------------------------------- comprobaciones

def check_dependencies() -> None:
    missing = [pkg for module, pkg in REQUIRED.items() if importlib.util.find_spec(module) is None]
    if missing:
        raise PipelineError(
            "Faltan dependencias: " + ", ".join(missing)
            + "\n  Instálalas con:  python run_pipeline.py install   (o: make install)"
        )

    stage_python = shutil.which("python", path=subprocess_env()["PATH"])
    if stage_python is None:
        print(
            "AVISO: DVC ejecuta las etapas con el comando `python`, que no se encuentra en el PATH.\n"
            "       Activa un entorno virtual (python -m venv .venv) antes de lanzar el pipeline.",
            flush=True,
        )
    elif Path(stage_python).resolve() != Path(sys.executable).resolve():
        print(
            f"AVISO: las etapas usarán {stage_python}, distinto del intérprete actual "
            f"({sys.executable}). Si fallan imports, activa tu entorno virtual.",
            flush=True,
        )


def ensure_dvc_initialized() -> None:
    """Inicializa DVC (y Git si hace falta) la primera vez."""
    if (ROOT / ".dvc").is_dir():
        return

    banner("Primera ejecución: inicializando DVC")
    if shutil.which("git") is None:
        run(dvc_cmd("init", "--no-scm"), "Inicializar DVC sin Git (Git no está instalado)")
        return

    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=ROOT, capture_output=True, text=True
    )
    if top.returncode != 0:
        run(["git", "init"], "Inicializar repositorio Git")
        run(dvc_cmd("init"), "Inicializar DVC")
    elif Path(top.stdout.strip()).resolve() == ROOT:
        run(dvc_cmd("init"), "Inicializar DVC")
    else:
        # El proyecto vive dentro de un repo más grande
        run(dvc_cmd("init", "--subdir"), "Inicializar DVC en subcarpeta del repositorio")


# --------------------------------------------------------------------------- resumen

def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def print_summary(stages_run: list[str], elapsed: float) -> None:
    """Muestra solo los resultados de las etapas incluidas en esta ejecución."""
    banner("RESUMEN")

    basic = read_json(ROOT / "reports" / "input_validation.json")
    if "validate_input" in stages_run and basic:
        rows = sum(f["rows_checked"] for f in basic["files"])
        issues = sum(len(f["issues"]) for f in basic["files"])
        estado = "OK" if basic["success"] else "FALLÓ"
        print(f"  Tipos/nulos/rangos  : {estado} ({rows} filas revisadas, {issues} problemas)")

    validation = read_json(ROOT / "reports" / "data_validation.json")
    if "validate" in stages_run and validation:
        failed = sum(len(d["failed"]) for d in validation["datasets"].values())
        estado = "OK" if validation["success"] else "FALLÓ"
        print(f"  Great Expectations  : {estado} ({len(validation['datasets'])} archivos, "
              f"{failed} expectativas fallidas)")

    metrics = read_json(ROOT / "reports" / "metrics.json")
    params = None
    try:
        import yaml
        params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — el resumen nunca debe romper el pipeline
        pass
    if "train" in stages_run and metrics:
        umbral = f"  (umbral {params['evaluate']['min_accuracy']})" if params else ""
        print(f"  Accuracy (test)     : {metrics['accuracy']:.4f}{umbral}")
        print(f"  MLflow Run ID       : {metrics.get('mlflow_run_id', '-')}")

    junit = ROOT / "reports" / "pytest.xml"
    if "test" in stages_run and junit.exists():
        try:
            root = ET.parse(junit).getroot()
            suite = root if root.tag == "testsuite" else root.find("testsuite")
            total = int(suite.get("tests", 0))
            bad = int(suite.get("failures", 0)) + int(suite.get("errors", 0))
            print(f"  Pruebas pytest      : {total - bad}/{total} superadas")
        except (ET.ParseError, AttributeError, ValueError):
            pass

    print(f"  Tiempo total        : {elapsed:.1f} s")


# --------------------------------------------------------------------------- comandos

def cmd_install(_args) -> None:
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], "Instalar dependencias")


def cmd_repro(args) -> None:
    check_dependencies()
    ensure_dvc_initialized()

    target = [] if args.command == "all" else [args.command]
    stages_run = STAGES if not target else STAGES[: STAGES.index(args.command) + 1]
    force = ["--force"] if args.force else []
    label = "pipeline completo" if not target else f"pipeline hasta la etapa '{args.command}'"

    banner(f"Ejecutando {label}: " + " → ".join(stages_run))
    start = time.perf_counter()
    run(dvc_cmd("repro", *force, *target), f"dvc repro ({label})")
    print_summary(stages_run, time.perf_counter() - start)
    print("\n✔ Pipeline completado correctamente.")


def cmd_metrics(_args) -> None:
    run(dvc_cmd("metrics", "show"), "Métricas registradas por DVC")


def cmd_dag(_args) -> None:
    run(dvc_cmd("dag"), "Grafo de etapas del pipeline")


def cmd_ui(_args) -> None:
    print("Abriendo MLflow UI en http://127.0.0.1:5000  (Ctrl+C para salir)")
    try:
        run([sys.executable, "-m", "mlflow", "ui"], "MLflow UI")
    except KeyboardInterrupt:
        pass


def cmd_clean(_args) -> None:
    for rel in GENERATED_DIRS:
        path = ROOT / rel
        if path.exists():
            shutil.rmtree(path)
            print(f"  eliminado: {rel}/")
    print("Limpieza lista. El historial de MLflow y la caché de DVC se conservan:\n"
          "el próximo 'all' restaura datos y modelo desde la caché si nada cambió y\n"
          "vuelve a ejecutar la validación y las pruebas (usa --force para re-ejecutar todo).")


COMMANDS = {
    "all": (cmd_repro, "Pipeline completo, las 5 etapas (por defecto)"),
    "prepare": (cmd_repro, "Solo preparación de datos"),
    "validate_input": (cmd_repro, "Hasta la validación de tipos, nulos y rangos (sin librerías)"),
    "validate": (cmd_repro, "Hasta la validación con Great Expectations"),
    "train": (cmd_repro, "Hasta el entrenamiento (MLflow)"),
    "test": (cmd_repro, "Hasta las pruebas del modelo (pytest) = pipeline completo"),
    "install": (cmd_install, "Instala requirements.txt en el entorno actual"),
    "metrics": (cmd_metrics, "Muestra las métricas (dvc metrics show)"),
    "dag": (cmd_dag, "Muestra el grafo de etapas (dvc dag)"),
    "ui": (cmd_ui, "Abre la interfaz de MLflow"),
    "clean": (cmd_clean, "Borra data/processed, models y reports"),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Orquestador del pipeline MLOps Iris: prepare → validate_input → validate → train → test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Comandos:\n" + "\n".join(f"  {name:<15} {desc}" for name, (_, desc) in COMMANDS.items()),
    )
    parser.add_argument("command", nargs="?", default="all", choices=COMMANDS, metavar="comando",
                        help="ver lista abajo (por defecto: all)")
    parser.add_argument("-f", "--force", action="store_true",
                        help="re-ejecuta las etapas aunque DVC no detecte cambios")
    args = parser.parse_args()

    # Evita UnicodeEncodeError en consolas de Windows con codificación antigua
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass

    os.chdir(ROOT)
    try:
        COMMANDS[args.command][0](args)
    except PipelineError as exc:
        print(f"\n✘ {exc}\n  El pipeline se detuvo; revisa el mensaje de error de arriba.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
