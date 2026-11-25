# MLOps Iris: pipeline reproducible de entrenamiento y validación

Pipeline de MLOps de punta a punta para clasificar flores Iris con un Random Forest. Con **un
solo comando** prepara los datos, valida su calidad, entrena y registra el modelo en MLflow y lo
somete a pruebas automáticas. Si cualquier paso falla, el pipeline se detiene con un código de
error distinto de 0.

```
prepare ──► validate_input ──► validate ──► train ──► test
(pandas)    (Python puro:      (Great        (MLflow)   (pytest)
             tipos, nulos,      Expectations)
             rangos)
```

Proyecto académico del Seminario de Grado II (Especialización en Ciencia de Datos), basado en
la guía [MLOps Open Source: Hands-on Lab](https://dialvebe.github.io/mlops-course-santoto/), Semana 2.

## Contenido

1. [Inicio rápido](#1-inicio-rápido)
2. [Requisitos previos](#2-requisitos-previos)
3. [Reproducción paso a paso](#3-reproducción-paso-a-paso)
4. [Cómo funciona el pipeline](#4-cómo-funciona-el-pipeline)
5. [Configuración y experimentos](#5-configuración-y-experimentos)
6. [Referencia de comandos](#6-referencia-de-comandos)
7. [Calidad de datos y contrato de inferencia](#7-calidad-de-datos-y-contrato-de-inferencia)
8. [Estructura del proyecto](#8-estructura-del-proyecto)
9. [Garantías de reproducibilidad](#9-garantías-de-reproducibilidad)
10. [Solución de problemas](#10-solución-de-problemas)

---

## 1. Inicio rápido

Si ya tienes Python 3.11 o superior y Git, estos son todos los pasos. La sección 3 explica cada uno.

**Linux / macOS**

```bash
cd mlops-iris
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
make                      # o: python run_pipeline.py
```

**Windows (PowerShell)**

```powershell
cd mlops-iris
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python run_pipeline.py
```

El resultado esperado termina con `✔ Pipeline completado correctamente.` y 49/49 pruebas superadas.

---

## 2. Requisitos previos

| Requisito | Versión | Obligatorio | Comprobar con |
|---|---|---|---|
| Python | 3.11, 3.12 o 3.13 (probado con 3.11 y 3.13) | Sí | `python --version` |
| Git | 2.x | Recomendado¹ | `git --version` |
| make | cualquiera | No² | `make --version` |
| Espacio en disco | ~1 GB para el entorno virtual | Sí | — |
| Internet | Solo para instalar dependencias | Sí | — |

¹ DVC funciona mejor dentro de un repositorio Git. Si Git no está instalado, el pipeline inicializa DVC en modo `--no-scm` y funciona igual, pero sin versionar el código.
² `make` viene en Linux y macOS. En Windows no viene instalado: usa `python run_pipeline.py`, que hace exactamente lo mismo.

> **¿Por qué Python 3.11 o superior?** Las versiones de referencia de pandas (3.1) y scikit-learn (1.9) lo exigen, y la de Great Expectations (1.25) requiere al menos Python 3.10.

Los datos **no se descargan**: el dataset Iris viene incluido en scikit-learn, así que, una vez
instaladas las dependencias, el pipeline funciona sin conexión.

---

## 3. Reproducción paso a paso

### 3.1 Obtener el código

```bash
git clone (https://github.com/pabno5/seminarioGrado2act2.git) mlops-iris   # o descomprime el .zip del proyecto
cd mlops-iris
```

Todos los comandos siguientes se ejecutan desde la carpeta raíz del proyecto (`mlops-iris/`).

### 3.2 Crear y activar un entorno virtual

El entorno virtual aísla las dependencias. Además, garantiza que DVC ejecute las etapas con el mismo Python en el que instalaste los paquetes.

| Sistema | Crear | Activar |
|---|---|---|
| Linux / macOS | `python3 -m venv .venv` | `source .venv/bin/activate` |
| Windows, PowerShell | `python -m venv .venv` | `.venv\Scripts\Activate.ps1` |
| Windows, cmd | `python -m venv .venv` | `.venv\Scripts\activate.bat` |
| Windows, Git Bash | `python -m venv .venv` | `source .venv/Scripts/activate` |

Al activarlo, el indicador de la terminal muestra `(.venv)`. **Actívalo en cada terminal nueva** antes de ejecutar el pipeline.

### 3.3 Instalar las dependencias

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

| Archivo | Qué instala | Cuándo usarlo |
|---|---|---|
| `requirements-lock.txt` | Las versiones exactas con las que se probó el proyecto | **Para reproducir resultados** (recomendado) |
| `requirements.txt` | Las versiones compatibles más recientes (`make install`) | Para desarrollo o para actualizar dependencias |

Para verificar la instalación:

```bash
python -c "import sklearn, pandas, mlflow, dvc, great_expectations; print('OK')"
```

### 3.4 Ejecutar el pipeline

```bash
make                       # Linux / macOS / Git Bash
python run_pipeline.py     # cualquier sistema, incluido Windows
```

En la primera ejecución el orquestador inicializa Git (si la carpeta aún no es un repositorio) y
DVC. Después ejecuta las 5 etapas en orden. Tarda entre 15 y 60 segundos según el equipo.

### 3.5 Verificar los resultados

La salida debe terminar con un resumen como este. El Run ID de MLflow cambia en cada entrenamiento:

```
================================================================
  RESUMEN
================================================================
  Tipos/nulos/rangos  : OK (150 filas revisadas, 0 problemas)
  Great Expectations  : OK (2 archivos, 0 expectativas fallidas)
  Accuracy (test)     : 1.0000  (umbral 0.8)
  MLflow Run ID       : f1b3deef403b49c686f5ffb94726189c
  Pruebas pytest      : 49/49 superadas
  Tiempo total        : 19.1 s

✔ Pipeline completado correctamente.
```

Lista de comprobación:

| Comprobación | Resultado esperado |
|---|---|
| Código de salida del comando | `0` (en bash: `echo $?`; en PowerShell: `$LASTEXITCODE`) |
| `data/processed/train.csv` y `test.csv` | 120 y 30 filas más la cabecera |
| `models/model.pkl` | Existe |
| `reports/metrics.json` | `"accuracy": 1.0` y el `mlflow_run_id` |
| `reports/input_validation.json` y `reports/data_validation.json` | `"success": true` |
| `python -m dvc status` | `Data and pipelines are up to date.` |
| Volver a ejecutar `make` | Las 5 etapas aparecen como `didn't change, skipping` en menos de 2 s |
| `make ui`, y abrir http://127.0.0.1:5000 | El experimento `Proyecto_Final_Grado` muestra el run con su accuracy |

### 3.6 Guardar el estado en Git

DVC crea los `.gitignore` necesarios para que los datos y el modelo no entren en Git. Lo que
sí se versiona es `dvc.lock`, la "huella" exacta de esta ejecución, junto con los reportes de
`reports/`:

```bash
git add .
git commit -m "Pipeline reproducido: accuracy 1.0"
```

### 3.7 (Opcional) Compartir datos y modelo con DVC

Para que otra persona obtenga los artefactos sin volver a entrenar, configura un almacenamiento remoto. Puede ser una carpeta compartida, S3, Google Drive, etc.:

```bash
python -m dvc remote add -d almacen /ruta/a/carpeta-compartida
python -m dvc push        # sube data/processed y models/
git add .dvc/config && git commit -m "Configura remoto DVC"
# La otra persona, tras clonar:  python -m dvc pull
```

---

## 4. Cómo funciona el pipeline

Hay tres capas, cada una con una responsabilidad:

- **`Makefile`**: atajos para quien tiene `make`. Cada objetivo llama a `run_pipeline.py`.
- **`run_pipeline.py`**: el orquestador. Funciona en Windows, macOS y Linux. Comprueba las dependencias, inicializa Git/DVC si hace falta, ejecuta `dvc repro` con el mismo Python del entorno activo y muestra el resumen.
- **`dvc.yaml`**: la definición del pipeline. DVC solo vuelve a ejecutar las etapas cuyas dependencias, parámetros o salidas cambiaron.

| Etapa | Script | Lee | Produce | Detiene el pipeline si… |
|---|---|---|---|---|
| `prepare` | `src/prepare.py` | `params.yaml → prepare` | `data/processed/train.csv`, `test.csv` | — |
| `validate_input` | `src/input_validation.py` | train/test CSV | `reports/input_validation.json` | hay tipos incorrectos, nulos, valores fuera de rango o columnas mal formadas |
| `validate` | `src/validate_data.py` | train/test CSV | `reports/data_validation.json` | falla alguna de las 12 expectativas de Great Expectations |
| `train` | `src/train.py` | train/test CSV, `params.yaml → train` | `models/model.pkl`, `reports/metrics.json`, run en MLflow | — |
| `test` | `pytest tests/` | modelo, datos, contrato, `params.yaml → evaluate` | `reports/pytest.xml` | falla alguna prueba (p. ej. accuracy < `min_accuracy`) |

Cada etapa depende del reporte de la anterior. Por eso el orden está garantizado: el modelo nunca se entrena con datos que no pasaron las dos validaciones.

---

## 5. Configuración y experimentos

Todos los parámetros están en `params.yaml`:

| Parámetro | Valor | Efecto |
|---|---|---|
| `prepare.test_size` | `0.2` | Proporción de datos para test (30 de 150 filas) |
| `prepare.random_state` | `42` | Semilla de la división train/test |
| `train.n_estimators` | `100` | Número de árboles del Random Forest |
| `train.random_state` | `42` | Semilla del modelo |
| `evaluate.min_accuracy` | `0.8` | Umbral mínimo de accuracy para aprobar el modelo |

Para correr un experimento, edita un valor y vuelve a ejecutar el pipeline. DVC solo repetirá
las etapas afectadas: si cambias `train.n_estimators`, se repiten `train` y `test`, pero no
`prepare` ni las validaciones. Para comparar con el último commit:

```bash
python -m dvc params diff     # qué parámetros cambiaron
python -m dvc metrics diff    # cómo cambió la accuracy
```

MLflow guarda cada entrenamiento como un run del experimento `Proyecto_Final_Grado`. Puedes
compararlos en `make ui`.

---

## 6. Referencia de comandos

Cada comando existe en dos formas equivalentes: `make <comando>` y `python run_pipeline.py <comando>`.

| Comando | Qué hace |
|---|---|
| `all` *(por defecto)* | Pipeline completo (las 5 etapas) |
| `prepare` · `validate_input` · `validate` · `train` · `test` | Ejecuta hasta esa etapa, incluidas las anteriores si hace falta |
| `install` | Instala `requirements.txt` en el entorno activo |
| `metrics` | Muestra las métricas registradas (`dvc metrics show`) |
| `dag` | Dibuja el grafo de etapas (`dvc dag`) |
| `ui` | Abre la interfaz de MLflow en http://127.0.0.1:5000 |
| `clean` | Borra `data/processed/`, `models/` y `reports/` |

Opciones:

- **Forzar la re-ejecución de todas las etapas**: `make all FORCE=1` o `python run_pipeline.py all --force`.
- **Elegir el intérprete** (solo con `make`): `make PYTHON=python`.

Para ejecutar solo las pruebas, sin el pipeline: `python -m pytest tests/ -v`. Necesitan que el modelo y los datos ya estén generados.

---

## 7. Calidad de datos y contrato de inferencia

### 7.1 Validación sin librerías externas (`validate_input`)

`src/input_validation.py` usa solo la biblioteca estándar de Python. Por eso también puede
reutilizarse en la API o en Docker sin añadir dependencias. Cada variable tiene una regla
declarativa:

| Variable | Tipo | Rango lógico | Observado en Iris |
|---|---|---|---|
| `sepal length (cm)` | decimal | 4.0 – 8.0 | 4.3 – 7.9 |
| `sepal width (cm)` | decimal | 1.5 – 5.0 | 2.0 – 4.4 |
| `petal length (cm)` | decimal | 0.5 – 7.5 | 1.0 – 6.9 |
| `petal width (cm)` | decimal | 0.1 – 3.0 | 0.1 – 2.5 |
| `species` | entero | {0, 1, 2} | 0, 1, 2 |

Las comprobaciones se aplican en cadena para cada valor:

1. **Estructura**: no faltan columnas, no hay duplicadas ni desconocidas, y cada fila tiene el número correcto de valores.
2. **Nulos**: rechaza `None`, `NaN`, las cadenas vacías y los marcadores `NA`, `N/A`, `null`, `None` y `?`.
3. **Tipo**: el valor debe convertirse sin ambigüedad. Se rechazan los booleanos, `"5,1"` (coma decimal), `"1.0"` donde se espera un entero, e `inf`.
4. **Rango**: el valor debe estar dentro de los límites (inclusivos) o en el conjunto de clases permitidas.

Si un valor es nulo no se evalúa su tipo, y si el tipo falla no se evalúa el rango. Así cada problema aparece una sola vez, con su causa real. Se revisan todas las filas y cada error indica la línea del archivo:

```
[FALLÓ] datos.csv: 3 filas revisadas, 2 con errores (estructura=0, nulo=1, tipo=2, rango=1)
    - [rango] línea 3, columna 'sepal length (cm)': 51.0 es mayor que el máximo lógico 8.0
    - [nulo] línea 3, columna 'petal length (cm)': valor nulo o vacío
    - [tipo] línea 4, columna 'sepal width (cm)': se esperaba número decimal, llegó el texto 'tres'
    - [tipo] línea 4, columna 'species': se esperaba número entero, llegó el texto '1.0'
```

Formas de usarlo:

- **Por separado**: `python src/input_validation.py archivo.csv`. Devuelve el código 1 si encuentra errores.
- **Para un solo registro**: `validate_record(registro, FEATURE_SCHEMA, strict_types=True)`. Con `strict_types=True` rechaza los números enviados como texto, como exige el contrato.

### 7.2 Great Expectations (`validate`)

`src/validate_data.py` aplica 12 expectativas por archivo con la API 1.x de Great Expectations. Toma columnas, rangos y clases del mismo esquema que `input_validation.py`, así que las dos validaciones nunca se contradicen.

### 7.3 Contrato de datos de inferencia

`contracts/iris_inference_contract.json` es el contrato oficial (v1.0.0) de lo que debe recibir
`POST /predict`. Incluye:

- **Esquema de entrada** en JSON Schema: 4 campos numéricos en cm, todos obligatorios y sin campos adicionales.
- **Reglas de validación**: el orden de las comprobaciones y la respuesta ante una violación (HTTP 422, sin predicción).
- **Esquema de salida**: `{"prediction": "setosa", "class_id": 0}`.
- **Ejemplos**: uno válido con su predicción esperada, y uno inválido que viola las cuatro reglas a la vez, con la lista exacta de errores que debe producir.

`tests/test_data_contract.py` hace que el contrato se cumpla. Si deja de coincidir con el validador o con el modelo, el pipeline falla.

---

## 8. Estructura del proyecto

```
mlops-iris/
├── Makefile                      # atajos: make, make train, make clean…
├── run_pipeline.py               # orquestador multiplataforma
├── dvc.yaml                      # definición del pipeline (etapas, deps, outs)
├── params.yaml                   # parámetros y umbral de calidad
├── requirements.txt              # dependencias (versiones compatibles más recientes)
├── requirements-lock.txt         # dependencias (versiones exactas probadas)
├── contracts/
│   └── iris_inference_contract.json
├── src/
│   ├── prepare.py                # carga Iris y genera train/test
│   ├── input_validation.py       # tipos, nulos y rangos, sin librerías externas
│   ├── validate_data.py          # Great Expectations
│   └── train.py                  # Random Forest + registro en MLflow
├── tests/
│   ├── conftest.py
│   ├── test_model.py             # 5 pruebas del modelo
│   ├── test_input_validation.py  # 38 pruebas del validador
│   └── test_data_contract.py     # 6 pruebas del contrato
│
│   Generado al ejecutar (no editar a mano):
├── dvc.lock                      # huella de la última ejecución → se versiona en Git
├── data/processed/               # train.csv, test.csv → versionado por DVC
├── models/                       # model.pkl → versionado por DVC
├── reports/                      # reportes JSON y pytest.xml
└── mlflow.db, mlruns/            # historial de MLflow → ignorado por Git
```

---

## 9. Garantías de reproducibilidad

| Fuente de variación | Cómo se controla |
|---|---|
| Datos | El dataset Iris viene en scikit-learn: siempre las mismas 150 filas, sin descargas |
| Aleatoriedad | Semillas fijas en `params.yaml` (`random_state: 42`): siempre la misma división y el mismo modelo |
| Versiones de librerías | `requirements-lock.txt` fija las versiones exactas probadas |
| Intérprete de Python | El orquestador obliga a DVC a usar el Python del entorno activo |
| Estado del pipeline | `dvc.lock` registra el hash de cada dependencia y salida. `dvc status` dice si el espacio de trabajo coincide |
| Orden y calidad | Cada etapa depende de la anterior, y las validaciones detienen el pipeline ante datos incorrectos |

**Lo que puede variar y es normal:**

- El **Run ID** de MLflow y las fechas cambian en cada entrenamiento.
- Los **hashes de `dvc.lock`** pueden diferir entre sistemas operativos o versiones de librerías. Por ejemplo, pandas escribe los CSV con saltos de línea `\r\n` en Windows. Para comparar ejecuciones, usa las métricas, no los hashes.
- En un **clon nuevo** no existe la caché de DVC, así que la primera ejecución regenera los datos y el modelo (`prepare` y `train`). Las validaciones y las pruebas solo se repiten si el resultado difiere del registrado en `dvc.lock`. Para verificar todo desde cero, usa `python run_pipeline.py --force`. Para no regenerar nada, usa `dvc pull` (sección 3.7).

---

## 10. Solución de problemas

| Síntoma | Causa | Solución |
|---|---|---|
| `make: command not found` o "make no se reconoce…" | Windows no trae `make` | Usa `python run_pipeline.py <comando>` |
| `✘ Faltan dependencias: …` | El entorno virtual no está activo o no se instaló | Activa `.venv` (3.2) e instala (3.3) |
| `AVISO: DVC ejecuta las etapas con el comando python…` | `python` no apunta al entorno virtual | Activa `.venv` antes de ejecutar |
| `No matching distribution found for pandas==3.1.1` (o scikit-learn) | Python < 3.11 | Instala Python 3.11 o superior y recrea `.venv` |
| PowerShell: "la ejecución de scripts está deshabilitada" | Política de ejecución de Windows | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y vuelve a activar |
| `Stage '…' didn't change, skipping` | No es un error: nada cambió desde la última ejecución | Para forzarla: `make all FORCE=1` |
| `make ui` falla con "Address already in use" | El puerto 5000 está ocupado (en macOS lo usa AirPlay) | `python -m mlflow ui --port 5001` |
| `MlflowException: … untrusted types` | Se usa el `train.py` original de la guía con MLflow ≥ 3 | Usa `src/train.py` de este proyecto, que declara los tipos confiables |
| `module 'great_expectations' has no attribute 'from_pandas'` | Código escrito para Great Expectations 0.x | Usa `src/validate_data.py` de este proyecto (API 1.x) |
| `git commit` pide nombre y correo | Git sin configurar | `git config --global user.name "Tu Nombre"` y `git config --global user.email tu@correo` |
| La etapa `test` falla en `test_model_accuracy` | La accuracy quedó por debajo de `evaluate.min_accuracy` | Revisa los parámetros de `train` o ajusta el umbral en `params.yaml` |

**Empezar de cero**, sin tocar el código ni Git:

```bash
python run_pipeline.py clean      # borra datos, modelo y reportes
# para borrar también el historial de MLflow, elimina la carpeta mlruns/ y el archivo mlflow.db
python run_pipeline.py --force    # vuelve a ejecutar las 5 etapas
```

Al inicializarse, DVC muestra un aviso de analítica anónima de uso. Para desactivarla:
`python -m dvc config core.analytics false`.

---

**Versiones de referencia:** Python 3.11 y 3.13 · MLflow 3.18 · DVC 3.67 · Great Expectations 1.25 ·
scikit-learn 1.9 · pandas 3.1 · pytest 9.2.
