# =====================================================================
#  Pipeline MLOps Iris — Semana 2
#  Atajos de make sobre run_pipeline.py (toda la lógica vive allí).
#
#  Un solo comando:    make            (= make all)
#  En Windows sin make: python run_pipeline.py
#
#  Opciones:  make all FORCE=1         re-ejecuta todo aunque no haya cambios
#             make all PYTHON=python   usa otro intérprete
# =====================================================================

ifeq ($(OS),Windows_NT)
PYTHON ?= python
else
PYTHON ?= python3
endif

RUN        := $(PYTHON) run_pipeline.py
FORCE_FLAG := $(if $(FORCE),--force,)

.DEFAULT_GOAL := all
.PHONY: all prepare validate_input validate train test install metrics dag ui clean help

all: ## Pipeline completo: prepare -> validate_input -> validate -> train -> test
	$(RUN) all $(FORCE_FLAG)

prepare validate_input validate train test: ## Ejecuta hasta esa etapa (y las previas si hace falta)
	$(RUN) $@ $(FORCE_FLAG)

install: ## Instala las dependencias de requirements.txt
	$(RUN) install

metrics dag ui clean: ## Utilidades: métricas, grafo, MLflow UI, limpieza
	$(RUN) $@

help: ## Muestra la ayuda
	$(RUN) --help
