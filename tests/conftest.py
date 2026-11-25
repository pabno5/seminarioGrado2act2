import sys
from pathlib import Path

# Permite importar los módulos de src/ desde las pruebas (p. ej. input_validation)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
