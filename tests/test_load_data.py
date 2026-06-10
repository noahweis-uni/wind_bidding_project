from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data import load_data


def test_get_production_path_returns_existing_directory():
    path = load_data.get_production_path()

    assert path is not None
    assert path.exists()
    assert path.name == "Daten zur Windkennlinie_2025-09-17"
