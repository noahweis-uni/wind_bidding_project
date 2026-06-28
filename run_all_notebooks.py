#!/usr/bin/env python3
"""
Run all notebooks in sequence: NB02 → NB03 → NB04 → NB05
"""

import subprocess
import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

NOTEBOOKS = [
    ("01_data_understanding.ipynb", "Data Understanding (Exploration)"),
    ("02_preprocessing.ipynb", "Preprocessing (NWP Download + Feature Engineering)"),
    ("03_forecasting_models.ipynb", "Forecasting Models (Training + Predictions)"),
    ("04_bidding_evaluation.ipynb", "Bidding Evaluation (Economic Assessment)"),
    ("05_xai_analysis.ipynb", "XAI Analysis (SHAP Explanations)"),
]

def run_notebook(notebook_file, description):
    """Execute a single notebook using jupyter nbconvert"""
    notebook_path = NOTEBOOKS_DIR / notebook_file

    if not notebook_path.exists():
        print(f"[FAIL] {notebook_file} not found!")
        return False

    print(f"\n{'='*70}")
    print(f"[RUN] Running: {description}")
    print(f"   File: {notebook_file}")
    print(f"{'='*70}")

    start_time = time.time()

    try:
        # Execute notebook with nbconvert
        result = subprocess.run(
            [
                sys.executable, "-m", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                str(notebook_path),
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"[OK] SUCCESS ({elapsed:.1f}s)\n")
            return True
        else:
            print(f"[ERROR] FAILED ({elapsed:.1f}s)")
            if result.stdout:
                print(f"Stdout:\n{result.stdout}\n")
            if result.stderr:
                print(f"Stderr:\n{result.stderr}\n")
            return False

    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] TIMEOUT (>1 hour)\n")
        return False
    except Exception as e:
        print(f"[FAIL] ERROR: {e}\n")
        return False

def main():
    print("\n" + "="*70)
    print("WIND BIDDING PROJECT - NOTEBOOK EXECUTION")
    print("="*70)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python: {sys.version.split()[0]}")

    # Check if jupyter is installed
    try:
        subprocess.run([sys.executable, "-m", "jupyter", "--version"],
                      capture_output=True, check=True)
    except subprocess.CalledProcessError:
        print("\n[FAIL] ERROR: jupyter not installed!")
        print("Install with: pip install jupyter")
        return 1

    results = {}
    total_time = time.time()

    for notebook_file, description in NOTEBOOKS:
        success = run_notebook(notebook_file, description)
        results[description] = success

        if not success:
            print(f"\n[WARN] Stopping here. Fix {notebook_file} and retry.\n")
            break

    total_elapsed = time.time() - total_time

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for description, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        print(f"{status} {description}")

    print(f"\nTotal time: {total_elapsed/60:.1f} minutes")
    print("="*70 + "\n")

    # Return exit code
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())
