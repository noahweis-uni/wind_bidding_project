#!/usr/bin/env python3
"""
Run all notebooks in sequence with error logging.
NB01 → NB02 → NB03 (all variants) → NB04 → NB05
"""

import subprocess
import sys
from pathlib import Path
import time
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# All notebooks to run (including all 03 variants)
NOTEBOOKS = [
    ("01_data_understanding.ipynb", "Data Understanding (Exploration)"),
    ("02_preprocessing.ipynb", "Preprocessing (NWP Download + Feature Engineering)"),
    ("03_forecasting_models.ipynb", "Forecasting Models (Day-Ahead)"),
    ("03_single_day_forecasting.ipynb", "Single-Day Forecasting"),
    ("03_single_day_forecasting_with_day_selection.ipynb", "Single-Day Forecasting (with Day Selection)"),
    ("04_bidding_evaluation.ipynb", "Bidding Evaluation (Economic Assessment)"),
    ("05_xai_analysis.ipynb", "XAI Analysis (SHAP Explanations)"),
]

def get_log_filename(notebook_file):
    """Generate log filename for a notebook"""
    name = notebook_file.replace(".ipynb", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{name}_{timestamp}.log"

def run_notebook(notebook_file, description):
    """Execute a single notebook using jupyter nbconvert with error logging"""
    notebook_path = NOTEBOOKS_DIR / notebook_file

    if not notebook_path.exists():
        print(f"[SKIP] {notebook_file} not found - skipping")
        return None  # None = skipped, not failed

    log_file = LOGS_DIR / get_log_filename(notebook_file)

    print(f"\n{'='*70}")
    print(f"[RUN] Running: {description}")
    print(f"   File: {notebook_file}")
    print(f"   Log: {log_file.name}")
    print(f"{'='*70}")

    start_time = time.time()

    try:
        # Execute notebook with nbconvert
        with open(log_file, "w") as logf:
            logf.write(f"Notebook: {notebook_file}\n")
            logf.write(f"Description: {description}\n")
            logf.write(f"Started: {datetime.now().isoformat()}\n")
            logf.write("="*70 + "\n\n")

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

            # Write full output to log
            if result.stdout:
                logf.write("STDOUT:\n")
                logf.write(result.stdout)
                logf.write("\n" + "="*70 + "\n\n")

            if result.stderr:
                logf.write("STDERR:\n")
                logf.write(result.stderr)
                logf.write("\n" + "="*70 + "\n\n")

            elapsed = time.time() - start_time
            logf.write(f"Status: {'SUCCESS' if result.returncode == 0 else 'FAILED'}\n")
            logf.write(f"Elapsed: {elapsed:.1f}s\n")
            logf.write(f"Completed: {datetime.now().isoformat()}\n")

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"[OK] SUCCESS ({elapsed:.1f}s)")
            return True
        else:
            print(f"[ERROR] FAILED ({elapsed:.1f}s)")
            print(f"[ERROR] See log: {log_file.name}")

            # Print last 20 lines of error if stderr exists
            if result.stderr:
                lines = result.stderr.split('\n')
                print("\n[ERROR] Last error lines:")
                for line in lines[-20:]:
                    if line.strip():
                        print(f"  {line}")

            return False

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"[TIMEOUT] TIMEOUT after {elapsed:.1f}s (>1 hour)")
        with open(log_file, "a") as logf:
            logf.write(f"\n[TIMEOUT] Process exceeded 1 hour timeout\n")
        return False
    except Exception as e:
        print(f"[FAIL] ERROR: {e}")
        with open(log_file, "a") as logf:
            logf.write(f"\n[FAIL] Exception: {str(e)}\n")
        return False

def main():
    print("\n" + "="*70)
    print("WIND BIDDING PROJECT - NOTEBOOK EXECUTION WITH LOGGING")
    print("="*70)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Log directory: {LOGS_DIR}")

    # Check if jupyter is installed
    try:
        subprocess.run([sys.executable, "-m", "jupyter", "--version"],
                      capture_output=True, check=True)
    except subprocess.CalledProcessError:
        print("\n[FAIL] ERROR: jupyter not installed!")
        print("Install with: pip install jupyter")
        return 1

    results = {}
    skipped = {}
    total_time = time.time()

    for notebook_file, description in NOTEBOOKS:
        result = run_notebook(notebook_file, description)

        if result is None:
            skipped[description] = True
            results[description] = None
        else:
            results[description] = result

            # Stop on failure UNLESS it's optional (03_single_day_forecasting variants)
            if not result and "Single-Day" not in description:
                print(f"\n[WARN] Critical notebook failed. Stopping here.")
                print(f"[WARN] Fix {notebook_file} and retry.")
                print(f"[INFO] Check error log: logs/{get_log_filename(notebook_file)}")
                break

    total_elapsed = time.time() - total_time

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    passed = 0
    failed = 0
    skipped_count = 0

    for description, success in results.items():
        if success is None:
            status = "[SKIP]"
            skipped_count += 1
        elif success:
            status = "[OK]"
            passed += 1
        else:
            status = "[FAIL]"
            failed += 1
        print(f"{status} {description}")

    print(f"\n{passed} passed, {failed} failed, {skipped_count} skipped")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"\nLogs saved to: {LOGS_DIR}")
    print("="*70 + "\n")

    # Return exit code (success only if no critical notebooks failed)
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
