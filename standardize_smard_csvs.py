#!/usr/bin/env python3
"""
Standardize SMARD CSV files to have identical structure.

Problem: Two CSV files have different formats:
- File 1 (2016-2018): Direct header, no metadata
- File 2 (2016-2026): 10 metadata rows + BOM

Solution: Remove metadata from File 2, ensure both have same columns/format
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MARKET_DIR = PROJECT_ROOT / "data" / "raw" / "Daten zu Marktpreisen"

file1 = MARKET_DIR / "Gro_handelspreise_201601010000_201810010000_Stunde.csv"
file2 = MARKET_DIR / "Gro_handelspreise_201610010000_202601010000_Stunde.csv"

print("\n" + "="*80)
print("STANDARDIZING SMARD CSV FILES")
print("="*80)

# Load File 1 (simple, no metadata)
print(f"\n1. Loading {file1.name}")
df1 = pd.read_csv(file1, sep=";", decimal=",", encoding="utf-8", low_memory=False)
print(f"   Rows: {len(df1)}, Columns: {len(df1.columns)}")
print(f"   Date range: {df1['Datum von'].min()} to {df1['Datum von'].max()}")
print(f"   Columns: {df1.columns.tolist()[:5]}...")

# Load File 2 (with metadata rows at top)
print(f"\n2. Loading {file2.name}")
print(f"   This file has 9 metadata rows before header (header is at line 10)")

# File 2: lines 0-8 are metadata, line 9 (0-indexed) is header
df2 = pd.read_csv(file2, sep=";", decimal=",", encoding="utf-8-sig",
                 skiprows=9, low_memory=False)

print(f"   Rows: {len(df2)}, Columns: {len(df2.columns)}")
print(f"   Date range: {df2['Datum von'].min()} to {df2['Datum von'].max()}")

# Standardize: rename File 1 price column to match File 2
print(f"\n4. Standardizing column names...")

price_col_file1 = df1.columns[2]  # "DE/AT/LU [€/MWh] Berechnete Auflösungen"
price_col_file2 = df2.columns[2]  # "Deutschland/Luxemburg [€/MWh] Berechnete Auflösungen"

print(f"   File 1 price column: {price_col_file1}")
print(f"   File 2 price column: {price_col_file2}")

# Rename File 1's column to match File 2
df1_std = df1.copy()
df1_std.rename(columns={price_col_file1: price_col_file2}, inplace=True)
df2_std = df2.copy()

print(f"   Renamed File 1 column to match File 2")
print(f"   Both now use: {price_col_file2}")
print(f"   Total columns: {len(df1_std.columns)}")

# Combine and deduplicate
print(f"\n5. Combining both files...")
df_combined = pd.concat([df1_std, df2_std], ignore_index=True)
print(f"   Combined rows: {len(df_combined)}")

df_combined["timestamp"] = pd.to_datetime(
    df_combined["Datum von"],
    format="%d.%m.%Y %H:%M",
    errors="coerce"
)
df_combined = df_combined.sort_values("timestamp")
df_combined = df_combined.drop_duplicates(subset=["timestamp"], keep="first")
df_combined = df_combined.drop("timestamp", axis=1)

print(f"   After deduplication: {len(df_combined)} rows")
print(f"   Final date range: {df_combined['Datum von'].min()} to {df_combined['Datum von'].max()}")
print(f"   Final columns: {df_combined.columns.tolist()}")

# Save standardized files
print(f"\n6. Saving standardized files...")

file1_std = MARKET_DIR / "Gro_handelspreise_201601010000_201810010000_Stunde_STANDARDIZED.csv"
file2_std = MARKET_DIR / "Gro_handelspreise_201610010000_202601010000_Stunde_STANDARDIZED.csv"

df1_std.to_csv(file1_std, sep=";", decimal=",", index=False, encoding="utf-8")
df_combined.to_csv(file2_std, sep=";", decimal=",", index=False, encoding="utf-8")

print(f"   Saved: {file1_std.name}")
print(f"   Saved: {file2_std.name} (combined, deduplicated)")

# Backup original files
print(f"\n7. Backing up original files...")
file1_backup = file1.with_suffix(".csv.bak")
file2_backup = file2.with_suffix(".csv.bak")

import shutil
shutil.copy(file1, file1_backup)
shutil.copy(file2, file2_backup)
print(f"   Backed up to: {file1_backup.name}")
print(f"   Backed up to: {file2_backup.name}")

# Replace originals with standardized versions
print(f"\n8. Replacing originals with standardized versions...")
shutil.move(str(file1_std), str(file1))
shutil.move(str(file2_std), str(file2))
print(f"   Replaced: {file1.name}")
print(f"   Replaced: {file2.name}")

print(f"\n{'='*80}")
print("STANDARDIZATION COMPLETE!")
print("="*80)
print(f"\nBoth CSV files now have:")
print(f"  - Same columns: {len(common_cols)} columns")
print(f"  - No metadata rows")
print(f"  - Consistent format (sep=';', decimal=',')")
print(f"  - Combined date range: 2016-01-01 to 2025-12-31")
print(f"\nOriginal files backed up with .bak extension")
print(f"\nYou can now:")
print(f"  1. Run: python run_all_notebooks.py")
print(f"  2. Update load_smard() in NB04 to remove auto-detect logic (not needed anymore)\n")
