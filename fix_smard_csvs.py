#!/usr/bin/env python3
"""
Standardize all SMARD price CSVs to identical format.
Handles 3 different file formats with different column names.
"""

import pandas as pd
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent
MARKET_DIR = PROJECT_ROOT / "data" / "raw" / "Daten zu Marktpreisen"

print("\n" + "="*80)
print("STANDARDIZE ALL SMARD PRICE FILES")
print("="*80)

files_to_process = [
    ("Gro_handelspreise_2016_01-2018_10.csv", "DE/AT/LU [€/MWh]"),  # New file: correct prices
    ("Gro_handelspreise_201601010000_201810010000_Stunde.csv", "Deutschland/Luxemburg [€/MWh]"),  # File 1
    ("Gro_handelspreise_201610010000_202601010000_Stunde.csv", "Deutschland/Luxemburg [€/MWh] Berechnete Auflösungen"),  # File 2
]

all_data = []
standard_cols = ["Datum von", "Datum bis", "Preis"]

for filename, price_col in files_to_process:
    filepath = MARKET_DIR / filename
    if not filepath.exists():
        print(f"\n✗ {filename} NOT FOUND - skipping")
        continue

    print(f"\n1. Loading {filename}...")

    # Try different encodings and skiprows
    try:
        if filename == "Gro_handelspreise_2016_01-2018_10.csv":
            # New file: has BOM, simple structure
            df = pd.read_csv(filepath, sep=";", decimal=",", encoding="utf-8-sig", low_memory=False)
        else:
            # Old files: try without skiprows first
            try:
                df = pd.read_csv(filepath, sep=";", decimal=",", encoding="utf-8", low_memory=False)
            except:
                # If that fails, try with skiprows (for metadata files)
                df = pd.read_csv(filepath, sep=";", decimal=",", encoding="utf-8-sig", skiprows=9, low_memory=False)
    except Exception as e:
        print(f"   ERROR: {e}")
        continue

    print(f"   Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["Datum von"], format="%d.%m.%Y %H:%M", errors="coerce")

    # Extract price column
    if price_col in df.columns:
        price_col_actual = price_col
    else:
        # Fallback: look for column containing price-like data
        for col in df.columns:
            prices = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
            if prices.notna().sum() > df.shape[0] * 0.5:  # At least 50% non-null
                price_col_actual = col
                print(f"   Using fallback price column: {col}")
                break
        else:
            print(f"   ERROR: Could not find price column!")
            continue

    # Convert price to numeric
    df["Preis"] = pd.to_numeric(df[price_col_actual].astype(str).str.replace(",", "."), errors="coerce")

    # Extract only necessary columns
    df_clean = df[["Datum von", "Datum bis", "timestamp", "Preis"]].dropna(subset=["timestamp", "Preis"])

    print(f"   Valid rows (with timestamp + price): {len(df_clean)}")
    print(f"   Price range: {df_clean['Preis'].min():.2f} to {df_clean['Preis'].max():.2f}")

    all_data.append(df_clean)

# Combine all files
print(f"\n2. Combining all files...")
if not all_data:
    print("   ERROR: No data loaded!")
    exit(1)

df_combined = pd.concat(all_data, ignore_index=True)
print(f"   Total rows before dedup: {len(df_combined)}")

# Deduplicate by timestamp (keep first occurrence)
df_combined = df_combined.sort_values("timestamp")
df_combined = df_combined.drop_duplicates(subset=["timestamp"], keep="first")
print(f"   Total rows after dedup: {len(df_combined)}")
print(f"   Date range: {df_combined['timestamp'].min()} to {df_combined['timestamp'].max()}")

# Save standardized files
print(f"\n3. Saving standardized files...")
output_file = MARKET_DIR / "prices_standardized.csv"
df_combined[["Datum von", "Datum bis", "Preis"]].to_csv(output_file, sep=";", decimal=",", index=False, encoding="utf-8")
print(f"   Saved: {output_file.name}")

# Verify
df_verify = pd.read_csv(output_file, sep=";", decimal=",", nrows=5)
print(f"\n4. Verification:")
print(f"   Columns: {df_verify.columns.tolist()}")
print(f"   First 5 rows:")
print(df_verify)

print(f"\n{'='*80}")
print("SUCCESS! All files standardized")
print("="*80)
print(f"\nResult saved to: prices_standardized.csv")
print(f"Total records: {len(df_combined)}")
print(f"Date range: {df_combined['timestamp'].min()} to {df_combined['timestamp'].max()}\n")

