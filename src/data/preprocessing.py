# preprocessing.py
# ---------------------------------------------------------------------------
# Reproduzierbare Feature-Pipeline fuer das Wind-Bidding-Projekt.
#
# Erzeugt aus vier Datenquellen einen modellfaehigen, STANDORTBEZOGENEN
# stuendlichen Feature-Datensatz:
#   1. Produktion  (SCADA-Messdaten der Windanlagen)         -> Zielvariable + Historie
#   2. ERA5        (Copernicus CDS, Reanalyse, UTC)          -> Wetter-Features (Perfect-Forecast-Baseline, s.u.)
#   3. Solar/pvlib (berechnet aus Standort + Zeit)           -> astronomische Features
#   4. CAMS        (Copernicus ADS, Meteosat/MSG, UTC)       -> Satellitenstrahlungs-Features
#
# CAMS Solar Radiation (cams-solar-radiation-timeseries):
#   Quelle: Copernicus Atmosphere Data Store (ADS), gleicher Account wie CDS.
#   Credentials: ~/.adsapirc (url + key, nie ins Repo).
#   Deckt 2004-heute ab (MSG-basiert, stundlich, ~4 km, geopunktuell).
#   Liefert: GHI/DNI/DHI (gemessen) + Clear-Sky-Varianten + Clear-Sky-Index.
#
# ERA5 als Perfect-Forecast-Baseline:
#   ERA5 ist eine Reanalyse (bestmoegliche Schaetzung des tatsaechlichen Wetters),
#   keine operative NWP-Prognose. Als Modell-Input entspricht das einem "Oracle"-
#   Modell mit perfektem Wetterwissen zur Gebotszeit -- was real nicht verfuegbar ist.
#   Methodische Einschaenkung: Die Ergebnisse zeigen die obere Leistungsgrenze
#   (Best-Case) eines wetterbasierten Gebotsmodells, nicht die operative Guete.
#   In der Praxis wuerden ERA5-Features durch archivierte NWP-Forecasts ersetzt
#   (z.B. ICON-EU oder GFS, ab ca. 2022 via Open-Meteo Historical Forecast API).
#   Fuer den Projektzeitraum ab 2017 existiert kein frei zugaengliches NWP-Archiv,
#   das eine konsistente Alternative bietet -- ERA5 wird daher bewusst als
#   Perfect-Forecast-Baseline eingesetzt und ist als solche im Paper deklariert.
#
# Zeitzonen-Konvention (siehe harmonize_timestamps):
#   - Produktion : lokale Zeit (Europe/Berlin, naiv)  -- ANNAHME, konfigurierbar
#   - ERA5       : UTC
#   - pvlib      : intern UTC, Ergebnis auf lokalen Master-Index
#   - CAMS       : UTC -> lokal (naiv), analog ERA5
#   Master-Zeitraster: stuendlich, naiv-lokal (konsistent mit NB02/NB03).
# ---------------------------------------------------------------------------

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_data import load_production

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEATHER_DIR = PROJECT_ROOT / "data" / "raw" / "weather"
SATELLITE_DIR = PROJECT_ROOT / "data" / "raw" / "satellite"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Annahme zur Produktions-Zeitzone (SCADA liefert i.d.R. Lokalzeit). Falls die
# Rohdaten doch UTC sind, hier auf "UTC" setzen -- der Rest der Pipeline zieht nach.
PRODUCTION_TZ = "Europe/Berlin"

# Reale Anlagenkoordinaten (lat, lon) -- Quelle: Plus Codes / Wikipedia (NB01).
SITES: dict[str, tuple[float, float]] = {
    "Schonungen":  (50.05028, 10.35056),
    "Schwanfeld":  (49.93361, 10.12667),
    "Trabelsdorf": (49.90889, 10.73500),
    "Obbach":      (50.08361, 10.07667),
}
# Grobe Nabenhoehe/Standorthoehe fuer pvlib-Clear-Sky (Unterfranken ~250-350 m).
SITE_ALTITUDE_M = 300.0


# ---------------------------------------------------------------------------
# 1) Produktion laden & robust bereinigen
# ---------------------------------------------------------------------------

def _to_numeric_robust(s: pd.Series) -> pd.Series:
    """Robuste numerische Konvertierung: Komma-Dezimal, Leerstrings, Mischformate."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    out = (
        s.astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)   # geschuetztes Leerzeichen
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)        # deutsches Dezimalkomma
    )
    out = out.replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": np.nan})
    return pd.to_numeric(out, errors="coerce")


def load_and_clean_production(sites: list[str] | None = None,
                              verbose: bool = True) -> pd.DataFrame:
    """
    Laedt SCADA-Rohdaten und liefert eine saubere LANG-Zeitreihe pro Standort.

    Returns
    -------
    DataFrame [timestamp, site, power, wind_speed]  (10-Min-Aufloesung, bereinigt)
    """
    sites = sites or list(SITES.keys())
    df = load_production(sites=sites, verbose=verbose)

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["power"] = _to_numeric_robust(df["power"])
    df["wind_speed"] = _to_numeric_robust(df["wind_speed"])

    n0 = len(df)
    # Zeilen ohne Zeitstempel verwerfen
    df = df.dropna(subset=["timestamp"])
    # Duplikate (site, timestamp) -> letzten Messwert behalten
    df = df.sort_values(["site", "timestamp"]).drop_duplicates(["site", "timestamp"], keep="last")
    # Unplausible Werte: negative Leistung -> 0 (Eigenverbrauch/Mess-Rauschen),
    # negative/extreme Windgeschwindigkeit -> NaN
    df["power"] = df["power"].clip(lower=0)
    df.loc[(df["wind_speed"] < 0) | (df["wind_speed"] > 60), "wind_speed"] = np.nan

    if verbose:
        print(f"Produktion bereinigt: {n0:,} -> {len(df):,} Zeilen "
              f"({df['site'].nunique()} Standorte)")
    return df[["timestamp", "site", "power", "wind_speed"]].reset_index(drop=True)


def aggregate_to_hourly_per_site(df: pd.DataFrame) -> pd.DataFrame:
    """
    10-Min -> Stundenmittel PRO STANDORT (keine Portfolio-Summe).
    power = mittlere Leistung [kW], wind_speed = mittlere Windgeschw. [m/s].
    """
    hourly = (
        df.set_index("timestamp")
        .groupby("site")
        .resample("h")[["power", "wind_speed"]]
        .mean()
        .reset_index()
        .rename(columns={"wind_speed": "wind_speed_obs"})
    )
    hourly["energy_mwh"] = hourly["power"].clip(lower=0) / 1000.0
    return hourly


# ---------------------------------------------------------------------------
# 2) ERA5 laden & Feature-Block bauen
# ---------------------------------------------------------------------------

def _pick(ds, *candidates):
    """Erste vorhandene Variable aus Kandidaten (robuste Namenserkennung)."""
    for c in candidates:
        if c in ds:
            return c
    # heuristische Teilstring-Suche
    for c in candidates:
        for v in ds.data_vars:
            if c.lower() in v.lower():
                return v
    raise KeyError(f"Keine der Variablen {candidates} im ERA5-Datensatz gefunden.")


def load_era5_data(weather_dir: Path | None = None):
    """Liest alle era5_*.nc Monatsdateien und konkateniert sie ueber die Zeit."""
    import xarray as xr
    base = Path(weather_dir) if weather_dir else WEATHER_DIR
    files = sorted(base.glob("era5_*.nc"))
    if not files:
        raise FileNotFoundError(f"Keine ERA5-Dateien in {base} (Download zuerst ausfuehren).")
    sample = xr.open_dataset(files[0])
    time_dim = "valid_time" if "valid_time" in sample.dims else "time"
    try:
        ds = xr.open_mfdataset(files, combine="by_coords")
    except Exception:
        ds = xr.concat([xr.open_dataset(f) for f in files], dim=time_dim)
    return ds, time_dim


def _meteo_wind_dir(u, v):
    """Meteorologische Windrichtung [deg], 'aus welcher Richtung der Wind weht'."""
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def build_era5_features(ds=None, time_dim: str = "valid_time",
                        sites: dict | None = None) -> pd.DataFrame:
    """
    Vollstaendiger ERA5-Feature-Block PRO STANDORT (bilineare Interpolation auf
    die exakte Anlagenkoordinate). Zeit wird von UTC auf lokale Zeit konvertiert.

    Returns (LANG-Form)
    -------
    DataFrame [timestamp(lokal naiv), site, ws100, wd100, ws10, wd10,
               temp2m, surface_pressure, total_cloud_cover,
               wind_shear_diff, wind_shear_ratio]
    """
    import xarray as xr
    sites = sites or SITES
    if ds is None:
        ds, time_dim = load_era5_data()

    u100 = _pick(ds, "u100", "100u", "100m_u_component_of_wind")
    v100 = _pick(ds, "v100", "100v", "100m_v_component_of_wind")
    u10 = _pick(ds, "u10", "10u", "10m_u_component_of_wind")
    v10 = _pick(ds, "v10", "10v", "10m_v_component_of_wind")
    t2m = _pick(ds, "t2m", "2t", "2m_temperature")
    sp = _pick(ds, "sp", "surface_pressure")
    tcc = _pick(ds, "tcc", "total_cloud_cover")

    frames = []
    for site, (lat, lon) in sites.items():
        cell = ds.interp(latitude=lat, longitude=lon)  # bilinear
        d = pd.DataFrame({
            "timestamp_utc": pd.to_datetime(cell[time_dim].values),
            "u100": np.asarray(cell[u100]), "v100": np.asarray(cell[v100]),
            "u10":  np.asarray(cell[u10]),  "v10":  np.asarray(cell[v10]),
            "temp2m": np.asarray(cell[t2m]),
            "surface_pressure": np.asarray(cell[sp]),
            "total_cloud_cover": np.asarray(cell[tcc]),
        })
        d["ws100"] = np.sqrt(d["u100"] ** 2 + d["v100"] ** 2)
        d["wd100"] = _meteo_wind_dir(d["u100"], d["v100"])
        d["ws10"] = np.sqrt(d["u10"] ** 2 + d["v10"] ** 2)
        d["wd10"] = _meteo_wind_dir(d["u10"], d["v10"])
        d["wind_shear_diff"] = d["ws100"] - d["ws10"]
        d["wind_shear_ratio"] = d["ws100"] / d["ws10"].replace(0, np.nan)
        d["site"] = site
        frames.append(d)

    era5 = pd.concat(frames, ignore_index=True)
    # UTC -> lokale (naive) Zeit, konsistent mit Produktion
    ts = pd.DatetimeIndex(era5["timestamp_utc"]).tz_localize("UTC").tz_convert(PRODUCTION_TZ)
    era5["timestamp"] = ts.tz_localize(None)
    era5 = era5.drop(columns=["timestamp_utc", "u100", "v100", "u10", "v10"])
    era5 = era5.drop_duplicates(["site", "timestamp"], keep="first")  # DST-Fallback

    cols = ["timestamp", "site", "ws100", "wd100", "ws10", "wd10", "temp2m",
            "surface_pressure", "total_cloud_cover", "wind_shear_diff", "wind_shear_ratio"]
    return era5[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3) Solar / astronomische Features (pvlib)  -- ersetzt den Satellitenblock
# ---------------------------------------------------------------------------

def build_solar_features(timestamps_local: pd.DatetimeIndex,
                         sites: dict | None = None) -> pd.DataFrame:
    """
    Berechnet pro Standort + Zeit die astronomischen / Clear-Sky-Groessen via pvlib.

    Quelle: Berechnung aus Koordinate + Zeit (kein Download, keine Satellitendaten).
    Deckt die Pinson-Features Zenitwinkel & Azimut ab; zusaetzlich Clear-Sky
    GHI/DNI/DHI als physikalisch fundierter Strahlungs-Proxy (ersetzt den nicht
    verfuegbaren satelliten-GHI / Clear-Sky-Index).

    Returns (LANG-Form)
    -------
    DataFrame [timestamp(lokal naiv), site, solar_zenith, solar_azimuth,
               clearsky_ghi, clearsky_dni, clearsky_dhi, is_day]
    """
    import pvlib
    sites = sites or SITES
    base = pd.DatetimeIndex(pd.unique(pd.DatetimeIndex(timestamps_local))).sort_values()

    # lokale naive Master-Zeit -> UTC (DST-sicher) fuer die Sonnenstandsberechnung
    idx_local = base.tz_localize(PRODUCTION_TZ, ambiguous="NaT", nonexistent="NaT")
    valid = ~idx_local.isna()
    idx_utc = idx_local[valid].tz_convert("UTC")
    base_valid = base[valid]

    frames = []
    for site, (lat, lon) in sites.items():
        loc = pvlib.location.Location(lat, lon, tz="UTC", altitude=SITE_ALTITUDE_M)
        solpos = pvlib.solarposition.get_solarposition(idx_utc, lat, lon, altitude=SITE_ALTITUDE_M)
        cs = loc.get_clearsky(idx_utc, model="ineichen")
        d = pd.DataFrame({
            "timestamp": base_valid,
            "site": site,
            "solar_zenith": solpos["apparent_zenith"].to_numpy(),
            "solar_azimuth": solpos["azimuth"].to_numpy(),
            "clearsky_ghi": cs["ghi"].to_numpy(),
            "clearsky_dni": cs["dni"].to_numpy(),
            "clearsky_dhi": cs["dhi"].to_numpy(),
        })
        d["is_day"] = (d["solar_zenith"] < 90).astype(int)
        frames.append(d)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 4) CAMS Satellitenstrahlungs-Features (Copernicus ADS)
# ---------------------------------------------------------------------------

def _read_adsapirc() -> tuple[str, str]:
    """
    Liest ADS-Credentials (url + key).
    Prioritaet: ~/.adsapirc -> ~/.cdsapirc (gleicher Copernicus-Key, andere URL).
    """
    def _parse_rc(path: Path) -> tuple[str | None, str | None]:
        url, key = None, None
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("url"):
                url = line.split(":", 1)[1].strip()
            elif line.startswith("key"):
                key = line.split(":", 1)[1].strip()
        return url, key

    adsrc = Path.home() / ".adsapirc"
    cdsrc = Path.home() / ".cdsapirc"

    if adsrc.exists():
        url, key = _parse_rc(adsrc)
    elif cdsrc.exists():
        # Gleicher Copernicus-Key, aber ADS-Endpunkt
        _, key = _parse_rc(cdsrc)
        url = "https://ads.atmosphere.copernicus.eu/api"
    else:
        raise FileNotFoundError(
            "Weder ~/.adsapirc noch ~/.cdsapirc gefunden. "
            "ADS-Key holen: https://ads.atmosphere.copernicus.eu -> Profil -> API Key"
        )

    if not url or not key:
        raise ValueError("~/.adsapirc / ~/.cdsapirc muss 'url' und 'key' enthalten.")
    return url, key


def download_cams_data(sites: dict | None = None,
                       start_date: str = "2017-01-01",
                       end_date: str = "2025-08-31",
                       satellite_dir: Path | None = None,
                       overwrite: bool = False) -> dict[str, Path]:
    """
    Laedt CAMS Solar Radiation Timeseries pro Standort vom Copernicus ADS.

    Credentials aus ~/.adsapirc (gleicher Copernicus-Account wie CDS,
    anderer Endpunkt: ads.atmosphere.copernicus.eu).

    Output: data/raw/satellite/cams_<site_lower>.csv  (UTC, Semikolon-getrennt)

    Returns
    -------
    Dict  {site_name: Path}
    """
    import cdsapi

    sites = sites or SITES
    base = Path(satellite_dir) if satellite_dir else SATELLITE_DIR
    base.mkdir(parents=True, exist_ok=True)

    ads_url, ads_key = _read_adsapirc()
    c = cdsapi.Client(url=ads_url, key=ads_key, quiet=False)

    paths: dict[str, Path] = {}
    for site, (lat, lon) in sites.items():
        target = base / f"cams_{site.lower()}.csv"
        if target.exists() and not overwrite:
            print(f"  {site}: bereits vorhanden ({target.name}), skip.")
            paths[site] = target
            continue

        print(f"  Downloading CAMS fuer {site} ({lat:.4f}N, {lon:.4f}E) ...")
        c.retrieve(
            "cams-solar-radiation-timeseries",
            {
                "sky_type": "observed_cloud",   # liefert observed + clear-sky in einer CSV
                "location": {"latitude": lat, "longitude": lon},
                "altitude": "-999.",
                "date": f"{start_date}/{end_date}",  # String "YYYY-MM-DD/YYYY-MM-DD"
                "time_step": "1hour",
                "time_reference": "universal_time",
                "data_format": "csv",
            },
            str(target),
        )
        paths[site] = target
        print(f"    -> {target.name} ({target.stat().st_size // 1024} kB)")

    return paths


def _parse_cams_csv(csv_path: Path, site: str) -> pd.DataFrame:
    """
    Parst eine einzelne CAMS-CSV-Datei.

    CAMS-CSV-Format (Semikolon, alle Kommentarzeilen mit '#', letzte davon = Header):
      # Observation period;TOA;Clear sky GHI;...;Reliability
      2023-06-01T00:00:00.0/2023-06-01T01:00:00.0;0.0;...
    """
    with open(csv_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    # Die letzte #-Zeile die "Observation" enthaelt ist die Spaltenheader-Zeile
    header_line_idx = None
    for i, l in enumerate(lines):
        if l.startswith("#") and "Observation" in l:
            header_line_idx = i

    if header_line_idx is None:
        raise ValueError(f"Keine Header-Zeile ('# Observation period;...') in {csv_path.name}")

    # Header-Zeile: '#'-Prefix + Leerzeichen entfernen
    col_names = [c.strip() for c in lines[header_line_idx].lstrip("# ").split(";")]

    raw = pd.read_csv(
        csv_path, sep=";",
        skiprows=header_line_idx + 1,   # Daten beginnen nach dem Header
        header=None,
        names=col_names,
        encoding="utf-8",
    )
    raw.columns = (
        raw.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"\(.*\)", "", regex=True)
        .str.strip("_")
    )

    # Zeitstempel: Anfang des Beobachtungsintervalls (UTC)
    # Format: "2023-06-01T00:00:00.0/2023-06-01T01:00:00.0"
    period_col = next(
        (c for c in raw.columns if "observation" in c or "period" in c), None
    )
    if period_col is None:
        raise ValueError(f"Keine 'Observation period'-Spalte in {csv_path.name}")

    raw["timestamp_utc"] = pd.to_datetime(
        raw[period_col].str.split("/").str[0].str.replace(r"\.\d+$", "", regex=True),
        utc=True,
    )

    # Spalten robust zuordnen (Gross/Klein, BNI=DNI, BHI=direkt-horizontal)
    def _find(raw_df, *keys):
        for k in keys:
            if k in raw_df.columns:
                return k
            matches = [c for c in raw_df.columns if k in c]
            if matches:
                return matches[0]
        return None

    mapping = {
        "cams_ghi":          _find(raw, "ghi"),
        "cams_dni":          _find(raw, "bni"),          # BNI = DNI
        "cams_dhi":          _find(raw, "dhi"),
        "cams_clearsky_ghi": _find(raw, "clear_sky_ghi"),
        "cams_clearsky_dni": _find(raw, "clear_sky_bni"),
        "cams_clearsky_dhi": _find(raw, "clear_sky_dhi"),
    }

    d = pd.DataFrame({"timestamp_utc": raw["timestamp_utc"]})
    for new_col, src_col in mapping.items():
        if src_col:
            d[new_col] = pd.to_numeric(raw[src_col], errors="coerce").clip(lower=0)

    # Clear-Sky-Index (Kt): gemessen / clearsky, kein Nachtwert
    if "cams_ghi" in d.columns and "cams_clearsky_ghi" in d.columns:
        d["cams_clearsky_index"] = (
            d["cams_ghi"] / d["cams_clearsky_ghi"].replace(0.0, np.nan)
        ).clip(0, 1.5)

    # UTC -> lokale naive Zeit (konsistent mit ERA5/Produktion)
    ts_local = (
        pd.DatetimeIndex(d["timestamp_utc"])
        .tz_convert(PRODUCTION_TZ)
        .tz_localize(None)
    )
    d["timestamp"] = ts_local
    d["site"] = site
    d = d.drop(columns=["timestamp_utc"])
    d = d.drop_duplicates(["site", "timestamp"], keep="first")  # DST-Fallback

    return d


def build_cams_features(sites: dict | None = None,
                        satellite_dir: Path | None = None,
                        verbose: bool = True) -> pd.DataFrame | None:
    """
    Liest die heruntergeladenen CAMS-CSV-Dateien und gibt einen LANG-DataFrame
    mit echten Satellitenstrahlungs-Features zurueck.

    Gibt None zurueck, wenn keine CAMS-Dateien vorhanden (graceful degradation).

    Returns (LANG-Form)
    -------
    DataFrame [timestamp(lokal naiv), site,
               cams_ghi, cams_dni, cams_dhi,
               cams_clearsky_ghi, cams_clearsky_dni, cams_clearsky_dhi,
               cams_clearsky_index]
    """
    sites = sites or SITES
    base = Path(satellite_dir) if satellite_dir else SATELLITE_DIR

    frames = []
    for site in sites:
        csv_path = base / f"cams_{site.lower()}.csv"
        if not csv_path.exists():
            if verbose:
                print(f"  {site}: kein CAMS CSV ({csv_path.name}), uebersprungen.")
            continue
        d = _parse_cams_csv(csv_path, site)
        if verbose:
            print(f"  {site}: {len(d):,} Stunden CAMS geladen "
                  f"({d['timestamp'].min().date()} .. {d['timestamp'].max().date()})")
        frames.append(d)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 5) Zeitachsen harmonisieren & Bloecke mergen
# ---------------------------------------------------------------------------

def harmonize_timestamps(prod_hourly: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Dokumentiert/erzeugt den gemeinsamen stuendlichen Master-Zeitindex (lokal, naiv).
    Produktion ist die Master-Quelle (lokale Zeit); ERA5/Solar/CAMS werden darauf gemappt.
    """
    return pd.DatetimeIndex(sorted(prod_hourly["timestamp"].unique()))


def merge_feature_blocks(prod_hourly: pd.DataFrame,
                         era5: pd.DataFrame | None,
                         solar: pd.DataFrame | None,
                         cams: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Left-Merge der Bloecke pro [site, timestamp] auf Produktionsbasis.
    Produktion = Master; ERA5, Solar, CAMS werden angehaengt (graceful, falls None).
    """
    out = prod_hourly.copy()
    for block in (era5, solar, cams):
        if block is not None:
            out = out.merge(block, on=["site", "timestamp"], how="left")
    return out.sort_values(["site", "timestamp"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5) Zyklische Features (Wind + Kalender)
# ---------------------------------------------------------------------------

def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Kalender-Features + zyklische Kodierung von Stunde, Wochentag, Windrichtung."""
    df = df.copy()
    ts = df["timestamp"].dt
    df["hour"] = ts.hour
    df["dayofweek"] = ts.dayofweek
    df["month"] = ts.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    for col in ("wd100", "wd10"):
        if col in df.columns:
            rad = np.radians(df[col])
            df[f"sin_{col}"] = np.sin(rad)
            df[f"cos_{col}"] = np.cos(rad)
    return df


# ---------------------------------------------------------------------------
# 6) Lags & Rolling Features
# ---------------------------------------------------------------------------

DEFAULT_LAGS = (1, 2, 3, 6, 12, 24)
DEFAULT_ROLLS = (3, 6, 24)


def build_lag_features(df: pd.DataFrame,
                       cols: list[str],
                       lags=DEFAULT_LAGS,
                       rolls=DEFAULT_ROLLS,
                       add_rolling: bool = True) -> pd.DataFrame:
    """
    Erzeugt PRO STANDORT Lags und (optional) Rolling-Mean/Std fuer `cols`.
    Annahme: lueckenloses Stundenraster pro Standort (sonst vorher reindexen).
    """
    df = df.sort_values(["site", "timestamp"]).copy()
    g = df.groupby("site", group_keys=False)
    for c in cols:
        if c not in df.columns:
            continue
        for L in lags:
            df[f"{c}_lag{L}h"] = g[c].shift(L)
        if add_rolling:
            for w in rolls:
                df[f"{c}_rollmean{w}h"] = g[c].transform(
                    lambda s: s.shift(1).rolling(w, min_periods=max(2, w // 2)).mean())
                df[f"{c}_rollstd{w}h"] = g[c].transform(
                    lambda s: s.shift(1).rolling(w, min_periods=max(2, w // 2)).std())
    return df


# ---------------------------------------------------------------------------
# 7) Datenqualitaet & Abdeckung
# ---------------------------------------------------------------------------

def run_data_quality_checks(df: pd.DataFrame,
                            blocks: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Pro Standort: Zeitraum, Zeilen, Fehlanteil je Block."""
    rows = []
    for site, g in df.groupby("site"):
        row = {
            "site": site,
            "von": g["timestamp"].min(),
            "bis": g["timestamp"].max(),
            "zeilen": len(g),
        }
        if blocks:
            for name, cols in blocks.items():
                present = [c for c in cols if c in g.columns]
                if present:
                    row[f"na_{name}_%"] = round(100 * g[present].isna().mean().mean(), 1)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8) Feature-Gruppen (A-H) fuer Modellvergleich & XAI
# ---------------------------------------------------------------------------

PRODUCTION_FEATURES = ["energy_mwh", "wind_speed_obs"]
ERA5_WIND_FEATURES = ["ws100", "wd100", "sin_wd100", "cos_wd100"]
ERA5_FULL_FEATURES = ERA5_WIND_FEATURES + [
    "ws10", "wd10", "sin_wd10", "cos_wd10", "temp2m", "surface_pressure",
    "total_cloud_cover", "wind_shear_diff", "wind_shear_ratio"]
# Solar/astronomisch: pvlib (zenith, azimuth, is_day) + pvlib Clear-Sky (Fallback)
SOLAR_FEATURES = ["solar_zenith", "solar_azimuth", "clearsky_ghi",
                  "clearsky_dni", "clearsky_dhi", "is_day"]
SOLAR_ASTRO_ONLY = ["solar_zenith", "solar_azimuth", "is_day"]
# CAMS: echte Meteosat-Strahlungsmessung (erfordert ADS-Download)
CAMS_FEATURES = [
    "cams_ghi", "cams_dni", "cams_dhi",
    "cams_clearsky_ghi", "cams_clearsky_dni", "cams_clearsky_dhi",
    "cams_clearsky_index",
]
CALENDAR_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month"]


def feature_groups() -> dict[str, list[str]]:
    """
    Ablationsgruppen A-H (ohne CAMS) + I-J (mit CAMS-Satellitendaten).
    Basisspalten, ohne Lags.
    """
    prod = PRODUCTION_FEATURES
    return {
        # -- Ohne CAMS (A-H, reproduzierbar ohne ADS-Download) --
        "A_production":            prod + CALENDAR_FEATURES,
        "B_prod_era5wind":         prod + ERA5_WIND_FEATURES + CALENDAR_FEATURES,
        "C_prod_era5full":         prod + ERA5_FULL_FEATURES + CALENDAR_FEATURES,
        "D_era5only":              ERA5_FULL_FEATURES + CALENDAR_FEATURES,
        "E_prod_solar":            prod + SOLAR_FEATURES + CALENDAR_FEATURES,
        "F_prod_era5_solar":       prod + ERA5_FULL_FEATURES + SOLAR_FEATURES + CALENDAR_FEATURES,
        "G_solaronly":             SOLAR_FEATURES + CALENDAR_FEATURES,
        "H_era5_solar_noprod":     ERA5_FULL_FEATURES + SOLAR_FEATURES + CALENDAR_FEATURES,
        # -- Mit CAMS-Satellitendaten (I-J, erfordern ADS-Download) --
        "I_prod_era5_cams":        prod + ERA5_FULL_FEATURES + SOLAR_ASTRO_ONLY + CAMS_FEATURES + CALENDAR_FEATURES,
        "J_all":                   prod + ERA5_FULL_FEATURES + SOLAR_ASTRO_ONLY + CAMS_FEATURES + CALENDAR_FEATURES,
    }


# ---------------------------------------------------------------------------
# 9) Feature-Dokumentation & Export
# ---------------------------------------------------------------------------

def build_feature_documentation() -> pd.DataFrame:
    """Tabelle: Feature, Beschreibung, Quelle, Einheit, empfohlene Gruppe."""
    rows = [
        ("power",             "Mittlere Anlagenleistung (Stunde)", "production", "kW",   "target"),
        ("energy_mwh",        "Stundenenergie (power/1000)",       "production", "MWh",  "A-F"),
        ("wind_speed_obs",    "Gemessene Windgeschw. (SCADA)",     "production", "m/s",  "A-F"),
        ("ws100",             "ERA5 Windgeschw. 100 m",            "ERA5",       "m/s",  "B,C,D,F,H"),
        ("wd100",             "ERA5 Windrichtung 100 m",           "derived",    "deg",  "B,C,D,F,H"),
        ("sin_wd100/cos_wd100", "Zyklische Windrichtung 100 m",    "derived",    "-",    "B,C,D,F,H"),
        ("ws10/wd10",         "ERA5 Wind 10 m + Richtung",         "ERA5/derived","m/s/deg","C,D,F,H"),
        ("temp2m",            "ERA5 2m-Temperatur",                "ERA5",       "K",    "C,D,F,H"),
        ("surface_pressure",  "ERA5 Oberflaechendruck",            "ERA5",       "Pa",   "C,D,F,H"),
        ("total_cloud_cover", "ERA5 Gesamtbedeckungsgrad (TCC)",   "ERA5",       "0-1",  "C,D,F,H"),
        ("wind_shear_diff",   "ws100 - ws10",                      "derived",    "m/s",  "C,D,F,H"),
        ("wind_shear_ratio",  "ws100 / ws10",                      "derived",    "-",    "C,D,F,H"),
        ("solar_zenith",      "Sonnen-Zenitwinkel (pvlib)",        "pvlib",      "deg",  "E,F,G,H,I,J"),
        ("solar_azimuth",     "Sonnen-Azimut (pvlib)",             "pvlib",      "deg",  "E,F,G,H,I,J"),
        ("clearsky_ghi",      "Clear-Sky GHI (pvlib, Ineichen)",   "pvlib",      "W/m2", "E,F,G,H"),
        ("clearsky_dni/dhi",  "Clear-Sky DNI/DHI (pvlib)",         "pvlib",      "W/m2", "E,F,G,H"),
        ("is_day",            "Tag-Indikator (zenith<90)",         "pvlib",      "0/1",  "E,F,G,H,I,J"),
        ("cams_ghi",          "CAMS GHI (Meteosat, gemessen)",     "CAMS/ADS",   "W/m2", "I,J"),
        ("cams_dni",          "CAMS DNI/BNI (Meteosat, gemessen)", "CAMS/ADS",   "W/m2", "I,J"),
        ("cams_dhi",          "CAMS DHI (Meteosat, gemessen)",     "CAMS/ADS",   "W/m2", "I,J"),
        ("cams_clearsky_ghi", "CAMS Clear-Sky GHI (McClear)",      "CAMS/ADS",   "W/m2", "I,J"),
        ("cams_clearsky_dni", "CAMS Clear-Sky DNI (McClear)",      "CAMS/ADS",   "W/m2", "I,J"),
        ("cams_clearsky_dhi", "CAMS Clear-Sky DHI (McClear)",      "CAMS/ADS",   "W/m2", "I,J"),
        ("cams_clearsky_index","GHI/Clear-Sky-GHI (Kt, CAMS)",     "derived",    "0-1.5","I,J"),
        ("hour_sin/cos",      "Zyklische Stunde",                  "calendar",   "-",    "alle"),
        ("dow_sin/cos",       "Zyklischer Wochentag",              "calendar",   "-",    "alle"),
        ("month",             "Monat",                             "calendar",   "1-12", "alle"),
        ("*_lag{1,2,3,6,12,24}h", "Zeitverzoegerte Inputs",        "derived",    "wie Basis", "Modell"),
        ("*_rollmean/std{3,6,24}h", "Rolling Mittel/Std",          "derived",    "wie Basis", "Modell"),
    ]
    return pd.DataFrame(rows, columns=["feature", "beschreibung", "quelle", "einheit", "gruppe"])


def export_feature_sets(df_base: pd.DataFrame,
                        df_model: pd.DataFrame | None = None,
                        out_dir: Path | None = None,
                        per_site: bool = True) -> dict:
    """
    Schreibt die finalen Datensaetze + Feature-Doku nach data/processed/.
    Gibt ein Dict der geschriebenen Pfade zurueck.
    """
    out = Path(out_dir) if out_dir else PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)
    written = {}

    p = out / "features_base_all_sites.csv"
    df_base.to_csv(p, index=False)
    written["base"] = p

    if df_model is not None:
        p = out / "features_model_all_sites.csv"
        df_model.to_csv(p, index=False)
        written["model"] = p

    if per_site:
        target = df_model if df_model is not None else df_base
        for site, g in target.groupby("site"):
            p = out / f"features_site_{site}.csv"
            g.to_csv(p, index=False)
            written[f"site_{site}"] = p

    p = out / "feature_documentation.csv"
    build_feature_documentation().to_csv(p, index=False)
    written["doc"] = p

    return written
