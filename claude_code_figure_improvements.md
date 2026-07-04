# Claude Code Prompt: Figure Improvements for WIBA Paper

## Aufgabe
Verbessere alle 31 Abbildungen in `results/figures/`. Alle Änderungen sollen im Code vorgenommen werden, der die Figures erzeugt (in den Notebooks NB04/NB05 oder entsprechenden Skripten). Speichere die finalen Figures mit `dpi=200` (beeswarm/force: `dpi=200`, große Figures: `dpi=150`).

---

## 0. Farb-Palette (CRISP-DM)

Extrahiere zuerst die dominanten Farben aus `03_Datenverarbeitung/CrispDM.jpg` (im LaTeX-Projekt) mit:

```python
from PIL import Image
import numpy as np

img = np.array(Image.open("path/to/CrispDM.jpg").convert("RGB"))
# K-Means mit k=6 auf non-white pixels
```

Nutze die extrahierten CRISP-DM-Farben als globale Palette. Falls die Datei nicht gefunden wird, nutze diesen Fallback:

```python
CRISP_COLORS = {
    "primary":   "#1F4E79",   # Dunkelblau
    "secondary": "#2E75B6",   # Mittelblau
    "light":     "#9DC3E6",   # Hellblau
    "orange":    "#ED7D31",   # Orange
    "green":     "#70AD47",   # Grün
    "yellow":    "#FFC000",   # Gelb/Gold
    "red":       "#C00000",   # Rot
    "gray":      "#7F7F7F",   # Grau
}
```

**Konsistente Modell-Farben** (global in allen Figures verwenden):
```python
MODEL_COLORS = {
    "QGB":         CRISP_COLORS["primary"],    # Dunkelblau
    "QRF":         CRISP_COLORS["secondary"],  # Mittelblau
    "XGBoost":     CRISP_COLORS["orange"],     # Orange
    "xgb":         CRISP_COLORS["orange"],
    "qgb":         CRISP_COLORS["primary"],
    "qrf":         CRISP_COLORS["secondary"],
    "Persistence": CRISP_COLORS["gray"],       # Grau
    "persistence": CRISP_COLORS["gray"],
    "ARIMA":       CRISP_COLORS["light"],      # Hellblau
    "arima":       CRISP_COLORS["light"],
    "Elastic_Net": CRISP_COLORS["green"],      # Grün
    "elastic_net": CRISP_COLORS["green"],
    "Oracle":      CRISP_COLORS["yellow"],     # Gold
}

SITE_COLORS = {
    "Schonungen":  CRISP_COLORS["primary"],
    "Schwanfeld":  CRISP_COLORS["orange"],
    "Trabelsdorf": CRISP_COLORS["green"],
    "Obbach":      CRISP_COLORS["red"],
}
```

**Heatmap Colormap**: Ersetze viridis/magma durch `"Blues"` oder eine CRISP-DM-kompatible sequentielle Palette. Für zweidimensionale Divergenz: `"RdBu_r"`.

**Globale Matplotlib-Einstellungen**:
```python
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})
```

---

## 1. Umlaut-Fix (ALLE Figures)

Ersetze in ALLEN Titeln, Achsenbeschriftungen, Legenden und Annotationen:
- `"ueber"` → `"über"`
- `"Fruehling"` → `"Frühling"`
- `"ue"` → `"ü"` (wo sinnvoll)
- Doppelstriche `"--"` in Titeln → `"–"` (em-dash oder Bindestrich)
- Englische Begriffe in gemischt-deutschen Titeln vereinheitlichen (entweder alles Deutsch oder alles Englisch – Zielsprache: **Deutsch**)

---

## 2. Figure-spezifische Verbesserungen

### 2.1 `shap_bar_importance.png`
- Umlaut in Titel und Achsenbeschriftungen fixen
- Feature-Name `nwp_ws100` → In Legende/Achse als `nwp\_ws100` rendern (kein LaTeX-Rendering-Problem in savefig)
- Balkenfarben: Verwende `MODEL_COLORS` für die drei Modelle (QGB, QRF, XGBoost), nicht einfarbige Balken
- Figsize: `(10, 6)`, DPI 150

### 2.2 `decision_aware_accuracy_vs_economics.png`
- Label-Overlap beheben: Verwende `adjustText` oder manuelle `xytext`-Offsets für alle Modell-Labels
- Achsenbeschriftungen auf Deutsch: x-Achse = "Pinball-Loss (↓ besser)", y-Achse = "NV-Loss [EUR/MWh] (↓ besser)"
- Punkt-Farben nach `MODEL_COLORS`
- Figsize: `(9, 7)`

### 2.3 `interpretable_vs_blackbox.png`
- **BUG FIX**: Persistence muss als **grau** (nicht grün) dargestellt werden — Persistence ist ein point benchmark, kein interpretierbares ML-Modell im engeren Sinne. Achte auf die Farbzuweisung im Code: `MODEL_COLORS["Persistence"]` verwenden
- Achsenbeschriftungen prüfen und ggf. auf Deutsch

### 2.4 `crosssite_economic_point_vs_prob.png`
- Konsistente Farben: Jeder Standort erhält eine Farbe aus `SITE_COLORS`, nicht wechselnd
- Linienmarkierungen unterschiedlich je Modell-Typ (point: `--`, prob: `-`)
- Legende: Vollständige Beschriftungen ohne Abkürzungen

### 2.5 `bid_failure_heatmap_schonungen.png`
- x-Achse (Monate): Zahlen 1–12 ersetzen durch `["Jan","Feb","Mär","Apr","Mai","Jun","Jul","Aug","Sep","Okt","Nov","Dez"]`
- Titel: Umlaut fixen
- Colormap: Ersetze Standard durch CRISP-DM-kompatible sequentielle Palette (`"Blues"` oder Custom)

### 2.6 `risk_profit_distribution.png`
- Histogramm: x-Achse Bereich erweitern um Tails zu zeigen (z.B. `xlim` auf 5. und 95. Perzentile der Daten, oder `clip=False` in seaborn)
- Bins erhöhen (z.B. 50 statt 20)
- Farben nach `MODEL_COLORS`
- Figsize: `(10, 5)`

### 2.7 `shap_beeswarm_qgb_schonungen.png`
- Figsize: `(12, 8)` — aktuell zu klein
- DPI: 200
- Feature-Namen in Achsenbeschriftung lesbar halten (ggf. `fontsize=9`)
- Titel auf Deutsch

### 2.8 `shap_dependence_nwp_ws100_schonungen.png`
- Figsize: `(10, 6)`
- DPI: 200
- Scatter-Farbe: aus CRISP-DM Palette (z.B. `CRISP_COLORS["primary"]`)
- Achsenbeschriftungen prüfen

### 2.9 `crosssite_forecast_accuracy.png`
- ARIMA und Persistence optisch unterscheidbar machen: unterschiedliche Linienstile (ARIMA: gestrichelt `"--"`, Persistence: gepunktet `":"`)
- Farben nach `MODEL_COLORS` für alle Modelle
- Legende mit vollständigen Modellnamen

### 2.10 `forecast_vs_actual_schonungen.png`
- Figsize: `(14, 5)` — aktuell zu klein
- DPI: 150
- Farben: actual = schwarz/dunkel, predicted = `MODEL_COLORS[model]`
- Legende außerhalb des Plots (bbox_to_anchor)

### 2.11 `forecast_bands_schonungen.png`
- Quantile-Band-Farbe: Statt lila → verwende CRISP-DM Palette:
  - Median (q50): `CRISP_COLORS["primary"]` (dunkelblau, solide Linie)
  - Band q25–q75: `CRISP_COLORS["secondary"]` mit alpha=0.3
  - Band q10–q90: `CRISP_COLORS["light"]` mit alpha=0.2
- Actual: schwarze solide Linie
- Figsize: `(14, 5)`, DPI 150
- Legende klar beschriften

### 2.12 `pinball_loss_schonungen.png`
- Farben: `MODEL_COLORS["xgb"]`, `MODEL_COLORS["qgb"]`, `MODEL_COLORS["qrf"]`
- Mehr Quantil-Punkte: Wenn möglich, auf 0.1, 0.2, ..., 0.9 verdichten (statt 5 Punkte)
- Titel: "Pinball-Loss – Schonungen" (Deutsch, mit Umlaut)
- Achsenbeschriftungen: x = "Quantil τ", y = "Pinball-Loss"
- Figsize: `(9, 5)`

### 2.13 `scatter_actual_pred_schonungen.png`
- Subplots (6 Panels): Jedes Panel erhält eine eigene Farbe aus CRISP-DM Palette statt alle blau
  - persistence: `CRISP_COLORS["gray"]`
  - arima: `CRISP_COLORS["light"]`
  - elastic_net: `CRISP_COLORS["green"]`
  - xgb: `CRISP_COLORS["orange"]`
  - qgb: `CRISP_COLORS["primary"]`
  - qrf: `CRISP_COLORS["secondary"]`
- Punkt-Alpha reduzieren (alpha=0.15) für weniger Überzeichnung
- Diagonale 1:1 Linie als schwarz gestrichelt
- Figsize: `(15, 10)`, DPI 150

### 2.14 `model_comparison_bars.png`
- Titel auf Deutsch: "Modellvergleich – Mittlere Fehlermetriken (alle Standorte)"
- Balkenfarben nach CRISP-DM: MAE = `CRISP_COLORS["primary"]`, RMSE = `CRISP_COLORS["orange"]`, Pinball = `CRISP_COLORS["green"]`
- Modell-Labels: "QGB", "QRF", "XGBoost" (korrekte Schreibweise)
- Y-Achse: "Verlust" oder jeweilige Einheit angeben
- Figsize: `(10, 6)`

### 2.15 `pinball_over_quantiles.png`
- Titel: "Pinball-Loss über Quantile" (kein "ueber")
- Farben: `MODEL_COLORS`
- Gleiche Verbesserungen wie 2.12

### 2.16 `bidding_portfolio_nvloss.png`
- Farb-Kodierung nach Strategie-Typ statt einheitliches Dunkelblau:
  - Oracle: `CRISP_COLORS["yellow"]` (Gold)
  - tau* dyn / tau* stat Strategien (QGB, QRF, XGBoost): `MODEL_COLORS[model]`
  - q50 Strategien: hellere Variante der Modellfarbe (alpha oder lightened)
  - DR-Strategien (DR-T1, DR-T2): `CRISP_COLORS["red"]`
  - Persistence / Elastic_Net: `MODEL_COLORS["Persistence"]` / `MODEL_COLORS["Elastic_Net"]`
- Figsize: `(12, 10)`, DPI 150
- Titel: "Portfolio-Bidding – Mittlerer Newsvendor-Verlust je Strategie"

### 2.17 `crosssite_bidfailure_wind.png`
- Farben nach `SITE_COLORS` (Schonungen=dunkelblau, Schwanfeld=orange, Trabelsdorf=grün, Obbach=rot)
- Linienstile: alle solid, Marker unterschiedlich je Standort
- Titel: "Gebotsverlust: NV-Loss vs. Windgeschwindigkeit × Standort (QGB τ*)"
- Figsize: `(10, 6)`

### 2.18 `crosssite_dependence_nwp_ws100.png`
- Farben nach `SITE_COLORS`
- Titel: Doppelstriche `--` durch `–` ersetzen: "Dependenz nwp\_ws100 × Standort – QGB"
- Figsize: `(10, 6)`

### 2.19 `crosssite_seasonal_wind.png`
- Winter: `CRISP_COLORS["primary"]` (Dunkelblau)
- Sommer: `CRISP_COLORS["orange"]` (Orange)
- Titel auf Deutsch korrekt
- Figsize: `(10, 6)`

### 2.20 `shap_seasonal_importance_schonungen.png`
- x-Achse: `"Fruehling"` → `"Frühling"` (alle anderen Jahreszeiten korrekt prüfen)
- Colormap: Ersetze viridis durch `"Blues"` (sequentiell) oder CRISP-DM Custom Colormap
- Titel: Sonderzeichen fixen
- Figsize: `(10, 7)`, DPI 150

### 2.21 `shap_heatmap_hour_month_schonungen.png`
- x-Achse (Monate): Zahlen 1–12 → `["Jan","Feb","Mär","Apr","Mai","Jun","Jul","Aug","Sep","Okt","Nov","Dez"]`
- Colormap: Ersetze magma durch `"Blues"` oder `cmap = LinearSegmentedColormap.from_list("crisp", [CRISP_COLORS["light"], CRISP_COLORS["primary"]])` 
- Titel: `"ueber"` → `"über"`
- Figsize: `(14, 8)`, DPI 150

### 2.22 `shap_waterfall_qgb_schonungen.png`
- SHAP Waterfall: Standard-Farben (rot/blau) beibehalten — das ist SHAP-Konvention
- Figsize erhöhen: `(12, 8)` (aktuell ausreichend, aber prüfen ob alle Labels lesbar)
- Titel auf Deutsch: "SHAP Waterfall – QGB / Schonungen (Beispiel-Stunde)"

### 2.23 `shap_force_qgb_schonungen.png`
- **KRITISCH**: Feature-Labels auf x-Achse sind extrem überlappend und unleserlich
- Lösung: `matplotlib.pyplot.tight_layout()` verwenden + Figsize deutlich erhöhen: `(20, 3)`
- Alternative: Feature-Labels um 45° rotieren mit `plt.xticks(rotation=45, ha='right')`
- Oder: Nur Top-5 Features beschriften, Rest als "..."
- DPI: 200

### 2.24 `quantile_band_qrf_schonungen.png`
- Quantile-Bänder farblich unterscheiden (nicht alle grün):
  - q10–q90: `CRISP_COLORS["light"]` alpha=0.2
  - q25–q75: `CRISP_COLORS["secondary"]` alpha=0.3
  - q50 Medianlinie: `CRISP_COLORS["primary"]` solid
- Actual: schwarz, `lw=1.5`
- Titel: "Probabilistische Prognoseband (QRF) – Schonungen"
- Figsize: `(14, 5)`, DPI 150

### 2.25 `shap_beeswarm_featured_qgb_schonungen.png`
- Figsize: `(12, 9)`, DPI 200
- Titel und Beschriftungen auf Deutsch prüfen

### 2.26 `shap_beeswarm_qrf_schonungen.png`
- Figsize: `(12, 8)`, DPI 200
- Titel: "Beeswarm – QRF / Schonungen"

### 2.27 `shap_beeswarm_xgb_schonungen.png`
- Figsize: `(12, 8)`, DPI 200
- Titel: "Beeswarm – XGBoost / Schonungen"

### 2.28 `shap_dependence_cubic_qgb_schonungen.png`
- Scatter-Farbe: `CRISP_COLORS["secondary"]` (statt lila)
- Referenzlinie (v^3): `CRISP_COLORS["red"]` gestrichelt — beibehalten
- Titel: Prüfen ob korrekt Deutsch
- Figsize: `(10, 6)`, DPI 150

### 2.29 `crosssite_shap_importance_qgb.png`
- Colormap: Ersetze viridis durch `"Blues"` oder CRISP-DM Custom Colormap
- Feature-Namen: Underscores in Achsenbeschriftungen ohne LaTeX rendern (kein Problem wenn `usetex=False`)
- Titel: Doppelstriche fixen
- Figsize: `(10, 8)`, DPI 150

### 2.30 `weather_nwp_sites.png`
- Farben: `SITE_COLORS["Schonungen"]` und `SITE_COLORS["Schwanfeld"]` (statt default blau/orange)
- Oder je nach dargestellten Sites die entsprechenden `SITE_COLORS` verwenden
- Achsenbeschriftungen auf Deutsch
- Figsize: `(12, 8)`, DPI 150

### 2.31 `bid_failure_conditions_schonungen.png`
- Linke Kurve (NV-Loss vs. Windgeschw.): `CRISP_COLORS["primary"]` (statt rot)
- Rechte Kurve (NV-Loss vs. Quantil-Spread): `CRISP_COLORS["orange"]` (statt lila)
- Oder beide mit `CRISP_COLORS["primary"]`, da es Schonungen-spezifisch ist
- Beide Subplots: Achsentitel auf Deutsch prüfen
- Figsize: `(14, 5)`, DPI 150

---

## 3. Zusammenfassung der globalen Änderungen

Erstelle eine Hilfsdatei `notebooks/plot_utils.py` (oder ergänze bestehende) mit:

```python
# plot_utils.py
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# CRISP-DM Palette (aus CrispDM.jpg extrahiert oder Fallback)
CRISP_COLORS = { ... }  # wie oben

MODEL_COLORS = { ... }  # wie oben

SITE_COLORS = { ... }   # wie oben

MONTH_NAMES_DE = ["Jan","Feb","Mär","Apr","Mai","Jun",
                   "Jul","Aug","Sep","Okt","Nov","Dez"]

def crisp_cmap(n=256):
    """Sequentielle CRISP-DM Colormap (hellblau → dunkelblau)."""
    return LinearSegmentedColormap.from_list(
        "crisp_blues", [CRISP_COLORS["light"], CRISP_COLORS["primary"]], N=n
    )

def apply_style():
    """Globale Matplotlib-Einstellungen setzen."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })
```

Importiere `plot_utils` in allen Notebooks/Skripten und ersetze alle hardcodierten Farben.

---

## 4. Prüfliste nach den Änderungen

- [ ] Alle 31 PNGs neu generieren
- [ ] Kein "ueber", "Fruehling", "--" mehr in Titeln
- [ ] Alle Modell-Farben konsistent über alle Figures
- [ ] Alle Figures haben DPI ≥ 150
- [ ] `bid_failure_heatmap` und `shap_heatmap` haben Monatsnamen statt -zahlen
- [ ] `interpretable_vs_blackbox.png`: Persistence ist grau
- [ ] `shap_force_qgb_schonungen.png`: Labels lesbar
- [ ] `bidding_portfolio_nvloss.png`: Farben nach Strategie-Typ
