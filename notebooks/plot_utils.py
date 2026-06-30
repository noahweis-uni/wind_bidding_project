"""Zentrale Plot-Styles fuer das WIBA-Projekt (CRISP-DM Farbpalette).

Dominante CRISP-DM-Farbe aus Datenverarbeitung/CrispDM.jpg extrahiert: #155F81 (Petrol-Blau).
Helle Teal-Toene ebenfalls aus der Grafik; Akzentfarben (orange/gruen/gelb/rot) ergaenzt,
um 5 Modelle und 4 Standorte distinkt darstellen zu koennen.
"""
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CRISP_COLORS = {
    "primary":   "#155F81",   # dominantes CRISP-DM Petrol-Blau (extrahiert, 95%)
    "secondary": "#2E75B6",   # Mittelblau (distinkt fuer QGB vs QRF)
    "light":     "#B7DAE8",   # helles Teal (extrahiert)
    "orange":    "#ED7D31",
    "green":     "#70AD47",
    "yellow":    "#FFC000",
    "red":       "#C00000",
    "gray":      "#7F7F7F",
}

MODEL_COLORS = {
    "QGB":         CRISP_COLORS["primary"],
    "QRF":         CRISP_COLORS["secondary"],
    "XGBoost":     CRISP_COLORS["orange"],
    "xgb":         CRISP_COLORS["orange"],
    "qgb":         CRISP_COLORS["primary"],
    "qrf":         CRISP_COLORS["secondary"],
    "Persistence": CRISP_COLORS["gray"],
    "persistence": CRISP_COLORS["gray"],
    "ARIMA":       CRISP_COLORS["light"],
    "arima":       CRISP_COLORS["light"],
    "Elastic_Net": CRISP_COLORS["green"],
    "elastic_net": CRISP_COLORS["green"],
    "Oracle":      CRISP_COLORS["yellow"],
}

SITE_COLORS = {
    "Schonungen":  CRISP_COLORS["primary"],
    "Schwanfeld":  CRISP_COLORS["orange"],
    "Trabelsdorf": CRISP_COLORS["green"],
    "Obbach":      CRISP_COLORS["red"],
}

MONTH_NAMES_DE = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
                  "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]


def crisp_cmap(n=256):
    """Sequentielle CRISP-DM Colormap (hellblau -> dunkelblau)."""
    return LinearSegmentedColormap.from_list(
        "crisp_blues", [CRISP_COLORS["light"], CRISP_COLORS["primary"]], N=n
    )


def apply_style():
    """Globale Matplotlib-Einstellungen."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })
