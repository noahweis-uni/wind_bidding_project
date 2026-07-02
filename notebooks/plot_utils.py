"""Zentrale Plot-Styles fuer das WIBA-Projekt (CRISP-DM Farbpalette).

Dominante CRISP-DM-Farbe aus Datenverarbeitung/CrispDM.jpg extrahiert: #155F81 (Petrol-Blau).
Helle Teal-Toene ebenfalls aus der Grafik; Akzentfarben (orange/gruen/gelb/rot) ergaenzt,
um 5 Modelle und 4 Standorte distinkt darstellen zu koennen.
"""
import numpy as np
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
    "purple":    "#7030A0",   # Quantile Regression (interpretierbar, probabilistisch)
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
    "QR":                  CRISP_COLORS["purple"],   # Quantile Regression (interpretierbar + probabilistisch)
    "qr":                  CRISP_COLORS["purple"],
    "Quantile_Regression": CRISP_COLORS["purple"],
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

def get_model_color(model_name):
    """
    Gibt die passende Modellfarbe aus der zentralen CRISP-DM-Farbpalette zurück.

    Funktioniert auch robust bei Modellnamen wie:
    - "xgb", "xgb_q50", "XGBoost", "Quantile XGBoost"
    - "qgb", "QGB"
    - "qrf", "QRF", "qrf_q50"
    - "qr", "Quantile_Regression"
    - "elastic_net", "Elastic_Net"
    - "persistence", "Persistence 24h"
    """

    name = str(model_name).lower().replace("-", "_").replace(" ", "_")

    if "oracle" in name:
        return CRISP_COLORS["yellow"]

    if "persistence" in name:
        return CRISP_COLORS["gray"]

    if "arima" in name:
        return CRISP_COLORS["light"]

    if "elastic" in name or name in ["en", "elastic_net"]:
        return CRISP_COLORS["green"]

    if "quantile_regression" in name or name == "qr" or name.startswith("qr_"):
        return CRISP_COLORS["purple"]

    if "qgb" in name or "quantile_gradient_boosting" in name:
        return CRISP_COLORS["primary"]

    if "qrf" in name or "quantile_random_forest" in name:
        return CRISP_COLORS["secondary"]

    if "xgb" in name or "xgboost" in name:
        return CRISP_COLORS["orange"]

    if "gradient_boosting" in name:
        return CRISP_COLORS["primary"]

    if "random_forest" in name:
        return CRISP_COLORS["secondary"]

    return CRISP_COLORS["gray"]


def get_site_color(site_name):
    """
    Gibt die passende Standortfarbe zurück.
    Falls der Standort unbekannt ist, wird Grau verwendet.
    """
    return SITE_COLORS.get(str(site_name), CRISP_COLORS["gray"])


# ---------------------------------------------------------------------------
# Automatische Scatter-Label-Platzierung (Leader-Lines)
# ---------------------------------------------------------------------------

_COMPASS_DIRS = {
    "E": (1, 0), "NE": (1, 1), "N": (0, 1), "NW": (-1, 1),
    "W": (-1, 0), "SW": (-1, -1), "S": (0, -1), "SE": (1, -1),
}
_COMPASS_ANGLES = {"E": 0, "NE": 45, "N": 90, "NW": 135,
                   "W": 180, "SW": 225, "S": 270, "SE": 315}


def _cluster_points(xn, yn, threshold):
    """Union-Find: Punkte werden transitiv geclustert, wenn ihr (normalisierter)
    Abstand < threshold ist."""
    n = len(xn)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(xn[i] - xn[j], yn[i] - yn[j]) < threshold:
                union(i, j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def compute_label_offsets(x, y, isolated_offset=(0.01, 0.05),
                          cluster_gap=0.16, density_threshold=0.05):
    """
    Berechnet automatisch Label-Positionen fuer einen Scatter-Plot, ohne
    hartkodierte Koordinaten. Distanzen werden auf [0,1] normalisiert
    (min-max je Achse), damit x/y trotz unterschiedlicher Skalen vergleichbar sind.

    - Isolierte Punkte (naechster Nachbar >= density_threshold, normalisiert):
      fester kleiner Versatz `isolated_offset` (Anteil der jeweiligen Achsspanne).
    - Dichte Cluster (naechster Nachbar < density_threshold): Labels werden
      radial um den Cluster-Schwerpunkt in 8 Himmelsrichtungen aufgefaechert.
      Jeder Punkt bekommt die freie Richtung, die seinem tatsaechlichen Winkel
      zum Schwerpunkt am naechsten liegt (Konfliktaufloesung durch Greedy-Zuweisung
      in Winkel-Reihenfolge).

    Parameters
    ----------
    x, y : array-like
        Datenkoordinaten der Punkte.
    isolated_offset : tuple(float, float)
        Versatz fuer isolierte Punkte, als Anteil der x-/y-Achsspanne.
    cluster_gap : float
        Abstand vom Cluster-Schwerpunkt zu den Labels, als Anteil der
        (normalisierten) Achsspanne. Bei langen Labels ggf. erhoehen.
    density_threshold : float
        Normalisierter Abstands-Schwellwert, ab dem Punkte als "dicht" gelten.

    Returns
    -------
    list[tuple[float, float]]
        Label-Position (x_label, y_label) in Datenkoordinaten, eine pro Punkt,
        in der Reihenfolge von `x`/`y`.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xr = x.max() - x.min() or 1.0
    yr = y.max() - y.min() or 1.0
    xn, yn = (x - x.min()) / xr, (y - y.min()) / yr

    clusters = _cluster_points(xn, yn, density_threshold)
    label_xy = [None] * len(x)

    for members in clusters:
        if len(members) == 1:
            i = members[0]
            label_xy[i] = (x[i] + isolated_offset[0] * xr,
                          y[i] + isolated_offset[1] * yr)
            continue

        cx, cy = xn[members].mean(), yn[members].mean()
        angles = {i: np.degrees(np.arctan2(yn[i] - cy, xn[i] - cx)) % 360
                  for i in members}
        assigned = {}
        for i in sorted(members, key=lambda i: angles[i]):
            free = [d for d in _COMPASS_ANGLES if d not in assigned.values()] or list(_COMPASS_ANGLES)
            best = min(free, key=lambda d: min(abs(angles[i] - _COMPASS_ANGLES[d]),
                                               360 - abs(angles[i] - _COMPASS_ANGLES[d])))
            assigned[i] = best

        for i in members:
            dx, dy = _COMPASS_DIRS[assigned[i]]
            label_xy[i] = (x[i] + dx * cluster_gap * xr,
                          y[i] + dy * cluster_gap * yr)

    return label_xy


def annotate_with_leaders(ax, x, y, labels, colors=None, fontsize=10, **offset_kwargs):
    """
    Zeichnet Scatter-Punkte + automatisch platzierte Labels mit gestrichelten
    Leader-Lines (statt fester Text-Offsets, die bei dichten Clustern ueberlappen).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x, y : array-like
        Datenkoordinaten.
    labels : list[str]
    colors : list[str] | None
        Punktfarben; falls None, wird `get_model_color(label)` je Punkt verwendet.
    offset_kwargs :
        Weitergereicht an `compute_label_offsets` (z.B. cluster_gap=0.2).
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if colors is None:
        colors = [get_model_color(lbl) for lbl in labels]
    label_xy = compute_label_offsets(x, y, **offset_kwargs)

    for xi, yi, lbl, col, (lx, ly) in zip(x, y, labels, colors, label_xy):
        ax.scatter(xi, yi, s=90, color=col, zorder=3, edgecolor="black", linewidth=0.4)
        ax.annotate(
            lbl, xy=(xi, yi), xytext=(lx, ly),
            arrowprops=dict(arrowstyle="-", linestyle="dashed", color="gray", lw=0.8),
            fontsize=fontsize, ha="center", va="center", zorder=4,
        )