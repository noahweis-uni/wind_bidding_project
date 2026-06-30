"""Regeneriert alle 31 WIBA-Figures aus gespeicherten Artefakten mit CRISP-DM-Styling.
Kein 40-Min-Re-Run noetig (nutzt predictions_*.csv, shap_*.pkl, bidding_detail_*.csv,
results/tables/*, NWP- und Preisdaten). Aufruf: python notebooks/regenerate_figures.py
"""
import sys, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT))
from plot_utils import CRISP_COLORS, MODEL_COLORS, SITE_COLORS, MONTH_NAMES_DE, crisp_cmap, apply_style
from src.bidding.newsvendor import compute_costs, optimal_quantile, loss, profit
import shap

apply_style()

FC = ROOT / "results" / "forecasts"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
SHAPD = ROOT / "results" / "shap_explanations"
NWP = ROOT / "data" / "raw" / "nwp"
MARKT = ROOT / "data" / "raw" / "Daten zu Marktpreisen"

PLANTS = ["Schonungen", "Schwanfeld", "Trabelsdorf", "Obbach"]
MODELS = {"xgb": "XGBoost", "qgb": "QGB", "qrf": "QRF"}
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
PP = "Schonungen"

# ---------- shared data ----------
preds = {p: pd.read_csv(FC / f"predictions_{p.lower()}.csv", parse_dates=["timestamp"]) for p in PLANTS}
shap_data = {}
for p in PLANTS:
    for m in MODELS:
        fp = SHAPD / f"shap_{m}_{p.lower()}.pkl"
        if fp.exists():
            shap_data[(p, m)] = pickle.load(open(fp, "rb"))
FEATURES = shap_data[(PP, "qgb")]["feature_names"]

def load_prices():
    from src.data.load_data import load_rebap
    reb = load_rebap()[["timestamp", "rebap"]]
    da = pd.read_csv(MARKT / "prices_standardized.csv", sep=";", decimal=",")
    da["timestamp"] = pd.to_datetime(da["Datum von"], format="%d.%m.%Y %H:%M", errors="coerce")
    da["da_price"] = pd.to_numeric(da["Preis"], errors="coerce")
    da = da[["timestamp", "da_price"]].dropna()
    pr = reb.merge(da, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    return pr

prices = load_prices()
TEST_START = min(preds[p]["timestamp"].min() for p in PLANTS)
train = prices[prices["timestamp"] < TEST_START]
c_under, c_over = compute_costs(train["da_price"].to_numpy(), train["rebap"].to_numpy())
tau_star = optimal_quantile(c_under, c_over) if (c_under + c_over) > 0 else 0.5

def qbid(d, prefix, tau):
    ex = f"{prefix}_q{int(tau*100):02d}"
    if ex in d.columns:
        return d[ex].to_numpy()
    lo = max((q for q in QUANTILES if q <= tau), default=QUANTILES[0])
    hi = min((q for q in QUANTILES if q >= tau), default=QUANTILES[-1])
    if lo == hi:
        return d[f"{prefix}_q{int(lo*100):02d}"].to_numpy()
    w = (tau - lo) / (hi - lo)
    return (1 - w) * d[f"{prefix}_q{int(lo*100):02d}"].to_numpy() + w * d[f"{prefix}_q{int(hi*100):02d}"].to_numpy()

done, failed = [], []
def save(fig, name, dpi=150):
    fig.savefig(FIG / name, dpi=dpi)
    plt.close(fig); done.append(name)

# ================= NB03-Figures =================
def fig_forecast_vs_actual():
    m = preds[PP].sort_values("timestamp").iloc[:28*24]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(m["timestamp"], m["y_true"], color="black", lw=1.3, label="Ist (Actual)")
    for col, lbl, key in [("elastic_net", "Elastic Net", "elastic_net"), ("xgb_q50", "XGBoost q50", "xgb"), ("qrf_q50", "QRF q50", "qrf")]:
        ax.plot(m["timestamp"], m[col], lw=0.9, alpha=0.85, label=lbl, color=MODEL_COLORS[key])
    ax.set_title(f"Prognose vs. Ist – {PP}"); ax.set_ylabel("Energie [MWh]"); ax.set_xlabel("Zeit")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    plt.xticks(rotation=45)
    save(fig, f"forecast_vs_actual_{PP.lower()}.png", 150)

def _band(ax, m, pre, name):
    ax.fill_between(m["timestamp"], m[f"{pre}_q10"], m[f"{pre}_q90"], alpha=0.2, color=CRISP_COLORS["light"], label="q10–q90")
    ax.fill_between(m["timestamp"], m[f"{pre}_q25"], m[f"{pre}_q75"], alpha=0.3, color=CRISP_COLORS["secondary"], label="q25–q75")
    ax.plot(m["timestamp"], m[f"{pre}_q50"], color=CRISP_COLORS["primary"], lw=1.2, label="Median (q50)")
    ax.plot(m["timestamp"], m["y_true"], color="black", lw=1.5, label="Ist (Actual)")
    ax.set_ylabel("Energie [MWh]"); ax.set_xlabel("Zeit"); ax.legend(fontsize=8)

def fig_forecast_bands():
    m = preds[PP].sort_values("timestamp").iloc[:28*24]
    fig, ax = plt.subplots(figsize=(14, 5)); _band(ax, m, "qgb", "QGB")
    ax.set_title(f"Probabilistisches Prognoseband (QGB) – {PP}"); plt.xticks(rotation=45)
    save(fig, f"forecast_bands_{PP.lower()}.png", 150)

def fig_quantile_band_qrf():
    m = preds[PP].sort_values("timestamp").iloc[:28*24]
    fig, ax = plt.subplots(figsize=(14, 5)); _band(ax, m, "qrf", "QRF")
    ax.set_title(f"Probabilistisches Prognoseband (QRF) – {PP}"); plt.xticks(rotation=45)
    save(fig, f"quantile_band_qrf_{PP.lower()}.png", 150)

def fig_pinball_single():
    pb = pd.read_csv(TAB / "pinball_evaluation.csv")
    sub = pb[pb["plant"] == PP]
    fig, ax = plt.subplots(figsize=(9, 5))
    for mdl in ["xgb", "qgb", "qrf"]:
        s = sub[sub["model"] == mdl].sort_values("quantile")
        ax.plot(s["quantile"], s["pinball_loss"], marker="o", label=MODELS[mdl], color=MODEL_COLORS[mdl])
    ax.set_xlabel("Quantil τ"); ax.set_ylabel("Pinball-Loss"); ax.set_title(f"Pinball-Loss – {PP}")
    ax.legend()
    save(fig, f"pinball_loss_{PP.lower()}.png", 150)

def fig_scatter():
    pred = preds[PP]
    sm = [("persistence", "Persistence", "gray"), ("arima", "ARIMA", "light"), ("elastic_net", "Elastic Net", "green"),
          ("xgb_q50", "XGBoost", "orange"), ("qgb_q50", "QGB", "primary"), ("qrf_q50", "QRF", "secondary")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    y = pred["y_true"].values; lim = float(np.nanpercentile(y, 99))
    for ax, (col, name, ckey) in zip(axes.ravel(), sm):
        ax.scatter(y, pred[col].values, s=4, alpha=0.15, color=CRISP_COLORS[ckey])
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_title(name)
        ax.set_xlabel("Ist [MWh]"); ax.set_ylabel("Prognose [MWh]")
    plt.suptitle(f"Ist vs. Prognose – {PP}"); plt.tight_layout()
    save(fig, f"scatter_actual_pred_{PP.lower()}.png", 150)

# ================= NB05 SHAP single =================
def fig_shap_bar():
    rows = []
    for (p, m), d in shap_data.items():
        imp = np.abs(d["shap_values"]).mean(0)
        for f, v in zip(FEATURES, imp):
            rows.append({"model": MODELS[m], "feature": f, "imp": v})
    glob = pd.DataFrame(rows).groupby(["model", "feature"])["imp"].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(10, 6))
    for ax, m in zip(axes, ["XGBoost", "QGB", "QRF"]):
        top = glob[glob["model"] == m].nlargest(15, "imp").sort_values("imp")
        ax.barh(top["feature"], top["imp"], color=MODEL_COLORS[m])
        ax.set_title(m); ax.set_xlabel("mean |SHAP|")
    plt.suptitle("Globale Feature-Importance (Mittel über 4 Standorte)"); plt.tight_layout()
    save(fig, "shap_bar_importance.png", 150)

def fig_beeswarm(m, fname, fs=(12, 8)):
    np.random.seed(42)  # SHAP-Jitter deterministisch -> reproduzierbare PNGs
    d = shap_data[(PP, m)]
    shap.summary_plot(d["shap_values"], np.asarray(d["X"]), feature_names=FEATURES, max_display=15, show=False)
    fig = plt.gcf(); fig.set_size_inches(*fs)
    plt.title(f"Beeswarm – {MODELS[m]} / {PP}", fontsize=12)
    plt.tick_params(labelsize=9); plt.tight_layout()
    save(fig, fname, 200)

def fig_dependence_3panel():
    j = FEATURES.index("nwp_ws100")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, m in zip(axes, ["xgb", "qgb", "qrf"]):
        d = shap_data[(PP, m)]; X = np.asarray(d["X"])
        ax.scatter(X[:, j], d["shap_values"][:, j], s=6, alpha=0.3, color=MODEL_COLORS[m])
        ax.axhline(0, color="k", lw=0.7, ls="--")
        ax.set_xlabel("nwp_ws100 [m/s]"); ax.set_ylabel("SHAP(nwp_ws100)"); ax.set_title(MODELS[m])
    plt.suptitle(f"Dependenz: nwp_ws100 → SHAP ({PP})"); plt.tight_layout()
    save(fig, f"shap_dependence_nwp_ws100_{PP.lower()}.png", 200)

def fig_dependence_cubic():
    d = shap_data[(PP, "qgb")]; X = np.asarray(d["X"]); sv = d["shap_values"]; j = FEATURES.index("nwp_ws100")
    ws, sh = X[:, j], sv[:, j]
    bins = np.linspace(np.nanmin(ws), np.nanmax(ws), 25); bi = np.digitize(ws, bins)
    bc, bm = [], []
    for b in range(1, len(bins)):
        m = bi == b
        if m.sum() > 5: bc.append(ws[m].mean()); bm.append(sh[m].mean())
    bc, bm = np.array(bc), np.array(bm); win = (bc >= 3) & (bc <= 13)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(ws, sh, s=6, alpha=0.18, color=CRISP_COLORS["secondary"], label="SHAP Samples")
    ax.plot(bc, bm, color="black", lw=2.2, marker="o", ms=4, label="Binned mean SHAP")
    if win.sum() >= 2:
        k = np.sum(bm[win] * bc[win]**3) / np.sum(bc[win]**6)
        ax.plot(bc[win], k * bc[win]**3, color=CRISP_COLORS["red"], lw=2.2, ls="--", label="~ v³ (Kubik) Referenz")
    ax.axhline(0, color="gray", lw=0.7); ax.set_xlabel("nwp_ws100 [m/s]"); ax.set_ylabel("SHAP(nwp_ws100) [MWh]")
    ax.set_title(f"Dependenz nwp_ws100 mit Kubik-Referenz – {MODELS['qgb']} / {PP}"); ax.legend()
    save(fig, f"shap_dependence_cubic_qgb_{PP.lower()}.png", 150)

def fig_waterfall():
    d = shap_data[(PP, "qgb")]; sv = d["shap_values"]; X = np.asarray(d["X"]); base = float(d["base_value"])
    i = int(np.argmax(np.abs(sv).sum(1)))
    expl = shap.Explanation(values=sv[i], base_values=base, data=X[i], feature_names=FEATURES)
    shap.plots.waterfall(expl, max_display=12, show=False)
    fig = plt.gcf(); fig.set_size_inches(12, 8); plt.title(f"SHAP Waterfall – {MODELS['qgb']} / {PP} (Beispiel-Stunde)")
    plt.tight_layout(); save(fig, f"shap_waterfall_qgb_{PP.lower()}.png", 200)

def fig_force():
    d = shap_data[(PP, "qgb")]; sv = d["shap_values"]; X = np.asarray(d["X"]); base = float(d["base_value"])
    i = int(np.argmax(np.abs(sv).sum(1)))
    # Nur Top-6 Features + Rest aggregiert (sonst ueberlappen 25 Labels)
    order = np.argsort(np.abs(sv[i]))[::-1]
    k = 6; keep = order[:k]; rest = order[k:]
    sv_red = np.concatenate([sv[i][keep], [sv[i][rest].sum()]])
    names = [f"{FEATURES[j]}={X[i][j]:.2f}" for j in keep] + [f"+{len(rest)} weitere"]
    shap.plots.force(base, sv_red, feature_names=names, matplotlib=True, show=False)
    fig = plt.gcf(); fig.set_size_inches(20, 3)
    for t in fig.axes[0].texts:
        t.set_fontsize(8)
    plt.tight_layout(); save(fig, f"shap_force_qgb_{PP.lower()}.png", 200)

def fig_seasonal_importance():
    d = shap_data[(PP, "qgb")]; ts = pd.to_datetime(d["timestamps"]); A = np.abs(d["shap_values"])
    SEAS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Frühling", 4: "Frühling", 5: "Frühling",
            6: "Sommer", 7: "Sommer", 8: "Sommer", 9: "Herbst", 10: "Herbst", 11: "Herbst"}
    season = pd.Series(ts.month).map(SEAS); order = ["Winter", "Frühling", "Sommer", "Herbst"]
    topj = np.argsort(A.mean(0))[::-1][:8]
    mat = np.zeros((len(topj), 4))
    for si, s in enumerate(order):
        msk = (season == s).values
        if msk.sum() > 0: mat[:, si] = A[msk][:, topj].mean(0)
    fig, ax = plt.subplots(figsize=(10, 7)); im = ax.imshow(mat, aspect="auto", cmap=crisp_cmap())
    ax.set_xticks(range(4)); ax.set_xticklabels(order)
    ax.set_yticks(range(len(topj))); ax.set_yticklabels([FEATURES[j] for j in topj])
    for yi in range(len(topj)):
        for xi in range(4):
            ax.text(xi, yi, f"{mat[yi,xi]:.3f}", ha="center", va="center",
                    color="white" if mat[yi, xi] > mat.max()*0.55 else "black", fontsize=8)
    ax.set_title(f"Saisonale Feature-Importance (mean|SHAP|) – {MODELS['qgb']} / {PP}")
    plt.colorbar(im, ax=ax, label="mean |SHAP|"); plt.tight_layout()
    save(fig, f"shap_seasonal_importance_{PP.lower()}.png", 150)

def fig_heatmap_hour_month():
    d = shap_data[(PP, "qgb")]; ts = pd.to_datetime(d["timestamps"]); j = FEATURES.index("nwp_ws100")
    hm = pd.DataFrame({"hour": ts.hour, "month": ts.month, "a": np.abs(d["shap_values"][:, j])})
    grid = hm.groupby(["hour", "month"])["a"].mean().unstack("month")
    fig, ax = plt.subplots(figsize=(14, 8)); im = ax.imshow(grid.values, aspect="auto", cmap=crisp_cmap(), origin="lower")
    ax.set_xticks(range(grid.shape[1])); ax.set_xticklabels([MONTH_NAMES_DE[c-1] for c in grid.columns])
    ax.set_yticks(range(grid.shape[0])); ax.set_yticklabels(grid.index)
    ax.set_xlabel("Monat"); ax.set_ylabel("Stunde des Tages")
    ax.set_title(f"|SHAP(nwp_ws100)| über Stunde × Monat – {MODELS['qgb']} / {PP}")
    plt.colorbar(im, ax=ax, label="mean |SHAP|"); plt.tight_layout()
    save(fig, f"shap_heatmap_hour_month_{PP.lower()}.png", 150)

# ================= NB05 model comparison =================
def fig_model_comparison():
    c = pd.read_csv(TAB / "model_comparison.csv")
    fig, ax = plt.subplots(figsize=(10, 6)); x = np.arange(len(c)); w = 0.25
    for i, (metric, col) in enumerate([("MAE", "primary"), ("RMSE", "orange"), ("Pinball", "green")]):
        ax.bar(x + (i-1)*w, c[metric], w, label=metric, color=CRISP_COLORS[col])
    ax.set_xticks(x); ax.set_xticklabels(c["model"]); ax.set_ylabel("Verlust [MWh / Pinball]")
    ax.set_title("Modellvergleich – Mittlere Fehlermetriken (alle Standorte)"); ax.legend()
    save(fig, "model_comparison_bars.png", 150)

def fig_pinball_over_quantiles():
    pb = pd.read_csv(TAB / "pinball_evaluation.csv")
    agg = pb.groupby(["model", "quantile"])["pinball_loss"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    for mdl in ["xgb", "qgb", "qrf"]:
        s = agg[agg["model"] == mdl].sort_values("quantile")
        ax.plot(s["quantile"], s["pinball_loss"], marker="o", label=MODELS[mdl], color=MODEL_COLORS[mdl])
    ax.set_xlabel("Quantil τ"); ax.set_ylabel("Pinball-Loss"); ax.set_title("Pinball-Loss über Quantile")
    ax.legend()
    save(fig, "pinball_over_quantiles.png", 150)

# ================= NB04 economics =================
def base_of(m):
    for k in ["Persistence", "Elastic_Net", "XGBoost", "QGB", "QRF", "Oracle"]:
        if m.startswith(k): return k
    return m

def fig_bidding_portfolio():
    br = pd.read_csv(TAB / "bidding_results.csv")
    port = br[br["scenario"] == "Portfolio"].sort_values("mean_nv_loss")
    import matplotlib.colors as mc
    def lighten(hexc, amt=0.45):
        c = np.array(mc.to_rgb(hexc)); return tuple(c + (1 - c) * amt)
    def col(m):
        b = base_of(m)
        if b == "Oracle": return CRISP_COLORS["yellow"]
        if "DR-" in m: return CRISP_COLORS["red"]
        if b in ("Persistence", "Elastic_Net"): return MODEL_COLORS[b]
        base_c = MODEL_COLORS.get(b, CRISP_COLORS["primary"])
        return lighten(base_c) if "q50" in m else base_c   # q50 heller als tau*
    colors = [col(m) for m in port["model"]]
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(port["model"], port["mean_nv_loss"], color=colors)
    ax.set_xlabel("Mittlerer Newsvendor-Verlust [EUR/MWh]")
    ax.set_title("Portfolio-Bidding – Mittlerer Newsvendor-Verlust je Strategie")
    ax.invert_yaxis()
    save(fig, "bidding_portfolio_nvloss.png", 150)

def _econ_best_per_base(br):
    e = br[(br["model"] != "Oracle") & (br["scenario"] != "Portfolio")].copy()
    e["base"] = e["model"].apply(base_of)
    return e

def fig_interpretable_vs_blackbox():
    br = pd.read_csv(TAB / "bidding_results.csv"); e = _econ_best_per_base(br)
    best = e.groupby("base")["mean_nv_loss"].min().reindex(["Persistence", "Elastic_Net", "XGBoost", "QGB", "QRF"])
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [MODEL_COLORS[b] for b in best.index]   # Persistence=grau via MODEL_COLORS
    ax.bar(best.index, best.values, color=colors)
    ax.set_ylabel("Bester NV-Verlust [EUR/MWh]")
    ax.set_title("Interpretierbar (Persistence grau / Elastic Net grün) vs. Black-Box")
    save(fig, "interpretable_vs_blackbox.png", 150)

def fig_decision_aware():
    br = pd.read_csv(TAB / "bidding_results.csv"); e = _econ_best_per_base(br)
    econ = e.groupby("base")["mean_nv_loss"].min()
    pb = pd.read_csv(TAB / "pinball_evaluation.csv").groupby("model")["pinball_loss"].mean()
    rows = [{"model": MODELS[m], "Pinball": pb[m], "NV": econ[{"xgb":"XGBoost","qgb":"QGB","qrf":"QRF"}[m]]} for m in ["xgb","qgb","qrf"]]
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 7))
    offs = {"XGBoost": (8, 8), "QGB": (8, -14), "QRF": (-40, 8)}
    for _, r in df.iterrows():
        ax.scatter(r["Pinball"], r["NV"], s=110, color=MODEL_COLORS[r["model"]], zorder=3)
        ax.annotate(r["model"], (r["Pinball"], r["NV"]), xytext=offs.get(r["model"], (6, 6)),
                    textcoords="offset points", fontsize=11)
    ax.set_xlabel("Pinball-Loss (↓ besser)"); ax.set_ylabel("NV-Verlust [EUR/MWh] (↓ besser)")
    ax.set_title("Decision-aware: Genauigkeit vs. ökonomischer Wert (probabilistische Modelle)")
    save(fig, "decision_aware_accuracy_vs_economics.png", 150)

def fig_risk_profit():
    dfp = preds[PP].merge(prices[["timestamp", "rebap", "da_price"]], on="timestamp", how="inner")
    y = dfp["y_true"].values
    fig, ax = plt.subplots(figsize=(10, 5))
    series = [("Elastic Net", dfp["elastic_net"].values, MODEL_COLORS["elastic_net"]),
              ("QGB τ*", qbid(dfp, "qgb", tau_star), MODEL_COLORS["qgb"])]
    allp = []
    for lbl, bid, c in series:
        pr = profit(np.clip(bid, 0, None), y, dfp["da_price"].values, dfp["rebap"].values); allp.append(pr)
    lo = min(np.percentile(p, 2) for p in allp); hi = max(np.percentile(p, 98) for p in allp)
    for (lbl, bid, c), pr in zip(series, allp):
        ax.hist(pr, bins=50, range=(lo, hi), alpha=0.55, label=lbl, color=c)
    ax.set_xlim(lo, hi); ax.set_xlabel("Stundenprofit [EUR]"); ax.set_ylabel("Häufigkeit")
    ax.set_title(f"Risk Exposure: Profit-Verteilung – {PP}"); ax.legend()
    save(fig, "risk_profit_distribution.png", 150)

# ================= NB04/NB05 cross-site =================
def fig_crosssite_forecast_accuracy():
    from sklearn.metrics import mean_absolute_error
    sites = PLANTS
    POINT = {"persistence": "Persistence", "arima": "ARIMA", "elastic_net": "Elastic Net"}
    STYLE = {"persistence": ":", "arima": "--", "elastic_net": "-"}
    pb = pd.read_csv(TAB / "pinball_evaluation.csv").groupby(["plant", "model"])["pinball_loss"].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5)); x = np.arange(len(sites))
    for c, n in POINT.items():
        vals = [mean_absolute_error(preds[s]["y_true"], preds[s][c]) for s in sites]
        axes[0].plot(x, vals, marker="o", ls=STYLE[c], label=n, color=MODEL_COLORS[c])
    axes[0].set_xticks(x); axes[0].set_xticklabels(sites, rotation=15); axes[0].set_ylabel("MAE [MWh]")
    axes[0].set_title("POINT Forecasts – MAE je Standort"); axes[0].legend()
    for m in ["xgb", "qgb", "qrf"]:
        vals = [pb[(pb["plant"] == s) & (pb["model"] == m)]["pinball_loss"].values[0] for s in sites]
        axes[1].plot(x, vals, marker="o", label=MODELS[m], color=MODEL_COLORS[m])
    axes[1].set_xticks(x); axes[1].set_xticklabels(sites, rotation=15); axes[1].set_ylabel("Pinball-Loss")
    axes[1].set_title("PROBABILISTIC Forecasts – Pinball je Standort"); axes[1].legend()
    plt.suptitle("Forecast-Genauigkeit getrennt: Point (MAE) | Probabilistic (Pinball)"); plt.tight_layout()
    save(fig, "crosssite_forecast_accuracy.png", 150)

def fig_crosssite_economic():
    br = pd.read_csv(TAB / "bidding_results.csv"); e = _econ_best_per_base(br)
    best = e.groupby(["scenario", "base"])["mean_nv_loss"].min().reset_index()
    sites = sorted(best["scenario"].unique()); bases = ["Persistence", "Elastic_Net", "XGBoost", "QGB", "QRF"]
    PT = {"Persistence": "Point", "Elastic_Net": "Point", "XGBoost": "Prob", "QGB": "Prob", "QRF": "Prob"}
    fig, ax = plt.subplots(figsize=(11, 6)); x = np.arange(len(bases))
    for s in sites:
        vals = [best[(best["scenario"] == s) & (best["base"] == b)]["mean_nv_loss"].values for b in bases]
        vals = [v[0] if len(v) else np.nan for v in vals]
        mk = ["s" if PT[b] == "Point" else "o" for b in bases]
        ax.plot(x, vals, "-", color=SITE_COLORS[s], label=s, alpha=0.9)
        for xi, b in enumerate(bases):
            ax.scatter(xi, vals[xi], marker="s" if PT[b] == "Point" else "o", color=SITE_COLORS[s], s=55, zorder=3)
    ax.axvline(1.5, color="gray", ls=":", lw=1)
    ax.text(0.5, ax.get_ylim()[1]*0.95, "Point", ha="center", fontsize=9, color="gray")
    ax.text(3, ax.get_ylim()[1]*0.95, "Probabilistic", ha="center", fontsize=9, color="gray")
    ax.set_xticks(x); ax.set_xticklabels(bases, rotation=15); ax.set_ylabel("Bester NV-Verlust [EUR/MWh]")
    ax.set_title("Ökonomie je Standort – Quadrat = Point, Kreis = Probabilistic"); ax.legend(title="Standort", fontsize=8)
    save(fig, "crosssite_economic_point_vs_prob.png", 150)

def fig_crosssite_shap_importance():
    rows = []
    for s in PLANTS:
        d = shap_data[(s, "qgb")]; imp = np.abs(d["shap_values"]).mean(0)
        for f, v in zip(FEATURES, imp): rows.append({"site": s, "feature": f, "imp": v})
    idf = pd.DataFrame(rows).pivot_table(index="feature", columns="site", values="imp")
    top = idf.mean(1).nlargest(12).index; idf = idf.loc[top]
    fig, ax = plt.subplots(figsize=(10, 8)); im = ax.imshow(idf.values, aspect="auto", cmap=crisp_cmap())
    ax.set_xticks(range(len(idf.columns))); ax.set_xticklabels(idf.columns, rotation=20)
    ax.set_yticks(range(len(idf.index))); ax.set_yticklabels(idf.index)
    ax.set_title("SHAP Feature-Importance × Standort – QGB (probabilistisch)")
    plt.colorbar(im, ax=ax, label="mean |SHAP|"); plt.tight_layout()
    save(fig, "crosssite_shap_importance_qgb.png", 150)

def fig_crosssite_dependence():
    j = FEATURES.index("nwp_ws100")
    fig, ax = plt.subplots(figsize=(10, 6))
    for s in PLANTS:
        d = shap_data[(s, "qgb")]; X = np.asarray(d["X"]); ws = X[:, j]; sh = d["shap_values"][:, j]
        b = np.linspace(np.nanmin(ws), np.nanmax(ws), 20); bi = np.digitize(ws, b); bc, bm = [], []
        for k in range(1, len(b)):
            m = bi == k
            if m.sum() > 5: bc.append(ws[m].mean()); bm.append(sh[m].mean())
        ax.plot(bc, bm, marker="o", ms=3, label=s, color=SITE_COLORS[s])
    ax.axhline(0, color="gray", lw=0.7); ax.set_xlabel("nwp_ws100 [m/s]"); ax.set_ylabel("SHAP(nwp_ws100) [MWh]")
    ax.set_title("Dependenz nwp_ws100 × Standort – QGB"); ax.legend()
    save(fig, "crosssite_dependence_nwp_ws100.png", 150)

def fig_crosssite_seasonal():
    j = FEATURES.index("nwp_ws100"); SEAS = {12: "Winter", 1: "Winter", 2: "Winter", 6: "Sommer", 7: "Sommer", 8: "Sommer"}
    rows = []
    for s in PLANTS:
        d = shap_data[(s, "qgb")]; ts = pd.to_datetime(d["timestamps"]); a = np.abs(d["shap_values"][:, j])
        se = pd.Series(ts.month).map(SEAS)
        for season in ["Winter", "Sommer"]:
            m = (se == season).values
            if m.sum() > 0: rows.append({"site": s, "season": season, "imp": float(a[m].mean())})
    sd = pd.DataFrame(rows); fig, ax = plt.subplots(figsize=(10, 6)); x = np.arange(len(PLANTS)); w = 0.35
    for i, (season, col) in enumerate([("Winter", "primary"), ("Sommer", "orange")]):
        vals = [sd[(sd["site"] == s) & (sd["season"] == season)]["imp"].values for s in PLANTS]
        vals = [v[0] if len(v) else 0 for v in vals]
        ax.bar(x + (i-0.5)*w, vals, w, label=season, color=CRISP_COLORS[col])
    ax.set_xticks(x); ax.set_xticklabels(PLANTS); ax.set_ylabel("mean |SHAP(nwp_ws100)|")
    ax.set_title("Saisonale Wind-Importance × Standort – QGB (Winter vs. Sommer)"); ax.legend()
    save(fig, "crosssite_seasonal_wind.png", 150)

def fig_crosssite_bidfailure():
    fig, ax = plt.subplots(figsize=(10, 6)); markers = {"Schonungen": "o", "Schwanfeld": "s", "Trabelsdorf": "^", "Obbach": "D"}
    for s in PLANTS:
        det = pd.read_csv(FC / f"bidding_detail_{s.lower()}.csv", parse_dates=["timestamp"])
        nwp = pd.read_csv(NWP / f"nwp_{s.lower()}.csv", parse_dates=["timestamp"])[["timestamp", "nwp_ws100"]]
        fa = det.merge(nwp, on="timestamp", how="left")
        b = np.linspace(fa["nwp_ws100"].min(), fa["nwp_ws100"].max(), 20); fa["wb"] = np.digitize(fa["nwp_ws100"], b)
        g = fa.groupby("wb").agg(ws=("nwp_ws100", "mean"), loss=("nv_loss", "mean"))
        ax.plot(g["ws"], g["loss"], marker=markers[s], ms=4, label=s, color=SITE_COLORS[s])
    ax.set_xlabel("nwp_ws100 [m/s]"); ax.set_ylabel("mean NV-Verlust")
    ax.set_title("Gebotsverlust: NV-Loss vs. Windgeschwindigkeit × Standort (QGB τ*)"); ax.legend()
    save(fig, "crosssite_bidfailure_wind.png", 150)

def fig_bidfailure_conditions():
    det = pd.read_csv(FC / f"bidding_detail_{PP.lower()}.csv", parse_dates=["timestamp"])
    nwp = pd.read_csv(NWP / f"nwp_{PP.lower()}.csv", parse_dates=["timestamp"])[["timestamp", "nwp_ws100"]]
    fa = det.merge(nwp, on="timestamp", how="left")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    b = np.linspace(fa["nwp_ws100"].min(), fa["nwp_ws100"].max(), 20); fa["wb"] = np.digitize(fa["nwp_ws100"], b)
    g = fa.groupby("wb").agg(ws=("nwp_ws100", "mean"), loss=("nv_loss", "mean"))
    axes[0].plot(g["ws"], g["loss"], marker="o", color=CRISP_COLORS["primary"])
    axes[0].set_xlabel("Windgeschwindigkeit [m/s]"); axes[0].set_ylabel("mean NV-Verlust"); axes[0].set_title("Gebotsverlust vs. Windgeschwindigkeit")
    b2 = np.linspace(fa["q_spread"].min(), fa["q_spread"].quantile(0.99), 20); fa["sb"] = np.digitize(fa["q_spread"], b2)
    g2 = fa.groupby("sb").agg(sp=("q_spread", "mean"), loss=("nv_loss", "mean"))
    axes[1].plot(g2["sp"], g2["loss"], marker="o", color=CRISP_COLORS["orange"])
    axes[1].set_xlabel("Quantil-Spread q90–q10 (Unsicherheit)"); axes[1].set_ylabel("mean NV-Verlust"); axes[1].set_title("Gebotsverlust vs. Prognose-Unsicherheit")
    plt.tight_layout(); save(fig, f"bid_failure_conditions_{PP.lower()}.png", 150)

def fig_bidfailure_heatmap():
    det = pd.read_csv(FC / f"bidding_detail_{PP.lower()}.csv", parse_dates=["timestamp"])
    det["hour"] = det["timestamp"].dt.hour; det["month"] = det["timestamp"].dt.month
    grid = det.groupby(["hour", "month"])["nv_loss"].mean().unstack("month")
    fig, ax = plt.subplots(figsize=(14, 8)); im = ax.imshow(grid.values, aspect="auto", cmap=crisp_cmap(), origin="lower")
    ax.set_xticks(range(grid.shape[1])); ax.set_xticklabels([MONTH_NAMES_DE[c-1] for c in grid.columns])
    ax.set_yticks(range(grid.shape[0])); ax.set_yticklabels(grid.index)
    ax.set_xlabel("Monat"); ax.set_ylabel("Stunde des Tages")
    ax.set_title(f"Gebotsverlust (NV-Loss) über Stunde × Monat – {PP}")
    plt.colorbar(im, ax=ax, label="mean NV-Verlust"); plt.tight_layout()
    save(fig, f"bid_failure_heatmap_{PP.lower()}.png", 150)

# ================= NB01 weather =================
def fig_weather():
    nwp_sites = {}
    for s in PLANTS:
        d = pd.read_csv(NWP / f"nwp_{s.lower()}.csv", parse_dates=["timestamp"])
        nwp_sites[s] = d.set_index("timestamp")["nwp_ws100"]
    W = pd.DataFrame(nwp_sites)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    win = W.loc["2024-01-01":"2024-01-14"]
    for s in PLANTS: axes[0].plot(win.index, win[s], lw=0.9, label=s, color=SITE_COLORS[s])
    axes[0].set_title("NWP Windgeschwindigkeit 100 m – 14-Tage-Ausschnitt (Jan 2024)")
    axes[0].set_ylabel("m/s"); axes[0].legend(fontsize=8)
    monthly = W.groupby(W.index.month).mean()
    x = np.arange(1, 13)
    for s in PLANTS: axes[1].plot(x, monthly[s].reindex(range(1,13)), marker="o", label=s, color=SITE_COLORS[s])
    axes[1].set_xticks(x); axes[1].set_xticklabels(MONTH_NAMES_DE)
    axes[1].set_title("Mittlere Windgeschwindigkeit 100 m je Monat")
    axes[1].set_ylabel("m/s"); axes[1].set_xlabel("Monat"); axes[1].legend(fontsize=8)
    plt.tight_layout(); save(fig, "weather_nwp_sites.png", 150)

ALL = [
    fig_forecast_vs_actual, fig_forecast_bands, fig_quantile_band_qrf, fig_pinball_single, fig_scatter,
    fig_shap_bar,
    lambda: fig_beeswarm("qgb", f"shap_beeswarm_qgb_{PP.lower()}.png", (12, 8)),
    lambda: fig_beeswarm("qgb", f"shap_beeswarm_featured_qgb_{PP.lower()}.png", (12, 9)),
    lambda: fig_beeswarm("qrf", f"shap_beeswarm_qrf_{PP.lower()}.png", (12, 8)),
    lambda: fig_beeswarm("xgb", f"shap_beeswarm_xgb_{PP.lower()}.png", (12, 8)),
    fig_dependence_3panel, fig_dependence_cubic, fig_waterfall, fig_force,
    fig_seasonal_importance, fig_heatmap_hour_month,
    fig_model_comparison, fig_pinball_over_quantiles,
    fig_bidding_portfolio, fig_interpretable_vs_blackbox, fig_decision_aware, fig_risk_profit,
    fig_crosssite_forecast_accuracy, fig_crosssite_economic, fig_crosssite_shap_importance,
    fig_crosssite_dependence, fig_crosssite_seasonal, fig_crosssite_bidfailure,
    fig_bidfailure_conditions, fig_bidfailure_heatmap, fig_weather,
]

if __name__ == "__main__":
    for fn in ALL:
        try:
            fn()
        except Exception as e:
            failed.append((getattr(fn, "__name__", "lambda"), str(e)[:120]))
            print("FAILED", getattr(fn, "__name__", "lambda"), "->", str(e)[:120])
    print(f"\nDONE: {len(done)} figures generated")
    for f in failed:
        print("  FAILED:", f)
