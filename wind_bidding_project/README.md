# Wind Bidding under Uncertainty

Seminar WIBA – Decision-Focused Wind Energy Forecasting

## Projektziel

Vergleich verschiedener Forecast-Modelle für Windproduktion und Bewertung ihrer
ökonomischen Auswirkungen auf Bidding-Entscheidungen im Strommarkt (Day-Ahead vs. Balancing).

Die zentrale Frage: **Ist das beste Forecast-Modell auch das beste Trading-Modell?**

## Modelle

| Modell              | Typ             | Output    | Warum                                      |
|---------------------|-----------------|-----------|--------------------------------------------|
| Linear Regression   | Baseline        | Punkt     | Interpretierbar, guter Referenzpunkt       |
| Random Forest       | Tree-based ML   | Punkt     | Robust, semi-interpretierbar               |
| Neural Network      | Black Box       | Punkt     | Performance-Vergleich                      |
| Quantile Regression | Probabilistisch | Quantile  | Direkt optimal für Newsvendor-Entscheidung |

## Bidding-Benchmarks

- **Naive Point Forecast**: Gebot = Punkt-Prognose, keine Unsicherheit
- **Quantile-based Bidding**: Gebot = Quantil aus Newsvendor-Logik
- **Oracle**: Perfektes Gebot mit echter Produktion (obere Schranke)

## Bewertungsmetriken

- Forecast: RMSE, MAE
- Economic: Profit, Newsvendor Loss, Regret vs. Oracle

## Projektstruktur

```
wind_bidding_project/
├── data/
│   ├── raw/          # Excel-Originaldaten (lokal, nicht im Repo)
│   ├── interim/      # Zwischenschritte
│   └── processed/    # final_dataset.csv für Modelle
├── notebooks/        # Explorative Analyse & Experimente
├── src/
│   ├── data/         # Laden & Feature Engineering
│   ├── models/       # Forecast-Modelle
│   ├── forecasting/  # Training, Prediction, Evaluation
│   ├── bidding/      # Entscheidungsregeln & Profit-Berechnung
│   ├── xai/          # SHAP & Interpretierbarkeit
│   └── utils/        # Metriken & Plotting
├── results/          # Outputs: Plots, Tabellen, Forecasts
└── reports/          # Hausarbeit, Präsentation, Literatur
```

## Setup

```bash
git clone https://github.com/<username>/wind_bidding_project.git
cd wind_bidding_project
pip install -r requirements.txt
```

## Ausführung (Reihenfolge)

```bash
# 1. Daten verstehen
jupyter notebook notebooks/01_data_understanding.ipynb

# 2. Daten aufbereiten → erzeugt data/processed/final_dataset.csv
jupyter notebook notebooks/02_preprocessing.ipynb

# 3. Modelle trainieren & vergleichen
jupyter notebook notebooks/03_forecasting_models.ipynb

# 4. Bidding-Performance bewerten
jupyter notebook notebooks/04_bidding_evaluation.ipynb

# 5. Interpretierbarkeit analysieren
jupyter notebook notebooks/05_xai_analysis.ipynb
```

## Daten

- Quelle: 10-Minuten Winddaten Schonungen 2016
- Aggregation: stündlich (Mittelwert)
- Zielvariable: `power` (Leistung in kW)
- Features: `wind_speed`, `hour`, `dayofweek`, (optional: `rotor_speed`, `gondel_position`)
- Ergänzung: Day-Ahead Preise + reBAP für Bidding-Evaluation

## Literatur

- Pinson (2023): Distributionally Robust Trading Strategies for Renewable Energy Producers
- Minh et al.: Explainable Artificial Intelligence – A Comprehensive Review
