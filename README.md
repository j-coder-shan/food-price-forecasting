# 🇱🇰 Sri Lanka Food Price Forecasting System

> **A production-quality machine learning system for forecasting monthly food prices in Sri Lanka using time-series analysis.**

[![Python](https://img.shields.io/badge/Python-3.13+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)](https://streamlit.io)
[![Models](https://img.shields.io/badge/Models-6+-green)](#models)
[![License](https://img.shields.io/badge/License-MIT-yellow)](#)

---

## 📖 Project Overview

This system ingests monthly Sri Lankan food price data (2013–present), engineers time-series features, trains multiple forecasting models, evaluates them on a held-out test set, and generates 3/6/12-month future price predictions with an interactive Streamlit dashboard.

Built for **university research and final-year projects** with clean modular architecture, full logging, and extensibility for future economic indicators (inflation, exchange rates, oil prices).

---

## 📊 Dataset

| Property | Value |
|---|---|
| File | `data/food_prices.xlsx` |
| Sheet | `Sheet1` |
| Date Range | January 2013 → present |
| Frequency | Monthly |
| Target Column | `Month` (date index) |
| Food Items | Rice, Vegetables, Fish, Fruits, Meat, Milk, Oils, Spices, Sugar, Bread, Tea, etc. |
| Economic Index | `Index` column |

---

## 📁 Project Structure

```
food-price-forecasting/
│
├── data/
│   └── food_prices.xlsx          ← Sri Lankan food price dataset
│
├── src/
│   ├── utils.py                  ← Logger, paths, constants
│   ├── preprocessing.py          ← Load, parse dates, handle missing values
│   ├── feature_engineering.py    ← Lags, rolling stats, date features
│   ├── train.py                  ← Train all forecasting models
│   ├── evaluate.py               ← MAE/RMSE/MAPE/R² metrics
│   ├── predict.py                ← Generate 3/6/12-month forecasts
│   └── visualize.py              ← Charts and graphs
│
├── models/
│   └── saved_models/             ← Trained model .pkl files (per target)
│
├── outputs/
│   ├── forecasts/                ← Forecast CSV files
│   ├── metrics/                  ← Evaluation metrics CSVs
│   └── graphs/                   ← PNG chart files
│
├── notebooks/
│   └── exploration.ipynb         ← EDA notebook
│
├── main.py                       ← Full pipeline CLI
├── dashboard.py                  ← Streamlit interactive dashboard
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Installation

### 1. Clone & Enter the Project

```bash
git clone https://github.com/your-username/food-price-forecasting.git
cd food-price-forecasting
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Place Your Dataset

```
data/food_prices.xlsx
```

---

## ▶️ Usage

### Run the Full Pipeline

```bash
# Full pipeline — all food items, 12-month forecast
python main.py

# Fast mode — skip statistical models (ARIMA/SARIMA/Prophet)
python main.py --skip-statistical

# Specific targets only
python main.py --targets "Rice ‐ (Kekulu white)" Tomatoes

# 6-month horizon
python main.py --horizon 6

# Limit targets for quick test
python main.py --max-targets 3 --skip-statistical

# Launch Streamlit dashboard after pipeline
python main.py --dashboard
```

### Launch Dashboard Only

```bash
streamlit run dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🤖 Models Implemented

| Category | Model | Description |
|---|---|---|
| ML | **Linear Regression** | Baseline model with lag/rolling features |
| ML | **Random Forest** | Ensemble of 200 decision trees |
| ML | **XGBoost** | Gradient boosting with early stopping |
| ML | **LightGBM** | Fast gradient boosting (Microsoft) |
| ML | **CatBoost** | Categorical boosting (Yandex) |
| Statistical | **ARIMA(1,1,1)** | Auto-regressive integrated moving average |
| Statistical | **SARIMA(1,1,1)(1,1,1,12)** | Seasonal ARIMA for monthly patterns |
| Statistical | **Prophet** | Facebook/Meta time-series model |

---

## ⚙️ Feature Engineering

For each food item, the following features are created:

| Feature Type | Features Created |
|---|---|
| **Lag** | lag_1, lag_3, lag_6, lag_12 |
| **Rolling Mean** | rolling_mean_3, _6, _12 |
| **Rolling Std** | rolling_std_3, _6, _12 |
| **Rolling Max** | rolling_max_3, _6, _12 |
| **Rolling Min** | rolling_min_3, _6, _12 |
| **Date** | month, quarter, year |
| **Cyclic** | month_sin, month_cos |
| **Momentum** | pct_change_1, pct_change_3 |

---

## 📈 Train / Test Split

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│             TRAIN (80%)              │TEST (20%)│
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 2013-01 ─────────────────────────── ─────── 2026
```

- **NO random shuffling** — preserves temporal order
- Models trained on the 80% train portion
- Evaluated on the 20% held-out test set
- Best model reused for future forecasting

---

## 📊 Evaluation Metrics

| Metric | Formula | Meaning |
|---|---|---|
| **MAE** | mean(|actual - pred|) | Average absolute error (LKR) |
| **RMSE** | √mean((actual - pred)²) | Penalizes large errors |
| **MAPE** | mean(|error/actual|) × 100 | Percentage error |
| **R²** | 1 - SS_res/SS_tot | Variance explained (1.0 = perfect) |

---

## 🔮 Forecasting

Future forecasts are generated for **3, 6, and 12 months** ahead.

Example output:

| Month | Rice Forecast (LKR) |
|---|---|
| 2026-06 | 187.50 |
| 2026-07 | 189.20 |
| 2026-08 | 191.00 |
| ... | ... |
| 2027-05 | 210.40 |

---

## 📊 Dashboard Features

| Tab | Description |
|---|---|
| **Historical Trends** | Interactive multi-item price trend chart |
| **Model Performance** | KPI cards, metrics table, RMSE comparison, actual vs predicted |
| **Forecast** | Future price chart with ±10% confidence band + CSV download |
| **Correlations** | Interactive Pearson correlation heatmap |
| **Seasonal Analysis** | Trend / seasonal / residual decomposition |

---

## 🔭 Future Extensions

This architecture is designed for easy extension with:

- 📉 Inflation rate integration
- 💱 Exchange rate features (USD/LKR)
- 🛢️ Global oil price impact
- 🌐 Spatial-Temporal GNN (ST-GNN) models
- 📊 Economic indicator dashboards
- 🤖 LLM-based price explanation

---

## 🧠 Author

Built for academic research on Sri Lankan food price dynamics.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
