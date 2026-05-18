"""
utils.py — Shared utilities, logger setup, path constants, and helpers.
Sri Lanka Food Price Forecasting System
"""

import logging
import os
from pathlib import Path

# ─────────────────────────────────────────────
# Project Root & Path Constants
# ─────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models" / "saved_models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FORECASTS_DIR = OUTPUTS_DIR / "forecasts"
METRICS_DIR = OUTPUTS_DIR / "metrics"
GRAPHS_DIR = OUTPUTS_DIR / "graphs"

DATA_FILE     = DATA_DIR / "food_prices.xlsx"
FUEL_FILE     = DATA_DIR / "fuel_prices.xls"
EXCHANGE_FILE = DATA_DIR / "exchange_rates.xlsx"
SHEET_NAME    = "Sheet1"

# Column names for merged economic indicators
FUEL_COL     = "Brent_USD"   # Brent crude oil price (USD/barrel)
EXCHANGE_COL = "USD_LKR"    # USD to LKR exchange rate

# Train/Test split ratio (80% train, 20% test/evaluation)
TRAIN_RATIO = 0.80

# Forecasting horizons (months)
HORIZONS = [3, 6, 12]

# Default sequence length for time-series sliding windows
SEQUENCE_LENGTH = 12

# Rolling window sizes for feature engineering
ROLLING_WINDOWS = [3, 6, 12]

# Lag periods
LAG_PERIODS = [1, 3, 6, 12]

# ─────────────────────────────────────────────
# Available Models Registry
# ─────────────────────────────────────────────
AVAILABLE_MODELS = {
    "ML": [
        "LinearRegression",
        "RandomForest",
        "XGBoost",
        "LightGBM",
        "CatBoost",
    ],
    "Statistical": [
        "ARIMA",
        "SARIMA",
        # "Prophet",  # uncomment when prophet is installed
    ],
}

DEFAULT_MODELS = ["LinearRegression", "RandomForest", "XGBoost", "LightGBM", "CatBoost"]


# ─────────────────────────────────────────────
# Logger Setup
# ─────────────────────────────────────────────
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a named logger with console + file handlers.
    Log file is saved to outputs/pipeline.log.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Avoid duplicate handlers

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    ensure_dirs()
    log_path = OUTPUTS_DIR / "pipeline.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────
# Directory Helpers
# ─────────────────────────────────────────────
def ensure_dirs() -> None:
    """Create all required output and model directories if they don't exist."""
    for d in [DATA_DIR, MODEL_DIR, FORECASTS_DIR, METRICS_DIR, GRAPHS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def model_save_dir(target: str) -> Path:
    """Returns and creates the save directory for a specific food item's models."""
    safe_name = sanitize_filename(target)
    path = MODEL_DIR / safe_name
    path.mkdir(parents=True, exist_ok=True)
    return path


# ─────────────────────────────────────────────
# Column Helpers
# ─────────────────────────────────────────────
def get_food_columns(df) -> list[str]:
    """
    Auto-detects all food item price columns by excluding known non-food columns.
    Returns a sorted list of column names.
    """
    exclude = {"Month", "Index"}
    return [c for c in df.columns if c not in exclude]


def sanitize_filename(name: str) -> str:
    """
    Converts a column name (which may contain special characters) into a
    safe filesystem name.
    """
    import re
    return re.sub(r"[^\w\-]", "_", name).strip("_")


# ─────────────────────────────────────────────
# Formatting Helpers
# ─────────────────────────────────────────────
def format_metric_table(metrics: dict) -> str:
    """
    Formats a metrics dictionary as a human-readable markdown table string.

    Args:
        metrics: {model_name: {metric_name: value}}
    Returns:
        Formatted markdown table string.
    """
    if not metrics:
        return "No metrics available."

    all_metrics = list(next(iter(metrics.values())).keys())
    header = "| Model | " + " | ".join(all_metrics) + " |"
    sep = "|---|" + "---|" * len(all_metrics)
    rows = [header, sep]

    for model, vals in metrics.items():
        row = f"| {model} | " + " | ".join(f"{vals[m]:.4f}" for m in all_metrics) + " |"
        rows.append(row)

    return "\n".join(rows)
