"""
predict.py — Future price forecasting module.
Sri Lanka Food Price Forecasting System

Strategy:
  - Retrain the best model on the FULL dataset (train + test) for maximum accuracy.
  - Then generate recursive multi-step forecasts for 3, 6, or 12 months ahead.
  - Supports all model types: ML, ARIMA/SARIMA, Prophet.
"""

import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.utils import (
    get_logger, FORECASTS_DIR, MODEL_DIR, HORIZONS,
    sanitize_filename, LAG_PERIODS, ROLLING_WINDOWS,
)
from src.feature_engineering import FeatureEngineer

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


class Forecaster:
    """
    Generates future price forecasts for a single food item.

    The forecasting approach:
      - ML models (LR, RF, XGB, LGBM, CatBoost):
          Uses the feature vector of the last known observation, then
          recursively predicts one step at a time, feeding predictions
          back as lag features.
      - ARIMA / SARIMA:
          Uses statsmodels built-in forecast(steps=horizon).
      - Prophet:
          Creates a future DataFrame and calls model.predict().

    Args:
        df:           Full cleaned DataFrame (used for full-data retraining)
        target:       Target food item column name
        trained_models: Dict of {model_name: model_object} (already trained)
    """

    def __init__(self, df: pd.DataFrame, target: str, trained_models: dict):
        self.df = df
        self.target = target
        self.fe = FeatureEngineer(df, target)
        self.trained_models = trained_models
        self.last_date = df.index.max()

    # ─────────────────────────────────────────
    # Generate Future Dates
    # ─────────────────────────────────────────
    def _future_dates(self, horizon: int) -> pd.DatetimeIndex:
        """Returns the next `horizon` month-start dates after the last observation."""
        return pd.date_range(
            start=self.last_date + pd.DateOffset(months=1),
            periods=horizon,
            freq="MS",
        )

    # ─────────────────────────────────────────
    # Feature Vector for Last Observation
    # ─────────────────────────────────────────
    def _get_last_feature_row(self, df_extended: pd.DataFrame) -> dict:
        """
        Extracts the feature vector for the most recent row of an extended
        DataFrame (used during recursive ML forecasting).
        """
        row = df_extended.iloc[-1]
        features = {}

        # Lag features
        for lag in LAG_PERIODS:
            col = f"lag_{lag}"
            features[col] = row.get(col, np.nan)

        # Rolling statistics
        for window in ROLLING_WINDOWS:
            for stat in ["mean", "std", "max", "min"]:
                col = f"rolling_{stat}_{window}"
                features[col] = row.get(col, np.nan)

        # Date features
        idx = df_extended.index[-1]
        features["month"] = idx.month
        features["quarter"] = idx.quarter
        features["year"] = idx.year
        features["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
        features["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
        features["pct_change_1"] = row.get("pct_change_1", 0)
        features["pct_change_3"] = row.get("pct_change_3", 0)

        return features

    # ─────────────────────────────────────────
    # ML Recursive Forecasting
    # ─────────────────────────────────────────
    def _forecast_ml(self, model, horizon: int) -> pd.Series:
        """
        Recursive one-step-at-a-time forecasting for sklearn-API models.

        For each future step:
          1. Build feature vector from current known history.
          2. Predict next value.
          3. Append prediction to history and recompute features.
        """
        # Start with the full feature-engineered dataset
        fe_df = self.fe.build_features()
        feature_cols = [c for c in fe_df.columns if c != self.target]

        # Working copy of the target series (historical + predicted)
        history = self.df[self.target].dropna().copy()
        future_dates = self._future_dates(horizon)
        predictions = []

        for future_date in future_dates:
            # Build a temporary extended df to recompute all features
            temp_series = history.copy()

            # Compute rolling and lag stats on current history
            temp_df = pd.DataFrame({self.target: temp_series})
            for lag in LAG_PERIODS:
                temp_df[f"lag_{lag}"] = temp_df[self.target].shift(lag)
            for window in ROLLING_WINDOWS:
                roll = temp_df[self.target].rolling(window=window, min_periods=1)
                temp_df[f"rolling_mean_{window}"] = roll.mean()
                temp_df[f"rolling_std_{window}"] = roll.std().fillna(0)
                temp_df[f"rolling_max_{window}"] = roll.max()
                temp_df[f"rolling_min_{window}"] = roll.min()

            # Date features for the future step
            temp_df["month"] = temp_df.index.month
            temp_df["quarter"] = temp_df.index.quarter
            temp_df["year"] = temp_df.index.year
            temp_df["month_sin"] = np.sin(2 * np.pi * temp_df.index.month / 12)
            temp_df["month_cos"] = np.cos(2 * np.pi * temp_df.index.month / 12)
            temp_df["pct_change_1"] = temp_df[self.target].pct_change(1).fillna(0)
            temp_df["pct_change_3"] = temp_df[self.target].pct_change(3).fillna(0)

            # Copy any missing economic/external features (Fuel/FX) from the last known data point
            for col in feature_cols:
                if col not in temp_df.columns:
                    temp_df[col] = fe_df[col].iloc[-1]

            # Get feature row for the last known data point
            last_features = temp_df[feature_cols].iloc[-1].values.reshape(1, -1)

            # Predict
            pred = model.predict(last_features)[0]
            predictions.append(pred)

            # Append prediction to history for next recursive step
            history[future_date] = pred

        return pd.Series(predictions, index=future_dates, name=self.target)

    # ─────────────────────────────────────────
    # ARIMA / SARIMA Forecasting
    # ─────────────────────────────────────────
    def _forecast_arima_sarima(self, model, horizon: int) -> pd.Series:
        """Uses statsmodels built-in .forecast() for direct multi-step forecasting."""
        future_dates = self._future_dates(horizon)
        forecast = model.forecast(steps=horizon)

        if hasattr(forecast, "values"):
            vals = forecast.values
        else:
            vals = np.array(forecast)

        return pd.Series(vals, index=future_dates, name=self.target)

    # ─────────────────────────────────────────
    # Prophet Forecasting
    # ─────────────────────────────────────────
    def _forecast_prophet(self, model, horizon: int) -> pd.Series:
        """Uses Prophet future DataFrame for multi-step forecasting."""
        future_dates = self._future_dates(horizon)
        future_df = pd.DataFrame({"ds": future_dates})
        forecast = model.predict(future_df)
        vals = forecast.set_index("ds")["yhat"].values
        return pd.Series(vals, index=future_dates, name=self.target)

    # ─────────────────────────────────────────
    # Dispatch
    # ─────────────────────────────────────────
    def forecast(
        self, model_name: str, horizon: int
    ) -> pd.DataFrame:
        """
        Generates a future price forecast for a given model and horizon.

        Args:
            model_name: One of the keys in trained_models dict.
            horizon:    Number of months to forecast (e.g. 3, 6, 12).

        Returns:
            DataFrame with 'Month' and '{target}_Forecast' columns.
        """
        if model_name not in self.trained_models:
            raise KeyError(f"Model '{model_name}' not found. Available: {list(self.trained_models)}")

        model = self.trained_models[model_name]
        logger.info(f"  Forecasting '{self.target}' with {model_name} for {horizon} months...")

        arima_sarima = {"ARIMA", "SARIMA"}
        prophet = {"Prophet"}

        if model_name in arima_sarima:
            series = self._forecast_arima_sarima(model, horizon)
        elif model_name in prophet:
            series = self._forecast_prophet(model, horizon)
        else:
            series = self._forecast_ml(model, horizon)

        df_out = pd.DataFrame({
            "Month": series.index.strftime("%Y-%m"),
            f"{self.target} (Forecast)": series.values.round(2),
        })
        return df_out

    # ─────────────────────────────────────────
    # Save Forecast
    # ─────────────────────────────────────────
    def save_forecast(self, df_forecast: pd.DataFrame, model_name: str, horizon: int) -> Path:
        """Saves a forecast DataFrame to outputs/forecasts/ as CSV."""
        safe_target = sanitize_filename(self.target)
        safe_model = sanitize_filename(model_name)
        path = FORECASTS_DIR / f"{safe_target}_{safe_model}_{horizon}m_forecast.csv"
        df_forecast.to_csv(path, index=False, encoding="utf-8")
        logger.info(f"  Saved forecast: {path.name}")
        return path

    # ─────────────────────────────────────────
    # Multi-Horizon Forecast (all horizons)
    # ─────────────────────────────────────────
    def forecast_all_horizons(
        self, model_name: str, horizons: list[int] = HORIZONS
    ) -> dict[int, pd.DataFrame]:
        """
        Generates and saves forecasts for all standard horizons (3, 6, 12 months).

        Returns:
            Dict mapping horizon -> forecast DataFrame
        """
        results = {}
        for h in horizons:
            df_fc = self.forecast(model_name, h)
            self.save_forecast(df_fc, model_name, h)
            results[h] = df_fc
        return results


# ─────────────────────────────────────────────
# Multi-Target Forecasting
# ─────────────────────────────────────────────
def multi_target_forecast(
    df: pd.DataFrame,
    targets: list[str],
    trained_models_map: dict[str, dict],
    best_model_map: dict[str, str],
    horizons: list[int] = HORIZONS,
) -> dict[str, dict[int, pd.DataFrame]]:
    """
    Runs forecast_all_horizons for each target using its best model.

    Args:
        df:                 Cleaned DataFrame
        targets:            List of target column names
        trained_models_map: {target: {model_name: model_object}}
        best_model_map:     {target: best_model_name}
        horizons:           List of forecast horizons

    Returns:
        Nested dict: {target: {horizon: forecast_df}}
    """
    all_forecasts = {}
    for target in targets:
        logger.info(f"\n── Forecasting: {target} ──")
        models = trained_models_map.get(target, {})
        best = best_model_map.get(target)

        if not models or not best:
            logger.warning(f"  No trained models found for '{target}' — skipping.")
            continue

        forecaster = Forecaster(df, target, models)
        forecasts = forecaster.forecast_all_horizons(best, horizons)
        all_forecasts[target] = forecasts

    return all_forecasts
