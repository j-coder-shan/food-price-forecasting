"""
evaluate.py — Model evaluation and metric computation.
Sri Lanka Food Price Forecasting System

Metrics:
  MAE   — Mean Absolute Error
  RMSE  — Root Mean Squared Error
  MAPE  — Mean Absolute Percentage Error
  R²    — Coefficient of Determination

The 20% held-out TEST set is used for all evaluations.
Results are saved as CSV to outputs/metrics/
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils import get_logger, METRICS_DIR, sanitize_filename
from src.feature_engineering import FeatureEngineer

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Core Metric Functions
# ─────────────────────────────────────────────
def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (returns %, handles zero denominators)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes MAE, RMSE, MAPE, and R² for a set of predictions.

    Args:
        y_true: Ground truth values.
        y_pred: Model predicted values.

    Returns:
        Dict with keys: MAE, RMSE, MAPE, R2
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "MAPE": round(compute_mape(y_true, y_pred), 4),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


# ─────────────────────────────────────────────
# ModelEvaluator
# ─────────────────────────────────────────────
class ModelEvaluator:
    """
    Evaluates trained models on the held-out 20% test split.

    For ML models: generates predictions using X_test feature matrix.
    For ARIMA/SARIMA: uses in-sample forecast with dynamic prediction.
    For Prophet: uses future DataFrame construction.

    Args:
        df:             Cleaned DataFrame
        target:         Target food item column name
        trained_models: Dict returned by ModelTrainer.train_all()
    """

    def __init__(self, df: pd.DataFrame, target: str, trained_models: dict):
        self.df = df
        self.target = target
        self.fe = FeatureEngineer(df, target)
        self.split = self.fe.get_split()
        self.trained_models = trained_models
        self.results: dict[str, dict] = {}
        self.predictions: dict[str, pd.Series] = {}

    # ─────────────────────────────────────────
    # Per-Model Prediction
    # ─────────────────────────────────────────
    def _predict_ml(self, model, model_name: str) -> pd.Series:
        """Generates predictions for sklearn-compatible models on X_test."""
        preds = model.predict(self.split.X_test)
        return pd.Series(preds, index=self.split.test_index, name=model_name)

    def _predict_arima_sarima(self, model, model_name: str) -> pd.Series:
        """
        Generates out-of-sample predictions for ARIMA/SARIMA.
        Forecasts exactly as many steps as the test set length.
        """
        n_steps = len(self.split.y_test)
        forecast = model.forecast(steps=n_steps)
        return pd.Series(
            forecast.values, index=self.split.test_index, name=model_name
        )

    def _predict_prophet(self, model, model_name: str) -> pd.Series:
        """Generates predictions from Prophet on the test date range."""
        future = pd.DataFrame({"ds": self.split.test_index})
        forecast = model.predict(future)
        preds = forecast.set_index("ds")["yhat"]
        preds.index = pd.DatetimeIndex(preds.index)
        return pd.Series(preds.values, index=self.split.test_index, name=model_name)

    # ─────────────────────────────────────────
    # Evaluation Loop
    # ─────────────────────────────────────────
    def evaluate_all(self) -> dict[str, dict]:
        """
        Evaluates all trained models and computes metrics on the test set.

        Returns:
            Nested dict: {model_name: {MAE, RMSE, MAPE, R2}}
        """
        y_true = self.split.y_test.values
        arima_sarima_names = {"ARIMA", "SARIMA"}
        prophet_names = {"Prophet"}

        for name, model in self.trained_models.items():
            try:
                if name in arima_sarima_names:
                    preds = self._predict_arima_sarima(model, name)
                elif name in prophet_names:
                    preds = self._predict_prophet(model, name)
                else:
                    preds = self._predict_ml(model, name)

                metrics = compute_metrics(y_true, preds.values)
                self.results[name] = metrics
                self.predictions[name] = preds

                logger.info(
                    f"  {name:<20} MAE={metrics['MAE']:.2f}  "
                    f"RMSE={metrics['RMSE']:.2f}  "
                    f"MAPE={metrics['MAPE']:.2f}%  "
                    f"R²={metrics['R2']:.4f}"
                )
            except Exception as e:
                logger.error(f"  Evaluation failed for {name}: {e}")

        return self.results

    # ─────────────────────────────────────────
    # Best Model Selection
    # ─────────────────────────────────────────
    def best_model(self) -> tuple[str, dict]:
        """
        Returns the model with the lowest RMSE.

        Returns:
            (model_name, metrics_dict)
        """
        if not self.results:
            raise RuntimeError("Call evaluate_all() first.")
        best = min(self.results.items(), key=lambda x: x[1]["RMSE"])
        logger.info(f"  Best model for '{self.target}': {best[0]} (RMSE={best[1]['RMSE']:.2f})")
        return best

    # ─────────────────────────────────────────
    # Save Results
    # ─────────────────────────────────────────
    def save_metrics(self) -> Path:
        """
        Saves evaluation metrics as a CSV to outputs/metrics/.

        Returns:
            Path to the saved CSV file.
        """
        if not self.results:
            raise RuntimeError("Call evaluate_all() first.")

        rows = []
        for model_name, metrics in self.results.items():
            row = {"Model": model_name, **metrics}
            rows.append(row)

        df_metrics = pd.DataFrame(rows).sort_values("RMSE")
        safe = sanitize_filename(self.target)
        path = METRICS_DIR / f"{safe}_metrics.csv"
        df_metrics.to_csv(path, index=False, encoding="utf-8")
        logger.info(f"  Metrics saved: {path}")
        return path

    def get_predictions_df(self) -> pd.DataFrame:
        """
        Returns a DataFrame with actual vs predicted values for all models
        on the test set.
        """
        df = pd.DataFrame({"Actual": self.split.y_test})
        for name, preds in self.predictions.items():
            df[name] = preds.values
        return df

    def get_metrics_df(self) -> pd.DataFrame:
        """Returns evaluation metrics as a formatted DataFrame."""
        rows = [{"Model": k, **v} for k, v in self.results.items()]
        return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
