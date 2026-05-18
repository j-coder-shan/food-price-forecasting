"""
train.py — Model training for all supported forecasting models.
Sri Lanka Food Price Forecasting System

Models implemented:
  ML:          LinearRegression, RandomForest, XGBoost, LightGBM, CatBoost
  Statistical: ARIMA, SARIMA, Prophet
  (Deep Learning skipped for Python 3.13 — TensorFlow not yet supported)

All models are saved to models/saved_models/<target>/
"""

import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.utils import get_logger, model_save_dir, sanitize_filename
from src.feature_engineering import FeatureEngineer, SplitData

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

# ─────────────────────────────────────────────
# Optional Imports (graceful fallback)
# ─────────────────────────────────────────────
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    logger.warning("LightGBM not available — skipping LGBM model.")

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    logger.warning("CatBoost not available — skipping CatBoost model.")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logger.warning("statsmodels not available — skipping ARIMA/SARIMA.")

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    logger.warning("Prophet not available — skipping Prophet model.")


# ─────────────────────────────────────────────
# ModelTrainer
# ─────────────────────────────────────────────
class ModelTrainer:
    """
    Trains multiple forecasting models for a single food item target.

    Workflow:
        1. Uses FeatureEngineer to get 80/20 train/test splits.
        2. Trains all available ML models on X_train / y_train.
        3. Trains statistical models on raw train time series.
        4. Saves all trained models to disk.
        5. Returns fitted model objects for evaluation.

    Args:
        df:     Cleaned DataFrame with DatetimeIndex.
        target: Target food item column name.
    """

    def __init__(self, df: pd.DataFrame, target: str):
        self.df = df
        self.target = target
        self.fe = FeatureEngineer(df, target)
        self.split: SplitData | None = None
        self.trained_models: dict = {}
        self.save_dir: Path = model_save_dir(target)

    # ─────────────────────────────────────────
    # ML Models
    # ─────────────────────────────────────────
    def _train_linear_regression(self) -> LinearRegression:
        logger.info("  → Training Linear Regression...")
        model = LinearRegression()
        model.fit(self.split.X_train, self.split.y_train)
        return model

    def _train_random_forest(self) -> RandomForestRegressor:
        logger.info("  → Training Random Forest (n=200)...")
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(self.split.X_train, self.split.y_train)
        return model

    def _train_xgboost(self) -> XGBRegressor:
        logger.info("  → Training XGBoost...")
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        model.fit(
            self.split.X_train,
            self.split.y_train,
            eval_set=[(self.split.X_test, self.split.y_test)],
            verbose=False,
        )
        return model

    def _train_lightgbm(self):
        logger.info("  → Training LightGBM...")
        model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )
        model.fit(self.split.X_train, self.split.y_train)
        return model

    def _train_catboost(self):
        logger.info("  → Training CatBoost...")
        model = CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            silent=True,
        )
        model.fit(self.split.X_train, self.split.y_train)
        return model

    # ─────────────────────────────────────────
    # Statistical Models
    # ─────────────────────────────────────────
    def _train_arima(self):
        logger.info("  → Training ARIMA(1,1,1)...")
        train_series, _ = self.fe.get_raw_split()
        model = ARIMA(train_series, order=(1, 1, 1))
        fitted = model.fit()
        return fitted

    def _train_sarima(self):
        logger.info("  → Training SARIMA(1,1,1)(1,1,1,12)...")
        train_series, _ = self.fe.get_raw_split()
        model = SARIMAX(
            train_series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)
        return fitted

    def _train_prophet(self):
        logger.info("  → Training Prophet...")
        train_series, _ = self.fe.get_raw_split()
        prophet_df = pd.DataFrame({
            "ds": train_series.index,
            "y": train_series.values,
        })
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
        )
        model.fit(prophet_df)
        return model

    # ─────────────────────────────────────────
    # Save / Load
    # ─────────────────────────────────────────
    def _save_model(self, model, name: str) -> None:
        """Saves a model to the target's model directory."""
        path = self.save_dir / f"{sanitize_filename(name)}.pkl"
        joblib.dump(model, path)
        logger.info(f"  Saved: {path.name}")

    def load_model(self, name: str):
        """Loads a previously saved model from disk."""
        path = self.save_dir / f"{sanitize_filename(name)}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No saved model found at: {path}")
        return joblib.load(path)

    # ─────────────────────────────────────────
    # Main Training Orchestrator
    # ─────────────────────────────────────────
    def train_all(
        self,
        skip_statistical: bool = False,
        selected_models: list[str] | None = None,
    ) -> dict:
        """
        Trains all available (or selected) models for this target.

        Args:
            skip_statistical:  If True, skips ARIMA/SARIMA/Prophet.
            selected_models:   Optional list of model names to train.
                               If None, trains all available models.
                               Example: ['RandomForest', 'XGBoost']

        Returns:
            Dict of {model_name: fitted_model_object}
        """
        logger.info(f"\n{'='*55}")
        logger.info(f"  Training models for: '{self.target}'")
        logger.info(f"{'='*55}")

        # Build feature split (80/20 chronological)
        self.split = self.fe.get_split()
        models = {}

        def _should_train(name: str) -> bool:
            """Returns True if this model is in the user-selected list."""
            if selected_models is None:
                return True
            return name in selected_models

        # ── ML Models ──────────────────────────
        try:
            if _should_train("LinearRegression"):
                m = self._train_linear_regression()
                models["LinearRegression"] = m
                self._save_model(m, "LinearRegression")
        except Exception as e:
            logger.error(f"LinearRegression failed: {e}")

        try:
            if _should_train("RandomForest"):
                m = self._train_random_forest()
                models["RandomForest"] = m
                self._save_model(m, "RandomForest")
        except Exception as e:
            logger.error(f"RandomForest failed: {e}")

        try:
            if _should_train("XGBoost"):
                m = self._train_xgboost()
                models["XGBoost"] = m
                self._save_model(m, "XGBoost")
        except Exception as e:
            logger.error(f"XGBoost failed: {e}")

        if HAS_LGBM and _should_train("LightGBM"):
            try:
                m = self._train_lightgbm()
                models["LightGBM"] = m
                self._save_model(m, "LightGBM")
            except Exception as e:
                logger.error(f"LightGBM failed: {e}")

        if HAS_CATBOOST and _should_train("CatBoost"):
            try:
                m = self._train_catboost()
                models["CatBoost"] = m
                self._save_model(m, "CatBoost")
            except Exception as e:
                logger.error(f"CatBoost failed: {e}")

        # ── Statistical Models ──────────────────
        if not skip_statistical and HAS_STATSMODELS:
            if _should_train("ARIMA"):
                try:
                    m = self._train_arima()
                    models["ARIMA"] = m
                    self._save_model(m, "ARIMA")
                except Exception as e:
                    logger.error(f"ARIMA failed: {e}")

            if _should_train("SARIMA"):
                try:
                    m = self._train_sarima()
                    models["SARIMA"] = m
                    self._save_model(m, "SARIMA")
                except Exception as e:
                    logger.error(f"SARIMA failed: {e}")

        if not skip_statistical and HAS_PROPHET and _should_train("Prophet"):
            try:
                m = self._train_prophet()
                models["Prophet"] = m
                self._save_model(m, "Prophet")
            except Exception as e:
                logger.error(f"Prophet failed: {e}")

        self.trained_models = models
        logger.info(
            f"  Done. Trained {len(models)} models: {list(models.keys())}"
        )
        return models
