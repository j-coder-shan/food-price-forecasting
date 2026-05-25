"""
feature_engineering.py — Time-series feature creation and train/test splitting.
Sri Lanka Food Price Forecasting System

Split strategy:
  - 80% → TRAIN  (model fitting)
  - 20% → TEST   (held-out evaluation + "predicted" output)
  - Full dataset → retrain for future forecasting
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.utils import get_logger, TRAIN_RATIO, LAG_PERIODS, ROLLING_WINDOWS, SEQUENCE_LENGTH

logger = get_logger(__name__)


@dataclass
class SplitData:
    """Container for train/test split results for ML models."""
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_names: list[str]
    train_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex


@dataclass
class SequenceData:
    """Container for LSTM/GRU sequence data."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: object
    train_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex


class FeatureEngineer:
    """
    Creates time-series features for a given target food item column.

    Features created:
      - Lag features (lag_1, lag_3, lag_6, lag_12)
      - Rolling statistics (mean, std, max, min at 3/6/12 month windows)
      - Date features (month, quarter, year, cyclic sin/cos encoding)
      - Momentum features (pct_change at 1 and 3 months)

    Split: 80% train / 20% test (preserving temporal order — NO shuffling).
    """

    def __init__(self, df: pd.DataFrame, target: str):
        """
        Args:
            df: Cleaned DataFrame with DatetimeIndex (output of DataPreprocessor)
            target: Column name of the food item to forecast
        """
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not found in DataFrame columns.")

        self.df = df.copy()
        self.target = target
        logger.info(f"FeatureEngineer initialized for target: '{target}'")

    # ─────────────────────────────────────────
    # Feature Creation
    # ─────────────────────────────────────────
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates lag features for the target column."""
        for lag in LAG_PERIODS:
            df[f"lag_{lag}"] = df[self.target].shift(lag)
        return df

    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates rolling window statistics for the target column."""
        for window in ROLLING_WINDOWS:
            roll = df[self.target].rolling(window=window, min_periods=1)
            df[f"rolling_mean_{window}"] = roll.mean()
            df[f"rolling_std_{window}"] = roll.std().fillna(0)
            df[f"rolling_max_{window}"] = roll.max()
            df[f"rolling_min_{window}"] = roll.min()
        return df

    def _create_ewma_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates Exponentially Weighted Moving Averages (EWMA)."""
        for span in ROLLING_WINDOWS:
            df[f"ewma_{span}"] = df[self.target].ewm(span=span, adjust=False).mean()
        return df

    def _create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates rolling volatility of percentage changes."""
        pct_change = df[self.target].pct_change(1).fillna(0)
        for window in ROLLING_WINDOWS:
            df[f"volatility_{window}"] = pct_change.rolling(window=window, min_periods=1).std().fillna(0)
        return df

    def _create_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates temporal date features from the DatetimeIndex."""
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        df["year"] = df.index.year

        # Cyclic encoding for month (captures Jan≈Dec proximity)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        return df

    def _create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates price momentum (percentage change) features."""
        df["pct_change_1"] = df[self.target].pct_change(1).fillna(0)
        df["pct_change_3"] = df[self.target].pct_change(3).fillna(0)
        return df

    def _create_economic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates features from the merged Brent oil price and USD/LKR exchange rate.
        Features are only added when the source columns exist in self.df
        (i.e., when the merged dataset is used). Skipped silently otherwise.

        Fuel (Brent_USD) features:
          fuel_lag_1, fuel_lag_3         — oil price 1/3 months ago
          fuel_rolling_mean_3            — 3-month oil price trend
          fuel_pct_change                — oil price momentum

        Exchange (USD_LKR) features:
          usd_lag_1                      — exchange rate last month
          usd_rolling_mean_3             — 3-month FX trend
          usd_pct_change                 — currency momentum
        """
        from src.utils import FUEL_COL, EXCHANGE_COL

        # Temporarily bring in economic columns from original df
        if FUEL_COL in self.df.columns:
            fuel = self.df[FUEL_COL]
            df["fuel_lag_1"]          = fuel.shift(1)
            df["fuel_lag_3"]          = fuel.shift(3)
            df["fuel_rolling_mean_3"] = fuel.rolling(3, min_periods=1).mean()
            df["fuel_pct_change"]     = fuel.pct_change(1).fillna(0)

        if EXCHANGE_COL in self.df.columns:
            fx = self.df[EXCHANGE_COL]
            df["usd_lag_1"]           = fx.shift(1)
            df["usd_rolling_mean_3"]  = fx.rolling(3, min_periods=1).mean()
            df["usd_pct_change"]      = fx.pct_change(1).fillna(0)

        # Interaction Terms (if both are present)
        if FUEL_COL in self.df.columns and EXCHANGE_COL in self.df.columns:
            # Local cost of fuel proxy
            local_fuel = self.df[FUEL_COL] * self.df[EXCHANGE_COL]
            df["local_fuel_lag_1"] = local_fuel.shift(1)
            df["local_fuel_pct_change"] = local_fuel.pct_change(1).fillna(0)

        return df

    def build_features(self) -> pd.DataFrame:
        """
        Applies all feature engineering steps and returns the full
        feature-engineered DataFrame (still contains the target column).
        """
        df = self.df[[self.target]].copy()

        df = self._create_lag_features(df)
        df = self._create_rolling_features(df)
        df = self._create_ewma_features(df)
        df = self._create_volatility_features(df)
        df = self._create_date_features(df)
        df = self._create_momentum_features(df)
        df = self._create_economic_features(df)   # fuel + FX (if available)

        # Drop rows with NaN from lagging (lag_12 creates 12 NaN rows at start)
        df = df.dropna()

        logger.info(
            f"Feature matrix for '{self.target}': "
            f"{len(df)} rows x {len(df.columns) - 1} features "
            f"(after dropping {len(self.df) - len(df)} NaN rows from lags)"
        )
        return df

    # ─────────────────────────────────────────
    # Train / Test Split (ML Models)
    # ─────────────────────────────────────────
    def get_split(self) -> SplitData:
        """
        Builds features and performs an 80/20 chronological train/test split.

        Returns:
            SplitData with X_train, y_train, X_test, y_test, feature_names.
        """
        df = self.build_features()

        # Define feature columns (everything except the target)
        feature_cols = [c for c in df.columns if c != self.target]

        X = df[feature_cols]
        y = df[self.target]

        split_idx = int(len(df) * TRAIN_RATIO)

        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]

        logger.info(
            f"  Train: {len(X_train)} rows ({X_train.index.min().date()} → {X_train.index.max().date()})"
        )
        logger.info(
            f"  Test : {len(X_test)} rows ({X_test.index.min().date()} → {X_test.index.max().date()})"
        )

        return SplitData(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_cols,
            train_index=X_train.index,
            test_index=X_test.index,
        )

    # ─────────────────────────────────────────
    # Sequence Data (Statistical / DL Models)
    # ─────────────────────────────────────────
    def get_raw_split(self) -> tuple[pd.Series, pd.Series]:
        """
        Returns the raw target time series split into train/test portions.
        Used by ARIMA, SARIMA, Prophet, and sequence-based models.

        Returns:
            (train_series, test_series)
        """
        series = self.df[self.target].dropna()
        split_idx = int(len(series) * TRAIN_RATIO)
        train = series.iloc[:split_idx]
        test = series.iloc[split_idx:]
        logger.info(
            f"  Raw split — Train: {len(train)}, Test: {len(test)} observations."
        )
        return train, test

    def get_sequence_data(
        self, seq_length: int = SEQUENCE_LENGTH
    ) -> SequenceData:
        """
        Generates sliding window sequences for deep learning models.
        Applies MinMaxScaler normalization and inverse-scaling info.

        Args:
            seq_length: Number of past time steps per input window.

        Returns:
            SequenceData with numpy arrays and the fitted scaler.
        """
        from sklearn.preprocessing import MinMaxScaler

        series = self.df[self.target].dropna().values.reshape(-1, 1)
        split_idx = int(len(series) * TRAIN_RATIO)

        train_raw = series[:split_idx]
        test_raw = series[split_idx:]

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_raw)
        # Scale test using train's scaler (no data leakage)
        full_scaled = scaler.transform(series)

        def make_sequences(data: np.ndarray, seq_len: int):
            X, y = [], []
            for i in range(seq_len, len(data)):
                X.append(data[i - seq_len : i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)

        # Train sequences from scaled train portion
        X_train, y_train = make_sequences(full_scaled[:split_idx], seq_length)
        # Test sequences — include look-back from train for continuity
        X_test, y_test = make_sequences(full_scaled[split_idx - seq_length :], seq_length)

        # Reshape for [samples, time_steps, features]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        idx = self.df[self.target].dropna().index
        return SequenceData(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            scaler=scaler,
            train_index=idx[:split_idx][seq_length:],
            test_index=idx[split_idx:][: len(y_test)],
        )


# ─────────────────────────────────────────────
# Convenience Function for Multi-target Use
# ─────────────────────────────────────────────
def prepare_target(df: pd.DataFrame, target: str) -> SplitData:
    """
    Shorthand to build features and get a train/test split for a single target.

    Args:
        df: Cleaned DataFrame
        target: Target column name

    Returns:
        SplitData object
    """
    fe = FeatureEngineer(df, target)
    return fe.get_split()
