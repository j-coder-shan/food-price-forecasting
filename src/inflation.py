"""
inflation.py  —  Inflation calculation from Food Price Index.
Sri Lanka AI Food Price & Inflation Forecasting Platform

Formulas:
  Monthly Inflation = (Index_t - Index_{t-1}) / Index_{t-1} × 100
  YoY Inflation     = (Index_t - Index_{t-12}) / Index_{t-12} × 100
"""

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

INDEX_COL = "Index"


class InflationCalculator:
    """
    Calculates monthly and year-over-year food price inflation
    from either the historical or forecasted food price index.

    Args:
        index_series: pd.Series with DatetimeIndex and food index values.
                      The index should use monthly frequency (MS).
    """

    def __init__(self, index_series: pd.Series):
        if not isinstance(index_series.index, pd.DatetimeIndex):
            raise ValueError("index_series must have a DatetimeIndex.")
        self.series = index_series.dropna().sort_index()

    # ─────────────────────────────────────────
    # Monthly Inflation
    # ─────────────────────────────────────────
    def monthly_inflation(self) -> pd.Series:
        """
        Month-over-month inflation rate (%).

        Formula:  (Index_t - Index_{t-1}) / Index_{t-1} × 100

        Returns:
            pd.Series named 'Monthly_Inflation_%' with same DatetimeIndex.
            First value is NaN (no previous month for comparison).
        """
        pct = self.series.pct_change(periods=1) * 100
        pct.name = "Monthly_Inflation_%"
        return pct

    # ─────────────────────────────────────────
    # Year-over-Year Inflation
    # ─────────────────────────────────────────
    def yoy_inflation(self) -> pd.Series:
        """
        Year-over-year inflation rate (%).

        Formula:  (Index_t - Index_{t-12}) / Index_{t-12} × 100

        Returns:
            pd.Series named 'YoY_Inflation_%'. First 12 values are NaN.
        """
        pct = self.series.pct_change(periods=12) * 100
        pct.name = "YoY_Inflation_%"
        return pct

    # ─────────────────────────────────────────
    # Full Inflation Table
    # ─────────────────────────────────────────
    def inflation_table(self) -> pd.DataFrame:
        """
        Returns a complete DataFrame with index values, monthly, and YoY inflation.

        Columns:
            Index              — food price index value
            Monthly_Inflation_%— month-over-month change (%)
            YoY_Inflation_%    — year-over-year change (%)
        """
        df = pd.DataFrame({
            INDEX_COL:            self.series,
            "Monthly_Inflation_%": self.monthly_inflation(),
            "YoY_Inflation_%":    self.yoy_inflation(),
        })
        df = df.round(4)
        logger.info(
            f"Inflation table: {len(df)} rows. "
            f"Avg monthly inflation: {df['Monthly_Inflation_%'].mean():.2f}%"
        )
        return df

    # ─────────────────────────────────────────
    # Forecasted Inflation
    # ─────────────────────────────────────────
    @staticmethod
    def forecast_inflation(
        forecasted_index: pd.Series,
        last_known_index: float,
        last_known_series: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Computes inflation for a forecasted index series.

        Monthly inflation: uses last_known_index as the "previous" value
        for the first forecasted month, then chains from there.

        YoY inflation: uses last_known_series (12 historical values) to
        compute YoY for the first 12 forecast months.

        Args:
            forecasted_index:   pd.Series of forecasted index values (DatetimeIndex).
            last_known_index:   Last actual index value (immediately before forecast).
            last_known_series:  Optional — the 12 most recent historical index values
                                for YoY computation.

        Returns:
            DataFrame with Month, Forecasted_Index, Monthly_Inflation_%, YoY_Inflation_%
        """
        # Prepend the last known value to enable pct_change on first step
        anchor = pd.Series(
            [last_known_index],
            index=pd.DatetimeIndex([forecasted_index.index[0] - pd.DateOffset(months=1)]),
            name=forecasted_index.name,
        )
        full = pd.concat([anchor, forecasted_index])
        monthly = full.pct_change(periods=1) * 100
        monthly = monthly.iloc[1:]  # drop anchor row

        # YoY: stitch historical + forecast
        yoy = pd.Series(np.nan, index=forecasted_index.index, name="YoY_Inflation_%")
        if last_known_series is not None and len(last_known_series) >= 12:
            hist_12 = last_known_series.iloc[-12:]
            for i, date in enumerate(forecasted_index.index):
                if i < 12:
                    hist_val = hist_12.iloc[i] if i < len(hist_12) else np.nan
                else:
                    # Use already-computed forecast value 12 steps back
                    hist_val = forecasted_index.iloc[i - 12]
                if hist_val and hist_val != 0:
                    yoy.iloc[i] = (forecasted_index.iloc[i] - hist_val) / hist_val * 100

        df = pd.DataFrame({
            "Month":                forecasted_index.index.strftime("%Y-%m"),
            "Forecasted_Index":     forecasted_index.values.round(4),
            "Monthly_Inflation_%":  monthly.values.round(4),
            "YoY_Inflation_%":      yoy.values.round(4),
        })
        return df

    # ─────────────────────────────────────────
    # Summary Statistics
    # ─────────────────────────────────────────
    def summary(self) -> dict:
        """Returns key inflation statistics."""
        monthly = self.monthly_inflation().dropna()
        yoy = self.yoy_inflation().dropna()
        return {
            "avg_monthly_inflation":  round(monthly.mean(), 3),
            "max_monthly_inflation":  round(monthly.max(), 3),
            "min_monthly_inflation":  round(monthly.min(), 3),
            "avg_yoy_inflation":      round(yoy.mean(), 3),
            "max_yoy_inflation":      round(yoy.max(), 3),
            "latest_monthly":         round(monthly.iloc[-1], 3) if len(monthly) else None,
            "latest_yoy":             round(yoy.iloc[-1], 3) if len(yoy) else None,
        }
