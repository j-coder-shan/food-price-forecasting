"""
merge_datasets.py  —  Loads and merges all three economic datasets.
Sri Lanka AI Food Price & Inflation Forecasting Platform

Datasets:
  food_prices.xlsx       : monthly food prices + Index (Jan 2013 →)
  fuel_prices.xls        : Europe Brent spot price USD/barrel (monthly, 1987 →)
  exchange_rates.xlsx    : USD/LKR exchange rate (monthly, 1986 →)

Merge key: Month (DatetimeIndex, monthly frequency MS)
"""

import io
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import get_logger, DATA_DIR

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

# ─────────────────────────────────────────────
# Default file paths
# ─────────────────────────────────────────────
FOOD_FILE     = DATA_DIR / "food_prices.xlsx"
FUEL_FILE     = DATA_DIR / "fuel_prices.xls"
EXCHANGE_FILE = DATA_DIR / "exchange_rates.xlsx"


class DatasetMerger:
    """
    Loads, cleans, and merges the three Sri Lankan economic datasets
    into a single unified DataFrame aligned on a monthly DatetimeIndex.

    Usage:
        merger = DatasetMerger()
        merged_df = merger.merge_all(food_df)   # food_df already has DatetimeIndex
        -- or --
        merged_df = merger.load_and_merge_all()  # loads food data internally too
    """

    def __init__(
        self,
        food_file:     Path | io.BytesIO | None = None,
        fuel_file:     Path | io.BytesIO | None = None,
        exchange_file: Path | io.BytesIO | None = None,
    ):
        self.food_file     = food_file     or FOOD_FILE
        self.fuel_file     = fuel_file     or FUEL_FILE
        self.exchange_file = exchange_file or EXCHANGE_FILE

        self.food_df:     pd.DataFrame | None = None
        self.fuel_df:     pd.DataFrame | None = None
        self.exchange_df: pd.DataFrame | None = None
        self.merged_df:   pd.DataFrame | None = None

    # ─────────────────────────────────────────
    # Fuel Price Loader
    # ─────────────────────────────────────────
    def load_fuel(self) -> pd.DataFrame:
        """
        Reads fuel_prices.xls (Brent crude oil USD/barrel).
        Supports both Path and BytesIO (from Streamlit file_uploader).
        """
        import io as _io
        import tempfile, os

        logger.info("Loading fuel prices (Brent crude)...")

        fuel_src = self.fuel_file
        is_bytes = isinstance(fuel_src, (_io.BytesIO, _io.RawIOBase, _io.BufferedIOBase))

        if is_bytes:
            # xlrd requires a real file path for .xls; write to temp
            fuel_src.seek(0)
            suffix = ".xls"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(fuel_src.read())
            tmp.close()
            read_path = tmp.name
            cleanup   = True
        else:
            read_path = str(fuel_src)
            cleanup   = False

        try:
            df = pd.read_excel(
                read_path, sheet_name="Data 1",
                skiprows=2, header=0, engine="xlrd",
            )
        finally:
            if cleanup:
                os.unlink(read_path)

        df.columns = ["Date", "Brent_USD"]
        df = df.dropna(subset=["Date"])
        df["Date"]      = pd.to_datetime(df["Date"], errors="coerce")
        df              = df.dropna(subset=["Date"]).set_index("Date")
        df["Brent_USD"] = pd.to_numeric(df["Brent_USD"], errors="coerce")
        df              = df.resample("MS").mean()
        df.index.name   = "Month"

        self.fuel_df = df
        logger.info(
            f"  Fuel prices: {len(df)} monthly rows "
            f"({df.index.min().date()} to {df.index.max().date()})"
        )
        return df

    # ─────────────────────────────────────────
    # Exchange Rate Loader
    # ─────────────────────────────────────────
    def load_exchange(self) -> pd.DataFrame:
        """
        Reads exchange_rates.xlsx (USD/LKR monthly).

        Structure:
          Year  : filled for Jan only, NaN for subsequent months → forward-fill
          Month : month name ('January', 'February', ...)
          USD   : LKR per 1 USD

        Steps:
          1. Forward-fill Year column.
          2. Build datetime from Year + Month name.
          3. Set as DatetimeIndex (MS frequency).
          4. Rename to USD_LKR.
        """
        logger.info("Loading exchange rates (USD/LKR)...")
        df = pd.read_excel(self.exchange_file, engine="openpyxl")
        df.columns = ["Year", "Month_Name", "USD_LKR"]

        # Drop rows where Month_Name is not a real month (e.g. 'Annual Averages')
        valid_months = [
            "January","February","March","April","May","June",
            "July","August","September","October","November","December",
        ]
        df = df[df["Month_Name"].isin(valid_months)].copy()

        # Forward-fill year (only first row of each year has the year value)
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df["Year"] = df["Year"].ffill().astype(int)

        # Build datetime string → parse
        df["Month"] = pd.to_datetime(
            df["Year"].astype(str) + " " + df["Month_Name"].astype(str),
            format="%Y %B",
            errors="coerce",
        )
        df = df.dropna(subset=["Month"]).set_index("Month")
        df = df[["USD_LKR"]]
        df["USD_LKR"] = pd.to_numeric(df["USD_LKR"], errors="coerce")
        df = df.resample("MS").mean()

        self.exchange_df = df
        logger.info(
            f"  Exchange rates: {len(df)} monthly rows "
            f"({df.index.min().date()} → {df.index.max().date()})"
        )
        return df

    # ─────────────────────────────────────────
    # Merge
    # ─────────────────────────────────────────
    def merge_all(self, food_df: pd.DataFrame) -> pd.DataFrame:
        """
        Left-joins fuel and exchange rate data onto the food price DataFrame.

        Only rows present in food_df are kept (left join), so the result
        always spans exactly the food price date range.
        Missing values from fuel/exchange are forward-filled then backward-filled.

        Args:
            food_df: Cleaned food price DataFrame with DatetimeIndex (MS).

        Returns:
            Merged DataFrame with all food columns + Brent_USD + USD_LKR.
        """
        if self.fuel_df is None:
            self.load_fuel()
        if self.exchange_df is None:
            self.load_exchange()

        merged = food_df.copy()
        merged = merged.join(self.fuel_df[["Brent_USD"]], how="left")
        merged = merged.join(self.exchange_df[["USD_LKR"]], how="left")

        # Fill gaps from fuel/exchange (they may not cover all food months)
        for col in ["Brent_USD", "USD_LKR"]:
            missing = merged[col].isnull().sum()
            if missing > 0:
                logger.warning(f"  {col}: {missing} missing months — interpolating.")
                merged[col] = merged[col].interpolate(method="time").ffill().bfill()

        logger.info(
            f"  Merged dataset: {merged.shape[0]} rows × {merged.shape[1]} cols. "
            f"Fuel NaN: {merged['Brent_USD'].isnull().sum()}, "
            f"FX NaN: {merged['USD_LKR'].isnull().sum()}"
        )
        self.merged_df = merged
        return merged

    # ─────────────────────────────────────────
    # Convenience: load food internally too
    # ─────────────────────────────────────────
    def load_and_merge_all(self) -> pd.DataFrame:
        """
        Full pipeline: loads food, fuel, exchange, and merges all.
        Uses DataPreprocessor to clean food data.
        """
        from src.preprocessing import DataPreprocessor
        pp = DataPreprocessor(filepath=self.food_file)
        food_df = pp.preprocess(silent=True)
        self.food_df = food_df
        return self.merge_all(food_df)

    # ─────────────────────────────────────────
    # Dataset Info (for dashboard display)
    # ─────────────────────────────────────────
    def get_dataset_info(self) -> dict:
        """Returns summary info for all three loaded datasets."""
        info = {}
        if self.food_df is not None:
            info["food"] = {
                "rows": len(self.food_df),
                "cols": len(self.food_df.columns),
                "start": self.food_df.index.min().strftime("%Y-%m"),
                "end":   self.food_df.index.max().strftime("%Y-%m"),
            }
        if self.fuel_df is not None:
            info["fuel"] = {
                "rows": len(self.fuel_df),
                "start": self.fuel_df.index.min().strftime("%Y-%m"),
                "end":   self.fuel_df.index.max().strftime("%Y-%m"),
                "latest_price": round(self.fuel_df["Brent_USD"].iloc[-1], 2),
            }
        if self.exchange_df is not None:
            info["exchange"] = {
                "rows": len(self.exchange_df),
                "start": self.exchange_df.index.min().strftime("%Y-%m"),
                "end":   self.exchange_df.index.max().strftime("%Y-%m"),
                "latest_rate": round(self.exchange_df["USD_LKR"].iloc[-1], 2),
            }
        return info
