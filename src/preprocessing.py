"""
preprocessing.py — Data loading and preprocessing for Sri Lankan food price data.
Sri Lanka Food Price Forecasting System
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.utils import get_logger, DATA_FILE, SHEET_NAME, get_food_columns

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Handles loading, date parsing, missing value treatment, and basic
    validation of the Sri Lankan food price dataset.

    Dataset structure:
        - Column 'Month': monthly dates from 2013 onward
        - Food columns: monthly LKR prices for each food item
        - Column 'Index': composite economic food price index
    """

    def __init__(self, filepath: Path = DATA_FILE, sheet: str = SHEET_NAME):
        self.filepath = filepath
        self.sheet = sheet
        self.df_raw: pd.DataFrame | None = None
        self.df: pd.DataFrame | None = None

    # ─────────────────────────────────────────
    # Step 1: Load
    # ─────────────────────────────────────────
    def load_data(self) -> "DataPreprocessor":
        """
        Reads food_prices.xlsx from data/ or from a BytesIO upload.
        Validates that 'Month' column exists.
        """
        import io
        is_bytes = isinstance(self.filepath, (io.BytesIO, io.RawIOBase, io.BufferedIOBase))

        if not is_bytes and not Path(self.filepath).exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.filepath}. "
                "Please place food_prices.xlsx in the data/ directory."
            )

        logger.info(f"Loading dataset from: {getattr(self.filepath, 'name', str(self.filepath))}")

        if is_bytes:
            self.filepath.seek(0)
            self.df_raw = pd.read_excel(self.filepath)
        else:
            self.df_raw = pd.read_excel(self.filepath, sheet_name=self.sheet)

        logger.info(
            f"Loaded {len(self.df_raw)} rows x {len(self.df_raw.columns)} columns."
        )

        if "Month" not in self.df_raw.columns:
            raise ValueError(
                "Expected 'Month' column not found. "
                f"Found columns: {list(self.df_raw.columns)}"
            )

        return self

    # ─────────────────────────────────────────
    # Step 2: Date Parsing
    # ─────────────────────────────────────────
    def parse_dates(self) -> "DataPreprocessor":
        """
        Converts 'Month' to datetime, sets as DatetimeIndex,
        sorts chronologically, and resamples to month-start frequency.
        """
        df = self.df_raw.copy()
        df["Month"] = pd.to_datetime(df["Month"])
        df = df.sort_values("Month").reset_index(drop=True)
        df.set_index("Month", inplace=True)

        # Enforce monthly frequency (MS = Month Start)
        df = df.asfreq("MS")

        self.df = df
        logger.info(
            f"Date range: {df.index.min().date()} → {df.index.max().date()} "
            f"({len(df)} months)"
        )
        return self

    # ─────────────────────────────────────────
    # Step 3: Missing Value Handling
    # ─────────────────────────────────────────
    def handle_missing(self) -> "DataPreprocessor":
        """
        Fills missing values using:
          1. Forward fill (propagate last known value)
          2. Backward fill (fill leading NaN at start)
          3. Linear interpolation (for any remaining gaps)

        Reports before/after missing counts.
        """
        before = self.df.isnull().sum().sum()
        logger.info(f"Missing values before cleaning: {before}")

        self.df = self.df.ffill().bfill()
        remaining = self.df.isnull().sum().sum()

        if remaining > 0:
            logger.warning(
                f"{remaining} NaN values remain after ffill/bfill — applying interpolation."
            )
            self.df = self.df.interpolate(method="linear", limit_direction="both")

        after = self.df.isnull().sum().sum()
        logger.info(f"Missing values after cleaning: {after}")
        return self

    # ─────────────────────────────────────────
    # Step 4: Data Type Enforcement
    # ─────────────────────────────────────────
    def enforce_dtypes(self) -> "DataPreprocessor":
        """
        Ensures all food price columns are float64.
        Coerces any non-numeric values (e.g., stray text) to NaN.
        """
        food_cols = get_food_columns(self.df)
        for col in food_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Run missing handler again in case coercion introduced new NaNs
        remaining = self.df.isnull().sum().sum()
        if remaining > 0:
            self.df = self.df.ffill().bfill()

        return self

    # ─────────────────────────────────────────
    # Step 5: Summary
    # ─────────────────────────────────────────
    def summary(self) -> None:
        """Prints a quick statistical summary of the cleaned dataset."""
        sep = "-" * 60
        print(f"\n{sep}")
        print("  DATASET SUMMARY")
        print(sep)
        print(f"  Shape       : {self.df.shape}")
        print(f"  Date Range  : {self.df.index.min().date()} to {self.df.index.max().date()}")
        print(f"  Total Months: {len(self.df)}")
        food_cols = get_food_columns(self.df)
        print(f"  Food Items  : {len(food_cols)}")
        print(f"  Has Index   : {'Index' in self.df.columns}")
        print(f"  Missing Val : {self.df.isnull().sum().sum()}")
        print(f"{sep}\n")

    # ─────────────────────────────────────────
    # Main Entry Point
    # ─────────────────────────────────────────
    def preprocess(self, silent: bool = False) -> pd.DataFrame:
        """
        Runs the full preprocessing pipeline and returns the cleaned DataFrame.

        Args:
            silent: If True, suppresses the summary() print
                    (use in dashboard / GUI contexts).

        Returns:
            pd.DataFrame with DatetimeIndex (monthly), all columns as float64.
        """
        self.load_data()
        self.parse_dates()
        self.handle_missing()
        self.enforce_dtypes()
        if not silent:
            self.summary()

        logger.info("Preprocessing complete.")
        return self.df

    def get_targets(self) -> list[str]:
        """Returns all forecastable target columns (food items + Index)."""
        if self.df is None:
            raise RuntimeError("Call preprocess() first.")
        targets = get_food_columns(self.df)
        if "Index" in self.df.columns:
            targets = targets + ["Index"]
        return targets
