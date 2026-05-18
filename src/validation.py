"""src/validation.py — Dataset validation for uploads and auto-loaded files."""
from dataclasses import dataclass, field
import pandas as pd
import streamlit as st


@dataclass
class ValidationResult:
    is_valid: bool = True
    errors:   list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info:     list[str] = field(default_factory=list)

    def add_error(self, msg: str):
        self.errors.append(msg); self.is_valid = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def add_info(self, msg: str):
        self.info.append(msg)


class DatasetValidator:

    @staticmethod
    def validate_food_prices(df: pd.DataFrame) -> ValidationResult:
        r = ValidationResult()
        if "Month" not in df.columns:
            r.add_error("Missing required column: 'Month'")
            return r
        if "Index" not in df.columns:
            r.add_warning("'Index' column not found — inflation analysis will be unavailable")
        food_cols = [c for c in df.columns if c not in ("Month", "Index")]
        if len(food_cols) == 0:
            r.add_error("No food item columns found after 'Month'")
            return r
        if len(food_cols) < 3:
            r.add_warning(f"Only {len(food_cols)} food columns found — expected more")
        try:
            dates = pd.to_datetime(df["Month"])
            dups = dates.duplicated().sum()
            if dups > 0:
                r.add_warning(f"{dups} duplicate month(s) detected — will be removed")
            non_monthly = (dates.sort_values().diff().dropna() > pd.Timedelta("35 days")).sum()
            if non_monthly > 0:
                r.add_warning(f"{non_monthly} gap(s) in monthly sequence detected")
        except Exception as e:
            r.add_error(f"Date parsing failed: {e}")
        nan_cols = df[food_cols].isnull().any()
        n_nan_cols = nan_cols.sum()
        if n_nan_cols > 0:
            r.add_info(f"{n_nan_cols} columns have missing values — will be filled")
        r.add_info(f"{len(food_cols)} food items, {len(df)} months loaded")
        return r

    @staticmethod
    def validate_fuel(df: pd.DataFrame) -> ValidationResult:
        r = ValidationResult()
        if df is None or len(df) == 0:
            r.add_error("Fuel price file is empty")
            return r
        if "Brent_USD" not in df.columns:
            r.add_error("Expected 'Brent_USD' column after processing")
        else:
            if df["Brent_USD"].isnull().mean() > 0.2:
                r.add_warning("More than 20% of Brent price values are missing")
            r.add_info(f"{len(df)} monthly fuel price records loaded")
        return r

    @staticmethod
    def validate_exchange(df: pd.DataFrame) -> ValidationResult:
        r = ValidationResult()
        if df is None or len(df) == 0:
            r.add_error("Exchange rate file is empty")
            return r
        if "USD_LKR" not in df.columns:
            r.add_error("Expected 'USD_LKR' column after processing")
        else:
            if df["USD_LKR"].isnull().mean() > 0.2:
                r.add_warning("More than 20% of exchange rate values are missing")
            r.add_info(f"{len(df)} monthly exchange rate records loaded")
        return r

    @staticmethod
    def check_date_continuity(df: pd.DataFrame) -> list[str]:
        """Returns list of missing month strings."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return []
        full = pd.date_range(df.index.min(), df.index.max(), freq="MS")
        missing = full.difference(df.index)
        return [d.strftime("%Y-%m") for d in missing]


def show_validation(result: ValidationResult, label: str = "Dataset"):
    """Renders a ValidationResult in Streamlit with appropriate styling."""
    for err in result.errors:
        st.error(f"**{label}**: {err}")
    for warn in result.warnings:
        st.warning(f"**{label}**: {warn}")
    for info in result.info:
        st.caption(f"ℹ️ {label}: {info}")
