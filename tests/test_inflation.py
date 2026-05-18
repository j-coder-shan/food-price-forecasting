"""tests/test_inflation.py — Unit tests for InflationCalculator."""
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inflation import InflationCalculator


@pytest.fixture
def index_series():
    dates = pd.date_range("2013-01-01", periods=36, freq="MS")
    vals  = [100.0 + i * 0.5 + np.sin(i / 3) * 2 for i in range(36)]
    return pd.Series(vals, index=dates, name="Index")


def test_monthly_inflation_length(index_series):
    calc = InflationCalculator(index_series)
    mom  = calc.monthly_inflation()
    assert len(mom) == len(index_series)


def test_monthly_inflation_first_is_nan(index_series):
    calc = InflationCalculator(index_series)
    mom  = calc.monthly_inflation()
    assert np.isnan(mom.iloc[0])


def test_yoy_first_12_nan(index_series):
    calc = InflationCalculator(index_series)
    yoy  = calc.yoy_inflation()
    assert all(np.isnan(yoy.iloc[:12]))


def test_inflation_table_columns(index_series):
    calc = InflationCalculator(index_series)
    tbl  = calc.inflation_table()
    assert "Monthly_Inflation_%" in tbl.columns
    assert "YoY_Inflation_%"     in tbl.columns
    assert "Index"               in tbl.columns


def test_summary_keys(index_series):
    calc = InflationCalculator(index_series)
    s    = calc.summary()
    for key in ("avg_monthly_inflation", "max_monthly_inflation",
                "avg_yoy_inflation", "latest_monthly", "latest_yoy"):
        assert key in s, f"Missing summary key: {key}"


def test_division_by_zero_safe():
    """Zero index values should not raise errors."""
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    vals  = [0.0] * 12
    calc  = InflationCalculator(pd.Series(vals, index=dates))
    mom   = calc.monthly_inflation()
    # Should return inf/nan but not raise
    assert len(mom) == 12


def test_forecast_inflation():
    hist_dates = pd.date_range("2013-01-01", periods=24, freq="MS")
    hist = pd.Series([100.0 + i for i in range(24)], index=hist_dates)
    fc_dates = pd.date_range("2015-01-01", periods=6, freq="MS")
    fc   = pd.Series([125.0 + i for i in range(6)], index=fc_dates)
    result = InflationCalculator.forecast_inflation(fc, float(hist.iloc[-1]), hist)
    assert len(result) == 6
    assert "Monthly_Inflation_%" in result.columns
    assert "Forecasted_Index"    in result.columns
