"""tests/test_inflation.py — Unit tests for InflationCalculator."""
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inflation import InflationCalculator, FoodIndexCalculator, load_weights_map


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


def test_load_weights_map():
    # Test weights mapping behaves correctly with simulated food columns
    test_cols = ["bandakka", "cabbage leave", "Index", "Unknown Food Item"]
    weights = load_weights_map(price_columns=test_cols)
    
    assert "bandakka" in weights
    assert "cabbage leave" in weights
    assert "Unknown Food Item" in weights
    assert "Index" not in weights
    # Mapped weights normalized to 1.0
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_food_index_calculator():
    # Test index calculator on synthetic data
    dates = pd.date_range("2021-01-01", periods=12, freq="MS")
    hist_df = pd.DataFrame({
        "bandakka": [100.0] * 12,
        "cabbage leave": [200.0] * 12,
    }, index=dates)
    
    # Init calculator
    calc = FoodIndexCalculator(hist_df)
    
    # Base prices should be average of 2021
    assert calc.base_prices["bandakka"] == 100.0
    assert calc.base_prices["cabbage leave"] == 200.0
    
    # Historical index should be 100 since all values are at their base price
    hist_idx = calc.calculate_historical_index()
    assert all(abs(hist_idx - 100.0) < 1e-9)
    
    # Test future index with prediction results
    # Predict bandakka to rise to 150 (50% increase) and cabbage leaves constant
    results = {
        "bandakka": {
            "forecasts": {
                3: pd.DataFrame({
                    "Month": ["2022-01", "2022-02", "2022-03"],
                    "bandakka (Forecast)": [150.0, 150.0, 150.0]
                })
            }
        }
    }
    
    fc_idx = calc.calculate_future_index(results, horizon=3)
    assert len(fc_idx) == 3
    assert "Index (Forecast)" in fc_idx.columns
    # The forecasted index should reflect the weighted increase
    assert all(fc_idx["Index (Forecast)"] > 100.0)
    assert all(fc_idx["Index (Forecast)"] < 150.0)
