"""tests/test_preprocessing.py — Unit tests for DataPreprocessor."""
import sys
from pathlib import Path
import pytest
import pandas as pd
import io

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import DataPreprocessor
from src.utils import DATA_FILE


@pytest.fixture
def preprocessor():
    pp = DataPreprocessor(filepath=DATA_FILE)
    return pp


def test_load_data(preprocessor):
    preprocessor.load_data()
    assert preprocessor.df_raw is not None
    assert "Month" in preprocessor.df_raw.columns
    assert len(preprocessor.df_raw) > 0


def test_preprocess_returns_dataframe(preprocessor):
    df = preprocessor.preprocess(silent=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_date_index_is_datetime(preprocessor):
    df = preprocessor.preprocess(silent=True)
    assert isinstance(df.index, pd.DatetimeIndex)


def test_no_missing_values(preprocessor):
    df = preprocessor.preprocess(silent=True)
    assert df.isnull().sum().sum() == 0, "Missing values remain after preprocessing"


def test_has_index_column(preprocessor):
    df = preprocessor.preprocess(silent=True)
    assert "Index" in df.columns, "Food price Index column must be present"


def test_all_numeric(preprocessor):
    df = preprocessor.preprocess(silent=True)
    non_numeric = [c for c in df.columns if df[c].dtype == object]
    assert len(non_numeric) == 0, f"Non-numeric columns: {non_numeric}"


def test_chronological_order(preprocessor):
    df = preprocessor.preprocess(silent=True)
    assert df.index.is_monotonic_increasing, "Index must be sorted chronologically"


def test_bytesio_input():
    """DataPreprocessor should accept BytesIO (Streamlit upload simulation)."""
    with open(DATA_FILE, "rb") as f:
        buf = io.BytesIO(f.read())
    pp = DataPreprocessor(filepath=buf)
    df = pp.preprocess(silent=True)
    assert len(df) > 0
    assert isinstance(df.index, pd.DatetimeIndex)
