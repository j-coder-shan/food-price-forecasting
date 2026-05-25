"""index_weights.py — Loads and parses base period weights for the food index."""
import re
import pandas as pd
import numpy as np
from src.utils import get_logger, DATA_DIR

logger = get_logger(__name__)

# Custom mappings from price column names (cleaned) to weight item names (cleaned)
COLUMN_TO_WEIGHT_MAP = {
    "bandakka": "ladies fingers",
    "cabbage leave": "cabbage leaves",
    "cucumber/kekiri": "cucumber",
    "turmeric/turmeric powder": "turmeric/ turmeric powder",
}

def clean_name(name: str) -> str:
    """Normalizes item names for robust matching.
    Replaces Unicode hyphens with ASCII hyphens, strips whitespace, and lowercases.
    """
    if pd.isna(name):
        return ""
    n = re.sub(r"[\u2010-\u2015\u002D\-]+", "-", str(name).strip())
    return re.sub(r"\s+", " ", n).lower()

def load_weights_map(
    weights_path=None, price_columns=None
) -> dict[str, float]:
    """Loads weights from BasePeriodWeights_.xlsx and maps them to price columns.
    
    Returns:
        dict: Mapping of {price_column_name: normalized_weight} (summing to 1.0).
    """
    if weights_path is None:
        weights_path = DATA_DIR / "BasePeriodWeights_.xlsx"
    
    try:
        w_df = pd.read_excel(weights_path)
    except Exception as e:
        logger.error(f"Failed to load weights from {weights_path}: {e}")
        # Fallback to empty dict or default weights
        return {}

    # Extract weights dict from excel
    weights_raw = {}
    for _, row in w_df.iterrows():
        item = row.get("Item")
        weight = row.get("Weight")
        if pd.isna(item) or pd.isna(weight) or clean_name(item) == "total":
            continue
        try:
            weights_raw[clean_name(item)] = float(weight)
        except ValueError:
            continue

    if price_columns is None:
        # Return raw cleaned weights if no columns provided
        return weights_raw

    # Map weights to actual price columns
    mapped_weights = {}
    total_matched_weight = 0.0

    for col in price_columns:
        if col in ["Month", "Index", "Date"]:
            continue
        
        col_clean = clean_name(col)
        # Apply custom mappings if present, otherwise use the cleaned column name
        mapped_name = COLUMN_TO_WEIGHT_MAP.get(col_clean, col_clean)
        
        if mapped_name in weights_raw:
            weight_val = weights_raw[mapped_name]
            mapped_weights[col] = weight_val
            total_matched_weight += weight_val
        else:
            # Assign a tiny default weight for unmapped columns to avoid ignoring them
            mapped_weights[col] = 0.01
            total_matched_weight += 0.01
            logger.debug(f"Weight not found for item: '{col}' (cleaned: '{col_clean}'). Assigned default 0.01.")

    # Normalize weights so they sum to 1.0
    if total_matched_weight > 0:
        mapped_weights = {k: v / total_matched_weight for k, v in mapped_weights.items()}
    else:
        # Equal weights fallback
        n_cols = len(mapped_weights)
        if n_cols > 0:
            mapped_weights = {k: 1.0 / n_cols for k in mapped_weights.keys()}

    logger.info(f"Loaded and mapped weights for {len(mapped_weights)} food items.")
    return mapped_weights
