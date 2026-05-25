# -*- coding: utf-8 -*-
"""
Unit test for STGNNLinearHybridRegressor.
"""
import sys
import os
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models.hybrid import STGNNLinearHybridRegressor

def test_hybrid_model():
    print("=== Step 1: Preprocessing & Data Loading ===")
    pp = DataPreprocessor()
    df = pp.preprocess(silent=True)
    target = df.columns[0] # samba rice or first commodity
    print(f"Target selected: {target}")

    print("=== Step 2: Feature Engineering ===")
    fe = FeatureEngineer(df, target)
    split = fe.get_split()
    print(f"Train Shape X: {split.X_train.shape}, y: {split.y_train.shape}")
    print(f"Test Shape X: {split.X_test.shape}, y: {split.y_test.shape}")

    print("=== Step 3: Instantiating and Fitting Hybrid Model ===")
    # Use very few epochs for fast execution in test
    hybrid_model = STGNNLinearHybridRegressor(
        epochs=5,
        batch_size=16,
        lr=5e-3,
        patience=2,
        seq_len=12,
        df=df,
        target=target
    )
    
    # Train the hybrid model
    hybrid_model.fit(split.X_train, split.y_train)
    print("Model fitted successfully!")

    print("=== Step 4: Scenario B - Test Set Predictions ===")
    preds = hybrid_model.predict(split.X_test)
    print(f"Generated predictions of shape: {preds.shape}")
    
    assert isinstance(preds, np.ndarray), "Predictions must be a NumPy array"
    assert preds.shape == split.y_test.shape, "Predictions shape must match test target shape"
    assert not np.isnan(preds).any(), "Predictions must not contain NaN values"
    print("Scenario B test assertions passed!")

    print("=== Step 5: Scenario A - Recursive Single Row Predictions ===")
    # Predict on a single row (similar to recursive forecasting)
    single_row = split.X_test.iloc[0:1]
    single_pred = hybrid_model.predict(single_row)
    print(f"Single prediction: {single_pred}")
    assert len(single_pred) == 1, "Single prediction must return 1 value"
    assert not np.isnan(single_pred)[0], "Single prediction must not be NaN"
    print("Scenario A single-row assertions passed!")

    print("\n=== HYBRID MODEL UNIT TEST PASSED ===")

if __name__ == "__main__":
    test_hybrid_model()
