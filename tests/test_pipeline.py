# -*- coding: utf-8 -*-
"""
Quick integration test — tests 2 food items through full pipeline:
  preprocess -> feature engineering -> train -> evaluate -> forecast -> visualize
"""
import sys
import os
sys.path.insert(0, ".")
os.environ["PYTHONIOENCODING"] = "utf-8"

from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.predict import Forecaster
from src.visualize import Visualizer
from src.utils import get_food_columns, ensure_dirs

ensure_dirs()

# Step 1: Load data
print("=== Step 1: Preprocessing ===")
pp = DataPreprocessor()
df = pp.preprocess()
print(f"Dataset: {df.shape[0]} rows x {df.shape[1]} cols")

food_cols = get_food_columns(df)
# Test on first 2 food items only
test_targets = food_cols[:2]
print(f"Test targets: {len(test_targets)} items")

results = {}
for target in test_targets:
    print(f"\n=== Processing: (food item) ===")

    # Step 2: Feature engineering
    fe = FeatureEngineer(df, target)
    split = fe.get_split()
    print(f"  Train: {split.X_train.shape}, Test: {split.X_test.shape}")

    # Step 3: Train (ML only, fast)
    trainer = ModelTrainer(df, target)
    trained = trainer.train_all(skip_statistical=True)
    print(f"  Trained models: {list(trained.keys())}")

    # Step 4: Evaluate
    evaluator = ModelEvaluator(df, target, trained)
    metrics = evaluator.evaluate_all()
    evaluator.save_metrics()
    best_name, best_m = evaluator.best_model()
    print(f"  Best model: {best_name}, RMSE={best_m['RMSE']:.2f}, R2={best_m['R2']:.4f}")

    # Step 5: Forecast
    forecaster = Forecaster(df, target, trained)
    fc = forecaster.forecast(best_name, 12)
    forecaster.save_forecast(fc, best_name, 12)
    print(f"  Forecast rows: {len(fc)}")
    print(fc.to_string(index=False))

    results[target] = {"best": best_name, "metrics": best_m}

# Step 6: Visualize
print("\n=== Step 6: Visualizations ===")
viz = Visualizer(df)
viz.plot_historical_trends(test_targets, max_items=2)
viz.plot_correlation_heatmap(test_targets)

for target in test_targets:
    evaluator_obj = ModelEvaluator(df, target, {})
    # Reload from metrics file
    pass

print("\n=== ALL TESTS PASSED ===")
print(f"Outputs in: outputs/")
print(f"Models in:  models/saved_models/")
