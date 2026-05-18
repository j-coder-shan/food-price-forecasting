"""
main.py — Full pipeline orchestrator for the Sri Lanka Food Price Forecasting System.

Usage:
    python main.py                             # Run all targets, all models, 12-month forecast
    python main.py --targets "Rice ‐ (Kekulu white)" Tomatoes
    python main.py --horizon 6
    python main.py --skip-statistical          # Skip ARIMA/SARIMA/Prophet (faster)
    python main.py --targets all --horizon 12
    python main.py --dashboard                 # Launch Streamlit after pipeline

Pipeline:
    Load → Preprocess → Feature Engineering → Train (80%) → Evaluate (20%) → Forecast → Save → Visualize
"""

import argparse
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────
# Ensure project root is on Python path
# ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import (
    get_logger, ensure_dirs, get_food_columns,
    format_metric_table, HORIZONS,
)
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.predict import Forecaster
from src.visualize import Visualizer

logger = get_logger("main")


# ─────────────────────────────────────────────
# CLI Argument Parser
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Sri Lanka Food Price Forecasting Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["auto"],
        help=(
            "Food item columns to forecast.\n"
            "Use 'auto' to forecast all items (default).\n"
            "Example: --targets 'Rice ‐ (Kekulu white)' Tomatoes"
        ),
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=12,
        choices=[3, 6, 12],
        help="Primary forecast horizon in months (3, 6, or 12). Default: 12",
    )
    parser.add_argument(
        "--skip-statistical",
        action="store_true",
        help="Skip ARIMA/SARIMA/Prophet models (much faster for many targets).",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="Limit the number of targets to process (useful for quick tests).",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch Streamlit dashboard after pipeline completes.",
    )
    return parser.parse_args()


# ---------------------------------------------
# Print Section Header
# ---------------------------------------------
def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ---------------------------------------------
# Main Pipeline
# ---------------------------------------------
def run_pipeline(args) -> None:
    start_time = time.time()
    ensure_dirs()

    # ----------------------------
    # STEP 1: Load & Preprocess
    # ----------------------------
    section("STEP 1: Load & Preprocess Dataset")
    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess()

    # ----------------------------
    # STEP 2: Select Targets
    # ----------------------------
    section("STEP 2: Select Forecast Targets")
    all_food_cols = get_food_columns(df)
    all_targets = all_food_cols + (["Index"] if "Index" in df.columns else [])

    if args.targets == ["auto"]:
        targets = all_targets
    else:
        targets = [t for t in args.targets if t in df.columns]
        if not targets:
            logger.error("None of the specified targets found in dataset. Use 'auto' or check column names.")
            sys.exit(1)

    if args.max_targets:
        targets = targets[: args.max_targets]

    print(f"\n  Targets selected ({len(targets)}):")
    for t in targets:
        print(f"    • {t}")

    # Storage for pipeline outputs
    trained_models_map: dict[str, dict] = {}
    best_model_map: dict[str, str] = {}
    evaluator_map: dict[str, ModelEvaluator] = {}
    all_metrics: dict[str, dict] = {}

    # ----------------------------
    # STEP 3–5: Train / Evaluate per Target
    # ----------------------------
    for i, target in enumerate(targets, 1):
        section(f"STEP 3-5 [{i}/{len(targets)}]: {target}")

        # TRAIN (80%)
        trainer = ModelTrainer(df, target)
        trained_models = trainer.train_all(skip_statistical=args.skip_statistical)
        trained_models_map[target] = trained_models

        if not trained_models:
            logger.warning(f"  No models trained for '{target}' — skipping.")
            continue

        # EVALUATE (20% held-out test set)
        evaluator = ModelEvaluator(df, target, trained_models)
        metrics = evaluator.evaluate_all()
        evaluator_map[target] = evaluator

        if not metrics:
            logger.warning(f"  No evaluation results for '{target}'.")
            continue

        # Save metrics CSV
        evaluator.save_metrics()
        all_metrics[target] = metrics

        # Select best model (lowest RMSE)
        best_name, best_metrics = evaluator.best_model()
        best_model_map[target] = best_name

        print(f"\n  Best model: {best_name} — RMSE={best_metrics['RMSE']:.2f} | MAE={best_metrics['MAE']:.2f} | MAPE={best_metrics['MAPE']:.2f}%")

    # ----------------------------
    # STEP 6: Generate Forecasts
    # ----------------------------
    section("STEP 6: Generate Future Forecasts")
    all_forecasts: dict[str, dict[int, dict]] = {}

    for target in targets:
        if target not in trained_models_map or target not in best_model_map:
            continue

        best_model_name = best_model_map[target]
        forecaster = Forecaster(df, target, trained_models_map[target])
        forecasts = forecaster.forecast_all_horizons(best_model_name, HORIZONS)
        all_forecasts[target] = forecasts

        # Print 12-month forecast table
        if 12 in forecasts:
            fc_df = forecasts[12]
            print(f"\n  📅 {target} — 12-Month Forecast ({best_model_name}):")
            print(f"  {fc_df.to_string(index=False)}")

    # ----------------------------
    # STEP 7: Generate Graphs
    # ----------------------------
    section("STEP 7: Generate Visualizations")
    viz = Visualizer(df)

    # Historical trends (up to 10 items)
    viz.plot_historical_trends(all_food_cols, max_items=10)
    logger.info("  ✓ Historical trends chart saved.")

    # Correlation heatmap
    viz.plot_correlation_heatmap()
    logger.info("  ✓ Correlation heatmap saved.")

    # Price dashboard grid
    viz.plot_price_dashboard(all_food_cols)
    logger.info("  ✓ Price dashboard grid saved.")

    # Per-target charts
    for target in targets:
        evaluator = evaluator_map.get(target)
        if evaluator:
            preds_df = evaluator.get_predictions_df()
            metrics_df = evaluator.get_metrics_df()

            viz.plot_actual_vs_predicted(target, preds_df)
            viz.plot_model_comparison(metrics_df, target)
            viz.plot_seasonal_decomposition(target)

            # Feature importance for best tree-based model
            best_model_name = best_model_map.get(target)
            if best_model_name in trained_models_map.get(target, {}):
                model = trained_models_map[target][best_model_name]
                viz.plot_feature_importance(
                    model,
                    best_model_name,
                    evaluator.split.feature_names,
                    target,
                )

        # Future forecast chart (12-month)
        if target in all_forecasts and 12 in all_forecasts[target]:
            viz.plot_future_forecast(target, all_forecasts[target][12], horizon=12)

    # ----------------------------
    # STEP 8: Summary Report
    # ----------------------------
    section("STEP 8: Summary")
    elapsed = time.time() - start_time
    print(f"  Pipeline completed in {elapsed:.1f}s")
    print(f"  Targets processed : {len(targets)}")
    print(f"  Forecast horizons : {HORIZONS}")
    print(f"\n  Output locations:")
    print(f"    • Metrics  : outputs/metrics/")
    print(f"    • Forecasts: outputs/forecasts/")
    print(f"    • Graphs   : outputs/graphs/")
    print(f"    • Models   : models/saved_models/")

    if all_metrics:
        print(f"  Best Models Summary:")
        print(f"  {'Target':<35} {'Best Model':<20} {'RMSE':>8}")
        print(f"  {'-'*65}")
        for target, best_name in best_model_map.items():
            rmse = all_metrics.get(target, {}).get(best_name, {}).get("RMSE", "N/A")
            print(f"  {target[:34]:<35} {best_name:<20} {rmse:>8}")


# ---------------------------------------------
# Entry Point
# ---------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    print("""
+===============================================================+
|      Sri Lanka Food Price Forecasting System                 |
|      Monthly Price Prediction -- 2013 onwards                |
+===============================================================+
""")

    run_pipeline(args)

    if args.dashboard:
        import subprocess
        print("\n  🚀 Launching Streamlit dashboard...")
        subprocess.run(["streamlit", "run", "dashboard.py"])
