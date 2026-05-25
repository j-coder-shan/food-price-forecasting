"""
dashboard/app.py — Sri Lanka AI Food Price & Inflation Forecasting Platform
Run: streamlit run dashboard/app.py
Force reload for STGNN
"""
import sys, warnings, io
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="SL Food Price AI | Inflation Forecasting",
    page_icon="🌾", layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.styles.theme import get_css
st.markdown(get_css(), unsafe_allow_html=True)


# ── File serialisation helpers (for safe caching) ──────────────────────────────
def _to_bytes(src) -> bytes | str:
    """Convert a file object to bytes (for cache key), or return path string."""
    if isinstance(src, (io.BytesIO, io.RawIOBase, io.BufferedIOBase)):
        src.seek(0); return src.read()
    return str(src)

def _from_bytes(b: bytes | str):
    """Reconstruct a usable file object from bytes or path string."""
    if isinstance(b, bytes):
        return io.BytesIO(b)
    return Path(b)


# ── Cached data loader ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading datasets…", ttl=3600)
def load_merged(food_b: bytes|str, fuel_b: bytes|str,
                exch_b: bytes|str, use_merged: bool) -> pd.DataFrame:
    from src.preprocessing import DataPreprocessor
    from src.merge_datasets import DatasetMerger

    food_src = _from_bytes(food_b)
    pp       = DataPreprocessor(filepath=food_src)
    food_df  = pp.preprocess(silent=True)

    if use_merged:
        fuel_src = _from_bytes(fuel_b)
        exch_src = _from_bytes(exch_b)
        merger   = DatasetMerger(food_src if isinstance(food_src, Path) else _from_bytes(food_b),
                                 fuel_src, exch_src)
        return merger.merge_all(food_df)
    return food_df


@st.cache_data(show_spinner=False, ttl=3600)
def _get_food_cols(path_str: str) -> list[str]:
    from src.preprocessing import DataPreprocessor
    from src.utils import get_food_columns
    pp = DataPreprocessor(filepath=Path(path_str))
    pp.load_data()
    return get_food_columns(pp.df_raw)


# ── Pipeline runner ────────────────────────────────────────────────────────────
def run_pipeline(df: pd.DataFrame, cfg: dict) -> dict:
    import src.utils, src.models.gnn, src.models.hybrid, src.train, src.evaluate, src.predict
    import importlib
    importlib.reload(src.utils)
    importlib.reload(src.models.gnn)
    importlib.reload(src.models.hybrid)
    importlib.reload(src.train)
    importlib.reload(src.evaluate)
    importlib.reload(src.predict)
    
    from src.utils import HORIZONS
    from src.train import ModelTrainer
    from src.evaluate import ModelEvaluator
    from src.predict import Forecaster

    targets  = [t for t in cfg["selected_targets"] if t != "Index"]
    if not targets:
        st.error("Please select at least one food item to forecast in addition to 'Index'.")
        return {}

    sel_mdls = cfg["selected_models"]
    skip_st  = cfg["skip_statistical"]
    horizon  = cfg["horizon"]
    all_h    = sorted(set(HORIZONS + [horizon]))

    import src.inflation
    importlib.reload(src.inflation)
    from src.inflation import FoodIndexCalculator

    results = {}
    prog    = st.progress(0, text="Initialising pipeline…")
    pipeline_errors = []


    for idx, target in enumerate(targets):
        prog.progress(
            int(idx / len(targets) * 95),
            text=f"[{idx+1}/{len(targets)}] Training → {target[:45]}…"
        )
        target_errors = []
        try:
            trainer = ModelTrainer(df, target)
            trained = trainer.train_all(skip_statistical=skip_st,
                                        selected_models=sel_mdls)
            if not trained:
                target_errors.append(f"No models trained successfully")
                results[target] = {"errors": target_errors}
                continue

            evaluator  = ModelEvaluator(df, target, trained)
            metrics    = evaluator.evaluate_all()
            evaluator.save_metrics()
            if not metrics:
                target_errors.append(f"Model evaluation failed")
                results[target] = {"errors": target_errors}
                continue

            best_name, _  = evaluator.best_model()
            preds_df      = evaluator.get_predictions_df()
            metrics_df    = evaluator.get_metrics_df()
            feat_names    = evaluator.split.feature_names

            forecaster = Forecaster(df, target, trained)
            forecasts  = {}
            for h in all_h:
                try:
                    fc = forecaster.forecast(best_name, h)
                    forecaster.save_forecast(fc, best_name, h)
                    forecasts[h] = fc
                except Exception as fe:
                    import traceback
                    target_errors.append(f"Horizon {h}m error: {fe}\n{traceback.format_exc()}")
                    pipeline_errors.append(f"{target} horizon {h}m: {fe}")

            results[target] = dict(
                best_model=best_name, trained_models=trained,
                feature_names=feat_names,
                metrics_df=metrics_df, predictions_df=preds_df,
                forecasts=forecasts,
                errors=target_errors,
            )
        except Exception as e:
            import traceback
            pipeline_errors.append(f"{target}: {e}")
            results[target] = {"errors": [f"Pipeline error: {e}\n{traceback.format_exc()}"]}

    # --- ADVANCED ST-GNN INJECTION ---
    if cfg.get("use_stgnn"):
        st.info("🧠 Running Advanced ST-GNN Multi-Target Pipeline...")
        try:
            import src.stgnn.demand_forecasting
            import src.stgnn.adjacency
            import src.stgnn.stgnn_model
            import src.stgnn.graph_visualization
            import importlib
            importlib.reload(src.stgnn.demand_forecasting)
            importlib.reload(src.stgnn.adjacency)
            importlib.reload(src.stgnn.stgnn_model)
            importlib.reload(src.stgnn.graph_visualization)
            
            from src.stgnn.demand_forecasting import DemandForecaster
            from src.stgnn.adjacency import EconomicGraphBuilder
            from src.stgnn.stgnn_model import AdvancedSTGNNRegressor
            from src.stgnn.graph_visualization import plot_economic_graph
            import pandas as pd
            import numpy as np

            # 1. Generate Demand Scores
            forecaster = DemandForecaster(df)
            demand_df = forecaster.generate_synthetic_demand()
            
            # 2. Build and Render Graph
            builder = EconomicGraphBuilder(df, demand_df)
            G = builder.build_correlation_graph()
            graph_fig = plot_economic_graph(G)
            st.session_state["stgnn_graph"] = graph_fig
            
            # 3. Train Model
            stgnn_model = AdvancedSTGNNRegressor(epochs=cfg.get("stgnn_epochs", 50))
            stgnn_model.fit(df, targets, demand_df)
            
            # 4. Predict Future (always predict at least 3 steps for the 3-month demand view)
            predict_steps = max(3, cfg["horizon"])
            preds = stgnn_model.predict_future(steps=predict_steps)
            
            # 5. Store Demand outputs (extract first 3 months)
            future_demands = []
            for i in range(3):
                if i < len(preds['demands']):
                    series = pd.Series(preds['demands'][i], index=[f"{c}_Demand" for c in targets])
                    future_demands.append(series)
            
            st.session_state["stgnn_demand"] = {
                "historical": demand_df,
                "forecast": future_demands
            }
            
            # Overwrite the target forecasts with ST-GNN predictions to show up in the main UI
            for idx, tgt in enumerate(targets):
                if tgt not in results:
                    results[tgt] = {"errors": [], "forecasts": {}}
                
                if "forecasts" not in results[tgt]:
                    results[tgt]["forecasts"] = {}
                
                # Extract predicted price curve for this target (limit to user's selected horizon)
                future_prices = [p[idx] for p in preds['prices']][:cfg["horizon"]]
                future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=cfg["horizon"], freq='MS')
                
                stgnn_fc_df = pd.DataFrame({
                    "Month": future_dates.strftime("%Y-%m"),
                    f"{tgt} (Forecast)": future_prices
                })
                
                results[tgt]["forecasts"][cfg["horizon"]] = stgnn_fc_df
                results[tgt]["best_model"] = "ST-GNN (Advanced)"
                if "trained_models" not in results[tgt]:
                    results[tgt]["trained_models"] = {"ST-GNN (Advanced)": stgnn_model}
                else:
                    results[tgt]["trained_models"]["ST-GNN (Advanced)"] = stgnn_model
                    
            st.success("✅ ST-GNN Pipeline Complete!")
            
        except Exception as e:
            import traceback
            st.error(f"ST-GNN Execution Failed: {e}\n{traceback.format_exc()}")
    # ---------------------------------

    # --- DYNAMIC FOOD INDEX & INFLATION INJECTION ---
    if "Index" in df.columns:
        st.info("📊 Computing Dynamic Food Price Index (Laspeyres)...")
        try:
            calculator = FoodIndexCalculator(df)
            
            # Find all horizons predicted in results
            run_horizons = set()
            for tgt in targets:
                if tgt in results and "forecasts" in results[tgt]:
                    run_horizons.update(results[tgt]["forecasts"].keys())
            
            if not run_horizons:
                run_horizons = {horizon}
                
            index_forecasts = {}
            for h in run_horizons:
                index_forecasts[h] = calculator.calculate_future_index(results, h)
                
            results["Index"] = dict(
                best_model="Dynamic Basket Index",
                trained_models={},
                feature_names=[],
                metrics_df=pd.DataFrame(),
                predictions_df=pd.DataFrame(),
                forecasts=index_forecasts,
                errors=[],
            )
            st.success("✅ Dynamic Food Price Index computed!")
        except Exception as e:
            import traceback
            st.error(f"Dynamic Index calculation failed: {e}\n{traceback.format_exc()}")
            pipeline_errors.append(f"Dynamic Index: {e}")
    # ------------------------------------------------

    prog.progress(100, text=f"Done — {len(results)}/{len(targets)} items forecasted.")

    if pipeline_errors:
        with st.expander(f"⚠️ {len(pipeline_errors)} warning(s) during pipeline"):
            for e in pipeline_errors:
                st.caption(f"• {e}")
    return results


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    from src.utils import DATA_DIR, get_food_columns

    # ── Header bar ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0D1629 0%,#111827 100%);
                border:1px solid #1E2D4A;border-radius:14px;
                padding:1rem 1.4rem;margin-bottom:1rem;">
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;">
            <div>
                <h1 style="margin:0;font-family:Poppins,sans-serif;font-weight:800;
                           font-size:1.55rem;color:#F1F5F9;line-height:1.2;">
                    🇱🇰 Sri Lanka Food Price &amp; Inflation
                </h1>
                <p style="margin:4px 0 0;color:#64748B;font-size:0.82rem;">
                    Multi-model forecasting &nbsp;·&nbsp; Fuel &amp; FX integration &nbsp;·&nbsp;
                    Inflation analytics &nbsp;·&nbsp; 2013 – present
                </p>
            </div>
            <div style="display:flex;gap:12px;flex-wrap:wrap;">
                <div style="background:#1A2238;border:1px solid #1E2D4A;border-radius:8px;
                            padding:6px 14px;text-align:center;">
                    <div style="font-size:0.65rem;color:#64748B;text-transform:uppercase;">Models</div>
                    <div style="font-family:Poppins;font-weight:700;color:#3B82F6;font-size:1.1rem;">8</div>
                </div>
                <div style="background:#1A2238;border:1px solid #1E2D4A;border-radius:8px;
                            padding:6px 14px;text-align:center;">
                    <div style="font-size:0.65rem;color:#64748B;text-transform:uppercase;">Datasets</div>
                    <div style="font-family:Poppins;font-weight:700;color:#06B6D4;font-size:1.1rem;">3</div>
                </div>
                <div style="background:#1A2238;border:1px solid #1E2D4A;border-radius:8px;
                            padding:6px 14px;text-align:center;">
                    <div style="font-size:0.65rem;color:#64748B;text-transform:uppercase;">Exports</div>
                    <div style="font-family:Poppins;font-weight:700;color:#10B981;font-size:1.1rem;">CSV·XLSX·PDF</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Initialise file session state ───────────────────────────────────────────
    food_path = DATA_DIR / "food_prices.xlsx"
    for key, val in [("food_file_obj", food_path),
                     ("fuel_file_obj",  DATA_DIR / "fuel_prices.xls"),
                     ("exch_file_obj",  DATA_DIR / "exchange_rates.xlsx")]:
        if key not in st.session_state:
            st.session_state[key] = val

    # Fast column preview (cached)
    try:
        food_cols_preview = _get_food_cols(str(food_path))
    except Exception:
        food_cols_preview = ["(Place food_prices.xlsx in data/)"]

    # ── Sidebar ─────────────────────────────────────────────────────────────────
    from dashboard.components.sidebar import render_sidebar
    cfg = render_sidebar(food_cols_preview)

    # ── Load / Merge data ────────────────────────────────────────────────────────
    try:
        df = load_merged(
            _to_bytes(cfg["food_file"]),
            _to_bytes(cfg["fuel_file"]),
            _to_bytes(cfg["exchange_file"]),
            cfg["use_merged"],
        )
    except Exception as e:
        st.error(f"**Data load failed**: {e}")
        st.info("Check that `data/food_prices.xlsx` exists or upload a valid file.")
        st.stop()

    # ── Run pipeline ─────────────────────────────────────────────────────────────
    if cfg["run"]:
        targets_to_forecast = [t for t in cfg["selected_targets"] if t != "Index"]
        if not targets_to_forecast:
            st.error("Please select at least one food item to forecast in the sidebar (in addition to 'Index')."); st.stop()
        if not cfg["selected_models"]:
            st.error("Select at least one model in the sidebar."); st.stop()
        with st.spinner("Running forecasting pipeline — this may take a few minutes…"):
            results = run_pipeline(df, cfg)
        st.session_state.update(results=results, run_cfg=cfg, run_df=df)
        if results:
            st.success(f"✅ Forecast complete for **{len(results)}** item(s). "
                       "Switch tabs to explore results.")

    # Retrieve persisted results
    results = st.session_state.get("results", {})
    cfg_s   = st.session_state.get("run_cfg", cfg)
    df_s    = st.session_state.get("run_df",  df)

    # ── Tabs ──────────────────────────────────────────────────────────────────────
    from dashboard.page_modules.overview      import render as r_overview
    from dashboard.page_modules.forecast      import render as r_forecast
    from dashboard.page_modules.inflation_tab import render as r_inflation
    from dashboard.page_modules.model_compare import render as r_compare
    from dashboard.page_modules.export_tab    import render as r_export
    from dashboard.page_modules.demand_tab    import render as r_demand
    from dashboard.page_modules.graph_tab     import render as r_graph

    tabs = st.tabs([
        "📈  Overview",
        "🔮  Forecast",
        "📉  Inflation",
        "🏆  Model Comparison",
        "🔥  Demand",
        "🕸  Economic Graph",
        "⬇️  Export",
    ])

    with tabs[0]: r_overview(df_s, cfg_s)
    with tabs[1]: r_forecast(df_s, cfg_s, results)
    with tabs[2]: r_inflation(df_s, results)
    with tabs[3]: r_compare(df_s, results)
    with tabs[4]: r_demand(df_s, results)
    with tabs[5]: r_graph(results)
    with tabs[6]: r_export(df_s, cfg_s, results)


if __name__ == "__main__":
    main()
