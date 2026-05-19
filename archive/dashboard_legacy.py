"""
dashboard.py — Streamlit interactive dashboard for Sri Lanka Food Price Forecasting.
Sri Lanka Food Price Forecasting System

Run with:
    streamlit run dashboard.py

Features:
    • Tab 1: Historical Trends (interactive Plotly)
    • Tab 2: Model Performance (metrics table + RMSE comparison chart)
    • Tab 3: Future Forecast (interactive chart + CSV download)
    • Tab 4: Correlation Analysis (interactive heatmap)
    • Tab 5: Seasonal Analysis (decomposition chart)
"""

import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import (
    get_food_columns, METRICS_DIR, FORECASTS_DIR, GRAPHS_DIR,
    sanitize_filename, HORIZONS,
)
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.predict import Forecaster

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🇱🇰 SL Food Price Forecasting",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — Dark Premium Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main { background-color: #0f1117; }

    .stApp { background-color: #0f1117; }

    .metric-card {
        background: linear-gradient(135deg, #1e1f2e 0%, #252638 100%);
        border: 1px solid #2a2a3a;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 4px;
    }

    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #4fc3f7;
    }

    .metric-card .label {
        font-size: 0.8rem;
        color: #9e9e9e;
        margin-top: 4px;
    }

    .header-badge {
        background: linear-gradient(135deg, #1565c0, #0d47a1);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block;
        margin-bottom: 12px;
    }

    .forecast-table {
        background: #1e1f2e;
        border-radius: 10px;
        padding: 16px;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid #2a2a3a;
    }

    .stTabs [data-baseweb="tab-list"] {
        background-color: #161b22;
        border-radius: 8px;
        gap: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        color: #9e9e9e;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1565c0 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Cached Data Loading
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading Sri Lankan food price data...")
def load_data():
    preprocessor = DataPreprocessor()
    return preprocessor.preprocess()


@st.cache_resource(show_spinner="Training forecasting models...")
def train_models_cached(target: str, skip_statistical: bool = True):
    """Cache trained models per target to avoid retraining on reruns."""
    df = load_data()
    trainer = ModelTrainer(df, target)
    return trainer.train_all(skip_statistical=skip_statistical)


# ─────────────────────────────────────────────
# Helper: Plotly Dark Theme
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0f1117",
    plot_bgcolor="#16171f",
    font=dict(family="Inter", color="#e0e0e0"),
    xaxis=dict(gridcolor="#2a2a3a", showgrid=True),
    yaxis=dict(gridcolor="#2a2a3a", showgrid=True),
    legend=dict(bgcolor="#1e1f2e", bordercolor="#2a2a3a", borderwidth=1),
    margin=dict(l=20, r=20, t=50, b=30),
)


# ─────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────
def main():
    # ── Header ──────────────────────────────
    st.markdown("""
    <div style="display:flex; align-items:center; gap:16px; margin-bottom:24px;">
        <div>
            <h1 style="margin:0; font-size:2rem; font-weight:700; color:#e0e0e0;">
                🇱🇰 Sri Lanka Food Price Forecasting
            </h1>
            <p style="margin:4px 0 0; color:#9e9e9e; font-size:0.95rem;">
                Monthly price predictions · 2013 onwards · Machine Learning powered
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load Data ────────────────────────────
    try:
        df = load_data()
    except FileNotFoundError as e:
        st.error(f"❌ Dataset not found: {e}")
        st.stop()

    food_cols = get_food_columns(df)
    all_targets = food_cols + (["Index"] if "Index" in df.columns else [])

    # ── Sidebar ──────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")

        selected_target = st.selectbox(
            "🎯 Food Item",
            options=all_targets,
            index=0,
            help="Select the food item to analyze and forecast.",
        )

        selected_horizon = st.selectbox(
            "📅 Forecast Horizon",
            options=[3, 6, 12],
            index=2,
            format_func=lambda x: f"{x} months",
        )

        skip_stat = st.checkbox(
            "⚡ Fast mode (skip ARIMA/SARIMA/Prophet)",
            value=True,
            help="Unchecking trains statistical models too (slower)."
        )

        st.markdown("---")
        st.markdown("**📊 Dataset Overview**")
        st.metric("Total Months", len(df))
        st.metric("Food Items", len(food_cols))
        st.metric(
            "Date Range",
            f"{df.index.min().strftime('%Y-%m')} → {df.index.max().strftime('%Y-%m')}"
        )

        st.markdown("---")
        run_btn = st.button("🚀 Run Forecast", type="primary", use_container_width=True)

    # ── Tabs ─────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Historical Trends",
        "🏆 Model Performance",
        "🔮 Forecast",
        "🔥 Correlations",
        "🌀 Seasonal Analysis",
    ])

    # ═════════════════════════════════════════
    # TAB 1: Historical Trends
    # ═════════════════════════════════════════
    with tab1:
        st.markdown(f"### 📈 Historical Price Trends")

        # Multi-item selector
        show_items = st.multiselect(
            "Select food items to display",
            options=food_cols,
            default=food_cols[:6],
        )

        if show_items:
            fig = go.Figure()
            colors = px.colors.qualitative.Plotly

            for i, col in enumerate(show_items):
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode="lines",
                    name=col,
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f"<b>{col}</b><br>Month: %{{x|%Y-%m}}<br>Price: LKR %{{y:,.2f}}<extra></extra>",
                ))

            fig.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text="Sri Lanka Monthly Food Prices (LKR)", font=dict(size=16)),
                xaxis_title="Month",
                yaxis_title="Price (LKR)",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Summary statistics
        st.markdown("#### 📊 Summary Statistics")
        if show_items:
            stats = df[show_items].describe().round(2)
            st.dataframe(stats.style.background_gradient(cmap="Blues"), use_container_width=True)

    # ═════════════════════════════════════════
    # TAB 2: Model Performance
    # ═════════════════════════════════════════
    with tab2:
        st.markdown(f"### 🏆 Model Performance — *{selected_target}*")

        if run_btn or st.session_state.get("models_trained"):
            with st.spinner(f"Training & evaluating models for **{selected_target}**..."):
                trained = train_models_cached(selected_target, skip_statistical=skip_stat)
                evaluator = ModelEvaluator(df, selected_target, trained)
                metrics = evaluator.evaluate_all()
                evaluator.save_metrics()
                st.session_state["models_trained"] = True
                st.session_state["trained"] = trained
                st.session_state["evaluator"] = evaluator
                st.session_state["metrics"] = metrics

        if "metrics" in st.session_state and st.session_state["metrics"]:
            metrics_df = st.session_state["evaluator"].get_metrics_df()
            best_name, best_m = st.session_state["evaluator"].best_model()

            # KPI cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="metric-card"><div class="value">{best_m["MAE"]:.1f}</div><div class="label">MAE (LKR)</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><div class="value">{best_m["RMSE"]:.1f}</div><div class="label">RMSE (LKR)</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-card"><div class="value">{best_m["MAPE"]:.1f}%</div><div class="label">MAPE</div></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="metric-card"><div class="value">{best_m["R2"]:.3f}</div><div class="label">R² Score</div></div>', unsafe_allow_html=True)

            st.success(f"🥇 Best Model: **{best_name}** (RMSE = {best_m['RMSE']:.2f} LKR)")

            # Metrics table
            st.markdown("#### All Models Comparison")
            st.dataframe(metrics_df.style.background_gradient(subset=["RMSE"], cmap="RdYlGn_r"), use_container_width=True)

            # RMSE bar chart
            fig = px.bar(
                metrics_df.sort_values("RMSE"),
                x="Model", y="RMSE",
                color="RMSE",
                color_continuous_scale="RdYlGn_r",
                title=f"RMSE Comparison — {selected_target}",
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Actual vs Predicted
            preds_df = st.session_state["evaluator"].get_predictions_df()
            st.markdown("#### Actual vs Predicted (Test Set — 20%)")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=preds_df.index, y=preds_df["Actual"],
                name="Actual", line=dict(color="white", width=2.5),
            ))
            model_colors = px.colors.qualitative.Set2
            for i, col in enumerate([c for c in preds_df.columns if c != "Actual"]):
                fig2.add_trace(go.Scatter(
                    x=preds_df.index, y=preds_df[col],
                    name=col, line=dict(color=model_colors[i % len(model_colors)], width=1.8, dash="dash"),
                ))
            fig2.update_layout(**PLOTLY_LAYOUT, height=420, title="Test Set: Actual vs Predicted")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("👆 Click **Run Forecast** in the sidebar to train models and see performance metrics.")

    # ═════════════════════════════════════════
    # TAB 3: Future Forecast
    # ═════════════════════════════════════════
    with tab3:
        st.markdown(f"### 🔮 Future Forecast — *{selected_target}* ({selected_horizon} months)")

        if "trained" in st.session_state and st.session_state["trained"]:
            best_name, _ = st.session_state["evaluator"].best_model()
            forecaster = Forecaster(df, selected_target, st.session_state["trained"])
            fc_df = forecaster.forecast(best_name, selected_horizon)
            forecaster.save_forecast(fc_df, best_name, selected_horizon)

            fc_col = f"{selected_target} (Forecast)"
            fc_dates = pd.to_datetime(fc_df["Month"])
            fc_vals = fc_df[fc_col].values

            # Chart: history + forecast
            hist = df[selected_target].dropna()
            band = fc_vals * 0.10

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist.values,
                name="Historical", line=dict(color="#4fc3f7", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=fc_dates, y=fc_vals,
                name=f"Forecast ({best_name})",
                line=dict(color="#ff7043", width=2.5, dash="dash"),
            ))
            fig.add_trace(go.Scatter(
                x=pd.concat([fc_dates.to_series(), fc_dates.to_series()[::-1]]),
                y=np.concatenate([fc_vals + band, (fc_vals - band)[::-1]]),
                fill="toself",
                fillcolor="rgba(255,112,67,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name="±10% band",
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                height=480,
                title=f"{selected_target} — {selected_horizon}-Month Forecast (LKR)",
                xaxis_title="Month",
                yaxis_title="Price (LKR)",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Table
            st.markdown("#### 📋 Forecast Table")
            st.dataframe(fc_df, use_container_width=True, hide_index=True)

            # Download CSV
            csv = fc_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Forecast CSV",
                data=csv,
                file_name=f"{sanitize_filename(selected_target)}_{best_name}_{selected_horizon}m_forecast.csv",
                mime="text/csv",
            )
        else:
            st.info("👆 Click **Run Forecast** in the sidebar first.")

    # ═════════════════════════════════════════
    # TAB 4: Correlation Analysis
    # ═════════════════════════════════════════
    with tab4:
        st.markdown("### 🔥 Food Price Correlation Matrix")

        corr_items = st.multiselect(
            "Items to include in correlation",
            options=food_cols,
            default=food_cols[:15],
        )

        if corr_items and len(corr_items) >= 2:
            corr = df[corr_items].corr()
            short = {c: c[:18] + "…" if len(c) > 18 else c for c in corr.columns}
            corr = corr.rename(columns=short, index=short)

            fig = px.imshow(
                corr,
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                aspect="auto",
                title="Pearson Correlation Coefficients",
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=600)
            fig.update_traces(
                text=corr.round(2).values,
                texttemplate="%{text}",
                textfont=dict(size=9),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Highest correlations
            st.markdown("#### 🔗 Top Correlated Pairs")
            corr_pairs = (
                corr.unstack()
                .reset_index()
                .rename(columns={"level_0": "Item A", "level_1": "Item B", 0: "Correlation"})
            )
            corr_pairs = corr_pairs[corr_pairs["Item A"] < corr_pairs["Item B"]]
            corr_pairs = corr_pairs.sort_values("Correlation", ascending=False).head(10)
            st.dataframe(corr_pairs.reset_index(drop=True), use_container_width=True, hide_index=True)
        else:
            st.info("Select at least 2 food items above.")

    # ═════════════════════════════════════════
    # TAB 5: Seasonal Analysis
    # ═════════════════════════════════════════
    with tab5:
        st.markdown(f"### 🌀 Seasonal Decomposition — *{selected_target}*")

        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            series = df[selected_target].dropna()
            if len(series) >= 24:
                decomp = seasonal_decompose(series, model="additive", period=12)

                components = {
                    "Observed": series,
                    "Trend": decomp.trend,
                    "Seasonal": decomp.seasonal,
                    "Residual": decomp.resid,
                }
                colors_map = {
                    "Observed": "#4fc3f7",
                    "Trend": "#a5d6a7",
                    "Seasonal": "#ce93d8",
                    "Residual": "#ef9a9a",
                }

                for comp_name, comp_data in components.items():
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=comp_data.index,
                        y=comp_data.values,
                        mode="lines",
                        line=dict(color=colors_map[comp_name], width=1.8),
                        name=comp_name,
                    ))
                    fig.update_layout(
                        **PLOTLY_LAYOUT,
                        height=220,
                        title=comp_name,
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough data points for seasonal decomposition (need ≥24 months).")
        except Exception as e:
            st.error(f"Seasonal decomposition error: {e}")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
