"""dashboard/page_modules/model_compare.py — Tab 4: Model Comparison."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard.components.charts import (
    rmse_bar, multi_metric_bar, actual_vs_predicted, feature_importance_chart
)
from dashboard.styles.theme import (
    compact_layout, COLORS, ACCENT_GREEN, ACCENT_AMBER, ACCENT_RED,
    TEXT_PRIMARY, TEXT_SECONDARY, BG_CARD, BORDER,
)

_MEDALS = ["🥇", "🥈", "🥉"]


def _ranking_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df.sort_values("RMSE").reset_index(drop=True)
    df.insert(0, "Rank", [
        _MEDALS[i] if i < 3 else f"#{i+1}" for i in range(len(df))
    ])
    return df


def _radar_chart(metrics_df: pd.DataFrame, target: str) -> go.Figure | None:
    """Multi-model radar chart for RMSE, MAE, R²."""
    if len(metrics_df) < 2: return None
    import numpy as np
    cats = ["RMSE", "MAE", "R2"]
    df = metrics_df.copy()
    # Normalise metrics 0–1 (lower is better for RMSE/MAE, higher for R²)
    for c in ("RMSE", "MAE"):
        mn, mx = df[c].min(), df[c].max()
        df[f"{c}_n"] = 1 - (df[c] - mn) / (mx - mn + 1e-9)
    mn, mx = df["R2"].min(), df["R2"].max()
    df["R2_n"] = (df["R2"] - mn) / (mx - mn + 1e-9)

    layout = compact_layout(title=f"Model Radar — {target}", height=380)
    fig = go.Figure()
    fig.update_layout(**layout)
    fig.update_layout(polar=dict(
        bgcolor=BG_CARD,
        radialaxis=dict(visible=True, range=[0, 1], showticklabels=False,
                        gridcolor=BORDER),
        angularaxis=dict(color=TEXT_SECONDARY, gridcolor=BORDER),
    ))

    categories = ["RMSE (norm)", "MAE (norm)", "R² (norm)"]
    for i, row in df.iterrows():
        vals = [row["RMSE_n"], row["MAE_n"], row["R2_n"]]
        vals_closed = vals + [vals[0]]
        cats_closed = categories + [categories[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed, theta=cats_closed, fill="toself",
            name=row["Model"],
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            fillcolor=f"rgba({int(COLORS[i%len(COLORS)][1:3],16)},"
                       f"{int(COLORS[i%len(COLORS)][3:5],16)},"
                       f"{int(COLORS[i%len(COLORS)][5:],16)},0.1)",
        ))
    return fig


def render(df: pd.DataFrame, results: dict) -> None:
    if not results:
        st.markdown("""<div class="empty-state"><div class="icon">🏆</div>
        <h3>No Results Yet</h3>
        <p style="font-size:0.85rem;">Run the forecast first to compare models.</p>
        </div>""", unsafe_allow_html=True)
        return

    target = st.selectbox("📌 Select food item", list(results.keys()), key="cmp_target")
    r          = results.get(target, {})
    metrics_df = r.get("metrics_df")
    preds_df   = r.get("predictions_df")
    trained    = r.get("trained_models", {})
    feat_names = r.get("feature_names", [])

    if metrics_df is None or len(metrics_df) == 0:
        st.warning("No metrics available for this item."); return

    # ── Ranking table ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>🏆 Model Ranking</div>",
                unsafe_allow_html=True)
    rank_df = _ranking_table(metrics_df)
    st.dataframe(
        rank_df.style
            .background_gradient(subset=["RMSE"], cmap="RdYlGn_r")
            .background_gradient(subset=["R2"],   cmap="RdYlGn")
            .format({"RMSE": "{:.3f}", "MAE": "{:.3f}",
                     "MAPE": "{:.2f}%", "R2": "{:.4f}"}),
        use_container_width=True, hide_index=True,
    )

    st.markdown("---")

    # ── Charts ─────────────────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)
    col_l.markdown("<div class='section-title'>📊 RMSE Comparison</div>",
                   unsafe_allow_html=True)
    col_l.plotly_chart(rmse_bar(metrics_df, target, height=330),
                       use_container_width=True)

    radar = _radar_chart(metrics_df, target)
    if radar:
        col_r.markdown("<div class='section-title'>🕸 Radar — Normalised Metrics</div>",
                       unsafe_allow_html=True)
        col_r.plotly_chart(radar, use_container_width=True)
    else:
        col_r.plotly_chart(multi_metric_bar(metrics_df, target, height=330),
                           use_container_width=True)

    # ── Actual vs Predicted (all models) ───────────────────────────────────────
    if preds_df is not None and len(preds_df) > 0:
        st.markdown("---")
        st.markdown("<div class='section-title'>📉 Actual vs Predicted (Test Set)</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(
            actual_vs_predicted(preds_df, target, height=420),
            use_container_width=True,
        )

    # ── Feature Importance ─────────────────────────────────────────────────────
    tree_models = {n: m for n, m in trained.items()
                   if n in ("RandomForest", "XGBoost", "LightGBM", "CatBoost")}
    if tree_models and feat_names:
        st.markdown("---")
        st.markdown("<div class='section-title'>🌿 Feature Importance</div>",
                    unsafe_allow_html=True)
        model_pick = st.selectbox("Select model for feature importance",
                                  list(tree_models.keys()), key="fi_model")
        if model_pick:
            fig = feature_importance_chart(
                tree_models[model_pick], feat_names, model_pick, target, height=420
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance not available for this model.")
