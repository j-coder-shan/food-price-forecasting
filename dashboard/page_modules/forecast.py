"""dashboard/page_modules/forecast.py — Tab 2: Price Forecast."""
import pandas as pd
import streamlit as st
from dashboard.components.charts import actual_vs_predicted, forecast_chart
from dashboard.styles.theme import ACCENT_GREEN, ACCENT_RED, BG_CARD, BORDER


def _empty_state():
    st.markdown("""
    <div class="empty-state">
        <div class="icon">🔮</div>
        <h3>No Forecasts Yet</h3>
        <p style="font-size:0.85rem;">Select food items and models in the sidebar,<br>
        then click <b>Generate Forecast</b> to get started.</p>
    </div>""", unsafe_allow_html=True)


def render(df: pd.DataFrame, cfg: dict, results: dict) -> None:
    if not results:
        _empty_state(); return

    horizon = cfg.get("horizon", 12)
    targets = list(results.keys())

    # ── Item selector ──────────────────────────────────────────────────────────
    col_sel, col_info = st.columns([2, 1])
    target = col_sel.selectbox("📌 Select food item", targets, key="fc_target")
    r = results.get(target, {})

    if not r:
        st.warning(f"No results found for **{target}**."); return

    best_name  = r.get("best_model", "N/A")
    metrics_df = r.get("metrics_df")
    preds_df   = r.get("predictions_df")
    forecasts  = r.get("forecasts", {})
    fc         = forecasts.get(horizon)

    # ── KPI row ────────────────────────────────────────────────────────────────
    if metrics_df is not None and len(metrics_df) > 0:
        best_row = metrics_df.sort_values("RMSE").iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Best Model", best_name)
        c2.metric("RMSE (LKR)", f"{best_row['RMSE']:.2f}")
        c3.metric("MAE (LKR)",  f"{best_row['MAE']:.2f}")
        c4.metric("MAPE",       f"{best_row.get('MAPE', 0):.2f}%")
        c5.metric("R²",         f"{best_row.get('R2', 0):.4f}")

    st.markdown("---")

    # ── Actual vs Predicted ────────────────────────────────────────────────────
    if preds_df is not None and len(preds_df) > 0:
        st.markdown("<div class='section-title'>📊 Actual vs Predicted (Test Set 20%)</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(
            actual_vs_predicted(preds_df, target, height=380),
            use_container_width=True,
        )

    # ── Future Forecast ────────────────────────────────────────────────────────
    st.markdown(f"<div class='section-title'>🔮 {horizon}-Month Future Forecast</div>",
                unsafe_allow_html=True)

    # Horizon selector (available horizons from results)
    available_h = sorted(forecasts.keys())
    if len(available_h) > 1:
        chosen_h = st.radio(
            "View horizon", available_h,
            format_func=lambda h: f"{h} months",
            horizontal=True, index=available_h.index(horizon) if horizon in available_h else 0,
            key="fc_horizon_pick",
        )
        fc = forecasts.get(chosen_h)
        horizon = chosen_h

    fc_col = f"{target} (Forecast)"
    if fc is not None and fc_col in fc.columns:
        hist = df[target].dropna().tail(36)  # show last 3 years for context
        st.plotly_chart(
            forecast_chart(hist, fc, fc_col, best_name, horizon, height=460),
            use_container_width=True, config={"displayModeBar": True},
        )

        # Forecast table
        st.markdown("<div class='section-title'>📋 Forecast Values</div>",
                    unsafe_allow_html=True)
        st.dataframe(
            fc.style.format({fc_col: "LKR {:.2f}"}),
            use_container_width=True, hide_index=True,
        )

        # Downloads
        col_d1, col_d2 = st.columns(2)
        csv = fc.to_csv(index=False).encode("utf-8")
        col_d1.download_button(
            f"⬇️ Forecast CSV ({horizon}m)",
            data=csv,
            file_name=f"{target[:30].replace(' ','_')}_{best_name}_{horizon}m.csv",
            mime="text/csv",
        )
    else:
        st.info(f"Forecast not available for **{target}** at **{horizon}m** horizon.")

    # ── All models metrics ─────────────────────────────────────────────────────
    if metrics_df is not None and len(metrics_df) > 0:
        with st.expander("📊 All Models — Evaluation Metrics"):
            styled = metrics_df.style\
                .background_gradient(subset=["RMSE"], cmap="RdYlGn_r")\
                .background_gradient(subset=["R2"],   cmap="RdYlGn")\
                .format({"RMSE": "{:.3f}", "MAE": "{:.3f}",
                         "MAPE": "{:.2f}%", "R2": "{:.4f}"})
            st.dataframe(styled, use_container_width=True, hide_index=True)
