import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard.components.charts import inflation_chart, economic_indicator_chart
from dashboard.styles.theme import (
    get_chart_layout, compact_layout,
    ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER, ACCENT_CYAN, ACCENT_BLUE,
    BG_CARD, TEXT_PRIMARY, TEXT_SECONDARY, BORDER,
)
from src.inflation import (
    InflationCalculator,
    FoodIndexCalculator,
    plot_index_comparison,
    plot_inflation_comparison,
    plot_contribution_analysis,
)


def _inflation_gauge(value: float, label: str) -> go.Figure:
    """Renders a simple gauge for current inflation."""
    color = ACCENT_GREEN if value < 3 else (ACCENT_AMBER if value < 8 else ACCENT_RED)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        number=dict(suffix="%", font=dict(size=26, color=TEXT_PRIMARY, family="Poppins")),
        title=dict(text=label, font=dict(size=13, color=TEXT_SECONDARY)),
        gauge=dict(
            axis=dict(range=[-5, 30], tickcolor=TEXT_SECONDARY,
                      tickfont=dict(color=TEXT_SECONDARY)),
            bar=dict(color=color, thickness=0.25),
            bgcolor=BG_CARD,
            bordercolor=BORDER, borderwidth=1,
            steps=[
                dict(range=[-5, 3],  color="rgba(16,185,129,0.12)"),
                dict(range=[3,  8],  color="rgba(245,158,11,0.12)"),
                dict(range=[8,  30], color="rgba(239,68,68,0.12)"),
            ],
            threshold=dict(line=dict(color="white", width=2), thickness=0.6, value=value),
        ),
    ))
    fig.update_layout(
        paper_bgcolor=BG_CARD, font=dict(color=TEXT_PRIMARY),
        margin=dict(l=20, r=20, t=40, b=10), height=240,
    )
    return fig


def render(df: pd.DataFrame, results: dict) -> None:
    if "Index" not in df.columns:
        st.warning("**No 'Index' column** in the dataset — inflation cannot be computed.")
        st.info("Ensure your food_prices.xlsx has an 'Index' column representing the composite food price index.")
        return

    from src.inflation import InflationCalculator
    calc       = InflationCalculator(df["Index"])
    hist_table = calc.inflation_table()
    summary    = calc.summary()

    # ── KPI row ────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Monthly Inflation",
              f"{summary['latest_monthly']}%",
              delta=f"{summary['latest_monthly'] - summary['avg_monthly_inflation']:.2f}% vs avg")
    c2.metric("Avg Monthly Inflation", f"{summary['avg_monthly_inflation']}%")
    c3.metric("Latest YoY Inflation",
              f"{summary['latest_yoy']}%",
              delta=f"{summary['latest_yoy'] - summary['avg_yoy_inflation']:.2f}% vs avg")
    c4.metric("Avg YoY Inflation", f"{summary['avg_yoy_inflation']}%")

    st.markdown("---")

    # ── Gauges ─────────────────────────────────────────────────────────────────
    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(_inflation_gauge(
            summary['latest_monthly'] or 0, "Current Monthly Inflation"
        ), use_container_width=True)
    with g2:
        st.plotly_chart(_inflation_gauge(
            summary['latest_yoy'] or 0, "Current YoY Inflation"
        ), use_container_width=True)
    with g3:
        st.plotly_chart(_inflation_gauge(
            summary['max_yoy_inflation'] or 0, "Peak YoY Inflation (Historical)"
        ), use_container_width=True)

    st.markdown("---")

    # ── Food Price Index Chart ──────────────────────────────────────────────────
    st.markdown("<div class='section-title'>📈 Food Price Index (Historical)</div>",
                unsafe_allow_html=True)
    st.plotly_chart(
        economic_indicator_chart(df, "Index", "Sri Lanka Food Price Index",
                                 color=ACCENT_CYAN, height=300),
        use_container_width=True,
    )

    # ── Inflation charts ────────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)
    mom = hist_table["Monthly_Inflation_%"].dropna()
    yoy = hist_table["YoY_Inflation_%"].dropna()

    col_l.markdown("<div class='section-title'>📉 Month-over-Month</div>",
                   unsafe_allow_html=True)
    col_l.plotly_chart(
        inflation_chart(mom, "Month-over-Month Inflation (%)",
                        color=ACCENT_GREEN, height=300),
        use_container_width=True,
    )

    col_r.markdown("<div class='section-title'>📉 Year-over-Year</div>",
                   unsafe_allow_html=True)
    col_r.plotly_chart(
        inflation_chart(yoy, "Year-over-Year Inflation (%)",
                        color=ACCENT_AMBER, height=300),
        use_container_width=True,
    )

    # ── Forecasted Inflation ───────────────────────────────────────────────────
    if results and "Index" in results:
        st.markdown("---")
        st.markdown("<div class='section-title'>🔮 Forecasted Inflation Analysis</div>",
                    unsafe_allow_html=True)
        r = results["Index"]
        fc_dict = r.get("forecasts", {})
        h = max(fc_dict.keys()) if fc_dict else None

        if h and f"Index (Forecast)" in fc_dict[h].columns:
            fc_df = fc_dict[h]
            fc_series = pd.Series(
                fc_df["Index (Forecast)"].values,
                index=pd.to_datetime(fc_df["Month"]),
                name="Index",
            )
            last_known  = float(df["Index"].dropna().iloc[-1])
            hist_series = df["Index"].dropna()

            fc_inf = InflationCalculator.forecast_inflation(fc_series, last_known, hist_series)
            fc_inf_clean = fc_inf.copy()
            fc_inf_clean["Monthly_Inflation_%"] = fc_inf_clean["Monthly_Inflation_%"].round(3)
            fc_inf_clean["YoY_Inflation_%"]     = fc_inf_clean["YoY_Inflation_%"].round(3)

            # ── Projected KPI Metrics ───────────────────────────────────────────
            c_p1, c_p2, c_p3, c_p4 = st.columns(4)
            c_p1.metric("Projected YoY (End)", f"{fc_inf['YoY_Inflation_%'].iloc[-1]:.2f}%")
            c_p2.metric("Projected MoM (End)", f"{fc_inf['Monthly_Inflation_%'].iloc[-1]:.2f}%")
            c_p3.metric("Peak Projected YoY", f"{fc_inf['YoY_Inflation_%'].max():.2f}%")
            c_p4.metric("Avg Projected YoY", f"{fc_inf['YoY_Inflation_%'].mean():.2f}%")

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            # ── Multi-column charts ─────────────────────────────────────────────
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    plot_index_comparison(
                        hist_series.tail(36),
                        fc_df,
                        r.get("best_model", "Dynamic Basket"),
                        h,
                        height=360
                    ),
                    use_container_width=True
                )
            with col2:
                st.plotly_chart(
                    plot_inflation_comparison(
                        hist_table.tail(36),
                        fc_inf,
                        r.get("best_model", "Dynamic Basket"),
                        h,
                        height=360
                    ),
                    use_container_width=True
                )

            # ── Contribution Analysis Card ──────────────────────────────────────
            st.markdown("<div class='section-title'>🍽️ Food Contribution Analysis</div>",
                        unsafe_allow_html=True)
            st.markdown(
                "<p style='font-size:0.82rem;color:#64748B;margin-top:-6px;margin-bottom:12px;'>"
                "This chart shows the index points contribution of individual food items to the projected food price index change over the forecast horizon. "
                "Price increases (positive contribution) push index up (Red), while price decreases (negative contribution) pull it down (Green)."
                "</p>",
                unsafe_allow_html=True
            )
            
            try:
                calc_inst = FoodIndexCalculator(df)
                fig_contrib = plot_contribution_analysis(calc_inst, results, h, top_n=15, height=440)
                st.plotly_chart(fig_contrib, use_container_width=True)
            except Exception as ce:
                st.error(f"Failed to generate contribution analysis: {ce}")

            # ── Forecast Values Table ───────────────────────────────────────────
            st.markdown("<div class='section-title'>📋 Forecasted Index & Inflation Table</div>",
                        unsafe_allow_html=True)
            st.dataframe(
                fc_inf_clean.style.background_gradient(
                    subset=["Monthly_Inflation_%"], cmap="RdYlGn_r"
                ),
                use_container_width=True, hide_index=True,
            )
            st.download_button(
                "⬇️ Download Inflation Forecast CSV",
                data=fc_inf_clean.to_csv(index=False).encode("utf-8"),
                file_name=f"inflation_forecast_{h}m.csv", mime="text/csv",
                use_container_width=True,
            )
    else:
        st.info("Include **Index** in the selected targets and run forecast to see projected inflation.")

    # ── Historical table ───────────────────────────────────────────────────────
    with st.expander("📋 Full Historical Inflation Table"):
        disp = hist_table.copy().reset_index()
        disp["Month"] = disp["Month"].dt.strftime("%Y-%m")
        st.dataframe(
            disp.style.background_gradient(subset=["YoY_Inflation_%"], cmap="RdYlGn_r"),
            use_container_width=True, hide_index=True,
        )
        st.download_button(
            "⬇️ Download Historical Inflation CSV",
            data=disp.to_csv(index=False).encode("utf-8"),
            file_name="historical_inflation.csv", mime="text/csv",
        )
