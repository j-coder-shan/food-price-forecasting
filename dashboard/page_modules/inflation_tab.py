"""dashboard/page_modules/inflation_tab.py — Tab 3: Inflation Analysis."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard.components.charts import inflation_chart, economic_indicator_chart
from dashboard.styles.theme import (
    get_chart_layout, compact_layout,
    ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER, ACCENT_CYAN, ACCENT_BLUE,
    BG_CARD, TEXT_PRIMARY, TEXT_SECONDARY, BORDER,
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
        st.markdown("<div class='section-title'>🔮 Forecasted Inflation</div>",
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

            from src.inflation import InflationCalculator as IC
            fc_inf = IC.forecast_inflation(fc_series, last_known, hist_series)
            fc_inf_clean = fc_inf.copy()
            fc_inf_clean["Monthly_Inflation_%"] = fc_inf_clean["Monthly_Inflation_%"].round(3)
            fc_inf_clean["YoY_Inflation_%"]     = fc_inf_clean["YoY_Inflation_%"].round(3)

            # Stitched inflation chart
            layout = compact_layout(
                title=f"Historical + {h}m Forecasted Monthly Inflation (%)",
                xaxis_title="Month", yaxis_title="Inflation (%)", height=360,
            )
            fig = go.Figure()
            fig.update_layout(**layout)
            fig.add_trace(go.Scatter(
                x=mom.index, y=mom.values, name="Historical",
                line=dict(color=ACCENT_GREEN, width=2), fill="tozeroy",
                fillcolor="rgba(16,185,129,0.06)",
                hovertemplate="%{x|%b %Y}: %{y:.2f}%<extra>Historical</extra>",
            ))
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(fc_inf["Month"]),
                y=fc_inf["Monthly_Inflation_%"],
                name=f"{h}m Forecast",
                line=dict(color=ACCENT_AMBER, width=2.5, dash="dash"),
                hovertemplate="%{x|%b %Y}: %{y:.2f}%<extra>Forecast</extra>",
            ))
            fig.add_hline(y=0, line_dash="dot", line_color=TEXT_SECONDARY,
                          line_width=1, opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)

            # Table + download
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
