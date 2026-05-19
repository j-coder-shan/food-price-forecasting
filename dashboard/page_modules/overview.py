"""dashboard/page_modules/overview.py — Tab 1: Economic Overview."""
import pandas as pd
import streamlit as st
from dashboard.components.charts import (
    line_chart, correlation_heatmap, economic_indicator_chart
)
from dashboard.styles.theme import ACCENT_CYAN, ACCENT_AMBER, ACCENT_PURPLE


def render(df: pd.DataFrame, cfg: dict) -> None:
    food_cols = [c for c in df.columns if c not in ("Brent_USD", "USD_LKR", "Index")]
    eco_cols  = [c for c in ("Brent_USD", "USD_LKR") if c in df.columns]

    # ── KPI strip ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Months",   f"{len(df)}")
    c2.metric("Food Items",     f"{len(food_cols)}")
    c3.metric("Date Range",
              f"{df.index.min().strftime('%Y-%m')} → {df.index.max().strftime('%Y-%m')}")

    if "Index" in df.columns:
        latest_inf = df["Index"].pct_change(12).iloc[-1] * 100
        c4.metric("Latest YoY Inflation",
                  f"{latest_inf:.2f}%",
                  delta=f"{latest_inf - df['Index'].pct_change(12).iloc[-2]*100:.2f}%")
    else:
        c4.metric("Economic Layers", f"{len(eco_cols)}")

    st.markdown("---")

    # ── Food price trends ─────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>📊 Food Price Trends (LKR)</div>",
                unsafe_allow_html=True)
    col_ctrl, _ = st.columns([3, 1])
    show_items = col_ctrl.multiselect(
        "Select items to plot", food_cols,
        default=food_cols[:8], key="ov_items",
        help="Choose food items to display on the trend chart.",
    )
    if show_items:
        st.plotly_chart(
            line_chart(df, show_items, "Sri Lanka Monthly Food Prices",
                       yaxis_title="Price (LKR)", height=460),
            use_container_width=True, config={"displayModeBar": True},
        )
    else:
        st.info("Select at least one food item above to see trends.")

    # ── Economic Indicators ────────────────────────────────────────────────────
    if eco_cols:
        st.markdown("---")
        st.markdown("<div class='section-title'>🌐 Economic Indicators</div>",
                    unsafe_allow_html=True)
        col_l, col_r = st.columns(2)

        if "Brent_USD" in df.columns:
            col_l.plotly_chart(
                economic_indicator_chart(df, "Brent_USD",
                                         "Brent Crude Oil (USD/barrel)",
                                         color=ACCENT_AMBER, height=300),
                use_container_width=True,
            )
            # Mini stats
            b = df["Brent_USD"]
            col_l.markdown(
                f"<small style='color:#94A3B8'>Latest: <b>${b.iloc[-1]:.1f}</b> &nbsp;|&nbsp; "
                f"Min: <b>${b.min():.1f}</b> &nbsp;|&nbsp; Max: <b>${b.max():.1f}</b></small>",
                unsafe_allow_html=True,
            )

        if "USD_LKR" in df.columns:
            col_r.plotly_chart(
                economic_indicator_chart(df, "USD_LKR",
                                         "USD/LKR Exchange Rate",
                                         color=ACCENT_PURPLE, height=300),
                use_container_width=True,
            )
            fx = df["USD_LKR"]
            col_r.markdown(
                f"<small style='color:#94A3B8'>Latest: <b>LKR {fx.iloc[-1]:.1f}</b> &nbsp;|&nbsp; "
                f"Min: <b>{fx.min():.1f}</b> &nbsp;|&nbsp; Max: <b>{fx.max():.1f}</b></small>",
                unsafe_allow_html=True,
            )

    # ── Correlation Heatmap ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-title'>🔗 Price Correlation Matrix</div>",
                unsafe_allow_html=True)
    corr_items = st.multiselect(
        "Items for correlation (recommend 10–20)",
        food_cols, default=food_cols[:16], key="ov_corr",
    )
    if len(corr_items) >= 2:
        short = {c: c[:18] for c in corr_items}
        corr  = df[corr_items].corr().rename(columns=short, index=short)
        st.plotly_chart(correlation_heatmap(corr), use_container_width=True)
    else:
        st.info("Select at least 2 items to generate the correlation matrix.")

    # ── Raw data preview ───────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📋 Raw Dataset Preview (first 30 rows)"):
        st.dataframe(df.head(30), use_container_width=True)
        st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
