"""dashboard/components/charts.py — Reusable Plotly chart builders (Bloomberg dark theme)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dashboard.styles.theme import (
    get_chart_layout, compact_layout, COLORS,
    ACCENT_BLUE, ACCENT_CYAN, ACCENT_GREEN, ACCENT_AMBER, ACCENT_RED,
    BG_CARD, BG_SECONDARY, TEXT_PRIMARY, TEXT_SECONDARY, BORDER,
)


def _fig(layout: dict) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**layout)
    return fig


def line_chart(df: pd.DataFrame, cols: list[str], title: str,
               yaxis_title: str = "Price (LKR)", height: int = 480) -> go.Figure:
    fig = _fig(get_chart_layout(title=title, xaxis_title="Month",
                                yaxis_title=yaxis_title, height=height))
    for i, col in enumerate(cols):
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], mode="lines", name=col,
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                hovertemplate=f"<b>{col}</b><br>%{{x|%b %Y}}: LKR %{{y:,.2f}}<extra></extra>",
            ))
    return fig


def actual_vs_predicted(preds_df: pd.DataFrame, target: str,
                         height: int = 400) -> go.Figure:
    layout = get_chart_layout(
        title=f"Actual vs Predicted — {target} (20% Test Set)",
        xaxis_title="Month", yaxis_title="Price (LKR)", height=height,
        show_rangeslider=False,
    )
    fig = _fig(layout)
    fig.add_trace(go.Scatter(
        x=preds_df.index, y=preds_df["Actual"],
        name="Actual", line=dict(color="white", width=2.5),
        hovertemplate="<b>Actual</b>: LKR %{y:,.2f}<extra></extra>",
    ))
    for i, col in enumerate([c for c in preds_df.columns if c != "Actual"]):
        fig.add_trace(go.Scatter(
            x=preds_df.index, y=preds_df[col], name=col,
            line=dict(color=COLORS[i % len(COLORS)], width=1.8, dash="dash"),
            hovertemplate=f"<b>{col}</b>: LKR %{{y:,.2f}}<extra></extra>",
        ))
    return fig


def forecast_chart(hist: pd.Series, fc_df: pd.DataFrame, fc_col: str,
                   model_name: str, horizon: int, height: int = 480) -> go.Figure:
    fc_dates = pd.to_datetime(fc_df["Month"])
    fc_vals  = fc_df[fc_col].values
    band     = fc_vals * 0.08  # ±8% confidence band

    item_name = fc_col.split(" (Forecast)")[0]
    layout = get_chart_layout(
        title=f"{item_name} — {horizon}-Month Forecast  ({model_name})",
        xaxis_title="Month", yaxis_title="Price (LKR)", height=height,
    )
    fig = _fig(layout)

    # Historical
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values, name="Historical",
        line=dict(color=ACCENT_CYAN, width=2.2),
        hovertemplate="<b>Historical</b><br>%{x|%b %Y}: LKR %{y:,.2f}<extra></extra>",
    ))
    # Confidence band
    x_band = list(fc_dates) + list(reversed(fc_dates))
    y_band = list(fc_vals + band) + list(reversed(fc_vals - band))
    fig.add_trace(go.Scatter(
        x=x_band, y=y_band, fill="toself", name="±8% band",
        fillcolor="rgba(59,130,246,0.10)", line=dict(color="rgba(0,0,0,0)"),
        showlegend=True, hoverinfo="skip",
    ))
    # Forecast line
    fig.add_trace(go.Scatter(
        x=fc_dates, y=fc_vals, name=f"Forecast ({model_name})",
        line=dict(color=ACCENT_AMBER, width=2.5, dash="dash"),
        hovertemplate="<b>Forecast</b><br>%{x|%b %Y}: LKR %{y:,.2f}<extra></extra>",
    ))
    # Vertical separator
    if len(hist) > 0:
        split_x = hist.index[-1]
        if isinstance(split_x, pd.Timestamp):
            split_x = split_x.timestamp() * 1000
            
        fig.add_vline(x=split_x, line_dash="dot", line_color=TEXT_SECONDARY,
                      line_width=1, opacity=0.6,
                      annotation_text="Forecast start",
                      annotation_font_color=TEXT_SECONDARY,
                      annotation_font_size=10)
    return fig


def rmse_bar(metrics_df: pd.DataFrame, target: str, height: int = 360) -> go.Figure:
    df = metrics_df.sort_values("RMSE").copy()
    colors = [ACCENT_GREEN if i == 0 else ACCENT_BLUE for i in range(len(df))]
    layout = compact_layout(title=f"RMSE by Model — {target}",
                            xaxis_title="Model", yaxis_title="RMSE (LKR)", height=height)
    fig = _fig(layout)
    fig.add_trace(go.Bar(
        x=df["Model"], y=df["RMSE"],
        marker_color=colors,
        text=df["RMSE"].round(2),
        textposition="outside",
        textfont=dict(color=TEXT_PRIMARY, size=11),
        hovertemplate="<b>%{x}</b><br>RMSE: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(showlegend=False)
    return fig


def multi_metric_bar(metrics_df: pd.DataFrame, target: str, height=380) -> go.Figure:
    """Side-by-side bars for RMSE, MAE, MAPE."""
    df = metrics_df.sort_values("RMSE").copy()
    layout = compact_layout(title=f"Model Metrics Comparison — {target}",
                            xaxis_title="Model", yaxis_title="Value", height=height)
    fig = _fig(layout)
    for metric, color in [("RMSE", ACCENT_RED), ("MAE", ACCENT_AMBER), ("MAPE", ACCENT_CYAN)]:
        if metric in df.columns:
            fig.add_trace(go.Bar(
                name=metric, x=df["Model"], y=df[metric], marker_color=color,
                hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.3f}}<extra></extra>",
            ))
    fig.update_layout(barmode="group")
    return fig


def correlation_heatmap(corr: pd.DataFrame, height: int = 540) -> go.Figure:
    layout = compact_layout(title="Food Price Correlation Matrix", height=height)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu_r", zmin=-1, zmax=1,
        colorbar=dict(title="r", tickfont=dict(color=TEXT_SECONDARY), len=0.8),
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(**layout)
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9))
    return fig


def inflation_chart(series: pd.Series, title: str, color: str = ACCENT_GREEN,
                    height: int = 320, ref_line: float = 0.0) -> go.Figure:
    vals = series.dropna()
    pos = vals.clip(lower=0)
    neg = vals.clip(upper=0)
    layout = compact_layout(title=title, xaxis_title="Month",
                            yaxis_title="Inflation (%)", height=height)
    fig = _fig(layout)
    fig.add_trace(go.Scatter(
        x=vals.index, y=vals.values, mode="lines",
        name=series.name or "Inflation",
        line=dict(color=color, width=2),
        fill="tozeroy", fillcolor=f"rgba(16,185,129,0.07)",
        hovertemplate="%{x|%b %Y}: %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=ref_line, line_dash="dot", line_color=TEXT_SECONDARY,
                  line_width=1, opacity=0.6)
    # Color zones
    fig.add_hrect(y0=3, y1=100, fillcolor="rgba(239,68,68,0.04)",
                  line_width=0, annotation_text="High", annotation_position="top right",
                  annotation_font_color=ACCENT_RED, annotation_font_size=9)
    return fig


def economic_indicator_chart(df: pd.DataFrame, col: str, title: str,
                              color: str = ACCENT_AMBER, height: int = 280) -> go.Figure:
    layout = compact_layout(title=title, xaxis_title="Month",
                            yaxis_title=col, height=height, show_rangeslider=False)
    fig = _fig(layout)
    if col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], mode="lines", name=col,
            line=dict(color=color, width=2),
            fill="tozeroy", fillcolor=f"rgba(245,158,11,0.06)",
            hovertemplate=f"%{{x|%b %Y}}: %{{y:,.2f}}<extra></extra>",
        ))
    return fig


def feature_importance_chart(model, feature_names: list[str],
                              model_name: str, target: str,
                              height: int = 420) -> go.Figure | None:
    importance = None
    try:
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "get_feature_importance"):
            importance = model.get_feature_importance()
    except Exception:
        return None
    if importance is None or len(importance) != len(feature_names):
        return None

    pairs = sorted(zip(feature_names, importance), key=lambda x: x[1])[-20:]
    names, vals = zip(*pairs)
    cmap = [ACCENT_BLUE if v < max(vals) * 0.5 else ACCENT_CYAN for v in vals]

    layout = compact_layout(title=f"Feature Importance — {model_name}",
                            xaxis_title="Importance Score", height=height)
    fig = _fig(layout)
    fig.add_trace(go.Bar(
        x=list(vals), y=list(names), orientation="h",
        marker_color=cmap,
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    return fig
