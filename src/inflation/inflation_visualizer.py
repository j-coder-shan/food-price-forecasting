"""inflation_visualizer.py — Plotly visualization builders for dynamic inflation analysis."""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dashboard.styles.theme import (
    get_chart_layout, compact_layout, COLORS,
    ACCENT_BLUE, ACCENT_CYAN, ACCENT_GREEN, ACCENT_AMBER, ACCENT_RED,
    TEXT_PRIMARY, TEXT_SECONDARY, BORDER, BG_CARD, BG_SECONDARY
)
from src.inflation.food_index_calculator import FoodIndexCalculator

def plot_index_comparison(
    hist_index: pd.Series,
    fc_index_df: pd.DataFrame,
    model_name: str,
    horizon: int,
    height: int = 420
) -> go.Figure:
    """Plots historical and forecasted Food Price Index.
    
    Args:
        hist_index: pd.Series with DatetimeIndex of historical index.
        fc_index_df: pd.DataFrame with 'Month' and 'Index (Forecast)' columns.
        model_name: Name of the forecasting model.
        horizon: Forecast horizon in months.
        height: Chart height.
    """
    fc_dates = pd.to_datetime(fc_index_df["Month"])
    fc_vals = fc_index_df["Index (Forecast)"].values
    
    layout = get_chart_layout(
        title=f"Food Price Index — {horizon}-Month Forecast ({model_name})",
        xaxis_title="Month",
        yaxis_title="Index (Base 2021 = 100)",
        height=height
    )
    fig = go.Figure()
    fig.update_layout(**layout)
    
    # Historical index
    fig.add_trace(go.Scatter(
        x=hist_index.index,
        y=hist_index.values,
        name="Historical Index",
        line=dict(color=ACCENT_CYAN, width=2.2),
        hovertemplate="<b>Historical Index</b><br>%{x|%b %Y}: %{y:.2f}<extra></extra>"
    ))
    
    # Forecast index
    fig.add_trace(go.Scatter(
        x=fc_dates,
        y=fc_vals,
        name=f"Forecast ({model_name})",
        line=dict(color=ACCENT_AMBER, width=2.5, dash="dash"),
        hovertemplate="<b>Forecast Index</b><br>%{x|%b %Y}: %{y:.2f}<extra></extra>"
    ))
    
    # Vertical separator
    if len(hist_index) > 0:
        split_x = hist_index.index[-1]
        if isinstance(split_x, pd.Timestamp):
            split_x = split_x.timestamp() * 1000
            
        fig.add_vline(
            x=split_x,
            line_dash="dot",
            line_color=TEXT_SECONDARY,
            line_width=1,
            opacity=0.6,
            annotation_text="Forecast start",
            annotation_font_color=TEXT_SECONDARY,
            annotation_font_size=10
        )
        
    return fig

def plot_inflation_comparison(
    hist_inflation: pd.DataFrame,
    fc_inflation: pd.DataFrame,
    model_name: str,
    horizon: int,
    height: int = 420
) -> go.Figure:
    """Plots historical and forecasted inflation rates (MoM and YoY).
    
    Args:
        hist_inflation: pd.DataFrame containing historical inflation ('Monthly_Inflation_%', 'YoY_Inflation_%').
        fc_inflation: pd.DataFrame containing forecasted inflation ('Monthly_Inflation_%', 'YoY_Inflation_%') with 'Month'.
        model_name: Name of the forecasting model.
        horizon: Forecast horizon in months.
        height: Chart height.
    """
    fc_dates = pd.to_datetime(fc_inflation["Month"])
    
    layout = get_chart_layout(
        title=f"Food Price Inflation Forecast — MoM & YoY ({model_name})",
        xaxis_title="Month",
        yaxis_title="Inflation Rate (%)",
        height=height
    )
    fig = go.Figure()
    fig.update_layout(**layout)
    
    # Historical YoY
    if "YoY_Inflation_%" in hist_inflation.columns:
        fig.add_trace(go.Scatter(
            x=hist_inflation.index,
            y=hist_inflation["YoY_Inflation_%"],
            name="Hist YoY Inflation",
            line=dict(color=ACCENT_GREEN, width=2),
            hovertemplate="<b>Historical YoY</b><br>%{x|%b %Y}: %{y:.2f}%<extra></extra>"
        ))
        
    # Historical MoM
    if "Monthly_Inflation_%" in hist_inflation.columns:
        fig.add_trace(go.Scatter(
            x=hist_inflation.index,
            y=hist_inflation["Monthly_Inflation_%"],
            name="Hist MoM Inflation",
            line=dict(color=ACCENT_BLUE, width=1.5),
            opacity=0.7,
            hovertemplate="<b>Historical MoM</b><br>%{x|%b %Y}: %{y:.2f}%<extra></extra>"
        ))
        
    # Forecast YoY
    if "YoY_Inflation_%" in fc_inflation.columns:
        fig.add_trace(go.Scatter(
            x=fc_dates,
            y=fc_inflation["YoY_Inflation_%"],
            name="Forecast YoY",
            line=dict(color=ACCENT_AMBER, width=2.5, dash="dash"),
            hovertemplate="<b>Forecast YoY</b><br>%{x|%b %Y}: %{y:.2f}%<extra></extra>"
        ))
        
    # Forecast MoM
    if "Monthly_Inflation_%" in fc_inflation.columns:
        fig.add_trace(go.Scatter(
            x=fc_dates,
            y=fc_inflation["Monthly_Inflation_%"],
            name="Forecast MoM",
            line=dict(color=ACCENT_CYAN, width=1.8, dash="dot"),
            hovertemplate="<b>Forecast MoM</b><br>%{x|%b %Y}: %{y:.2f}%<extra></extra>"
        ))
        
    # Horizontal line at 0%
    fig.add_hline(y=0.0, line_dash="dot", line_color=TEXT_SECONDARY, line_width=1, opacity=0.4)
    
    # Vertical separator
    if len(hist_inflation) > 0:
        split_x = hist_inflation.index[-1]
        if isinstance(split_x, pd.Timestamp):
            split_x = split_x.timestamp() * 1000
            
        fig.add_vline(
            x=split_x,
            line_dash="dot",
            line_color=TEXT_SECONDARY,
            line_width=1,
            opacity=0.6,
            annotation_text="Forecast start",
            annotation_font_color=TEXT_SECONDARY,
            annotation_font_size=10
        )
        
    return fig

def plot_contribution_analysis(
    calculator: FoodIndexCalculator,
    results: dict,
    horizon: int,
    top_n: int = 15,
    height: int = 480
) -> go.Figure:
    """Computes food item contribution to changes in the food index.
    
    Contribution_i = Weight_i * (Price_i_forecast_end - Price_i_hist_end) / Price_i_base_2021 * 100
    
    Args:
        calculator: FoodIndexCalculator instance.
        results: Prediction results dictionary.
        horizon: Forecast horizon (months).
        top_n: Number of top contributors to display.
        height: Chart height.
    """
    future_prices = calculator.get_future_prices(results, horizon)
    
    # Last historical prices and end-of-forecast prices
    price_start = calculator.historical_df[calculator.food_items].iloc[-1]
    price_end = future_prices[calculator.food_items].iloc[-1]
    
    # Base prices
    base_prices = calculator.base_prices
    
    # Normalized weights
    weights = pd.Series(calculator.weights)
    
    # Calculate index point contribution: weights * (P_t - P_0) / P_base * 100
    # Change in index contribution = weights * (P_end - P_start) / P_base * 100
    contrib = weights * (price_end - price_start) / base_prices * 100
    
    # Create DataFrame for analysis
    df = pd.DataFrame({
        "Item": contrib.index,
        "Weight": weights[contrib.index],
        "Price_Start": price_start[contrib.index],
        "Price_End": price_end[contrib.index],
        "Price_Change_Pct": ((price_end - price_start) / price_start * 100).round(2),
        "Contribution": contrib.values.round(4)
    })
    
    # Remove items with virtually 0 contribution
    df = df[df["Contribution"].abs() > 0.0001]
    
    # Sort by absolute contribution and take top N
    df["Abs_Contribution"] = df["Contribution"].abs()
    df_sorted = df.sort_values("Abs_Contribution", ascending=False).head(top_n)
    
    # Sort again so largest contribution is on top in the horizontal bar chart
    df_sorted = df_sorted.sort_values("Contribution", ascending=True)
    
    # Map colors: positive contribution (raising cost of living) -> Red/Amber, negative -> Green
    colors = [ACCENT_RED if val >= 0 else ACCENT_GREEN for val in df_sorted["Contribution"]]
    
    layout = compact_layout(
        title=f"Top {top_n} Food Items Driving Food Index Change ({horizon}m Horizon)",
        xaxis_title="Index Points Contribution",
        yaxis_title="",
        height=height
    )
    
    fig = go.Figure()
    fig.update_layout(**layout)
    
    fig.add_trace(go.Bar(
        x=df_sorted["Contribution"],
        y=df_sorted["Item"],
        orientation="h",
        marker_color=colors,
        customdata=np.stack((
            df_sorted["Weight"] * 100,
            df_sorted["Price_Start"],
            df_sorted["Price_End"],
            df_sorted["Price_Change_Pct"]
        ), axis=-1),
        hovertemplate=(
            "<b>%{y}</b><br>" +
            "Index Contribution: %{x:+.3f} pts<br>" +
            "Basket Weight: %{customdata[0]:.2f}%<br>" +
            "Start Price: LKR %{customdata[1]:,.2f}<br>" +
            "End Price: LKR %{customdata[2]:,.2f}<br>" +
            "Price Change: %{customdata[3]:+,.2f}%<extra></extra>"
        )
    ))
    
    return fig
