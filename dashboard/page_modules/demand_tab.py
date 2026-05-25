import streamlit as st
import pandas as pd
import plotly.express as px
from dashboard.styles.theme import BG_CARD, BORDER, ACCENT_BLUE, ACCENT_CYAN, ACCENT_PURPLE, TEXT_PRIMARY, TEXT_SECONDARY

def render(df: pd.DataFrame, results: dict):
    st.markdown("### 🔥 Demand Forecasting (Synthetic)")
    
    if "stgnn_demand" not in st.session_state:
        st.info("💡 Enable Advanced ST-GNN mode in the sidebar to generate Demand Forecasts.")
        return
        
    demand_df = st.session_state["stgnn_demand"]["historical"]
    future_demand = st.session_state["stgnn_demand"]["forecast"]
    
    # 1. Historical Trends (Full Width)
    st.markdown("#### Demand Trends Over Time")
    top_foods = demand_df.mean().sort_values(ascending=False).head(5).index
    fig = px.line(demand_df[top_foods], title="Top 5 Foods by Demand Score")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 2. Future Demand Ranking (Next 3 Months)
    st.markdown("#### 🔮 Future Demand Ranking")
    
    # We support the new list of Series format for 3 months, and fallback to single Series
    if isinstance(future_demand, list):
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        colors = [ACCENT_BLUE, ACCENT_CYAN, ACCENT_PURPLE]
        
        for i in range(3):
            if i < len(future_demand):
                with cols[i]:
                    try:
                        # Find the target date
                        target_date = df.index[-1] + pd.DateOffset(months=i+1)
                        month_name = target_date.strftime("%B %Y")
                    except Exception:
                        month_name = f"Month {i+1}"
                        
                    # Styled Bloomberg-style header card
                    st.markdown(f"""
                    <div style="background-color: {BG_CARD}; padding: 12px; border-radius: 8px; border-top: 3px solid {colors[i]}; border-bottom: 1px solid {BORDER}; border-left: 1px solid {BORDER}; border-right: 1px solid {BORDER}; margin-bottom: 12px;">
                        <div style="font-size: 0.72rem; color: {TEXT_SECONDARY}; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 500;">📅 Month {i+1} Forecast</div>
                        <div style="font-family: 'Poppins', sans-serif; font-weight: 700; color: {TEXT_PRIMARY}; font-size: 1.1rem; margin-top: 2px;">{month_name}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    fd_series = future_demand[i]
                    top_future = fd_series.sort_values(ascending=False).head(10)
                    # Clean food names (remove suffix _Demand)
                    top_future.index = [idx.replace("_Demand", "") for idx in top_future.index]
                    st.dataframe(top_future.rename("Predicted Score"), use_container_width=True)
                    
    elif isinstance(future_demand, pd.Series):
        st.caption("Forecasted demand score for next month.")
        top_future = future_demand.sort_values(ascending=False).head(10)
        top_future.index = [idx.replace("_Demand", "") for idx in top_future.index]
        st.dataframe(top_future.rename("Predicted Score"), use_container_width=True)
        
    st.markdown("---")
    st.markdown("#### Explore Demand by Item")
    selected_item = st.selectbox("Select Food Item", options=[c.replace("_Demand", "") for c in demand_df.columns])
    
    if selected_item:
        demand_col = f"{selected_item}_Demand"
        item_fig = px.area(demand_df, y=demand_col, title=f"Historical Demand: {selected_item}")
        item_fig.update_layout(height=300)
        st.plotly_chart(item_fig, use_container_width=True)
