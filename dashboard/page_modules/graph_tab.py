import streamlit as st
import pandas as pd

def render(results: dict):
    st.markdown("### 🕸 Economic Graph Analysis")
    
    if "stgnn_graph" not in st.session_state:
        st.info("💡 Enable Advanced ST-GNN mode in the sidebar to construct and analyze the Economic Graph.")
        return
        
    st.markdown("This interactive graph displays the relationships learned by the Spatio-Temporal Graph Neural Network. Nodes represent economic metrics and food items, while edges represent correlations and domain-based influences.")
    
    fig = st.session_state["stgnn_graph"]
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    #### Legend
    - <span style='color:#ef4444; font-weight:bold;'>Red Nodes</span>: Macroeconomic indicators (Fuel, Exchange Rate, Inflation)
    - <span style='color:#eab308; font-weight:bold;'>Yellow Nodes</span>: Demand Scores
    - <span style='color:#3b82f6; font-weight:bold;'>Blue Nodes</span>: Food Prices
    """, unsafe_allow_html=True)
