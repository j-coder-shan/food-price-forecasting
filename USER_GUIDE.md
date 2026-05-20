# 📖 User Guide: Sri Lanka Food Price & Inflation Forecasting Platform

Welcome to the **Sri Lanka Food Price & Inflation** platform. This dashboard is an advanced, production-grade analytical tool designed to help researchers, economists, and policymakers analyze historical food price trends, evaluate sophisticated machine learning models, and forecast future market behavior.

This guide provides a detailed walkthrough of the platform's layout, visual components, and evaluation metrics to help you interpret the data accurately.

---

### 1. General Dashboard Layout & Sidebar

The sidebar on the left acts as the control center for your analytical pipeline. 

* **Dataset Sources & Item Selection:** 
  The platform allows you to select specific commodities from the Sri Lankan food basket (e.g., *Thalapath*, *Ulundu flour*, *Rice*). You can analyze a single item in isolation or process multiple items simultaneously. You can also define the **Forecast Horizon**, allowing the AI to predict market prices for the next **3, 6, or 12 months**.
* **Model Selection Checklist:** 
  Rather than relying on a single algorithm, the platform simultaneously trains and compares up to **8 diverse forecasting models** to find the most accurate fit for your selected data. These include:
  * **Baseline Statistical Models:** Linear Regression, ARIMA, and SARIMA (excellent for capturing stable, seasonal trends).
  * **Tree-Based Machine Learning:** Random Forest, XGBoost, LightGBM, and CatBoost (powerful algorithms capable of capturing complex, non-linear market shocks).
  * **Deep Learning:** Spatio-Temporal Graph Neural Networks (ST-GNN) (an advanced architecture that treats macroeconomic factors like fuel and exchange rates as interconnected economic nodes).

---

### 2. Single-Item Model Comparison Section (Individual Analytics)

When viewing the analytics for a specific food item, the dashboard provides a suite of visual tools to help you determine which model learned the underlying price dynamics best.

#### 📊 Model Ranking Table
This table ranks all trained models based on their forecasting accuracy against a held-out test dataset. The rows represent the different algorithms, while the columns break down the specific mathematical errors.
* **MAE (Mean Absolute Error):** This measures how much the model's prediction misses the actual price *on average*, measured in Sri Lankan Rupees (LKR). For example, an MAE of 12.5 means the forecast is usually off by about LKR 12.50.
* **RMSE (Root Mean Squared Error):** Similar to MAE, but the mathematics of RMSE penalize larger forecasting errors more heavily. This is a crucial metric for food security analysis, as a model with a low RMSE is proven to be highly stable and less likely to miss sudden, volatile price spikes.
* **MAPE (Mean Absolute Percentage Error):** The average relative percentage error. If a model has a MAPE of 1.92%, it means its forecasts are, on average, 98.08% accurate relative to the true price scale.
* **R² (Coefficient of Determination):** This explains how much of the actual price variance is successfully captured by the model. 
  * `1.0` is a perfect score (flawless prediction).
  * `Positive values` (e.g., 0.85) mean the model is highly capable of following the actual price curve.
  * `Negative values` indicate catastrophic failure—meaning the model performs *worse* than simply drawing a flat, horizontal line at the historical average.
* **Color Coding (Heatmap):** The table uses a conditional color styler. Dark green signifies highly accurate, reliable models, while red highlights trailing, unstable, or wildly inaccurate models.

#### 📉 RMSE Comparison Bar Chart
This horizontal bar chart provides a rapid visual benchmark of error magnitudes across all models. **Shorter bars represent better, tighter forecasts.** It allows you to see at a glance if your deep learning models are significantly outperforming your baseline statistical models.

#### 🕸️ Model Radar Chart (Normalised Metrics)
A radar chart normalizes the different evaluation metrics (scaling them between 0 and 1) to visualize a model's holistic "footprint." 
Because lower errors are better, **a smaller, tighter web area typically represents a superior model.** It helps you assess balance—for instance, checking if a model has an excellent R² score but suffers from a dangerously high RMSE due to a few extreme outlier errors.

---

### 3. Overall / Macro Model Comparison Section (Multi-Item Analytics)

If you have selected multiple food items in the sidebar, scrolling to the bottom of the Model Comparison tab reveals the **Macro-Aggregation View**.

* **Overall Model Performance Table:** Instead of looking at a single commodity, this table calculates the mathematical **macro-average** of all metrics across *all selected food items combined*. For a researcher or policymaker, this is invaluable: it reveals which algorithm is the most robust generalizer across the entire Sri Lankan food basket, rather than just being good at predicting a single easy commodity.
* **Overall RMSE & Radar Visuals:** These aggregated charts summarize the global performance footprint. They allow you to easily compare the baseline statistical capabilities against the advanced ST-GNN architecture across massive, diverse datasets simultaneously.

---

### 4. Export & Reporting Utilities

The platform offers comprehensive export utilities allowing you to seamlessly transition from dashboard analysis to academic or stakeholder reporting.

* **CSV & PDF Export:** You can download isolated forecast spreadsheets for individual items or generate an automated, formatted PDF document containing the overarching performance tables and inflation statistics.
* **Overall Pipeline Summary Report (.xlsx):** This is the most powerful export option. It downloads a beautifully structured, multi-sheet Excel workbook containing:
  * **Sheet 1 (`Overall_Summary`):** The macro-averaged metrics table showcasing the global performance of all models across the entire session.
  * **Sheet 2 (`Item_Level_Breakdown`):** A granular, flat data table mapping the exact MAE, RMSE, MAPE, and R² for every single food item and every single model processed. This sheet is perfectly structured for pivot tables and secondary academic analysis.
