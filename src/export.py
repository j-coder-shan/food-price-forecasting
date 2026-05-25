"""
export.py  —  Export helpers for forecasts, metrics, and inflation reports.
Sri Lanka AI Food Price & Inflation Forecasting Platform

Supports:
  - CSV bytes (for Streamlit st.download_button)
  - Multi-sheet Excel bytes (openpyxl)
  - PDF summary report (fpdf2, optional)
"""

import io
import warnings
import re
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from src.utils import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


# ─────────────────────────────────────────────
# CSV Helpers
# ─────────────────────────────────────────────
def get_csv_bytes(df: pd.DataFrame) -> bytes:
    """Returns a DataFrame as UTF-8 CSV bytes (for Streamlit download)."""
    return df.to_csv(index=False).encode("utf-8")


def save_csv(df: pd.DataFrame, path: Path) -> Path:
    """Saves DataFrame to CSV file."""
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info(f"Saved CSV: {path}")
    return path


# ─────────────────────────────────────────────
# Excel Helpers
# ─────────────────────────────────────────────
def get_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    """
    Returns a multi-sheet Excel file as bytes (for Streamlit download).

    Args:
        sheets: dict mapping sheet name → DataFrame.
                Sheet names are truncated to 31 chars (Excel limit).

    Returns:
        bytes of the .xlsx file.
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_name = re.sub(r'[\\/?*\[\]:]', '_', str(sheet_name))[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
    return buf.getvalue()


def save_excel(sheets: dict[str, pd.DataFrame], path: Path) -> Path:
    """Saves multiple DataFrames as sheets in a single .xlsx file."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_name = re.sub(r'[\\/?*\[\]:]', '_', str(sheet_name))[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
    logger.info(f"Saved Excel: {path} ({len(sheets)} sheets)")
    return path


# ─────────────────────────────────────────────
# Full Report Builder
# ─────────────────────────────────────────────
def build_full_report(
    forecasts_by_target: dict[str, dict[int, pd.DataFrame]],
    metrics_by_target:   dict[str, pd.DataFrame],
    inflation_table:     pd.DataFrame | None = None,
    horizon:             int = 12,
    forecasted_inflation_table: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Assembles a dictionary of sheets suitable for export_excel / get_excel_bytes.

    Sheets produced:
      - Summary           : best model + RMSE per target
      - Forecasts_<N>m    : all-target forecast table for horizon N
      - Metrics_<target>  : per-target model metrics (one sheet each, max 10)
      - Inflation         : historical monthly & YoY inflation table (if provided)
      - Forecasted_Inflation: forecasted monthly & YoY inflation table (if provided)

    Args:
        forecasts_by_target: {target: {horizon: forecast_df}}
        metrics_by_target:   {target: metrics_df}
        inflation_table:     Optional DataFrame from InflationCalculator
        horizon:             Primary forecast horizon to include
        forecasted_inflation_table: Optional forecasted inflation DataFrame
    """
    sheets = {}

    # Summary sheet
    summary_rows = []
    for target, metrics_df in metrics_by_target.items():
        if metrics_df is not None and len(metrics_df) > 0:
            best = metrics_df.sort_values("RMSE").iloc[0]
            summary_rows.append({
                "Food Item": target,
                "Best Model": best["Model"],
                "RMSE":  round(best["RMSE"], 4),
                "MAE":   round(best["MAE"], 4),
                "MAPE":  round(best["MAPE"], 4),
                "R2":    round(best["R2"], 4),
            })
    if summary_rows:
        sheets["Summary"] = pd.DataFrame(summary_rows)

    # Combined forecast sheet for primary horizon
    fc_cols = ["Month"]
    fc_data = {}
    for target, fc_dict in forecasts_by_target.items():
        if horizon in fc_dict:
            fc_df = fc_dict[horizon]
            for col in fc_df.columns:
                if col != "Month":
                    fc_data[col] = fc_df[col].values
            if "Month" not in fc_data:
                fc_data["Month"] = fc_df["Month"].values
    if fc_data:
        sheets[f"Forecasts_{horizon}m"] = pd.DataFrame(fc_data)

    # Per-target metrics (max 10 sheets to keep file manageable)
    for i, (target, metrics_df) in enumerate(metrics_by_target.items()):
        if i >= 10:
            break
        if metrics_df is not None and len(metrics_df) > 0:
            short = target[:25]
            sheets[f"Metrics_{short}"] = metrics_df

    # Inflation sheet
    if inflation_table is not None and len(inflation_table) > 0:
        # Reset index to make Month a column
        inf_df = inflation_table.copy()
        if isinstance(inf_df.index, pd.DatetimeIndex):
            inf_df = inf_df.reset_index()
            inf_df.rename(columns={"Month": "Month"}, inplace=True)
        sheets["Inflation"] = inf_df

    # Forecasted Inflation sheet
    if forecasted_inflation_table is not None and len(forecasted_inflation_table) > 0:
        sheets["Forecasted_Inflation"] = forecasted_inflation_table

    logger.info(f"Built report with {len(sheets)} sheets.")
    return sheets


def build_macro_summary_report(metrics_by_target: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Builds the macro-averaged Overall Pipeline Summary Report.
    Returns a dict with 'Overall_Summary' and 'Item_Level_Breakdown' sheets.
    """
    all_metrics = []
    for target, metrics_df in metrics_by_target.items():
        if metrics_df is not None and not metrics_df.empty:
            df_copy = metrics_df.copy()
            # Ensure Food Item is the first column
            if "Food Item" not in df_copy.columns:
                df_copy.insert(0, "Food Item", target)
            all_metrics.append(df_copy)
            
    if not all_metrics:
        return {}
        
    master_df = pd.concat(all_metrics, ignore_index=True)
    
    # Macro Average
    macro_df = master_df.groupby("Model").agg({
        "MAE": "mean",
        "RMSE": "mean",
        "MAPE": "mean",
        "R2": "mean"
    }).reset_index().sort_values("RMSE")
    
    return {
        "Overall_Summary": macro_df,
        "Item_Level_Breakdown": master_df
    }



# ─────────────────────────────────────────────
# PDF Report (optional — requires fpdf2)
# ─────────────────────────────────────────────
def _safe(text: str) -> str:
    """Replace Unicode special chars that Helvetica can't encode."""
    return (
        str(text)
        .replace("\u2010", "-")   # unicode hyphen -> ASCII hyphen
        .replace("\u2013", "-")   # en dash
        .replace("\u2014", "-")   # em dash
        .replace("\u2192", "->")  # arrow
        .replace("\u00d7", "x")   # multiplication sign
        .encode("latin-1", errors="replace").decode("latin-1")
    )


def get_pdf_bytes(
    title: str,
    summary_df: pd.DataFrame | None,
    inflation_summary: dict | None,
    generated_at: str | None = None,
    forecast_df: pd.DataFrame | None = None,
) -> bytes | None:
    """
    Generates a simple PDF summary report.
    Returns None if fpdf2 is not installed.

    Args:
        title:             Report title string.
        summary_df:        Summary metrics DataFrame (from build_full_report).
        inflation_summary: Dict from InflationCalculator.summary().
        generated_at:      Timestamp string.

    Returns:
        PDF bytes or None.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        logger.warning("fpdf2 not installed — PDF export unavailable.")
        return None

    ts = generated_at or datetime.now().strftime("%Y-%m-%d %H:%M")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, _safe(title), ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, _safe(f"Generated: {ts}"), ln=True, align="C")
    pdf.ln(6)

    # Inflation summary
    if inflation_summary:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, "Inflation Statistics", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for k, v in inflation_summary.items():
            label = k.replace("_", " ").title()
            pdf.cell(0, 7, _safe(f"  {label}: {v}%"), ln=True)
        pdf.ln(4)

    # Model summary table
    if summary_df is not None and len(summary_df) > 0:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, "Best Model Per Food Item", ln=True)
        pdf.set_font("Helvetica", "B", 9)

        # Header
        col_widths = [65, 35, 20, 20, 20, 18]
        headers = ["Food Item", "Best Model", "RMSE", "MAE", "MAPE", "R2"]
        for h, w in zip(headers, col_widths):
            pdf.cell(w, 7, h, border=1)
        pdf.ln()

        # Rows
        pdf.set_font("Helvetica", "", 8)
        for _, row in summary_df.head(30).iterrows():
            vals = [
                _safe(str(row.get("Food Item", ""))[:28]),
                _safe(str(row.get("Best Model", ""))[:18]),
                _safe(str(row.get("RMSE", ""))),
                _safe(str(row.get("MAE", ""))),
                _safe(str(row.get("MAPE", ""))),
                _safe(str(row.get("R2", ""))),
            ]
            for v, w in zip(vals, col_widths):
                pdf.cell(w, 6, v, border=1)
            pdf.ln()

    # Forecasts table
    if forecast_df is not None and len(forecast_df) > 0:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, "Forecasted Values", ln=True)
        pdf.set_font("Helvetica", "B", 8)

        # Limit to 6 columns total to fit standard A4 portrait width
        cols = list(forecast_df.columns)
        if len(cols) > 6:
            cols = cols[:6]
            
        col_w = 190 / len(cols)
        
        # Headers
        for col in cols:
            # strip "(Forecast)" to save space
            c_name = col.replace(" (Forecast)", "")[:18]
            pdf.cell(col_w, 7, _safe(c_name), border=1)
        pdf.ln()

        # Rows
        pdf.set_font("Helvetica", "", 8)
        for _, row in forecast_df.iterrows():
            for col in cols:
                val = row[col]
                if isinstance(val, (float, np.floating)):
                    val_str = f"{val:,.2f}"
                else:
                    val_str = str(val)[:15]
                pdf.cell(col_w, 6, _safe(val_str), border=1)
            pdf.ln()
        
        if len(forecast_df.columns) > 6:
            pdf.ln(2)
            pdf.set_font("Helvetica", "I", 8)
            pdf.cell(0, 5, f"* Note: Only the first 5 food items are shown due to page width constraints.", ln=True)

    return bytes(pdf.output())
