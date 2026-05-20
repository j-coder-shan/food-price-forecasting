"""dashboard/page_modules/export_tab.py — Tab 5: Export Reports."""
from datetime import datetime
import pandas as pd
import streamlit as st
import importlib
import src.export
importlib.reload(src.export)
from src.export import get_csv_bytes, get_excel_bytes, get_pdf_bytes, build_full_report, build_macro_summary_report
from src.inflation import InflationCalculator


def _dl_card(col, label: str, icon: str, desc: str, btn_label: str,
             data: bytes, filename: str, mime: str):
    """Renders a styled download card in the given column."""
    col.markdown(f"""
    <div class="fpc-card" style="text-align:center;padding:1rem;">
        <div style="font-size:2rem;margin-bottom:6px;">{icon}</div>
        <div style="font-weight:600;font-size:0.9rem;margin-bottom:4px;">{label}</div>
        <div style="font-size:0.75rem;color:#64748B;margin-bottom:10px;">{desc}</div>
    </div>""", unsafe_allow_html=True)
    col.download_button(btn_label, data=data, file_name=filename,
                        mime=mime, use_container_width=True)


def render(df: pd.DataFrame, cfg: dict, results: dict) -> None:
    if not results:
        st.markdown("""<div class="empty-state"><div class="icon">⬇️</div>
        <h3>No Data to Export</h3>
        <p style="font-size:0.85rem;">Run the forecast first, then download your results here.</p>
        </div>""", unsafe_allow_html=True)
        return

    horizon = cfg.get("horizon", 12)
    ts      = datetime.now().strftime("%Y%m%d_%H%M")

    # Build report sheets
    forecasts_by_target = {t: r.get("forecasts", {}) for t, r in results.items()}
    metrics_by_target   = {t: r.get("metrics_df")   for t, r in results.items()}

    inf_table = None
    inf_summary = None
    if "Index" in df.columns:
        calc      = InflationCalculator(df["Index"])
        inf_table = calc.inflation_table().reset_index()
        inf_table["Month"] = inf_table["Month"].dt.strftime("%Y-%m")
        inf_summary = calc.summary()

    report_sheets = build_full_report(
        forecasts_by_target, metrics_by_target, inf_table, horizon
    )

    # ── Section 1: Per-item CSV ────────────────────────────────────────────────
    st.markdown("<div class='section-title'>📌 Per-Item Forecast Download</div>",
                unsafe_allow_html=True)
    targets = list(results.keys())
    target  = st.selectbox("Select food item", targets, key="exp_target")
    r = results.get(target, {})
    avail_h = sorted(r.get("forecasts", {}).keys())
    if avail_h:
        sel_h = st.radio("Horizon", avail_h, horizontal=True,
                         format_func=lambda h: f"{h}m", key="exp_h")
        fc_df = r["forecasts"].get(sel_h)
        if fc_df is not None:
            st.dataframe(fc_df, use_container_width=True, hide_index=True)
            st.download_button(
                f"⬇️ Download Forecast CSV — {target[:30]} ({sel_h}m)",
                data=get_csv_bytes(fc_df),
                file_name=f"{target[:30].replace(' ','_')}_{sel_h}m_{ts}.csv",
                mime="text/csv", use_container_width=True,
            )

    # ── Section 2: Full Reports ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-title'>📦 Full System Reports</div>",
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    # Summary CSV
    if "Summary" in report_sheets:
        size_kb = round(len(get_csv_bytes(report_sheets["Summary"])) / 1024, 1)
        _dl_card(c1, "Metrics Summary", "📊",
                 f"Best model + RMSE/MAE/R² per item · {size_kb} KB",
                 "⬇️ Summary CSV",
                 get_csv_bytes(report_sheets["Summary"]),
                 f"metrics_summary_{ts}.csv", "text/csv")

    # Inflation CSV
    if inf_table is not None:
        _dl_card(c2, "Inflation Report", "📉",
                 "Historical MoM & YoY inflation table",
                 "⬇️ Inflation CSV",
                 get_csv_bytes(inf_table),
                 f"inflation_report_{ts}.csv", "text/csv")

    # Full Excel
    excel_bytes = get_excel_bytes(report_sheets)
    size_kb_xl  = round(len(excel_bytes) / 1024, 1)
    _dl_card(c3, "Full Excel Report", "📁",
             f"All sheets: forecasts + metrics + inflation · {size_kb_xl} KB",
             "⬇️ Excel (.xlsx)",
             excel_bytes,
             f"sl_food_price_report_{ts}.xlsx",
             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ── Section 2.5: Macro Summary ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-title'>🌍 Overall Pipeline Summary</div>",
                unsafe_allow_html=True)
    
    macro_sheets = build_macro_summary_report(metrics_by_target)
    if macro_sheets:
        macro_excel_bytes = get_excel_bytes(macro_sheets)
        st.download_button(
            "📥 Export Overall Pipeline Summary Report (.xlsx)",
            data=macro_excel_bytes,
            file_name=f"overall_macro_summary_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="primary"
        )
        st.info("Contains **Overall_Summary** (averages across all items) and **Item_Level_Breakdown**.")

    # ── Section 3: PDF ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-title'>📄 PDF Summary Report</div>",
                unsafe_allow_html=True)
    col_pdf, col_info = st.columns([1, 2])
    pdf_bytes = get_pdf_bytes(
        title="Sri Lanka Food Price & Inflation Forecast Report",
        summary_df=report_sheets.get("Summary"),
        inflation_summary=inf_summary,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        forecast_df=report_sheets.get(f"Forecasts_{horizon}m"),
    )
    if pdf_bytes:
        col_pdf.download_button(
            "⬇️ Download PDF Summary",
            data=pdf_bytes,
            file_name=f"sl_food_inflation_report_{ts}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        col_info.info(f"PDF includes: summary metrics table, inflation statistics, forecast predictions, and generation timestamp.")
    else:
        col_pdf.warning("Install `fpdf2` to enable PDF export: `pip install fpdf2`")

    # ── Section 4: Preview ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-title'>👁 Report Contents Preview</div>",
                unsafe_allow_html=True)
    for sheet_name, sheet_df in report_sheets.items():
        with st.expander(f"📋 {sheet_name}  ({len(sheet_df)} rows)"):
            st.dataframe(sheet_df.head(25), use_container_width=True, hide_index=True)
