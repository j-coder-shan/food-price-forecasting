"""dashboard/components/sidebar.py — Redesigned sidebar with validation, status badges, and model selection."""
import io
import streamlit as st
from pathlib import Path
from src.utils import AVAILABLE_MODELS, DEFAULT_MODELS, DATA_DIR
from src.validation import DatasetValidator, show_validation
from dashboard.styles.theme import (
    ACCENT_BLUE, ACCENT_GREEN, ACCENT_AMBER, TEXT_SECONDARY, BG_CARD
)

_HORIZONS = {"3 months": 3, "6 months": 6, "12 months": 12}


def _resolve_file(upload, default_path: Path):
    """Returns BytesIO from upload or Path if using default."""
    if upload is not None:
        return io.BytesIO(upload.getvalue()), upload.name, True
    return default_path, default_path.name, False


def _file_badge(uploaded: bool, name: str) -> str:
    if uploaded:
        return f'<span class="badge-success">✓ Uploaded: {name[:22]}</span>'
    return f'<span class="badge-info">⟳ Auto: {name[:22]}</span>'


def render_sidebar(food_cols: list[str]) -> dict:
    """Renders the full sidebar and returns user config dict."""
    cfg = {}

    with st.sidebar:
        # ── Branding ──────────────────────────────────────────────────────────
        st.markdown("""
        <div style="background:linear-gradient(135deg,#0D1629,#1A2238);
                    border-bottom:1px solid #1E2D4A; padding:14px 16px 12px; margin:-1rem -1rem 0.8rem;">
            <div style="display:flex;align-items:center;gap:10px;">
                <span style="font-size:1.6rem;">🌾</span>
                <div>
                    <div style="font-family:Poppins,sans-serif;font-weight:700;
                                font-size:0.95rem;color:#F1F5F9;line-height:1.2;">
                        SL Food Price AI
                    </div>
                    <div style="font-size:0.68rem;color:#64748B;letter-spacing:0.06em;
                                text-transform:uppercase;">Inflation Forecasting Platform</div>
                </div>
                <div style="margin-left:auto;">
                    <span class="live-dot"></span>
                    <span style="font-size:0.65rem;color:#10B981;">LIVE</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── 1. Dataset Sources ─────────────────────────────────────────────────
        with st.expander("📂 Dataset Sources", expanded=True):
            food_up = st.file_uploader("🍚 Food Prices (.xlsx / .csv)",
                                       type=["xlsx","csv"], key="up_food",
                                       help="Monthly food prices. Auto-loads food_prices.xlsx if not uploaded.")
            fuel_up = st.file_uploader("⛽ Fuel Prices (.xls / .xlsx)",
                                       type=["xls","xlsx"], key="up_fuel",
                                       help="Brent crude oil prices. Auto-loads fuel_prices.xls if not uploaded.")
            exch_up = st.file_uploader("💱 Exchange Rates (.xlsx)",
                                       type=["xlsx","csv"], key="up_exch",
                                       help="USD/LKR rates. Auto-loads exchange_rates.xlsx if not uploaded.")

            food_obj, food_name, food_upl = _resolve_file(food_up, DATA_DIR / "food_prices.xlsx")
            fuel_obj, fuel_name, fuel_upl = _resolve_file(fuel_up, DATA_DIR / "fuel_prices.xls")
            exch_obj, exch_name, exch_upl = _resolve_file(exch_up, DATA_DIR / "exchange_rates.xlsx")

            cfg["food_file"]     = food_obj
            cfg["fuel_file"]     = fuel_obj
            cfg["exchange_file"] = exch_obj

            st.markdown(
                _file_badge(food_upl, food_name) + "&nbsp;" +
                _file_badge(fuel_upl, fuel_name) + "&nbsp;" +
                _file_badge(exch_upl, exch_name),
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            cfg["use_merged"] = st.checkbox(
                "Merge fuel & FX features into model",
                value=True,
                help="Adds Brent oil price and USD/LKR as lag/rolling predictor features.",
            )

        # ── 2. Food Item Selection ─────────────────────────────────────────────
        with st.expander("🎯 Food Item Selection", expanded=True):
            search = st.text_input("🔍 Search food items", placeholder="e.g. Rice, Sugar, Fish…",
                                   key="sb_search")
            filtered = [c for c in food_cols
                        if search.lower() in c.lower()] if search else food_cols

            col_a, col_b = st.columns([1, 1])
            select_all = col_a.checkbox("All", key="sb_all",
                                        help="Select all visible items")
            # Include Index if forecast of index is desired
            include_index = col_b.checkbox("+ Index", value=True, key="sb_index",
                                           help="Also forecast the Food Price Index (needed for inflation)")

            if select_all:
                selected = list(filtered)
                st.caption(f"All {len(selected)} items selected.")
            else:
                default_picks = filtered[:5] if len(filtered) >= 5 else filtered
                selected = st.multiselect(
                    f"{len(filtered)} items available",
                    options=filtered, default=default_picks, key="sb_items",
                )

            if include_index and "Index" in food_cols and "Index" not in selected:
                selected = selected + ["Index"]

            cfg["selected_targets"] = selected
            if not selected:
                st.warning("⚠️ Select at least one food item.")

        # ── 3. Forecast Settings ───────────────────────────────────────────────
        with st.expander("⚙️ Forecast Settings", expanded=True):
            horizon_label = st.radio("Forecast Horizon",
                                     list(_HORIZONS.keys()) + ["Custom"],
                                     index=2, horizontal=True)
            if horizon_label == "Custom":
                cfg["horizon"] = st.slider("Months ahead", 1, 24, 12)
            else:
                cfg["horizon"] = _HORIZONS[horizon_label]

            st.markdown(
                "<div class='section-title' style='font-size:0.8rem;margin-top:0.8rem;'>Models</div>",
                unsafe_allow_html=True,
            )
            ml_sel, stat_sel = [], []
            ml_cols = st.columns(2)
            for i, m in enumerate(AVAILABLE_MODELS["ML"]):
                checked = ml_cols[i % 2].checkbox(m, value=(m in DEFAULT_MODELS),
                                                   key=f"m_{m}")
                if checked:
                    ml_sel.append(m)

            st.caption("Statistical (slower, ~2 min/item)")
            stat_cols = st.columns(2)
            for i, m in enumerate(AVAILABLE_MODELS["Statistical"]):
                if stat_cols[i % 2].checkbox(m, value=False, key=f"m_{m}"):
                    stat_sel.append(m)

            cfg["selected_models"]  = ml_sel + stat_sel
            cfg["skip_statistical"] = len(stat_sel) == 0

            if not cfg["selected_models"]:
                st.error("Select at least one model.")

        # ── 4. Advanced Settings ───────────────────────────────────────────────
        with st.expander("🔧 Advanced Settings", expanded=False):
            cfg["train_ratio"] = st.slider(
                "Train/Test Split", 0.6, 0.95, 0.80, 0.05,
                help="Fraction of data used for training. Default 80%.",
            )
            cfg["rolling_windows"] = st.multiselect(
                "Rolling Window Sizes (months)",
                [3, 6, 12, 24], default=[3, 6, 12],
                help="Window sizes for rolling mean/std features.",
            )

        st.markdown("---")

        # ── 5. Generate Button ─────────────────────────────────────────────────
        btn_disabled = (not cfg["selected_targets"] or not cfg["selected_models"])
        cfg["run"] = st.button(
            "🚀  Generate Forecast",
            type="primary",
            use_container_width=True,
            disabled=btn_disabled,
            help="Train models and generate forecasts for selected items.",
        )
        if btn_disabled and not cfg["selected_targets"]:
            st.caption("↑ Select food items to enable")

        # ── 6. System Status ───────────────────────────────────────────────────
        st.markdown(
            f"""<div style="margin-top:0.6rem;padding:10px 12px;
                background:#0F1C32;border:1px solid #1E2D4A;border-radius:8px;">
            <div style="font-size:0.7rem;color:#64748B;text-transform:uppercase;
                        letter-spacing:0.05em;margin-bottom:6px;">System Status</div>
            <div style="font-size:0.78rem;color:#94A3B8;line-height:1.8;">
                🌾 Items: <b style="color:#F1F5F9">{len(food_cols)}</b><br>
                🔢 Horizon: <b style="color:#F1F5F9">{cfg['horizon']}m</b><br>
                🤖 Models: <b style="color:#F1F5F9">{len(cfg['selected_models'])}</b>
            </div></div>""",
            unsafe_allow_html=True,
        )

    return cfg
