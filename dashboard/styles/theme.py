"""dashboard/styles/theme.py — Centralized Bloomberg-style dark fintech theme."""

BG_PRIMARY   = "#0B1020"
BG_SECONDARY = "#111827"
BG_CARD      = "#1A2238"
BG_CARD_HOVER= "#1E2D4A"
BG_INPUT     = "#0F1C32"
BORDER       = "#1E2D4A"
BORDER_LIGHT = "#253355"

ACCENT_BLUE  = "#3B82F6"
ACCENT_CYAN  = "#06B6D4"
ACCENT_GREEN = "#10B981"
ACCENT_AMBER = "#F59E0B"
ACCENT_RED   = "#EF4444"
ACCENT_PURPLE= "#8B5CF6"
ACCENT_ORANGE= "#F97316"

TEXT_PRIMARY  = "#F1F5F9"
TEXT_SECONDARY= "#94A3B8"
TEXT_MUTED    = "#64748B"

COLORS = [
    "#3B82F6","#06B6D4","#10B981","#F59E0B",
    "#EF4444","#8B5CF6","#F97316","#EC4899",
    "#14B8A6","#6366F1","#84CC16","#F43F5E",
]

def get_chart_layout(title="", xaxis_title="", yaxis_title="",
                     height=420, show_rangeslider=True,
                     legend_orientation="v", **kwargs) -> dict:
    layout = dict(
        paper_bgcolor=BG_CARD, plot_bgcolor=BG_SECONDARY,
        font=dict(family="Inter, sans-serif", color=TEXT_PRIMARY, size=12),
        title=dict(text=title, font=dict(size=15, color=TEXT_PRIMARY,
                   family="Poppins, sans-serif"), x=0.01, xanchor="left"),
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(size=12, color=TEXT_SECONDARY)),
            gridcolor=BORDER, showgrid=True, zeroline=False,
            tickfont=dict(size=11, color=TEXT_SECONDARY),
            showspikes=True, spikecolor=ACCENT_BLUE, spikethickness=1, spikedash="dot",
            rangeslider=dict(visible=show_rangeslider, bgcolor=BG_PRIMARY,
                             thickness=0.04, bordercolor=BORDER),
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(size=12, color=TEXT_SECONDARY)),
            gridcolor=BORDER, showgrid=True, zeroline=False,
            tickfont=dict(size=11, color=TEXT_SECONDARY),
            showspikes=True, spikecolor=ACCENT_BLUE, spikethickness=1, spikedash="dot",
        ),
        legend=dict(
            bgcolor=BG_CARD, bordercolor=BORDER, borderwidth=1,
            font=dict(size=11, color=TEXT_PRIMARY), orientation=legend_orientation,
            yanchor="top" if legend_orientation=="v" else "bottom",
            y=1.0 if legend_orientation=="v" else -0.25,
            xanchor="right" if legend_orientation=="v" else "center",
            x=1.0 if legend_orientation=="v" else 0.5,
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=BG_CARD, bordercolor=ACCENT_BLUE,
                        font=dict(size=12, color=TEXT_PRIMARY), namelength=-1),
        margin=dict(l=60, r=20, t=55, b=50), height=height,
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=TEXT_MUTED,
                     activecolor=ACCENT_BLUE, orientation="h"),
        dragmode="pan",
    )
    layout.update(kwargs)
    return layout

def compact_layout(title="", xaxis_title="", yaxis_title="", height=280, **kwargs):
    kwargs.pop("show_rangeslider", None)  # prevent duplicate kwarg
    return get_chart_layout(title=title, xaxis_title=xaxis_title,
                            yaxis_title=yaxis_title, height=height,
                            show_rangeslider=False, **kwargs)

def get_css() -> str:
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700;800&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: {TEXT_PRIMARY}; }}
.stApp {{ background-color: {BG_PRIMARY} !important; }}
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stToolbar"] {{ display: none; }}

section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #0D1629 0%, {BG_SECONDARY} 100%) !important;
  border-right: 1px solid {BORDER} !important;
}}

.stTabs [data-baseweb="tab-list"] {{
  background: {BG_SECONDARY}; border-radius: 10px; padding: 4px;
  gap: 2px; border: 1px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{
  border-radius: 7px; color: {TEXT_SECONDARY}; font-weight: 500;
  font-size: 0.88rem; padding: 8px 16px; transition: all 0.2s ease; background: transparent;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: {TEXT_PRIMARY}; background: {BG_CARD} !important; }}
.stTabs [aria-selected="true"] {{
  background: linear-gradient(135deg, {ACCENT_BLUE}, #1D4ED8) !important;
  color: white !important; font-weight: 600 !important;
  box-shadow: 0 2px 10px rgba(59,130,246,0.35);
}}

[data-testid="metric-container"], div[data-testid="stMetric"] {{
  background: linear-gradient(135deg, {BG_CARD} 0%, {BG_CARD_HOVER} 100%);
  border: 1px solid {BORDER}; border-radius: 12px; padding: 1rem 1.2rem;
  transition: all 0.2s ease; position: relative; overflow: hidden;
}}
[data-testid="metric-container"]:hover, div[data-testid="stMetric"]:hover {{
  border-color: {ACCENT_BLUE}; transform: translateY(-1px);
  box-shadow: 0 4px 20px rgba(59,130,246,0.2);
}}
[data-testid="stMetricValue"], div[data-testid="stMetricValue"] {{
  color: {ACCENT_CYAN} !important; font-size: 1.6rem !important;
  font-weight: 700 !important; font-family: 'Poppins', sans-serif !important;
}}
[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] {{
  color: {TEXT_SECONDARY} !important; font-size: 0.76rem !important;
  text-transform: uppercase; letter-spacing: 0.05em;
}}

.stButton > button {{
  border-radius: 8px !important; font-weight: 500; transition: all 0.18s ease !important;
  border: 1px solid {BORDER} !important; background: {BG_CARD} !important; color: {TEXT_PRIMARY} !important;
}}
.stButton > button:hover {{ border-color: {ACCENT_BLUE} !important; transform: translateY(-1px); }}
.stButton > button[kind="primary"] {{
  background: linear-gradient(135deg, {ACCENT_BLUE}, #1D4ED8) !important;
  border: none !important; color: white !important; font-weight: 600 !important;
  box-shadow: 0 4px 15px rgba(59,130,246,0.3);
}}
.stButton > button[kind="primary"]:hover {{
  transform: translateY(-2px) !important; box-shadow: 0 6px 22px rgba(59,130,246,0.45) !important;
}}

.stDownloadButton > button {{
  background: linear-gradient(135deg, #065F46, {ACCENT_GREEN}) !important;
  border: none !important; border-radius: 7px !important; color: white !important;
  font-weight: 500 !important; box-shadow: 0 2px 8px rgba(16,185,129,0.2);
}}
.stDownloadButton > button:hover {{ transform: translateY(-1px) !important; }}

.stSelectbox > div > div, .stMultiSelect > div > div {{
  background: {BG_INPUT} !important; border: 1px solid {BORDER} !important;
  border-radius: 8px !important; color: {TEXT_PRIMARY} !important;
}}
[data-baseweb="tag"] {{
  background: rgba(59,130,246,0.2) !important; border-radius: 4px !important;
  border: 1px solid rgba(59,130,246,0.35) !important; color: {ACCENT_BLUE} !important;
}}

.stTextInput > div > div > input {{
  background: {BG_INPUT} !important; border: 1px solid {BORDER} !important;
  border-radius: 8px !important; color: {TEXT_PRIMARY} !important;
}}

[data-testid="stFileUploadDropzone"] {{
  background: {BG_INPUT} !important; border: 2px dashed {BORDER} !important;
  border-radius: 10px !important; transition: all 0.2s ease !important;
}}
[data-testid="stFileUploadDropzone"]:hover {{
  border-color: {ACCENT_BLUE} !important; background: rgba(59,130,246,0.04) !important;
}}

[data-testid="stExpander"] {{
  background: {BG_CARD} !important; border: 1px solid {BORDER} !important;
  border-radius: 10px !important; margin-bottom: 0.5rem !important;
}}
[data-testid="stExpander"] summary {{
  color: {TEXT_PRIMARY} !important; font-weight: 500 !important; padding: 0.7rem 1rem !important;
}}

[data-testid="stSuccess"] {{ background: rgba(16,185,129,0.08) !important; border-left-color: {ACCENT_GREEN} !important; }}
[data-testid="stWarning"] {{ background: rgba(245,158,11,0.08) !important; border-left-color: {ACCENT_AMBER} !important; }}
[data-testid="stError"]   {{ background: rgba(239,68,68,0.08)  !important; border-left-color: {ACCENT_RED} !important; }}
[data-testid="stInfo"]    {{ background: rgba(59,130,246,0.08) !important; border-left-color: {ACCENT_BLUE} !important; }}

[data-testid="stDataFrame"] {{ border-radius: 10px !important; border: 1px solid {BORDER} !important; }}

.stProgress > div > div > div {{
  background: linear-gradient(90deg, {ACCENT_BLUE}, {ACCENT_CYAN}) !important;
}}

::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {BG_PRIMARY}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {ACCENT_BLUE}; }}

hr {{ border-color: {BORDER} !important; margin: 0.8rem 0 !important; }}

.fpc-card {{
  background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 12px;
  padding: 1.2rem 1.4rem; margin-bottom: 0.8rem; position: relative; overflow: hidden;
  transition: all 0.2s ease;
}}
.section-title {{
  font-family: 'Poppins', sans-serif; font-size: 1rem; font-weight: 600;
  color: {TEXT_PRIMARY}; margin: 0.8rem 0 0.6rem; padding-bottom: 0.4rem;
  border-bottom: 1px solid {BORDER};
}}
.badge-success {{
  display: inline-block; background: rgba(16,185,129,0.15);
  border: 1px solid rgba(16,185,129,0.35); color: #6EE7B7;
  border-radius: 10px; padding: 1px 8px; font-size: 0.7rem; font-weight: 500;
}}
.badge-warning {{
  display: inline-block; background: rgba(245,158,11,0.15);
  border: 1px solid rgba(245,158,11,0.35); color: #FCD34D;
  border-radius: 10px; padding: 1px 8px; font-size: 0.7rem; font-weight: 500;
}}
.badge-info {{
  display: inline-block; background: rgba(59,130,246,0.15);
  border: 1px solid rgba(59,130,246,0.35); color: #93C5FD;
  border-radius: 10px; padding: 1px 8px; font-size: 0.7rem; font-weight: 500;
}}
@keyframes pulse {{
  0%, 100% {{ opacity:1; transform:scale(1); }}
  50% {{ opacity:0.5; transform:scale(0.8); }}
}}
.live-dot {{
  display:inline-block; width:7px; height:7px; background:{ACCENT_GREEN};
  border-radius:50%; animation:pulse 2s infinite; margin-right:5px; vertical-align:middle;
}}
.empty-state {{ text-align:center; padding:3rem 2rem; color:{TEXT_MUTED}; }}
.empty-state .icon {{ font-size:2.8rem; opacity:0.4; margin-bottom:0.8rem; }}
.empty-state h3 {{ color:{TEXT_SECONDARY}; font-size:1.05rem; margin-bottom:0.4rem; }}
</style>"""
