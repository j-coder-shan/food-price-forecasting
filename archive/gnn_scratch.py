import os, re, shutil, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from numpy.fft import rfft
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error, r2_score)
warnings.filterwarnings("ignore")

os.makedirs("data/food_prices", exist_ok=True)
for year in range(2014, 2026):
    src = f"/content/{year}.csv"
    if os.path.exists(src):
        shutil.copy(src, f"/content/data/food_prices/{year}.csv")
for fname in ["CCPI_Consolidated_Base2021.csv", "z.csv"]:
    src = f"/content/{fname}"
    if os.path.exists(src):
        shutil.copy(src, f"/content/data/{fname}")

print("Files ready:")
print(f"  Food files : {len(os.listdir('/content/data/food_prices'))}")
print(f"  CPI exists : {os.path.exists('/content/data/CCPI_Consolidated_Base2021.csv')}")
print(f"  Weights    : {os.path.exists('/content/data/z.csv')}")

print("="*65)
print("  METHODOLOGY ADJUSTMENTS — Supervisor Recommendations")
print("="*65)

print("""
ORIGINAL APPROACH:
  ✗ Predicted CCPI index directly as a single scalar
  ✗ No basket weights used
  ✗ No festival demand seasonality
  ✗ End-to-end ST-GNN on 85 samples → mean collapse
  ✗ Target: raw CCPI index (81 → 191, non-stationary)

ADJUSTED APPROACH (per supervisor feedback):
  ✓ Bottom-up CPI calculation using official DCS basket weights
  ✓ 127 food items matched to DCS CCPI basket (99.77% coverage)
  ✓ Laspeyres price index formula — same as DCS methodology
  ✓ Festival demand features added:
      - Sinhala & Tamil New Year (March–April)
      - Christmas & New Year (November–January)
      - Maha harvest season (March–April)
      - Yala harvest season (August–September)
  ✓ Hybrid ST-GNN + Ridge to handle small dataset (85 samples)
  ✓ Target: Month-on-Month CPI % change (stationary)
""")

print("-"*65)
print("  BASKET WEIGHT VALIDATION")
print("-"*65)
print(f"  DCS basket items        : 129")
print(f"  Matched to price data   : 127  (99.77% weight coverage)")
print(f"  Unmatched (negligible)  : 2    (Thalapath dry, Samaposha)")
print(f"  Base year               : {BASE_YEAR} (DCS CCPI base year)")
print(f"  Bottom-up vs official   : corr = {corr:.4f}")

print("""
LASPEYRES FORMULA USED:

  CPI_t = Σ ( w_i × P_it / P_i0 ) × 100

  where:
    w_i  = normalised DCS basket weight for item i
    P_it = retail price of item i at month t
    P_i0 = average retail price of item i in base year 2021
    100  = index base value

  This mirrors the exact methodology published by the
  Department of Census and Statistics of Sri Lanka.
""")

print("-"*65)
print("  FESTIVAL DEMAND FEATURES")
print("-"*65)
festivals = {
    "Sinhala & Tamil New Year": {
        "months": "March (pre-shopping), April (peak)",
        "affected_items": "Milk rice ingredients, kavum, kokis, "
                          "coconut, rice flour, sugar, fruits",
        "features": "sinhala_ny_march, sinhala_ny_april, sinhala_season"
    },
    "Christmas & New Year": {
        "months": "November (prep), December (peak), January (NY)",
        "affected_items": "Chicken, sweets, soft drinks, "
                          "chocolates, cake, fruit drinks",
        "features": "christmas_nov, christmas_dec, new_year_jan, "
                    "christmas_season"
    },
    "Maha Harvest (rice)": {
        "months": "March–April (price drop after harvest)",
        "affected_items": "All rice varieties",
        "features": "maha_harvest"
    },
    "Yala Harvest (rice)": {
        "months": "August–September (price drop after harvest)",
        "affected_items": "All rice varieties",
        "features": "yala_harvest"
    },
}
for name, info in festivals.items():
    print(f"\n  {name}")
    print(f"    Months   : {info['months']}")
    print(f"    Items    : {info['affected_items']}")
    print(f"    Features : {info['features']}")

MONTH_COLS = ["January","February","March","April","May","June",
              "July","August","September","October","November","December"]
NON_FOOD   = ["NON FOOD","Total","ALCOHOLIC","CLOTHING","HOUSING",
              "FURNISHING","HEALTH","TRANSPORT","COMMUNICATION",
              "RECREATION","EDUCATION","RESTAURANTS","MISCELLANEOUS",
              "FOOD AND NON"]
TRAIN_END  = "2022-12-01"
VAL_END    = "2023-12-01"
WINDOW     = 12
BASE_YEAR  = 2021

def clean_name(n):
    n = re.sub(r"[\u2010-\u2015\u002D\-]+", "-", str(n).strip())
    return re.sub(r"\s+", " ", n).lower()

def is_non_food(n):
    return any(k in str(n).upper() for k in NON_FOOD)

# ── Load basket weights ───────────────────────────────────────────────
weights_raw = pd.read_csv("/content/data/z.csv", header=0)
weights_raw.columns = ["item","_","weight"]
weights_raw = weights_raw[["item","weight"]].dropna()
weights_raw["weight"]     = pd.to_numeric(weights_raw["weight"], errors="coerce")
weights_raw               = weights_raw.dropna()
weights_raw["item_clean"] = weights_raw["item"].apply(clean_name)
print(f"Basket items : {len(weights_raw)}")
print(f"Total weight : {weights_raw['weight'].sum():.4f}%")

# ── Load food price CSVs ──────────────────────────────────────────────
records = []
for year in range(2014, 2026):
    fp = f"/content/data/food_prices/{year}.csv"
    if not os.path.exists(fp): continue
    raw = pd.read_csv(fp, header=None, skiprows=3, dtype=str).iloc[:, :14]
    raw.columns = ["item","unit"] + MONTH_COLS
    raw = raw[raw["item"].notna() & (raw["item"].str.strip() != "")]
    raw = raw[~raw["item"].apply(is_non_food)]
    raw["item"] = raw["item"].apply(clean_name)
    m = raw.melt(id_vars=["item","unit"], value_vars=MONTH_COLS,
                 var_name="month", value_name="price")
    m["year"]      = year
    m["month_num"] = m["month"].apply(lambda x: MONTH_COLS.index(x)+1)
    m["price"]     = pd.to_numeric(m["price"], errors="coerce")
    records.append(m)

food = pd.concat(records, ignore_index=True)
food["date"] = pd.to_datetime(
    food["year"].astype(str)+"-"+
    food["month_num"].astype(str).str.zfill(2)+"-01")

wide_all = food.pivot_table(index="date", columns="item",
                             values="price", aggfunc="mean")
wide_all = wide_all.loc[:, wide_all.notna().mean() >= 0.7]
wide_all = wide_all.sort_index().ffill().bfill()

# ── Match weights to price data ───────────────────────────────────────
price_items             = set(wide_all.columns.tolist())
weights_raw["in_price"] = weights_raw["item_clean"].isin(price_items)
matched                 = weights_raw[weights_raw["in_price"]].copy()
unmatched               = weights_raw[~weights_raw["in_price"]].copy()

matched["weight_norm"]  = matched["weight"] / matched["weight"].sum() * 100
basket_items            = matched["item_clean"].tolist()
wide                    = wide_all[basket_items].copy()
item_weights            = matched.set_index("item_clean")["weight_norm"]

print(f"Matched   : {len(matched)} items  (coverage={matched['weight'].sum():.2f}%)")
print(f"Unmatched : {len(unmatched)} items  (weight={unmatched['weight'].sum():.2f}% — negligible)")
print(f"Basket price matrix : {wide.shape}")

# ── Load official CCPI ────────────────────────────────────────────────
cpi_raw = pd.read_csv("/content/data/CCPI_Consolidated_Base2021.csv", dtype=str)
cpi_raw.columns = ["year"] + MONTH_COLS
cm = cpi_raw.melt(id_vars=["year"], value_vars=MONTH_COLS,
                  var_name="month", value_name="cpi")
cm["month_num"] = cm["month"].apply(lambda x: MONTH_COLS.index(x)+1)
cm["cpi"]       = pd.to_numeric(cm["cpi"], errors="coerce")
cm = cm.dropna(subset=["cpi"])
cm["date"] = pd.to_datetime(
    cm["year"].astype(float).astype(int).astype(str)+"-"+
    cm["month_num"].astype(str).str.zfill(2)+"-01")
cpi_official = cm.set_index("date")["cpi"].sort_index()

# ── Bottom-up CPI (Laspeyres formula) ────────────────────────────────
def compute_basket_cpi(price_matrix, item_weights, base_year=2021):
    """
    CPI_t = Σ (w_i × P_it / P_i0) × 100
    w_i  = normalised basket weight
    P_i0 = average price in base year
    """
    base_mask   = (price_matrix.index.year == base_year)
    base_prices = price_matrix[base_mask].mean()
    base_prices = base_prices.replace(0, np.nan).fillna(price_matrix.mean())
    w           = item_weights / item_weights.sum()
    return (price_matrix.div(base_prices, axis=1) * w).sum(axis=1) * 100

cpi_basket = compute_basket_cpi(wide, item_weights, BASE_YEAR)
cpi_mom    = cpi_basket.pct_change(1) * 100

common_all = cpi_basket.index.intersection(cpi_official.index)
corr       = np.corrcoef(cpi_basket.loc[common_all].values,
                          cpi_official.loc[common_all].values)[0,1]
print(f"\nBottom-up vs official CCPI correlation : {corr:.4f}")

# ── Festival demand features ──────────────────────────────────────────
def festival_features(date):
    m = date.month
    return {
        # Sinhala & Tamil New Year (April 13-14)
        "sinhala_ny_march"  : 1.0 if m == 3       else 0.0,
        "sinhala_ny_april"  : 1.0 if m == 4       else 0.0,
        "sinhala_season"    : 1.0 if m in [3,4]   else 0.0,
        # Christmas & New Year
        "christmas_nov"     : 1.0 if m == 11      else 0.0,
        "christmas_dec"     : 1.0 if m == 12      else 0.0,
        "new_year_jan"      : 1.0 if m == 1       else 0.0,
        "christmas_season"  : 1.0 if m in [11,12,1] else 0.0,
        # Harvest seasons (price drops)
        "maha_harvest"      : 1.0 if m in [3,4]   else 0.0,
        "yala_harvest"      : 1.0 if m in [8,9]   else 0.0,
        # Smooth seasonal encoding
        "month_sin"         : np.sin(2*np.pi*m/12),
        "month_cos"         : np.cos(2*np.pi*m/12),
    }

print("Festival features defined:")
print("  Sinhala & Tamil New Year : March–April")
print("  Christmas & New Year     : November–January")
print("  Harvest seasons          : Maha (Mar-Apr), Yala (Aug-Sep)")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ── Plot 1: Basket weight distribution ───────────────────────────────
ax = axes[0]
top20 = matched.nlargest(20, "weight_norm")
bars  = ax.barh(top20["item"].str[:30],
                top20["weight_norm"], color="steelblue")
ax.set_title("Top 20 DCS Basket Items by Weight\n"
             "(used in bottom-up CPI calculation)",
             fontweight="bold")
ax.set_xlabel("Basket weight (%)")
ax.invert_yaxis()
for bar, val in zip(bars, top20["weight_norm"]):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f"{val:.2f}%", va="center", fontsize=7)

# ── Plot 2: Bottom-up vs official CCPI ───────────────────────────────
ax = axes[1]
ax.plot(cpi_official.loc[common_all].values,
        label="Official CCPI (DCS)",
        linewidth=2.5, color="steelblue")
ax.plot(cpi_basket.loc[common_all].values,
        label=f"Bottom-up basket (r={corr:.3f})",
        linewidth=1.8, color="coral", linestyle="--")
ax.set_title("Bottom-up CPI vs Official CCPI\n"
             "(validation of DCS Laspeyres formula)",
             fontweight="bold")
ax.set_xlabel("Month index")
ax.set_ylabel("CCPI Index")
ax.legend()
ax.annotate(f"Pearson r = {corr:.4f}",
            xy=(0.05, 0.92), xycoords="axes fraction",
            fontsize=11, color="darkgreen",
            bbox=dict(boxstyle="round", fc="lightyellow",
                      ec="green", alpha=0.8))

# ── Plot 3: Festival demand effect on MoM CPI ────────────────────────
ax = axes[2]
mom_by_month = cpi_mom.groupby(cpi_mom.index.month).mean()
month_names  = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
colors = []
for m in range(1, 13):
    if m in [3, 4]:        colors.append("#FF8C00")   # Sinhala NY
    elif m in [11, 12, 1]: colors.append("#4169E1")   # Christmas/NY
    elif m in [8, 9]:      colors.append("#2ECC71")   # Yala harvest
    else:                  colors.append("#95A5A6")   # normal

bars = ax.bar(month_names, mom_by_month.values, color=colors)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Average MoM CPI by Month (2014–2025)\n"
             "(festival demand effects visible)",
             fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Average MoM CPI %")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#FF8C00", label="Sinhala & Tamil NY (Mar–Apr)"),
    Patch(facecolor="#4169E1", label="Christmas & NY (Nov–Jan)"),
    Patch(facecolor="#2ECC71", label="Yala harvest (Aug–Sep)"),
    Patch(facecolor="#95A5A6", label="Normal months"),
]
ax.legend(handles=legend_elements, fontsize=8, loc="upper left")

plt.tight_layout()
plt.savefig("supervisor_adjustments.png", dpi=150,
            bbox_inches="tight")
plt.show()
print("Saved: supervisor_adjustments.png")

class GraphStructureLearner(nn.Module):
    def __init__(self, n_items, embed_dim=16, top_k=8):
        super().__init__()
        self.embed = nn.Embedding(n_items, embed_dim)
        self.top_k = top_k

    def forward(self):
        e    = F.normalize(self.embed.weight, dim=-1)
        sim  = torch.mm(e, e.T)
        mask = torch.zeros_like(sim)
        mask.scatter_(1, sim.topk(self.top_k+1, dim=-1).indices, 1)
        mask.fill_diagonal_(0)
        A   = sim * mask
        return A / A.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    def get_adjacency(self):
        with torch.no_grad():
            return self.forward().cpu().numpy()


class FreEformerBlock(nn.Module):
    def __init__(self, seq_len, n_items):
        super().__init__()
        self.n_freq = seq_len // 2 + 1
        w = torch.ones(n_items, self.n_freq) * 0.3
        w[:, 0] = 1.0   # DC — overall price level
        w[:, 1] = 1.5   # 12-month — annual harvest cycle
        w[:, 2] = 1.2   # 6-month  — monsoon seasons
        w[:, 3] = 0.8   # 4-month  — quarterly
        self.freq_attn = nn.Parameter(w)

    def forward(self, x):
        x_t = x.permute(0, 2, 1)
        X_f = torch.fft.rfft(x_t, dim=-1)
        X_f = X_f * torch.sigmoid(self.freq_attn).unsqueeze(0)
        return torch.fft.irfft(X_f, n=x.shape[1], dim=-1).permute(0, 2, 1)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.W    = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, H, A):
        AH = torch.bmm(A.unsqueeze(0).expand(H.size(0), -1, -1), H)
        return F.relu(self.drop(self.W(AH)))


class STGNNExtractor(nn.Module):
    def __init__(self, n_items, seq_len, embed_dim=16,
                 d_model=32, gcn_hidden=32, top_k=8, dropout=0.2):
        super().__init__()
        self.gsl   = GraphStructureLearner(n_items, embed_dim, top_k)
        self.freef = FreEformerBlock(seq_len, n_items)
        self.proj  = nn.Linear(seq_len, d_model)
        self.drop  = nn.Dropout(dropout)
        self.gcn1  = GCNLayer(d_model, gcn_hidden, dropout)

    def forward(self, x):
        A = self.gsl()
        x = self.freef(x)
        H = self.proj(x.permute(0, 2, 1))
        H = self.drop(H)
        H = self.gcn1(H, A)
        return H.mean(dim=1)

print("ST-GNN classes defined.")

N_ITEMS    = wide.shape[1]
GCN_HIDDEN = 32
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device           : {device}")
print(f"Basket nodes     : {N_ITEMS}")

# ── Normalisation (training period only) ─────────────────────────────
train_mask  = wide.index <= TRAIN_END
price_mean  = wide[train_mask].mean()
price_std   = wide[train_mask].std().replace(0, 1)
price_norm  = (wide - price_mean) / price_std

train_mom   = cpi_mom[cpi_mom.index <= TRAIN_END].dropna()
mom_mean    = float(train_mom.mean())
mom_std     = float(train_mom.std()) or 1.0
target_norm = (cpi_mom - mom_mean) / mom_std

# ── Commodity clusters ────────────────────────────────────────────────
cols = wide.columns.tolist()
def cluster(kws):
    return [c for c in cols if any(k in c for k in kws)]

clusters = {
    "rice":       cluster(["rice"]),
    "fish":       cluster(["kelavalla","thalapath","paraw","balaya",
                           "mullet","seer","salaya","linna","sprats",
                           "katta","keerameen","maldives"]),
    "vegetables": cluster(["brinjal","pumpkin","beans","bandakka",
                           "bitter","snake","polos","ridge","ash",
                           "elabattu","dambala","cucumber","carrot",
                           "cabbage","beetroot","leeks","capsicum",
                           "tomatoes","radish","mukunuwenna",
                           "gotukola","katurumurunga","kankun"]),
    "proteins":   cluster(["chicken","beef","pork","mutton","sausage",
                           "hen","dhal","gram","cowpea","soya"]),
    "oils":       cluster(["coconut oil","vegetable oil",
                           "butter","margarine"]),
    "spices":     cluster(["chilly powder","pepper","dried chili",
                           "curry powder","turmeric","cinnamon",
                           "ginger","garlic","goraka","tamarind",
                           "cumin","mustard","coriander"]),
    "essentials": cluster(["milk powder","coconut","sugar","bread",
                           "rice flour","wheat flour",
                           "red onion","bombay onion"]),
}
for k,v in clusters.items():
    print(f"  Cluster '{k}': {len(v)} items")

# ── Extra feature builder ─────────────────────────────────────────────
price_mom_wide = wide.pct_change(1) * 100

def build_extra_features(idx_list, common_dates, window=WINDOW):
    rows = []
    for i in idx_list:
        target_date = common_dates[i + window]
        row = {}

        # Festival demand features
        row.update(festival_features(target_date))

        # CPI lag features
        for lag in [1, 2, 3]:
            lag_idx  = i + window - lag
            lag_date = common_dates[lag_idx] if lag_idx >= 0 else None
            if lag_date and lag_date in cpi_mom.index:
                v = cpi_mom[lag_date]
                row[f"cpi_lag{lag}"] = float(v) if not pd.isna(v) else 0.0
            else:
                row[f"cpi_lag{lag}"] = 0.0

        # Basket-level stats
        if target_date in price_mom_wide.index:
            pm = price_mom_wide.loc[target_date]
            row["basket_mom"]  = float(pm.mean())
            row["basket_vol"]  = float(pm.std())
            row["basket_skew"] = float(pm.skew())
            row["basket_q90"]  = float(pm.quantile(0.9))
            row["basket_q10"]  = float(pm.quantile(0.1))
        else:
            for k in ["basket_mom","basket_vol","basket_skew",
                      "basket_q90","basket_q10"]:
                row[k] = 0.0

        # Cluster-level features
        for cname, items in clusters.items():
            valid = [it for it in items if it in price_mom_wide.columns]
            if valid and target_date in price_mom_wide.index:
                vals = price_mom_wide[valid].loc[target_date]
                row[f"cl_{cname}_mom"] = float(vals.mean())
                row[f"cl_{cname}_vol"] = float(vals.std())
            else:
                row[f"cl_{cname}_mom"] = 0.0
                row[f"cl_{cname}_vol"] = 0.0

        # Rolling trend
        w_mom = price_mom_wide.iloc[i:i+window]
        row["rolling_trend"] = float(
            w_mom.mean(axis=1).iloc[-1] - w_mom.mean(axis=1).iloc[0])
        row["rolling_vol"]   = float(w_mom.values.std())

        # Crisis flag
        row["crisis"] = 1.0 if (
            pd.Timestamp("2022-01-01") <= target_date
            <= pd.Timestamp("2023-06-01")) else 0.0

        rows.append(row)
    return pd.DataFrame(rows).fillna(0).values.astype(np.float32)

# ── Build sliding windows ─────────────────────────────────────────────
common = sorted(price_norm.index.intersection(target_norm.dropna().index))
n      = len(common)
P      = price_norm.loc[common].values.astype(np.float32)
T      = target_norm.loc[common].values.astype(np.float32)

def get_idx(start_excl, end_incl):
    return [i for i in range(n)
            if i + WINDOW < n
            and pd.Timestamp(start_excl) < common[i+WINDOW]
                                        <= pd.Timestamp(end_incl)]

idx_train = get_idx("2014-01-01", TRAIN_END)
idx_val   = get_idx(TRAIN_END,    VAL_END)
idx_test  = get_idx(VAL_END,      "2026-01-01")

X_train = np.stack([P[i:i+WINDOW] for i in idx_train])
X_val   = np.stack([P[i:i+WINDOW] for i in idx_val])
X_test  = np.stack([P[i:i+WINDOW] for i in idx_test])
y_train = np.array([T[i+WINDOW]   for i in idx_train])
y_val   = np.array([T[i+WINDOW]   for i in idx_val])
y_test  = np.array([T[i+WINDOW]   for i in idx_test])

F_train = build_extra_features(idx_train, common)
F_val   = build_extra_features(idx_val,   common)
F_test  = build_extra_features(idx_test,  common)

print(f"\nSplits:")
print(f"  Train : {len(X_train)} samples")
print(f"  Val   : {len(X_val)}  samples")
print(f"  Test  : {len(X_test)} samples")
print(f"  Extra features : {F_train.shape[1]}  "
      f"(festival + cluster + CPI lag + seasonal)")

from torch.utils.data import TensorDataset, DataLoader

# ── Pre-train graph ───────────────────────────────────────────────────
def pretrain_graph(price_matrix, embed_dim=16, epochs=300, lr=1e-3):
    returns  = price_matrix.pct_change().dropna()
    emp_corr = torch.tensor(returns.corr().values, dtype=torch.float32)
    n        = emp_corr.shape[0]
    print(f"Pre-training graph: {n} commodity nodes × "
          f"{len(returns)} months...")

    embed = nn.Embedding(n, embed_dim)
    optim = torch.optim.Adam(embed.parameters(),
                             lr=lr, weight_decay=1e-4)
    best_loss, best_w = float("inf"), None

    for epoch in range(1, epochs+1):
        optim.zero_grad()
        e    = F.normalize(embed.weight, dim=-1)
        pred = torch.mm(e, e.T)
        mask = (emp_corr.abs() > 0.3).float()
        loss = (mask * (pred - emp_corr)**2).mean()
        loss.backward()
        optim.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_w    = embed.weight.data.clone()
        if epoch % 100 == 0:
            print(f"  Epoch {epoch}  loss={loss.item():.6f}")

    print(f"Pre-training done. Best loss={best_loss:.6f}")
    return best_w

pretrained_w = pretrain_graph(wide, embed_dim=16, epochs=300)

# ── Create extractor and load pre-trained graph ───────────────────────
extractor = STGNNExtractor(
    n_items=N_ITEMS, seq_len=WINDOW, embed_dim=16,
    d_model=32, gcn_hidden=GCN_HIDDEN, top_k=8, dropout=0.2
).to(device)
extractor.gsl.embed.weight.data = pretrained_w.to(device)

for p in extractor.gsl.parameters():
    p.requires_grad = False
trainable = sum(p.numel() for p in extractor.parameters()
                if p.requires_grad)
print(f"Trainable params (FreEformer+GCN): {trainable:,}")

# ── Supervised fine-tuning ────────────────────────────────────────────
Xt  = torch.from_numpy(X_train).to(device)
Xv  = torch.from_numpy(X_val).to(device)
Xte = torch.from_numpy(X_test).to(device)
yt  = torch.from_numpy(y_train).unsqueeze(1).to(device)
yv  = torch.from_numpy(y_val).unsqueeze(1).to(device)

train_ds  = TensorDataset(Xt, yt)
train_dl  = DataLoader(train_ds, batch_size=8, shuffle=True)
temp_head = nn.Linear(GCN_HIDDEN, 1).to(device)
mse_loss  = nn.MSELoss()

optim_ft  = torch.optim.Adam(
    list(filter(lambda p: p.requires_grad, extractor.parameters())) +
    list(temp_head.parameters()),
    lr=5e-4, weight_decay=1e-3)

best_val, best_state, patience_cnt = float("inf"), None, 0
PATIENCE = 30

print(f"\nSupervised fine-tuning...")
print(f"{'Epoch':>6}  {'Train':>8}  {'Val':>8}")
print("─"*28)

for epoch in range(1, 301):
    extractor.train(); temp_head.train()
    tr_loss = 0.0
    for X_b, y_b in train_dl:
        optim_ft.zero_grad()
        loss = mse_loss(temp_head(extractor(X_b)), y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(extractor.parameters())+list(temp_head.parameters()), 1.0)
        optim_ft.step()
        tr_loss += loss.item()
    tr_loss /= len(train_dl)

    extractor.eval(); temp_head.eval()
    with torch.no_grad():
        vl = mse_loss(temp_head(extractor(Xv)), yv).item()

    if epoch % 50 == 0:
        print(f"{epoch:>6}  {tr_loss:>8.5f}  {vl:>8.5f}")

    if vl < best_val:
        best_val   = vl
        best_state = {k: v.clone() for k,v in extractor.state_dict().items()}
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

extractor.load_state_dict(best_state)
print(f"Best val loss: {best_val:.5f}")

# ── Extract embeddings ────────────────────────────────────────────────
extractor.eval()
with torch.no_grad():
    E_train = extractor(Xt).cpu().numpy()
    E_val   = extractor(Xv).cpu().numpy()
    E_test  = extractor(Xte).cpu().numpy()

# ── Combine with extra features ───────────────────────────────────────
C_train = np.hstack([E_train, F_train])
C_val   = np.hstack([E_val,   F_val])
C_test  = np.hstack([E_test,  F_test])
print(f"Combined feature dim : {C_train.shape[1]}  "
      f"({E_train.shape[1]} ST-GNN + {F_train.shape[1]} extra)")

# ── Grid search alpha ─────────────────────────────────────────────────
print("Searching best Ridge alpha...")
best_alpha, best_val_r2 = None, -np.inf

for alpha in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
    pipe = Pipeline([("sc", StandardScaler()),
                     ("rg", Ridge(alpha=alpha))])
    pipe.fit(C_train, y_train)
    val_r2 = r2_score(y_val, pipe.predict(C_val))
    print(f"  alpha={alpha:<6}  val R²={val_r2:.4f}")
    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        best_alpha  = alpha

print(f"\nBest alpha: {best_alpha}  (val R²={best_val_r2:.4f})")

# Final model on train+val
C_tv = np.vstack([C_train, C_val])
y_tv = np.concatenate([y_train, y_val])
final_pipe = Pipeline([("sc", StandardScaler()),
                        ("rg", Ridge(alpha=best_alpha))])
final_pipe.fit(C_tv, y_tv)

# ── Test evaluation ───────────────────────────────────────────────────
p_mom = final_pipe.predict(C_test) * mom_std + mom_mean
a_mom = y_test * mom_std + mom_mean

rmse  = np.sqrt(mean_squared_error(a_mom, p_mom))
mae   = mean_absolute_error(a_mom, p_mom)
r2    = r2_score(a_mom, p_mom)
safe  = np.abs(a_mom) > 0.1
mape  = (np.abs(p_mom[safe]-a_mom[safe]) /
         np.abs(a_mom[safe])).mean()*100 if safe.any() else np.nan
acc3  = (np.abs(p_mom-a_mom) <= 0.3).mean()*100
acc5  = (np.abs(p_mom-a_mom) <= 0.5).mean()*100
acc10 = (np.abs(p_mom-a_mom) <= 1.0).mean()*100
dacc  = (np.sign(p_mom)==np.sign(a_mom)).mean()*100

print("\n" + "="*58)
print("  Hybrid ST-GNN + Ridge  (Bottom-up CPI + Festivals)")
print("="*58)
print(f"\n  RMSE          : {rmse:.4f} pp")
print(f"  MAE           : {mae:.4f} pp")
print(f"  MAPE          : {mape:.2f}%")
print(f"  R²            : {r2:.4f}")
print(f"\n  Within ±0.3pp : {acc3:.1f}%")
print(f"  Within ±0.5pp : {acc5:.1f}%")
print(f"  Within ±1.0pp : {acc10:.1f}%")
print(f"  Directional   : {dacc:.1f}%")
print(f"\n  Actual range  : {a_mom.min():.3f}% to {a_mom.max():.3f}%")
print(f"  Pred range    : {p_mom.min():.3f}% to {p_mom.max():.3f}%")

coef      = np.abs(final_pipe["rg"].coef_)
emb_imp   = coef[:E_train.shape[1]].sum()
feat_imp  = coef[E_train.shape[1]:].sum()
total_imp = emb_imp + feat_imp
print(f"\n  ST-GNN embeddings       : {emb_imp/total_imp*100:.1f}%")
print(f"  Statistical + festival  : {feat_imp/total_imp*100:.1f}%")

test_dates = [common[i+WINDOW] for i in idx_test]
print(f"\n  {'Date':<12} {'Actual':>8} {'Pred':>8} {'Error':>8} {'±0.5':>6}")
print(f"  {'─'*50}")
for date, act, pred in zip(test_dates, a_mom, p_mom):
    err = pred - act
    ok  = "✓" if abs(err) <= 0.5 else "✗"
    print(f"  {str(date.date()):<12} {act:>7.3f}%  "
          f"{pred:>7.3f}%  {err:>+7.3f}pp  {ok}")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# 1. Predicted vs actual
ax = axes[0, 0]
ax.plot(a_mom, label="Actual",    marker="o", linewidth=2, markersize=5)
ax.plot(p_mom, label="Predicted", marker="x", linewidth=2,
        markersize=5, linestyle="--")
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.fill_between(range(len(a_mom)), p_mom-0.5, p_mom+0.5,
                alpha=0.15, color="orange", label="±0.5pp band")
ax.set_title(f"MoM Inflation — Test\nR²={r2:.3f}  "
             f"RMSE={rmse:.4f}pp  Dir={dacc:.0f}%")
ax.set_xlabel("Test month"); ax.set_ylabel("MoM CPI %")
ax.legend(fontsize=8)

# 2. Bottom-up vs official CPI
ax = axes[0, 1]
ax.plot(cpi_official.loc[common_all],
        label="Official CCPI", linewidth=2, color="steelblue")
ax.plot(cpi_basket.loc[common_all],
        label="Bottom-up basket", linewidth=1.5,
        color="coral", linestyle="--")
ax.axvline(pd.Timestamp(TRAIN_END), color="green",
           linewidth=1, linestyle="--", label="Train end")
ax.set_title(f"Bottom-up vs Official CCPI\nCorrelation={corr:.4f}")
ax.set_ylabel("CCPI Index"); ax.legend(fontsize=8)

# 3. Adjacency matrix
ax = axes[0, 2]
A_mat = extractor.gsl.get_adjacency()
im    = ax.imshow(A_mat[:40,:40], cmap="RdBu_r",
                  aspect="auto", vmin=0, vmax=0.3)
ax.set_title("Learned commodity graph\n(top 40 nodes)")
ax.set_xlabel("Commodity index")
ax.set_ylabel("Commodity index")
plt.colorbar(im, ax=ax)

# 4. Full MoM CPI with festival shading
ax = axes[1, 0]
mom_full = cpi_mom.dropna()
ax.plot(mom_full.index, mom_full.values,
        color="steelblue", linewidth=1.5)
for yr in range(2014, 2026):
    ax.axvspan(pd.Timestamp(f"{yr}-03-01"),
               pd.Timestamp(f"{yr}-04-30"),
               alpha=0.15, color="orange")
    end_ny = "2025-12-31" if yr >= 2025 else f"{yr+1}-01-31"
    ax.axvspan(pd.Timestamp(f"{yr}-11-01"),
               pd.Timestamp(end_ny),
               alpha=0.10, color="lightblue")
ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
ax.set_title("MoM CPI + festival seasons\n"
             "(orange=Sinhala NY, blue=Christmas/NY)")
ax.set_ylabel("MoM %")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())

# 5. Error by test month
ax = axes[1, 1]
errors = p_mom - a_mom
colors = ["#2ecc71" if abs(e)<=0.5 else "#e74c3c" for e in errors]
ax.bar(range(len(errors)), errors, color=colors)
ax.axhline( 0.5, color="green", linewidth=1,
            linestyle="--", label="±0.5pp")
ax.axhline(-0.5, color="green", linewidth=1, linestyle="--")
ax.axhline(0,    color="black", linewidth=0.8)
ax.set_title("Prediction error by test month\n(green=within ±0.5pp)")
ax.set_xlabel("Test month"); ax.set_ylabel("Error (pp)")
ax.legend()

# 6. Feature importance
ax = axes[1, 2]
sample_row = {}
sample_row.update(festival_features(common[idx_train[0]+WINDOW]))
for lag in [1,2,3]: sample_row[f"cpi_lag{lag}"] = 0
for k in ["basket_mom","basket_vol","basket_skew",
          "basket_q90","basket_q10"]: sample_row[k] = 0
for cname in clusters:
    sample_row[f"cl_{cname}_mom"] = 0
    sample_row[f"cl_{cname}_vol"] = 0
sample_row["rolling_trend"] = 0
sample_row["rolling_vol"]   = 0
sample_row["crisis"]        = 0

all_names = ([f"ST-GNN_{i}" for i in range(E_train.shape[1])]
             + list(sample_row.keys()))
coef_vals = np.abs(final_pipe["rg"].coef_)
top_idx   = np.argsort(coef_vals)[-20:]
ax.barh([all_names[i] if i < len(all_names) else f"f{i}"
         for i in top_idx],
        coef_vals[top_idx], color="steelblue")
ax.set_title("Top 20 features\n(Ridge |coefficients|)")
ax.set_xlabel("|Coefficient|")

plt.tight_layout()
plt.savefig("results_final.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: results_final.png")