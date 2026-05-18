"""
visualize.py — Visualization module for food price forecasting system.
Sri Lanka Food Price Forecasting System

Generates:
  1. Historical price trends (multi-item line chart)
  2. Actual vs Predicted (per model on test set)
  3. Future forecast chart (with shading)
  4. Model comparison bar chart (RMSE)
  5. Correlation heatmap
  6. Seasonal decomposition plots
  7. Feature importance chart (tree-based models)

All charts saved to outputs/graphs/
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

from src.utils import get_logger, GRAPHS_DIR, sanitize_filename

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

# ─────────────────────────────────────────────
# Global Style
# ─────────────────────────────────────────────
PALETTE = "husl"
BG_COLOR = "#0f1117"
TEXT_COLOR = "#e0e0e0"
ACCENT = "#4fc3f7"
GRID_COLOR = "#2a2a3a"

def _apply_dark_style():
    """Applies a consistent dark theme to all matplotlib plots."""
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": "#16171f",
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.titlecolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "legend.facecolor": "#1e1f2e",
        "legend.edgecolor": GRID_COLOR,
        "font.family": "DejaVu Sans",
        "font.size": 10,
    })


class Visualizer:
    """
    Creates and saves all charts for the Sri Lankan food price forecasting system.

    Args:
        df: Cleaned DataFrame with DatetimeIndex.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        _apply_dark_style()
        logger.info("Visualizer initialized.")

    def _save(self, fig: plt.Figure, filename: str) -> Path:
        """Saves a figure to outputs/graphs/ and closes it."""
        path = GRAPHS_DIR / filename
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        plt.close(fig)
        logger.info(f"  Saved graph: {path.name}")
        return path

    # ─────────────────────────────────────────
    # 1. Historical Price Trends
    # ─────────────────────────────────────────
    def plot_historical_trends(self, targets: list[str], max_items: int = 8) -> Path:
        """
        Multi-line chart showing historical price trends for selected food items.

        Args:
            targets:   List of food item columns to plot.
            max_items: Maximum lines to show (avoids overcluttering).
        """
        items = targets[:max_items]
        colors = sns.color_palette(PALETTE, len(items))

        fig, ax = plt.subplots(figsize=(14, 6))
        for col, color in zip(items, colors):
            if col in self.df.columns:
                ax.plot(self.df.index, self.df[col], label=col, color=color, linewidth=1.5, alpha=0.85)

        ax.set_title("🇱🇰 Sri Lanka Food Price Trends (LKR)", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Month", fontsize=11)
        ax.set_ylabel("Price (LKR)", fontsize=11)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=45)
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return self._save(fig, "01_historical_trends.png")

    # ─────────────────────────────────────────
    # 2. Actual vs Predicted (Test Set)
    # ─────────────────────────────────────────
    def plot_actual_vs_predicted(
        self,
        target: str,
        predictions_df: pd.DataFrame,
    ) -> Path:
        """
        Overlays actual test values against all model predictions.

        Args:
            target:         Target food item name.
            predictions_df: DataFrame with 'Actual' + one column per model.
        """
        model_cols = [c for c in predictions_df.columns if c != "Actual"]
        colors = sns.color_palette(PALETTE, len(model_cols))

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(
            predictions_df.index,
            predictions_df["Actual"],
            label="Actual",
            color="white",
            linewidth=2.5,
            zorder=5,
        )

        for col, color in zip(model_cols, colors):
            ax.plot(
                predictions_df.index,
                predictions_df[col],
                label=col,
                color=color,
                linewidth=1.5,
                linestyle="--",
                alpha=0.85,
            )

        ax.set_title(
            f"Actual vs Predicted — {target} (Test Set 20%)",
            fontsize=13, fontweight="bold", pad=12
        )
        ax.set_xlabel("Month")
        ax.set_ylabel("Price (LKR)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        safe = sanitize_filename(target)
        return self._save(fig, f"02_actual_vs_predicted_{safe}.png")

    # ─────────────────────────────────────────
    # 3. Future Forecast Chart
    # ─────────────────────────────────────────
    def plot_future_forecast(
        self,
        target: str,
        forecast_df: pd.DataFrame,
        horizon: int,
    ) -> Path:
        """
        Plots historical prices + future forecast with a shaded forecast region.

        Args:
            target:      Target column name.
            forecast_df: DataFrame with columns ['Month', '{target} (Forecast)'].
            horizon:     Forecast horizon in months.
        """
        hist = self.df[target].dropna()
        fc_col = f"{target} (Forecast)"
        fc_vals = forecast_df[fc_col].values
        fc_dates = pd.to_datetime(forecast_df["Month"])

        fig, ax = plt.subplots(figsize=(14, 6))

        # Historical line
        ax.plot(hist.index, hist.values, color=ACCENT, linewidth=2, label="Historical")

        # Forecast line
        ax.plot(fc_dates, fc_vals, color="#ff7043", linewidth=2.5,
                linestyle="--", label=f"Forecast ({horizon}m)", zorder=5)

        # Confidence-style shading (±10% band as approximation)
        band = fc_vals * 0.10
        ax.fill_between(
            fc_dates,
            fc_vals - band,
            fc_vals + band,
            alpha=0.20,
            color="#ff7043",
            label="±10% band",
        )

        # Vertical separator
        ax.axvline(x=hist.index.max(), color="white", linestyle=":", linewidth=1.5, alpha=0.5)
        ax.text(
            hist.index.max(), ax.get_ylim()[1] * 0.95,
            " Forecast →", color="white", fontsize=9, alpha=0.7
        )

        ax.set_title(
            f"🔮 {target} — {horizon}-Month Future Forecast (LKR)",
            fontsize=13, fontweight="bold", pad=12
        )
        ax.set_xlabel("Month")
        ax.set_ylabel("Price (LKR)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.xticks(rotation=45)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        safe = sanitize_filename(target)
        return self._save(fig, f"03_forecast_{safe}_{horizon}m.png")

    # ─────────────────────────────────────────
    # 4. Model Comparison Bar Chart
    # ─────────────────────────────────────────
    def plot_model_comparison(self, metrics_df: pd.DataFrame, target: str) -> Path:
        """
        Horizontal bar chart comparing RMSE across all models for a target.

        Args:
            metrics_df: DataFrame with 'Model' and 'RMSE' columns.
            target:     Target name (used in title and filename).
        """
        df = metrics_df.sort_values("RMSE", ascending=True)
        colors = sns.color_palette(PALETTE, len(df))

        fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.7)))

        bars = ax.barh(df["Model"], df["RMSE"], color=colors, edgecolor="none", height=0.6)

        for bar, val in zip(bars, df["RMSE"]):
            ax.text(
                bar.get_width() + bar.get_width() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center",
                fontsize=9,
                color=TEXT_COLOR,
            )

        ax.set_title(
            f"Model Comparison — {target} (RMSE ↓ lower is better)",
            fontsize=12, fontweight="bold", pad=12
        )
        ax.set_xlabel("RMSE (LKR)")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()

        safe = sanitize_filename(target)
        return self._save(fig, f"04_model_comparison_{safe}.png")

    # ─────────────────────────────────────────
    # 5. Correlation Heatmap
    # ─────────────────────────────────────────
    def plot_correlation_heatmap(self, cols: list[str] | None = None) -> Path:
        """
        Heatmap showing Pearson correlation between food item prices.

        Args:
            cols: Subset of columns to include (defaults to all numeric).
        """
        df_corr = self.df[cols].corr() if cols else self.df.corr(numeric_only=True)

        # Shorten column names for readability
        short_names = {c: c[:20] for c in df_corr.columns}
        df_corr = df_corr.rename(columns=short_names, index=short_names)

        fig, ax = plt.subplots(figsize=(max(12, len(df_corr) * 0.6), max(10, len(df_corr) * 0.55)))

        mask = np.triu(np.ones_like(df_corr, dtype=bool))
        sns.heatmap(
            df_corr,
            mask=mask,
            annot=len(df_corr) <= 20,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1, vmax=1,
            linewidths=0.3,
            linecolor=GRID_COLOR,
            ax=ax,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 7},
        )

        ax.set_title("🔥 Food Price Correlation Matrix", fontsize=13, fontweight="bold", pad=15)
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.yticks(rotation=0, fontsize=7)
        fig.tight_layout()

        return self._save(fig, "05_correlation_heatmap.png")

    # ─────────────────────────────────────────
    # 6. Seasonal Decomposition
    # ─────────────────────────────────────────
    def plot_seasonal_decomposition(self, target: str) -> Path:
        """
        Decomposes the target time series into trend, seasonal, and residual components.
        Uses statsmodels seasonal_decompose with additive model.
        """
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            series = self.df[target].dropna()
            if len(series) < 24:
                logger.warning(f"  Too few data points for decomposition of '{target}' — skipping.")
                return None

            decomp = seasonal_decompose(series, model="additive", period=12)

            fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
            components = [
                (series, "Observed", ACCENT),
                (decomp.trend, "Trend", "#a5d6a7"),
                (decomp.seasonal, "Seasonal", "#ce93d8"),
                (decomp.resid, "Residual", "#ef9a9a"),
            ]

            for ax, (data, label, color) in zip(axes, components):
                ax.plot(data.index, data.values, color=color, linewidth=1.5)
                ax.set_ylabel(label, fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_facecolor("#16171f")

            axes[0].set_title(
                f"Seasonal Decomposition — {target}",
                fontsize=13, fontweight="bold", pad=10
            )
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            plt.xticks(rotation=45)
            fig.tight_layout()

            safe = sanitize_filename(target)
            return self._save(fig, f"06_seasonal_decomp_{safe}.png")

        except Exception as e:
            logger.error(f"  Seasonal decomposition failed for '{target}': {e}")
            return None

    # ─────────────────────────────────────────
    # 7. Feature Importance
    # ─────────────────────────────────────────
    def plot_feature_importance(
        self, model, model_name: str, feature_names: list[str], target: str
    ) -> Path:
        """
        Horizontal bar chart of feature importances for tree-based models.
        Supported: RandomForest, XGBoost, LightGBM, CatBoost.
        """
        importance = None

        try:
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
            elif hasattr(model, "get_feature_importance"):
                importance = model.get_feature_importance()
        except Exception:
            pass

        if importance is None:
            logger.info(f"  Feature importance not available for {model_name} — skipping.")
            return None

        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance,
        }).sort_values("Importance", ascending=True).tail(20)

        colors = sns.color_palette("viridis", len(fi_df))
        fig, ax = plt.subplots(figsize=(10, max(5, len(fi_df) * 0.4)))
        ax.barh(fi_df["Feature"], fi_df["Importance"], color=colors, edgecolor="none")
        ax.set_title(
            f"Feature Importance — {model_name} | {target}",
            fontsize=12, fontweight="bold", pad=12
        )
        ax.set_xlabel("Importance Score")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()

        safe_t = sanitize_filename(target)
        safe_m = sanitize_filename(model_name)
        return self._save(fig, f"07_feature_importance_{safe_t}_{safe_m}.png")

    # ─────────────────────────────────────────
    # 8. All-in-One Summary Dashboard (static)
    # ─────────────────────────────────────────
    def plot_price_dashboard(self, targets: list[str], n_cols: int = 3) -> Path:
        """
        Grid of individual price trend subplots — one per target food item.

        Args:
            targets: List of food item columns.
            n_cols:  Number of columns in the grid.
        """
        n = len(targets)
        n_rows = (n + n_cols - 1) // n_cols
        colors = sns.color_palette(PALETTE, n)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))
        axes = axes.flatten() if n > 1 else [axes]

        for i, (target, color) in enumerate(zip(targets, colors)):
            ax = axes[i]
            if target in self.df.columns:
                ax.plot(self.df.index, self.df[target], color=color, linewidth=1.2)
                ax.set_title(target[:25], fontsize=8, fontweight="bold")
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                ax.xaxis.set_major_locator(mdates.YearLocator(2))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=6)
                ax.grid(True, alpha=0.25)
                ax.set_facecolor("#16171f")

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            "🇱🇰 Sri Lanka Monthly Food Prices (LKR)",
            fontsize=14, fontweight="bold", y=1.01
        )
        fig.tight_layout()
        return self._save(fig, "08_price_dashboard.png")
