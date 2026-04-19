"""
visualizations.py
=================
All charts and visualizations for the Expense Tracker App.
Uses Matplotlib + Seaborn for static charts (saved to /outputs).
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
warnings.filterwarnings("ignore")

# ── Global style ──────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":       120,
    "figure.facecolor": "#0F1117",
    "axes.facecolor":   "#1A1D27",
    "axes.edgecolor":   "#3A3D4D",
    "axes.labelcolor":  "#E0E0E0",
    "xtick.color":      "#A0A0B0",
    "ytick.color":      "#A0A0B0",
    "text.color":       "#E0E0E0",
    "grid.color":       "#2A2D3D",
    "grid.linestyle":   "--",
    "font.family":      "DejaVu Sans",
})
PALETTE = ["#6C63FF", "#FF6584", "#43E97B", "#FFD166",
           "#EF476F", "#06D6A0", "#118AB2", "#FFB347",
           "#A8DADC", "#E63946"]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save(fig: plt.Figure, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  📊 Saved: {name}")
    return path


# ════════════════════════════════════════════════════════════════
# 1. CATEGORY PIE CHART
# ════════════════════════════════════════════════════════════════

def plot_category_pie(df_exp: pd.DataFrame) -> str:
    cat_totals = df_exp.groupby("Category")["Amount"].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 7))
    wedges, texts, autotexts = ax.pie(
        cat_totals.values,
        labels=cat_totals.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=PALETTE[:len(cat_totals)],
        pctdistance=0.82,
        wedgeprops={"linewidth": 2, "edgecolor": "#0F1117"},
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("#0F1117")
    ax.set_title("Category-wise Expense Distribution", fontsize=14,
                 fontweight="bold", pad=15, color="#E0E0E0")
    return _save(fig, "01_category_pie.png")


# ════════════════════════════════════════════════════════════════
# 2. MONTHLY EXPENSE vs INCOME BAR CHART
# ════════════════════════════════════════════════════════════════

def plot_monthly_expense_income(summary: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(14, 5))
    x  = np.arange(len(summary))
    w  = 0.38

    ax.bar(x - w / 2, summary["Total_Income"],   w, label="Income",  color="#43E97B", alpha=0.85)
    ax.bar(x + w / 2, summary["Total_Expense"],   w, label="Expense", color="#FF6584", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(summary["Month_Year"], rotation=60, ha="right", fontsize=7)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"₹{v/1000:.0f}k"))
    ax.set_title("Monthly Income vs Expense", fontsize=14, fontweight="bold", pad=12)
    ax.legend(framealpha=0.3)
    ax.grid(axis="y", alpha=0.4)
    return _save(fig, "02_monthly_income_expense.png")


# ════════════════════════════════════════════════════════════════
# 3. SAVINGS TREND LINE
# ════════════════════════════════════════════════════════════════

def plot_savings_trend(summary: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(summary["Month_Year"], summary["Savings"],
            color="#6C63FF", linewidth=2.5, marker="o", markersize=5)
    ax.fill_between(summary["Month_Year"], summary["Savings"], 0,
                    where=(summary["Savings"] >= 0),
                    alpha=0.25, color="#43E97B", label="Positive savings")
    ax.fill_between(summary["Month_Year"], summary["Savings"], 0,
                    where=(summary["Savings"] < 0),
                    alpha=0.25, color="#FF6584", label="Deficit")
    ax.axhline(0, color="#A0A0B0", linewidth=1, linestyle="--")
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(summary["Month_Year"], rotation=60, ha="right", fontsize=7)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"₹{v/1000:.0f}k"))
    ax.set_title("Monthly Savings Trend", fontsize=14, fontweight="bold", pad=12)
    ax.legend(framealpha=0.3)
    ax.grid(alpha=0.3)
    return _save(fig, "03_savings_trend.png")


# ════════════════════════════════════════════════════════════════
# 4. CATEGORY HORIZONTAL BAR
# ════════════════════════════════════════════════════════════════

def plot_category_bar(df_exp: pd.DataFrame) -> str:
    cat_totals = df_exp.groupby("Category")["Amount"].sum().sort_values()
    fig, ax    = plt.subplots(figsize=(10, 6))
    bars = ax.barh(cat_totals.index, cat_totals.values,
                   color=PALETTE[:len(cat_totals)], edgecolor="#0F1117", linewidth=0.8)

    for bar, val in zip(bars, cat_totals.values):
        ax.text(val + 500, bar.get_y() + bar.get_height() / 2,
                f"₹{val:,.0f}", va="center", fontsize=8, color="#E0E0E0")

    ax.set_title("Total Spending by Category", fontsize=14, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"₹{v/1000:.0f}k"))
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, "04_category_bar.png")


# ════════════════════════════════════════════════════════════════
# 5. HEATMAP — Monthly spending per category
# ════════════════════════════════════════════════════════════════

def plot_heatmap(df_exp: pd.DataFrame) -> str:
    pivot = (df_exp.groupby(["Month_Year", "Category"])["Amount"]
             .sum().unstack(fill_value=0))
    # Keep only last 12 months if more
    pivot = pivot.tail(12)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot.T, ax=ax, cmap="YlOrRd", fmt=".0f", annot=True,
                annot_kws={"size": 6},
                linewidths=0.5, linecolor="#0F1117",
                cbar_kws={"label": "Amount (₹)"})
    ax.set_title("Spending Heatmap — Category × Month", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Month", fontsize=9)
    ax.set_ylabel("Category", fontsize=9)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=8)
    return _save(fig, "05_spending_heatmap.png")


# ════════════════════════════════════════════════════════════════
# 6. PAYMENT METHOD DISTRIBUTION
# ════════════════════════════════════════════════════════════════

def plot_payment_methods(df_exp: pd.DataFrame) -> str:
    pm = df_exp.groupby("Payment_Method")["Amount"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(pm.index, pm.values, color=PALETTE[:len(pm)], edgecolor="#0F1117", linewidth=0.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"₹{v/1000:.0f}k"))
    ax.set_title("Spending by Payment Method", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Payment Method")
    ax.grid(axis="y", alpha=0.3)
    for i, (idx, val) in enumerate(pm.items()):
        ax.text(i, val + 500, f"₹{val/1000:.1f}k", ha="center", fontsize=8)
    return _save(fig, "06_payment_methods.png")


# ════════════════════════════════════════════════════════════════
# 7. WEEKEND vs WEEKDAY SPENDING
# ════════════════════════════════════════════════════════════════

def plot_weekend_vs_weekday(df_exp: pd.DataFrame) -> str:
    wk = df_exp.groupby("Is_Weekend")["Amount"].agg(["sum", "mean"])
    labels = ["Weekday", "Weekend"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(labels, wk["sum"],  color=["#6C63FF", "#FFD166"], edgecolor="#0F1117")
    axes[0].set_title("Total Spend", fontweight="bold")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"₹{v/1000:.0f}k"))

    axes[1].bar(labels, wk["mean"], color=["#43E97B", "#FF6584"], edgecolor="#0F1117")
    axes[1].set_title("Avg Transaction", fontweight="bold")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"₹{v:.0f}"))

    fig.suptitle("Weekend vs Weekday Spending", fontsize=13, fontweight="bold", y=1.02)
    return _save(fig, "07_weekend_vs_weekday.png")


# ════════════════════════════════════════════════════════════════
# 8. ML PREDICTION CHART
# ════════════════════════════════════════════════════════════════

def plot_prediction(summary: pd.DataFrame, next_pred: float) -> str:
    months    = list(summary["Month_Year"]) + ["Next Month"]
    actuals   = list(summary["Total_Expense"]) + [None]
    predicted = [None] * len(summary) + [next_pred]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(months[:-1], actuals[:-1], color="#6C63FF", marker="o",
            markersize=4, linewidth=2, label="Actual Expense")
    ax.plot(months[-2:],
            [actuals[-2], next_pred], color="#FFD166",
            marker="*", markersize=12, linewidth=2,
            linestyle="--", label=f"Predicted: ₹{next_pred:,.0f}")
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=60, ha="right", fontsize=7)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"₹{v/1000:.0f}k"))
    ax.set_title("Expense Trend + ML Prediction for Next Month",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(framealpha=0.3)
    ax.grid(alpha=0.3)
    return _save(fig, "08_ml_prediction.png")


# ════════════════════════════════════════════════════════════════
# 9. FINANCIAL HEALTH SCORE GAUGE
# ════════════════════════════════════════════════════════════════

def plot_health_score(score: float) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#FF4C4C", "#FF9800", "#FFD166", "#43E97B"]
    bands  = [0, 40, 60, 80, 100]

    for i in range(len(bands) - 1):
        ax.barh(0, bands[i + 1] - bands[i], left=bands[i], height=0.5,
                color=colors[i], alpha=0.7)

    ax.barh(0, min(score, 100), height=0.5, color="white", alpha=0.0)   # invisible
    ax.axvline(score, color="white", linewidth=3, linestyle="-")
    ax.text(score, 0.35, f"{score:.1f}", ha="center", fontsize=18,
            fontweight="bold", color="white")

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xticks([0, 40, 60, 80, 100])
    ax.set_xticklabels(["0\nPoor", "40\nFair", "60\nGood", "80\nExcellent", "100"])
    ax.set_title("Financial Health Score", fontsize=13, fontweight="bold", pad=12)
    ax.grid(False)
    return _save(fig, "09_health_score.png")


# ════════════════════════════════════════════════════════════════
# 10. SAVINGS RATE OVER TIME
# ════════════════════════════════════════════════════════════════

def plot_savings_rate(summary: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(summary["Month_Year"], summary["Savings_Rate"], 0,
                    alpha=0.35, color="#6C63FF")
    ax.plot(summary["Month_Year"], summary["Savings_Rate"],
            color="#6C63FF", linewidth=2, marker="o", markersize=4)
    ax.axhline(20, color="#43E97B", linewidth=1.5, linestyle="--",
               label="Target: 20%")
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(summary["Month_Year"], rotation=60, ha="right", fontsize=7)
    ax.set_title("Savings Rate Over Time (%)", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Savings Rate %")
    ax.legend(framealpha=0.3)
    ax.grid(alpha=0.3)
    return _save(fig, "10_savings_rate.png")


# ════════════════════════════════════════════════════════════════
# RUN ALL CHARTS
# ════════════════════════════════════════════════════════════════

def generate_all_charts(df_exp: pd.DataFrame,
                         summary: pd.DataFrame,
                         next_pred: float,
                         health_score: float) -> list:
    print("\n🎨 Generating visualizations...")
    paths = []
    paths.append(plot_category_pie(df_exp))
    paths.append(plot_monthly_expense_income(summary))
    paths.append(plot_savings_trend(summary))
    paths.append(plot_category_bar(df_exp))
    paths.append(plot_heatmap(df_exp))
    paths.append(plot_payment_methods(df_exp))
    paths.append(plot_weekend_vs_weekday(df_exp))
    paths.append(plot_prediction(summary, next_pred))
    paths.append(plot_health_score(health_score))
    paths.append(plot_savings_rate(summary))
    print(f"✅ {len(paths)} charts saved to /outputs/")
    return paths
