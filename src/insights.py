"""
insights.py
===========
Intelligent Insight Engine
--------------------------
Converts raw analytics into human-readable, actionable financial insights.
Mimics what CRED / Mint display on their insight cards.

Insight types:
  - Overspending alerts
  - Category comparisons (MoM)
  - Savings warnings
  - Prediction-based advice
  - Anomaly alerts
  - Positive reinforcement
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Dict


# ════════════════════════════════════════════════════════════════
# HELPER
# ════════════════════════════════════════════════════════════════

def _pct_change(new: float, old: float) -> float:
    return round((new - old) / old * 100, 1) if old else 0.0


def _fmt(n: float) -> str:
    """Format rupee amounts nicely."""
    return f"₹{n:,.0f}"


# ════════════════════════════════════════════════════════════════
# 1. MONTH-OVER-MONTH CATEGORY INSIGHTS
# ════════════════════════════════════════════════════════════════

def mom_category_insights(df_exp: pd.DataFrame, n_months: int = 2) -> List[str]:
    """
    Compare the last two months for each category and surface
    significant changes (>15%).
    """
    insights = []
    monthly_cat = (df_exp.groupby(["Month_Year", "Category"])["Amount"]
                   .sum().reset_index())
    months = sorted(monthly_cat["Month_Year"].unique())

    if len(months) < 2:
        return ["Not enough data for month-over-month comparison."]

    last, prev = months[-1], months[-2]

    for cat in monthly_cat["Category"].unique():
        curr_amt = monthly_cat.query("Month_Year==@last and Category==@cat")["Amount"]
        prev_amt = monthly_cat.query("Month_Year==@prev and Category==@cat")["Amount"]

        curr_val = float(curr_amt.values[0]) if not curr_amt.empty else 0
        prev_val = float(prev_amt.values[0]) if not prev_amt.empty else 0

        if prev_val == 0:
            continue

        pct = _pct_change(curr_val, prev_val)

        if pct > 30:
            insights.append(
                f"⚠️  Your **{cat}** spend jumped by {pct}% in {last} "
                f"({_fmt(prev_val)} → {_fmt(curr_val)}). Review if this was planned."
            )
        elif pct > 15:
            insights.append(
                f"📈 **{cat}** spending increased by {pct}% compared to {prev} "
                f"({_fmt(curr_val)} vs {_fmt(prev_val)})."
            )
        elif pct < -20:
            insights.append(
                f"✅ Great! You reduced **{cat}** expenses by {abs(pct)}% "
                f"({_fmt(prev_val)} → {_fmt(curr_val)}). Keep it up!"
            )

    return insights if insights else ["✅ Spending patterns are stable across categories."]


# ════════════════════════════════════════════════════════════════
# 2. OVERSPENDING INSIGHTS
# ════════════════════════════════════════════════════════════════

def overspending_insights(overspend_df: pd.DataFrame) -> List[str]:
    """Generate insights from the overspending detection output."""
    insights = []
    flagged  = overspend_df[overspend_df["Overspent"]].head(5)

    for _, row in flagged.iterrows():
        insights.append(
            f"🚨 **{row['Category']}** in {row['Month_Year']}: "
            f"You spent {_fmt(row['Amount'])} — that's {row['Pct_Deviation']:.1f}% "
            f"above your average ({_fmt(row['Historical_Mean'])})."
        )

    if not insights:
        insights.append("✅ No significant overspending detected across any category.")
    return insights


# ════════════════════════════════════════════════════════════════
# 3. SAVINGS INSIGHTS
# ════════════════════════════════════════════════════════════════

def savings_insights(summary: pd.DataFrame) -> List[str]:
    """Generate savings-focused insights from monthly summary."""
    insights = []
    avg_rate = summary["Savings_Rate"].mean()
    neg_months = summary[summary["Savings"] < 0]

    if avg_rate < 10:
        insights.append(
            f"⚠️ Your average savings rate is only {avg_rate:.1f}%. "
            "Financial experts recommend saving at least 20% of income. "
            "Try the 50/30/20 rule: 50% needs, 30% wants, 20% savings."
        )
    elif avg_rate < 20:
        insights.append(
            f"📊 Your savings rate is {avg_rate:.1f}% — decent, but you can do better! "
            "Target 20–30% to build a strong emergency fund."
        )
    else:
        insights.append(
            f"🌟 Excellent! You're saving {avg_rate:.1f}% of your income on average. "
            "Consider investing surplus savings in mutual funds or SIPs."
        )

    if len(neg_months) > 0:
        months_list = ", ".join(neg_months["Month_Year"].tolist()[:3])
        insights.append(
            f"🔴 You spent more than you earned in {len(neg_months)} month(s): "
            f"{months_list}. Identify what drove these deficits."
        )

    best   = summary.loc[summary["Savings"].idxmax()]
    worst  = summary.loc[summary["Savings"].idxmin()]
    insights.append(
        f"📅 Your best savings month was **{best['Month_Year']}** "
        f"({_fmt(best['Savings'])}), and the worst was **{worst['Month_Year']}** "
        f"({_fmt(worst['Savings'])})."
    )
    return insights


# ════════════════════════════════════════════════════════════════
# 4. PREDICTION-BASED INSIGHTS
# ════════════════════════════════════════════════════════════════

def prediction_insights(next_pred: float, summary: pd.DataFrame) -> List[str]:
    """Generate forward-looking insights from ML prediction."""
    insights = []
    avg_exp  = summary["Total_Expense"].mean()
    avg_inc  = summary["Total_Income"].mean()
    pct_diff = _pct_change(next_pred, avg_exp)

    if next_pred > avg_inc:
        insights.append(
            f"🚨 **High Risk!** Next month's predicted expense ({_fmt(next_pred)}) "
            f"exceeds your typical income ({_fmt(avg_inc)}). "
            "Immediate budget review recommended!"
        )
    elif pct_diff > 15:
        insights.append(
            f"⚠️ Next month is predicted to be an **above-average spending month** "
            f"({_fmt(next_pred)} vs usual {_fmt(avg_exp)} — +{pct_diff}%). "
            "Plan your discretionary spending carefully."
        )
    elif pct_diff < -10:
        insights.append(
            f"✅ Predicted spend for next month ({_fmt(next_pred)}) is "
            f"{abs(pct_diff)}% below your average. Potential for higher savings!"
        )
    else:
        insights.append(
            f"📊 Next month's expenses are predicted to be around {_fmt(next_pred)}, "
            "consistent with your recent trend."
        )
    return insights


# ════════════════════════════════════════════════════════════════
# 5. ANOMALY INSIGHTS
# ════════════════════════════════════════════════════════════════

def anomaly_insights(anomaly_df: pd.DataFrame) -> List[str]:
    """Surface the top unusual transactions as insights."""
    insights = []
    if anomaly_df.empty:
        return ["✅ No unusual transactions detected in your spending history."]

    insights.append(
        f"🔍 **{len(anomaly_df)} unusual transactions** were flagged by our AI engine:"
    )
    for _, row in anomaly_df.head(4).iterrows():
        insights.append(
            f"   • {row['Date'].strftime('%d %b %Y')} | {row['Category']} | "
            f"{_fmt(row['Amount'])} — *{row['Description']}*"
        )
    return insights


# ════════════════════════════════════════════════════════════════
# 6. BUDGET RECOMMENDATION SYSTEM  (WOW FEATURE)
# ════════════════════════════════════════════════════════════════

def budget_recommendations(df_exp: pd.DataFrame,
                            monthly_income: float = 75000) -> pd.DataFrame:
    """
    Generates a recommended budget for each category using the
    historical spend ratio, capped to leave 20% savings.

    Returns: DataFrame with Category | Avg_Spend | Recommended_Budget | Status
    """
    cat_avg = (df_exp.groupby("Category")["Amount"]
               .mean().reset_index(name="Avg_Monthly_Spend"))

    total_avg = cat_avg["Avg_Monthly_Spend"].sum()
    budget_pool = monthly_income * 0.80   # leave 20% for savings

    cat_avg["Recommended_Budget"] = (
        (cat_avg["Avg_Monthly_Spend"] / total_avg) * budget_pool
    ).round(2)

    cat_avg["Variance"] = (
        cat_avg["Avg_Monthly_Spend"] - cat_avg["Recommended_Budget"]
    ).round(2)

    cat_avg["Status"] = cat_avg["Variance"].apply(
        lambda v: "🔴 Over Budget" if v > 0 else "✅ Under Budget"
    )

    return cat_avg.sort_values("Avg_Monthly_Spend", ascending=False).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════
# 7. MASTER INSIGHT AGGREGATOR
# ════════════════════════════════════════════════════════════════

def generate_all_insights(df_exp: pd.DataFrame,
                           summary: pd.DataFrame,
                           overspend_df: pd.DataFrame,
                           anomaly_df: pd.DataFrame,
                           next_pred: float) -> Dict[str, List[str]]:
    """
    Aggregate all insights into a single structured dictionary.
    Used by both the Streamlit app and the main script.
    """
    return {
        "Month-over-Month":   mom_category_insights(df_exp),
        "Overspending":       overspending_insights(overspend_df),
        "Savings":            savings_insights(summary),
        "Predictions":        prediction_insights(next_pred, summary),
        "Anomalies":          anomaly_insights(anomaly_df),
    }
