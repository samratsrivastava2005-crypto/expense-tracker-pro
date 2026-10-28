"""
analytics.py
============
Advanced Analytics Layer for the Expense Tracker App.

Provides:
  - Overspending detection (category + time-based)
  - Spending anomaly detection (IQR method)
  - Category-wise behavioural insights
  - Savings vs expense analysis
  - Financial Health Score (composite formula)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List


# ════════════════════════════════════════════════════════════════
# 1. DATA PREPARATION
# ════════════════════════════════════════════════════════════════

def load_and_prepare(expense_path: str, income_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load CSVs and add derived columns."""
    df_exp = pd.read_csv(expense_path, parse_dates=["Date"])
    df_inc = pd.read_csv(income_path,  parse_dates=["Date"])

    df_exp["Month_Year"] = df_exp["Date"].dt.to_period("M").astype(str)
    df_exp["DayOfWeek"]  = df_exp["Date"].dt.day_name()
    df_exp["Quarter"]    = df_exp["Date"].dt.quarter

    return df_exp, df_inc


# ════════════════════════════════════════════════════════════════
# 2. MONTHLY SUMMARY
# ════════════════════════════════════════════════════════════════

def monthly_summary(df_exp: pd.DataFrame, df_inc: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a month-level DataFrame:
        Month_Year | Total_Expense | Total_Income | Savings | Savings_Rate
    """
    exp_m = (df_exp.groupby("Month_Year")["Amount"]
             .sum().reset_index(name="Total_Expense"))

    inc_m = (df_inc.groupby(
                df_inc["Date"].dt.to_period("M").astype(str)
             )["Amount"].sum().reset_index())
    inc_m.columns = ["Month_Year", "Total_Income"]

    summary = pd.merge(exp_m, inc_m, on="Month_Year", how="left").fillna(0)
    summary["Savings"]      = summary["Total_Income"] - summary["Total_Expense"]
    
summary["Savings_Rate"] = np.where(
    summary["Total_Income"] != 0,
    (summary["Savings"] / summary["Total_Income"] * 100).round(2),
    np.nan
)

    summary = summary.sort_values("Month_Year").reset_index(drop=True)
    return summary


# ════════════════════════════════════════════════════════════════
# 3. OVERSPENDING DETECTION
# ════════════════════════════════════════════════════════════════

def detect_overspending(df_exp: pd.DataFrame,
                        threshold_pct: float = 20.0) -> pd.DataFrame:
    """
    Flags months where a category's spending exceeds its own historical
    mean by `threshold_pct` percent.

    Returns DataFrame with columns:
        Month_Year | Category | Amount | Historical_Mean | Pct_Deviation | Overspent
    """
    monthly_cat = (df_exp.groupby(["Month_Year", "Category"])["Amount"]
                   .sum().reset_index())

    cat_means = (monthly_cat.groupby("Category")["Amount"]
                 .mean().reset_index(name="Historical_Mean"))

    result = pd.merge(monthly_cat, cat_means, on="Category")
    result["Pct_Deviation"] = (
        (result["Amount"] - result["Historical_Mean"])
        / result["Historical_Mean"] * 100
    ).round(2)
    result["Overspent"] = result["Pct_Deviation"] > threshold_pct
    return result.sort_values("Pct_Deviation", ascending=False).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════
# 4. ANOMALY DETECTION  (IQR method — no ML dependency)
# ════════════════════════════════════════════════════════════════

def detect_anomalies(df_exp: pd.DataFrame) -> pd.DataFrame:
    """
    Flags individual transactions that are statistical outliers
    within their category using the IQR fence method.

    Outlier if: amount > Q3 + 1.5*IQR  for that category.
    """
    df = df_exp.copy()
    df["Is_Anomaly"] = False
    df["Anomaly_Reason"] = ""

    for cat, grp in df.groupby("Category"):
        q1 = grp["Amount"].quantile(0.25)
        q3 = grp["Amount"].quantile(0.75)
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr

        mask = (df["Category"] == cat) & (df["Amount"] > upper_fence)
        df.loc[mask, "Is_Anomaly"]     = True
        df.loc[mask, "Anomaly_Reason"] = (
            f"Amount > ₹{upper_fence:.0f} (Q3+1.5×IQR for {cat})"
        )

    anomalies = df[df["Is_Anomaly"]].sort_values("Amount", ascending=False)
    return anomalies.reset_index(drop=True)


# ════════════════════════════════════════════════════════════════
# 5. CATEGORY-WISE BEHAVIOURAL INSIGHTS
# ════════════════════════════════════════════════════════════════

def category_behaviour(df_exp: pd.DataFrame) -> pd.DataFrame:
    """
    Per-category statistics useful for behavioural profiling:
        Category | Total | Mean_Monthly | Pct_of_Total |
        Max_Month | Trend (slope sign)
    """
    monthly_cat = (df_exp.groupby(["Month_Year", "Category"])["Amount"]
                   .sum().unstack(fill_value=0))

    stats = []
    total_all = df_exp["Amount"].sum()

    for cat in monthly_cat.columns:
        series = monthly_cat[cat]
        total  = series.sum()
        mean_m = series.mean()
        pct    = round(total / total_all * 100, 2)
        max_m  = series.idxmax()

        # Simple linear trend (positive/negative/flat)
        x = np.arange(len(series))
        slope = np.polyfit(x, series.values, 1)[0]
        trend = "↑ Rising" if slope > 50 else ("↓ Falling" if slope < -50 else "→ Stable")

        stats.append({
            "Category":     cat,
            "Total_Spent":  round(total, 2),
            "Mean_Monthly": round(mean_m, 2),
            "Pct_of_Total": pct,
            "Peak_Month":   max_m,
            "Trend":        trend,
        })

    return (pd.DataFrame(stats)
              .sort_values("Total_Spent", ascending=False)
              .reset_index(drop=True))


# ════════════════════════════════════════════════════════════════
# 6. SAVINGS ANALYSIS
# ════════════════════════════════════════════════════════════════

def savings_analysis(summary: pd.DataFrame) -> Dict:
    """
    High-level savings KPIs from the monthly summary DataFrame.
    Returns a dict of scalar metrics.
    """
    avg_savings      = summary["Savings"].mean()
    avg_savings_rate = summary["Savings_Rate"].mean()
    negative_months  = (summary["Savings"] < 0).sum()
    best_month       = summary.loc[summary["Savings"].idxmax(), "Month_Year"]
    worst_month      = summary.loc[summary["Savings"].idxmin(), "Month_Year"]
    cumulative_saved = summary["Savings"].sum()

    return {
        "Average_Monthly_Savings":    round(avg_savings, 2),
        "Average_Savings_Rate_%":     round(avg_savings_rate, 2),
        "Negative_Savings_Months":    int(negative_months),
        "Best_Savings_Month":         best_month,
        "Worst_Savings_Month":        worst_month,
        "Cumulative_Savings":         round(cumulative_saved, 2),
    }


# ════════════════════════════════════════════════════════════════
# 7. FINANCIAL HEALTH SCORE  (0–100)
# ════════════════════════════════════════════════════════════════

def financial_health_score(summary: pd.DataFrame,
                            overspend_df: pd.DataFrame) -> Dict:
    """
    Composite Financial Health Score (0–100).

    Formula (weighted):
      ① Savings Rate Score   (40 pts)  — based on average savings rate
      ② Overspend Penalty    (30 pts)  — deducted for overspending months
      ③ Consistency Score    (20 pts)  — low variance in savings = consistent
      ④ Income Utilisation   (10 pts)  — keeping expenses < 80% of income

    Grades: 80–100 Excellent | 60–79 Good | 40–59 Fair | <40 Poor
    """
    # ① Savings Rate Score (max 40)
    avg_rate = summary["Savings_Rate"].mean()
    savings_score = min(40, max(0, avg_rate / 30 * 40))   # 30% rate → full marks

    # ② Overspend Penalty (max 30)
    overspend_months = overspend_df[overspend_df["Overspent"]]["Month_Year"].nunique()
    total_months     = summary.shape[0]
    overspend_ratio  = overspend_months / total_months if total_months else 0
    overspend_score  = 30 * (1 - overspend_ratio)

    # ③ Consistency Score (max 20)
    cv = summary["Savings"].std() / (summary["Savings"].mean() + 1e-9)
    consistency_score = max(0, 20 - cv * 10)

    # ④ Income Utilisation Score (max 10)
    avg_util = (summary["Total_Expense"] / (summary["Total_Income"] + 1e-9)).mean()
    util_score = 10 if avg_util < 0.8 else max(0, 10 - (avg_util - 0.8) * 50)

    total_score = round(savings_score + overspend_score + consistency_score + util_score, 1)
    total_score = min(100, max(0, total_score))

    if total_score >= 80:
        grade = "🟢 Excellent"
    elif total_score >= 60:
        grade = "🟡 Good"
    elif total_score >= 40:
        grade = "🟠 Fair"
    else:
        grade = "🔴 Poor"

    return {
        "Financial_Health_Score": total_score,
        "Grade":                  grade,
        "Savings_Rate_Score":     round(savings_score, 1),
        "Overspend_Score":        round(overspend_score, 1),
        "Consistency_Score":      round(consistency_score, 1),
        "Utilisation_Score":      round(util_score, 1),
        "Avg_Savings_Rate_%":     round(avg_rate, 2),
        "Overspent_Months":       overspend_months,
    }


# ════════════════════════════════════════════════════════════════
# 8. BUSINESS KPIS
# ════════════════════════════════════════════════════════════════

def compute_kpis(summary: pd.DataFrame, df_exp: pd.DataFrame) -> Dict:
    """
    Professional KPIs as used in fintech dashboards.
    """
    total_expense  = summary["Total_Expense"].sum()
    total_income   = summary["Total_Income"].sum()
    total_savings  = summary["Savings"].sum()
    avg_burn_rate  = summary["Total_Expense"].mean()
    avg_savings_rt = summary["Savings_Rate"].mean()

    # Month-over-month expense growth rate (last vs second-last month)
    if len(summary) >= 2:
        last   = summary["Total_Expense"].iloc[-1]
        prev   = summary["Total_Expense"].iloc[-2]
        growth = round((last - prev) / prev * 100, 2) if prev else 0
    else:
        growth = 0

    # Category contribution % for top category
    cat_totals   = df_exp.groupby("Category")["Amount"].sum()
    top_cat      = cat_totals.idxmax()
    top_cat_pct  = round(cat_totals.max() / total_expense * 100, 2)

    return {
        "Total_Expense":          round(total_expense, 2),
        "Total_Income":           round(total_income, 2),
        "Total_Savings":          round(total_savings, 2),
        "Avg_Monthly_Burn_Rate":  round(avg_burn_rate, 2),
        "Avg_Savings_Rate_%":     round(avg_savings_rt, 2),
        "Expense_Growth_Rate_%":  growth,
        "Top_Category":           top_cat,
        "Top_Category_Share_%":   top_cat_pct,
    }
