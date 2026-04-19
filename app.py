"""
app.py
======
Streamlit Interactive Dashboard — Expense Tracker App v2.0
==========================================================
Features:
  ✅ KPI cards (income, expense, savings, health score)
  ✅ Filters (date range, category, payment method)
  ✅ Add new expense form
  ✅ All visualizations
  ✅ ML prediction
  ✅ Insight engine
  ✅ Anomaly detection
  ✅ Budget recommendations
  ✅ Excel download button

Run: streamlit run app.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Internal modules ──────────────────────────────────────────
from data.generate_data import generate_expenses, generate_income
from src.analytics      import (load_and_prepare, monthly_summary,
                                detect_overspending, detect_anomalies,
                                category_behaviour, financial_health_score,
                                compute_kpis, savings_analysis)
from src.ml_models      import (train_expense_predictor,
                                train_spending_segmentation,
                                ml_anomaly_detection)
from src.insights       import (generate_all_insights, budget_recommendations)
from src.export         import to_excel_bytes

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="💰 Expense Tracker Pro",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #0F1117; }
  .kpi-card {
      background: linear-gradient(135deg, #1A1D27, #252836);
      border-radius: 12px; padding: 20px;
      border: 1px solid #3A3D4D;
      text-align: center;
  }
  .kpi-value  { font-size: 28px; font-weight: 700; color: #6C63FF; }
  .kpi-label  { font-size: 13px; color: #A0A0B0; margin-top: 4px; }
  .insight-card {
      background: #1A1D27; border-radius: 8px;
      padding: 12px 16px; margin: 6px 0;
      border-left: 3px solid #6C63FF;
  }
  .section-title { color: #6C63FF; font-size: 20px; font-weight: 700;
                   margin-top: 20px; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EXP_PATH = os.path.join(DATA_DIR, "expenses.csv")
INC_PATH = os.path.join(DATA_DIR, "income.csv")
os.makedirs(DATA_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════
# DATA LOADING (cached)
# ════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="🔄 Loading financial data...")
def load_data():
    if not os.path.exists(EXP_PATH):
        df_e = generate_expenses()
        df_i = generate_income()
        df_e.to_csv(EXP_PATH, index=False)
        df_i.to_csv(INC_PATH, index=False)

    df_e, df_i = load_and_prepare(EXP_PATH, INC_PATH)
    return df_e, df_i


@st.cache_data(show_spinner="🤖 Training ML models...")
def get_ml(summary_hash: str, _summary: pd.DataFrame, _df_exp: pd.DataFrame):
    _, _, _, reg_metrics = train_expense_predictor(_summary)
    cluster_pivot, _, cluster_info = train_spending_segmentation(_df_exp)
    ml_anom = ml_anomaly_detection(_df_exp)
    return reg_metrics, cluster_info, ml_anom


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/color/96/money-bag.png", width=70)
    st.title("💰 Expense Tracker Pro")
    st.caption("Fintech-grade personal finance dashboard")
    st.divider()

    st.markdown("### 🔍 Filters")
    df_exp_raw, df_inc_raw = load_data()

    min_d = df_exp_raw["Date"].min().date()
    max_d = df_exp_raw["Date"].max().date()
    date_range = st.date_input("Date Range", value=(min_d, max_d),
                               min_value=min_d, max_value=max_d)

    cats = st.multiselect("Category", options=sorted(df_exp_raw["Category"].unique()),
                          default=list(df_exp_raw["Category"].unique()))

    pay_methods = st.multiselect(
        "Payment Method",
        options=sorted(df_exp_raw["Payment_Method"].unique()),
        default=list(df_exp_raw["Payment_Method"].unique()),
    )

    st.divider()
    st.markdown("### ➕ Add New Expense")
    with st.expander("Log a transaction"):
        new_date   = st.date_input("Date")
        new_cat    = st.selectbox("Category", sorted(df_exp_raw["Category"].unique()))
        new_amount = st.number_input("Amount (₹)", min_value=1.0, step=10.0)
        new_pay    = st.selectbox("Payment Method",
                                  ["UPI", "Credit Card", "Debit Card", "Cash", "Net Banking"])
        new_desc   = st.text_input("Description", placeholder="e.g. Zomato order")
        if st.button("💾 Save Expense", use_container_width=True):
            new_row = pd.DataFrame([{
                "Date": pd.to_datetime(new_date), "Category": new_cat,
                "Amount": new_amount, "Payment_Method": new_pay,
                "Description": new_desc,
                "Is_Weekend": int(pd.to_datetime(new_date).weekday() >= 5),
                "Month": new_date.month, "Year": new_date.year,
                "Month_Year": str(pd.to_datetime(new_date).to_period("M")),
                "DayOfWeek": pd.to_datetime(new_date).day_name(),
                "Quarter": (new_date.month - 1) // 3 + 1,
            }])
            updated = pd.concat([df_exp_raw, new_row], ignore_index=True)
            updated.to_csv(EXP_PATH, index=False)
            st.success("✅ Expense saved!")
            st.cache_data.clear()
            st.rerun()

    st.divider()
    monthly_budget = st.number_input("Monthly Budget (₹)", value=60000, step=1000)


# ════════════════════════════════════════════════════════════════
# FILTER DATA
# ════════════════════════════════════════════════════════════════

if len(date_range) == 2:
    start_d, end_d = date_range
    df_exp = df_exp_raw[
        (df_exp_raw["Date"].dt.date >= start_d) &
        (df_exp_raw["Date"].dt.date <= end_d)  &
        (df_exp_raw["Category"].isin(cats))     &
        (df_exp_raw["Payment_Method"].isin(pay_methods))
    ].copy()
else:
    df_exp = df_exp_raw.copy()

summary = monthly_summary(df_exp, df_inc_raw)
overspend_df = detect_overspending(df_exp)
health       = financial_health_score(summary, overspend_df)
kpis         = compute_kpis(summary, df_exp)


# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

st.markdown("## 💼 Financial Dashboard")
st.caption(f"Period: {df_exp['Date'].min().date()} → {df_exp['Date'].max().date()} "
           f"| {len(df_exp):,} transactions")

# ════════════════════════════════════════════════════════════════
# KPI CARDS
# ════════════════════════════════════════════════════════════════

c1, c2, c3, c4, c5 = st.columns(5)

def kpi_card(col, label: str, value: str, color: str = "#6C63FF"):
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-value" style="color:{color}">{value}</div>
      <div class="kpi-label">{label}</div>
    </div>""", unsafe_allow_html=True)

kpi_card(c1, "Total Income",   f"₹{kpis['Total_Income']:,.0f}",   "#43E97B")
kpi_card(c2, "Total Expense",  f"₹{kpis['Total_Expense']:,.0f}",  "#FF6584")
kpi_card(c3, "Total Savings",  f"₹{kpis['Total_Savings']:,.0f}",  "#6C63FF")
kpi_card(c4, "Savings Rate",   f"{kpis['Avg_Savings_Rate_%']:.1f}%", "#FFD166")
kpi_card(c5, "Health Score",   f"{health['Financial_Health_Score']} {health['Grade'].split()[0]}", "#EF476F")

st.divider()

# ════════════════════════════════════════════════════════════════
# MAIN TABS
# ════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "🤖 ML Insights", "💡 Smart Insights", "🎯 Budget Planner", "📥 Export"]
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1: OVERVIEW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-title">📈 Monthly Income vs Expense</div>',
                    unsafe_allow_html=True)
        import matplotlib
        matplotlib.use("Agg")
        fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="#1A1D27")
        ax.set_facecolor("#1A1D27")
        x = np.arange(len(summary))
        w = 0.4
        ax.bar(x - w/2, summary["Total_Income"],  w, color="#43E97B", alpha=0.85, label="Income")
        ax.bar(x + w/2, summary["Total_Expense"], w, color="#FF6584", alpha=0.85, label="Expense")
        ax.set_xticks(x)
        ax.set_xticklabels(summary["Month_Year"], rotation=60, fontsize=6, color="#A0A0B0")
        ax.tick_params(colors="#A0A0B0")
        ax.legend(framealpha=0.3, labelcolor="#E0E0E0")
        ax.set_facecolor("#1A1D27")
        fig.patch.set_facecolor("#1A1D27")
        for spine in ax.spines.values(): spine.set_edgecolor("#3A3D4D")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_right:
        st.markdown('<div class="section-title">🥧 Category Distribution</div>',
                    unsafe_allow_html=True)
        cat_totals = df_exp.groupby("Category")["Amount"].sum().sort_values(ascending=False)
        PALETTE = ["#6C63FF","#FF6584","#43E97B","#FFD166","#EF476F",
                   "#06D6A0","#118AB2","#FFB347","#A8DADC","#E63946"]
        fig2, ax2 = plt.subplots(figsize=(6, 3.5), facecolor="#1A1D27")
        wedges, texts, autotexts = ax2.pie(
            cat_totals.values, labels=cat_totals.index,
            autopct="%1.0f%%", startangle=140,
            colors=PALETTE[:len(cat_totals)],
            pctdistance=0.82,
            wedgeprops={"linewidth":1.5,"edgecolor":"#1A1D27"},
            textprops={"color":"#E0E0E0","fontsize":7},
        )
        for at in autotexts: at.set_fontsize(6); at.set_color("#0F1117")
        fig2.patch.set_facecolor("#1A1D27")
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    st.markdown('<div class="section-title">💰 Savings Trend</div>', unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(12, 2.8), facecolor="#1A1D27")
    ax3.set_facecolor("#1A1D27")
    ax3.plot(summary["Month_Year"], summary["Savings"],
             color="#6C63FF", linewidth=2, marker="o", markersize=4)
    ax3.fill_between(summary["Month_Year"], summary["Savings"], 0,
                     where=(summary["Savings"] >= 0), alpha=0.25, color="#43E97B")
    ax3.fill_between(summary["Month_Year"], summary["Savings"], 0,
                     where=(summary["Savings"] < 0), alpha=0.25, color="#FF6584")
    ax3.axhline(0, color="#A0A0B0", linewidth=1, linestyle="--")
    ax3.set_xticks(range(len(summary)))
    ax3.set_xticklabels(summary["Month_Year"], rotation=60, fontsize=6, color="#A0A0B0")
    ax3.tick_params(colors="#A0A0B0")
    for spine in ax3.spines.values(): spine.set_edgecolor("#3A3D4D")
    fig3.patch.set_facecolor("#1A1D27")
    st.pyplot(fig3, use_container_width=True)
    plt.close()

    st.markdown('<div class="section-title">📋 Transaction Data</div>', unsafe_allow_html=True)
    st.dataframe(
        df_exp[["Date","Category","Amount","Payment_Method","Description"]]
              .sort_values("Date", ascending=False).head(50),
        use_container_width=True, height=280,
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: ML INSIGHTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    reg_metrics, cluster_info, ml_anom = get_ml(
        str(len(df_exp)), summary, df_exp
    )

    col_ml1, col_ml2 = st.columns(2)
    with col_ml1:
        st.markdown("### 🔮 Expense Prediction (Linear Regression)")
        next_pred = reg_metrics["Next_Month_Pred"]
        st.metric("Predicted Next Month Expense", f"₹{next_pred:,.0f}")
        st.metric("Model MAE",  f"₹{reg_metrics['MAE']:,.0f}")
        st.metric("R² Score",   str(reg_metrics["R2_Score"]))

        # Prediction chart
        months  = list(summary["Month_Year"]) + ["Next Month"]
        actuals = list(summary["Total_Expense"]) + [None]
        fig4, ax4 = plt.subplots(figsize=(8, 3.5), facecolor="#1A1D27")
        ax4.set_facecolor("#1A1D27")
        ax4.plot(summary["Month_Year"], summary["Total_Expense"],
                 color="#6C63FF", marker="o", markersize=4, linewidth=2, label="Actual")
        ax4.plot(months[-2:], [actuals[-2], next_pred],
                 color="#FFD166", marker="*", markersize=12,
                 linewidth=2, linestyle="--", label=f"Predicted: ₹{next_pred:,.0f}")
        ax4.set_xticks(range(len(summary)))
        ax4.set_xticklabels(summary["Month_Year"], rotation=60, fontsize=6, color="#A0A0B0")
        ax4.tick_params(colors="#A0A0B0")
        ax4.legend(framealpha=0.3, labelcolor="#E0E0E0", fontsize=8)
        for spine in ax4.spines.values(): spine.set_edgecolor("#3A3D4D")
        fig4.patch.set_facecolor("#1A1D27")
        st.pyplot(fig4, use_container_width=True)
        plt.close()

    with col_ml2:
        st.markdown("### 🎯 Spending Segmentation (KMeans)")
        st.metric("Silhouette Score", cluster_info["Silhouette_Score"])
        for seg, cnt in cluster_info["Segment_Counts"].items():
            st.progress(cnt / sum(cluster_info["Segment_Counts"].values()),
                        text=f"{seg} — {cnt} months")

        st.markdown("### 🚨 ML Anomaly Detection (Isolation Forest)")
        st.metric("Anomalous Transactions Flagged", len(ml_anom))
        if not ml_anom.empty:
            st.dataframe(ml_anom.head(8), use_container_width=True, height=200)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3: SMART INSIGHTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    next_pred_val = reg_metrics["Next_Month_Pred"] if "reg_metrics" in dir() else 0
    all_insights  = generate_all_insights(df_exp, summary, overspend_df,
                                          ml_anom, next_pred_val)
    for section, msgs in all_insights.items():
        st.markdown(f'<div class="section-title">📌 {section}</div>', unsafe_allow_html=True)
        for m in msgs:
            clean = m.replace("**", "**")    # keep bold markdown
            st.markdown(f'<div class="insight-card">{m}</div>', unsafe_allow_html=True)

    # Overspending details
    st.markdown('<div class="section-title">⚠️ Overspending Flags</div>',
                unsafe_allow_html=True)
    st.dataframe(
        overspend_df[overspend_df["Overspent"]][
            ["Month_Year","Category","Amount","Historical_Mean","Pct_Deviation"]
        ].head(15),
        use_container_width=True, height=250,
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4: BUDGET PLANNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.markdown("### 🎯 Smart Budget Recommendations")
    st.caption("Recommended budgets are calculated based on your spending history "
               "capped to leave 20% for savings.")
    budget_rec = budget_recommendations(df_exp, monthly_income=monthly_budget)

    st.dataframe(
        budget_rec.rename(columns={
            "Avg_Monthly_Spend": "Your Avg Spend (₹)",
            "Recommended_Budget": "Recommended (₹)",
            "Variance": "Over/Under (₹)",
        }),
        use_container_width=True, height=350,
    )

    over  = budget_rec[budget_rec["Status"] == "🔴 Over Budget"]
    under = budget_rec[budget_rec["Status"] == "✅ Under Budget"]
    c_b1, c_b2 = st.columns(2)
    c_b1.metric("Categories Over Budget",  len(over))
    c_b2.metric("Categories Under Budget", len(under))

    st.markdown("### 📊 Category Health")
    cat_beh = category_behaviour(df_exp)
    st.dataframe(cat_beh, use_container_width=True, height=300)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5: EXPORT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab5:
    st.markdown("### 📥 Download Your Financial Reports")
    cat_beh_exp = category_behaviour(df_exp)

    excel_bytes = to_excel_bytes(df_exp, summary, cat_beh_exp)
    st.download_button(
        label="📊 Download Excel Report (Multi-Sheet)",
        data=excel_bytes,
        file_name="expense_tracker_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    csv_exp = df_exp.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📄 Download Transactions CSV",
        data=csv_exp,
        file_name="transactions.csv",
        mime="text/csv",
        use_container_width=True,
    )

    csv_sum = summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📄 Download Monthly Summary CSV",
        data=csv_sum,
        file_name="monthly_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.divider()
    st.markdown("### 📈 Quick Stats")
    q1, q2, q3 = st.columns(3)
    q1.metric("Total Transactions", f"{len(df_exp):,}")
    q2.metric("Date Range", f"{df_exp['Date'].min().date()} → {df_exp['Date'].max().date()}")
    q3.metric("Unique Categories", df_exp["Category"].nunique())

    st.markdown("### 🔥 Top 10 Largest Transactions")
    st.dataframe(
        df_exp.nlargest(10, "Amount")[
            ["Date","Category","Amount","Description","Payment_Method"]
        ],
        use_container_width=True,
    )

# ── Footer ─────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>💰 Expense Tracker Pro v2.0 — Built for Placements & Internships "
    "| Powered by Python, Scikit-learn & Streamlit</small></center>",
    unsafe_allow_html=True,
)
