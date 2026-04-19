"""
main.py
=======
Expense Tracker App — Main Pipeline Runner
==========================================
Orchestrates the full data science pipeline:
  1. Generate synthetic data
  2. Load & prepare
  3. Analytics (overspending, anomalies, health score)
  4. Machine learning (regression + clustering)
  5. Insight engine
  6. Visualizations
  7. Export (CSV, Excel, report)

Run: python main.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

# ── Internal modules ──────────────────────────────────────────
from data.generate_data    import generate_expenses, generate_income
from src.analytics         import (load_and_prepare, monthly_summary,
                                   detect_overspending, detect_anomalies,
                                   category_behaviour, savings_analysis,
                                   financial_health_score, compute_kpis)
from src.ml_models         import (train_expense_predictor,
                                   train_spending_segmentation,
                                   ml_anomaly_detection)
from src.insights          import (generate_all_insights, budget_recommendations)
from src.visualizations    import generate_all_charts
from src.export            import (export_csv, export_excel,
                                   generate_text_report)

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXP_PATH = os.path.join(DATA_DIR, "expenses.csv")
INC_PATH = os.path.join(DATA_DIR, "income.csv")

BANNER = """
╔══════════════════════════════════════════════════════════╗
║        💰 EXPENSE TRACKER APP  — Data Science v2.0       ║
║            Built for Placements & Internships             ║
╚══════════════════════════════════════════════════════════╝
"""


def main():
    print(BANNER)

    # ──────────────────────────────────────────────
    # PHASE 1: DATA GENERATION
    # ──────────────────────────────────────────────
    print("=" * 55)
    print("📂 PHASE 1: Generating Synthetic Financial Data")
    print("=" * 55)

    df_exp  = generate_expenses()
    df_inc  = generate_income()
    df_exp.to_csv(EXP_PATH, index=False)
    df_inc.to_csv(INC_PATH, index=False)

    print(f"  ✅ Expenses: {len(df_exp):,} transactions")
    print(f"  ✅ Income  : {len(df_inc):,} entries")

    # ──────────────────────────────────────────────
    # PHASE 2: LOAD & PREPARE
    # ──────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("🔧 PHASE 2: Loading & Preparing Data")
    print("=" * 55)
    df_exp, df_inc = load_and_prepare(EXP_PATH, INC_PATH)
    summary        = monthly_summary(df_exp, df_inc)
    print(f"  ✅ Date range  : {df_exp['Date'].min().date()} → {df_exp['Date'].max().date()}")
    print(f"  ✅ Total spent : ₹{df_exp['Amount'].sum():,.0f}")
    print(f"  ✅ Categories  : {df_exp['Category'].nunique()}")
    print(f"  ✅ Months      : {len(summary)}")

    # ──────────────────────────────────────────────
    # PHASE 3: ANALYTICS
    # ──────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("📊 PHASE 3: Advanced Analytics")
    print("=" * 55)

    overspend_df = detect_overspending(df_exp, threshold_pct=20)
    n_overspend  = overspend_df[overspend_df["Overspent"]].shape[0]
    print(f"  ⚠️  Overspending instances detected: {n_overspend}")

    anomaly_df = detect_anomalies(df_exp)
    print(f"  🔍 Statistical anomalies (IQR)    : {len(anomaly_df)}")

    cat_beh = category_behaviour(df_exp)
    print(f"  📈 Category trends computed.")

    sav_kpis = savings_analysis(summary)
    print(f"  💰 Avg monthly savings: ₹{sav_kpis['Average_Monthly_Savings']:,.0f}")

    health = financial_health_score(summary, overspend_df)
    print(f"  🏆 Financial Health Score: {health['Financial_Health_Score']} {health['Grade']}")

    kpis = compute_kpis(summary, df_exp)
    print(f"  📌 KPIs computed: Burn rate ₹{kpis['Avg_Monthly_Burn_Rate']:,.0f}/month")

    # ──────────────────────────────────────────────
    # PHASE 4: MACHINE LEARNING
    # ──────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("🤖 PHASE 4: Machine Learning")
    print("=" * 55)

    print("  Training expense predictor (Linear Regression)...")
    model, scaler, feat_cols, reg_metrics = train_expense_predictor(summary)
    next_pred = reg_metrics["Next_Month_Pred"]
    print(f"  ✅ MAE: ₹{reg_metrics['MAE']:,.0f} | R²: {reg_metrics['R2_Score']}")
    print(f"  🔮 Next month predicted expense: ₹{next_pred:,.0f}")

    print("\n  Training spending segmentation (KMeans)...")
    cluster_pivot, kmeans_model, cluster_info = train_spending_segmentation(df_exp)
    print(f"  ✅ Silhouette Score: {cluster_info['Silhouette_Score']}")
    for seg, cnt in cluster_info["Segment_Counts"].items():
        print(f"     {seg}: {cnt} months")

    print("\n  Running ML anomaly detection (Isolation Forest)...")
    ml_anomalies = ml_anomaly_detection(df_exp)
    print(f"  ✅ ML anomalies flagged: {len(ml_anomalies)}")

    # ──────────────────────────────────────────────
    # PHASE 5: INSIGHTS
    # ──────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("💡 PHASE 5: Intelligent Insight Engine")
    print("=" * 55)

    insights = generate_all_insights(df_exp, summary, overspend_df,
                                      ml_anomalies, next_pred)

    for section, msgs in insights.items():
        print(f"\n  [{section}]")
        for m in msgs[:2]:          # show first 2 per section
            clean = m.replace("**", "").replace("*", "")
            print(f"    • {clean[:110]}")

    # Budget recommendations
    budget_rec = budget_recommendations(df_exp, monthly_income=75000)
    budget_path = os.path.join(OUTPUT_DIR, "budget_recommendations.csv")
    budget_rec.to_csv(budget_path, index=False)
    print(f"\n  ✅ Budget recommendations saved.")

    # ──────────────────────────────────────────────
    # PHASE 6: VISUALIZATIONS
    # ──────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("🎨 PHASE 6: Generating Visualizations")
    print("=" * 55)
    generate_all_charts(df_exp, summary, next_pred, health["Financial_Health_Score"])

    # ──────────────────────────────────────────────
    # PHASE 7: EXPORT
    # ──────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("💾 PHASE 7: Exporting Reports")
    print("=" * 55)

    export_csv(df_exp,        "transactions.csv")
    export_csv(summary,       "monthly_summary.csv")
    export_csv(overspend_df,  "overspending_report.csv")
    export_csv(ml_anomalies,  "ml_anomalies.csv")
    export_excel(df_exp, summary, cat_beh, overspend_df)

    # Build KPI dict for report
    full_kpis = {**kpis, **sav_kpis}
    full_kpis = {k: (f"₹{v:,.2f}" if isinstance(v, float) and v > 100 else v)
                 for k, v in full_kpis.items()}
    generate_text_report(full_kpis, health, insights)

    # ──────────────────────────────────────────────
    # DONE
    # ──────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("🎉 PIPELINE COMPLETE!")
    print("=" * 55)
    print(f"  📁 All outputs saved to: {OUTPUT_DIR}")
    print(f"  🌐 Run Streamlit app  : streamlit run app.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
