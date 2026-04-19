"""
export.py
=========
Export & Reporting System
--------------------------
Generates CSV, Excel, and a formatted text summary report.
Used by both the CLI (main.py) and Streamlit dashboard (app.py).
"""

import os
import io
import pandas as pd
import datetime

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════
# 1. CSV EXPORT
# ════════════════════════════════════════════════════════════════

def export_csv(df: pd.DataFrame, filename: str) -> str:
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  💾 CSV exported: {filename}")
    return path


# ════════════════════════════════════════════════════════════════
# 2. EXCEL EXPORT — multi-sheet
# ════════════════════════════════════════════════════════════════

def export_excel(df_exp: pd.DataFrame,
                 summary: pd.DataFrame,
                 cat_behaviour: pd.DataFrame,
                 overspend: pd.DataFrame,
                 filename: str = "expense_report.xlsx") -> str:
    path = os.path.join(OUTPUT_DIR, filename)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_exp.to_excel(writer,      sheet_name="All Transactions", index=False)
        summary.to_excel(writer,     sheet_name="Monthly Summary",  index=False)
        cat_behaviour.to_excel(writer, sheet_name="Category Analysis", index=False)
        overspend.to_excel(writer,   sheet_name="Overspending Flags", index=False)

    print(f"  📊 Excel exported: {filename}")
    return path


# ════════════════════════════════════════════════════════════════
# 3. IN-MEMORY EXCEL (for Streamlit download button)
# ════════════════════════════════════════════════════════════════

def to_excel_bytes(df_exp: pd.DataFrame,
                   summary: pd.DataFrame,
                   cat_behaviour: pd.DataFrame) -> bytes:
    """Returns an Excel file as bytes — used for Streamlit's download_button."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_exp.to_excel(writer,        sheet_name="Transactions",   index=False)
        summary.to_excel(writer,       sheet_name="Monthly Summary", index=False)
        cat_behaviour.to_excel(writer, sheet_name="Categories",      index=False)
    return buf.getvalue()


# ════════════════════════════════════════════════════════════════
# 4. TEXT SUMMARY REPORT
# ════════════════════════════════════════════════════════════════

def generate_text_report(kpis: dict,
                          health: dict,
                          insights: dict,
                          filename: str = "summary_report.txt") -> str:
    now   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []

    lines += [
        "=" * 65,
        "   💼 EXPENSE TRACKER APP — FINANCIAL SUMMARY REPORT",
        f"   Generated: {now}",
        "=" * 65,
        "",
        "📌 KEY PERFORMANCE INDICATORS",
        "-" * 40,
    ]
    for k, v in kpis.items():
        lines.append(f"  {k:35s}: {v}")

    lines += [
        "",
        "🏆 FINANCIAL HEALTH SCORE",
        "-" * 40,
    ]
    for k, v in health.items():
        lines.append(f"  {k:35s}: {v}")

    lines += ["", "💡 AUTOMATED INSIGHTS", "-" * 40]
    for section, msgs in insights.items():
        lines.append(f"\n  [{section}]")
        for msg in msgs:
            clean = msg.replace("**", "").replace("*", "")
            lines.append(f"    • {clean}")

    lines += ["", "=" * 65, "  End of Report — Expense Tracker App v2.0", "=" * 65]

    report = "\n".join(lines)
    path   = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"  📝 Report saved: {filename}")
    return path
