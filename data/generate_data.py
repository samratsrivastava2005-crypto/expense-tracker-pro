"""
generate_data.py
================
Generates realistic synthetic financial data for the Expense Tracker App.
Simulates: salary credits, fixed rent, weekend spikes, festival spikes,
random unexpected expenses — mimicking real-world personal finance.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random

# ── Reproducibility ──────────────────────────────────────────────
np.random.seed(42)
random.seed(42)

# ── Configuration ────────────────────────────────────────────────
START_DATE   = "2023-01-01"
END_DATE     = "2024-12-31"
MONTHLY_SALARY = 75000          # ₹75,000 / month
MONTHLY_RENT   = 15000          # fixed rent
USER_NAME      = "Aryan Sharma"

CATEGORIES = {
    "Food & Dining":    {"base": 8000,  "std": 1500},
    "Transport":        {"base": 3000,  "std": 800},
    "Rent":             {"base": 15000, "std": 0},      # fixed
    "Entertainment":    {"base": 2500,  "std": 1000},
    "Shopping":         {"base": 4000,  "std": 2000},
    "Healthcare":       {"base": 1500,  "std": 1200},
    "Utilities":        {"base": 2000,  "std": 400},
    "Education":        {"base": 3000,  "std": 500},
    "Travel":           {"base": 0,     "std": 5000},   # occasional
    "Miscellaneous":    {"base": 1000,  "std": 800},
}

PAYMENT_METHODS = ["UPI", "Credit Card", "Debit Card", "Cash", "Net Banking"]

# Festival months → spending spikes (India-specific)
FESTIVAL_MONTHS = {10: 1.4, 11: 1.3, 12: 1.2, 1: 1.15}

# ── Helper: transaction-level data ───────────────────────────────
def _rand_date_in_month(year: int, month: int) -> str:
    """Pick a random calendar day within a given month."""
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1)
    else:
        end = datetime(year, month + 1, 1)
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta - 1))).strftime("%Y-%m-%d")


def generate_expenses(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """
    Generate a transaction-level expense DataFrame.

    Each row = one expense transaction with:
        Date | Category | Amount | Payment_Method | Description | Is_Weekend | Month | Year
    """
    rows = []
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")

    year_months = []
    cur = start_dt
    while cur <= end_dt:
        year_months.append((cur.year, cur.month))
        # Advance month
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)

    for year, month in year_months:
        festival_mult = FESTIVAL_MONTHS.get(month, 1.0)

        for cat, params in CATEGORIES.items():
            base, std = params["base"], params["std"]

            # Travel: only 3–4 months a year have travel expenses
            if cat == "Travel" and random.random() > 0.3:
                continue

            # Rent: exactly one transaction per month
            if cat == "Rent":
                rows.append({
                    "Date": f"{year}-{month:02d}-01",
                    "Category": cat,
                    "Amount": MONTHLY_RENT,
                    "Payment_Method": "Net Banking",
                    "Description": "Monthly Rent Payment",
                    "Is_Weekend": 0,
                    "Month": month,
                    "Year": year,
                })
                continue

            # Number of transactions this month for this category
            n_txn = max(1, int(np.random.normal(4, 1)))
            monthly_budget = max(0, np.random.normal(base, std)) * festival_mult

            if monthly_budget < 10:
                continue

            per_txn = monthly_budget / n_txn

            for _ in range(n_txn):
                date_str = _rand_date_in_month(year, month)
                date_obj  = datetime.strptime(date_str, "%Y-%m-%d")
                is_weekend = int(date_obj.weekday() >= 5)

                # Weekend premium (food & entertainment spike)
                weekend_mult = 1.3 if (is_weekend and cat in ("Food & Dining", "Entertainment")) else 1.0

                amount = max(10, np.random.normal(per_txn, per_txn * 0.15) * weekend_mult)

                rows.append({
                    "Date": date_str,
                    "Category": cat,
                    "Amount": round(amount, 2),
                    "Payment_Method": random.choice(PAYMENT_METHODS),
                    "Description": _description(cat),
                    "Is_Weekend": is_weekend,
                    "Month": month,
                    "Year": year,
                })

        # ── Inject random unexpected expenses (2–5 per month) ──
        for _ in range(random.randint(2, 5)):
            date_str = _rand_date_in_month(year, month)
            rows.append({
                "Date": date_str,
                "Category": "Miscellaneous",
                "Amount": round(random.uniform(200, 3000), 2),
                "Payment_Method": random.choice(PAYMENT_METHODS),
                "Description": random.choice([
                    "Emergency purchase", "Surprise gift", "Car repair",
                    "Medical emergency", "House repair", "Lost wallet recharge"
                ]),
                "Is_Weekend": 0,
                "Month": month,
                "Year": year,
            })

    df_expenses = pd.DataFrame(rows)
    df_expenses["Date"] = pd.to_datetime(df_expenses["Date"])
    df_expenses = df_expenses.sort_values("Date").reset_index(drop=True)
    return df_expenses


def _description(category: str) -> str:
    desc_map = {
        "Food & Dining":  ["Zomato order", "Swiggy", "Restaurant dinner", "Grocery store",
                           "Coffee shop", "Street food", "Bakery"],
        "Transport":      ["Ola/Uber ride", "Metro card recharge", "Petrol", "Auto fare",
                           "Bus pass", "Parking fee"],
        "Entertainment":  ["Netflix", "Movie tickets", "OTT subscription", "Concert",
                           "Gaming purchase", "Sports event"],
        "Shopping":       ["Amazon order", "Flipkart", "Clothing store", "Electronics",
                           "Home décor", "Footwear"],
        "Healthcare":     ["Pharmacy", "Doctor consultation", "Lab test", "Gym membership",
                           "Health supplement"],
        "Utilities":      ["Electricity bill", "Water bill", "Internet bill", "Mobile recharge",
                           "DTH recharge"],
        "Education":      ["Online course", "Books", "Stationery", "Coaching fee",
                           "Certification exam"],
        "Travel":         ["Flight ticket", "Hotel booking", "Cab to airport",
                           "Train ticket", "Travel insurance"],
        "Miscellaneous":  ["Miscellaneous", "Other expense"],
    }
    return random.choice(desc_map.get(category, ["General expense"]))


def generate_income(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """
    Generate monthly income DataFrame.
        Date | Source | Amount | Month | Year
    """
    rows = []
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")

    cur = start_dt
    while cur <= end_dt:
        # Salary on 1st of each month
        rows.append({
            "Date": cur.strftime("%Y-%m-%d"),
            "Source": "Salary",
            "Amount": MONTHLY_SALARY,
            "Month": cur.month,
            "Year": cur.year,
        })
        # Occasional freelance (30% probability)
        if random.random() < 0.30:
            rows.append({
                "Date": cur.strftime("%Y-%m-%d"),
                "Source": "Freelance",
                "Amount": round(random.uniform(5000, 20000), 2),
                "Month": cur.month,
                "Year": cur.year,
            })
        # Advance month
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)

    df_income = pd.DataFrame(rows)
    df_income["Date"] = pd.to_datetime(df_income["Date"])
    return df_income


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__))

    expenses = generate_expenses()
    income   = generate_income()

    expenses.to_csv(os.path.join(out_dir, "expenses.csv"), index=False)
    income.to_csv(os.path.join(out_dir, "income.csv"),     index=False)

    print(f"✅ expenses.csv  → {len(expenses)} rows")
    print(f"✅ income.csv    → {len(income)} rows")
    print("\nExpense sample:")
    print(expenses.head(8).to_string(index=False))
