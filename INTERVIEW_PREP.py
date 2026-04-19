"""
INTERVIEW PREPARATION GUIDE
============================
Expense Tracker App — Data Science / Data Analyst / Business Analyst
Roles: DA | BA | FA | ML Engineer | Python Developer
"""

QUESTIONS_AND_ANSWERS = """
╔══════════════════════════════════════════════════════════════════╗
║          EXPENSE TRACKER APP — INTERVIEW PREP GUIDE              ║
╚══════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION A: HR / BEHAVIOURAL QUESTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q1. Tell me about this project.

A: "I built an end-to-end personal finance intelligence platform called
Expense Tracker Pro. It simulates 2 years of realistic financial data —
including salary credits, rent, festival spending spikes, and random
expenses — and analyzes them using Python, Pandas, and Scikit-learn.
The system generates automated insights, predicts next-month spending
using regression, segments spending patterns using KMeans clustering,
and detects anomalies using Isolation Forest. The entire pipeline is
wrapped in a Streamlit dashboard deployed on Streamlit Cloud — live
and publicly accessible."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION B: TECHNICAL QUESTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q2. Why did you use Linear Regression for expense prediction?

A: "Linear Regression is interpretable, fast, and effective for
time-series data with temporal features like month index and
rolling averages. My features included month index, rolling 3-month
average, previous month expense, savings rate, and income level.
The model achieved an MAE of ~₹3,000 on test data (last 4 months),
which is acceptable for personal finance forecasting. More complex
models like XGBoost could improve accuracy, which I listed as a
future enhancement."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q3. How does your anomaly detection work?

A: "I implemented two approaches. The first is statistical — IQR-based
detection that flags individual transactions where the amount exceeds
Q3 + 1.5×IQR within that category. This is fast and explainable.
The second is ML-based — Isolation Forest, an unsupervised algorithm
that isolates anomalies by randomly partitioning the feature space.
It uses Amount, Is_Weekend, Month, and DayOfWeek as features with
5% contamination. This is similar to how banks detect unusual
transactions in real-time fraud detection pipelines."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q4. Explain your Financial Health Score formula.

A: "The score is a weighted composite on a 0–100 scale:
  - Savings Rate Score (40 pts): proportional to average savings rate,
    full marks for 30%+ savings rate.
  - Overspend Penalty (30 pts): deducted based on proportion of months
    where categories exceeded their historical mean by 20%.
  - Consistency Score (20 pts): inversely proportional to the
    coefficient of variation in monthly savings.
  - Income Utilisation Score (10 pts): full marks if expenses are
    below 80% of income.
The formula mimics how CIBIL and fintech apps compute financial wellness."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q5. How did you generate realistic synthetic data?

A: "I designed the data generator with real-world financial patterns:
  - Fixed monthly salary (₹75,000) credited on the 1st
  - Rent is a fixed ₹15,000 transaction every month
  - Festival months (Oct/Nov/Dec) have a 1.2–1.4x spending multiplier
  - Weekends have 30% higher Food & Entertainment transactions
  - 2–5 random unexpected expenses per month (emergency purchases)
  - Occasional freelance income (30% probability each month)
  - Category-wise normal distribution for transaction amounts with
    realistic means and standard deviations
This created ~900 transactions over 24 months, closely mimicking
a real personal finance dataset."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q6. What are the KPIs you computed and why?

A: "I computed 8 business KPIs:
  1. Total Income / Expense / Savings — baseline P&L
  2. Monthly Burn Rate — average spend per month; used by CFOs
  3. Savings Rate — % of income saved; key metric for financial health
  4. Expense Growth Rate — month-over-month % change; signals trends
  5. Category Contribution % — which bucket consumes most budget
  6. Cumulative Savings — total wealth accumulated over the period
These mirror KPIs tracked by finance teams at startups and used in
fintech dashboards like Razorpay Dashboard and CRED Insights."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q7. How does KMeans clustering help in this project?

A: "KMeans groups months based on their category-level spending pattern.
Each month becomes a feature vector (spending per category). After
StandardScaler normalization, KMeans with k=3 segments months into:
  - Minimal Spender: low overall spending, likely savings months
  - Balanced Spender: typical months near average
  - High Spender: above-average across multiple categories
This is how Mint and Fi recommend 'spending personas'. Silhouette
Score validates cluster quality. In a product, this enables
personalised advice: 'This month you are a High Spender — here are
tips to cut Food expenses.'"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q8. How would you scale this to production?

A: "For production, I would:
  1. Replace synthetic data with bank API integration (Plaid, Setu)
  2. Store data in PostgreSQL with a FastAPI backend
  3. Schedule ML retraining monthly using Apache Airflow
  4. Add real-time alerts via Twilio SMS or Firebase push
  5. Containerise with Docker and deploy on AWS/GCP
  6. Add user authentication (Auth0 / Firebase Auth)
  7. Implement A/B testing for insight recommendations"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q9. What was the hardest challenge?

A: "The hardest part was designing the insight engine. Raw numbers
are useless — the value is in context. I built rule-based logic
that compares current vs historical, flags percentage deviations,
and generates human-readable text. Balancing sensitivity (catching
real overspending) vs noise (false alerts) required tuning the
20% threshold through experimentation — similar to how data teams
at fintech companies calibrate alert systems."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q10. How is this better than just using Excel?

A: "Excel can store and sum data. This project does what Excel cannot:
  - Automatically detects anomalous transactions
  - Predicts future spending using ML regression
  - Segments behaviour across months using clustering
  - Generates natural language insights automatically
  - Provides a real-time interactive dashboard accessible anywhere
  - Exports professional multi-sheet Excel reports programmatically
  - Computes a quantitative Financial Health Score
This is the difference between a tool and a platform."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESUME BULLET POINTS (Top 1% Level)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. "Built an end-to-end Fintech Data Science platform with Python and
   Scikit-learn; implemented ML pipeline (Linear Regression, KMeans,
   Isolation Forest) achieving <₹3,000 MAE on expense forecasting and
   0.72+ Silhouette Score on behavioural segmentation."

2. "Engineered a composite Financial Health Scoring system (0–100)
   and automated Insight Engine generating 20+ data-driven financial
   recommendations — deployed on Streamlit Cloud with live public access."

3. "Designed a realistic financial data simulation engine modelling
   salary credits, festival spending spikes, weekend multipliers, and
   900+ synthetic transactions across 24 months; built multi-format
   reporting (CSV, Excel, text) and interactive dashboard with real-time
   filters and ML visualizations."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO EXPLAIN IN INTERVIEW (Script)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"I built this project to solve a real problem: most people don't
understand where their money goes. I created a full data science
pipeline — from synthetic data generation to an interactive dashboard.

The system generates realistic financial transactions, applies advanced
analytics to detect overspending and anomalies, uses machine learning
to predict future expenses and segment spending behaviour, and converts
all this into human-readable insights.

I deployed it as a live Streamlit app — you can filter by date, add
new transactions, see ML predictions in real-time, download Excel
reports, and check your Financial Health Score.

This isn't a college assignment — it's a functioning fintech product
that demonstrates every skill in a data science workflow."
"""

if __name__ == "__main__":
    print(QUESTIONS_AND_ANSWERS)
