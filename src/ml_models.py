"""
ml_models.py
============
Machine Learning Integration for the Expense Tracker App.

Models:
  1. Expense Prediction     — Linear Regression  (next-month forecast)
  2. Spending Segmentation  — KMeans Clustering   (user behaviour groups)
  3. Anomaly Detection      — Isolation Forest    (ML-based outlier detection)

Why each model:
  - Regression: Forecasting is a core fintech feature (budgeting apps, CRED).
  - Clustering: User segmentation drives personalised recommendations (Mint, Fi).
  - Isolation Forest: Used in fraud/anomaly detection in banking pipelines.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LinearRegression
from sklearn.cluster         import KMeans
from sklearn.ensemble        import IsolationForest
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import silhouette_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# ════════════════════════════════════════════════════════════════
# 1. EXPENSE PREDICTION — Linear Regression
# ════════════════════════════════════════════════════════════════

def build_monthly_features(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering on monthly summary for regression.

    Features: month index, rolling 3-month avg, rolling std,
              savings rate, income, previous month expense
    """
    df = summary.copy().reset_index(drop=True)
    df["Month_Idx"]     = np.arange(len(df))
    df["Roll3_Avg"]     = df["Total_Expense"].rolling(3, min_periods=1).mean()
    df["Roll3_Std"]     = df["Total_Expense"].rolling(3, min_periods=1).std().fillna(0)
    df["Prev_Expense"]  = df["Total_Expense"].shift(1).fillna(df["Total_Expense"].mean())
    df["Prev2_Expense"] = df["Total_Expense"].shift(2).fillna(df["Total_Expense"].mean())
    return df


def train_expense_predictor(summary: pd.DataFrame):
    """
    Train a Linear Regression model to predict next-month total expense.

    Returns: (model, scaler, feature_cols, metrics_dict, next_month_prediction)
    """
    df = build_monthly_features(summary)

    feature_cols = ["Month_Idx", "Roll3_Avg", "Roll3_Std",
                    "Prev_Expense", "Prev2_Expense", "Total_Income", "Savings_Rate"]
    target_col   = "Total_Expense"
    
    import numpy as np
    X = df[feature_cols]
    y = df[target_col]
    # ✅ CLEAN DATA (VERY IMPORTANT - BEFORE SPLIT)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    # Train / test split (last 4 months as test)
    split = max(int(len(X) * 0.80), len(X) - 4)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # ✅ THEN SCALE
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae    = round(mean_absolute_error(y_test, y_pred), 2)
    r2     = round(r2_score(y_test, y_pred), 4)
    last_row = build_monthly_features(summary).iloc[[-1]].copy()
    # Prepare next input (KEEP AS DATAFRAME)
    X_next = last_row[feature_cols]

    # Clean BEFORE scaling
    X_next = X_next.replace([np.inf, -np.inf], np.nan)
    X_next = X_next.fillna(0)

    # Scale
    X_next = scaler.transform(X_next)

    # Final safety (NOW it's numpy → safe to use this)
    X_next = np.nan_to_num(X_next)
    next_pred = model.predict(X_next)[0]

    metrics = {
        "MAE":              mae,
        "R2_Score":         r2,
        "Test_Actuals":     list(y_test.round(2)),
        "Test_Predictions": list(y_pred.round(2)),
        "Next_Month_Pred":  next_pred,
    }
    return model, scaler, feature_cols, metrics


# ════════════════════════════════════════════════════════════════
# 2. SPENDING SEGMENTATION — KMeans Clustering
# ════════════════════════════════════════════════════════════════

def prepare_cluster_features(df_exp: pd.DataFrame) -> pd.DataFrame:
    """
    Create a month-level feature matrix for clustering.
    Each row = one month; features = spending per category.
    """
    pivot = (df_exp.groupby(["Month_Year", "Category"])["Amount"]
             .sum().unstack(fill_value=0))
    pivot = pivot.reset_index()
    return pivot


def train_spending_segmentation(df_exp: pd.DataFrame, n_clusters: int = 3):
    """
    KMeans clustering to identify spending personality segments.

    Segments (typical):
      Cluster 0 → "Balanced Saver"     — moderate spending, high savings
      Cluster 1 → "High Spender"       — above-average across categories
      Cluster 2 → "Minimal Spender"    — low discretionary, mostly essentials

    Returns: (pivot_with_cluster, cluster_labels, silhouette, interpretation)
    """
    pivot    = prepare_cluster_features(df_exp)
    cat_cols = [c for c in pivot.columns if c != "Month_Year"]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(pivot[cat_cols])

    kmeans   = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels   = kmeans.fit_predict(X_scaled)

    pivot["Cluster"] = labels
    sil = round(silhouette_score(X_scaled, labels), 4) if n_clusters > 1 else 1.0

    # Auto-label clusters by mean total spending
    pivot["Total_Month"] = pivot[cat_cols].sum(axis=1)
    cluster_means        = pivot.groupby("Cluster")["Total_Month"].mean().sort_values()

    label_map = {}
    for rank, cluster_id in enumerate(cluster_means.index):
        if rank == 0:
            label_map[cluster_id] = "💰 Minimal Spender"
        elif rank == len(cluster_means) - 1:
            label_map[cluster_id] = "💸 High Spender"
        else:
            label_map[cluster_id] = "⚖️ Balanced Spender"

    pivot["Segment"] = pivot["Cluster"].map(label_map)

    interpretation = {
        "Silhouette_Score": sil,
        "Cluster_Means":    cluster_means.round(2).to_dict(),
        "Label_Map":        label_map,
        "Segment_Counts":   pivot["Segment"].value_counts().to_dict(),
    }
    return pivot, kmeans, interpretation


# ════════════════════════════════════════════════════════════════
# 3. ML ANOMALY DETECTION — Isolation Forest
# ════════════════════════════════════════════════════════════════

def ml_anomaly_detection(df_exp: pd.DataFrame,
                          contamination: float = 0.05) -> pd.DataFrame:
    """
    Isolation Forest anomaly detection on transaction-level data.

    Features: Amount, Is_Weekend, Month, DayOfWeek (encoded)

    Used in: fraud detection, unusual-spend alerts (CRED, banks).
    contamination = expected proportion of anomalies (5% default).

    Returns: df_exp with 'ML_Anomaly' column (True = anomalous).
    """
    df = df_exp.copy()

    # Encode day-of-week
    df["DayNum"] = df["Date"].dt.dayofweek

    features = ["Amount", "Is_Weekend", "Month", "DayNum"]
    X = df[features].fillna(0).values

    iso = IsolationForest(n_estimators=100, contamination=contamination,
                          random_state=42)
    preds = iso.fit_predict(X)                # -1 = anomaly, 1 = normal
    df["ML_Anomaly"]       = preds == -1
    df["Anomaly_Score"]    = iso.decision_function(X).round(4)

    anomalies = df[df["ML_Anomaly"]].sort_values("Amount", ascending=False)
    return anomalies[["Date", "Category", "Amount", "Description",
                       "Payment_Method", "Anomaly_Score"]].reset_index(drop=True)


# ════════════════════════════════════════════════════════════════
# 4. ELBOW CURVE — choose optimal K
# ════════════════════════════════════════════════════════════════

def elbow_inertia(df_exp: pd.DataFrame, k_range: range = range(2, 8)) -> dict:
    """Return inertia values for elbow plot (choose optimal K)."""
    pivot    = prepare_cluster_features(df_exp)
    cat_cols = [c for c in pivot.columns if c != "Month_Year"]
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(pivot[cat_cols])

    inertias = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias[k] = round(km.inertia_, 2)
    return inertias
