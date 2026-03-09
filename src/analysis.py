# src/analysis.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

TABLE_DIR = Path("reports/tables")
TABLE_DIR.mkdir(parents=True, exist_ok=True)

def build_feature_table(patients, stores, prescriptions_diab, adherence, interventions, financial):
    df = adherence.merge(patients, on="patient_id", how="left")

    store_agg = prescriptions_diab.groupby("store_id").agg(
        diab_rx_count=("rx_id", "count"),
        diab_margin=("walgreens_margin", "sum"),
    )
    store_agg = store_agg.merge(stores, on="store_id", how="left")

    last_rx = prescriptions_diab.sort_values("fill_date").groupby("patient_id").tail(1)[
        ["patient_id", "store_id", "drug_class", "days_supply", "copay_amount", "plan_paid_amount"]
    ]

    df = df.merge(last_rx, on="patient_id", how="left")
    df = df.merge(store_agg, on="store_id", how="left")
    df = df.merge(interventions, on="patient_id", how="left")
    df = df.merge(financial, on="patient_id", how="left")

    df["age_bucket"] = pd.cut(df["age"], bins=[0, 40, 60, 80, 120], labels=["<40", "40-60", "60-80", "80+"])
    df["chronic_bucket"] = pd.cut(df["chronic_conditions"], bins=[-1, 0, 1, 3, 10], labels=["0", "1", "2-3", "4+"])

    df.to_csv(TABLE_DIR / "feature_table_diabetes.csv", index=False)
    return df

def encode_features(df):
    target = "is_non_adherent_next_90d"

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target]

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    df = df.copy()
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=True)

    X = df.drop(columns=[target])
    y = df[target].astype(int)

    return X, y

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    logit = LogisticRegression(max_iter=1000, n_jobs=-1)
    logit.fit(X_train, y_train)
    y_logit = logit.predict_proba(X_test)[:, 1]

    rf = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_rf = rf.predict_proba(X_test)[:, 1]

    results = {
        "logit": (y_test, y_logit),
        "rf": (y_test, y_rf),
    }

    for name, (yt, yp) in results.items():
        print(f"\n=== {name.upper()} ===")
        print("AUC:", roc_auc_score(yt, yp))
        print("PR-AUC:", average_precision_score(yt, yp))
        print(classification_report(yt, (yp > 0.5).astype(int)))

    return results
