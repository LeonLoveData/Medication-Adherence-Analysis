# src/data_process.py
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/raw/synthetic")
TABLE_DIR = Path("reports/tables")
TABLE_DIR.mkdir(parents=True, exist_ok=True)

def load_synthetic_data():
    patients = pd.read_csv(DATA_DIR / "patients.csv")
    prescriptions = pd.read_csv(DATA_DIR / "prescriptions.csv", parse_dates=["fill_date"])
    stores = pd.read_csv(DATA_DIR / "stores.csv")
    interventions = pd.read_csv(DATA_DIR / "program_interventions.csv", parse_dates=["intervention_date"])
    financial = pd.read_csv(DATA_DIR / "financial_summary.csv")
    return patients, prescriptions, stores, interventions, financial

def filter_diabetes_prescriptions(prescriptions):
    diabetes_classes = ["Diabetes", "GLP1", "SGLT2", "Insulin"]
    mask = prescriptions["drug_class"].str.contains("diab", case=False, na=False) | prescriptions[
        "drug_class"
    ].isin(diabetes_classes)
    return prescriptions[mask].copy()

def compute_adherence_metrics(prescriptions):
    df = prescriptions.copy()
    df = df.sort_values(["patient_id", "drug_class", "fill_date"])

    df["prev_fill_date"] = df.groupby(["patient_id", "drug_class"])["fill_date"].shift(1)
    df["prev_days_supply"] = df.groupby(["patient_id", "drug_class"])["days_supply"].shift(1)

    df["gap_days"] = (
        (df["fill_date"] - df["prev_fill_date"]).dt.days - df["prev_days_supply"]
    )
    df["gap_days"] = df["gap_days"].fillna(0).clip(lower=0)

    obs_start = df["fill_date"].min()
    obs_end = df["fill_date"].max()
    obs_days = (obs_end - obs_start).days + 1

    agg = df.groupby("patient_id").agg(
        total_days_supply=("days_supply", "sum"),
        rx_count=("rx_id", "count"),
        avg_gap=("gap_days", "mean"),
        max_gap=("gap_days", "max"),
        last_fill_date=("fill_date", "max"),
    )

    agg["mpr"] = agg["total_days_supply"] / obs_days
    agg["mpr"] = agg["mpr"].clip(0, 1.5)
    agg["pdc"] = agg["mpr"].clip(0, 1.0)
    agg["is_non_adherent_next_90d"] = (agg["pdc"] < 0.8).astype(int)

    agg = agg.reset_index()
    agg.to_csv(TABLE_DIR / "adherence_metrics_diabetes.csv", index=False)
    return agg
