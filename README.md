# ChannelRx-Insight-Predicting-Pharmacy-Adherence-Program-ROI
ABC Pharma – Diabetes Medication Analytics Platform
A Production‑Style End‑to‑End Data Science System for Medication Adherence, Risk Prediction, and Financial Impact Modeling

🧭 Overview
This project simulates a real-world analytics platform used by ABC Pharma, a large retail pharmacy and healthcare services organization. The system integrates public healthcare datasets with synthetic enterprise patient‑level data to analyze diabetes medication utilization, predict non‑adherence risk, segment patients and prescribers, and model financial impact under multiple intervention scenarios.

The platform demonstrates advanced capabilities expected from senior data scientists and analytics leaders:

Predictive modeling

Statistical analysis

Segmentation

Scenario simulation

Business impact quantification

Dashboarding

AI‑assisted insights

All components are built in a modular, production‑ready architecture.

🏗️ System Architecture
```
Code
                ┌──────────────────────────┐
                │  Public Data Downloader   │
                └──────────────┬───────────┘
                               ▼
                ┌──────────────────────────┐
                │  Synthetic Data Generator │
                └──────────────┬───────────┘
                               ▼
                ┌──────────────────────────┐
                │     Feature Pipeline      │
                └──────────────┬───────────┘
                               ▼
                ┌──────────────────────────┐
                │  ML Models (Risk, Seg.)  │
                └──────────────┬───────────┘
                               ▼
                ┌──────────────────────────┐
                │ Scenario & ROI Simulator │
                └──────────────┬───────────┘
                               ▼
                ┌──────────────────────────┐
                │   Dashboard + Assistant  │
                └──────────────────────────┘
```
📂 Repository Structure
```
Code
abc-pharma-diabetes-analytics/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── public/          # CMS, CDC, Census, OpenFDA
│   │   └── synthetic/       # Generated patient-level data
│   ├── interim/
│   └── processed/
├── src/
│   ├── data/
│   │   ├── download_public_data.py
│   │   └── generate_synthetic_data.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train_adherence_model.py
│   │   ├── train_segmentation.py
│   │   └── scenario_simulator.py
│   ├── visualization/
│   │   └── visualize.py
│   ├── dashboard/
│   │   └── app.py
│   └── assistant/
│       └── cli_assistant.py
├── models/
├── reports/
│   ├── figures/
│   └── tables/
└── notebooks/
```
🔍 Focus Area: Diabetes Medication Analytics
This project focuses specifically on diabetes medications, combining:

CMS Medicare Part D Drug Spending

Part D Prescriber Diabetes 2021

Synthetic patient‑level pharmacy data

The goal is to replicate how a large pharmacy like ABC Pharma would analyze, predict, and optimize diabetes medication adherence and financial outcomes.

📊 1. Descriptive Analytics — What is happening?
1.1 Diabetes Drug Utilization Trends (CMS Part D)
Total spending by diabetes drug class (GLP‑1, SGLT2, insulin)

Year‑over‑year growth in claims and beneficiaries

Average spend per beneficiary

Identification of high‑cost therapies

1.2 Prescriber Behavior (Part D Prescriber Diabetes 2021)
Top prescribers by diabetes drug volume

Specialty mix (endocrinology vs primary care)

Geographic prescribing patterns

Prescriber concentration metrics

1.3 Patient‑Level Descriptives (Synthetic Data)
Age, gender, payer mix

Chronic condition burden

Loyalty tier distribution

Store‑level diabetes prescription volume

🧪 2. Diagnostic Analytics — Why is it happening?
2.1 Adherence Drivers
Relationship between days supply and PDC/MPR

Impact of payer type on adherence

Influence of chronic condition count

Store‑level operational factors

2.2 Drug Class Differences
GLP‑1 vs SGLT2 vs insulin adherence patterns

Cost‑related non‑adherence (copay vs refill behavior)

2.3 Prescriber Influence
Prescribers associated with higher adherence outcomes

Prescriber‑level refill gap patterns

🤖 3. Predictive Analytics — What will happen?
3.1 Non‑Adherence Risk Model
Predicts which diabetes patients are likely to become non‑adherent in the next 90 days.

Features include:

Historical MPR/PDC

Refill gaps

Drug class

Copay + plan paid

Chronic condition burden

Loyalty tier

Store region

Prior interventions

Models:

Logistic Regression

RandomForest / XGBoost

Calibration + lift charts

3.2 Prescriber‑Level Forecasting
Predict future diabetes prescribing volume

Forecast spend per beneficiary

Identify prescribers likely to increase GLP‑1 usage

3.3 Store‑Level Opportunity Forecast
Predict stores with highest diabetes script growth

Estimate revenue at risk due to non‑adherence

🧩 4. Segmentation — Who behaves similarly?
4.1 Patient Segmentation
Clusters patients using:

Adherence behavior

Drug class mix

Chronic condition burden

Value (margin, revenue)

Loyalty tier

Intervention responsiveness

4.2 Prescriber Segmentation
Clusters prescribers by:

Volume

Drug class mix

GLP‑1 adoption

Patient adherence outcomes

Region

💰 5. Prescriptive & Financial Analytics — What should we do?
5.1 Intervention ROI Modeling
Estimate incremental fills from outreach

Estimate incremental margin

Compute ROI by patient segment

5.2 Scenario Simulation
Simulates multiple program strategies:

High‑touch vs low‑touch outreach

Different response rates

Different cost structures

Targeting high‑risk vs high‑value patients

5.3 Revenue‑at‑Risk Analysis
Identify stores with highest diabetes revenue at risk

Quantify financial upside of improving adherence

📈 6. Dashboard
The Streamlit dashboard includes:

Executive KPIs

Risk score explorer

Segmentation explorer

Scenario simulator

Store & region opportunity maps
