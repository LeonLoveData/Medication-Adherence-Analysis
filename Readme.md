# ABC Pharma – Pharmacy Patient Adherence Risk Prediction System

# Business Context

Retail pharmacies like Walgreens serve millions of patients managing chronic diseases such as diabetes. One of the biggest operational challenges is medication non-adherence, which leads to:
- poor health outcomes
- increased healthcare costs
- lost pharmacy revenue

This project simulates how a retail pharmacy analytics team can use healthcare data and machine learning to:
- identify patients at risk of medication non-adherence
- forecast refill gaps
- quantify revenue at risk
- prioritize pharmacy interventions

A production-style end-to-end analytics system for analyzing diabetes medication adherence, predicting non-adherence risk, and generating actionable insights for pharmacy operations and patient engagement.

---

# 🧭 Overview

This project simulates how a large pharmacy organization like **ABC Pharma** analyzes diabetes medication behavior using:

- Public datasets (CMS Part D Drug Spending, Prescriber Diabetes)
- Synthetic enterprise-grade patient-level data
- A full adherence analytics pipeline

The system includes:

- Data processing
- Adherence metric computation (**MPR, PDC, refill gaps**)
- Feature engineering
- Predictive modeling (**Logistic Regression, RandomForest**)
- Visualization
- Modular architecture

---
# 🎯Business Impact 

This analytics system enables pharmacy organizations to:
- identify high-risk patients before medication gaps occur
- prioritize pharmacist outreach programs
- improve medication adherence rates
- reduce revenue loss from missed refills
- optimize patient engagement strategies

Potential measurable outcomes:
- 10–15% improvement in adherence
- improved patient outcomes
- millions in recovered pharmacy revenue
- 
# 🏗️ Project Structure
```
src/
  data_process.py        # Load data, filter diabetes drugs, compute MPR/PDC/gaps
  analysis.py            # Feature engineering + ML models
  visualization.py       # Plots for adherence + model curves
  main.py                # Orchestration pipeline

reports/
  tables/                # adherence_metrics_diabetes.csv, feature_table_diabetes.csv
  figures/               # PDC/MPR/gap distributions, ROC/PR curves

data/
  raw/synthetic/         # Synthetic enterprise data
```
```
Data Sources
    ↓
CMS Medicare Part D Data
Synthetic Patient Dataset
Pharmacy Store Data

    ↓
Data Engineering Layer
Python ETL Pipelines
Feature Engineering

    ↓
Analytics Layer
Adherence Metrics
PDC / MPR
Refill Gap Detection

    ↓
Machine Learning
Logistic Regression
Random Forest

    ↓
Business Insights
Patient Risk Scores
Revenue-at-Risk Analysis
Dashboard Reporting
```

---

# 📊 Outputs

## 1. Tables (reports/tables/)

| File | Description |
|-----|-------------|
| adherence_metrics_diabetes.csv | MPR, PDC, refill gaps, non-adherence label |
| feature_table_diabetes.csv | Full ML feature table |

---

## 2. Figures (reports/figures/)

- `pdc_distribution.png`
- `mpr_distribution.png`
- `max_gap_distribution.png`
- `roc_logit.png`
- `roc_rf.png`
- `pr_logit.png`
- `pr_rf.png`

---

## 3. Console Model Results

Model evaluation outputs include:

- **AUC**
- **PR-AUC**
- **Classification Report**
- **Positive Rate**
- **Feature Table Shape**

---

# ⚙️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Synthetic healthcare data generation

---

# 📌 Future Improvements

- Add **XGBoost / LightGBM models**
- Deploy a **Streamlit dashboard**
- Integrate **real CMS Medicare Part D claims data**
- Implement **patient segmentation analytics**

---

# 📜 License

This project is for **educational and learning purposes**.
