# Telecom Credit Risk Hybrid Model

A credit risk model for phone instalment lending in Zimbabwe, combining **Cox Proportional Hazards** survival analysis with **gradient boosting (XGBoost)** to predict not only *if* a customer will default, but *when*.

Final-year project, B.Tech (Honours) Financial Engineering, Harare Institute of Technology, 2026.

## Problem
Traditional logistic regression models only predict whether a client will default. They ignore timing, rely on limited behavioural data, struggle in unstable macroeconomic conditions, and exclude clients with no formal credit history.

## Approach
- Trained on 5,234 phone instalment contracts (Jan 2024 – Dec 2025) from a leading Zimbabwean telecom operator (data not included, for confidentiality)
- Cox PH model (lifelines) for time-to-default and survival probabilities
- XGBoost classifier for default probability
- Hybrid score combining both models (weight α = 0.3, chosen by cross-validation)
- Payment-to-income (PTI) ratio and behavioural features such as payment history and past defaults

## Key results
| Model | Performance |
|---|---|
| Logistic regression | AUC 0.752 |
| XGBoost | AUC 0.843 |
| Cox PH | C-index 0.814 |
| **Hybrid** | **AUC 0.871 (+15.8%)** |

- Improvement validated with DeLong's test (p = 0.008) and McNemar's test (p < 0.001)
- PTI ratio was the strongest default driver (hazard ratio 2.84)
- Stress test: expected default rate rose from 14.98% to 19.32% under a 30% income drop

## Features
- Risk scoring with four tiers (Low, Medium, High, Critical)
- Asset protection terms by tier: maximum term, minimum down payment, monitoring frequency
- Stress testing across five macroeconomic scenarios
- IFRS 9 stage mapping
- SHAP explainability
- Web dashboard for assessment and portfolio monitoring

## Tech stack
Python, XGBoost, lifelines, scikit-learn, SHAP, Streamlit

## Author
Tadiwanashe Makwangudze, Financial Engineering graduate, Harare, Zimbabwe
