# Credit Card Default Prediction

**Module:** MSIN0097 Predictive Analytics (UCL) — Individual Coursework 2025-26

## Overview

An end-to-end predictive analytics pipeline to predict whether a credit card holder will default on their next payment, using the [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) dataset.

## Repository Structure

```
├── notebooks/
│   └── credit_default_analysis.ipynb   # Main analysis notebook
├── src/
│   ├── data_prep.py                    # Preprocessing functions
│   ├── models.py                       # Training & evaluation helpers
│   └── visualisation.py                # Plotting utilities
├── data/
│   └── default of credit card clients.xls
│   └── DATA_DICTIONARY.md
├── outputs/
│   ├── figures/                        # Saved plots
│   └── models/                         # Serialised models
├── tests/
│   ├── test_data_prep.py               # 9 unit tests for preprocessing
│   └── test_pipeline.py                # 3 end-to-end pipeline smoke tests
├── requirements.txt
└── README.md
```

## Setup & Run

```bash
# 1. Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook notebooks/credit_default_analysis.ipynb

# 4. Or execute non-interactively
jupyter nbconvert --to notebook --execute notebooks/credit_default_analysis.ipynb
```

## Dataset

- **Source:** UCI Machine Learning Repository
- **Rows:** 30,000 clients
- **Features:** 24 (demographics, credit limit, repayment history, bill/payment amounts)
- **Target:** `default payment next month` (binary: 0 = no default, 1 = default)

## Models Evaluated

| Model | Rationale |
|---|---|
| Logistic Regression | Linear baseline; interpretable |
| Random Forest | Non-linear ensemble; feature importance |
| XGBoost | State-of-the-art gradient boosting |
| MLP Neural Network | Deep learning baseline (128-64 hidden units, ReLU, Adam) |

## Key Metrics

- **Primary:** AUC-ROC
- **Secondary:** F1-Score, Precision, Recall, Accuracy

## AI Acknowledgement

Code was co-developed with **Antigravity** (Google DeepMind), an agentic AI coding assistant powered by Claude Opus 4.6 (Anthropic). All agent outputs were reviewed, modified where necessary, and validated by the author. A full 34-entry decision register documenting every agent interaction, including accepted, modified, and rejected suggestions, is provided in the appendix of the report.
