"""
Smoke test for the training pipeline.
Verifies that the full pipeline (feature engineering → scaling → SMOTE → model fit → predict)
runs end-to-end without errors and produces outputs with expected shapes and value ranges.

Code co-developed with Antigravity (Google DeepMind), powered by Claude (Anthropic).
All outputs reviewed, tested, and validated by the author.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


@pytest.fixture
def sample_data():
    """Create a minimal dataset mimicking the credit default schema."""
    np.random.seed(42)
    n = 200
    data = {
        'LIMIT_BAL': np.random.randint(10000, 500000, n),
        'SEX': np.random.choice([1, 2], n),
        'EDUCATION': np.random.choice([1, 2, 3, 4], n),
        'MARRIAGE': np.random.choice([1, 2, 3], n),
        'AGE': np.random.randint(21, 65, n),
    }
    # Add PAY and BILL_AMT and PAY_AMT columns
    for i in range(6):
        data[f'PAY_{i}'] = np.random.choice([-1, 0, 1, 2, 3], n)
        data[f'BILL_AMT{i+1}'] = np.random.randint(0, 100000, n)
        data[f'PAY_AMT{i+1}'] = np.random.randint(0, 50000, n)
    data['DEFAULT'] = np.random.choice([0, 1], n, p=[0.78, 0.22])
    return pd.DataFrame(data)


def test_pipeline_runs_end_to_end(sample_data):
    """Full pipeline smoke test: engineer → scale → SMOTE → fit → predict."""
    df = sample_data.copy()

    # Feature engineering (simplified)
    df['UTILISATION_RATIO'] = df[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
                                   'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].mean(axis=1) / (df['LIMIT_BAL'] + 1)
    df['AVG_PAY_AMT'] = df[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
                              'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].mean(axis=1)
    df['MAX_DELAY'] = df[['PAY_0', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5']].max(axis=1)

    avg_bill = df[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
                    'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].mean(axis=1)
    df['PAY_BILL_RATIO'] = df['AVG_PAY_AMT'] / (avg_bill + 1)

    # Split
    X = df.drop('DEFAULT', axis=1)
    y = df['DEFAULT']
    split_idx = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_sc, y_train)

    # Fit
    model = XGBClassifier(
        n_estimators=10, max_depth=3, random_state=42,
        use_label_encoder=False, eval_metric='logloss'
    )
    model.fit(X_train_sm, y_train_sm)

    # Predict
    y_pred = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    # Assertions
    assert y_pred.shape == (len(X_test),), "Prediction shape mismatch"
    assert y_proba.shape == (len(X_test),), "Probability shape mismatch"
    assert set(y_pred).issubset({0, 1}), "Predictions must be binary"
    assert all(0 <= p <= 1 for p in y_proba), "Probabilities must be in [0, 1]"
    assert len(X_train_sm) > len(X_train), "SMOTE should increase training set size"


def test_model_predicts_both_classes(sample_data):
    """Model should predict at least some instances of each class (not degenerate)."""
    df = sample_data.copy()
    X = df.drop('DEFAULT', axis=1)
    y = df['DEFAULT']

    model = XGBClassifier(
        n_estimators=50, max_depth=3, random_state=42,
        use_label_encoder=False, eval_metric='logloss',
        scale_pos_weight=3.5
    )
    model.fit(X, y)
    y_pred = model.predict(X)

    assert len(set(y_pred)) == 2, "Model should predict both classes on training data"


def test_feature_importance_non_negative(sample_data):
    """Feature importances should be non-negative and sum to ~1."""
    df = sample_data.copy()
    X = df.drop('DEFAULT', axis=1)
    y = df['DEFAULT']

    model = XGBClassifier(
        n_estimators=10, max_depth=3, random_state=42,
        use_label_encoder=False, eval_metric='logloss'
    )
    model.fit(X, y)
    importances = model.feature_importances_

    assert all(i >= 0 for i in importances), "Importances must be non-negative"
    assert abs(sum(importances) - 1.0) < 0.01, "Importances should sum to ~1"
