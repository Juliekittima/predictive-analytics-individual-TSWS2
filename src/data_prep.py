"""
Data preparation utilities for credit card default prediction.
Handles cleaning, feature engineering, and train/val/test splitting.

Code co-developed with Antigravity (Google DeepMind), powered by Claude (Anthropic).
All outputs reviewed, tested, and validated by the author.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from ucimlrepo import fetch_ucirepo


def load_data(dataset_id: int = 350) -> pd.DataFrame:
    """
    Load the credit card default dataset from UCI ML Repository.
    Returns a single DataFrame with features + target combined,
    with columns renamed to their original meaningful names.
    """
    dataset = fetch_ucirepo(id=dataset_id)

    # Combine features and target into one DataFrame
    X = dataset.data.features
    y = dataset.data.targets

    df = pd.concat([X, y], axis=1)

    # UCI repo returns generic names (X1–X23, Y).
    # Rename to the original, descriptive column names.
    column_mapping = {
        "X1": "LIMIT_BAL",
        "X2": "SEX",
        "X3": "EDUCATION",
        "X4": "MARRIAGE",
        "X5": "AGE",
        "X6": "PAY_0",
        "X7": "PAY_2",
        "X8": "PAY_3",
        "X9": "PAY_4",
        "X10": "PAY_5",
        "X11": "PAY_6",
        "X12": "BILL_AMT1",
        "X13": "BILL_AMT2",
        "X14": "BILL_AMT3",
        "X15": "BILL_AMT4",
        "X16": "BILL_AMT5",
        "X17": "BILL_AMT6",
        "X18": "PAY_AMT1",
        "X19": "PAY_AMT2",
        "X20": "PAY_AMT3",
        "X21": "PAY_AMT4",
        "X22": "PAY_AMT5",
        "X23": "PAY_AMT6",
        "Y": "default payment next month",
    }
    df = df.rename(columns=column_mapping)

    # Print metadata summary
    print(f"Dataset: {dataset.metadata.name}")
    print(f"Rows: {len(df):,}  |  Features: {X.shape[1]}  |  Target: default payment next month")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset:
    - Drop the ID column (if present)
    - Fix undocumented EDUCATION values (0, 5, 6 → 4 = 'Other')
    - Fix undocumented MARRIAGE value (0 → 3 = 'Other')
    - Rename target for convenience
    """
    df = df.copy()

    # Drop ID column (present in the XLS version)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Fix undocumented EDUCATION categories (0, 5, 6 → 4: Other)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

    # Fix undocumented MARRIAGE category (0 → 3: Other)
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    # Rename target for convenience
    df = df.rename(columns={"default payment next month": "DEFAULT"})

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features:
    - UTILISATION_RATIO: average bill amount / credit limit
    - AVG_PAY_AMT: average payment amount over 6 months
    - MAX_DELAY: worst repayment delay across all 6 months
    - PAY_AMT_TO_BILL_RATIO: average ratio of payment to bill
    """
    df = df.copy()

    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    pay_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    repayment_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

    # Average utilisation ratio
    avg_bill = df[bill_cols].mean(axis=1)
    df["UTILISATION_RATIO"] = avg_bill / df["LIMIT_BAL"].replace(0, np.nan)
    df["UTILISATION_RATIO"] = df["UTILISATION_RATIO"].fillna(0)

    # Average payment amount
    df["AVG_PAY_AMT"] = df[pay_cols].mean(axis=1)

    # Maximum repayment delay
    df["MAX_DELAY"] = df[repayment_cols].max(axis=1)

    # Payment-to-bill ratio (how much of the bill is typically paid)
    avg_pay = df[pay_cols].mean(axis=1)
    df["PAY_BILL_RATIO"] = avg_pay / avg_bill.replace(0, np.nan)
    df["PAY_BILL_RATIO"] = df["PAY_BILL_RATIO"].fillna(0)
    # Cap extreme ratios
    df["PAY_BILL_RATIO"] = df["PAY_BILL_RATIO"].clip(upper=10)

    return df


def split_data(
    df: pd.DataFrame,
    target_col: str = "DEFAULT",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split into train (60%), validation (20%), test (20%) with stratification.
    Returns X_train, X_val, X_test, y_train, y_val, y_test.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split: 80% train+val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Second split: from the 80%, take 25% as validation (0.25 * 0.8 = 0.2 of total)
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction, stratify=y_temp, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test, continuous_cols):
    """
    Apply StandardScaler fitted on training data only.
    Returns scaled copies and the fitted scaler.
    """
    scaler = StandardScaler()

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_val[continuous_cols] = scaler.transform(X_val[continuous_cols])
    X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

    return X_train, X_val, X_test, scaler


def apply_smote(X_train, y_train, random_state=42):
    """Apply SMOTE oversampling to the training set only."""
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def get_continuous_cols(df):
    """Return list of continuous feature columns."""
    return [
        "LIMIT_BAL", "AGE",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
        "UTILISATION_RATIO", "AVG_PAY_AMT", "PAY_BILL_RATIO",
    ]


def validate_splits(X_train, X_val, X_test, y_train, y_val, y_test):
    """Run data-integrity assertions on the splits."""
    total = len(X_train) + len(X_val) + len(X_test)

    # No missing values
    assert X_train.isnull().sum().sum() == 0, "NaN in X_train"
    assert X_val.isnull().sum().sum() == 0, "NaN in X_val"
    assert X_test.isnull().sum().sum() == 0, "NaN in X_test"

    # Approximate split ratios (with 5% tolerance)
    assert abs(len(X_train) / total - 0.6) < 0.05, f"Train ratio off: {len(X_train)/total:.2f}"
    assert abs(len(X_val) / total - 0.2) < 0.05, f"Val ratio off: {len(X_val)/total:.2f}"
    assert abs(len(X_test) / total - 0.2) < 0.05, f"Test ratio off: {len(X_test)/total:.2f}"

    # No index overlap
    assert len(set(X_train.index) & set(X_test.index)) == 0, "Train/test overlap!"
    assert len(set(X_train.index) & set(X_val.index)) == 0, "Train/val overlap!"
    assert len(set(X_val.index) & set(X_test.index)) == 0, "Val/test overlap!"

    print(f"✅ All validation checks passed!")
    print(f"   Train: {len(X_train):,} ({len(X_train)/total:.1%})")
    print(f"   Val:   {len(X_val):,} ({len(X_val)/total:.1%})")
    print(f"   Test:  {len(X_test):,} ({len(X_test)/total:.1%})")
