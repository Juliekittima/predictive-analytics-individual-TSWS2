"""
Unit tests for the data preparation pipeline.
Validates that preprocessing steps maintain data integrity and avoid leakage.

Code co-developed with Antigravity (Google DeepMind), powered by Claude (Anthropic).
All outputs reviewed, tested, and validated by the author.
"""
import sys
import os
import pytest
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_prep import clean_data, engineer_features, split_data, scale_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_df():
    """Load and clean the dataset once for all tests."""
    data_path = os.path.join(os.path.dirname(__file__), "..",
                              "default of credit card clients.xls")
    df = pd.read_excel(data_path, header=1)
    df.columns = [c.strip().upper() for c in df.columns]
    if "DEFAULT PAYMENT NEXT MONTH" in df.columns:
        df.rename(columns={"DEFAULT PAYMENT NEXT MONTH": "DEFAULT"}, inplace=True)
    return clean_data(df)


@pytest.fixture
def featured_df(raw_df):
    """Apply feature engineering."""
    return engineer_features(raw_df)


@pytest.fixture
def split_sets(featured_df):
    """Split the data."""
    return split_data(featured_df)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCleanData:
    """Tests for the clean_data function."""

    def test_no_missing_values(self, raw_df):
        """Cleaned data should have no NaN values."""
        assert raw_df.isnull().sum().sum() == 0, "Cleaned data contains missing values"

    def test_low_duplicate_rate(self, raw_df):
        """Duplicate rows should be minimal (< 1% of data)."""
        dup_rate = raw_df.duplicated().sum() / len(raw_df)
        assert dup_rate < 0.01, f"Duplicate rate {dup_rate:.3%} exceeds 1%"

    def test_target_is_binary(self, raw_df):
        """DEFAULT column should only contain 0 and 1."""
        assert set(raw_df["DEFAULT"].unique()) == {0, 1}, \
            f"Target has unexpected values: {raw_df['DEFAULT'].unique()}"

    def test_education_recoded(self, raw_df):
        """Undocumented EDUCATION codes (0, 5, 6) should be mapped to 4."""
        assert not raw_df["EDUCATION"].isin([0, 5, 6]).any(), \
            "Undocumented EDUCATION categories still present"


class TestFeatureEngineering:
    """Tests for the engineer_features function."""

    def test_new_features_exist(self, featured_df):
        """All engineered features should be present."""
        expected = ["UTILISATION_RATIO", "AVG_PAY_AMT", "MAX_DELAY", "PAY_BILL_RATIO"]
        for feat in expected:
            assert feat in featured_df.columns, f"Missing engineered feature: {feat}"

    def test_no_inf_values(self, featured_df):
        """Engineered features should not contain infinity."""
        numeric = featured_df.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any(), "Infinite values found in features"


class TestSplitData:
    """Tests for the split_data function."""

    def test_no_index_overlap(self, split_sets):
        """Train, validation and test sets should have no overlapping indices."""
        X_train, X_val, X_test, y_train, y_val, y_test = split_sets
        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)

        assert train_idx.isdisjoint(val_idx), "Train and validation sets overlap"
        assert train_idx.isdisjoint(test_idx), "Train and test sets overlap"
        assert val_idx.isdisjoint(test_idx), "Validation and test sets overlap"

    def test_class_proportions_preserved(self, split_sets):
        """Stratified split should preserve approximate class proportions."""
        X_train, X_val, X_test, y_train, y_val, y_test = split_sets
        train_ratio = y_train.mean()
        val_ratio = y_val.mean()
        test_ratio = y_test.mean()

        # All should be close to 0.221 (±0.02)
        for name, ratio in [("train", train_ratio), ("val", val_ratio), ("test", test_ratio)]:
            assert abs(ratio - 0.221) < 0.02, \
                f"{name} set class ratio {ratio:.3f} deviates from expected 0.221"

    def test_split_sizes(self, split_sets):
        """60/20/20 split should produce expected set sizes."""
        X_train, X_val, X_test, y_train, y_val, y_test = split_sets
        total = len(X_train) + len(X_val) + len(X_test)

        assert abs(len(X_train) / total - 0.6) < 0.02, "Train set is not ~60%"
        assert abs(len(X_val) / total - 0.2) < 0.02, "Validation set is not ~20%"
        assert abs(len(X_test) / total - 0.2) < 0.02, "Test set is not ~20%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
