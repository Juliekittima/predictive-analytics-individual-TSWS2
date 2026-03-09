"""
Model training and evaluation utilities for credit card default prediction.

Code co-developed with Antigravity (Google DeepMind), powered by Claude (Anthropic).
All outputs reviewed, tested, and validated by the author.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
)
import warnings
warnings.filterwarnings("ignore")


def get_candidate_models(random_state=42):
    """Return a dictionary of candidate models with sensible defaults."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=3.5,  # approx. ratio of neg/pos classes
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(128, 64),  # 2 hidden layers
            activation="relu",
            solver="adam",
            alpha=0.001,           # L2 regularisation
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,   # stop when validation score plateaus
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=random_state,
        ),
    }
    return models


def cross_validate_models(models, X_train, y_train, cv=5, scoring="roc_auc", random_state=42):
    """
    Run stratified cross-validation for each candidate model.
    Returns a DataFrame with mean and std scores.
    """
    results = []
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1)
        results.append({
            "Model": name,
            "Mean AUC-ROC": scores.mean(),
            "Std AUC-ROC": scores.std(),
            "Scores": scores,
        })
        print(f"  {name}: AUC = {scores.mean():.4f} ± {scores.std():.4f}")

    return pd.DataFrame(results)


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a fitted model on a test set.
    Returns a dict of metrics and the predicted probabilities.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
        "AUC-ROC": roc_auc_score(y_test, y_prob),
        "F1-Score": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Average Precision": average_precision_score(y_test, y_prob),
    }

    return metrics, y_pred, y_prob


def get_xgb_param_grid():
    """Return hyperparameter search space for XGBoost."""
    return {
        "max_depth": [3, 4, 5, 6, 7, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
        "n_estimators": [100, 200, 300, 500],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "scale_pos_weight": [2.5, 3.0, 3.5, 4.0],
    }


def get_rf_param_grid():
    """Return hyperparameter search space for Random Forest."""
    return {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [5, 8, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced", "balanced_subsample"],
    }


def tune_model(model, param_grid, X_train, y_train, n_iter=50, cv=5, random_state=42):
    """
    Run RandomizedSearchCV for hyperparameter tuning.
    Returns the best estimator and search results.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=skf,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )

    search.fit(X_train, y_train)

    print(f"  Best AUC-ROC: {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")

    return search.best_estimator_, search


def print_classification_report(y_test, y_pred, model_name="Model"):
    """Print a formatted classification report."""
    print(f"\n{'='*60}")
    print(f"Classification Report — {model_name}")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))


def cost_sensitive_evaluation(y_true, y_pred, cost_fp=1, cost_fn=5, model_name="Model"):
    """
    Evaluate a model using cost-sensitive metrics.

    In credit default prediction:
      - False Negative (missed default): bank loses the outstanding balance.
        Typically 5-10x more costly than a false positive.
      - False Positive (false alarm on good client): intervention cost
        (e.g. reduced credit limit), plus customer friction.

    Parameters
    ----------
    y_true : array-like — true labels
    y_pred : array-like — predicted labels
    cost_fp : float — cost assigned to each false positive (default=1)
    cost_fn : float — cost assigned to each false negative (default=5)
    model_name : str — label for the model

    Returns
    -------
    dict with FP count, FN count, FP cost, FN cost, total cost,
    and cost per client (normalised by sample size).
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fp_cost = fp * cost_fp
    fn_cost = fn * cost_fn
    total_cost = fp_cost + fn_cost
    n = len(y_true)

    return {
        "Model": model_name,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "FP_Cost": fp_cost,
        "FN_Cost": fn_cost,
        "Total_Cost": total_cost,
        "Cost_Per_Client": total_cost / n,
    }
