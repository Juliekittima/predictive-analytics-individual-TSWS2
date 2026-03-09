"""
Visualisation utilities for credit card default prediction.
All plots use a consistent style and colour palette.

Code co-developed with Antigravity (Google DeepMind), powered by Claude (Anthropic).
All outputs reviewed, tested, and validated by the author.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc
from sklearn.calibration import calibration_curve

# ---------- Style configuration ----------
PALETTE = ["#3B82F6", "#EF4444"]  # Blue (no default), Red (default)
PALETTE_MULTI = ["#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6"]
BG_COLOR = "#FAFAFA"
GRID_COLOR = "#E5E7EB"

sns.set_theme(
    style="whitegrid",
    palette=PALETTE,
    font_scale=1.1,
    rc={
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": GRID_COLOR,
        "grid.color": GRID_COLOR,
        "figure.dpi": 120,
    },
)


# Project root detection — works from both scripts and notebooks
def _find_project_root():
    """Find project root by locating the src/ package directory."""
    try:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(src_dir)
    except Exception:
        return os.getcwd()

_PROJECT_ROOT = _find_project_root()

# Allow external override of the output directory
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "outputs", "figures")


def set_output_dir(path):
    """Override the default figure output directory."""
    global OUTPUT_DIR
    OUTPUT_DIR = path
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_fig(fig, name, output_dir=None):
    """Save figure to the outputs directory."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{name}.png")
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    print(f"Saved: {filepath}")


# ---------- EDA Plots ----------

def plot_target_distribution(y, save=True):
    """Bar chart of target class distribution with percentages."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    counts = y.value_counts().sort_index()
    bars = ax.bar(
        ["No Default (0)", "Default (1)"],
        counts.values,
        color=PALETTE,
        edgecolor="white",
        linewidth=1.5,
        width=0.5,
    )

    for bar, count in zip(bars, counts.values):
        pct = count / len(y) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
            f"{count:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontweight="bold", fontsize=11,
        )

    ax.set_title("Target Variable Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_ylim(0, max(counts.values) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save:
        save_fig(fig, "01_target_distribution")
    return fig


def plot_correlation_heatmap(df, save=True):
    """Annotated correlation heatmap with diverging palette."""
    fig, ax = plt.subplots(figsize=(16, 13))

    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, linewidths=0.5,
        annot_kws={"size": 7}, ax=ax,
        cbar_kws={"label": "Pearson Correlation", "shrink": 0.8},
    )

    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    if save:
        save_fig(fig, "02_correlation_heatmap")
    return fig


def plot_numerical_distributions(df, save=True, filename="18_numerical_distributions"):
    """Histograms of key numerical features with skewness annotation."""
    from scipy.stats import skew

    num_cols = ["LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT3",
                "PAY_AMT1", "PAY_AMT3"]
    col_labels = ["Credit Limit", "Age", "Bill Amount (Sep)",
                  "Bill Amount (Jul)", "Payment (Sep)", "Payment (Jul)"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    for ax, col, label in zip(axes, num_cols, col_labels):
        vals = df[col].dropna()
        sk = skew(vals)

        ax.hist(vals, bins=50, color=PALETTE[0], edgecolor="white",
                linewidth=0.5, alpha=0.8)
        ax.axvline(vals.median(), color="#EF4444", linewidth=1.5,
                   linestyle="--", label=f"Median: {vals.median():,.0f}")
        ax.set_title(f"{label} ({col})", fontsize=11, fontweight="bold")
        ax.set_ylabel("Count", fontsize=9)
        ax.text(0.97, 0.93, f"Skew: {sk:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D1D5DB", alpha=0.9))
        ax.legend(fontsize=8, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Numerical Feature Distributions",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save:
        save_fig(fig, filename)
    return fig


def plot_categorical_distributions(df, save=True, filename="19_categorical_distributions"):
    """Bar charts showing value counts for categorical features."""
    cat_config = {
        "SEX": {1: "Male", 2: "Female"},
        "EDUCATION": {1: "Graduate", 2: "University", 3: "High School", 4: "Other"},
        "MARRIAGE": {1: "Married", 2: "Single", 3: "Other"},
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, (col, label_map) in zip(axes, cat_config.items()):
        counts = df[col].value_counts().sort_index()
        labels = [label_map.get(k, str(k)) for k in counts.index]
        bars = ax.bar(labels, counts.values, color=PALETTE[0],
                      edgecolor="white", linewidth=0.8, alpha=0.8)

        # Add count + percentage labels above each bar
        total = counts.sum()
        for bar, val in zip(bars, counts.values):
            pct = val / total * 100
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

        ax.set_title(col, fontsize=12, fontweight="bold")
        ax.set_ylabel("Count", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Categorical Feature Distributions",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save:
        save_fig(fig, filename)
    return fig


def plot_repayment_by_default(df, target_col="DEFAULT", save=True):
    """Repayment status distribution (PAY_0–PAY_6) split by default status."""
    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    month_labels = ["Sep", "Aug", "Jul", "Jun", "May", "Apr"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for ax, col, month in zip(axes.ravel(), pay_cols, month_labels):
        for label, color in zip([0, 1], PALETTE):
            subset = df[df[target_col] == label][col]
            ax.hist(
                subset, bins=range(-3, 10), alpha=0.55, color=color,
                label=f"{'No Default' if label == 0 else 'Default'}",
                edgecolor="white", density=True, linewidth=0.8,
            )
        ax.set_title(f"{col} ({month} 2005)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Repayment Status", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=9, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Repayment Status by Default Status", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save:
        save_fig(fig, "03_repayment_by_default")
    return fig


def plot_limit_bal_by_default(df, target_col="DEFAULT", save=True):
    """Overlaid KDE of LIMIT_BAL by default status."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for label, color, name in zip([0, 1], PALETTE, ["No Default", "Default"]):
        subset = df[df[target_col] == label]["LIMIT_BAL"]
        ax.hist(subset, bins=50, alpha=0.5, color=color, label=name, density=True, edgecolor="white")
        subset.plot.kde(ax=ax, color=color, linewidth=2)

    ax.set_title("Credit Limit Distribution by Default Status", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Credit Limit (NT$)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save:
        save_fig(fig, "04_limit_bal_by_default")
    return fig


def plot_default_rate_by_category(df, target_col="DEFAULT", save=True):
    """Default rate across EDUCATION and MARRIAGE categories."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    edu_labels = {1: "Grad School", 2: "University", 3: "High School", 4: "Other"}
    mar_labels = {1: "Married", 2: "Single", 3: "Other"}

    for ax, col, labels in zip(axes, ["EDUCATION", "MARRIAGE"], [edu_labels, mar_labels]):
        rates = df.groupby(col)[target_col].mean() * 100
        rates.index = [labels.get(i, str(i)) for i in rates.index]

        bars = ax.bar(rates.index, rates.values, color=PALETTE_MULTI[:len(rates)],
                      edgecolor="white", linewidth=1.5, width=0.5)

        for bar, val in zip(bars, rates.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10,
            )

        ax.set_title(f"Default Rate by {col}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Default Rate (%)", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Default Rates across Demographic Categories", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save:
        save_fig(fig, "05_default_rate_by_category")
    return fig


def plot_boxplots(df, cols, target_col="DEFAULT", save=True, filename="06_boxplots"):
    """Box plots of selected features by default status."""
    n_cols = min(len(cols), 3)
    n_rows = (len(cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    for i, col in enumerate(cols):
        sns.boxplot(data=df, x=target_col, y=col, palette=PALETTE, ax=axes[i], width=0.4)
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].set_xticklabels(["No Default", "Default"])
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Default Status", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save:
        save_fig(fig, filename)
    return fig


# ---------- Model Evaluation Plots ----------

def plot_roc_curves(models_results, save=True):
    """
    Plot ROC curves for multiple models on the same axes.
    models_results: list of dicts with keys 'name', 'y_test', 'y_prob'
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, res in enumerate(models_results):
        fpr, tpr, _ = roc_curve(res["y_test"], res["y_prob"])
        auc_score = auc(fpr, tpr)
        color = PALETTE_MULTI[i % len(PALETTE_MULTI)]
        ax.plot(fpr, tpr, color=color, linewidth=2.5, label=f'{res["name"]} (AUC = {auc_score:.4f})')

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=11, frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save:
        save_fig(fig, "07_roc_curves")
    return fig


def plot_precision_recall_curves(models_results, save=True):
    """Plot Precision-Recall curves for multiple models."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, res in enumerate(models_results):
        precision, recall, _ = precision_recall_curve(res["y_test"], res["y_prob"])
        ap = res.get("ap", None)
        color = PALETTE_MULTI[i % len(PALETTE_MULTI)]
        label = f'{res["name"]}'
        if ap is not None:
            label += f" (AP = {ap:.4f})"
        ax.plot(recall, precision, color=color, linewidth=2.5, label=label)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper right", fontsize=11, frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save:
        save_fig(fig, "08_precision_recall_curves")
    return fig


def plot_confusion_matrix(y_test, y_pred, model_name="Model", save=True, filename="09_confusion_matrix"):
    """Plot a styled confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues",
        xticklabels=["No Default", "Default"],
        yticklabels=["No Default", "Default"],
        linewidths=1, linecolor="white",
        annot_kws={"size": 14, "fontweight": "bold"},
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    if save:
        save_fig(fig, filename)
    return fig


def plot_feature_importance(model, feature_names, top_n=15, save=True, filename="10_feature_importance"):
    """Plot feature importance bar chart for tree-based models."""
    fig, ax = plt.subplots(figsize=(9, 6))

    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    ax.barh(
        range(len(indices)),
        importances[indices],
        color=PALETTE[0],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save:
        save_fig(fig, filename)
    return fig


def plot_calibration(y_test, y_prob, model_name="Model", save=True, filename="11_calibration"):
    """Plot a reliability / calibration curve."""
    fig, ax = plt.subplots(figsize=(7, 6))

    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")

    ax.plot(prob_pred, prob_true, marker="o", linewidth=2.5, color=PALETTE[0], label=model_name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Perfectly Calibrated")
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curve", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save:
        save_fig(fig, filename)
    return fig


def plot_cost_analysis(cost_results, cost_fp=1, cost_fn=5, save=True, filename="14_cost_analysis"):
    """
    Grouped bar chart comparing FP cost vs. FN cost for each model.

    Parameters
    ----------
    cost_results : list of dicts from cost_sensitive_evaluation()
    cost_fp : float — cost per false positive (for subtitle)
    cost_fn : float — cost per false negative (for subtitle)
    """
    import pandas as pd

    df = pd.DataFrame(cost_results)
    models = df["Model"]
    x = np.arange(len(models))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_fp = ax.bar(x - width / 2, df["FP_Cost"], width,
                     label="FP Cost (false alarms)", color="#F59E0B",
                     edgecolor="white", linewidth=1)
    bars_fn = ax.bar(x + width / 2, df["FN_Cost"], width,
                     label="FN Cost (missed defaults)", color="#EF4444",
                     edgecolor="white", linewidth=1)

    # Annotate totals above bar groups
    for i, (fp_c, fn_c, total) in enumerate(zip(df["FP_Cost"], df["FN_Cost"], df["Total_Cost"])):
        max_bar = max(fp_c, fn_c)
        ax.text(i, max_bar + total * 0.03,
                f"Total: {total:,.0f}", ha="center", va="bottom",
                fontweight="bold", fontsize=10, color="#374151")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Cost (arbitrary units)", fontsize=12)
    ax.set_title("Cost-Sensitive Evaluation: FP vs. FN Loss by Model",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11, loc="upper right", frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add subtitle with cost ratio
    ax.text(0.5, -0.12,
            f"Cost ratio: FP = {cost_fp},  FN = {cost_fn}   (1 missed default costs {cost_fn}x a false alarm)",
            transform=ax.transAxes, ha="center", fontsize=10, color="#6B7280", style="italic")

    plt.tight_layout()
    if save:
        save_fig(fig, filename)
    return fig


def plot_cramers_v(df, cat_cols, target="DEFAULT", save=True, filename="15_cramers_v"):
    """
    Compute and plot Cramér's V association between categorical features
    and a binary target variable.

    Cramér's V is the appropriate association measure for categorical variables,
    unlike Pearson correlation which assumes continuous data.

    Parameters
    ----------
    df : DataFrame with the categorical columns and target
    cat_cols : list of categorical column names
    target : str — name of the target column
    """
    from scipy.stats import chi2_contingency

    results = {}
    for col in cat_cols:
        contingency = pd.crosstab(df[col], df[target])
        chi2, p, dof, expected = chi2_contingency(contingency)
        n = contingency.sum().sum()
        k = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * k)) if k > 0 else 0
        results[col] = {"cramers_v": cramers_v, "chi2": chi2, "p_value": p}

    res_df = pd.DataFrame(results).T.sort_values("cramers_v", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(cat_cols) * 0.6)))

    colors = ["#3B82F6" if p < 0.05 else "#9CA3AF" for p in res_df["p_value"]]
    bars = ax.barh(res_df.index, res_df["cramers_v"], color=colors,
                   edgecolor="white", linewidth=1, height=0.6)

    # Annotate values
    for bar, (_, row) in zip(bars, res_df.iterrows()):
        v = row["cramers_v"]
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else "ns"
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{v:.3f} {sig}', va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Cramér's V", fontsize=12)
    ax.set_title(f"Cramér's V: Categorical Features vs {target}",
                 fontsize=14, fontweight="bold", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(res_df["cramers_v"]) * 1.25)

    # Legend for significance
    ax.text(0.98, 0.02, "*** p<.001  ** p<.01  * p<.05  ns = not significant",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#6B7280", style="italic")

    plt.tight_layout()
    if save:
        save_fig(fig, filename)
    return fig, res_df


def plot_error_analysis(X_test, y_test, y_pred, feature_names=None,
                        top_n=6, save=True, filename="16_error_analysis"):
    """
    Analyse characteristics of misclassified samples.

    Compares KDE distributions of missed defaults (FN) vs caught defaults (TP)
    for the top features that differ most between these two groups.

    Parameters
    ----------
    X_test : array-like — test features (scaled or unscaled)
    y_test : array-like — true labels
    y_pred : array-like — predicted labels
    feature_names : list of str — column names
    top_n : int — number of top features to compare
    """
    from scipy.stats import gaussian_kde

    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df["y_true"] = np.array(y_test).ravel()
    test_df["y_pred"] = np.array(y_pred).ravel()

    # Split into outcome groups
    tp = test_df[(test_df["y_true"] == 1) & (test_df["y_pred"] == 1)]
    fn = test_df[(test_df["y_true"] == 1) & (test_df["y_pred"] == 0)]

    # Select top features by mean difference between FN and TP
    diff = (fn[feature_names].mean() - tp[feature_names].mean()).abs()
    top_features = diff.nlargest(top_n).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    for idx, feat in enumerate(top_features):
        ax = axes[idx]
        tp_vals = tp[feat].dropna().values
        fn_vals = fn[feat].dropna().values

        # Shared x range
        all_vals = np.concatenate([tp_vals, fn_vals])
        x_min, x_max = np.percentile(all_vals, [1, 99])
        x_grid = np.linspace(x_min, x_max, 200)

        # KDE curves
        if len(tp_vals) > 5:
            kde_tp = gaussian_kde(tp_vals, bw_method=0.3)
            ax.fill_between(x_grid, kde_tp(x_grid), alpha=0.35, color="#10B981", linewidth=0)
            ax.plot(x_grid, kde_tp(x_grid), color="#10B981", linewidth=2, label=f"TP — caught (n={len(tp_vals)})")
            ax.axvline(np.median(tp_vals), color="#10B981", linewidth=1.5, linestyle="--", alpha=0.7)

        if len(fn_vals) > 5:
            kde_fn = gaussian_kde(fn_vals, bw_method=0.3)
            ax.fill_between(x_grid, kde_fn(x_grid), alpha=0.35, color="#EF4444", linewidth=0)
            ax.plot(x_grid, kde_fn(x_grid), color="#EF4444", linewidth=2, label=f"FN — missed (n={len(fn_vals)})")
            ax.axvline(np.median(fn_vals), color="#EF4444", linewidth=1.5, linestyle="--", alpha=0.7)

        ax.set_title(feat, fontsize=11, fontweight="bold")
        ax.set_ylabel("Density", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        if idx == 0:
            ax.legend(fontsize=8, loc="upper right", frameon=False)

    fig.suptitle("Error Analysis: Caught Defaults (TP) vs Missed Defaults (FN)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.text(0.5, -0.01, "Dashed lines = median values",
             ha="center", fontsize=9, color="#6B7280", style="italic")
    plt.tight_layout()
    if save:
        save_fig(fig, filename)
    return fig


def plot_error_profile(X_test, y_test, y_pred, feature_names=None,
                       top_n=8, save=True, filename="17_error_profile"):
    """
    Bar chart comparing mean feature values for FN vs TP
    to reveal what makes a default 'hard to catch'.

    Parameters
    ----------
    X_test : array-like — test features
    y_test : array-like — true labels
    y_pred : array-like — predicted labels
    feature_names : list of str — column names
    top_n : int — number of features to display
    """
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df["y_true"] = np.array(y_test).ravel()
    test_df["y_pred"] = np.array(y_pred).ravel()

    # TP = caught defaults; FN = missed defaults
    tp = test_df[(test_df["y_true"] == 1) & (test_df["y_pred"] == 1)]
    fn = test_df[(test_df["y_true"] == 1) & (test_df["y_pred"] == 0)]

    # Mean difference (FN - TP) to show what's different about missed defaults
    tp_mean = tp[feature_names].mean()
    fn_mean = fn[feature_names].mean()
    diff = (fn_mean - tp_mean)
    top_diff = diff.abs().nlargest(top_n)
    selected = diff[top_diff.index].sort_values()

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.5)))

    colors = ["#EF4444" if v > 0 else "#3B82F6" for v in selected.values]
    ax.barh(selected.index, selected.values, color=colors,
            edgecolor="white", linewidth=1, height=0.6)

    ax.axvline(0, color="#374151", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Mean Difference (Missed Defaults − Caught Defaults)", fontsize=11)
    ax.set_title("Error Profile: What Makes a Default Hard to Catch?",
                 fontsize=14, fontweight="bold", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(0.98, 0.02,
            "Red = higher in missed defaults | Blue = lower in missed defaults",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#6B7280", style="italic")

    plt.tight_layout()
    if save:
        save_fig(fig, filename)
    return fig


def plot_final_model_comparison(final_df, save=True, filename="20_final_comparison"):
    """Grouped bar chart comparing all models on key metrics (test set)."""
    metrics = ["AUC-ROC", "F1-Score", "Recall", "Precision"]
    available = [m for m in metrics if m in final_df.columns]

    models = final_df.index.tolist()
    n_models = len(models)
    n_metrics = len(available)
    x = np.arange(n_metrics)
    bar_width = 0.8 / n_models

    colours = PALETTE_MULTI[:n_models] if n_models <= len(PALETTE_MULTI) else \
        PALETTE_MULTI + ["#9CA3AF"] * (n_models - len(PALETTE_MULTI))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(models):
        vals = [final_df.loc[model, m] for m in available]
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, label=model,
                      color=colours[i], edgecolor="white", linewidth=1.0, alpha=0.95)

        # Value labels above each bar
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(available, fontsize=11, fontweight="bold")
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Final Model Comparison — Test Set Performance",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(y=0.5, color="#D1D5DB", linestyle="--", linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    if save:
        save_fig(fig, filename)
    return fig
