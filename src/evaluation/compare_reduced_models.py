import os
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import ttest_rel, wilcoxon, binomtest, spearmanr


warnings.filterwarnings("ignore")


# ============================================================
# Project paths
# ============================================================

RESULTS_DIR = Path("outputs/results")
FIGURE_DIR = Path("outputs/figures")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# File prefixes
# ============================================================

BASELINE_PREFIX = "baseline_reduced"
DISTILBERT_PREFIX = "distilbert_reduced"


# ============================================================
# Utility functions
# ============================================================

def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def pretty_metric_name(metric: str) -> str:
    names = {
        "macro_f1": "Macro F1",
        "micro_f1": "Micro F1",
        "macro_precision": "Macro Precision",
        "micro_precision": "Micro Precision",
        "macro_recall": "Macro Recall",
        "micro_recall": "Micro Recall",
        "hamming_loss": "Hamming Loss",
        "exact_match_accuracy": "Exact-Match Accuracy",
    }
    return names.get(metric, metric)


def metric_direction(metric: str) -> str:
    """
    Most metrics are better when higher.
    Hamming loss is better when lower.
    """

    if metric == "hamming_loss":
        return "lower_is_better"

    return "higher_is_better"


def directional_improvement(
    baseline_value: float,
    distilbert_value: float,
    metric: str,
) -> float:
    """
    Positive value always means DistilBERT is better.
    """

    if metric_direction(metric) == "lower_is_better":
        return baseline_value - distilbert_value

    return distilbert_value - baseline_value


def relative_directional_change(
    baseline_value: float,
    distilbert_value: float,
    metric: str,
) -> float:
    """
    Relative percentage change where positive means improvement by DistilBERT.
    """

    if baseline_value == 0:
        return np.nan

    improvement = directional_improvement(
        baseline_value,
        distilbert_value,
        metric,
    )

    return (improvement / baseline_value) * 100


def check_required_files() -> None:
    required_files = [
        RESULTS_DIR / f"{BASELINE_PREFIX}_test_per_label_threshold_summary_metrics.json",
        RESULTS_DIR / f"{BASELINE_PREFIX}_test_per_label_threshold_per_label_metrics.csv",
        RESULTS_DIR / f"{BASELINE_PREFIX}_global_threshold_tuning.csv",
        RESULTS_DIR / f"{DISTILBERT_PREFIX}_test_per_label_threshold_summary_metrics.json",
        RESULTS_DIR / f"{DISTILBERT_PREFIX}_test_per_label_threshold_per_label_metrics.csv",
        RESULTS_DIR / f"{DISTILBERT_PREFIX}_global_threshold_tuning.csv",
    ]

    missing = [str(path) for path in required_files if not path.exists()]

    if missing:
        print("\nMissing required files:")
        for path in missing:
            print("-", path)

        raise FileNotFoundError(
            "Some required result files are missing. "
            "Make sure both reduced baseline and reduced DistilBERT experiments completed."
        )


def ordered_main_metrics() -> List[str]:
    return [
        "macro_f1",
        "micro_f1",
        "macro_precision",
        "micro_precision",
        "macro_recall",
        "micro_recall",
        "hamming_loss",
        "exact_match_accuracy",
    ]


# ============================================================
# Summary metric comparison
# ============================================================

def build_summary_comparison() -> pd.DataFrame:
    baseline = load_json(
        RESULTS_DIR / f"{BASELINE_PREFIX}_test_per_label_threshold_summary_metrics.json"
    )

    distilbert = load_json(
        RESULTS_DIR / f"{DISTILBERT_PREFIX}_test_per_label_threshold_summary_metrics.json"
    )

    records = []

    for metric in ordered_main_metrics():
        if metric not in baseline or metric not in distilbert:
            continue

        baseline_value = float(baseline[metric])
        distilbert_value = float(distilbert[metric])

        raw_difference = distilbert_value - baseline_value

        improvement = directional_improvement(
            baseline_value,
            distilbert_value,
            metric,
        )

        relative_change = relative_directional_change(
            baseline_value,
            distilbert_value,
            metric,
        )

        records.append({
            "metric": metric,
            "metric_pretty": pretty_metric_name(metric),
            "tfidf_logreg_reduced": baseline_value,
            "distilbert_reduced": distilbert_value,
            "raw_difference_distilbert_minus_tfidf": raw_difference,
            "directional_improvement_distilbert": improvement,
            "relative_directional_change_percent": relative_change,
            "better_model": (
                "DistilBERT"
                if improvement > 0
                else "TF-IDF Logistic Regression"
                if improvement < 0
                else "Tie"
            ),
            "interpretation_note": (
                "Lower is better"
                if metric_direction(metric) == "lower_is_better"
                else "Higher is better"
            ),
        })

    comparison_df = pd.DataFrame(records)

    output_path = RESULTS_DIR / "reduced_baseline_vs_distilbert_summary_comparison.csv"
    comparison_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")

    return comparison_df


# ============================================================
# Per-label comparison
# ============================================================

def build_per_label_comparison() -> pd.DataFrame:
    baseline = pd.read_csv(
        RESULTS_DIR / f"{BASELINE_PREFIX}_test_per_label_threshold_per_label_metrics.csv"
    )

    distilbert = pd.read_csv(
        RESULTS_DIR / f"{DISTILBERT_PREFIX}_test_per_label_threshold_per_label_metrics.csv"
    )

    required_cols = {"label", "support", "precision", "recall", "f1"}

    if not required_cols.issubset(set(baseline.columns)):
        raise ValueError("Baseline per-label metrics file has missing columns.")

    if not required_cols.issubset(set(distilbert.columns)):
        raise ValueError("DistilBERT per-label metrics file has missing columns.")

    merged = baseline.merge(
        distilbert,
        on="label",
        suffixes=("_tfidf", "_distilbert"),
        how="inner",
    )

    if len(merged) == 0:
        raise ValueError("No overlapping labels found between baseline and DistilBERT files.")

    merged["support"] = merged["support_tfidf"]

    support_mismatch = (
        merged["support_tfidf"].astype(int) != merged["support_distilbert"].astype(int)
    ).sum()

    if support_mismatch > 0:
        print(
            f"Warning: {support_mismatch} labels have different support values "
            "between the two result files."
        )

    merged["precision_difference"] = (
        merged["precision_distilbert"] - merged["precision_tfidf"]
    )

    merged["recall_difference"] = (
        merged["recall_distilbert"] - merged["recall_tfidf"]
    )

    merged["f1_difference"] = (
        merged["f1_distilbert"] - merged["f1_tfidf"]
    )

    merged["relative_f1_change_percent"] = np.where(
        merged["f1_tfidf"] != 0,
        (merged["f1_difference"] / merged["f1_tfidf"]) * 100,
        np.nan,
    )

    merged = merged.sort_values("f1_difference", ascending=False)

    output_path = RESULTS_DIR / "reduced_per_label_baseline_vs_distilbert_comparison.csv"
    merged.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")

    return merged


# ============================================================
# Statistical analysis
# ============================================================

def compute_label_level_statistical_tests(per_label_df: pd.DataFrame) -> pd.DataFrame:
    """
    Statistical evidence over paired label-level metrics.

    Important dissertation note:
    These are label-level paired tests across the same 11 labels.
    They are useful supporting evidence, but they are not a full
    instance-level significance test such as McNemar's test.
    """

    records = []

    for metric_key, metric_name in [
        ("f1", "F1"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ]:
        tfidf_values = per_label_df[f"{metric_key}_tfidf"].astype(float).values
        distilbert_values = per_label_df[f"{metric_key}_distilbert"].astype(float).values

        differences = distilbert_values - tfidf_values

        mean_diff = float(np.mean(differences))
        median_diff = float(np.median(differences))
        std_diff = float(np.std(differences, ddof=1))

        cohens_dz = mean_diff / std_diff if std_diff > 0 else np.nan

        improved = int((differences > 0).sum())
        worsened = int((differences < 0).sum())
        tied = int((differences == 0).sum())

        t_stat, t_p = ttest_rel(distilbert_values, tfidf_values)

        try:
            w_stat, w_p = wilcoxon(
                distilbert_values,
                tfidf_values,
                zero_method="wilcox",
                alternative="two-sided",
            )
        except ValueError:
            w_stat, w_p = np.nan, np.nan

        non_tied_n = improved + worsened

        if non_tied_n > 0:
            sign_result = binomtest(
                improved,
                n=non_tied_n,
                p=0.5,
                alternative="two-sided",
            )
            sign_p = float(sign_result.pvalue)
        else:
            sign_p = np.nan

        records.append({
            "metric": metric_name,
            "number_of_labels": int(len(differences)),
            "mean_difference_distilbert_minus_tfidf": mean_diff,
            "median_difference_distilbert_minus_tfidf": median_diff,
            "std_difference": std_diff,
            "cohens_dz": cohens_dz,
            "labels_improved": improved,
            "labels_worsened": worsened,
            "labels_tied": tied,
            "paired_t_statistic": float(t_stat),
            "paired_t_test_p_value": float(t_p),
            "wilcoxon_statistic": float(w_stat) if not np.isnan(w_stat) else np.nan,
            "wilcoxon_p_value": float(w_p) if not np.isnan(w_p) else np.nan,
            "sign_test_p_value": sign_p,
            "note": "Paired label-level test across reduced emotion labels",
        })

    stats_df = pd.DataFrame(records)

    output_path = RESULTS_DIR / "reduced_model_label_level_statistical_tests.csv"
    stats_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")

    return stats_df


def bootstrap_mean_difference(
    per_label_df: pd.DataFrame,
    metric_key: str,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap 95% confidence interval for mean paired label-level difference.
    """

    rng = np.random.default_rng(seed)

    differences = (
        per_label_df[f"{metric_key}_distilbert"].astype(float).values
        - per_label_df[f"{metric_key}_tfidf"].astype(float).values
    )

    n = len(differences)

    bootstrap_indices = rng.integers(0, n, size=(n_bootstrap, n))
    bootstrap_means = differences[bootstrap_indices].mean(axis=1)

    mean_diff = float(np.mean(differences))
    lower = float(np.percentile(bootstrap_means, 2.5))
    upper = float(np.percentile(bootstrap_means, 97.5))

    return mean_diff, lower, upper


def compute_bootstrap_intervals(per_label_df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for metric_key, metric_pretty in [
        ("f1", "F1"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ]:
        mean_diff, lower, upper = bootstrap_mean_difference(
            per_label_df,
            metric_key=metric_key,
        )

        records.append({
            "metric": metric_pretty,
            "mean_difference_distilbert_minus_tfidf": mean_diff,
            "bootstrap_95_ci_lower": lower,
            "bootstrap_95_ci_upper": upper,
            "ci_excludes_zero": bool(lower > 0 or upper < 0),
            "note": "Bootstrap interval over paired label-level differences",
        })

    bootstrap_df = pd.DataFrame(records)

    output_path = RESULTS_DIR / "reduced_model_bootstrap_mean_difference_ci.csv"
    bootstrap_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")

    return bootstrap_df


def compute_support_correlations(per_label_df: pd.DataFrame) -> pd.DataFrame:
    """
    Spearman correlations between label support and metrics.
    This helps discuss label imbalance.
    """

    records = []

    for model_name, suffix in [
        ("TF-IDF Logistic Regression", "tfidf"),
        ("DistilBERT", "distilbert"),
    ]:
        for metric_key, metric_name in [
            ("f1", "F1"),
            ("precision", "Precision"),
            ("recall", "Recall"),
        ]:
            corr, p_value = spearmanr(
                per_label_df["support"].astype(float),
                per_label_df[f"{metric_key}_{suffix}"].astype(float),
            )

            records.append({
                "model": model_name,
                "metric": metric_name,
                "spearman_correlation_support_vs_metric": float(corr),
                "p_value": float(p_value),
                "note": "Positive values suggest higher-support labels tend to perform better",
            })

    corr_delta, p_delta = spearmanr(
        per_label_df["support"].astype(float),
        per_label_df["f1_difference"].astype(float),
    )

    records.append({
        "model": "Difference: DistilBERT minus TF-IDF",
        "metric": "F1 difference",
        "spearman_correlation_support_vs_metric": float(corr_delta),
        "p_value": float(p_delta),
        "note": "Correlation between support and DistilBERT F1 gain",
    })

    corr_df = pd.DataFrame(records)

    output_path = RESULTS_DIR / "reduced_support_metric_correlations.csv"
    corr_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")

    return corr_df


# ============================================================
# Training-time comparison
# ============================================================

def get_baseline_training_time_seconds() -> Optional[float]:
    """
    Baseline training time is usually stored in tuning results.
    Use the best validation macro F1 row as the selected baseline config.
    """

    tuning_path = RESULTS_DIR / f"{BASELINE_PREFIX}_model_tuning_results.csv"

    if not tuning_path.exists():
        return None

    tuning = pd.read_csv(tuning_path)

    if "training_time_seconds" not in tuning.columns:
        return None

    if "val_macro_f1_threshold_0_5" in tuning.columns:
        best_row = tuning.sort_values(
            "val_macro_f1_threshold_0_5",
            ascending=False,
        ).iloc[0]
    else:
        best_row = tuning.iloc[0]

    return float(best_row["training_time_seconds"])


def get_distilbert_training_time_seconds() -> Optional[float]:
    info_path = RESULTS_DIR / f"{DISTILBERT_PREFIX}_experiment_info.json"
    info = safe_load_json(info_path)

    if info is None:
        return None

    value = info.get("training_time_seconds", None)

    if value is None:
        return None

    return float(value)


def build_training_time_comparison() -> Optional[pd.DataFrame]:
    baseline_time = get_baseline_training_time_seconds()
    distilbert_time = get_distilbert_training_time_seconds()

    if baseline_time is None or distilbert_time is None:
        print("Skipping training-time comparison because timing information is incomplete.")
        return None

    df = pd.DataFrame({
        "model": ["TF-IDF Logistic Regression", "DistilBERT"],
        "training_time_seconds": [baseline_time, distilbert_time],
        "training_time_minutes": [baseline_time / 60, distilbert_time / 60],
    })

    output_path = RESULTS_DIR / "reduced_training_time_comparison.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")

    return df


# ============================================================
# Visualisation helpers
# ============================================================

def save_current_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_summary_metric_comparison(comparison_df: pd.DataFrame) -> None:
    df = comparison_df.copy()
    df = df.set_index("metric").loc[ordered_main_metrics()].reset_index()

    x = np.arange(len(df))
    width = 0.38

    plt.figure(figsize=(13, 6))
    plt.bar(
        x - width / 2,
        df["tfidf_logreg_reduced"],
        width,
        label="TF-IDF Logistic Regression",
    )
    plt.bar(
        x + width / 2,
        df["distilbert_reduced"],
        width,
        label="DistilBERT",
    )

    plt.xticks(x, df["metric_pretty"], rotation=30, ha="right")
    plt.ylabel("Score")
    plt.title("Reduced-label task: TF-IDF Logistic Regression vs DistilBERT")
    plt.legend()

    save_current_figure(
        FIGURE_DIR / "reduced_baseline_vs_distilbert_metric_comparison.png"
    )


def plot_summary_metric_delta(comparison_df: pd.DataFrame) -> None:
    df = comparison_df.copy()
    df = df.sort_values("directional_improvement_distilbert", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(
        df["metric_pretty"],
        df["directional_improvement_distilbert"],
    )

    plt.axvline(0, linewidth=1)
    plt.xlabel("Directional improvement by DistilBERT")
    plt.ylabel("Metric")
    plt.title("Reduced-label task: DistilBERT improvement over TF-IDF")

    save_current_figure(
        FIGURE_DIR / "reduced_distilbert_minus_baseline_metric_delta.png"
    )


def plot_per_label_f1_comparison(per_label_df: pd.DataFrame) -> None:
    df = per_label_df.sort_values("f1_distilbert", ascending=True)

    y = np.arange(len(df))
    height = 0.38

    plt.figure(figsize=(10, 7))
    plt.barh(
        y - height / 2,
        df["f1_tfidf"],
        height,
        label="TF-IDF Logistic Regression",
    )
    plt.barh(
        y + height / 2,
        df["f1_distilbert"],
        height,
        label="DistilBERT",
    )

    plt.yticks(y, df["label"])
    plt.xlabel("F1 score")
    plt.ylabel("Emotion label")
    plt.title("Reduced-label task: per-label F1 comparison")
    plt.legend()

    save_current_figure(
        FIGURE_DIR / "reduced_per_label_f1_baseline_vs_distilbert.png"
    )


def plot_per_label_f1_difference(per_label_df: pd.DataFrame) -> None:
    df = per_label_df.sort_values("f1_difference", ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(df["label"], df["f1_difference"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("F1 difference: DistilBERT minus TF-IDF")
    plt.ylabel("Emotion label")
    plt.title("Reduced-label task: per-label F1 improvement")

    save_current_figure(
        FIGURE_DIR / "reduced_per_label_f1_difference_distilbert_minus_baseline.png"
    )


def plot_support_vs_f1(per_label_df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 6))

    plt.scatter(
        per_label_df["support"],
        per_label_df["f1_tfidf"],
        label="TF-IDF Logistic Regression",
    )

    plt.scatter(
        per_label_df["support"],
        per_label_df["f1_distilbert"],
        label="DistilBERT",
    )

    for _, row in per_label_df.iterrows():
        plt.annotate(
            row["label"],
            (row["support"], row["f1_distilbert"]),
            fontsize=8,
            alpha=0.8,
        )

    plt.xlabel("Test support")
    plt.ylabel("F1 score")
    plt.title("Reduced-label task: relationship between support and F1")
    plt.legend()

    save_current_figure(
        FIGURE_DIR / "reduced_support_vs_f1_baseline_vs_distilbert.png"
    )


def plot_f1_gain_vs_support(per_label_df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 6))

    plt.scatter(
        per_label_df["support"],
        per_label_df["f1_difference"],
    )

    plt.axhline(0, linewidth=1)

    for _, row in per_label_df.iterrows():
        plt.annotate(
            row["label"],
            (row["support"], row["f1_difference"]),
            fontsize=8,
            alpha=0.8,
        )

    plt.xlabel("Test support")
    plt.ylabel("F1 gain: DistilBERT minus TF-IDF")
    plt.title("Reduced-label task: does DistilBERT gain depend on label support?")

    save_current_figure(
        FIGURE_DIR / "reduced_f1_gain_vs_support.png"
    )


def plot_precision_recall_space(per_label_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 6))

    plt.scatter(
        per_label_df["precision_tfidf"],
        per_label_df["recall_tfidf"],
        label="TF-IDF Logistic Regression",
    )

    plt.scatter(
        per_label_df["precision_distilbert"],
        per_label_df["recall_distilbert"],
        label="DistilBERT",
    )

    for _, row in per_label_df.iterrows():
        plt.annotate(
            row["label"],
            (row["precision_distilbert"], row["recall_distilbert"]),
            fontsize=8,
            alpha=0.8,
        )

    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title("Reduced-label task: precision-recall behaviour by label")
    plt.legend()

    save_current_figure(
        FIGURE_DIR / "reduced_precision_recall_space_baseline_vs_distilbert.png"
    )


def plot_threshold_tuning_comparison() -> None:
    baseline = pd.read_csv(
        RESULTS_DIR / f"{BASELINE_PREFIX}_global_threshold_tuning.csv"
    )

    distilbert = pd.read_csv(
        RESULTS_DIR / f"{DISTILBERT_PREFIX}_global_threshold_tuning.csv"
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        baseline["threshold"],
        baseline["macro_f1"],
        marker="o",
        label="TF-IDF macro F1",
    )

    plt.plot(
        distilbert["threshold"],
        distilbert["macro_f1"],
        marker="o",
        label="DistilBERT macro F1",
    )

    plt.plot(
        baseline["threshold"],
        baseline["micro_f1"],
        marker="x",
        label="TF-IDF micro F1",
    )

    plt.plot(
        distilbert["threshold"],
        distilbert["micro_f1"],
        marker="x",
        label="DistilBERT micro F1",
    )

    plt.xlabel("Global decision threshold")
    plt.ylabel("Validation F1 score")
    plt.title("Reduced-label task: validation threshold tuning comparison")
    plt.legend()

    save_current_figure(
        FIGURE_DIR / "reduced_threshold_tuning_baseline_vs_distilbert.png"
    )


def plot_per_label_thresholds() -> None:
    baseline_path = RESULTS_DIR / f"{BASELINE_PREFIX}_per_label_thresholds.csv"
    distilbert_path = RESULTS_DIR / f"{DISTILBERT_PREFIX}_per_label_thresholds.csv"

    if not baseline_path.exists() or not distilbert_path.exists():
        print("Skipping per-label threshold plot because threshold CSV files are missing.")
        return

    baseline = pd.read_csv(baseline_path)
    distilbert = pd.read_csv(distilbert_path)

    required = {"label", "best_threshold"}

    if not required.issubset(set(baseline.columns)):
        print("Skipping threshold plot: baseline threshold file missing required columns.")
        return

    if not required.issubset(set(distilbert.columns)):
        print("Skipping threshold plot: DistilBERT threshold file missing required columns.")
        return

    merged = baseline.merge(
        distilbert,
        on="label",
        suffixes=("_tfidf", "_distilbert"),
        how="inner",
    )

    merged = merged.sort_values("best_threshold_distilbert", ascending=True)

    y = np.arange(len(merged))
    height = 0.38

    plt.figure(figsize=(10, 7))
    plt.barh(
        y - height / 2,
        merged["best_threshold_tfidf"],
        height,
        label="TF-IDF Logistic Regression",
    )
    plt.barh(
        y + height / 2,
        merged["best_threshold_distilbert"],
        height,
        label="DistilBERT",
    )

    plt.yticks(y, merged["label"])
    plt.xlabel("Best validation threshold")
    plt.ylabel("Emotion label")
    plt.title("Reduced-label task: per-label threshold comparison")
    plt.legend()

    save_current_figure(
        FIGURE_DIR / "reduced_per_label_thresholds_baseline_vs_distilbert.png"
    )


def plot_training_time_comparison(training_time_df: Optional[pd.DataFrame]) -> None:
    if training_time_df is None:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(
        training_time_df["model"],
        training_time_df["training_time_minutes"],
    )

    plt.ylabel("Training time in minutes")
    plt.title("Reduced-label task: training time comparison")
    plt.xticks(rotation=15, ha="right")

    save_current_figure(
        FIGURE_DIR / "reduced_training_time_baseline_vs_distilbert.png"
    )


def plot_label_metric_heatmap(per_label_df: pd.DataFrame) -> None:
    """
    Creates a compact heatmap of label-level metric differences.
    Positive values mean DistilBERT improved over TF-IDF.
    """

    df = per_label_df.sort_values("f1_difference", ascending=False).copy()

    matrix = df[
        [
            "precision_difference",
            "recall_difference",
            "f1_difference",
        ]
    ].values

    labels = df["label"].tolist()
    metric_labels = ["Precision gain", "Recall gain", "F1 gain"]

    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label="DistilBERT minus TF-IDF")

    plt.yticks(np.arange(len(labels)), labels)
    plt.xticks(np.arange(len(metric_labels)), metric_labels, rotation=20, ha="right")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.title("Reduced-label task: label-level metric gains")

    save_current_figure(
        FIGURE_DIR / "reduced_label_metric_gain_heatmap.png"
    )


# ============================================================
# Markdown summary
# ============================================================

def save_findings_summary(
    comparison_df: pd.DataFrame,
    per_label_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    training_time_df: Optional[pd.DataFrame],
) -> None:
    def metric_row(metric: str) -> pd.Series:
        return comparison_df[comparison_df["metric"] == metric].iloc[0]

    macro_f1 = metric_row("macro_f1")
    micro_f1 = metric_row("micro_f1")
    macro_recall = metric_row("macro_recall")
    hamming = metric_row("hamming_loss")
    exact = metric_row("exact_match_accuracy")

    f1_stats = stats_df[stats_df["metric"] == "F1"].iloc[0]
    f1_bootstrap = bootstrap_df[bootstrap_df["metric"] == "F1"].iloc[0]

    best_f1_gains = per_label_df.sort_values(
        "f1_difference",
        ascending=False,
    ).head(5)

    weakest_f1_gains = per_label_df.sort_values(
        "f1_difference",
        ascending=True,
    ).head(5)

    lines = []

    lines.append("# Reduced-label DistilBERT vs TF-IDF Logistic Regression Findings")
    lines.append("")
    lines.append("## Summary metric comparison")
    lines.append("")
    lines.append(
        f"Macro F1 increased from {macro_f1['tfidf_logreg_reduced']:.4f} "
        f"for TF-IDF Logistic Regression to {macro_f1['distilbert_reduced']:.4f} "
        f"for DistilBERT. The absolute improvement was "
        f"{macro_f1['directional_improvement_distilbert']:.4f}."
    )
    lines.append(
        f"Micro F1 increased from {micro_f1['tfidf_logreg_reduced']:.4f} "
        f"to {micro_f1['distilbert_reduced']:.4f}, giving an absolute improvement "
        f"of {micro_f1['directional_improvement_distilbert']:.4f}."
    )
    lines.append(
        f"Macro recall increased from {macro_recall['tfidf_logreg_reduced']:.4f} "
        f"to {macro_recall['distilbert_reduced']:.4f}."
    )
    lines.append(
        f"Exact-match accuracy increased from {exact['tfidf_logreg_reduced']:.4f} "
        f"to {exact['distilbert_reduced']:.4f}."
    )
    lines.append(
        f"Hamming loss decreased from {hamming['tfidf_logreg_reduced']:.4f} "
        f"to {hamming['distilbert_reduced']:.4f}. Since lower Hamming loss is better, "
        "this also favours DistilBERT."
    )

    lines.append("")
    lines.append("## Label-level statistical evidence")
    lines.append("")
    lines.append(
        f"Across the reduced label set, the mean per-label F1 difference "
        f"(DistilBERT minus TF-IDF) was {f1_stats['mean_difference_distilbert_minus_tfidf']:.4f}. "
        f"The median difference was {f1_stats['median_difference_distilbert_minus_tfidf']:.4f}."
    )
    lines.append(
        f"DistilBERT improved F1 on {int(f1_stats['labels_improved'])} labels, "
        f"performed worse on {int(f1_stats['labels_worsened'])} labels, "
        f"and tied on {int(f1_stats['labels_tied'])} labels."
    )
    lines.append(
        f"The paired t-test p-value for label-level F1 was "
        f"{f1_stats['paired_t_test_p_value']:.6f}; the Wilcoxon signed-rank "
        f"test p-value was {f1_stats['wilcoxon_p_value']:.6f}; and the sign-test "
        f"p-value was {f1_stats['sign_test_p_value']:.6f}."
    )
    lines.append(
        f"The bootstrap 95% confidence interval for the mean per-label F1 difference "
        f"was [{f1_bootstrap['bootstrap_95_ci_lower']:.4f}, "
        f"{f1_bootstrap['bootstrap_95_ci_upper']:.4f}]."
    )
    lines.append(
        "These tests are based on the 11 paired emotion labels and should be interpreted "
        "as supporting label-level statistical evidence rather than as an instance-level "
        "significance test."
    )

    lines.append("")
    lines.append("## Strongest F1 improvements")
    lines.append("")
    for _, row in best_f1_gains.iterrows():
        lines.append(
            f"- {row['label']}: TF-IDF F1={row['f1_tfidf']:.4f}, "
            f"DistilBERT F1={row['f1_distilbert']:.4f}, "
            f"difference={row['f1_difference']:.4f}."
        )

    lines.append("")
    lines.append("## Weakest F1 improvements")
    lines.append("")
    for _, row in weakest_f1_gains.iterrows():
        lines.append(
            f"- {row['label']}: TF-IDF F1={row['f1_tfidf']:.4f}, "
            f"DistilBERT F1={row['f1_distilbert']:.4f}, "
            f"difference={row['f1_difference']:.4f}."
        )

    lines.append("")
    lines.append("## Visual evidence files")
    lines.append("")
    figure_files = [
        "outputs/figures/reduced_baseline_vs_distilbert_metric_comparison.png",
        "outputs/figures/reduced_distilbert_minus_baseline_metric_delta.png",
        "outputs/figures/reduced_per_label_f1_baseline_vs_distilbert.png",
        "outputs/figures/reduced_per_label_f1_difference_distilbert_minus_baseline.png",
        "outputs/figures/reduced_support_vs_f1_baseline_vs_distilbert.png",
        "outputs/figures/reduced_f1_gain_vs_support.png",
        "outputs/figures/reduced_precision_recall_space_baseline_vs_distilbert.png",
        "outputs/figures/reduced_threshold_tuning_baseline_vs_distilbert.png",
        "outputs/figures/reduced_per_label_thresholds_baseline_vs_distilbert.png",
        "outputs/figures/reduced_label_metric_gain_heatmap.png",
        "outputs/figures/reduced_training_time_baseline_vs_distilbert.png",
    ]

    for file_path in figure_files:
        lines.append(f"- {file_path}")

    output_path = RESULTS_DIR / "reduced_distilbert_vs_baseline_findings_summary.md"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved: {output_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("\nChecking required files...")
    check_required_files()

    print("\nBuilding summary metric comparison...")
    comparison_df = build_summary_comparison()

    print("\nBuilding per-label comparison...")
    per_label_df = build_per_label_comparison()

    print("\nComputing label-level statistical tests...")
    stats_df = compute_label_level_statistical_tests(per_label_df)

    print("\nComputing bootstrap confidence intervals...")
    bootstrap_df = compute_bootstrap_intervals(per_label_df)

    print("\nComputing support-performance correlations...")
    corr_df = compute_support_correlations(per_label_df)

    print("\nBuilding training-time comparison where possible...")
    training_time_df = build_training_time_comparison()

    print("\nCreating visualisations...")
    plot_summary_metric_comparison(comparison_df)
    plot_summary_metric_delta(comparison_df)
    plot_per_label_f1_comparison(per_label_df)
    plot_per_label_f1_difference(per_label_df)
    plot_support_vs_f1(per_label_df)
    plot_f1_gain_vs_support(per_label_df)
    plot_precision_recall_space(per_label_df)
    plot_threshold_tuning_comparison()
    plot_per_label_thresholds()
    plot_training_time_comparison(training_time_df)
    plot_label_metric_heatmap(per_label_df)

    print("\nSaving markdown findings summary...")
    save_findings_summary(
        comparison_df=comparison_df,
        per_label_df=per_label_df,
        stats_df=stats_df,
        bootstrap_df=bootstrap_df,
        corr_df=corr_df,
        training_time_df=training_time_df,
    )

    print("\nReduced-label model comparison completed successfully.")
    print("\nKey outputs:")
    print("- outputs/results/reduced_baseline_vs_distilbert_summary_comparison.csv")
    print("- outputs/results/reduced_per_label_baseline_vs_distilbert_comparison.csv")
    print("- outputs/results/reduced_model_label_level_statistical_tests.csv")
    print("- outputs/results/reduced_model_bootstrap_mean_difference_ci.csv")
    print("- outputs/results/reduced_support_metric_correlations.csv")
    print("- outputs/results/reduced_distilbert_vs_baseline_findings_summary.md")


if __name__ == "__main__":
    main()