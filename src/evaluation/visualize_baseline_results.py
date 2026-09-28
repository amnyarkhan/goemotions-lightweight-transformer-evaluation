import os
import json
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = "outputs/results"
FIGURE_DIR = "outputs/figures"

os.makedirs(FIGURE_DIR, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_summary_metrics():
    """
    Loads full and reduced baseline summary metrics.

    This assumes both experiments used per-label thresholds, which is the
    main thresholding method for imbalanced multi-label classification.
    """

    full_path = os.path.join(
        RESULTS_DIR,
        "baseline_full_test_per_label_threshold_summary_metrics.json",
    )

    reduced_path = os.path.join(
        RESULTS_DIR,
        "baseline_reduced_test_per_label_threshold_summary_metrics.json",
    )

    full_metrics = load_json(full_path)
    reduced_metrics = load_json(reduced_path)

    records = []

    for metric, value in full_metrics.items():
        records.append({
            "setting": "Full 28-label task",
            "metric": metric,
            "value": value,
        })

    for metric, value in reduced_metrics.items():
        records.append({
            "setting": "Reduced 11-label task",
            "metric": metric,
            "value": value,
        })

    return pd.DataFrame(records)


def pretty_metric_name(metric):
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


def plot_metric_comparison(df):
    """
    Bar chart comparing full vs reduced label settings.
    """

    selected_metrics = [
        "macro_f1",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "hamming_loss",
        "exact_match_accuracy",
    ]

    plot_df = df[df["metric"].isin(selected_metrics)].copy()
    plot_df["metric_pretty"] = plot_df["metric"].apply(pretty_metric_name)

    pivot_df = plot_df.pivot(
        index="metric_pretty",
        columns="setting",
        values="value",
    )

    pivot_df = pivot_df.loc[
        [
            "Macro F1",
            "Micro F1",
            "Macro Precision",
            "Macro Recall",
            "Hamming Loss",
            "Exact-Match Accuracy",
        ]
    ]

    ax = pivot_df.plot(kind="bar", figsize=(12, 6))

    plt.title("Baseline Performance: Full 28-label Task vs Reduced 11-label Task")
    plt.xlabel("Evaluation Metric")
    plt.ylabel("Score")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Label Setting")
    plt.tight_layout()

    output_path = os.path.join(
        FIGURE_DIR,
        "baseline_full_vs_reduced_metric_comparison.png",
    )
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_metric_delta(df):
    """
    Shows how much the reduced-label setting changes each metric
    compared with the full-label setting.
    """

    pivot_df = df.pivot(
        index="metric",
        columns="setting",
        values="value",
    )

    pivot_df["change_reduced_minus_full"] = (
        pivot_df["Reduced 11-label task"] - pivot_df["Full 28-label task"]
    )

    selected_metrics = [
        "macro_f1",
        "micro_f1",
        "macro_precision",
        "micro_precision",
        "macro_recall",
        "micro_recall",
        "hamming_loss",
        "exact_match_accuracy",
    ]

    delta_df = pivot_df.loc[selected_metrics, ["change_reduced_minus_full"]].copy()
    delta_df["metric_pretty"] = [pretty_metric_name(m) for m in delta_df.index]

    ax = delta_df.plot(
        x="metric_pretty",
        y="change_reduced_minus_full",
        kind="bar",
        figsize=(12, 6),
        legend=False,
    )

    plt.axhline(0, linewidth=1)
    plt.title("Change in Baseline Performance after Reducing the Label Space")
    plt.xlabel("Evaluation Metric")
    plt.ylabel("Reduced task score minus full task score")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_path = os.path.join(
        FIGURE_DIR,
        "baseline_reduced_minus_full_metric_change.png",
    )
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_per_label_f1(label_mode):
    """
    Creates a per-label F1 bar chart.
    """

    path = os.path.join(
        RESULTS_DIR,
        f"baseline_{label_mode}_test_per_label_threshold_per_label_metrics.csv",
    )

    df = pd.read_csv(path)
    df = df.sort_values("f1", ascending=True)

    title_label = "Full 28-label task" if label_mode == "full" else "Reduced 11-label task"

    plt.figure(figsize=(10, max(6, len(df) * 0.35)))
    plt.barh(df["label"], df["f1"])
    plt.title(f"Per-Label F1 Scores for Baseline Model: {title_label}")
    plt.xlabel("F1 Score")
    plt.ylabel("Emotion Label")
    plt.tight_layout()

    output_path = os.path.join(
        FIGURE_DIR,
        f"baseline_{label_mode}_per_label_f1.png",
    )
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_support_vs_f1(label_mode):
    """
    Scatterplot showing relation between label frequency/support and F1.
    This directly supports the label imbalance research question.
    """

    path = os.path.join(
        RESULTS_DIR,
        f"baseline_{label_mode}_test_per_label_threshold_per_label_metrics.csv",
    )

    df = pd.read_csv(path)

    plt.figure(figsize=(9, 6))
    plt.scatter(df["support"], df["f1"])

    for _, row in df.iterrows():
        plt.annotate(
            row["label"],
            (row["support"], row["f1"]),
            fontsize=8,
            alpha=0.8,
        )

    title_label = "Full 28-label task" if label_mode == "full" else "Reduced 11-label task"

    plt.title(f"Relationship between Label Support and F1: {title_label}")
    plt.xlabel("Test Support: Number of True Examples")
    plt.ylabel("F1 Score")
    plt.tight_layout()

    output_path = os.path.join(
        FIGURE_DIR,
        f"baseline_{label_mode}_support_vs_f1.png",
    )
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_threshold_curve(label_mode):
    """
    Plots validation macro F1 and micro F1 across global thresholds.
    This supports the methodology decision to tune thresholds on validation data.
    """

    path = os.path.join(
        RESULTS_DIR,
        f"baseline_{label_mode}_global_threshold_tuning.csv",
    )

    df = pd.read_csv(path)

    plt.figure(figsize=(9, 6))
    plt.plot(df["threshold"], df["macro_f1"], marker="o", label="Macro F1")
    plt.plot(df["threshold"], df["micro_f1"], marker="o", label="Micro F1")

    title_label = "Full 28-label task" if label_mode == "full" else "Reduced 11-label task"

    plt.title(f"Validation Threshold Tuning Curve: {title_label}")
    plt.xlabel("Global Decision Threshold")
    plt.ylabel("Validation F1 Score")
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(
        FIGURE_DIR,
        f"baseline_{label_mode}_threshold_tuning_curve.png",
    )
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def save_summary_table(df):
    pivot_df = df.pivot(
        index="metric",
        columns="setting",
        values="value",
    )

    pivot_df["change_reduced_minus_full"] = (
        pivot_df["Reduced 11-label task"] - pivot_df["Full 28-label task"]
    )

    pivot_df["relative_change_percent"] = (
        pivot_df["change_reduced_minus_full"] / pivot_df["Full 28-label task"] * 100
    )

    pivot_df = pivot_df.reset_index()
    pivot_df["metric"] = pivot_df["metric"].apply(pretty_metric_name)

    output_path = os.path.join(
        RESULTS_DIR,
        "baseline_full_vs_reduced_summary_table.csv",
    )

    pivot_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")


def main():
    metrics_df = load_summary_metrics()

    save_summary_table(metrics_df)

    plot_metric_comparison(metrics_df)
    plot_metric_delta(metrics_df)

    plot_per_label_f1("full")
    plot_per_label_f1("reduced")

    plot_support_vs_f1("full")
    plot_support_vs_f1("reduced")

    plot_threshold_curve("full")
    plot_threshold_curve("reduced")

    print("\nAll baseline visualisations created successfully.")


if __name__ == "__main__":
    main()