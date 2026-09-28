import os
import json
import pandas as pd


RESULTS_DIR = "outputs/results"


def load_json(path):
    """Load a JSON result file from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    """Create a concise Markdown summary of baseline model performance."""
    full_summary = load_json(
        os.path.join(
            RESULTS_DIR,
            "baseline_full_test_per_label_threshold_summary_metrics.json",
        )
    )

    reduced_summary = load_json(
        os.path.join(
            RESULTS_DIR,
            "baseline_reduced_test_per_label_threshold_summary_metrics.json",
        )
    )

    full_per_label = pd.read_csv(
        os.path.join(
            RESULTS_DIR,
            "baseline_full_test_per_label_threshold_per_label_metrics.csv",
        )
    )

    reduced_per_label = pd.read_csv(
        os.path.join(
            RESULTS_DIR,
            "baseline_reduced_test_per_label_threshold_per_label_metrics.csv",
        )
    )

    strongest_full = full_per_label.sort_values("f1", ascending=False).head(10)
    weakest_full = full_per_label.sort_values("f1", ascending=True).head(10)

    summary_lines = []

    summary_lines.append("# Baseline Result Summary")
    summary_lines.append("")
    summary_lines.append("## Full vs Reduced Baseline")
    summary_lines.append("")
    summary_lines.append(
        f"Macro F1 improved from {full_summary['macro_f1']:.3f} "
        f"to {reduced_summary['macro_f1']:.3f}."
    )
    summary_lines.append(
        f"Micro F1 improved from {full_summary['micro_f1']:.3f} "
        f"to {reduced_summary['micro_f1']:.3f}."
    )
    summary_lines.append(
        f"Macro recall improved from {full_summary['macro_recall']:.3f} "
        f"to {reduced_summary['macro_recall']:.3f}."
    )
    summary_lines.append(
        f"Exact-match accuracy improved from {full_summary['exact_match_accuracy']:.3f} "
        f"to {reduced_summary['exact_match_accuracy']:.3f}."
    )
    summary_lines.append(
        f"Hamming loss increased from {full_summary['hamming_loss']:.3f} "
        f"to {reduced_summary['hamming_loss']:.3f}; this should be interpreted carefully "
        "because Hamming loss is affected by the number of labels."
    )

    summary_lines.append("")
    summary_lines.append("## Strongest Full-Label Emotions")
    summary_lines.append("")

    for _, row in strongest_full.iterrows():
        summary_lines.append(
            f"- {row['label']}: F1={row['f1']:.3f}, "
            f"precision={row['precision']:.3f}, recall={row['recall']:.3f}, "
            f"support={int(row['support'])}"
        )

    summary_lines.append("")
    summary_lines.append("## Weakest Full-Label Emotions")
    summary_lines.append("")

    for _, row in weakest_full.iterrows():
        summary_lines.append(
            f"- {row['label']}: F1={row['f1']:.3f}, "
            f"precision={row['precision']:.3f}, recall={row['recall']:.3f}, "
            f"support={int(row['support'])}"
        )

    output_path = os.path.join(RESULTS_DIR, "baseline_result_summary.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"Saved baseline result summary to: {output_path}")


if __name__ == "__main__":
    main()