"""
Computational-cost analysis for the GoEmotions dissertation experiments.

Place this file in:
    src/evaluation/analyse_computational_cost.py

Run from the project root:
    python src/evaluation/analyse_computational_cost.py

The script reads existing experiment metadata and evaluation metrics, estimates saved
model sizes, and produces CSV/PNG outputs for the dissertation computational-cost section.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path.cwd()
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = [
    {
        "model": "TF-IDF Logistic Regression",
        "label_setting": "Full 28-label",
        "prefix": "baseline_full",
        "model_paths": [
            MODELS_DIR / "baseline" / "baseline_full_tfidf_logreg.joblib",
            MODELS_DIR / "baseline" / "baseline_full_tfidf_logreg.pkl",
            MODELS_DIR / "baseline" / "baseline_full_model.joblib",
        ],
    },
    {
        "model": "DistilBERT",
        "label_setting": "Full 28-label",
        "prefix": "distilbert_full",
        "model_paths": [
            MODELS_DIR / "distilbert" / "distilbert_full",
            MODELS_DIR / "distilbert" / "full",
        ],
    },
    {
        "model": "TF-IDF Logistic Regression",
        "label_setting": "Reduced 11-label",
        "prefix": "baseline_reduced",
        "model_paths": [
            MODELS_DIR / "baseline" / "baseline_reduced_tfidf_logreg.joblib",
            MODELS_DIR / "baseline" / "baseline_reduced_tfidf_logreg.pkl",
            MODELS_DIR / "baseline" / "baseline_reduced_model.joblib",
        ],
    },
    {
        "model": "DistilBERT",
        "label_setting": "Reduced 11-label",
        "prefix": "distilbert_reduced",
        "model_paths": [
            MODELS_DIR / "distilbert" / "distilbert_reduced",
            MODELS_DIR / "distilbert" / "reduced",
        ],
    },
]


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Warning: could not read {path}: {exc}")
        return {}


def first_existing(paths: list[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def get_first_number(d: Dict[str, Any], keys: list[str]) -> Optional[float]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    # also search one nested level in case metadata was nested under timing/hardware
    for nested in d.values():
        if isinstance(nested, dict):
            for k in keys:
                v = nested.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    return float(v)
    return None


def get_first_text(d: Dict[str, Any], keys: list[str]) -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    for nested in d.values():
        if isinstance(nested, dict):
            for k in keys:
                v = nested.get(k)
                if isinstance(v, str) and v.strip():
                    return v
    return "not recorded"


def seconds_to_minutes(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return x / 60.0


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


def build_summary() -> pd.DataFrame:
    rows = []
    for exp in EXPERIMENTS:
        prefix = exp["prefix"]
        info = load_json(RESULTS_DIR / f"{prefix}_experiment_info.json")
        metrics = load_json(RESULTS_DIR / f"{prefix}_test_per_label_threshold_summary_metrics.json")

        train_seconds = get_first_number(
            info,
            [
                "training_time_seconds",
                "train_time_seconds",
                "fit_time_seconds",
                "training_seconds",
                "total_training_time_seconds",
                "runtime_train_seconds",
            ],
        )
        inference_seconds = get_first_number(
            info,
            [
                "test_prediction_time_seconds",
                "test_inference_time_seconds",
                "prediction_time_seconds",
                "inference_time_seconds",
                "test_runtime_seconds",
            ],
        )
        n_test = get_first_number(info, ["n_test", "test_size", "num_test_examples", "test_examples"])
        if n_test is None:
            # GoEmotions official test split used in this dissertation.
            n_test = 5427.0
        inference_ms_per_comment = None
        if inference_seconds is not None and n_test:
            inference_ms_per_comment = (inference_seconds / n_test) * 1000

        model_path = first_existing(exp["model_paths"])
        model_size_mb = directory_size_bytes(model_path) / (1024 ** 2) if model_path else None

        rows.append(
            {
                "model": exp["model"],
                "label_setting": exp["label_setting"],
                "prefix": prefix,
                "training_time_seconds": train_seconds,
                "training_time_minutes": seconds_to_minutes(train_seconds),
                "test_inference_time_seconds": inference_seconds,
                "inference_ms_per_comment": inference_ms_per_comment,
                "model_size_mb": model_size_mb,
                "device_or_hardware": get_first_text(info, ["device", "device_name", "hardware", "accelerator"]),
                "macro_f1": safe_float(metrics.get("macro_f1")),
                "micro_f1": safe_float(metrics.get("micro_f1")),
                "exact_match_accuracy": safe_float(metrics.get("exact_match_accuracy")),
                "hamming_loss": safe_float(metrics.get("hamming_loss")),
                "metrics_file_found": (RESULTS_DIR / f"{prefix}_test_per_label_threshold_summary_metrics.json").exists(),
                "experiment_info_found": (RESULTS_DIR / f"{prefix}_experiment_info.json").exists(),
                "model_path_found": str(model_path) if model_path else "not found",
            }
        )
    return pd.DataFrame(rows)


def bar_plot(df: pd.DataFrame, y: str, ylabel: str, title: str, output: Path) -> None:
    plot_df = df.dropna(subset=[y]).copy()
    if plot_df.empty:
        print(f"Skipping {output.name}: no data for {y}")
        return
    labels = plot_df["model"] + "\n" + plot_df["label_setting"]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, plot_df[y])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Saved {output}")


def scatter_tradeoff(df: pd.DataFrame, output: Path) -> None:
    plot_df = df.dropna(subset=["training_time_minutes", "macro_f1"]).copy()
    if plot_df.empty:
        print(f"Skipping {output.name}: no training-time data available")
        return
    plt.figure(figsize=(8, 5))
    plt.scatter(plot_df["training_time_minutes"], plot_df["macro_f1"])
    for _, row in plot_df.iterrows():
        label = f"{row['model']}\n{row['label_setting']}"
        plt.annotate(label, (row["training_time_minutes"], row["macro_f1"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    plt.xlabel("Training time (minutes)")
    plt.ylabel("Macro F1")
    plt.title("Predictive performance versus computational cost")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Saved {output}")


def main() -> None:
    df = build_summary()
    out_csv = RESULTS_DIR / "computational_cost_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    print(df.to_string(index=False))

    bar_plot(
        df,
        y="training_time_minutes",
        ylabel="Training time (minutes)",
        title="Training time by model and label setting",
        output=FIGURES_DIR / "computational_cost_training_time.png",
    )
    bar_plot(
        df,
        y="model_size_mb",
        ylabel="Saved model size (MB)",
        title="Saved model size by model and label setting",
        output=FIGURES_DIR / "computational_cost_model_size.png",
    )
    bar_plot(
        df,
        y="inference_ms_per_comment",
        ylabel="Inference time per comment (ms)",
        title="Test-set inference time per comment",
        output=FIGURES_DIR / "computational_cost_inference_time.png",
    )
    scatter_tradeoff(df, FIGURES_DIR / "computational_cost_tradeoff_macro_f1_vs_training_time.png")

    missing_time = df["training_time_minutes"].isna().any()
    if missing_time:
        print("\nNote: Some training-time values were missing. Check each *_experiment_info.json file.")
        


if __name__ == "__main__":
    main()
