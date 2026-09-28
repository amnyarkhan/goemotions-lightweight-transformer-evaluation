import os

# MacOs Safety Code to prevent CPU bottleneck when training without GPU:
'''
Important Note: If this script is run on Windows based machine and have access to GPU 
or TPU vice versa comments this piece of code as this is optimization code for my macos machine

'''
# Prevent too many background CPU threads from fighting each other.
# This is especially useful when using scikit-learn, NumPy and scipy sparse matrices.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# =================== Till here ====================================

import json
import time
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    accuracy_score,
    precision_recall_fscore_support,
)

warnings.filterwarnings("ignore")


PROCESSED_DIR = "data/processed"
RESULTS_DIR = "outputs/results"
MODEL_DIR = "models/baseline"

RANDOM_STATE = 42


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    val_path = os.path.join(PROCESSED_DIR, "val.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")

    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing processed data file: {path}. "
                "Run python src/data/prepare_data.py first."
            )

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    label_cols = [
        col for col in train.columns
        if col not in ["comment_id", "text", "split"]
    ]

    if len(label_cols) != 28:
        raise ValueError(
            f"Expected 28 label columns for GoEmotions, found {len(label_cols)}."
        )

    return train, val, test, label_cols


def select_reduced_labels(train, label_cols, top_k=10):
    """
    Select top-k most frequent non-neutral emotion labels using training data only.
    Neutral is always retained but is not counted inside top_k.

    This avoids test-set leakage and follows the dissertation methodology.
    """

    label_counts = train[label_cols].sum().sort_values(ascending=False)

    non_neutral_labels = [
        label for label in label_counts.index
        if label != "neutral"
    ]

    selected = list(non_neutral_labels[:top_k])

    if "neutral" in label_cols:
        selected.append("neutral")

    return selected


def get_xy(df, label_cols):
    x = df["text"].fillna("").astype(str).values
    y = df[label_cols].values.astype(np.int32)
    return x, y


def build_model(
    max_features=30000,
    ngram_max=2,
    c_value=2.0,
    class_weight=None,
):
    """
    Build TF-IDF + One-vs-Rest Logistic Regression baseline.

    MacBook Air M2 optimisation choices:
    - dtype=np.float32 reduces memory usage.
    - n_jobs=1 avoids joblib/loky read-only sparse matrix errors.
    - liblinear is stable for one-vs-rest binary Logistic Regression.
    """

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, ngram_max),
        max_features=max_features,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        dtype=np.float32,
    )

    base_classifier = LogisticRegression(
        C=c_value,
        solver="liblinear",
        max_iter=1000,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
    )

    # Important Mac fix:
    # n_jobs=-1 can create joblib memory-mapping problems with scipy sparse matrices.
    model = OneVsRestClassifier(base_classifier, n_jobs=1)

    return vectorizer, model


def make_sparse_safe(matrix):
    """
    Convert TF-IDF output into a writable CSR matrix.

    This prevents scipy/joblib read-only sparse matrix issues on macOS.
    """

    matrix = matrix.tocsr().copy()
    matrix.sort_indices()
    return matrix


def tune_global_threshold(y_true, y_scores, metric="macro_f1"):
    """
    Tune one global threshold using validation data only.
    """

    thresholds = np.arange(0.05, 0.96, 0.05)

    best_threshold = 0.5
    best_score = -1.0
    results = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(np.int32)

        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

        selected_score = macro_f1 if metric == "macro_f1" else micro_f1

        results.append({
            "threshold": round(float(threshold), 2),
            "macro_f1": float(macro_f1),
            "micro_f1": float(micro_f1),
        })

        if selected_score > best_score:
            best_score = selected_score
            best_threshold = threshold

    return float(best_threshold), pd.DataFrame(results)


def tune_per_label_thresholds(y_true, y_scores, label_cols):
    """
    Tune one threshold per label using validation data only.

    This is important for imbalanced multi-label classification because rare labels
    may require different probability cut-offs from frequent labels.
    """

    thresholds = np.arange(0.05, 0.96, 0.05)
    best_thresholds = {}
    threshold_records = []

    for i, label in enumerate(label_cols):
        best_threshold = 0.5
        best_f1 = -1.0

        for threshold in thresholds:
            y_pred_label = (y_scores[:, i] >= threshold).astype(np.int32)
            current_f1 = f1_score(
                y_true[:, i],
                y_pred_label,
                zero_division=0,
            )

            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold

        best_thresholds[label] = float(best_threshold)

        threshold_records.append({
            "label": label,
            "best_threshold": float(best_threshold),
            "validation_f1": float(best_f1),
            "support": int(y_true[:, i].sum()),
        })

    threshold_df = pd.DataFrame(threshold_records).sort_values(
        "validation_f1",
        ascending=False,
    )

    return best_thresholds, threshold_df


def apply_per_label_thresholds(y_scores, label_cols, thresholds):
    y_pred = np.zeros_like(y_scores, dtype=np.int32)

    for i, label in enumerate(label_cols):
        threshold = thresholds[label]
        y_pred[:, i] = (y_scores[:, i] >= threshold).astype(np.int32)

    return y_pred


def evaluate_model(y_true, y_pred, label_cols, prefix):
    """
    Evaluate multi-label classification performance.
    """

    summary = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "exact_match_accuracy": float(accuracy_score(y_true, y_pred)),
    }

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    per_label_df = pd.DataFrame({
        "label": label_cols,
        "support": support.astype(int),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }).sort_values("f1", ascending=False)

    summary_path = os.path.join(RESULTS_DIR, f"{prefix}_summary_metrics.json")
    per_label_path = os.path.join(RESULTS_DIR, f"{prefix}_per_label_metrics.csv")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    per_label_df.to_csv(per_label_path, index=False)

    return summary, per_label_df


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def print_metric_block(title, metrics):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


def run_experiment(label_mode="full", top_k=10, quick=False):
    ensure_dirs()

    train, val, test, all_label_cols = load_data()

    if label_mode == "reduced":
        label_cols = select_reduced_labels(train, all_label_cols, top_k=top_k)
    else:
        label_cols = all_label_cols

    print("\nSelected label mode:", label_mode)
    print("Number of labels:", len(label_cols))
    print("Labels:", label_cols)

    x_train, y_train = get_xy(train, label_cols)
    x_val, y_val = get_xy(val, label_cols)
    x_test, y_test = get_xy(test, label_cols)

    print("\nTraining examples:", len(x_train))
    print("Validation examples:", len(x_val))
    print("Test examples:", len(x_test))

    label_frequency = pd.DataFrame({
        "label": label_cols,
        "train_support": y_train.sum(axis=0).astype(int),
        "val_support": y_val.sum(axis=0).astype(int),
        "test_support": y_test.sum(axis=0).astype(int),
    }).sort_values("train_support", ascending=False)

    label_frequency_path = os.path.join(
        RESULTS_DIR,
        f"baseline_{label_mode}_label_frequency.csv",
    )
    label_frequency.to_csv(label_frequency_path, index=False)

    # Smaller and safer config list for MacBook Air M2.
    # It still gives a fair baseline because it tests unigram, unigram+bigram,
    # and class-weighted Logistic Regression.
    if quick:
        configs = [
            {
                "name": "tfidf_lr_unigram_bigram_30k",
                "max_features": 30000,
                "ngram_max": 2,
                "c_value": 2.0,
                "class_weight": None,
            }
        ]
    else:
        configs = [
            {
                "name": "tfidf_lr_unigram_30k",
                "max_features": 30000,
                "ngram_max": 1,
                "c_value": 1.0,
                "class_weight": None,
            },
            {
                "name": "tfidf_lr_unigram_bigram_40k",
                "max_features": 40000,
                "ngram_max": 2,
                "c_value": 2.0,
                "class_weight": None,
            },
            {
                "name": "tfidf_lr_unigram_bigram_40k_balanced",
                "max_features": 40000,
                "ngram_max": 2,
                "c_value": 2.0,
                "class_weight": "balanced",
            },
        ]

    best_config = None
    best_vectorizer = None
    best_model = None
    best_val_scores = None
    best_val_macro_f1 = -1.0

    tuning_records = []

    for config in configs:
        print("\n" + "-" * 70)
        print("Training config:", config["name"])
        print("-" * 70)

        start_time = time.time()

        vectorizer, model = build_model(
            max_features=config["max_features"],
            ngram_max=config["ngram_max"],
            c_value=config["c_value"],
            class_weight=config["class_weight"],
        )

        print("Vectorising training data...")
        x_train_tfidf = make_sparse_safe(vectorizer.fit_transform(x_train))

        print("Vectorising validation data...")
        x_val_tfidf = make_sparse_safe(vectorizer.transform(x_val))

        print("TF-IDF train matrix shape:", x_train_tfidf.shape)
        print("TF-IDF validation matrix shape:", x_val_tfidf.shape)

        print("Training One-vs-Rest Logistic Regression...")
        model.fit(x_train_tfidf, y_train)

        print("Predicting validation probabilities...")
        val_scores = model.predict_proba(x_val_tfidf)

        val_pred_default = (val_scores >= 0.5).astype(np.int32)

        val_macro_f1 = f1_score(
            y_val,
            val_pred_default,
            average="macro",
            zero_division=0,
        )

        val_micro_f1 = f1_score(
            y_val,
            val_pred_default,
            average="micro",
            zero_division=0,
        )

        training_time = time.time() - start_time

        record = {
            "config": config["name"],
            "val_macro_f1_threshold_0_5": float(val_macro_f1),
            "val_micro_f1_threshold_0_5": float(val_micro_f1),
            "training_time_seconds": float(training_time),
            "max_features": config["max_features"],
            "ngram_max": config["ngram_max"],
            "c_value": config["c_value"],
            "class_weight": str(config["class_weight"]),
        }

        tuning_records.append(record)

        print("Validation macro F1 @ 0.5:", round(val_macro_f1, 4))
        print("Validation micro F1 @ 0.5:", round(val_micro_f1, 4))
        print("Training time seconds:", round(training_time, 2))

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_config = config
            best_vectorizer = vectorizer
            best_model = model
            best_val_scores = val_scores

        # Free memory before next config
        del x_train_tfidf
        del x_val_tfidf

    tuning_df = pd.DataFrame(tuning_records)
    tuning_path = os.path.join(
        RESULTS_DIR,
        f"baseline_{label_mode}_model_tuning_results.csv",
    )
    tuning_df.to_csv(tuning_path, index=False)

    print("\n" + "=" * 70)
    print("Best validation configuration")
    print("=" * 70)
    print("Best config:", best_config["name"])
    print("Best validation macro F1 @ 0.5:", round(best_val_macro_f1, 4))

    # Threshold tuning on validation set only
    print("\nTuning global threshold on validation set...")
    best_global_threshold, threshold_df = tune_global_threshold(
        y_val,
        best_val_scores,
        metric="macro_f1",
    )

    threshold_path = os.path.join(
        RESULTS_DIR,
        f"baseline_{label_mode}_global_threshold_tuning.csv",
    )
    threshold_df.to_csv(threshold_path, index=False)

    print("Tuning per-label thresholds on validation set...")
    per_label_thresholds, per_label_threshold_df = tune_per_label_thresholds(
        y_val,
        best_val_scores,
        label_cols,
    )

    per_label_threshold_json_path = os.path.join(
        RESULTS_DIR,
        f"baseline_{label_mode}_per_label_thresholds.json",
    )
    per_label_threshold_csv_path = os.path.join(
        RESULTS_DIR,
        f"baseline_{label_mode}_per_label_thresholds.csv",
    )

    save_json(per_label_thresholds, per_label_threshold_json_path)
    per_label_threshold_df.to_csv(per_label_threshold_csv_path, index=False)

    print("Best global threshold:", round(best_global_threshold, 2))

    # Final test evaluation
    print("\nVectorising test data...")
    x_test_tfidf = make_sparse_safe(best_vectorizer.transform(x_test))

    print("Predicting test probabilities...")
    test_scores = best_model.predict_proba(x_test_tfidf)

    test_pred_global = (test_scores >= best_global_threshold).astype(np.int32)

    test_pred_per_label = apply_per_label_thresholds(
        test_scores,
        label_cols,
        per_label_thresholds,
    )

    global_summary, global_per_label = evaluate_model(
        y_test,
        test_pred_global,
        label_cols,
        prefix=f"baseline_{label_mode}_test_global_threshold",
    )

    per_label_summary, per_label_metrics = evaluate_model(
        y_test,
        test_pred_per_label,
        label_cols,
        prefix=f"baseline_{label_mode}_test_per_label_threshold",
    )

    selected_labels_path = os.path.join(
        RESULTS_DIR,
        f"baseline_{label_mode}_selected_labels.json",
    )

    save_json(label_cols, selected_labels_path)

    experiment_info = {
        "label_mode": label_mode,
        "top_k": top_k if label_mode == "reduced" else None,
        "number_of_labels": len(label_cols),
        "labels": label_cols,
        "best_config": best_config,
        "best_global_threshold": best_global_threshold,
        "random_state": RANDOM_STATE,
        "train_examples": int(len(x_train)),
        "validation_examples": int(len(x_val)),
        "test_examples": int(len(x_test)),
        "model_type": "TF-IDF + One-vs-Rest Logistic Regression",
        "threshold_tuning": "Validation set only",
    }

    experiment_info_path = os.path.join(
        RESULTS_DIR,
        f"baseline_{label_mode}_experiment_info.json",
    )
    save_json(experiment_info, experiment_info_path)

    model_path = os.path.join(
        MODEL_DIR,
        f"baseline_{label_mode}_tfidf_logreg.joblib",
    )

    print("\nSaving trained baseline model...")
    joblib.dump(
        {
            "vectorizer": best_vectorizer,
            "model": best_model,
            "label_cols": label_cols,
            "best_config": best_config,
            "best_global_threshold": best_global_threshold,
            "per_label_thresholds": per_label_thresholds,
        },
        model_path,
        compress=3,
    )

    print_metric_block(
        "Final test results using global threshold",
        global_summary,
    )

    print_metric_block(
        "Final test results using per-label thresholds",
        per_label_summary,
    )

    print("\nBest per-label F1 scores using per-label thresholds:")
    print(per_label_metrics.head(10).to_string(index=False))

    print("\nWeakest per-label F1 scores using per-label thresholds:")
    print(per_label_metrics.tail(10).to_string(index=False))

    print("\nSaved outputs:")
    print("Model:", model_path)
    print("Tuning results:", tuning_path)
    print("Global threshold tuning:", threshold_path)
    print("Per-label thresholds:", per_label_threshold_csv_path)
    print("Global test metrics:", f"baseline_{label_mode}_test_global_threshold_summary_metrics.json")
    print("Per-label test metrics:", f"baseline_{label_mode}_test_per_label_threshold_summary_metrics.json")


def main():
    parser = argparse.ArgumentParser(
        description="Train TF-IDF + One-vs-Rest Logistic Regression baseline for GoEmotions."
    )

    parser.add_argument(
        "--label_mode",
        choices=["full", "reduced"],
        default="full",
        help="Use all 28 labels or reduced top-k emotion labels plus neutral.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of non-neutral frequent emotion labels for reduced mode.",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only one smaller baseline configuration for faster testing.",
    )

    args = parser.parse_args()

    run_experiment(
        label_mode=args.label_mode,
        top_k=args.top_k,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()