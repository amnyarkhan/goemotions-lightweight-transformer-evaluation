import os

# MacOs Safety Code to prevent CPU bottleneck when training without GPU:
'''
Important Note: If this script is run on Windows based machine and have access to GPU 
or TPU vice versa comments this piece of code as this is optimization code for my macos machine

'''
# Prevent too many background CPU threads from fighting each other.
# This is especially useful when using scikit-learn, NumPy and scipy sparse matrices.

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Avoid CPU thread oversubscription on MacBook Air.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# =================== Till here ====================================


import json
import time
import argparse
import random
import inspect
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    accuracy_score,
    precision_recall_fscore_support,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

warnings.filterwarnings("ignore")


# ============================================================
# Paths and constants
# ============================================================

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "outputs/results"
MODEL_DIR = "models/distilbert"

RANDOM_STATE = 42
MODEL_NAME = "distilbert-base-uncased"


# ============================================================
# Utility functions
# ============================================================

def ensure_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.join("outputs", "logs"), exist_ok=True)


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass


def configure_torch() -> None:
    """
    Safe PyTorch settings for MacOS based laptops with limited resources and 
    without GPU.
    """

    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass


def get_device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def save_json(data, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_metric_block(title: str, metrics: Dict[str, float]) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


def sigmoid_numpy(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid for logits.
    """

    logits = np.clip(logits, -40, 40)
    return 1.0 / (1.0 + np.exp(-logits))


# ============================================================
# Data loading
# ============================================================

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
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
            f"Expected 28 GoEmotions label columns, found {len(label_cols)}."
        )

    return train, val, test, label_cols


def select_reduced_labels(
    train: pd.DataFrame,
    label_cols: List[str],
    top_k: int = 10,
) -> List[str]:
    """
    Select top-k most frequent non-neutral labels using training data only.

    This avoids test-set leakage and matches the dissertation methodology.
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


def maybe_quick_sample(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    quick: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not quick:
        return train, val, test

    print("\nQuick mode enabled: using smaller subsets for testing only.")

    train = train.sample(n=min(2000, len(train)), random_state=RANDOM_STATE)
    val = val.sample(n=min(500, len(val)), random_state=RANDOM_STATE)
    test = test.sample(n=min(500, len(test)), random_state=RANDOM_STATE)

    return train, val, test


# ============================================================
# Dataset class
# ============================================================

class GoEmotionsDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        tokenizer,
        max_length: int = 128,
    ):
        self.texts = [str(text) for text in texts]
        self.labels = labels.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=None,
        )

        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }


# ============================================================
# Threshold tuning
# ============================================================

def tune_global_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric: str = "macro_f1",
) -> Tuple[float, pd.DataFrame]:
    """
    Tune one global threshold using validation data only.
    """

    thresholds = np.arange(0.05, 0.96, 0.05)

    best_threshold = 0.5
    best_score = -1.0
    records = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(np.int32)

        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

        selected_score = macro_f1 if metric == "macro_f1" else micro_f1

        records.append({
            "threshold": round(float(threshold), 2),
            "macro_f1": float(macro_f1),
            "micro_f1": float(micro_f1),
        })

        if selected_score > best_score:
            best_score = selected_score
            best_threshold = threshold

    return float(best_threshold), pd.DataFrame(records)


def tune_per_label_thresholds(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    label_cols: List[str],
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Tune one threshold per label using validation data only.
    """

    thresholds = np.arange(0.05, 0.96, 0.05)

    best_thresholds = {}
    records = []

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

        records.append({
            "label": label,
            "best_threshold": float(best_threshold),
            "validation_f1": float(best_f1),
            "support": int(y_true[:, i].sum()),
        })

    threshold_df = pd.DataFrame(records).sort_values(
        "validation_f1",
        ascending=False,
    )

    return best_thresholds, threshold_df


def apply_per_label_thresholds(
    y_scores: np.ndarray,
    label_cols: List[str],
    thresholds: Dict[str, float],
) -> np.ndarray:
    y_pred = np.zeros_like(y_scores, dtype=np.int32)

    for i, label in enumerate(label_cols):
        y_pred[:, i] = (y_scores[:, i] >= thresholds[label]).astype(np.int32)

    return y_pred


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_cols: List[str],
    prefix: str,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Evaluate multi-label classification using the same metrics as the baseline.
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

    save_json(summary, summary_path)
    per_label_df.to_csv(per_label_path, index=False)

    return summary, per_label_df


def compute_metrics_for_trainer(eval_pred):
    """
    Used only for validation monitoring during training.
    Final dissertation metrics are calculated separately after threshold tuning.
    """

    logits, labels = eval_pred
    y_true = labels.astype(np.int32)
    y_scores = sigmoid_numpy(logits)
    y_pred = (y_scores >= 0.5).astype(np.int32)

    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
    }


# ============================================================
# Transformers compatibility helpers
# ============================================================

def create_training_arguments(
    output_dir: str,
    label_mode: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_accumulation_steps: int,
) -> TrainingArguments:
    """
    Creates TrainingArguments using only parameters supported by the installed
    Transformers version.

    Your local signature accepts eval_strategy but not logging_dir or use_mps_device.
    """

    requested_args = {
        "output_dir": output_dir,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": epochs,
        "weight_decay": weight_decay,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 1,
        "logging_strategy": "steps",
        "logging_steps": 100,
        "report_to": "none",
        "run_name": f"distilbert_{label_mode}",
        "seed": RANDOM_STATE,
        "data_seed": RANDOM_STATE,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_grad_norm": 1.0,
        "optim": "adamw_torch",
        "warmup_steps": 0,
        "fp16": False,
        "bf16": False,
        "remove_unused_columns": True,
        "disable_tqdm": False,
    }

    accepted_params = inspect.signature(TrainingArguments.__init__).parameters

    safe_args = {
        key: value
        for key, value in requested_args.items()
        if key in accepted_params
    }

    ignored_args = sorted(set(requested_args) - set(safe_args))

    if ignored_args:
        print("\nIgnored unsupported TrainingArguments:")
        print(ignored_args)

    return TrainingArguments(**safe_args)


def create_trainer(
    model,
    training_args,
    train_dataset,
    val_dataset,
    tokenizer,
):
    """
    Trainer API changed across Transformers versions.
    This helper supports both tokenizer= and processing_class=.
    """

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "compute_metrics": compute_metrics_for_trainer,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=2)],
    }

    trainer_params = inspect.signature(Trainer.__init__).parameters

    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    return Trainer(**trainer_kwargs)


# ============================================================
# Main experiment
# ============================================================

def run_experiment(
    label_mode: str = "full",
    top_k: int = 10,
    max_length: int = 128,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 1,
    quick: bool = False,
) -> None:
    ensure_dirs()
    configure_torch()
    set_seed(RANDOM_STATE)

    device_name = get_device_name()
    print("\nUsing device:", device_name)

    train, val, test, all_label_cols = load_data()

    if label_mode == "reduced":
        label_cols = select_reduced_labels(train, all_label_cols, top_k=top_k)
    else:
        label_cols = all_label_cols

    train, val, test = maybe_quick_sample(train, val, test, quick=quick)

    if quick:
        epochs = 1

    print("\nSelected label mode:", label_mode)
    print("Number of labels:", len(label_cols))
    print("Labels:", label_cols)

    x_train = train["text"].fillna("").astype(str).tolist()
    x_val = val["text"].fillna("").astype(str).tolist()
    x_test = test["text"].fillna("").astype(str).tolist()

    y_train = train[label_cols].values.astype(np.float32)
    y_val = val[label_cols].values.astype(np.float32)
    y_test = test[label_cols].values.astype(np.int32)

    print("\nTraining examples:", len(x_train))
    print("Validation examples:", len(x_val))
    print("Test examples:", len(x_test))
    print("Max token length:", max_length)
    print("Epochs:", epochs)
    print("Batch size:", batch_size)
    print("Gradient accumulation steps:", gradient_accumulation_steps)
    print("Effective batch size:", batch_size * gradient_accumulation_steps)

    label_frequency = pd.DataFrame({
        "label": label_cols,
        "train_support": y_train.sum(axis=0).astype(int),
        "val_support": y_val.sum(axis=0).astype(int),
        "test_support": y_test.sum(axis=0).astype(int),
    }).sort_values("train_support", ascending=False)

    label_frequency.to_csv(
        os.path.join(RESULTS_DIR, f"distilbert_{label_mode}_label_frequency.csv"),
        index=False,
    )

    print("\nLoading tokenizer:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Creating datasets...")
    train_dataset = GoEmotionsDataset(
        x_train,
        y_train,
        tokenizer,
        max_length=max_length,
    )

    val_dataset = GoEmotionsDataset(
        x_val,
        y_val,
        tokenizer,
        max_length=max_length,
    )

    test_dataset = GoEmotionsDataset(
        x_test,
        y_test.astype(np.float32),
        tokenizer,
        max_length=max_length,
    )

    print("Loading model:", MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_cols),
        problem_type="multi_label_classification",
    )

    output_dir = os.path.join(MODEL_DIR, f"distilbert_{label_mode}")

    training_args = create_training_arguments(
        output_dir=output_dir,
        label_mode=label_mode,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    trainer = create_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    print("\nTraining DistilBERT...")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time

    print("\nTraining time seconds:", round(training_time, 2))
    print("Training time minutes:", round(training_time / 60, 2))

    print("\nPredicting validation logits...")
    val_output = trainer.predict(val_dataset)
    val_logits = val_output.predictions
    val_scores = sigmoid_numpy(val_logits)

    print("Tuning global threshold on validation set...")
    best_global_threshold, threshold_df = tune_global_threshold(
        y_val.astype(np.int32),
        val_scores,
        metric="macro_f1",
    )

    threshold_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            f"distilbert_{label_mode}_global_threshold_tuning.csv",
        ),
        index=False,
    )

    print("Tuning per-label thresholds on validation set...")
    per_label_thresholds, per_label_threshold_df = tune_per_label_thresholds(
        y_val.astype(np.int32),
        val_scores,
        label_cols,
    )

    save_json(
        per_label_thresholds,
        os.path.join(
            RESULTS_DIR,
            f"distilbert_{label_mode}_per_label_thresholds.json",
        ),
    )

    per_label_threshold_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            f"distilbert_{label_mode}_per_label_thresholds.csv",
        ),
        index=False,
    )

    print("Best global threshold:", round(best_global_threshold, 2))

    print("\nPredicting test logits...")
    inference_start = time.time()
    test_output = trainer.predict(test_dataset)
    inference_time = time.time() - inference_start

    test_logits = test_output.predictions
    test_scores = sigmoid_numpy(test_logits)

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
        prefix=f"distilbert_{label_mode}_test_global_threshold",
    )

    per_label_summary, per_label_metrics = evaluate_model(
        y_test,
        test_pred_per_label,
        label_cols,
        prefix=f"distilbert_{label_mode}_test_per_label_threshold",
    )

    selected_labels_path = os.path.join(
        RESULTS_DIR,
        f"distilbert_{label_mode}_selected_labels.json",
    )
    save_json(label_cols, selected_labels_path)

    experiment_info = {
        "model_type": "DistilBERT multi-label classifier",
        "base_model": MODEL_NAME,
        "label_mode": label_mode,
        "top_k": top_k if label_mode == "reduced" else None,
        "number_of_labels": len(label_cols),
        "labels": label_cols,
        "max_length": max_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "best_global_threshold": best_global_threshold,
        "training_time_seconds": training_time,
        "training_time_minutes": training_time / 60,
        "test_inference_time_seconds": inference_time,
        "test_examples": int(len(x_test)),
        "device": device_name,
        "threshold_tuning": "Validation set only",
        "random_state": RANDOM_STATE,
    }

    save_json(
        experiment_info,
        os.path.join(
            RESULTS_DIR,
            f"distilbert_{label_mode}_experiment_info.json",
        ),
    )

    print("\nSaving final DistilBERT model and tokenizer...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print_metric_block(
        "Final DistilBERT test results using global threshold",
        global_summary,
    )

    print_metric_block(
        "Final DistilBERT test results using per-label thresholds",
        per_label_summary,
    )

    print("\nBest per-label F1 scores using per-label thresholds:")
    print(per_label_metrics.head(10).to_string(index=False))

    print("\nWeakest per-label F1 scores using per-label thresholds:")
    print(per_label_metrics.tail(10).to_string(index=False))

    print("\nSaved outputs:")
    print("Model directory:", output_dir)
    print("Results directory:", RESULTS_DIR)
    print("Training time minutes:", round(training_time / 60, 2))
    print("Test inference time seconds:", round(inference_time, 2))


# ============================================================
# CLI Based options to train and fine-tune by passing the choice reduced labels set or Full
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for GoEmotions multi-label classification."
    )

    parser.add_argument(
        "--label_mode",
        choices=["full", "reduced"],
        default="full",
        help="Use full 28-label task or reduced top-k emotions plus neutral.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of non-neutral labels for reduced mode.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum token length.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training and evaluation batch size.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use small subsets for pipeline testing only.",
    )

    args = parser.parse_args()

    run_experiment(
        label_mode=args.label_mode,
        top_k=args.top_k,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()