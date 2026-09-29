# GoEmotions Dissertation Implementation README

## Project title

**Evaluating Lightweight Transformer Models for Fine-Grained Multi-Label Emotion Classification under Label Imbalance**

This repository contains the practical implementation used for the MSc Artificial Intelligence dissertation. The project compares a classical **TF-IDF + One-vs-Rest Logistic Regression** baseline with **DistilBERT** for fine-grained multi-label emotion classification using the **GoEmotions** dataset.

The implementation is designed to be reproducible from raw data preparation through exploratory data analysis, model training, evaluation, statistical comparison, computational-cost analysis and appendix export.

---

## 1. Project aim

The main aim is to test whether a lightweight transformer model, **DistilBERT**, provides a practically meaningful improvement over a traditional **TF-IDF-based linear classifier** for fine-grained multi-label emotion classification under label imbalance.

The implementation supports:

- official GoEmotions train, validation and test splits;
- full 28-label classification;
- reduced 11-label classification;
- TF-IDF + One-vs-Rest Logistic Regression baseline;
- DistilBERT multi-label fine-tuning;
- validation-only threshold tuning;
- final test-set evaluation;
- per-label analysis;
- statistical comparison;
- computational-cost analysis.

---

## 2. Recommended project location

The original implementation was developed using the following project root:

```bash
~/Desktop/goemotions-lightweight-transformer-evaluation
```

If the full project folder is provided directly, place it on the Desktop or another preferred location and enter the folder:

```bash
cd ~/Desktop/goemotions-lightweight-transformer-evaluation
```

If using the GitHub repository instead, clone the repository and enter it:

```bash
git clone https://github.com/amnyarkhan/goemotions-lightweight-transformer-evaluation.git
cd goemotions-lightweight-transformer-evaluation
```

All scripts should be run from this project root unless stated otherwise. The repository/folder should already contain the source code, `requirements.txt`, and the expected directory structure.

---

## 3. Project directory structure

The uploaded project folder or GitHub repository should already follow the structure below:

```text
goemotions-lightweight-transformer-evaluation/
│
├── data/
│   ├── raw/
│   │   └── splits/
│   │       ├── train.tsv
│   │       ├── dev.tsv
│   │       ├── test.tsv
│   │       └── emotions.txt
│   │
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── all_splits.csv
│
├── src/
│   ├── data/
│   │   └── prepare_goemotions.py
│   │
│   ├── eda/
│   │   └── run_eda.py
│   │
│   ├── models/
│   │   ├── train_baseline.py
│   │   └── train_distilbert.py
│   │
│   └── evaluation/
│       ├── visualize_baseline_results.py
│       ├── compare_reduced_models.py
│       ├── compare_full_models.py
│       └── analyse_computational_cost.py
│
├── models/
│   ├── baseline/
│   └── distilbert/
│
├── outputs/
│   ├── figures/
│   ├── results/
│   └── appendix_exports/
│
├── requirements.txt
└── README.md
```

If any output folders are missing after cloning or extracting the project, they can be recreated with:

```bash
mkdir -p data/raw/splits
mkdir -p data/processed
mkdir -p models/baseline
mkdir -p models/distilbert
mkdir -p outputs/figures
mkdir -p outputs/results
mkdir -p outputs/appendix_exports
```

The `src/` folder and `requirements.txt` should already be present in the submitted folder or GitHub repository.

---

## 4. Python environment setup

The project already includes a `requirements.txt` file. The supervisor or examiner does not need to manually install packages one by one. Only create a virtual environment and install the packages from `requirements.txt`.

Create a virtual environment from the project root:

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install all required packages from the existing `requirements.txt` file:

```bash
pip install -r requirements.txt
```

After installation, the environment is ready to run the full pipeline. There is no need to regenerate `requirements.txt` unless the environment is deliberately changed.

Optional verification:

```bash
python -c "import pandas, sklearn, torch, transformers; print('Environment ready')"
```

---

## 5. Raw data placement

Place the official GoEmotions split files inside:

```text
data/raw/splits/
```

The folder must contain:

```text
train.tsv
dev.tsv
test.tsv
emotions.txt
```

The expected raw-data structure is:

```text
data/raw/splits/train.tsv
data/raw/splits/dev.tsv
data/raw/splits/test.tsv
data/raw/splits/emotions.txt
```

The `emotions.txt` file must contain the ordered emotion labels used by the GoEmotions split files.

### How the raw GoEmotions files were downloaded

In the original implementation, the official GoEmotions split files were downloaded directly from the Google Research GitHub repository using `curl`. If the raw files are not already included in the submitted folder, they can be downloaded with the following commands from the project root:

```bash
mkdir -p data/raw/splits

curl -L -o data/raw/splits/train.tsv \
  https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/train.tsv

curl -L -o data/raw/splits/dev.tsv \
  https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/dev.tsv

curl -L -o data/raw/splits/test.tsv \
  https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/test.tsv

curl -L -o data/raw/splits/emotions.txt \
  https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/emotions.txt
```

After downloading, verify that the four required files exist:

```bash
ls data/raw/splits
```

Expected output:

```text
dev.tsv
emotions.txt
test.tsv
train.tsv
```

---

## 6. Step-by-step pipeline execution

Run the scripts in the following order.

---

### Step 1: Prepare the GoEmotions data

Script:

```text
src/data/prepare_goemotions.py
```

Run:

```bash
python src/data/prepare_goemotions.py
```

This script should:

- read `train.tsv`, `dev.tsv`, `test.tsv`;
- read `emotions.txt`;
- convert comma-separated label IDs into binary multi-hot columns;
- save processed CSV files.

Expected outputs:

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
data/processed/all_splits.csv
```

Expected shapes from the dissertation implementation:

```text
train.csv       43,410 rows
val.csv          5,426 rows
test.csv         5,427 rows
all_splits.csv  54,263 rows
```

---

### Step 2: Run exploratory data analysis

Script:

```text
src/eda/run_eda.py
```

Run:

```bash
python src/eda/run_eda.py
```

This script generates exploratory summaries and figures for:

- label frequency distribution;
- number of labels per comment;
- comment length distribution;
- label co-occurrence matrix.

Expected outputs:

```text
outputs/results/label_frequencies.csv
outputs/results/labels_per_comment_summary.csv
outputs/results/text_length_summary.csv
outputs/results/label_cooccurrence_matrix.csv

outputs/figures/label_frequency_distribution.png
outputs/figures/labels_per_comment.png
outputs/figures/comment_length_distribution.png
outputs/figures/label_cooccurrence_matrix.png
```

These outputs support the dissertation discussion of label imbalance, multi-label structure, semantic overlap and tokenisation choices.

---

### Step 3: Train the TF-IDF baseline

Script:

```text
src/models/train_baseline.py
```

Run the full 28-label baseline:

```bash
python src/models/train_baseline.py --label_mode full
```

Run the reduced 11-label baseline:

```bash
python src/models/train_baseline.py --label_mode reduced
```

The baseline model uses:

- TF-IDF text features;
- One-vs-Rest Logistic Regression;
- validation-based model selection;
- validation-based threshold tuning;
- test-only final evaluation.

Main baseline settings used in the dissertation included:

```text
ngram options: unigram and unigram-bigram
max features: 30,000 and 40,000
min_df: 2
max_df: 0.95
sublinear_tf: True
Logistic Regression C: 2.0
solver: liblinear
max_iter: 1000
```

Expected output examples:

```text
models/baseline/baseline_full_tfidf_logreg.joblib
models/baseline/baseline_reduced_tfidf_logreg.joblib

outputs/results/baseline_full_test_per_label_threshold_summary_metrics.json
outputs/results/baseline_reduced_test_per_label_threshold_summary_metrics.json

outputs/results/baseline_full_test_per_label_threshold_per_label_metrics.csv
outputs/results/baseline_reduced_test_per_label_threshold_per_label_metrics.csv
```

---

### Step 4: Visualise baseline results

Script:

```text
src/evaluation/visualize_baseline_results.py
```

Run:

```bash
python src/evaluation/visualize_baseline_results.py
```

Expected outputs include:

```text
outputs/results/baseline_full_vs_reduced_summary_table.csv

outputs/figures/baseline_full_vs_reduced_metric_comparison.png
outputs/figures/baseline_reduced_minus_full_metric_change.png
outputs/figures/baseline_full_per_label_f1.png
outputs/figures/baseline_reduced_per_label_f1.png
outputs/figures/baseline_full_support_vs_f1.png
outputs/figures/baseline_reduced_support_vs_f1.png
outputs/figures/baseline_full_threshold_tuning_curve.png
outputs/figures/baseline_reduced_threshold_tuning_curve.png
```

---

### Step 5: Fine-tune DistilBERT

Script:

```text
src/models/train_distilbert.py
```

Run the reduced 11-label DistilBERT experiment first:

```bash
python src/models/train_distilbert.py --label_mode reduced
```

Run the full 28-label DistilBERT experiment:

```bash
python src/models/train_distilbert.py --label_mode full
```

The dissertation implementation used:

```text
checkpoint: distilbert-base-uncased
task type: multi-label classification
loss: binary cross-entropy with logits
activation: sigmoid
maximum sequence length: 128
epochs: 3
batch size: 8
learning rate: 2e-5
weight decay: 0.01
gradient accumulation: 1
```

Expected outputs:

```text
models/distilbert/distilbert_reduced/
models/distilbert/distilbert_full/

outputs/results/distilbert_reduced_test_per_label_threshold_summary_metrics.json
outputs/results/distilbert_full_test_per_label_threshold_summary_metrics.json

outputs/results/distilbert_reduced_test_per_label_threshold_per_label_metrics.csv
outputs/results/distilbert_full_test_per_label_threshold_per_label_metrics.csv
```

Note: DistilBERT training is much slower than the TF-IDF baseline. In the dissertation implementation, DistilBERT required approximately two hours per experiment on the reported hardware.

---

### Step 6: Compare reduced-label TF-IDF and DistilBERT

Script:

```text
src/evaluation/compare_reduced_models.py
```

Run:

```bash
python src/evaluation/compare_reduced_models.py
```

Expected outputs:

```text
outputs/results/reduced_baseline_vs_distilbert_summary_comparison.csv
outputs/results/reduced_per_label_baseline_vs_distilbert_comparison.csv
outputs/results/reduced_model_label_level_statistical_tests.csv
outputs/results/reduced_model_bootstrap_mean_difference_ci.csv
outputs/results/reduced_support_metric_correlations.csv

outputs/figures/reduced_baseline_vs_distilbert_metric_comparison.png
outputs/figures/reduced_distilbert_minus_baseline_metric_delta.png
outputs/figures/reduced_per_label_f1_baseline_vs_distilbert.png
outputs/figures/reduced_per_label_f1_difference_distilbert_minus_baseline.png
outputs/figures/reduced_support_vs_f1_baseline_vs_distilbert.png
outputs/figures/reduced_f1_gain_vs_support.png
outputs/figures/reduced_precision_recall_space_baseline_vs_distilbert.png
outputs/figures/reduced_threshold_tuning_baseline_vs_distilbert.png
outputs/figures/reduced_per_label_thresholds_baseline_vs_distilbert.png
outputs/figures/reduced_label_metric_gain_heatmap.png
```

---

### Step 7: Compare full-label TF-IDF and DistilBERT

Script:

```text
src/evaluation/compare_full_models.py
```

Run:

```bash
python src/evaluation/compare_full_models.py
```

Expected outputs:

```text
outputs/results/full_baseline_vs_distilbert_summary_comparison.csv
outputs/results/full_per_label_baseline_vs_distilbert_comparison.csv
outputs/results/full_model_label_level_statistical_tests.csv
outputs/results/full_model_bootstrap_mean_difference_ci.csv
outputs/results/full_support_metric_correlations.csv

outputs/figures/full_baseline_vs_distilbert_metric_comparison.png
outputs/figures/full_distilbert_minus_baseline_metric_delta.png
outputs/figures/full_per_label_f1_baseline_vs_distilbert.png
outputs/figures/full_per_label_f1_difference_distilbert_minus_baseline.png
outputs/figures/full_support_vs_f1_baseline_vs_distilbert.png
outputs/figures/full_f1_gain_vs_support.png
outputs/figures/full_precision_recall_space_baseline_vs_distilbert.png
outputs/figures/full_threshold_tuning_baseline_vs_distilbert.png
outputs/figures/full_per_label_thresholds_baseline_vs_distilbert.png
outputs/figures/full_label_metric_gain_heatmap.png
```

---

### Step 8: Analyse computational cost

Script:

```text
src/evaluation/analyse_computational_cost.py
```

Run:

```bash
python src/evaluation/analyse_computational_cost.py
```

Expected outputs:

```text
outputs/results/computational_cost_summary.csv

outputs/figures/computational_cost_training_time.png
outputs/figures/computational_cost_model_size.png
outputs/figures/computational_cost_inference_time.png
outputs/figures/computational_cost_tradeoff_macro_f1_vs_training_time.png
```

The dissertation reports the following main computational-cost values:

```text
Full TF-IDF:
runtime = 12.54 seconds
model size = 8.59 MB

Full DistilBERT:
training time = 148.48 minutes
model size = 1023.48 MB

Reduced TF-IDF:
runtime = 7.52 seconds
model size = 3.63 MB

Reduced DistilBERT:
training time = 127.50 minutes
model size = 1023.28 MB
```

---

### Step 9: Export appendix result tables

Script:

```text
extract_appendix_results.py
```

Run:

```bash
python extract_appendix_results.py
```

Expected outputs:

```text
outputs/appendix_exports/appendix_results_raw_output.txt
outputs/appendix_exports/appendix_results_tables.xlsx
```

These files collect statistical tests, bootstrap confidence intervals, per-label comparison tables, support correlations and computational-cost summaries for the dissertation appendices.

---

## 7. Main final results to verify

The reconstructed implementation should reproduce the following headline results.

### Full 28-label task

```text
TF-IDF macro F1:       0.4458
DistilBERT macro F1:  0.5182

TF-IDF exact-match:       0.2937
DistilBERT exact-match:  0.4074
```

### Reduced 11-label task

```text
TF-IDF macro F1:       0.5457
DistilBERT macro F1:  0.6060

TF-IDF exact-match:       0.3947
DistilBERT exact-match:  0.4964
```

### Main interpretation

DistilBERT improves predictive performance in both the full-label and reduced-label settings. However, TF-IDF remains much faster, smaller and more interpretable. Therefore, the final recommendation is conditional: use DistilBERT when predictive reliability is the priority, and use TF-IDF when speed, transparency, compactness and easy deployment are more important.

---

## 8. Troubleshooting notes

### File not found errors

Make sure the raw data files are inside:

```text
data/raw/splits/
```

and that commands are run from the project root:

```bash
cd ~/Desktop/goemotions-dissertation
```

### Missing package errors

All required packages should be installed from `requirements.txt`. If a package error appears, first make sure the virtual environment is active and reinstall the requirements:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

For example, if the export script reports `ModuleNotFoundError: No module named 'openpyxl'`, reinstalling from `requirements.txt` should resolve it because `openpyxl` is included in the project dependencies.

### Slow DistilBERT training

DistilBERT training is expected to be much slower than TF-IDF. On Apple Silicon M1-M2 chips, PyTorch may use `mps` if available. If MPS causes issues, run on CPU or use a CUDA GPU environment.

Useful environment variables for macOS:

```bash
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Reproducibility

For best reproducibility:

- keep the official train, validation and test splits unchanged;
- tune thresholds only on the validation set;
- evaluate final models only once on the test set;
- select reduced labels using only training-set frequencies;
- save all outputs under `outputs/results/` and `outputs/figures/`.

---

## 9. Recommended execution order summary

```bash
# 1. Create and activate environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2. Prepare data
python src/data/prepare_goemotions.py

# 3. Run EDA
python src/eda/run_eda.py

# 4. Train TF-IDF baselines
python src/models/train_baseline.py --label_mode full
python src/models/train_baseline.py --label_mode reduced

# 5. Visualise baseline results
python src/evaluation/visualize_baseline_results.py

# 6. Train DistilBERT models
python src/models/train_distilbert.py --label_mode reduced
python src/models/train_distilbert.py --label_mode full

# 7. Compare models
python src/evaluation/compare_reduced_models.py
python src/evaluation/compare_full_models.py

# 8. Analyse computational cost
python src/evaluation/analyse_computational_cost.py

# 9. Export summary tables
python ResultsSummary.py
```

---

## 10. Important Notes

This repository is not intended as a new model package. It is a reproducible experimental system for a dissertation. The key objective is to demonstrate a fair comparison between a classical lexical baseline and a lightweight transformer under the same dataset splits, labels, thresholding strategy and evaluation metrics.

The implementation should be reproducable:

- the processed GoEmotions data files;
- EDA figures;
- baseline results;
- DistilBERT results;
- full-label and reduced-label comparisons;
- statistical tables;
- computational-cost outputs;
- result table summary exports.
