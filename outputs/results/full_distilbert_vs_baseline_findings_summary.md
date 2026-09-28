# Full-label DistilBERT vs TF-IDF Logistic Regression Findings

## Summary metric comparison

Macro F1 increased from 0.4458 for TF-IDF Logistic Regression to 0.5182 for DistilBERT. The absolute improvement was 0.0724.
Micro F1 increased from 0.5228 to 0.6052, giving an absolute improvement of 0.0824.
Macro recall increased from 0.4981 to 0.5442.
Exact-match accuracy increased from 0.2937 to 0.4074.
Hamming loss decreased from 0.0477 to 0.0366. Since lower Hamming loss is better, this also favours DistilBERT.

## Label-level statistical evidence

Across the full label set, the mean per-label F1 difference (DistilBERT minus TF-IDF) was 0.0724. The median difference was 0.0816.
DistilBERT improved F1 on 26 labels, performed worse on 2 labels, and tied on 0 labels.
The paired t-test p-value for label-level F1 was 0.002369; the Wilcoxon signed-rank test p-value was 0.000028; and the sign-test p-value was 0.000003.
The bootstrap 95% confidence interval for the mean per-label F1 difference was [0.0259, 0.1081].
These tests are based on the 11 paired emotion labels and should be interpreted as supporting label-level statistical evidence rather than as an instance-level significance test.

## Strongest F1 improvements

- nervousness: TF-IDF F1=0.1818, DistilBERT F1=0.4103, difference=0.2284.
- embarrassment: TF-IDF F1=0.2821, DistilBERT F1=0.5091, difference=0.2270.
- relief: TF-IDF F1=0.1176, DistilBERT F1=0.3243, difference=0.2067.
- caring: TF-IDF F1=0.2606, DistilBERT F1=0.4364, difference=0.1758.
- curiosity: TF-IDF F1=0.4012, DistilBERT F1=0.5769, difference=0.1757.

## Weakest F1 improvements

- grief: TF-IDF F1=0.4000, DistilBERT F1=0.0000, difference=-0.4000.
- realization: TF-IDF F1=0.2661, DistilBERT F1=0.2252, difference=-0.0408.
- optimism: TF-IDF F1=0.5625, DistilBERT F1=0.5645, difference=0.0020.
- fear: TF-IDF F1=0.6456, DistilBERT F1=0.6484, difference=0.0028.
- gratitude: TF-IDF F1=0.9049, DistilBERT F1=0.9220, difference=0.0171.

## Visual evidence files

- outputs/figures/full_baseline_vs_distilbert_metric_comparison.png
- outputs/figures/full_distilbert_minus_baseline_metric_delta.png
- outputs/figures/full_per_label_f1_baseline_vs_distilbert.png
- outputs/figures/full_per_label_f1_difference_distilbert_minus_baseline.png
- outputs/figures/full_support_vs_f1_baseline_vs_distilbert.png
- outputs/figures/full_f1_gain_vs_support.png
- outputs/figures/full_precision_recall_space_baseline_vs_distilbert.png
- outputs/figures/full_threshold_tuning_baseline_vs_distilbert.png
- outputs/figures/full_per_label_thresholds_baseline_vs_distilbert.png
- outputs/figures/full_label_metric_gain_heatmap.png
- outputs/figures/full_training_time_baseline_vs_distilbert.png