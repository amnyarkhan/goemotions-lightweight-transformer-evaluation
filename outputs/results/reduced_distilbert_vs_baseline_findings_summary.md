# Reduced-label DistilBERT vs TF-IDF Logistic Regression Findings

## Summary metric comparison

Macro F1 increased from 0.5457 for TF-IDF Logistic Regression to 0.6060 for DistilBERT. The absolute improvement was 0.0603.
Micro F1 increased from 0.5596 to 0.6341, giving an absolute improvement of 0.0745.
Macro recall increased from 0.6330 to 0.6719.
Exact-match accuracy increased from 0.3947 to 0.4964.
Hamming loss decreased from 0.0871 to 0.0664. Since lower Hamming loss is better, this also favours DistilBERT.

## Label-level statistical evidence

Across the reduced label set, the mean per-label F1 difference (DistilBERT minus TF-IDF) was 0.0603. The median difference was 0.0536.
DistilBERT improved F1 on 11 labels, performed worse on 0 labels, and tied on 0 labels.
The paired t-test p-value for label-level F1 was 0.001034; the Wilcoxon signed-rank test p-value was 0.000977; and the sign-test p-value was 0.000977.
The bootstrap 95% confidence interval for the mean per-label F1 difference was [0.0370, 0.0867].
These tests are based on the 11 paired emotion labels and should be interpreted as supporting label-level statistical evidence rather than as an instance-level significance test.

## Strongest F1 improvements

- curiosity: TF-IDF F1=0.4012, DistilBERT F1=0.5614, difference=0.1602.
- disapproval: TF-IDF F1=0.3143, DistilBERT F1=0.4080, difference=0.0937.
- approval: TF-IDF F1=0.2969, DistilBERT F1=0.3879, difference=0.0910.
- admiration: TF-IDF F1=0.6321, DistilBERT F1=0.7064, difference=0.0744.
- anger: TF-IDF F1=0.4188, DistilBERT F1=0.4778, difference=0.0589.

## Weakest F1 improvements

- optimism: TF-IDF F1=0.5625, DistilBERT F1=0.5686, difference=0.0061.
- gratitude: TF-IDF F1=0.9049, DistilBERT F1=0.9177, difference=0.0128.
- amusement: TF-IDF F1=0.7897, DistilBERT F1=0.8190, difference=0.0293.
- love: TF-IDF F1=0.7579, DistilBERT F1=0.7959, difference=0.0380.
- neutral: TF-IDF F1=0.6437, DistilBERT F1=0.6886, difference=0.0449.

## Visual evidence files

- outputs/figures/reduced_baseline_vs_distilbert_metric_comparison.png
- outputs/figures/reduced_distilbert_minus_baseline_metric_delta.png
- outputs/figures/reduced_per_label_f1_baseline_vs_distilbert.png
- outputs/figures/reduced_per_label_f1_difference_distilbert_minus_baseline.png
- outputs/figures/reduced_support_vs_f1_baseline_vs_distilbert.png
- outputs/figures/reduced_f1_gain_vs_support.png
- outputs/figures/reduced_precision_recall_space_baseline_vs_distilbert.png
- outputs/figures/reduced_threshold_tuning_baseline_vs_distilbert.png
- outputs/figures/reduced_per_label_thresholds_baseline_vs_distilbert.png
- outputs/figures/reduced_label_metric_gain_heatmap.png
- outputs/figures/reduced_training_time_baseline_vs_distilbert.png