# Baseline Result Summary

## Full vs Reduced Baseline

Macro F1 improved from 0.446 to 0.546.
Micro F1 improved from 0.523 to 0.560.
Macro recall improved from 0.498 to 0.633.
Exact-match accuracy improved from 0.294 to 0.395.
Hamming loss increased from 0.048 to 0.087; this should be interpreted carefully because Hamming loss is affected by the number of labels.

## Strongest Full-Label Emotions

- gratitude: F1=0.905, precision=0.918, recall=0.892, support=352
- amusement: F1=0.790, precision=0.725, recall=0.867, support=264
- love: F1=0.758, precision=0.718, recall=0.803, support=238
- fear: F1=0.646, precision=0.637, recall=0.654, support=78
- neutral: F1=0.644, precision=0.532, recall=0.815, support=1787
- admiration: F1=0.632, precision=0.576, recall=0.700, support=504
- remorse: F1=0.631, precision=0.505, recall=0.839, support=56
- optimism: F1=0.562, precision=0.545, recall=0.581, support=186
- joy: F1=0.551, precision=0.522, recall=0.584, support=161
- sadness: F1=0.486, precision=0.473, recall=0.500, support=156

## Weakest Full-Label Emotions

- relief: F1=0.118, precision=0.075, recall=0.273, support=11
- nervousness: F1=0.182, precision=0.300, recall=0.130, support=23
- disappointment: F1=0.234, precision=0.185, recall=0.318, support=151
- caring: F1=0.261, precision=0.233, recall=0.296, support=135
- realization: F1=0.266, precision=0.397, recall=0.200, support=145
- annoyance: F1=0.281, precision=0.191, recall=0.528, support=320
- embarrassment: F1=0.282, precision=0.268, recall=0.297, support=37
- approval: F1=0.297, precision=0.244, recall=0.379, support=351
- disapproval: F1=0.314, precision=0.233, recall=0.483, support=267
- excitement: F1=0.315, precision=0.241, recall=0.456, support=103