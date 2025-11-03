# Model Comparison Report

**Generated:** 2025-11-02 18:22:24

## Executive Summary

**Best Performing Model:** Gradient Boosting

**ROC AUC:** 0.9805

## Performance Metrics

|                   |   ROC AUC |   F1 Score |   Precision |   Recall |   Accuracy |
|:------------------|----------:|-----------:|------------:|---------:|-----------:|
| Isolation Forest  |  0.780403 |   0.750031 |    0.709665 | 0.795266 |   0.708133 |
| SGD Classifier    |  0.902491 |   0.833393 |    0.729492 | 0.971808 |   0.786061 |
| Random Forest     |  0.978898 |   0.929213 |    0.966741 | 0.89449  |   0.924962 |
| Gradient Boosting |  0.980499 |   0.932175 |    0.95815  | 0.907571 |   0.927282 |
| Extra Trees       |  0.971885 |   0.91708  |    0.965331 | 0.873423 |   0.913035 |

## Detailed Metrics

|                   |   ROC AUC |   Average Precision |   Precision |   Recall |   F1 Score |   Accuracy |   Balanced Accuracy |      MCC |   Cohen Kappa |   Optimal Threshold |   True Positives |   True Negatives |   False Positives |   False Negatives |       FPR |      TPR |
|:------------------|----------:|--------------------:|------------:|---------:|-----------:|-----------:|--------------------:|---------:|--------------:|--------------------:|-----------------:|-----------------:|------------------:|------------------:|----------:|---------:|
| Isolation Forest  |  0.780403 |            0.821586 |    0.709665 | 0.795266 |   0.750031 |   0.708133 |            0.698322 | 0.405879 |      0.402106 |           -0.108729 |            36051 |            22251 |             14749 |              9281 | 0.398622  | 0.795266 |
| SGD Classifier    |  0.902491 |            0.919026 |    0.729492 | 0.971808 |   0.833393 |   0.786061 |            0.765147 | 0.596623 |      0.550898 |           -0.812192 |            44054 |            20664 |             16336 |              1278 | 0.441514  | 0.971808 |
| Random Forest     |  0.978898 |            0.984811 |    0.966741 | 0.89449  |   0.929213 |   0.924962 |            0.928393 | 0.85254  |      0.849637 |            0.668849 |            40549 |            35605 |              1395 |              4783 | 0.0377027 | 0.89449  |
| Gradient Boosting |  0.980499 |            0.985955 |    0.95815  | 0.907571 |   0.932175 |   0.927282 |            0.929502 | 0.855387 |      0.853928 |            1.40913  |            41142 |            35203 |              1797 |              4190 | 0.0485676 | 0.907571 |
| Extra Trees       |  0.971885 |            0.980005 |    0.965331 | 0.873423 |   0.91708  |   0.913035 |            0.917495 | 0.830709 |      0.826134 |            0.645238 |            39594 |            35578 |              1422 |              5738 | 0.0384324 | 0.873423 |

## Model Rankings

### By ROC AUC

🥇 **Gradient Boosting**: 0.9805

🥈 **Random Forest**: 0.9789

🥉 **Extra Trees**: 0.9719

4. **SGD Classifier**: 0.9025

5. **Isolation Forest**: 0.7804


### By F1 Score

🥇 **Gradient Boosting**: 0.9322

🥈 **Random Forest**: 0.9292

🥉 **Extra Trees**: 0.9171

4. **SGD Classifier**: 0.8334

5. **Isolation Forest**: 0.7500


### By Precision

🥇 **Random Forest**: 0.9667

🥈 **Extra Trees**: 0.9653

🥉 **Gradient Boosting**: 0.9581

4. **SGD Classifier**: 0.7295

5. **Isolation Forest**: 0.7097


### By Recall

🥇 **SGD Classifier**: 0.9718

🥈 **Gradient Boosting**: 0.9076

🥉 **Random Forest**: 0.8945

4. **Extra Trees**: 0.8734

5. **Isolation Forest**: 0.7953


## Confusion Matrix Analysis

### Isolation Forest

- True Positives: 36,051
- True Negatives: 22,251
- False Positives: 14,749
- False Negatives: 9,281
- FPR: 0.3986
- TPR: 0.7953

### SGD Classifier

- True Positives: 44,054
- True Negatives: 20,664
- False Positives: 16,336
- False Negatives: 1,278
- FPR: 0.4415
- TPR: 0.9718

### Random Forest

- True Positives: 40,549
- True Negatives: 35,605
- False Positives: 1,395
- False Negatives: 4,783
- FPR: 0.0377
- TPR: 0.8945

### Gradient Boosting

- True Positives: 41,142
- True Negatives: 35,203
- False Positives: 1,797
- False Negatives: 4,190
- FPR: 0.0486
- TPR: 0.9076

### Extra Trees

- True Positives: 39,594
- True Negatives: 35,578
- False Positives: 1,422
- False Negatives: 5,738
- FPR: 0.0384
- TPR: 0.8734

