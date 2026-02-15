# Classification Metrics

---

## Table of Contents

- [Overview](#overview)
- [Accuracy, Precision, Recall, F1](#accuracy-precision-recall-f1)
- [Classification Report](#classification-report)
- [Confusion Matrix](#confusion-matrix)
- [ROC Curve and AUC](#roc-curve-and-auc)
- [Precision-Recall Curve](#precision-recall-curve)

---

## Overview

Classification metrics quantify how well a classifier predicts class labels. Different metrics emphasize different aspects: **accuracy** for overall correctness, **precision** for positive prediction quality, **recall** for positive class coverage, and **F1** for their balance.

---

## Accuracy, Precision, Recall, F1

### accuracy_score

**accuracy_score** is the fraction of correct predictions over total predictions.

\[
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\]

For balanced datasets, accuracy is meaningful. For imbalanced data, it can be misleading (e.g., 99% accuracy when 99% of samples are one class).

```python
from sklearn.metrics import accuracy_score

y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]
accuracy_score(y_true, y_pred)  # 0.8
```

### precision_score

**Precision** is the fraction of predicted positives that are actually positive.

\[
\text{Precision} = \frac{TP}{TP + FP}
\]

Use when false positives are costly (e.g., spam filter marking legitimate email as spam).

```python
from sklearn.metrics import precision_score

precision_score(y_true, y_pred, zero_division=0)
```

### recall_score

**Recall** (sensitivity, true positive rate) is the fraction of actual positives that are correctly predicted.

\[
\text{Recall} = \frac{TP}{TP + FN}
\]

Use when false negatives are costly (e.g., missing a disease diagnosis).

```python
from sklearn.metrics import recall_score

recall_score(y_true, y_pred, zero_division=0)
```

### f1_score

**F1 score** is the harmonic mean of precision and recall.

\[
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
\]

Use when you need a single metric balancing precision and recall.

```python
from sklearn.metrics import f1_score

f1_score(y_true, y_pred, zero_division=0)
```

### average Parameter (Multiclass)

For multiclass problems, use the **average** parameter:

| Value | Description |
|-------|-------------|
| **macro** | Unweighted mean across classes |
| **micro** | Aggregate TP, FP, FN then compute |
| **weighted** | Weight by support (number of true instances per class) |
| **None** | Return per-class scores |

```python
precision_score(y_true, y_pred, average='macro')
recall_score(y_true, y_pred, average='weighted')
f1_score(y_true, y_pred, average=None)  # Per-class
```

---

## Classification Report

### classification_report

**classification_report** produces a text report with precision, recall, f1-score, and support per class.

```python
from sklearn.metrics import classification_report

report = classification_report(y_true, y_pred, target_names=['neg', 'pos'])
print(report)
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| **target_names** | Display names for classes |
| **digits** | Number of decimal places |
| **zero_division** | Value when division by zero (0 or 1) |
| **output_dict** | Return dict instead of string |

### output_dict

With `output_dict=True`, you get a dictionary with keys like `'0'`, `'1'`, `'macro avg'`, `'weighted avg'`, each containing `precision`, `recall`, `f1-score`, `support`.

---

## Confusion Matrix

### confusion_matrix

A **confusion matrix** shows counts of true vs predicted labels. Rows = true, columns = predicted.

|  | Pred 0 | Pred 1 |
|---|--------|--------|
| True 0 | TN | FP |
| True 1 | FN | TP |

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
```

### normalize Parameter

- `normalize='pred'`: Normalize by predicted (columns)
- `normalize='true'`: Normalize by true (rows)
- `normalize='all'`: Normalize by total count

### ConfusionMatrixDisplay

**ConfusionMatrixDisplay.from_estimator** and **ConfusionMatrixDisplay.from_predictions** create visualizations.

```python
from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
disp.plot()
```

---

## ROC Curve and AUC

### roc_curve

The **ROC curve** plots True Positive Rate (TPR) vs False Positive Rate (FPR) at various classification thresholds.

\[
\text{TPR} = \frac{TP}{TP + FN}, \quad \text{FPR} = \frac{FP}{FP + TN}
\]

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
```

### roc_auc_score

**AUC** (Area Under the Curve) summarizes ROC performance. AUC = 1 is perfect, 0.5 is random.

```python
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_true, y_scores)
```

### Multi-class AUC

For multiclass, use `multi_class='ovr'` (one-vs-rest), `'ovr_weighted'`, `'macro'`, or `'micro'`.

```python
roc_auc_score(y_true, y_proba, multi_class='ovr')
```

### RocCurveDisplay

```python
from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_estimator(clf, X_test, y_test)
```

---

## Precision-Recall Curve

### precision_recall_curve

When classes are imbalanced, **precision-recall** curves are often more informative than ROC. They plot precision vs recall at various thresholds.

```python
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
```

### average_precision_score

**Average Precision (AP)** is the area under the precision-recall curve. For binary classification, AP equals the mean precision at each recall level.

```python
from sklearn.metrics import average_precision_score

ap = average_precision_score(y_true, y_scores)
```

### PrecisionRecallDisplay

```python
from sklearn.metrics import PrecisionRecallDisplay

PrecisionRecallDisplay.from_estimator(clf, X_test, y_test)
```

### Multiclass

For multiclass, use `average='macro'` or `'micro'` with probability predictions.

---
