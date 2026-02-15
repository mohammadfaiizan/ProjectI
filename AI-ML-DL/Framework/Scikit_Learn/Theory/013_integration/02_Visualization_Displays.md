# Visualization Displays

---

## Table of Contents

- [Overview](#overview)
- [DecisionBoundaryDisplay](#decisionboundarydisplay)
- [RocCurveDisplay](#roccurvedisplay)
- [ConfusionMatrixDisplay](#confusionmatrixdisplay)
- [PrecisionRecallDisplay](#precisionrecalldisplay)
- [LearningCurveDisplay](#learningcurvedisplay)
- [ValidationCurveDisplay](#validationcurvedisplay)
- [Combining Displays](#combining-displays)
- [Best Practices](#best-practices)

---

## Overview

Scikit-learn provides object-oriented display classes for model evaluation and diagnostics. These replace deprecated function-based plotting APIs and offer consistent interfaces, matplotlib axes integration, and reusable components. This document covers **DecisionBoundaryDisplay**, **RocCurveDisplay**, **ConfusionMatrixDisplay**, **PrecisionRecallDisplay**, **LearningCurveDisplay**, and **ValidationCurveDisplay**.

---

## DecisionBoundaryDisplay

### Purpose

**DecisionBoundaryDisplay** visualizes the decision boundary of a classifier in 2D feature space. It is useful for understanding how linear and non-linear classifiers separate classes.

### from_estimator

Create a display from a fitted estimator:

```python
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC

clf = SVC(kernel="rbf").fit(X_train, y_train)
disp = DecisionBoundaryDisplay.from_estimator(
    clf, X_train,
    response_method="predict",
    xlabel="Feature 0", ylabel="Feature 1",
    alpha=0.5
)
disp.ax_.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors="k")
```

### response_method

| Value | Use Case |
|-------|----------|
| **predict** | Class labels (default for classifiers) |
| **predict_proba** | Probability for each class; produces multiple subplots for multiclass |
| **decision_function** | Raw scores (SVM, etc.) |

### Pipeline Support

Works with **Pipeline** objects. Pass the pipeline as the estimator; it will call `predict` or `predict_proba` through the pipeline.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([("scaler", StandardScaler()), ("clf", SVC())])
pipe.fit(X_train, y_train)
disp = DecisionBoundaryDisplay.from_estimator(pipe, X_train, response_method="predict")
```

### Limitations

- Only for **2D** feature space; use PCA or feature selection to reduce to 2 features
- **grid_resolution** controls mesh density; higher values are slower

---

## RocCurveDisplay

### Purpose

**RocCurveDisplay** plots the Receiver Operating Characteristic (ROC) curve, showing the trade-off between true positive rate (TPR) and false positive rate (FPR) at various thresholds.

### from_predictions

Plot from ground truth and predicted probabilities:

```python
from sklearn.metrics import RocCurveDisplay

y_proba = clf.predict_proba(X_test)[:, 1]
disp = RocCurveDisplay.from_predictions(y_test, y_proba)
```

### from_estimator

Plot directly from an estimator and test data:

```python
disp = RocCurveDisplay.from_estimator(clf, X_test, y_test)
# disp.roc_auc gives the AUC value
```

### Multiclass: One-vs-Rest

For multiclass, use **label_binarize** and plot one curve per class:

```python
from sklearn.preprocessing import label_binarize

y_bin = label_binarize(y_test, classes=[0, 1, 2])
y_proba = clf.predict_proba(X_test)
for i in range(n_classes):
    RocCurveDisplay.from_predictions(y_bin[:, i], y_proba[:, i], ax=ax, name=f"Class {i}")
```

### AUC

Access the AUC via `disp.roc_auc` when using `from_estimator`, or compute with `sklearn.metrics.auc(fpr, tpr)`.

---

## ConfusionMatrixDisplay

### Purpose

**ConfusionMatrixDisplay** visualizes the confusion matrix with optional normalization and custom labels.

### from_predictions

```python
from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
# Or with labels:
disp = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Negative", "Positive"]
)
```

### from_estimator

```python
disp = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
```

### Normalization

Use **normalize** parameter: `"true"` (rows), `"pred"` (columns), or `"all"` (total).

```python
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize="true")
```

---

## PrecisionRecallDisplay

### Purpose

**PrecisionRecallDisplay** plots precision vs recall, especially useful for **imbalanced** classification where ROC can be misleading.

### from_predictions

```python
from sklearn.metrics import PrecisionRecallDisplay

disp = PrecisionRecallDisplay.from_predictions(y_test, y_proba)
```

### from_estimator

```python
disp = PrecisionRecallDisplay.from_estimator(clf, X_test, y_test)
# disp.average_precision gives AP score
```

### Average Precision

**average_precision_score** summarizes the precision-recall curve. Higher is better. Compare with a no-skill baseline (horizontal line at positive class prevalence).

---

## LearningCurveDisplay

### Purpose

**LearningCurveDisplay** plots training and validation scores as a function of training set size. Helps diagnose underfitting, overfitting, and whether more data would help.

### from_estimator

```python
from sklearn.model_selection import LearningCurveDisplay

disp = LearningCurveDisplay.from_estimator(
    clf, X_train, y_train,
    train_sizes=[50, 100, 200, 400, 600],
    cv=5, n_jobs=-1
)
# disp.train_sizes_, disp.train_scores_, disp.test_scores_
```

### Interpretation

- **Large gap** between train and test: overfitting
- **Both low**: underfitting
- **Converging with more data**: may benefit from more samples

---

## ValidationCurveDisplay

### Purpose

**ValidationCurveDisplay** plots train and validation scores as a function of a **single hyperparameter**. Useful for tuning (e.g., C, gamma, max_depth).

### from_estimator

```python
from sklearn.model_selection import ValidationCurveDisplay

disp = ValidationCurveDisplay.from_estimator(
    clf, X_train, y_train,
    param_name="gamma",
    param_range=[1e-4, 1e-3, 1e-2, 1e-1],
    cv=5
)
```

### param_name

Use the parameter name as seen in `clf.get_params()`. For pipelines, use `"clf__gamma"`.

---

## Combining Displays

### Subplots

All display classes return or use a matplotlib axes. Combine multiple displays on a figure:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2)
RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=axes[0])
PrecisionRecallDisplay.from_estimator(clf, X_test, y_test, ax=axes[1])
plt.tight_layout()
plt.show()
```

### plot Method

Some displays have a **plot** method to render on a given axes. Check the API for `plot(ax=ax)`.

---

## Best Practices

| Practice | Recommendation |
|----------|----------------|
| **2D for decision boundary** | Reduce to 2 features via PCA or selection |
| **Imbalanced data** | Prefer PrecisionRecallDisplay over ROC |
| **Multiclass ROC** | Use one-vs-rest with label_binarize |
| **Learning curves** | Use multiple train_sizes and CV folds |
| **Validation curves** | Sweep one parameter at a time |
| **Reproducibility** | Set random_state in CV and estimators |

---

## Summary

- **DecisionBoundaryDisplay**: 2D decision boundaries; use `response_method` for predict/proba
- **RocCurveDisplay**: ROC curve and AUC; supports multiclass one-vs-rest
- **ConfusionMatrixDisplay**: Confusion matrix with optional normalization
- **PrecisionRecallDisplay**: Precision-recall for imbalanced problems
- **LearningCurveDisplay**: Train/test score vs sample size
- **ValidationCurveDisplay**: Train/test score vs hyperparameter value
- All displays integrate with matplotlib and support custom axes
