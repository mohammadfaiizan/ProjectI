# Semi-Supervised Learning and Calibration

---

## Table of Contents

- [Overview](#overview)
- [Semi-Supervised Learning](#semi-supervised-learning)
- [LabelPropagation](#labelpropagation)
- [LabelSpreading](#labelspreading)
- [SelfTrainingClassifier](#selftrainingclassifier)
- [TransformedTargetRegressor](#transformedtargetregressor)
- [Probability Calibration](#probability-calibration)
- [CalibratedClassifierCV](#calibratedclassifiercv)
- [calibration_curve](#calibration_curve)
- [When to Use Each Method](#when-to-use-each-method)

---

## Overview

**Semi-supervised learning** uses both labeled and unlabeled data. **Calibration** adjusts classifier probability outputs so they reflect true likelihoods. **TransformedTargetRegressor** applies transformations to regression targets. This document covers these advanced techniques.

---

## Semi-Supervised Learning

When labeled data is scarce but unlabeled data is abundant, semi-supervised methods leverage the unlabeled data to improve performance. Unlabeled samples are typically marked with **-1** in the target array.

---

## LabelPropagation

**LabelPropagation** propagates labels through a graph built from the data. Similar samples (by kernel) tend to share labels. Uses a **label propagation** algorithm on the affinity matrix.

```python
from sklearn.semi_supervised import LabelPropagation

lp = LabelPropagation(kernel="rbf", gamma=0.25)
lp.fit(X_train, y_train_semi)
y_pred = lp.predict(X_test)
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **kernel** | "rbf", "knn", or "knn" |
| **gamma** | RBF kernel parameter |
| **n_neighbors** | For kNN kernel |

### Limitations

- Can be sensitive to noise
- No explicit regularization in the basic formulation

---

## LabelSpreading

**LabelSpreading** adds **regularization** (clamping) to LabelPropagation. The **alpha** parameter controls the amount of regularization; higher alpha means more smoothing and less reliance on initial labels.

```python
from sklearn.semi_supervised import LabelSpreading

ls = LabelSpreading(kernel="rbf", gamma=0.25, alpha=0.2)
ls.fit(X_train, y_train_semi)
y_pred = ls.predict(X_test)
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **kernel** | "rbf" or "knn" |
| **gamma** | RBF kernel parameter |
| **n_neighbors** | For kNN kernel |
| **alpha** | Clamping factor (0-1); higher = more regularization |

LabelSpreading is often more robust than LabelPropagation when labels are noisy.

---

## SelfTrainingClassifier

**SelfTrainingClassifier** iteratively trains a base classifier, predicts on unlabeled samples, and adds high-confidence predictions to the labeled set. Repeats until convergence or **max_iter**.

```python
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.linear_model import LogisticRegression

st = SelfTrainingClassifier(
    LogisticRegression(),
    threshold=0.9,
    criterion="threshold",
    max_iter=10,
)
st.fit(X_train, y_train_semi)
y_pred = st.predict(X_test)
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **threshold** | Min probability for a sample to be self-labeled |
| **criterion** | "threshold" (use threshold) or "k_best" (label k most confident) |
| **k** | Number of samples to add per iteration (when criterion="k_best") |
| **max_iter** | Maximum self-training iterations |

### Base Estimator

The base estimator should implement **predict_proba** for threshold-based selection. If not, **decision_function** is used (e.g., for SVM).

---

## TransformedTargetRegressor

**TransformedTargetRegressor** fits a regressor on transformed targets and applies the inverse transform to predictions. Useful when the target has a skewed distribution (e.g., log-normal).

```python
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge

reg = TransformedTargetRegressor(
    regressor=Ridge(),
    func=np.log1p,
    inverse_func=np.expm1,
)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

### Options

| Option | Use Case |
|--------|----------|
| **transformer** | Use a fitted transformer (e.g., QuantileTransformer) |
| **func, inverse_func** | Use explicit transform functions |

The regressor is fit on `func(y)`; predictions are `inverse_func(regressor.predict(X))`.

---

## Probability Calibration

Many classifiers produce **poorly calibrated** probabilities: predicted P(y=1) does not match the empirical frequency. **Calibration** adjusts these probabilities to be better aligned with reality.

### When Calibration Matters

- Decision thresholds (e.g., when to act)
- Cost-sensitive decisions
- Ensembles combining probability outputs

---

## CalibratedClassifierCV

**CalibratedClassifierCV** wraps a classifier and calibrates its probability outputs using a held-out set. Supports **isotonic regression** and **sigmoid** (Platt scaling) methods.

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

base_clf = LogisticRegression()
cal = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
cal.fit(X_train, y_train)
prob = cal.predict_proba(X_test)
```

### Methods

| Method | Description |
|--------|-------------|
| **isotonic** | Non-parametric; flexible, can overfit with few samples |
| **sigmoid** | Platt scaling; parametric, more stable with few samples |

### cv Parameter

- **cv=3** (or similar): use cross-validation to generate calibration data
- **cv="prefit"**: use a pre-fitted base estimator; provide separate calibration data

---

## calibration_curve

**calibration_curve** computes the relationship between predicted probabilities and true frequencies. Used to plot reliability diagrams.

```python
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
```

- **prob_true**: fraction of positives in each bin
- **prob_pred**: mean predicted probability in each bin

A well-calibrated model has prob_true close to prob_pred.

---

## When to Use Each Method

| Method | Use When |
|--------|----------|
| **LabelPropagation** | Graph-based; many unlabeled samples; fast |
| **LabelSpreading** | Same as above but more robust to noise |
| **SelfTrainingClassifier** | Any base classifier; iterative self-labeling |
| **TransformedTargetRegressor** | Skewed regression targets |
| **CalibratedClassifierCV** | Need reliable probabilities for decision-making |

Semi-supervised methods help when labels are expensive. Calibration improves probability estimates for downstream decisions.
