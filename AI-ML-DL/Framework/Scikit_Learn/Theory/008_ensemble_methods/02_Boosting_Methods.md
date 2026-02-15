# Boosting Methods

---

## Table of Contents

- [Overview](#overview)
- [Gradient Boosting](#gradient-boosting)
- [GradientBoostingClassifier and Regressor](#gradientboostingclassifier-and-regressor)
- [AdaBoost](#adaboost)
- [Histogram-Based Gradient Boosting](#histogram-based-gradient-boosting)
- [Key Parameters](#key-parameters)
- [Comparison of Boosting Methods](#comparison-of-boosting-methods)

---

## Overview

**Boosting** builds an ensemble sequentially: each new model focuses on correcting the errors of the previous ones. Unlike bagging, base learners are not independent. **Gradient Boosting**, **AdaBoost**, and **HistGradientBoosting** are the main boosting methods in scikit-learn.

---

## Gradient Boosting

**Gradient Boosting** fits additive models of the form:

\[ F(x) = F_0 + \eta \sum_{m=1}^{M} h_m(x) \]

where \(h_m\) are decision trees (or other weak learners) and \(\eta\) is the **learning rate**. Each new tree \(h_m\) is fit to the **negative gradient** of the loss function with respect to the current prediction.

### Algorithm (Regression, MSE Loss)

1. Initialize \(F_0(x) = \bar{y}\)
2. For m = 1 to M:
   - Compute residuals: \(r_i = y_i - F_{m-1}(x_i)\)
   - Fit tree \(h_m\) to \((x_i, r_i)\)
   - Update: \(F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)\)
3. Output \(F_M(x)\)

### Key Ideas

- **Additive expansion**: Each tree adds a correction term
- **Gradient descent**: Residuals are the negative gradient of MSE
- **Shrinkage**: Learning rate \(\eta < 1\) regularizes by slowing updates

---

## GradientBoostingClassifier and Regressor

**GradientBoostingClassifier** and **GradientBoostingRegressor** implement gradient boosting with decision trees as base learners.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **n_estimators** | Number of boosting stages (trees) |
| **learning_rate** | Shrinkage factor (default: 0.1); smaller = more trees needed |
| **max_depth** | Maximum depth of each tree (default: 3) |
| **min_samples_split** | Minimum samples to split a node |
| **min_samples_leaf** | Minimum samples per leaf |
| **subsample** | Fraction of samples per tree (default: 1.0); < 1 = stochastic GB |
| **max_features** | Features per split (default: None = all) |

```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

gb_clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42
)

gb_reg = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
```

### staged_predict

**staged_predict** returns a generator of predictions after each boosting stage, useful for early stopping or learning curves.

```python
for pred in gb_reg.staged_predict(X_test):
    # pred is prediction after one more tree
    pass
```

---

## AdaBoost

**AdaBoost** (Adaptive Boosting) assigns weights to training samples. Misclassified samples get higher weights so subsequent learners focus on them.

### Algorithm

1. Initialize sample weights \(w_i = 1/n\)
2. For m = 1 to M:
   - Train weak learner \(h_m\) on weighted data
   - Compute weighted error \(\epsilon_m\)
   - Set \(\alpha_m = \frac{1}{2} \ln((1-\epsilon_m)/\epsilon_m)\)
   - Update weights: increase for misclassified, decrease for correct
   - Renormalize weights
3. Final model: \(H(x) = \text{sign}(\sum_m \alpha_m h_m(x))\)

### AdaBoostClassifier and AdaBoostRegressor

| Parameter | Description |
|-----------|-------------|
| **n_estimators** | Number of weak learners |
| **learning_rate** | Shrinks contribution of each learner |
| **estimator** | Base estimator (default: DecisionTreeClassifier(max_depth=1)) |
| **algorithm** | `SAMME` or `SAMME.R` (real-valued, uses probabilities) |

```python
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=2),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
ada.fit(X_train, y_train)
print(ada.estimator_weights_)
print(ada.estimator_errors_)
```

### estimator_weights_ and estimator_errors_

- **estimator_weights_**: Weight \(\alpha_m\) of each learner in the final vote
- **estimator_errors_**: Weighted error of each learner on the training set

---

## Histogram-Based Gradient Boosting

**HistGradientBoostingClassifier** and **HistGradientBoostingRegressor** use histogram-based binning for faster training on large datasets. They are inspired by LightGBM.

### Advantages

- **Speed**: Histogram binning reduces cost of finding split points
- **Native missing values**: Handles NaN without imputation
- **Categorical features**: Supports **categorical_features** parameter for native encoding

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **max_iter** | Number of boosting iterations (replaces n_estimators) |
| **learning_rate** | Shrinkage factor |
| **max_depth** | Maximum depth of trees |
| **max_bins** | Number of bins per feature (default: 255) |
| **categorical_features** | Indices or mask of categorical features |

```python
from sklearn.ensemble import HistGradientBoostingClassifier

hgb = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
hgb.fit(X_train, y_train)
```

### Missing Values

Pass data with NaN; HistGradientBoosting learns optimal missing-value strategy during training.

```python
X_with_nan = X.copy()
X_with_nan[0, 0] = np.nan
hgb.fit(X_with_nan, y)
```

### Categorical Features

Specify which columns are categorical. Each must have fewer than **max_bins** unique values.

```python
hgb = HistGradientBoostingClassifier(
    categorical_features=[2, 5],
    max_iter=100
)
```

---

## Key Parameters

### n_estimators / max_iter

More stages improve fit but risk overfitting. Use early stopping or validation to choose.

### learning_rate

- **High (e.g., 0.5)**: Fewer trees needed, faster; may overfit
- **Low (e.g., 0.05)**: More trees needed, slower; often better generalization

Rule of thumb: **n_estimators × learning_rate** should be tuned together.

### max_depth

- **Shallow (1–3)**: Weak learners; many trees; less overfitting
- **Deep (5–10)**: Strong learners; fewer trees; more overfitting risk

### subsample (Gradient Boosting)

**subsample < 1** enables stochastic gradient boosting: each tree uses a random subset of samples. Increases diversity and can reduce overfitting.

---

## Comparison of Boosting Methods

| Method | Speed | Missing Values | Categorical | Use Case |
|--------|-------|----------------|-------------|----------|
| **GradientBoosting** | Slow | Requires imputation | Requires encoding | General purpose; interpretable |
| **AdaBoost** | Medium | Requires imputation | Requires encoding | Simpler; small trees |
| **HistGradientBoosting** | Fast | Native support | Native support | Large data; mixed types |

---

## Summary

- **Gradient Boosting** fits trees to residuals in a stage-wise manner
- **AdaBoost** uses sample weights to focus on hard examples
- **HistGradientBoosting** uses histograms for speed and supports missing/categorical data
- Tune **n_estimators**, **learning_rate**, and **max_depth** together
- **subsample** enables stochastic gradient boosting for regularization
