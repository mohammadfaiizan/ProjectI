# Scoring, Distributions, and CV Strategies

---

## Table of Contents

- [Parameter Distributions](#parameter-distributions)
- [Scoring and Refit](#scoring-and-refit)
- [CV Strategies in Tuning](#cv-strategies-in-tuning)
- [Multi-Metric Search](#multi-metric-search)
- [Custom Scorers](#custom-scorers)

---

## Parameter Distributions

### scipy.stats for RandomizedSearchCV

Use **scipy.stats** distributions in **param_distributions** to sample continuous or discrete values. Common choices: **uniform**, **randint**, **loguniform**.

```python
from scipy.stats import uniform, randint, loguniform

param_distributions = {
    "n_estimators": randint(10, 200),
    "max_depth": randint(2, 20),
    "C": loguniform(1e-2, 1e2),
    "gamma": loguniform(1e-4, 1e-1),
}
```

| Distribution | Use Case |
|--------------|----------|
| **randint(low, high)** | Discrete integers in [low, high) |
| **uniform(loc, scale)** | Continuous uniform |
| **loguniform(a, b)** | Log-uniform for scale-sensitive params (C, gamma) |

### Log-Uniform for Scale Parameters

Parameters like **C** and **gamma** in SVM span orders of magnitude. **loguniform** samples uniformly in log-space, giving better coverage.

```python
from scipy.stats import loguniform

# Samples between 1e-2 and 1e2, log-uniform
C_dist = loguniform(1e-2, 1e2)
```

### Mixing Lists and Distributions

You can mix lists (for categorical) with distributions:

```python
param_distributions = {
    "C": loguniform(1e-2, 1e2),
    "kernel": ["rbf", "linear"],
}
```

---

## Scoring and Refit

### scoring Parameter

**scoring** defines the metric to maximize (or minimize). Use string names or **make_scorer** for custom metrics.

```python
grid = GridSearchCV(model, param_grid, scoring="accuracy")
grid = GridSearchCV(model, param_grid, scoring="f1_macro")
grid = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error")
```

| scoring | Use Case |
|---------|----------|
| **accuracy** | Default for classifiers |
| **f1_macro** | Imbalanced classification |
| **precision_macro** | When false positives costly |
| **recall_macro** | When false negatives costly |
| **neg_mean_squared_error** | Regression (minimize MSE) |

### refit Strategies

**refit** selects which model to retrain on full data when using multiple metrics. Options: metric name string or callable.

```python
# Refit on best accuracy (default)
grid = GridSearchCV(model, param_grid, scoring=["accuracy", "f1_macro"], refit="accuracy")

# Refit with callable: custom selection logic
def my_refit(cv_results):
    return np.argmax(0.5 * cv_results["mean_test_accuracy"] + 0.5 * cv_results["mean_test_f1_macro"])
grid = GridSearchCV(model, param_grid, scoring=["accuracy", "f1_macro"], refit=my_refit)
```

### callable refit

A callable receives **cv_results_** and returns the index of the best candidate. Use for custom composite objectives.

---

## CV Strategies in Tuning

### KFold

**KFold** splits data into K folds. Use for regression or when class balance is not critical.

```python
from sklearn.model_selection import GridSearchCV, KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(model, param_grid, cv=kf)
```

### StratifiedKFold

**StratifiedKFold** preserves class proportions. Default for classifiers when `cv` is an integer.

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(clf, param_grid, cv=skf)
```

### GroupKFold

**GroupKFold** ensures samples from the same group stay together. Use when data has natural groups (e.g., patients, subjects).

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
grid = GridSearchCV(model, param_grid, cv=gkf)
grid.fit(X, y, groups=groups)
```

### cv as Integer

Passing `cv=5` uses **StratifiedKFold** for classifiers and **KFold** for regressors. Shuffle and random_state are not set by default.

---

## Multi-Metric Search

### Multiple scoring Metrics

Pass a list to **scoring** to compute multiple metrics. **cv_results_** will contain keys like `mean_test_accuracy`, `mean_test_f1_macro`, etc.

```python
grid = GridSearchCV(
    clf,
    param_grid,
    cv=5,
    scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
    refit="f1_macro",
)
grid.fit(X_train, y_train)
```

### refit with Multiple Metrics

**refit** must be one of the metric names or a callable. The refitted model is used for `predict` and `predict_proba`.

### Analyzing Multi-Metric Results

Different metrics may favor different parameter combinations. Inspect **cv_results_** to compare:

```python
for metric in ["accuracy", "f1_macro"]:
    key = f"mean_test_{metric}"
    best_idx = np.argmax(grid.cv_results_[key])
    print(f"{metric}: {grid.cv_results_['params'][best_idx]}")
```

---

## Custom Scorers

### make_scorer

**make_scorer** wraps a metric function for use in search. Key arguments: **greater_is_better**, **needs_proba**, **needs_threshold**.

```python
from sklearn.metrics import make_scorer, fbeta_score

f2_scorer = make_scorer(fbeta_score, beta=2, average="macro")
grid = GridSearchCV(clf, param_grid, scoring=f2_scorer)
```

### greater_is_better

Set **greater_is_better=False** for loss-like metrics (e.g., MSE). The search will minimize the score.

```python
from sklearn.metrics import mean_squared_error

mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
```

### needs_proba

For metrics requiring probability estimates (e.g., **roc_auc**), set **needs_proba=True**. The estimator must implement `predict_proba`.

```python
from sklearn.metrics import roc_auc_score

auc_scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class="ovr")
grid = GridSearchCV(LogisticRegression(), param_grid, scoring=auc_scorer)
```

### needs_threshold

For metrics using decision function (e.g., **average_precision** with SVM), set **needs_threshold=True**. The estimator must implement `decision_function`.

---
