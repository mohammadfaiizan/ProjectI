# Curves, Custom Scoring, and Practices

## Table of Contents

1. [Learning Curves](#1-learning-curves)
2. [Validation Curves](#2-validation-curves)
3. [Custom Scoring with make_scorer](#3-custom-scoring-with-make_scorer)
4. [Multi-Metric Evaluation](#4-multi-metric-evaluation)
5. [Refit Strategies](#5-refit-strategies)
6. [Evaluation Best Practices](#6-evaluation-best-practices)

---

## 1. Learning Curves

**Learning curves** plot model performance (e.g., training and validation score) as a function of training set size. They help diagnose bias-variance tradeoff and whether more data would help.

### learning_curve

`sklearn.model_selection.learning_curve` computes cross-validated scores for different training set sizes.

```python
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
estimator = SVC(gamma=0.001)

train_sizes, train_scores, test_scores = learning_curve(
    estimator,
    X, y,
    train_sizes=[100, 500, 1000, 1500, 2000],
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
```

### Parameters

| Parameter | Purpose |
|-----------|---------|
| `train_sizes` | Absolute or relative sizes to evaluate |
| `cv` | Cross-validation strategy |
| `scoring` | Metric name or callable |
| `n_jobs` | Parallel jobs |

### LearningCurveDisplay

`LearningCurveDisplay` provides a convenient plotting interface.

```python
from sklearn.model_selection import LearningCurveDisplay

LearningCurveDisplay.from_estimator(
    estimator,
    X, y,
    train_sizes=train_sizes,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
```

### Interpreting Learning Curves

| Pattern | Interpretation |
|---------|----------------|
| Large gap between train and validation | Overfitting; try regularization or simpler model |
| Both curves low | Underfitting; try more complex model or features |
| Both curves plateau, gap small | Sufficient model capacity; more data may help |
| Validation still rising | More data likely to improve performance |

### Example with Plotting

```python
import matplotlib.pyplot as plt
import numpy as np

train_sizes, train_scores, test_scores = learning_curve(
    estimator, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2)
plt.plot(train_sizes, train_mean, 'o-', label='Training score')
plt.plot(train_sizes, test_mean, 'o-', label='Validation score')
plt.xlabel('Training set size')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
```

---

## 2. Validation Curves

**Validation curves** plot performance as a function of a hyperparameter (e.g., `C`, `gamma`, `max_depth`). They help select hyperparameter values.

### validation_curve

`sklearn.model_selection.validation_curve` computes cross-validated scores for different values of a single hyperparameter.

```python
from sklearn.model_selection import validation_curve
from sklearn.svm import SVC

param_name = 'gamma'
param_range = np.logspace(-6, -1, 6)

train_scores, test_scores = validation_curve(
    SVC(),
    X, y,
    param_name=param_name,
    param_range=param_range,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
```

### ValidationCurveDisplay

```python
from sklearn.model_selection import ValidationCurveDisplay

ValidationCurveDisplay.from_estimator(
    SVC(),
    X, y,
    param_name='C',
    param_range=np.logspace(-2, 2, 5),
    cv=5,
    scoring='accuracy'
)
```

### Use Cases

| Hyperparameter | Typical Range |
|----------------|---------------|
| SVM `C` | Log scale, e.g., 1e-3 to 1e3 |
| SVM `gamma` | Log scale, e.g., 1e-6 to 1e-1 |
| Tree `max_depth` | 1 to 20 |
| `n_estimators` (ensemble) | 10 to 500 |

---

## 3. Custom Scoring with make_scorer

**make_scorer** creates a scorer object from a metric function. Scorers are used in `cross_val_score`, `GridSearchCV`, and other evaluation tools.

### Basic Usage

```python
from sklearn.metrics import make_scorer, f1_score, mean_squared_error

# From existing metric
f1_scorer = make_scorer(f1_score, average='macro')

# With extra parameters
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
```

### greater_is_better

- `True`: Higher score is better (e.g., accuracy, F1)
- `False`: Lower score is better (e.g., MSE, MAE)

```python
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
# GridSearchCV will maximize the negated MSE, i.e., minimize MSE
```

### Custom Metric Function

The metric function signature must be `(y_true, y_pred, **kwargs)` or `(y_true, y_pred)`.

```python
def custom_metric(y_true, y_pred):
    """Example: weighted accuracy favoring minority class"""
    correct = (y_true == y_pred).astype(float)
    weights = np.where(y_true == 1, 2.0, 1.0)  # Weight minority class
    return np.average(correct, weights=weights)

custom_scorer = make_scorer(custom_metric)
```

### needs_threshold and needs_proba

For metrics that need probability estimates or decision thresholds:

```python
from sklearn.metrics import roc_auc_score

# needs_proba=True: pass y_pred as probabilities
auc_scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr')

# needs_threshold=True: pass raw decision function (e.g., SVM)
# Used with estimators that have decision_function
```

### response_method

Specify which method of the estimator to use for predictions.

```python
# For classifiers: 'predict', 'predict_proba', 'decision_function'
scorer = make_scorer(roc_auc_score, response_method='predict_proba')
```

---

## 4. Multi-Metric Evaluation

### cross_validate

`cross_validate` supports multiple metrics in a single run, avoiding redundant fitting.

```python
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score

scoring = ['accuracy', 'f1_macro', 'recall_macro']

scores = cross_validate(
    SVC(),
    X, y,
    cv=5,
    scoring=scoring,
    return_train_score=True
)

print(scores['test_accuracy'])
print(scores['test_f1_macro'])
print(scores['test_recall_macro'])
```

### Dict of Scorers

```python
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score, average='macro'),
    'custom': make_scorer(custom_metric)
}
scores = cross_validate(estimator, X, y, cv=5, scoring=scoring)
```

### Return Values

| Key | Content |
|-----|---------|
| `test_<metric>` | Test scores per fold |
| `train_<metric>` | Train scores (if `return_train_score=True`) |
| `fit_time` | Fit time per fold |
| `score_time` | Score time per fold |

---

## 5. Refit Strategies

In `GridSearchCV` and similar, **refit** determines how to select the best model when multiple metrics are used.

### refit with Single Metric

```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    SVC(),
    param_grid={'C': [0.1, 1, 10], 'gamma': [0.01, 0.1]},
    cv=5,
    scoring='accuracy',
    refit=True  # Refit on full data with best params (default)
)
grid.fit(X_train, y_train)
# grid.best_estimator_ is refit on full X_train, y_train
```

### refit with Multiple Metrics

When `scoring` is a dict, `refit` must be a key from that dict (or `True` to use the first).

```python
scoring = {'accuracy': 'accuracy', 'f1': 'f1_macro'}
grid = GridSearchCV(
    SVC(),
    param_grid={'C': [0.1, 1, 10]},
    cv=5,
    scoring=scoring,
    refit='f1'  # Refit using best params for f1
)
grid.fit(X_train, y_train)
# Best params are those that maximize f1_macro
```

### refit=False

Use `refit=False` when you only need the CV results and do not need a refit estimator.

---

## 6. Evaluation Best Practices

### Data Leakage

**Data leakage** occurs when information from outside the training set influences the model. It inflates performance estimates.

| Source | Prevention |
|--------|------------|
| Scaling/normalization | Fit on train only; transform train and test |
| Feature selection | Perform inside CV loop |
| Imputation | Fit on train; transform test |
| Target encoding | Encode using train statistics only |

```python
# Wrong: fit on full data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
scores = cross_val_score(model, X_scaled, y, cv=5)  # Leakage

# Correct: use pipeline so scaling is inside CV
from sklearn.pipeline import Pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
scores = cross_val_score(pipe, X, y, cv=5)
```

### Class Imbalance Awareness

With imbalanced classes, accuracy can be misleading. Use metrics that account for imbalance.

| Metric | When to Use |
|--------|-------------|
| `precision`, `recall`, `f1` | Imbalanced; focus on positive class |
| `f1_macro`, `f1_weighted` | Multi-class imbalance |
| `roc_auc` | Threshold-independent; good for ranking |
| `average_precision` | Imbalanced binary; precision-recall tradeoff |

```python
from sklearn.metrics import classification_report, confusion_matrix

# Prefer stratified CV for imbalanced data
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
```

### Overfitting to the Test Set

Repeated use of the test set for model selection leads to **overfitting to the test set**. The test set should be used only for final evaluation.

| Practice | Purpose |
|----------|---------|
| Hold out test set | Final evaluation only |
| Use validation/CV for tuning | Model and hyperparameter selection |
| Nested CV | Unbiased performance estimate with tuning |

```python
# Nested CV: outer loop for evaluation, inner for tuning
from sklearn.model_selection import cross_val_score, GridSearchCV

inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(estimator, param_grid, cv=inner_cv, scoring='f1_macro')
scores = cross_val_score(grid, X, y, cv=outer_cv, scoring='f1_macro')
# Each outer fold: grid search on inner folds, evaluate on outer test fold
```

### Summary Checklist

| Practice | Description |
|----------|-------------|
| Use pipelines | Avoid leakage from preprocessing |
| Stratified splits | Preserve class distribution in CV |
| Multiple metrics | Capture different aspects of performance |
| Nested CV | When reporting performance with tuning |
| Single final test | Reserve for one-time final evaluation |
