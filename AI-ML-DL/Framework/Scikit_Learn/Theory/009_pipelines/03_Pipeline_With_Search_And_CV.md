# Pipeline with Search and Cross-Validation

---

## Table of Contents

- [Overview](#overview)
- [GridSearchCV with Pipeline](#gridsearchcv-with-pipeline)
- [RandomizedSearchCV with Pipeline](#randomizedsearchcv-with-pipeline)
- [Parameter Grids for Pipelines](#parameter-grids-for-pipelines)
- [Nested Pipelines](#nested-pipelines)
- [Cross-Validation with Pipelines](#cross-validation-with-pipelines)
- [Refit and Best Estimator](#refit-and-best-estimator)
- [Multi-Metric Search](#multi-metric-search)
- [Best Practices](#best-practices)

---

## Overview

**GridSearchCV** and **RandomizedSearchCV** integrate with **Pipeline** to tune hyperparameters across preprocessing and model steps. The pipeline ensures that preprocessing is fitted only on training folds, avoiding data leakage. **cross_val_score** and **cross_validate** work the same way.

| Concept | Purpose |
|---------|---------|
| **GridSearchCV** | Exhaustive search over parameter grid |
| **RandomizedSearchCV** | Random sampling of parameter space |
| **step__param** | Access nested pipeline parameters |
| **refit** | Refit best estimator on full data |

---

## GridSearchCV with Pipeline

**GridSearchCV** fits the pipeline for each parameter combination and evaluates with cross-validation. Use **step__param** for parameter names.

### Basic Usage

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])

param_grid = {
    "scaler__with_mean": [True, False],
    "scaler__with_std": [True, False],
    "clf__C": [0.1, 1.0, 10.0],
}

gs = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy")
gs.fit(X_train, y_train)
print(gs.best_params_)
print(gs.best_score_)
y_pred = gs.predict(X_test)
```

### Key Points

- **gs.best_estimator_** is the pipeline refit on full training data (when refit=True)
- **gs.cv_results_** contains all fold scores and parameters
- **gs.predict** uses the best estimator

---

## RandomizedSearchCV with Pipeline

**RandomizedSearchCV** samples a fixed number of parameter combinations. More efficient for large parameter spaces.

### Usage

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

param_distributions = {
    "clf__C": loguniform(1e-3, 1e3),
    "clf__max_iter": [500, 1000, 2000],
}

rs = RandomizedSearchCV(pipe, param_distributions, n_iter=20, cv=5, random_state=42)
rs.fit(X_train, y_train)
```

---

## Parameter Grids for Pipelines

### Nested Parameters

Use double underscore for nested steps:

```python
param_grid = {
    "preprocessor__num__imputer__strategy": ["mean", "median"],
    "preprocessor__num__scaler__with_std": [True, False],
    "clf__C": [0.1, 1.0, 10.0],
}
```

### Conditional Parameters

Some parameters apply only when others have certain values. Use **ParameterGrid** or filter manually. **GridSearchCV** will skip invalid combinations (estimator may raise).

---

## Nested Pipelines

Pipelines can contain other pipelines. Parameter access uses the full path.

```python
num_pipe = Pipeline([
    ("imputer", SimpleImputer()),
    ("scaler", StandardScaler()),
])
preprocessor = ColumnTransformer([
    ("num", num_pipe, [0, 1, 2]),
])
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression()),
])

param_grid = {
    "preprocessor__num__imputer__strategy": ["mean", "median"],
    "clf__C": [0.1, 1.0],
}
```

---

## Cross-Validation with Pipelines

**cross_val_score** and **cross_validate** work with pipelines. Each CV fold fits the entire pipeline on the training fold and evaluates on the test fold.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
print(scores.mean(), scores.std())
```

### Why Pipeline Matters

Without a pipeline, fitting the scaler on the full dataset before CV would leak test information. With a pipeline, each fold fits the scaler only on that fold's training data.

---

## Refit and Best Estimator

**GridSearchCV** refits the best estimator on the full training data by default (**refit=True**). This gives the final model for prediction.

### refit Parameter

- **refit=True** (default): Refit best estimator on full data
- **refit=False**: best_estimator_ is from the best CV fold (not refit)
- **refit="metric_name"**: When using multiple metrics, refit based on that metric

### Accessing Results

```python
gs.best_estimator_   # Best pipeline (refit)
gs.best_params_      # Best parameter combination
gs.best_score_       # Mean CV score of best params
gs.cv_results_       # Full results dict
```

---

## Multi-Metric Search

**GridSearchCV** can use multiple scoring metrics. Use **refit** to choose which metric selects the best estimator.

```python
gs = GridSearchCV(
    pipe, param_grid, cv=5,
    scoring={"accuracy": "accuracy", "f1": "f1_macro"},
    refit="f1"
)
gs.fit(X_train, y_train)
# Best params optimize f1
# gs.cv_results_ has mean_test_accuracy, mean_test_f1, etc.
```

---

## Best Practices

| Practice | Reason |
|----------|--------|
| Use **Pipeline** with GridSearchCV | Correct CV, no leakage |
| Use **step__param** for parameter names | Correct nesting |
| Use **n_jobs=-1** for parallel search | Speed |
| Use **RandomizedSearchCV** for large spaces | Efficiency |
| Inspect **cv_results_** for analysis | Understand parameter effects |
