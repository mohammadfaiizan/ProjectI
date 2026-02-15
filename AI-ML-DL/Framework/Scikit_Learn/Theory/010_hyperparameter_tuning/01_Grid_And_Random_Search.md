# Grid and Random Search

---

## Table of Contents

- [Overview](#overview)
- [GridSearchCV](#gridsearchcv)
- [RandomizedSearchCV](#randomizedsearchcv)
- [HalvingGridSearchCV](#halvinggridsearchcv)
- [HalvingRandomSearchCV](#halvingrandomsearchcv)
- [Comparison and When to Use](#comparison-and-when-to-use)

---

## Overview

**Hyperparameter tuning** finds optimal model settings that cannot be learned from data. Scikit-learn provides several search strategies: **GridSearchCV** exhaustively evaluates all combinations, **RandomizedSearchCV** samples randomly, and **HalvingGridSearchCV** and **HalvingRandomSearchCV** use successive halving to reduce computational cost.

---

## GridSearchCV

### Core Parameters

**GridSearchCV** evaluates every combination in **param_grid** using cross-validation. The best parameters are selected by **scoring**, and **refit** retrains the best model on the full training data.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {
    "C": [0.1, 1.0, 10.0],
    "gamma": ["scale", "auto", 0.01],
    "kernel": ["rbf", "linear"],
}
grid = GridSearchCV(SVC(), param_grid, cv=5, scoring="accuracy", refit=True)
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
```

| Parameter | Description |
|-----------|-------------|
| **param_grid** | Dict of param name -> list of values |
| **cv** | Number of folds or CV splitter |
| **scoring** | Metric to optimize |
| **refit** | Retrain best model on full data |

### best_params_ and best_score_

After fitting, **best_params_** holds the best parameter combination and **best_score_** the mean cross-validation score. **best_estimator_** is the refitted model (when `refit=True`).

```python
best_params = grid.best_params_
best_score = grid.best_score_
best_model = grid.best_estimator_
```

### Limitations

Grid search is exhaustive: with many parameters and values, the number of combinations grows exponentially. Use **RandomizedSearchCV** or halving variants for large search spaces.

---

## RandomizedSearchCV

### Core Parameters

**RandomizedSearchCV** samples **n_iter** parameter combinations from **param_distributions** instead of evaluating all. Use **scipy.stats** distributions for continuous or discrete ranges.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, loguniform

param_distributions = {
    "n_estimators": randint(10, 200),
    "max_depth": randint(2, 20),
    "min_samples_split": randint(2, 20),
}
random_search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    random_state=42,
)
random_search.fit(X_train, y_train)
```

| Parameter | Description |
|-----------|-------------|
| **param_distributions** | Dict of param name -> distribution |
| **n_iter** | Number of parameter settings to sample |
| **random_state** | Reproducibility |

### Advantages

Randomized search often finds good parameters with far fewer evaluations than grid search. It scales better to high-dimensional parameter spaces.

---

## HalvingGridSearchCV

### Successive Halving

**HalvingGridSearchCV** uses **successive halving**: start with many candidates on a small subset of data, keep the best half, double the data, repeat. Fewer fits than full grid search.

```python
from sklearn.model_selection import HalvingGridSearchCV

halving_grid = HalvingGridSearchCV(
    RandomForestClassifier(),
    param_grid=param_grid,
    factor=3,
    resource="n_samples",
    min_resources="exhaust",
    cv=3,
)
halving_grid.fit(X_train, y_train)
```

| Parameter | Description |
|-----------|-------------|
| **factor** | Fraction of candidates kept each iteration (1/factor) |
| **resource** | Resource to increase ("n_samples" or "n_iter") |
| **min_resources** | Initial resource ("smallest", "exhaust", or int) |

### How It Works

1. First iteration: all candidates on `min_resources` samples.
2. Keep top 1/factor candidates.
3. Next iteration: surviving candidates on more samples.
4. Repeat until one candidate remains or resources exhausted.

---

## HalvingRandomSearchCV

### Halving + Random Sampling

**HalvingRandomSearchCV** combines successive halving with random parameter sampling. Use **n_candidates** to control how many random combinations to try initially.

```python
from sklearn.model_selection import HalvingRandomSearchCV

halving_random = HalvingRandomSearchCV(
    RandomForestClassifier(),
    param_distributions=param_distributions,
    n_candidates="exhaust",
    factor=3,
    resource="n_samples",
    min_resources="smallest",
    cv=3,
)
halving_random.fit(X_train, y_train)
```

| Parameter | Description |
|-----------|-------------|
| **n_candidates** | Initial number of random candidates ("exhaust" or int) |
| **factor** | Same as HalvingGridSearchCV |
| **resource** | "n_samples" or "n_iter" |

### n_resources_ and n_candidates_

After fitting, **n_resources_** lists samples per iteration and **n_candidates_** lists candidates per iteration. **n_iterations_** gives the total number of halving rounds.

---

## Comparison and When to Use

| Method | Use Case |
|--------|----------|
| **GridSearchCV** | Small param grids, exhaustive search |
| **RandomizedSearchCV** | Large spaces, limited budget |
| **HalvingGridSearchCV** | Medium grids, faster than full grid |
| **HalvingRandomSearchCV** | Large spaces, faster than random search |

### Computational Cost

- Grid: O(n_combinations × cv_folds × fit_cost)
- Random: O(n_iter × cv_folds × fit_cost)
- Halving: Early iterations cheap (few samples), late iterations expensive but fewer candidates

---
