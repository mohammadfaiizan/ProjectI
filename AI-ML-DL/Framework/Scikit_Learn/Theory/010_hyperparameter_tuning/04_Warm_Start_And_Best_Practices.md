# Warm Start and Best Practices

---

## Table of Contents

- [Warm Starting](#warm-starting)
- [Tuning Strategies](#tuning-strategies)
- [Overfitting to Validation](#overfitting-to-validation)
- [Nested Cross-Validation](#nested-cross-validation)
- [Best Practices Summary](#best-practices-summary)

---

## Warm Starting

### warm_start Parameter

**warm_start** allows incremental training: calling **fit** again continues from the previous model instead of reinitializing. Supported by **RandomForestClassifier**, **GradientBoostingClassifier**, and others.

```python
rf = RandomForestClassifier(n_estimators=10, warm_start=True)
rf.fit(X_train, y_train)
rf.n_estimators = 20
rf.fit(X_train, y_train)
```

### Use Case

Use **warm_start** when you want to iteratively increase a resource (e.g., **n_estimators**) without retraining from scratch. Each **fit** adds more trees or iterations.

### In Search

**GridSearchCV** and **RandomizedSearchCV** treat each parameter combination independently. **warm_start** does not carry state between candidates. It helps when manually iterating over **n_estimators** on the same model.

### Manual Warm Start Loop

```python
best_score = 0
model = RandomForestClassifier(n_estimators=5, warm_start=True)
for n in [5, 10, 20, 40, 80]:
    model.n_estimators = n
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    if score > best_score:
        best_score = score
        best_n = n
```

---

## Tuning Strategies

### Coarse-to-Fine Grid

Start with a coarse grid to locate promising regions, then refine:

```python
# Coarse
param_coarse = {"C": [0.1, 1.0, 100.0]}
grid1 = GridSearchCV(SVC(), param_coarse, cv=5)
grid1.fit(X_train, y_train)
best_c = grid1.best_params_["C"]

# Fine around best_c
if best_c == 0.1:
    param_fine = {"C": [0.01, 0.05, 0.1, 0.5]}
elif best_c == 100.0:
    param_fine = {"C": [50.0, 100.0, 200.0]}
else:
    param_fine = {"C": [0.5, 1.0, 2.0]}
grid2 = GridSearchCV(SVC(), param_fine, cv=5)
grid2.fit(X_train, y_train)
```

### Random First, Grid Second

Use **RandomizedSearchCV** to explore broadly, then **GridSearchCV** in the best region.

### Halving for Speed

Use **HalvingGridSearchCV** or **HalvingRandomSearchCV** when the full grid or many random samples would be too slow.

### Parameter Importance

Not all parameters matter equally. Focus tuning on the most impactful (e.g., **C**, **gamma** for SVM; **n_estimators**, **max_depth** for trees).

---

## Overfitting to Validation

### The Problem

Selecting hyperparameters based on validation performance can lead to **overfitting to the validation set**. The reported CV score may be optimistically biased.

### Mitigations

1. **Hold out a test set** for final evaluation; never use it for tuning.
2. **Nested CV** for unbiased performance estimates when reporting results.
3. **Fewer parameters** or **coarser grids** to reduce the search space.
4. **Regularization** in the model to reduce overfitting.

### Train vs Test Gap in cv_results_

When **return_train_score=True**, compare **mean_train_score** and **mean_test_score**. A large gap indicates overfitting. Consider simpler models or stronger regularization.

---

## Nested Cross-Validation

### Structure

**Nested CV** uses an outer loop for performance estimation and an inner loop for hyperparameter selection. The inner loop never sees the outer test fold.

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=43)
outer_scores = []

for train_idx, test_idx in outer_cv.split(X, y):
    X_in, X_out = X[train_idx], X[test_idx]
    y_in, y_out = y[train_idx], y[test_idx]
    grid = GridSearchCV(SVC(), param_grid, cv=inner_cv)
    grid.fit(X_in, y_in)
    score = grid.score(X_out, y_out)
    outer_scores.append(score)

print(np.mean(outer_scores), np.std(outer_scores))
```

### When to Use

Use nested CV when you need an **unbiased estimate** of performance that accounts for hyperparameter selection. It is computationally expensive (outer_folds × inner_folds × n_combinations).

### Single Split vs Nested CV

- **Single split**: train/val for tuning, test for final eval. Simpler, but variance in the estimate.
- **Nested CV**: More robust estimate, higher cost.

---

## Best Practices Summary

### Search Method Selection

| Scenario | Recommendation |
|----------|----------------|
| Small grid (< 100 combos) | **GridSearchCV** |
| Large continuous space | **RandomizedSearchCV** or **BayesSearchCV** |
| Need speed | **HalvingGridSearchCV** or **HalvingRandomSearchCV** |
| Advanced optimization | **Optuna** or **scikit-optimize** |

### Scoring and CV

- Use **StratifiedKFold** for classification.
- Use **GroupKFold** when data has groups.
- Choose **scoring** to match the business objective (e.g., **f1_macro** for imbalanced data).

### Reporting Performance

- Use **nested CV** when reporting final performance in papers or reports.
- Keep a held-out test set for final validation when possible.
- Report mean and standard deviation of scores.

### Computational Budget

- Start with **n_iter=20** for RandomizedSearchCV; increase if budget allows.
- Use **n_jobs=-1** for parallelization.
- Consider **Halving*** when the full search is too slow.

### Avoiding Overfitting

- Prefer simpler models when performance is similar.
- Use **return_train_score=True** to check for overfitting.
- Apply proper preprocessing (e.g., scaling) inside a **Pipeline** so it is fit only on training folds.

---
