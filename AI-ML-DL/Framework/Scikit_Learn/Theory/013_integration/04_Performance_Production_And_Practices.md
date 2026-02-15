# Performance, Production, and Best Practices

---

## Table of Contents

- [Overview](#overview)
- [Performance Optimization](#performance-optimization)
- [n_jobs Parallelism](#n_jobs-parallelism)
- [warm_start and Incremental Training](#warm_start-and-incremental-training)
- [Large Dataset Strategies](#large-dataset-strategies)
- [partial_fit and Incremental Learning](#partial_fit-and-incremental-learning)
- [Out-of-Core with HashingVectorizer](#out-of-core-with-hashingvectorizer)
- [Comparison with Other Frameworks](#comparison-with-other-frameworks)
- [XGBoost, LightGBM, CatBoost](#xgboost-lightgbm-catboost)
- [Production Patterns](#production-patterns)
- [Validation and Monitoring](#validation-and-monitoring)
- [Model Versioning](#model-versioning)
- [Best Practices Guide](#best-practices-guide)
- [Preprocessing and Selection](#preprocessing-and-selection)
- [Evaluation and Metrics](#evaluation-and-metrics)
- [Summary](#summary)

---

## Overview

This document covers **performance optimization** (n_jobs, warm_start), **large dataset strategies** (partial_fit, out-of-core), **framework comparison** (XGBoost, LightGBM, CatBoost), **production patterns** (validation, monitoring, versioning), and a **comprehensive best practices guide**.

---

## Performance Optimization

### Key Levers

- **Parallelism**: n_jobs for cross-validation, grid search, and ensemble fitting
- **Algorithm choice**: Sparse-friendly models for text; tree ensembles for tabular
- **Pipeline efficiency**: Avoid redundant transforms; cache when possible
- **Warm start**: Reuse previous fit for incremental training

---

## n_jobs Parallelism

### Where to Use n_jobs

| Component | n_jobs | Effect |
|-----------|--------|--------|
| **cross_val_score** | -1 | Parallel CV folds |
| **GridSearchCV** | -1 | Parallel parameter combinations |
| **RandomForest** | -1 | Parallel tree building |
| **SVC** | -1 | Parallel binary subproblems (libsvm) |

### Example

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, X, y, cv=5, n_jobs=-1)
# Uses all CPU cores for CV
```

### Caveats

- **n_jobs=-1** uses all cores; may contend with other processes
- Nested parallelism (e.g., GridSearchCV with n_jobs and RF with n_jobs) can oversubscribe; set inner n_jobs=1 sometimes
- **threading** vs **multiprocessing**: sklearn uses joblib; avoid large data copy overhead

---

## warm_start and Incremental Training

### Supported Estimators

**GradientBoostingClassifier**, **GradientBoostingRegressor** support **warm_start**. Set `warm_start=True`, then increase `n_estimators` and call `fit` again to add more trees without retraining from scratch.

```python
gb = GradientBoostingClassifier(n_estimators=10, warm_start=True)
gb.fit(X_train, y_train)
gb.n_estimators = 20
gb.fit(X_train, y_train)
# Now has 20 trees; first 10 reused
```

### Use Case

- Hyperparameter search over n_estimators with early stopping
- Incremental model improvement without full retrain

---

## Large Dataset Strategies

### When Data Does Not Fit in Memory

1. **Incremental learning**: partial_fit with mini-batches
2. **Out-of-core preprocessing**: HashingVectorizer (no fit), IncrementalPCA
3. **Sampling**: Train on a representative subset
4. **External libraries**: XGBoost, LightGBM with native large-data support

---

## partial_fit and Incremental Learning

### Estimators Supporting partial_fit

| Estimator | Use Case |
|-----------|----------|
| **SGDClassifier** | Linear classification |
| **SGDRegressor** | Linear regression |
| **MiniBatchKMeans** | Clustering (via batch fit) |
| **IncrementalPCA** | Dimensionality reduction |
| **MultinomialNB** | Naive Bayes (partial_fit) |

### Pattern

```python
sgd = SGDClassifier(max_iter=1, random_state=42)
for i in range(0, len(X), batch_size):
    X_batch = X[i:i+batch_size]
    y_batch = y[i:i+batch_size]
    sgd.partial_fit(X_batch, y_batch, classes=np.unique(y))
```

### Important

- **classes** must be passed on first call for classifiers
- **max_iter=1** per partial_fit call (one pass over batch)
- Shuffle batches for better convergence

---

## Out-of-Core with HashingVectorizer

### HashingVectorizer

**HashingVectorizer** does not require fit; it transforms text to a fixed-size sparse matrix via hashing. Suitable for streaming or very large corpora.

```python
from sklearn.feature_extraction.text import HashingVectorizer

hv = HashingVectorizer(n_features=2**18)
# No fit; transform only
X_chunk = hv.transform(text_chunk)
sgd.partial_fit(X_chunk, y_chunk, classes=[0, 1])
```

### Trade-offs

- **Pros**: No vocabulary storage; constant memory; streaming
- **Cons**: No inverse mapping (feature names); possible collisions

---

## Comparison with Other Frameworks

### When to Use sklearn vs Gradient Boosting Libraries

| Criterion | sklearn | XGBoost/LightGBM/CatBoost |
|-----------|---------|---------------------------|
| **Ease of use** | Native, well documented | Extra dependency |
| **Speed** | Slower for large data | Faster, optimized |
| **Categorical** | Requires encoding | Native support (LightGBM, CatBoost) |
| **Sparse** | Limited | Good support |
| **Pipeline** | Full integration | sklearn API compatible |

---

## XGBoost, LightGBM, CatBoost

### sklearn-Compatible API

All three provide **fit**, **predict**, **predict_proba** and work in **Pipeline** and **GridSearchCV**:

```python
import xgboost as xgb
clf = xgb.XGBClassifier(n_estimators=100)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
```

### Feature Comparison

| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| **Categorical** | One-hot | Native | Native |
| **Sparse** | Yes | Yes | Yes |
| **GPU** | Yes | Yes | Yes |
| **Missing values** | Learned | Learned | Learned |
| **Default quality** | Good | Good | Good |

### Migration

Replace `GradientBoostingClassifier` with `XGBClassifier` or `LGBMClassifier` for similar hyperparameters (n_estimators, max_depth, learning_rate) and often better performance.

---

## Production Patterns

### Validation

- **Input validation**: Check shape, dtype, NaN/Inf before predict
- **Feature count**: Ensure test data has same number of features as training
- **Schema**: Validate column names and types for DataFrame input

```python
from sklearn.utils.validation import check_array

def validate_input(X, n_features):
    X = check_array(X, dtype=np.float64)
    if X.shape[1] != n_features:
        raise ValueError(f"Expected {n_features} features")
    return X
```

### Monitoring

- **Latency**: Track predict time (p50, p99)
- **Throughput**: Requests per second
- **Prediction distribution**: Monitor class/score distribution for drift
- **Data drift**: Compare input distribution to training

### Graceful Degradation

- Catch validation errors; return meaningful HTTP status or error message
- Fallback to cached/default prediction when appropriate
- Log failures for debugging

---

## Model Versioning

### Storage

- Save model with **joblib** or **sklearn.model_export**
- Include **metadata**: version, sklearn version, training date, config

```python
import joblib

artifact = {
    "model": pipe,
    "version": "v1.2",
    "sklearn_version": sklearn.__version__,
    "n_features": 4
}
joblib.dump(artifact, "model_v1.2.joblib")
```

### Loading

```python
artifact = joblib.load("model_v1.2.joblib")
pipe = artifact["model"]
n_features = artifact["n_features"]
```

---

## Best Practices Guide

### Preprocessing and Selection

- **Pipeline**: Always use Pipeline for preprocessing + model to avoid leakage
- **Fit on train only**: Transform test with scaler/encoder fitted on train
- **Feature selection**: Use SelectKBest or RFE inside pipeline; fit on train

### Model Selection

- **Problem type**: Classification vs regression; binary vs multiclass
- **Imbalanced**: Use class_weight, SMOTE, or precision-recall metrics
- **High-dimensional**: Consider L1/L2, PCA, or tree-based feature importance

### Evaluation

- **Cross-validation**: Report mean and std; use stratified CV for classification
- **Metrics**: Accuracy for balanced; precision/recall/F1 for imbalanced
- **Multiple metrics**: Log precision, recall, AUC, etc. for full picture

### Reproducibility

- **random_state**: Set on all estimators and splitters
- **Version pinning**: Pin sklearn, numpy, scipy versions
- **Experiment log**: Record config, data hash, and environment

---

## Evaluation and Metrics

### Classification

| Scenario | Primary Metric |
|----------|----------------|
| Balanced | Accuracy |
| Imbalanced | Precision, Recall, F1, AUC-PR |
| Multiclass | Macro/micro F1 |
| Ranking | AUC-ROC |

### Regression

- **RMSE**, **MAE** for general use
- **R2** for variance explained
- **MAPE** when relative error matters

---

## Summary

- **n_jobs**: Use for CV, GridSearch, and ensemble fitting
- **warm_start**: Incremental trees for GradientBoosting
- **partial_fit**: SGD, IncrementalPCA, MiniBatchKMeans for large data
- **HashingVectorizer**: Out-of-core text; no fit
- **XGBoost/LightGBM/CatBoost**: Faster, categorical support; sklearn-compatible
- **Production**: Validate input, monitor latency and drift, version models
- **Best practices**: Pipeline, CV, random_state, appropriate metrics
