# Persistence, Debugging, and Practices

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Persistence](#pipeline-persistence)
- [joblib dump and load](#joblib-dump-and-load)
- [Compression and Compatibility](#compression-and-compatibility)
- [Pipeline Debugging](#pipeline-debugging)
- [Step-by-Step Inspection](#step-by-step-inspection)
- [Common Issues](#common-issues)
- [Best Practices](#best-practices)
- [Design Patterns](#design-patterns)
- [Summary](#summary)

---

## Overview

**Persistence** of pipelines is done with **joblib** (dump/load). **Debugging** pipelines involves inspecting intermediate outputs, verifying step order, and using **set_config** for verbose output. **Best practices** cover design patterns, avoiding data leakage, and structuring pipelines for maintainability.

---

## Pipeline Persistence

Fitted pipelines can be saved to disk and loaded later for deployment or sharing. Scikit-learn recommends **joblib** for this purpose (previously `sklearn.externals.joblib`, now the standalone **joblib** package).

### Why joblib

- Optimized for NumPy arrays and scikit-learn estimators
- Handles large arrays efficiently
- Supports compression
- Compatible with pipeline nesting and custom transformers (if picklable)

---

## joblib dump and load

### Basic Usage

```python
import joblib

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=500)),
])
pipe.fit(X_train, y_train)

joblib.dump(pipe, "model.joblib")
pipe_loaded = joblib.load("model.joblib")
y_pred = pipe_loaded.predict(X_test)
```

### File Paths

Pass a file path string or a file-like object:

```python
joblib.dump(pipe, "/path/to/model.joblib")
pipe_loaded = joblib.load("/path/to/model.joblib")
```

### Nested Pipelines

Nested pipelines, **ColumnTransformer**, and **FeatureUnion** are serialized as part of the pipeline. All components must be picklable.

```python
nested = Pipeline([
    ("preprocess", Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=5)),
    ])),
    ("clf", LogisticRegression()),
])
joblib.dump(nested, "nested_model.joblib")
```

---

## Compression and Compatibility

### compress Parameter

```python
joblib.dump(pipe, "model.joblib", compress=3)
```

- **compress=0**: No compression
- **compress=1-9**: Higher values mean more compression, slower save/load
- **compress=True**: Default compression level

### Version Compatibility

- Joblib format may change between versions
- For long-term storage, document the scikit-learn and joblib versions used
- Consider using **ONNX** or **PMML** for cross-framework deployment

---

## Pipeline Debugging

### set_config(print_changed_only=False)

Show full configuration of all steps:

```python
from sklearn import set_config

set_config(print_changed_only=False)
print(pipe)
```

### Inspect Step Order

Verify that preprocessing steps come before the final estimator:

```python
for i, (name, step) in enumerate(pipe.steps):
    print(f"{i}: {name} -> {type(step).__name__}")
```

### Check get_params

List all parameter names for GridSearchCV:

```python
params = pipe.get_params()
for k in sorted(params.keys()):
    if "__" in k:
        print(k)
```

---

## Step-by-Step Inspection

### Transform Up to a Step

Use slicing to get the preprocessing pipeline (all steps except the last):

```python
preprocess = pipe[:-1]
X_transformed = preprocess.transform(X_train)
print(X_transformed.shape)
print(np.isfinite(X_transformed).all())
```

### Manual Step-by-Step

Fit the pipeline, then manually apply each step to inspect outputs:

```python
pipe.fit(X_train, y_train)
X_scaled = pipe.named_steps["scaler"].transform(X_train)
X_pca = pipe.named_steps["pca"].transform(X_scaled)
print("After scaler:", X_scaled.shape, X_scaled.mean(), X_scaled.std())
print("After PCA:", X_pca.shape)
```

### Check for NaN/Inf

Intermediate outputs may contain NaN or Inf if data or transformers are misconfigured:

```python
X_transformed = pipe[:-1].transform(X_train)
assert np.isfinite(X_transformed).all(), "NaN or Inf in pipeline output"
```

---

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| **Param not found** | Wrong param string for GridSearchCV | Use `stepname__param`; check `get_params().keys()` |
| **Shape mismatch** | Transformer changes number of features | Inspect output of each step; ensure ColumnTransformer columns match |
| **Data leakage** | Preprocessing fit on full data before CV | Use pipeline in cross_val_score/GridSearchCV |
| **Pickle error** | Custom transformer not picklable | Avoid lambdas, file handles; use joblib-friendly types |
| **Different results after load** | Random state not fixed | Set `random_state` in all stochastic components |

---

## Best Practices

### One Pipeline, One Model

Treat the pipeline as the unit of deployment. Include all preprocessing and the model. Do not save the model alone and expect to apply preprocessing separately in production.

### Preprocessing Before Model

Always order steps as: preprocessing (imputation, encoding, scaling, feature selection) then the final estimator.

### Use Explicit Names for Tuning

When using GridSearchCV, prefer **Pipeline** with explicit step names over **make_pipeline** for clarity in param grids.

### Persist the Full Pipeline

```python
joblib.dump(pipe, "model.joblib")
# In production:
pipe = joblib.load("model.joblib")
y_pred = pipe.predict(X_new)
```

### Keep Pipelines in CV

Always use `cross_val_score(pipe, X, y)` or `GridSearchCV(pipe, param_grid)` rather than fitting preprocessing outside the CV loop.

---

## Design Patterns

### Numeric-Only Pipeline

```python
pipe = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=500),
)
```

### Mixed Data with ColumnTransformer

```python
ct = make_column_transformer(
    (make_pipeline(SimpleImputer(), StandardScaler()), numeric_cols),
    (OneHotEncoder(drop="first"), categorical_cols),
)
pipe = make_pipeline(ct, LogisticRegression(max_iter=500))
```

### Pipeline with Feature Selection

```python
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("select", SelectKBest(score_func=f_classif, k=10)),
    ("clf", LogisticRegression(max_iter=500)),
])
```

### Nested Preprocessing

```python
preprocess = Pipeline([
    ("impute", SimpleImputer()),
    ("scale", StandardScaler()),
    ("pca", PCA(n_components=0.95)),
])
pipe = Pipeline([
    ("preprocess", preprocess),
    ("clf", LogisticRegression(max_iter=500)),
])
```

---

## Summary

- Use **joblib.dump** and **joblib.load** for pipeline persistence
- **compress** reduces file size at the cost of speed
- Debug with **pipe[:-1].transform(X)**, **named_steps**, and **get_params**
- Avoid data leakage by keeping preprocessing inside the pipeline for CV
- One pipeline = one reproducible model; persist and deploy the full pipeline
- Use **ColumnTransformer** for mixed data, **Pipeline** for sequential steps
