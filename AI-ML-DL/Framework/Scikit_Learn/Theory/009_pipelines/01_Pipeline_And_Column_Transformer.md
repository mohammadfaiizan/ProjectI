# Pipeline and ColumnTransformer

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Basics](#pipeline-basics)
- [Pipeline Structure](#pipeline-structure)
- [fit, transform, predict](#fit-transform-predict)
- [ColumnTransformer](#columntransformer)
- [make_column_transformer](#make_column_transformer)
- [Combining Pipeline and ColumnTransformer](#combining-pipeline-and-columntransformer)
- [Parameter Access](#parameter-access)
- [Best Practices](#best-practices)

---

## Overview

**Pipeline** chains multiple estimators (transformers and a final estimator) into a single object. It ensures preprocessing is fitted only on training data and applied consistently, preventing **data leakage**. **ColumnTransformer** applies different transformers to different column subsets, essential for mixed numeric and categorical data.

| Component | Purpose |
|-----------|---------|
| **Pipeline** | Chain fit/transform/predict |
| **ColumnTransformer** | Per-column preprocessing |
| **make_column_transformer** | Convenience for ColumnTransformer |
| **make_column_selector** | Select columns by type or name |

---

## Pipeline Basics

**Pipeline** applies a sequence of steps. Each step is a (name, estimator) tuple. The last step can be a transformer (has transform) or a predictor (has predict).

### Basic Usage

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression()),
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

### Key Behaviors

- **fit**: Calls fit on each step; for transformers, fit_transform is passed to the next step
- **predict**: Applies transform through all steps except the last, then predict on the final estimator
- Steps are accessible via **named_steps** or indexing: `pipe["scaler"]`, `pipe.named_steps["clf"]`

---

## Pipeline Structure

### Steps

Each step is a tuple `(name, estimator)`. Names must be unique and are used for parameter access in GridSearchCV.

```python
pipe.steps
# [("scaler", StandardScaler()), ("clf", LogisticRegression())]
```

### Slicing

Pipeline supports slicing: `pipe[:2]` returns a new pipeline with the first two steps. Useful for inspecting intermediate outputs.

```python
pipe_preprocess = pipe[:-1]
X_transformed = pipe_preprocess.transform(X)
```

---

## fit, transform, predict

### fit

**fit** trains each step in sequence. Transformers are fitted on the output of the previous step. The final estimator is fitted on the transformed data and targets.

### transform

If the final step has **transform**, the pipeline has **transform**. Otherwise, use **predict** for the final step.

### predict

**predict** applies **transform** to all but the last step, then **predict** on the last step.

```python
y_pred = pipe.predict(X_test)
# Equivalent to: pipe["clf"].predict(pipe["scaler"].transform(X_test))
```

---

## ColumnTransformer

**ColumnTransformer** applies different transformers to different column subsets. Outputs are concatenated horizontally.

### Basic Usage

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

ct = ColumnTransformer([
    ("num", StandardScaler(), [0, 1, 2]),
    ("cat", OneHotEncoder(), [3, 4]),
])
X_transformed = ct.fit_transform(X)
```

### remainder

- **remainder="drop"** (default): Drop columns not specified
- **remainder="passthrough"**: Pass through unchanged

```python
ct = ColumnTransformer(
    [("num", StandardScaler(), [0, 1])],
    remainder="passthrough"
)
```

### sparse_threshold

Control output format: sparse matrix vs dense array. Default 0.3.

---

## make_column_transformer

**make_column_transformer** simplifies ColumnTransformer creation. Automatically generates step names.

```python
from sklearn.compose import make_column_transformer

ct = make_column_transformer(
    (StandardScaler(), [0, 1, 2]),
    (OneHotEncoder(), [3, 4]),
)
```

### make_column_selector

Select columns by dtype or pattern:

```python
from sklearn.compose import make_column_selector

ct = make_column_transformer(
    (StandardScaler(), make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(), make_column_selector(dtype_include=object)),
)
```

---

## Combining Pipeline and ColumnTransformer

Use **ColumnTransformer** as a step inside **Pipeline**, or use **Pipeline** as a step inside **ColumnTransformer**.

### Preprocessor + Model

```python
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), [0, 1, 2]),
    ("cat", OneHotEncoder(), [3, 4]),
])

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression()),
])
```

---

## Parameter Access

Use **stepname__param** for nested parameters. Essential for **GridSearchCV**.

```python
pipe.get_params()
# Includes "clf__C", "clf__max_iter", "scaler__with_mean", etc.

pipe.set_params(clf__C=0.5)
```

### In GridSearchCV

```python
param_grid = {
    "preprocessor__num__imputer__strategy": ["mean", "median"],
    "clf__C": [0.1, 1.0, 10.0],
}
GridSearchCV(pipe, param_grid, cv=5)
```

---

## Best Practices

| Practice | Reason |
|----------|--------|
| Use **Pipeline** for preprocessing + model | Prevents data leakage in CV |
| Use **ColumnTransformer** for mixed types | Clean per-column preprocessing |
| Use **make_column_selector** with DataFrames | Robust column selection |
| Set **random_state** in pipeline steps | Reproducibility |
| Use descriptive step names | Easier debugging and parameter access |
