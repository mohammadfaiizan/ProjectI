# Pipelines and Interoperability

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Basics](#pipeline-basics)
- [ColumnTransformer](#columntransformer)
- [FeatureUnion and Feature Selection in Pipelines](#featureunion-and-feature-selection-in-pipelines)
- [NumPy Interoperability](#numpy-interoperability)
- [Pandas Interoperability](#pandas-interoperability)
- [Sparse Matrix Support](#sparse-matrix-support)
- [Pipeline with GridSearchCV](#pipeline-with-gridsearchcv)
- [Best Practices](#best-practices)

---

## Overview

**Pipeline** chains multiple estimators (transformers and a final estimator) into a single object. It ensures that preprocessing is fitted only on training data and applied consistently. **ColumnTransformer** applies different transformers to different columns. Interoperability with **NumPy** and **Pandas** is central to real-world workflows.

| Component | Purpose |
|-----------|---------|
| **Pipeline** | Chain fit/transform/predict |
| **ColumnTransformer** | Per-column preprocessing |
| **FeatureUnion** | Concatenate transformer outputs |
| **make_column_selector** | Select columns by type or name |

---

## Pipeline Basics

**Pipeline** applies a sequence of transformers followed by a final estimator. Each step is named for access and parameter tuning.

### Structure

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

- **fit**: Calls `fit` then `fit_transform` on each transformer, `fit` on the final estimator
- **predict**: Applies `transform` through all steps, then `predict` on the final estimator
- **transform**: Available if the final step has `transform` (e.g., some classifiers)
- Steps are accessible via **named_steps** or indexing: `pipe["scaler"]`

### Parameter Access

Use **stepname__param** for GridSearchCV:

```python
param_grid = {"clf__C": [0.1, 1.0, 10.0], "scaler__with_mean": [True, False]}
```

---

## ColumnTransformer

**ColumnTransformer** applies different transformers to different subsets of columns. Useful for mixed numeric and categorical data.

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

### make_column_selector

```python
from sklearn.compose import make_column_selector

ct = ColumnTransformer([
    ("num", StandardScaler(), make_column_selector(dtype_include=np.number)),
    ("cat", OneHotEncoder(), make_column_selector(dtype_include=object)),
])
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

---

## FeatureUnion and Feature Selection in Pipelines

**FeatureUnion** concatenates outputs of multiple transformers. Useful for combining different feature extraction methods.

```python
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

union = FeatureUnion([
    ("pca", PCA(n_components=5)),
    ("select", SelectKBest(k=10)),
])
X_combined = union.fit_transform(X, y)
```

### In a Pipeline

```python
pipe = Pipeline([
    ("union", FeatureUnion([...])),
    ("clf", LogisticRegression()),
])
```

---

## NumPy Interoperability

Scikit-learn expects **NumPy arrays** or **scipy.sparse** matrices. Most estimators accept `np.ndarray` with shape **(n_samples, n_features)**.

### Requirements

- **X**: 2D array, float or int
- **y**: 1D array, int (classification) or float (regression)
- **C-order** (row-major) is preferred for performance

### Conversion

```python
import numpy as np

# From list
X = np.array([[1, 2], [3, 4]])

# Ensure 2D
X = np.atleast_2d(X)

# Ensure float
X = X.astype(np.float64)
```

---

## Pandas Interoperability

**Pandas** DataFrames can be passed directly to many estimators. Column names are preserved where supported (e.g., ColumnTransformer with **make_column_selector**).

### DataFrame to Array

```python
import pandas as pd

X_df = pd.DataFrame(X, columns=["f1", "f2", "f3"])
X_array = X_df.values  # or X_df.to_numpy()
```

### Preserving Column Names

```python
from sklearn.compose import make_column_transformer

ct = make_column_transformer(
    (StandardScaler(), make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(), make_column_selector(dtype_include=object)),
)
# Column names flow through when using get_feature_names_out()
```

---

## Sparse Matrix Support

Many estimators accept **scipy.sparse** matrices (CSR, CSC). Useful for high-dimensional sparse data (e.g., text).

```python
from scipy.sparse import csr_matrix

X_sparse = csr_matrix(X_dense)
model.fit(X_sparse, y)
```

### Sparse Output

Some transformers produce sparse output (e.g., OneHotEncoder with **sparse_output=True**). Check estimator documentation for sparse support.

---

## Pipeline with GridSearchCV

**GridSearchCV** and **RandomizedSearchCV** work seamlessly with pipelines. Use **step__param** for parameter names.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "scaler__with_std": [True, False],
    "clf__C": [0.1, 1.0, 10.0],
    "clf__penalty": ["l2"],
}
gs = GridSearchCV(pipe, param_grid, cv=5)
gs.fit(X_train, y_train)
print(gs.best_params_)
```

### Refitting

**GridSearchCV** refits the best estimator on the full training data by default. Use **refit** to control this behavior.

---

## Best Practices

| Practice | Reason |
|---------|--------|
| Use **Pipeline** for preprocessing + model | Prevents data leakage in CV |
| Use **ColumnTransformer** for mixed types | Clean per-column preprocessing |
| Use **make_column_selector** with DataFrames | Robust column selection |
| Set **random_state** in pipeline steps | Reproducibility |
| Use **n_jobs=-1** in GridSearchCV | Parallel search |
