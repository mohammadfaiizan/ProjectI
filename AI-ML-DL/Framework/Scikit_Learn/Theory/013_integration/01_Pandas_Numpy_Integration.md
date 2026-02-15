# Pandas and NumPy Integration

---

## Table of Contents

- [Overview](#overview)
- [Pandas Integration](#pandas-integration)
- [set_output API](#set_output-api)
- [DataFrame Input and Output](#dataframe-input-and-output)
- [Feature Names](#feature-names)
- [ColumnTransformer with Pandas](#columntransformer-with-pandas)
- [NumPy Interoperability](#numpy-interoperability)
- [Array Handling](#array-handling)
- [Memory Layout: C vs F Order](#memory-layout-c-vs-f-order)
- [Dtype and Copy Behavior](#dtype-and-copy-behavior)
- [Best Practices](#best-practices)

---

## Overview

Scikit-learn integrates seamlessly with **Pandas** and **NumPy**, the foundational data structures for data science in Python. Understanding how transformers and estimators handle **DataFrame** input/output and **NumPy** array semantics is essential for building robust pipelines and avoiding subtle bugs.

---

## Pandas Integration

### Why Pandas Integration Matters

- Preserve **column names** and **index** through transformations
- Debug pipelines by inspecting intermediate DataFrames
- Integrate with data validation and schema checks
- Improve readability when working with tabular data

### set_output API

Scikit-learn 1.2+ introduced **set_output** for transformers. Call `set_output(transform="pandas")` on any transformer to get **DataFrame** output instead of NumPy arrays.

```python
from sklearn.preprocessing import StandardScaler
import pandas as pd

scaler = StandardScaler().set_output(transform="pandas")
X = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
X_scaled = scaler.fit_transform(X)
# X_scaled is a DataFrame with columns ["a", "b"]
```

### Global Configuration

Use **set_config** to apply pandas output globally:

```python
from sklearn import set_config

set_config(transform_output="pandas")
scaler = StandardScaler()
# All transformers now return DataFrames by default
X_out = scaler.fit_transform(X)

set_config(transform_output="default")  # Reset
```

---

## DataFrame Input and Output

### Input Acceptance

Most estimators accept **DataFrame** input. Scikit-learn extracts the underlying array via `np.asarray` or preserves structure when `set_output` is used. Column order is preserved.

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
clf = LogisticRegression()
clf.fit(df, y)
pred = clf.predict(df)
```

### Output with set_output

When `transform_output="pandas"` is set, the output DataFrame has:

- **Column names** from `get_feature_names_out()`
- **Index** preserved from input (when applicable)
- **dtype** typically float64

---

## Feature Names

### get_feature_names_out

Transformers implement **get_feature_names_out** to return output feature names. Use `verbose_feature_names_out=False` in **ColumnTransformer** to avoid long prefixed names.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

ct = ColumnTransformer([
    ("num", StandardScaler(), ["num1", "num2"]),
    ("cat", OneHotEncoder(drop="first"), ["cat"])
], verbose_feature_names_out=False)
ct.fit(df)
print(ct.get_feature_names_out())
# ['num1', 'num2', 'cat_B', 'cat_C']
```

---

## ColumnTransformer with Pandas

### Preserving Column Semantics

**ColumnTransformer** selects columns by name when given a DataFrame. Combine with `set_output("pandas")` for end-to-end DataFrame pipelines.

```python
ct = ColumnTransformer([
    ("num", StandardScaler(), ["num1", "num2"]),
    ("cat", OneHotEncoder(), ["cat"])
]).set_output(transform="pandas")

X_trans = ct.fit_transform(df)
# X_trans is DataFrame with named columns
```

### Pipeline Compatibility

Pipelines pass output to the next step. If the first step returns a DataFrame, the next step receives it. **LogisticRegression** and most estimators accept DataFrame input.

---

## NumPy Interoperability

### Core Contract

Scikit-learn expects **array-like** input: NumPy arrays, SciPy sparse matrices, or objects convertible via `np.asarray`. Output is typically **NumPy** unless `set_output("pandas")` is used.

### Array Conversion

- Input is converted to **float64** for numerical estimators when needed
- **int** and **float32** are often accepted; internal computation may use float64
- Sparse input is preserved when the estimator supports it

---

## Array Handling

### Shape Requirements

- **fit(X, y)**: X is 2D (n_samples, n_features), y is 1D (n_samples) for supervised learning
- **predict(X)**: X must have the same number of features as training data
- Single-sample prediction: use `X.reshape(1, -1)` to ensure 2D

### Contiguity

Scikit-learn generally works with both **C-contiguous** and **F-contiguous** arrays. Internal copies may be made for efficiency. Use `np.ascontiguousarray` or `np.asfortranarray` if layout matters for downstream code.

---

## Memory Layout: C vs F Order

### C-Order (Row-Major)

Default for NumPy. Rows are stored contiguously. Best for row-wise access patterns.

```python
X_c = np.ascontiguousarray(X)
print(X_c.flags["C_CONTIGUOUS"])  # True
```

### F-Order (Column-Major)

Columns stored contiguously. Useful for column-wise operations.

```python
X_f = np.asfortranarray(X)
print(X_f.flags["F_CONTIGUOUS"])  # True
```

### Impact on sklearn

Most sklearn operations work with both. Transformations typically produce a **copy** with C-contiguous layout. No need to worry unless interfacing with external C/Fortran code.

---

## Dtype and Copy Behavior

### Dtype

- **fit** and **transform** may convert input to float64
- **predict** output is typically int for classifiers, float for regressors
- **predict_proba** returns float64 array of shape (n_samples, n_classes)

### Copy Behavior

- **fit(X, y)** does not modify X or y
- **transform(X)** may return a copy; in-place modification is not guaranteed
- Use `copy=False` in some transformers (e.g., **StandardScaler**) when available to avoid unnecessary copies

---

## Best Practices

| Practice | Recommendation |
|----------|----------------|
| **Pipeline preprocessing** | Use Pipeline for preprocessing + model; avoid data leakage |
| **Feature names** | Use `set_output("pandas")` when debugging or when column names matter |
| **Reproducibility** | Set `random_state` on all estimators |
| **Validation** | Use `check_array` for custom code; validate feature count at predict time |
| **Memory** | For large data, prefer sparse matrices or incremental learning when applicable |

---

## Summary

- **set_output("pandas")** enables DataFrame output from transformers
- **set_config** can set global transform output
- **get_feature_names_out** provides interpretable column names
- **ColumnTransformer** with column names works well with DataFrames
- NumPy arrays are the native format; C-contiguous and F-contiguous both work
- Dtype conversion to float64 is common; single-sample input must be 2D
