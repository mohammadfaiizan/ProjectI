# Custom Transformers and Inspection

---

## Table of Contents

- [Overview](#overview)
- [Custom Transformers](#custom-transformers)
- [BaseEstimator and TransformerMixin](#baseestimator-and-transformermixin)
- [Implementing fit and transform](#implementing-fit-and-transform)
- [Pipeline Inspection](#pipeline-inspection)
- [get_params and set_params](#get_params-and-set_params)
- [Pipeline Caching](#pipeline-caching)
- [HTML Display](#html-display)
- [Best Practices](#best-practices)

---

## Overview

**Custom transformers** extend scikit-learn's preprocessing capabilities. They must inherit from **BaseEstimator** and **TransformerMixin** to work with **Pipeline**, **GridSearchCV**, and other meta-estimators. **Pipeline inspection** uses **named_steps**, **get_params**, and **set_params**. **Caching** via the **memory** parameter speeds up repeated fits, and **set_config(display='diagram')** produces HTML representations for Jupyter.

---

## Custom Transformers

A custom transformer is a class that implements `fit` and `transform`. It should be compatible with scikit-learn's API for use in pipelines and grid search.

### Requirements

1. Inherit from **BaseEstimator** and **TransformerMixin**
2. Implement `fit(X, y=None)` returning `self`
3. Implement `transform(X)` returning the transformed array
4. Store learned state in attributes ending with `_` (e.g., `mean_`, `components_`)

---

## BaseEstimator and TransformerMixin

**BaseEstimator** provides `get_params` and `set_params` for hyperparameter handling. **TransformerMixin** provides `fit_transform` as a combination of `fit` and `transform`.

```python
from sklearn.base import BaseEstimator, TransformerMixin

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, offset=1.0):
        self.offset = offset

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log1p(np.abs(X) + self.offset)
```

---

## Implementing fit and transform

### Stateless Transformers

If the transformer does not learn from data, `fit` can simply return `self`:

```python
def fit(self, X, y=None):
    return self
```

### Stateful Transformers

Store learned parameters in attributes with a trailing underscore:

```python
class ThresholdFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.mask_ = None

    def fit(self, X, y=None):
        self.mask_ = np.var(X, axis=0) > self.threshold
        return self

    def transform(self, X):
        return X[:, self.mask_]
```

### Parameter Naming for Grid Search

Constructor parameters become accessible as `transformer__param` in pipelines:

```python
pipe = Pipeline([
    ("log", LogTransformer(offset=0.1)),
    ("clf", LogisticRegression()),
])
pipe.set_params(log__offset=0.5)
```

---

## Pipeline Inspection

### named_steps

**named_steps** is a dictionary mapping step names to estimator instances:

```python
for name, step in pipe.named_steps.items():
    print(f"{name}: {type(step).__name__}")
```

### steps Attribute

**steps** is a list of (name, estimator) tuples. Modifying it directly is not recommended; use `set_params` instead.

### Indexing

```python
scaler = pipe["scaler"]
scaler = pipe[0]
preprocess = pipe[:-1]
```

---

## get_params and set_params

**get_params** returns a flat dictionary of all parameters, including nested ones. Nested parameters use the **double underscore** convention: `stepname__param`.

```python
params = pipe.get_params()
# {"scaler", "clf", "clf__C", "clf__max_iter", ...}
```

**set_params** updates parameters by keyword:

```python
pipe.set_params(clf__C=0.1, clf__max_iter=500)
```

This is essential for **GridSearchCV**, which uses `set_params` to try different hyperparameter combinations.

---

## Pipeline Caching

The **memory** parameter caches fitted transformers to avoid recomputing them when the same pipeline is fit multiple times (e.g., in grid search with overlapping parameter combinations).

### Using a Directory

```python
from joblib import Memory
from sklearn.pipeline import Pipeline

memory = Memory(location="/tmp/cache", verbose=0)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=10)),
    ("clf", LogisticRegression()),
], memory=memory)
```

### Using a String Path

```python
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=10)),
    ("clf", LogisticRegression()),
], memory="/tmp/cache")
```

### When Caching Helps

- **GridSearchCV** or **RandomizedSearchCV** with expensive preprocessing
- Repeated fits with the same data
- Pipelines with **PCA**, **PolynomialFeatures**, or custom heavy transformers

### Caveats

- Caching uses **joblib**; the cache directory can grow large
- Transformers must be picklable
- Do not use caching when transformers have non-serializable state

---

## HTML Display

**set_config** controls how estimators are displayed. With `display="diagram"`, pipelines render as HTML diagrams in Jupyter.

```python
from sklearn import set_config

set_config(display="diagram")
print(pipe)
```

### _repr_html_

In Jupyter, pipelines with `display="diagram"` use **_repr_html_** to produce an interactive diagram showing the flow of data through steps.

```python
set_config(display="diagram")
html = pipe._repr_html_()
```

### Resetting

```python
set_config(display="text")
```

---

## Best Practices

| Practice | Description |
|----------|-------------|
| **Use trailing underscore** | Learned attributes should end with `_` |
| **fit returns self** | Always `return self` in `fit` |
| **Check for None in fit** | `y` may be `None` for unsupervised transformers |
| **Avoid in-place modification** | `transform` should not modify `X` in place |
| **Use memory for expensive steps** | Cache when preprocessing is costly |
| **Explicit step names** | Use **Pipeline** with names when tuning with GridSearchCV |

---

## Summary

- Custom transformers inherit **BaseEstimator** and **TransformerMixin**
- **fit** returns `self`; **transform** returns the transformed array
- **named_steps**, **get_params**, and **set_params** enable inspection and tuning
- **memory** caches fitted transformers for faster grid search
- **set_config(display="diagram")** produces HTML pipeline diagrams in Jupyter
