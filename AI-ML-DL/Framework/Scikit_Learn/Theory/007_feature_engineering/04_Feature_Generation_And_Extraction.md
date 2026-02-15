# Feature Generation and Extraction

---

## Table of Contents

- [Overview](#overview)
- [PolynomialFeatures](#polynomialfeatures)
- [Interaction Terms](#interaction-terms)
- [Text Feature Extraction](#text-feature-extraction)
- [Image Feature Extraction](#image-feature-extraction)
- [Custom Feature Transformers](#custom-feature-transformers)
- [FunctionTransformer](#functiontransformer)
- [Feature Pipelines](#feature-pipelines)
- [Best Practices](#best-practices)

---

## Overview

**Feature generation** creates new features from existing ones (e.g., polynomial terms, interactions). **Feature extraction** derives features from raw data (e.g., text, images). Both expand the representation to capture non-linear or domain-specific structure.

| Method | Input | Output | Use Case |
|--------|-------|--------|----------|
| **PolynomialFeatures** | Numeric | Polynomial terms | Non-linear relationships |
| **CountVectorizer** | Text | Bag-of-words | Text classification |
| **TfidfVectorizer** | Text | TF-IDF vectors | Text, information retrieval |
| **FunctionTransformer** | Any | Custom transform | Log, sqrt, custom ops |
| **FeatureHasher** | Dicts | Sparse vectors | High-cardinality categoricals |

---

## PolynomialFeatures

**PolynomialFeatures** generates polynomial and interaction terms. Captures non-linear relationships for linear models.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **degree** | Polynomial degree (default 2) |
| **interaction_only** | Only interaction terms, no powers |
| **include_bias** | Include constant (intercept) term |

### Usage

```python
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

X = np.array([[1, 2], [3, 4]])
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
print(poly.get_feature_names_out())
# [1, x0, x1, x0^2, x0*x1, x1^2]
```

### Interaction Only

```python
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_inter = poly.fit_transform(X)
# [1, x0, x1, x0*x1]
```

### Caveats

- **Curse of dimensionality**: High degree or many features lead to explosion of columns
- **Scale** features first; polynomial terms amplify scale differences

---

## Interaction Terms

**Interaction terms** capture combined effects of two or more features. Use **PolynomialFeatures(interaction_only=True)** or manual construction.

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_inter = poly.fit_transform(X)
```

---

## Text Feature Extraction

**CountVectorizer** builds a bag-of-words matrix. **TfidfVectorizer** applies TF-IDF weighting to reduce importance of frequent terms.

### CountVectorizer

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = ["hello world", "world of python", "python programming"]
vec = CountVectorizer()
X = vec.fit_transform(corpus)
print(vec.get_feature_names_out())
```

### TfidfVectorizer

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X = tfidf.fit_transform(corpus)
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **max_features** | Limit vocabulary size |
| **ngram_range** | (1, 1) unigrams, (1, 2) bigrams |
| **min_df, max_df** | Document frequency bounds |
| **stop_words** | Remove common words |

---

## Image Feature Extraction

For images, common approaches include flattening pixels, **PCA** on patches, or using pre-trained CNNs (outside scikit-learn). Scikit-learn provides **PatchExtractor** for extracting patches.

```python
from sklearn.feature_extraction import image
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.images
# Flatten for simple models
X_flat = X.reshape(X.shape[0], -1)
```

### Patch Extraction

```python
from sklearn.feature_extraction.image import extract_patches_2d

patches = extract_patches_2d(image, patch_size=(8, 8))
```

---

## Custom Feature Transformers

Create custom transformers by inheriting **BaseEstimator** and **TransformerMixin**. Implement **fit** and **transform**.

```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log1p(np.abs(X))
```

---

## FunctionTransformer

**FunctionTransformer** wraps a function (and optional inverse) as a transformer. No fit needed for stateless transforms.

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1)
X_transformed = transformer.fit_transform(X)
```

### With Keyword Arguments

```python
transformer = FunctionTransformer(np.power, kw_args={"exponent": 2})
X_squared = transformer.fit_transform(X)
```

---

## Feature Pipelines

Chain feature generation and extraction in a **Pipeline** with scaling and model.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

pipe = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("scaler", StandardScaler()),
    ("reg", LinearRegression()),
])
pipe.fit(X_train, y_train)
```

---

## Best Practices

| Practice | Reason |
|----------|--------|
| **Scale** before PolynomialFeatures | Avoid numerical issues |
| Limit **degree** and **max_features** | Control dimensionality |
| Use **TfidfVectorizer** for text | Better than raw counts |
| Use **FunctionTransformer** for simple ops | Clean, pipeline-compatible |
| Combine in **Pipeline** | Consistent transform flow |
