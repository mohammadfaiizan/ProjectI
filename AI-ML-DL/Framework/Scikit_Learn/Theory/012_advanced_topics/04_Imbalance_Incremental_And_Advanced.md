# Class Imbalance, Incremental Learning, and Advanced Patterns

---

## Table of Contents

- [Overview](#overview)
- [Class Imbalance](#class-imbalance)
- [class_weight](#class_weight)
- [sample_weight](#sample_weight)
- [SMOTE and imblearn](#smote-and-imblearn)
- [Incremental Learning](#incremental-learning)
- [partial_fit](#partial_fit)
- [Sparse Data Handling](#sparse-data-handling)
- [scipy.sparse Matrices](#scipy-sparse-matrices)
- [Sparse-Compatible Estimators](#sparse-compatible-estimators)
- [Advanced Patterns](#advanced-patterns)
- [clone](#clone)
- [Meta-Estimators and Duck Typing](#meta-estimators-and-duck-typing)
- [Summary](#summary)

---

## Overview

This document covers **class imbalance** strategies, **incremental learning** with `partial_fit`, **sparse data** handling, and **advanced sklearn patterns** such as `clone` and meta-estimators.

---

## Class Imbalance

When classes are imbalanced (e.g., 99% negative, 1% positive), standard classifiers tend to predict the majority class. Strategies include reweighting, resampling, and specialized algorithms.

---

## class_weight

**class_weight** assigns higher loss to misclassifying minority samples. Use `class_weight="balanced"` to set weights inversely proportional to class frequencies.

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(class_weight="balanced")
clf.fit(X_train, y_train)
```

### Custom Weights

```python
clf = LogisticRegression(class_weight={0: 0.5, 1: 2.0})
```

Many classifiers support **class_weight**: LogisticRegression, SVC, RandomForestClassifier, etc.

---

## sample_weight

**sample_weight** assigns per-sample importance. Useful when some samples are more important or when combining multiple datasets.

```python
sample_weights = np.ones(len(y_train))
sample_weights[y_train == 1] = 5.0
clf.fit(X_train, y_train, sample_weight=sample_weights)
```

---

## SMOTE and imblearn

**SMOTE** (Synthetic Minority Over-sampling Technique) generates synthetic minority samples. It is not in scikit-learn; use the **imbalanced-learn** package.

```python
pip install imbalanced-learn
```

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
clf.fit(X_resampled, y_resampled)
```

imblearn provides SMOTE, ADASYN, RandomOverSampler, RandomUnderSampler, and pipeline-compatible versions.

---

## Incremental Learning

**Incremental** (online) learning processes data in batches. Useful for large datasets that do not fit in memory or for streaming data.

---

## partial_fit

Estimators supporting **partial_fit** can be updated with new batches of data. Call `partial_fit` repeatedly instead of a single `fit`.

```python
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
for X_batch, y_batch in batches:
    sgd.partial_fit(X_batch, y_batch, classes=classes)
```

### Estimators with partial_fit

| Estimator | Use Case |
|-----------|----------|
| **SGDClassifier** | Large-scale classification |
| **SGDRegressor** | Large-scale regression |
| **MiniBatchKMeans** | Large-scale clustering |
| **MultinomialNB** | Text classification |
| **BernoulliNB** | Binary features |
| **PassiveAggressiveClassifier** | Online classification |

### Requirements

- For classification, **classes** must be passed on the first call (or when new classes appear)
- Batches can have varying sizes

---

## Sparse Data Handling

Text, recommender systems, and high-dimensional categorical data often produce **sparse** matrices where most entries are zero. Storing and computing with sparse matrices saves memory and time.

---

## scipy.sparse Matrices

**scipy.sparse** provides sparse matrix types. Common formats:

| Format | Best For |
|--------|----------|
| **csr_matrix** | Row slicing, matrix-vector products |
| **csc_matrix** | Column slicing |
| **lil_matrix** | Incremental construction |

```python
from scipy.sparse import csr_matrix

X_sparse = csr_matrix(X_dense)
```

**TfidfVectorizer** and **CountVectorizer** return sparse matrices by default.

---

## Sparse-Compatible Estimators

Not all estimators accept sparse input. Check documentation. Sparse-compatible estimators include:

- **LogisticRegression**
- **LinearSVC**, **SGDClassifier**
- **MultinomialNB**, **BernoulliNB**
- **Ridge**, **Lasso**, **ElasticNet**
- **DecisionTreeClassifier** (converts to dense internally in some versions)

Use **X_sparse** directly when the estimator supports it; no need to convert to dense.

---

## Advanced Patterns

### clone

**clone** creates an unfitted copy of an estimator with the same parameters. Used internally by **GridSearchCV**, **cross_val_score**, and **BaggingClassifier**.

```python
from sklearn.base import clone

clf_fitted = LogisticRegression().fit(X, y)
clf_copy = clone(clf_fitted)
# clf_copy is unfitted but has same C, max_iter, etc.
```

---

## Meta-Estimators and Duck Typing

**Meta-estimators** wrap base estimators: **BaggingClassifier**, **MultiOutputClassifier**, **CalibratedClassifierCV**, etc. They use **clone** to create copies of the base estimator for each sub-model.

**Duck typing**: sklearn does not require inheritance from BaseEstimator. Any object with **fit** and **predict** (or **transform**) can be used in many contexts. Full compatibility (clone, get_params, set_params) requires proper implementation.

```python
class MyClassifier:
    def fit(self, X, y):
        ...
        return self
    def predict(self, X):
        ...
```

---

## Summary

| Topic | Key Takeaway |
|-------|--------------|
| **Class imbalance** | Use class_weight, sample_weight; consider imblearn for SMOTE |
| **Incremental learning** | Use partial_fit with SGDClassifier, MiniBatchKMeans, etc. |
| **Sparse data** | Use scipy.sparse; choose sparse-compatible estimators |
| **clone** | Creates unfitted copy; required for meta-estimators and CV |
| **Duck typing** | fit/predict interface enables integration; full API needs get_params/set_params |

These patterns support scalable, flexible machine learning workflows within the scikit-learn ecosystem.
