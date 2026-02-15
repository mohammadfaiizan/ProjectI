# Anomaly Detection

---

## Table of Contents

- [Overview](#overview)
- [Isolation Forest](#isolation-forest)
- [Local Outlier Factor](#local-outlier-factor)
- [One-Class SVM](#one-class-svm)
- [Comparison and Use Cases](#comparison-and-use-cases)

---

## Overview

Anomaly detection identifies outliers or novelties in data. Scikit-learn provides **IsolationForest**, **LocalOutlierFactor**, and **OneClassSVM**. All use **fit_predict** returning +1 (inlier) or -1 (outlier).

---

## Isolation Forest

**IsolationForest** isolates anomalies by random splits. Anomalies require fewer splits to isolate, yielding lower path lengths and **decision_function** scores.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **n_estimators** | Number of trees |
| **contamination** | Expected fraction of outliers (0.0–0.5) |
| **max_samples** | Samples per tree (`auto` or int) |

### Key Attributes and Methods

| Attribute/Method | Description |
|------------------|-------------|
| **decision_function** | Higher = more normal |
| **score_samples** | Negative log anomaly score |
| **predict** | +1 inlier, -1 outlier |

### Code Example

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
pred = iso.fit_predict(X)
scores = iso.decision_function(X)
```

### Contamination

**contamination** sets the decision threshold. If unknown, use **contamination='auto'** or tune via validation.

---

## Local Outlier Factor

**LOF** compares local density of a point to its neighbors. Anomalies have lower density than their neighbors.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **n_neighbors** | Number of neighbors for density |
| **contamination** | Expected fraction of outliers |
| **novelty** | If True, fit on train and predict on new data |

### Key Attributes

| Attribute | Description |
|-----------|-------------|
| **negative_outlier_factor_** | LOF score (higher = more anomalous) |
| **decision_function** | Available when **novelty=True** |

### Unsupervised Mode

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
pred = lof.fit_predict(X)
print(lof.negative_outlier_factor_)
```

### Novelty Detection

When **novelty=True**, fit on normal data only; predict on new samples.

```python
lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
lof.fit(X_train)
pred = lof.predict(X_test)
scores = lof.decision_function(X_test)
```

---

## One-Class SVM

**OneClassSVM** learns a boundary around the training data in feature space. Points outside are anomalies.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **kernel** | `rbf`, `linear`, `poly` |
| **nu** | Upper bound on outlier fraction and lower bound on support vectors |
| **gamma** | RBF/poly kernel coefficient |

### Key Attributes

| Attribute | Description |
|-----------|-------------|
| **n_support_** | Number of support vectors per class |
| **decision_function** | Signed distance to boundary |

### Code Example

```python
from sklearn.svm import OneClassSVM

ocsvm = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
pred = ocsvm.fit_predict(X)
scores = ocsvm.decision_function(X)
```

### Scaling

OneClassSVM is sensitive to feature scale; use **StandardScaler** before fitting.

---

## Comparison and Use Cases

### Algorithm Comparison

| Algorithm | Scalability | Multimodal | High-Dim | Training Data |
|-----------|-------------|------------|----------|---------------|
| IsolationForest | High | Yes | Yes | Mixed |
| LOF | Medium | Yes | Medium | Mixed |
| OneClassSVM | Low | No | Yes (kernel) | Normal only (novelty) |

### When to Use

- **IsolationForest**: Large datasets, high dimensions, mixed training data
- **LOF**: Local density matters, variable density clusters
- **OneClassSVM**: Small/medium data, kernel flexibility, novelty detection

### Score Interpretation

| Algorithm | Score | Interpretation |
|-----------|-------|-----------------|
| IsolationForest | decision_function | Higher = more normal |
| IsolationForest | score_samples | More negative = more anomalous |
| LOF | negative_outlier_factor_ | Higher = more anomalous |
| OneClassSVM | decision_function | More negative = more anomalous |
