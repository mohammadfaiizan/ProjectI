# Non-Linear Regression

---

## Table of Contents

- [Overview](#overview)
- [Support Vector Regression](#support-vector-regression)
- [K-Nearest Neighbors Regressor](#k-nearest-neighbors-regressor)
- [Decision Tree Regressor](#decision-tree-regressor)

---

## Overview

Non-linear models capture complex relationships without explicit feature transformation.

---

## Support Vector Regression

**SVR** finds a function with at most **epsilon** deviation from targets while maximizing margin.

### Key Concepts

- **epsilon-tube**: Predictions within epsilon of true values incur no loss
- **kernel trick**: Maps data to higher dimensions for non-linear fits

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **kernel** | `linear`, `rbf`, `poly`, `sigmoid` |
| **C** | Regularization; smaller = more margin, fewer support vectors |
| **epsilon** | Width of epsilon-tube |

### Code Example

```python
from sklearn.svm import SVR, LinearSVR

svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr.fit(X_train, y_train)

# LinearSVR: faster for linear case
lsvr = LinearSVR(C=1.0, epsilon=0.1)
lsvr.fit(X_train, y_train)
```

---

## K-Nearest Neighbors Regressor

**KNeighborsRegressor** predicts by averaging the target values of the k nearest training samples.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **n_neighbors** | Number of neighbors (k) |
| **weights** | `uniform` or `distance` (inverse distance weighting) |
| **metric** | `euclidean`, `manhattan`, `minkowski` |

### Code Example

```python
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=10, weights='distance')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
```

### Scaling

KNN is distance-based; **StandardScaler** or **MinMaxScaler** is recommended.

---

## Decision Tree Regressor

**DecisionTreeRegressor** partitions the feature space recursively using axis-aligned splits.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **max_depth** | Maximum tree depth (pruning) |
| **min_samples_split** | Minimum samples to split a node |
| **min_samples_leaf** | Minimum samples per leaf |

### Pruning

```python
from sklearn.tree import DecisionTreeRegressor

# Limit depth to prevent overfitting
dt = DecisionTreeRegressor(max_depth=10, min_samples_leaf=5)
dt.fit(X_train, y_train)
print(dt.get_depth())
```

### Pros and Cons

| Pros | Cons |
|------|------|
| No scaling needed | Prone to overfitting |
| Handles non-linearity | High variance |
| Interpretable | No extrapolation |
