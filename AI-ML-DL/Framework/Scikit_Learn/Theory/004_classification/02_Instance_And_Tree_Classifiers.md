# Instance-Based and Tree Classifiers

---

## Table of Contents

- [Overview](#overview)
- [K-Nearest Neighbors Classifier](#k-nearest-neighbors-classifier)
- [Decision Tree Classifier](#decision-tree-classifier)
- [Naive Bayes](#naive-bayes)
- [Passive Aggressive Classifier](#passive-aggressive-classifier)
- [Nearest Centroid](#nearest-centroid)

---

## Overview

Instance-based methods use stored examples; tree methods partition the feature space; probabilistic methods use Bayes' theorem.

---

## K-Nearest Neighbors Classifier

**KNeighborsClassifier** assigns the majority class among k nearest training samples.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **n_neighbors** | Number of neighbors (k) |
| **weights** | `uniform` or `distance` |
| **metric** | `euclidean`, `manhattan`, `minkowski` |

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_train, y_train)
probs = knn.predict_proba(X_test)
```

---

## Decision Tree Classifier

**DecisionTreeClassifier** recursively splits on features using **Gini impurity** or **entropy**.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **criterion** | `gini`, `entropy`, `log_loss` |
| **max_depth** | Maximum tree depth |
| **min_samples_split** | Minimum samples to split a node |
| **min_samples_leaf** | Minimum samples per leaf |

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion='gini', max_depth=10)
dt.fit(X_train, y_train)
print(dt.feature_importances_)
```

---

## Naive Bayes

Assumes feature independence given class. Fast and works well with high-dimensional data.

| Model | Use Case |
|-------|----------|
| **GaussianNB** | Continuous features |
| **MultinomialNB** | Count data (e.g., text) |
| **BernoulliNB** | Binary features |
| **ComplementNB** | Imbalanced text classification |

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

mnb = MultinomialNB()
mnb.fit(X_count_train, y_train)
```

---

## Passive Aggressive Classifier

**PassiveAggressiveClassifier** updates aggressively on margin violations, passively otherwise. Good for online learning.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **C** | Aggressiveness (larger = more updates) |
| **loss** | `hinge`, `squared_hinge` |

```python
from sklearn.linear_model import PassiveAggressiveClassifier

pac = PassiveAggressiveClassifier(C=1.0)
pac.fit(X_train, y_train)
pac.partial_fit(X_batch, y_batch, classes=classes)
```

---

## Nearest Centroid

**NearestCentroid** assigns the class whose centroid is closest. Simple and fast.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **metric** | `euclidean`, `manhattan` |
| **shrink_threshold** | Shrink centroids toward global centroid |

```python
from sklearn.neighbors import NearestCentroid

nc = NearestCentroid(shrink_threshold=0.5)
nc.fit(X_train, y_train)
print(nc.centroids_)
```
