# Linear Classifiers

---

## Table of Contents

- [Overview](#overview)
- [Logistic Regression](#logistic-regression)
- [Support Vector Classifier](#support-vector-classifier)
- [SGD Classifier](#sgd-classifier)
- [Perceptron](#perceptron)

---

## Overview

Linear classifiers learn a linear decision boundary. They are fast, interpretable, and scale well to large datasets.

---

## Logistic Regression

**LogisticRegression** models class probabilities via the logistic (sigmoid) function.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **penalty** | `l2`, `l1`, `elasticnet`, `none` |
| **solver** | `lbfgs`, `liblinear`, `sag`, `saga` (saga for L1/elasticnet) |
| **C** | Inverse regularization strength (smaller = stronger regularization) |
| **multi_class** | `ovr`, `multinomial`, `auto` |

### Code Example

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l2', C=1.0, multi_class='multinomial')
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
probs = lr.predict_proba(X_test)
```

---

## Support Vector Classifier

**SVC** finds the maximum-margin hyperplane separating classes. **LinearSVC** is faster for linear kernels.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **kernel** | `linear`, `rbf`, `poly`, `sigmoid` |
| **C** | Regularization; smaller = wider margin |
| **gamma** | Kernel coefficient for rbf/poly/sigmoid |

### Code Example

```python
from sklearn.svm import SVC, LinearSVC

svc = SVC(kernel='rbf', C=1.0, gamma='scale')
svc.fit(X_train, y_train)
dec = svc.decision_function(X_test)

lsvc = LinearSVC(C=1.0, max_iter=5000)
lsvc.fit(X_train, y_train)
```

---

## SGD Classifier

**SGDClassifier** uses Stochastic Gradient Descent for scalable, incremental learning.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **loss** | `hinge`, `log_loss`, `perceptron`, `squared_hinge` |
| **penalty** | `l2`, `l1`, `elasticnet` |

### partial_fit for Online Learning

```python
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(max_iter=1)
classes = np.unique(y_train)
for X_batch, y_batch in batches:
    sgd.partial_fit(X_batch, y_batch, classes=classes)
```

---

## Perceptron

**Perceptron** is a simple linear classifier; updates weights only on misclassification.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **penalty** | `l2`, `l1`, `elasticnet`, `none` |
| **alpha** | Regularization strength when penalty is used |
| **max_iter** | Maximum passes over training data |

```python
from sklearn.linear_model import Perceptron

perc = Perceptron(penalty='l2', alpha=0.0001)
perc.fit(X_train, y_train)
print(perc.n_iter_)  # Epochs to converge
```
