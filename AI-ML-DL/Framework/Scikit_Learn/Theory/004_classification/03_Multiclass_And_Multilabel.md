# Multiclass and Multilabel Classification

---

## Table of Contents

- [Overview](#overview)
- [Ridge Classifier](#ridge-classifier)
- [One-vs-Rest and One-vs-One](#one-vs-rest-and-one-vs-one)
- [Multilabel Classification](#multilabel-classification)

---

## Overview

**Multiclass**: each sample belongs to exactly one class. **Multilabel**: each sample can have multiple labels.

---

## Ridge Classifier

**RidgeClassifier** uses Ridge regression with thresholding for classification. Fast and works well with many features.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **alpha** | Regularization strength |

### RidgeClassifierCV

```python
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV

rc = RidgeClassifier(alpha=1.0)
rc.fit(X_train, y_train)

rc_cv = RidgeClassifierCV(alphas=[0.01, 0.1, 1, 10], cv=5)
rc_cv.fit(X_train, y_train)
print(rc_cv.alpha_)
```

---

## One-vs-Rest and One-vs-One

### OneVsRestClassifier (OvR)

- One binary classifier per class
- For K classes: K classifiers
- Each classifier: class k vs rest

### OneVsOneClassifier (OvO)

- One binary classifier per pair
- For K classes: K(K-1)/2 classifiers
- Each classifier: class i vs class j

```python
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC

ovr = OneVsRestClassifier(SVC(kernel='linear'))
ovr.fit(X_train, y_train)

ovo = OneVsOneClassifier(SVC(kernel='linear'))
ovo.fit(X_train, y_train)
```

---

## Multilabel Classification

Samples can have multiple labels. Use **MultiLabelBinarizer** to encode targets.

### MultiOutputClassifier

Trains one classifier per label (independent):

```python
from sklearn.multioutput import MultiOutputClassifier

moc = MultiOutputClassifier(LogisticRegression())
moc.fit(X_train, y_train)
```

### ClassifierChain

Chains classifiers; each uses prior predictions as features:

```python
from sklearn.multioutput import ClassifierChain

cc = ClassifierChain(LogisticRegression(), order=[0, 1, 2])
cc.fit(X_train, y_train)
```

### MultiLabelBinarizer

```python
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
y_binary = mlb.fit_transform([(0, 1), (1, 2), (0,)])
```
