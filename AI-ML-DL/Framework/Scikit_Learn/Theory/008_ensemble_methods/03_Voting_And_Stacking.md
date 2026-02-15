# Voting and Stacking

---

## Table of Contents

- [Overview](#overview)
- [Voting Classifier](#voting-classifier)
- [Voting Regressor](#voting-regressor)
- [Stacking Classifier](#stacking-classifier)
- [Stacking Regressor](#stacking-regressor)
- [Voting vs Stacking](#voting-vs-stacking)
- [Best Practices](#best-practices)

---

## Overview

**Voting** and **Stacking** are meta-ensemble methods that combine predictions from multiple base estimators. **Voting** uses a simple rule (average or majority). **Stacking** trains a meta-learner on the base estimators' outputs to learn an optimal combination.

---

## Voting Classifier

**VotingClassifier** combines predictions from multiple classifiers. Two voting strategies:

### Hard Voting

Each classifier predicts a class label; the final prediction is the **majority vote**.

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

estimators = [
    ("lr", LogisticRegression()),
    ("dt", DecisionTreeClassifier()),
    ("svc", SVC()),
]
vc = VotingClassifier(estimators=estimators, voting="hard")
vc.fit(X_train, y_train)
```

### Soft Voting

Each classifier outputs class probabilities; the final prediction is the class with the **highest averaged probability**. Requires base estimators to implement **predict_proba**.

```python
vc = VotingClassifier(estimators=estimators, voting="soft")
vc.fit(X_train, y_train)
```

**Note**: For soft voting, **SVC** must use `probability=True` to enable probability estimates.

### Weights

Assign different weights to base estimators. For soft voting, probabilities are weighted before averaging.

```python
vc = VotingClassifier(
    estimators=estimators,
    voting="soft",
    weights=[2, 1, 1]
)
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **estimators** | List of (name, estimator) tuples |
| **voting** | `"hard"` (majority) or `"soft"` (averaged probabilities) |
| **weights** | Optional weights per estimator |
| **n_jobs** | Parallel fit/predict |

### named_estimators_

Access fitted base estimators by name:

```python
vc.fit(X_train, y_train)
lr_model = vc.named_estimators_["lr"]
```

---

## Voting Regressor

**VotingRegressor** averages the predictions of multiple regressors. No hard/soft distinction; it always averages.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **estimators** | List of (name, estimator) tuples |
| **weights** | Optional weights per estimator |

```python
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

estimators = [
    ("lr", LinearRegression()),
    ("ridge", Ridge()),
    ("dt", DecisionTreeRegressor()),
]
vr = VotingRegressor(estimators=estimators, weights=[1, 2, 1])
vr.fit(X_train, y_train)
y_pred = vr.predict(X_test)
```

---

## Stacking Classifier

**StackingClassifier** uses cross-validation to generate meta-features: for each sample, base estimators predict on out-of-fold data. A **final_estimator** (meta-learner) is trained on these meta-features to produce the final prediction.

### Algorithm

1. Split training data into K folds
2. For each fold k: train base estimators on the other K-1 folds; predict on fold k
3. Stack out-of-fold predictions into meta-features
4. Train **final_estimator** on meta-features and true labels
5. At prediction time: base estimators predict; final_estimator combines their outputs

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **estimators** | List of (name, estimator) tuples |
| **final_estimator** | Meta-learner (default: LogisticRegression) |
| **cv** | Cross-validation strategy (default: 5-fold) |
| **stack_method** | `"auto"`, `"predict"`, `"predict_proba"`, or `"decision_function"` |

```python
from sklearn.ensemble import StackingClassifier

stk = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,
    stack_method="predict_proba"
)
stk.fit(X_train, y_train)
```

### stack_method

Controls what base estimators output as meta-features:

- **predict**: Class labels (one column per estimator for multi-class)
- **predict_proba**: Class probabilities (multiple columns per estimator)
- **decision_function**: Decision function values (e.g., SVM)
- **auto**: Chooses based on estimator capabilities

### named_estimators_ and final_estimator_

```python
stk.fit(X_train, y_train)
base_models = stk.named_estimators_
meta_model = stk.final_estimator_
```

---

## Stacking Regressor

**StackingRegressor** follows the same logic: base regressors produce out-of-fold predictions; **final_estimator** learns to combine them.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| **estimators** | List of (name, estimator) tuples |
| **final_estimator** | Meta-learner (default: Ridge) |
| **cv** | Cross-validation strategy |

```python
from sklearn.ensemble import StackingRegressor

stk = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=5
)
stk.fit(X_train, y_train)
```

### transform

**transform** returns the meta-features (base estimator predictions) for new data. Useful for analysis or feeding into another model.

```python
meta_features = stk.transform(X_test)
```

---

## Voting vs Stacking

| Aspect | Voting | Stacking |
|--------|--------|----------|
| **Combination rule** | Fixed (average, majority) | Learned by meta-learner |
| **Training** | Only base estimators | Base + meta-learner |
| **Overfitting risk** | Lower | Higher (meta-learner can overfit) |
| **Flexibility** | Limited | Can learn non-linear combinations |
| **Computation** | Faster | Slower (cross-validation) |

### When to Use Voting

- Base estimators are diverse and similarly accurate
- Need simple, interpretable combination
- Want fast training

### When to Use Stacking

- Base estimators have different strengths
- Willing to invest in cross-validation and meta-learner tuning
- Want to learn optimal weights or non-linear combination

---

## Best Practices

### Diversity

Use diverse base estimators (e.g., linear, tree-based, kernel-based) so their errors are uncorrelated.

### Cross-Validation in Stacking

Larger **cv** reduces overfitting of meta-features but increases computation. Typical: 5 or 10 folds.

### Final Estimator

- **Simple** (e.g., LogisticRegression, Ridge): Reduces overfitting
- **Complex** (e.g., RandomForest): Can capture non-linear combinations but may overfit

### Weights in Voting

Use **weights** when some estimators are more reliable. Can be tuned via cross-validation.

---

## Summary

- **VotingClassifier**: Hard (majority) or soft (averaged probabilities) combination
- **VotingRegressor**: Weighted average of predictions
- **StackingClassifier/Regressor**: Meta-learner trained on out-of-fold base predictions
- **Stacking** is more flexible but requires more computation and care to avoid overfitting
- Use diverse base estimators for both voting and stacking
