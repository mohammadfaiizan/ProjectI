# Calibration, Class Imbalance, and Model Comparison

---

## Table of Contents

- [Probability Calibration](#probability-calibration)
- [Class Imbalance](#class-imbalance)
- [Model Comparison](#model-comparison)

---

## Probability Calibration

Raw **decision_function** or **predict_proba** outputs may not reflect true probabilities. **CalibratedClassifierCV** recalibrates them.

### Methods

| Method | Description |
|--------|-------------|
| **sigmoid** | Platt scaling; fits logistic regression on scores |
| **isotonic** | Non-parametric; preserves order |

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

base = SVC(probability=False)
cal = CalibratedClassifierCV(base, method='sigmoid', cv=3)
cal.fit(X_train, y_train)
probs = cal.predict_proba(X_test)
```

### When to Use

- Need reliable probability estimates (e.g., cost-sensitive decisions)
- Base estimator has uncalibrated scores (e.g., SVM, tree ensembles)

---

## Class Imbalance

When classes are imbalanced, models may favor the majority class.

### class_weight='balanced'

Weights inversely proportional to class frequency:

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(class_weight='balanced')
lr.fit(X_train, y_train)
```

### Custom class_weight

```python
lr = LogisticRegression(class_weight={0: 0.5, 1: 2.0})
lr.fit(X_train, y_train)
```

### sample_weight

Per-sample weights for fine-grained control:

```python
sample_weights = np.ones(len(y_train))
sample_weights[y_train == 1] = 5.0  # Upweight minority
lr.fit(X_train, y_train, sample_weight=sample_weights)
```

---

## Model Comparison

### Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Fraction correct (careful with imbalance) |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1** | Harmonic mean of precision and recall |

### Comparison Workflow

```python
from sklearn.model_selection import cross_val_score

models = {
    'LogisticRegression': LogisticRegression(),
    'SVC': SVC(kernel='rbf'),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'DecisionTree': DecisionTreeClassifier(max_depth=10),
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
```

### Considerations

| Factor | Notes |
|--------|-------|
| **Scalability** | Linear models, SGD scale well |
| **Interpretability** | Logistic, trees more interpretable |
| **Imbalance** | Use class_weight, sample_weight |
| **Probabilities** | Use calibration if needed |
