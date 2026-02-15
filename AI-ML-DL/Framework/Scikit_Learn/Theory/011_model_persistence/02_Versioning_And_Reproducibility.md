# Versioning and Reproducibility

---

## Table of Contents

- [Overview](#overview)
- [Model Versioning Strategies](#model-versioning-strategies)
- [Semantic Versioning](#semantic-versioning)
- [Metadata Alongside Models](#metadata-alongside-models)
- [random_state and Reproducibility](#random_state-and-reproducibility)
- [Seed Management](#seed-management)
- [Environment Tracking](#environment-tracking)
- [Reproducibility Checklist](#reproducibility-checklist)

---

## Overview

**Model versioning** ensures you can track, compare, and roll back model artifacts. **Reproducibility** guarantees that the same code and data produce the same model and predictions. Both require disciplined use of **random_state**, **metadata**, and **environment tracking**.

---

## Model Versioning Strategies

### File-Based Versioning

Store models with version suffixes or in versioned directories:

```python
version = "v1.0.0"
joblib.dump(model, f"model_{version}.joblib")
```

### Directory-Based Versioning

```
models/
  20250215_001/
    model.joblib
    metadata.json
  20250215_002/
    model.joblib
    metadata.json
```

### Registry Pattern

Maintain an index that points to the latest or production version:

```python
registry = {
    "versions": ["v1.0.0", "v1.1.0"],
    "latest": "v1.1.0",
    "production": "v1.0.0",
}
```

---

## Semantic Versioning

Use **semantic versioning** (major.minor.patch) for models:

| Component | When to Increment |
|-----------|-------------------|
| **Major** | Breaking changes (e.g., different features, incompatible API) |
| **Minor** | New training, improved metrics, backward compatible |
| **Patch** | Bug fixes, metadata updates |

```python
version = "v2.1.3"  # major=2, minor=1, patch=3
```

---

## Metadata Alongside Models

Store **metadata** in a separate file (JSON, YAML) or bundled with the model:

```python
metadata = {
    "version": "v1.0.0",
    "created_at": "2025-02-15T10:00:00",
    "model_type": "RandomForestClassifier",
    "train_samples": 120,
    "test_accuracy": 0.95,
    "sklearn_version": "1.3.2",
}
with open("model_meta.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

### Recommended Metadata Fields

| Field | Description |
|-------|-------------|
| **version** | Model version string |
| **created_at** | ISO timestamp |
| **model_type** | Estimator class name |
| **train_samples** | Number of training samples |
| **metrics** | Test accuracy, RMSE, etc. |
| **sklearn_version** | For compatibility checks |

---

## random_state and Reproducibility

**random_state** controls randomness in scikit-learn. Setting it ensures reproducible results across runs.

### In Estimators

```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### In train_test_split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### In Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, random_state=42)
```

### Without random_state

Models and splits will differ between runs. Useful only when you explicitly want non-determinism (e.g., robustness testing).

---

## Seed Management

### NumPy Global Seed

```python
import numpy as np
np.random.seed(42)
```

Affects NumPy random operations. Scikit-learn uses its own RNG when **random_state** is set; numpy seed affects custom code and some third-party libraries.

### Best Practice

Prefer **random_state** in each component over global seeds. It makes dependencies explicit and avoids side effects.

---

## Environment Tracking

### pip freeze

Capture the exact package versions:

```bash
pip freeze > requirements.txt
pip install -r requirements.txt
```

### conda export

For conda environments:

```bash
conda env export > environment.yml
conda env create -f environment.yml
```

### Minimal Requirements for sklearn

```
scikit-learn==1.3.2
numpy>=1.24,<2.0
joblib>=1.2.0
```

Add **scipy** if using sparse matrices or certain algorithms.

### Version Pinning

| Strategy | Example | Use Case |
|----------|---------|----------|
| **Exact** | scikit-learn==1.3.2 | Maximum reproducibility |
| **Minimum** | numpy>=1.24 | Allow patches |
| **Range** | numpy>=1.24,<2.0 | Avoid breaking changes |

---

## Reproducibility Checklist

1. **Set random_state** in all estimators (RandomForest, SVC, train_test_split, CV)
2. **Pin versions** of scikit-learn, numpy, and other key packages
3. **Store metadata** (version, metrics, timestamp) with each model
4. **Use version control** for code and configs
5. **Document data** sources and preprocessing steps
6. **Export environment** (pip freeze or conda export) with each release

---

## Summary

- Use **semantic versioning** and store **metadata** with models
- Set **random_state** everywhere for reproducible training and evaluation
- Track **environment** via pip freeze or conda export
- Prefer explicit **random_state** over global numpy seed
