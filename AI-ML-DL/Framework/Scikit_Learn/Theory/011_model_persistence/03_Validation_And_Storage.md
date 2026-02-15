# Validation and Storage

---

## Table of Contents

- [Overview](#overview)
- [Model Validation on Load](#model-validation-on-load)
- [Version Compatibility](#version-compatibility)
- [Storage Strategies](#storage-strategies)
- [File Organization](#file-organization)
- [Metadata and Checksums](#metadata-and-checksums)
- [Environment Tracking](#environment-tracking)
- [Testing Loaded Models](#testing-loaded-models)
- [Best Practices](#best-practices)

---

## Overview

**Validation** ensures loaded models behave correctly and are compatible with the current environment. **Storage** strategies (file layout, naming, metadata) support reproducibility and deployment. Proper validation catches version mismatches, corrupted files, and API changes.

| Concept | Purpose |
|---------|---------|
| **Version check** | Ensure sklearn/Python compatibility |
| **Sanity prediction** | Verify model produces valid output |
| **Metadata** | Store version, timestamp, config |
| **Checksums** | Detect file corruption |

---

## Model Validation on Load

After loading a model, validate that it works before deployment.

### Basic Validation

```python
import joblib

model = joblib.load("model.joblib")

# Check type
assert hasattr(model, "predict")

# Sanity check on known input
X_test = [[1.0, 2.0, 3.0, 4.0]]
y_pred = model.predict(X_test)
assert len(y_pred) == len(X_test)
assert not np.any(np.isnan(y_pred))
```

### Schema Validation

For production, validate input shape and dtypes before prediction:

```python
def validate_input(X, expected_features=4):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    assert X.shape[1] == expected_features
    assert np.isfinite(X).all()
    return X
```

---

## Version Compatibility

Scikit-learn models can break across versions. Store and check versions.

### Storing Version

```python
import sklearn
import joblib

metadata = {
    "sklearn_version": sklearn.__version__,
    "python_version": ".".join(map(str, __import__("sys").version_info[:2])),
}
joblib.dump({"model": model, "metadata": metadata}, "model_with_meta.joblib")
```

### Checking on Load

```python
data = joblib.load("model_with_meta.joblib")
model = data["model"]
meta = data.get("metadata", {})
current_sklearn = sklearn.__version__
if meta.get("sklearn_version") != current_sklearn:
    import warnings
    warnings.warn(f"Model trained with sklearn {meta.get('sklearn_version')}, "
                 f"loading with {current_sklearn}")
```

---

## Storage Strategies

### Single File

Store model (and optionally metadata) in one file. Simple for small models.

```python
joblib.dump(model, "model.joblib")
```

### Split Model and Config

Store model and config separately. Config can be human-editable.

```python
joblib.dump(model, "model.joblib")
with open("config.json", "w") as f:
    json.dump({"version": "1.0", "features": [...]}, f)
```

### Directory Layout

For multiple artifacts (model, scaler, feature names):

```
models/
  v1/
    model.joblib
    scaler.joblib
    metadata.json
  v2/
    ...
```

---

## File Organization

### Naming Conventions

- Include version: `model_v1.2.joblib`
- Include timestamp: `model_20240115.joblib`
- Include metric: `model_auc0.95.joblib`

### Compression

```python
joblib.dump(model, "model.joblib", compress=3)
# compress=0: none; 1-9: gzip level
```

---

## Metadata and Checksums

### Metadata Dict

```python
metadata = {
    "sklearn_version": sklearn.__version__,
    "created_at": datetime.now().isoformat(),
    "training_samples": len(X_train),
    "metrics": {"accuracy": 0.95, "f1": 0.93},
}
joblib.dump({"model": model, "metadata": metadata}, "model.joblib")
```

### Checksums

Use hashlib to detect corruption:

```python
import hashlib

def save_with_checksum(model, path):
    joblib.dump(model, path)
    with open(path, "rb") as f:
        checksum = hashlib.sha256(f.read()).hexdigest()
    with open(path + ".sha256", "w") as f:
        f.write(checksum)

def load_with_checksum(path):
    with open(path, "rb") as f:
        data = f.read()
    expected = open(path + ".sha256").read().strip()
    actual = hashlib.sha256(data).hexdigest()
    if expected != actual:
        raise ValueError("Checksum mismatch: file may be corrupted")
    return joblib.loads(data)
```

---

## Environment Tracking

Record Python version, package versions, and system info for reproducibility.

```python
import sklearn
import sys

def get_environment():
    return {
        "python": sys.version,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
    }
```

Store with model and verify on load if needed.

---

## Testing Loaded Models

Run a minimal test suite after loading:

```python
def test_loaded_model(model, X_sample, y_sample=None):
    y_pred = model.predict(X_sample)
    assert y_pred.shape[0] == X_sample.shape[0]
    if y_sample is not None:
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_sample, y_pred)
        assert acc > 0.5
```

---

## Best Practices

| Practice | Reason |
|----------|--------|
| Store **metadata** with model | Reproducibility, debugging |
| **Validate** on load | Catch corruption, version issues |
| Use **checksums** for critical models | Detect file corruption |
| **Version** file names or directories | Rollback, A/B testing |
| **Test** loaded model on holdout | Ensure it works before deploy |
