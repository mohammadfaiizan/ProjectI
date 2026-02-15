# Serialization Formats

---

## Table of Contents

- [Overview](#overview)
- [Joblib](#joblib)
- [Pickle](#pickle)
- [Skops](#skops)
- [ONNX Export](#onnx-export)
- [Format Comparison](#format-comparison)

---

## Overview

**Model persistence** is the process of saving trained scikit-learn models to disk and loading them later for inference or deployment. Scikit-learn recommends **joblib** for most use cases because it is optimized for large NumPy arrays. **Pickle** is the Python standard but less efficient for numerical data. **Skops** provides secure serialization with type checking. **ONNX** enables cross-framework deployment.

---

## Joblib

**joblib** is part of the scikit-learn ecosystem and is the recommended way to persist scikit-learn models. It uses pickle under the hood but handles large NumPy arrays efficiently through memory-mapping and compression.

### joblib.dump

```python
import joblib
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "model.joblib")
joblib.dump(model, "model_compressed.joblib", compress=3)
```

### joblib.load

```python
loaded_model = joblib.load("model.joblib")
predictions = loaded_model.predict(X_test)
```

### compress Parameter

The **compress** parameter controls compression level (0-9):

| Value | Description |
|-------|-------------|
| **0** | No compression |
| **1-9** | gzip compression; higher = smaller file, slower I/O |
| **3** | Good default balance |

```python
joblib.dump(model, "model.joblib", compress=3)
```

### When to Use Joblib

- Default choice for scikit-learn models
- Models with large NumPy arrays (e.g., Random Forest, neural nets)
- Python-only deployment
- Need for compression and memory-mapped loading

---

## Pickle

**pickle** is Python's built-in serialization module. Any Python object can be pickled, including scikit-learn estimators. Joblib uses pickle internally but adds optimizations for arrays.

### pickle.dump and pickle.load

```python
import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(model, f, protocol=4)

with open("model.pkl", "rb") as f:
    loaded_model = pickle.load(f)
```

### protocol Parameter

The **protocol** parameter controls pickle format:

| Protocol | Python | Format | Notes |
|----------|--------|--------|-------|
| **0-3** | All | ASCII | Human-readable, larger files |
| **4** | 3.4+ | Binary | Default in Python 3.4-3.7 |
| **5** | 3.8+ | Binary | Out-of-band data, faster |

```python
pickle.dump(model, f, protocol=5)
```

### Pickle Security Warning

**Unpickling** data from untrusted sources can execute arbitrary code. Never load pickle files from unknown origins. Use **skops** for safer loading when provenance is uncertain.

---

## Skops

**Skops** (scikit-learn serialization) provides secure serialization with **trusted type** checking. Before loading, you can inspect which types would be instantiated and reject untrusted ones.

### skops.io.dump and skops.io.load

```python
import skops.io as sio

sio.dump(model, "model.skops")
loaded = sio.load("model.skops", trusted=True)
```

### get_untrusted_types

Inspect which types would be loaded before trusting:

```python
untrusted = sio.get_untrusted_types("model.skops")
print(untrusted)

if not untrusted:
    model = sio.load("model.skops", trusted=False)
```

### trusted Parameter

| Value | Behavior |
|-------|----------|
| **trusted=True** | Load all types (equivalent to pickle) |
| **trusted=False** | Raise if any untrusted type present |
| **trusted=set** | Custom set of trusted type names |

### When to Use Skops

- Loading models from untrusted or external sources
- Need to audit what types are deserialized
- Compliance or security requirements

---

## ONNX Export

**ONNX** (Open Neural Network Exchange) is a format for representing machine learning models across frameworks. **skl2onnx** converts scikit-learn models to ONNX for deployment in non-Python runtimes.

### convert_sklearn

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [("float_input", FloatTensorType([None, n_features]))]
onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)

with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

### to_onnx (Alternative)

Some packages provide a `to_onnx` helper for simpler conversion.

### Input Type Specification

You must specify the input shape and type:

```python
FloatTensorType([None, 4])  # batch_size variable, 4 features
```

### Running ONNX Models

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("model.onnx")
inputs = {"float_input": X_test.astype(np.float32)}
predictions = sess.run(None, inputs)[0]
```

### When to Use ONNX

- Deploy to C#, Java, JavaScript, or mobile
- Use ONNX Runtime for optimized inference
- Integrate with TensorFlow, PyTorch ecosystems
- Edge or embedded deployment

---

## Format Comparison

| Format | Use Case | Security | Cross-Platform | Array Optimization |
|--------|----------|----------|----------------|-------------------|
| **joblib** | Default for sklearn | Same as pickle | Python only | Yes |
| **pickle** | General Python | Unsafe from untrusted | Python only | No |
| **skops** | Secure loading | Trusted types | Python only | Yes |
| **ONNX** | Cross-framework | Safer (no code exec) | Broad | N/A (different format) |

---

## Summary

- **joblib** is the default for scikit-learn; use **compress=3** for smaller files
- **pickle** works but joblib is preferred for models with arrays
- **skops** adds trusted-type checking for secure loading
- **ONNX** enables deployment outside Python via skl2onnx and ONNX Runtime
