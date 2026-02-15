# Registry, Export, and Practices

---

## Table of Contents

- [Overview](#overview)
- [Model Registry Patterns](#model-registry-patterns)
- [ONNX Export](#onnx-export)
- [Cross-Framework Export](#cross-framework-export)
- [Model Cards](#model-cards)
- [Production Practices](#production-practices)
- [Deployment Considerations](#deployment-considerations)
- [Security and Trust](#security-and-trust)
- [Best Practices](#best-practices)

---

## Overview

**Model registry** patterns organize models for versioning, promotion, and rollback. **Export** to ONNX or other formats enables deployment outside Python. **Practices** cover production readiness, security, and maintainability.

| Concept | Purpose |
|---------|---------|
| **Registry** | Central store for model versions |
| **ONNX** | Cross-framework, cross-language deployment |
| **Model cards** | Documentation, limitations, ethics |
| **Reproducibility** | Same environment, same results |

---

## Model Registry Patterns

### Simple File-Based Registry

```
registry/
  models/
    prod/
      current -> v2
      v1/
      v2/
    staging/
      v3/
```

### Metadata Index

```python
registry = {
    "v1": {"path": "models/v1/model.joblib", "accuracy": 0.92, "promoted": False},
    "v2": {"path": "models/v2/model.joblib", "accuracy": 0.95, "promoted": True},
}
```

### MLflow and Similar

Tools like **MLflow** provide model registry, versioning, and deployment. Integrate with sklearn via `mlflow.sklearn.log_model()`.

---

## ONNX Export

**ONNX** (Open Neural Network Exchange) enables deployment in C++, JavaScript, mobile, etc. Use **skl2onnx** or **onnxruntime** for sklearn models.

### skl2onnx

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [("float_input", FloatTensorType([None, 4]))]
onx = convert_sklearn(model, initial_types=initial_type)
with open("model.onnx", "wb") as f:
    f.write(onx.SerializeToString())
```

### Inference with ONNX Runtime

```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: X.astype(np.float32)})
```

### Limitations

- Not all sklearn estimators are supported
- Complex pipelines may require manual conversion
- Check skl2onnx compatibility for your model

---

## Cross-Framework Export

### CoreML (Apple)

Use **coremltools** to convert sklearn models for iOS/macOS.

### PMML

**PMML** (Predictive Model Markup Language) is XML-based. Use **sklearn2pmml** or **nyoka** for export. Less common than ONNX.

### Pickle/Joblib

**Pickle** and **joblib** are Python-specific. Use for Python-only deployment. Prefer **joblib** for sklearn.

---

## Model Cards

**Model cards** document model purpose, training data, limitations, and ethical considerations. Store as JSON or Markdown alongside the model.

### Example Structure

```json
{
  "model_id": "classifier_v1",
  "description": "Binary classifier for X",
  "training_data": "dataset_v2, 10k samples",
  "metrics": {"accuracy": 0.95, "f1": 0.93},
  "limitations": "Trained on 2020 data; may not generalize",
  "bias_considerations": "Class imbalance was addressed with class_weight"
}
```

---

## Production Practices

### Preprocessing in Pipeline

Always save the full **Pipeline** (preprocessing + model). Loading raw features and applying the same preprocessing manually is error-prone.

```python
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression()),
])
pipe.fit(X_train, y_train)
joblib.dump(pipe, "pipeline.joblib")
```

### Input Validation

Validate inputs (shape, dtype, range, missing values) before prediction. Reject invalid inputs with clear errors.

### Logging and Monitoring

Log predictions, latencies, and errors. Monitor for distribution shift and performance degradation.

---

## Deployment Considerations

### Latency

- **joblib** load can be slow for large models; consider lazy loading or warm-up
- **ONNX** often faster for inference in production runtimes
- Use **n_jobs=1** or small batches if thread contention is an issue

### Memory

- Large models (e.g., many trees) can use significant memory
- **compress** in joblib reduces disk size but may increase load time
- Consider model distillation or pruning for edge deployment

---

## Security and Trust

### Pickle Security

**Pickle** and **joblib** can execute arbitrary code when loading. Only load models from trusted sources. Prefer **skops** for secure loading when available.

### Integrity

Use checksums or signatures to verify model files have not been tampered with.

---

## Best Practices

| Practice | Reason |
|----------|--------|
| Save full **Pipeline** | Consistent preprocessing |
| Use **model registry** | Versioning, rollback |
| Document with **model cards** | Transparency, maintenance |
| **Validate** inputs and outputs | Robustness |
| Use **ONNX** for non-Python deployment | Portability |
| **Version** dependencies | Reproducibility |
| **Test** before promotion | Quality assurance |
