# Optimization, Versioning, and Monitoring

## Table of Contents

1. [Inference Optimization](#1-inference-optimization)
2. [Model Versioning](#2-model-versioning)
3. [Production Monitoring](#3-production-monitoring)
4. [A/B Testing](#4-a-b-testing)
5. [Model Registry with MLflow](#5-model-registry-with-mlflow)

---

## 1. Inference Optimization

### 1.1 Operator Fusion

Fusing Conv-BN-ReLU reduces memory bandwidth and kernel launches:

```python
import torch.quantization as quantization

modules_to_fuse = [
    ['features.0', 'features.1', 'features.2'],
    ['features.3', 'features.4', 'features.5']
]
fused_model = quantization.fuse_modules(model, modules_to_fuse)
```

### 1.2 Memory Planning

**Channels-last** memory format improves cache utilization on GPUs:

```python
model = model.to(memory_format=torch.channels_last)
input_tensor = input_tensor.to(memory_format=torch.channels_last)
```

### 1.3 Batch Scheduling

Find the optimal batch size for throughput:

```python
def find_optimal_batch_size(model, input_shape, max_batch=64):
    best_throughput = 0
    optimal_batch = 1
    for batch_size in [2**i for i in range(int(np.log2(max_batch)) + 1)]:
        try:
            x = torch.randn(batch_size, *input_shape[1:]).to(device)
            times = [time_inference(model, x) for _ in range(20)]
            throughput = batch_size / np.mean(times)
            if throughput > best_throughput:
                best_throughput = throughput
                optimal_batch = batch_size
        except RuntimeError:
            break
    return optimal_batch
```

### 1.4 Dynamic Batching

Accumulate requests until a target batch size or timeout, then run inference. Implement with queues and worker threads, or use TorchServe's built-in batching.

### 1.5 Caching

Cache predictions for repeated inputs (e.g., by hash of input tensor or request ID) to avoid redundant inference.

### 1.6 Async Inference

Run inference in a separate thread or process to avoid blocking the main request handler:

```python
import concurrent.futures
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
future = executor.submit(model, input_tensor)
result = future.result(timeout=5.0)
```

### 1.7 TorchScript Optimization

```python
scripted = torch.jit.script(model)
scripted = torch.jit.freeze(scripted)
scripted = torch.jit.optimize_for_inference(scripted)
```

---

## 2. Model Versioning

### 2.1 Version Tracking

Use semantic versioning (major.minor.patch):

| Level | When to increment |
|-------|-------------------|
| Major | Breaking changes, new architecture |
| Minor | New features, significant accuracy change |
| Patch | Bug fixes, minor tweaks |

```python
def increment_version(current: str, level: str) -> str:
    major, minor, patch = map(int, current.split('.'))
    if level == "major":
        return f"{major+1}.0.0"
    elif level == "minor":
        return f"{major}.{minor+1}.0"
    else:
        return f"{major}.{minor}.{patch+1}"
```

### 2.2 Model Metadata

Store metadata with each version:

```python
metadata = {
    "model_name": "image_classifier",
    "version": "1.2.0",
    "description": "Improved regularization",
    "tags": ["cnn", "classification"],
    "model_hash": hashlib.md5(open(model_path, 'rb').read()).hexdigest(),
    "created_at": datetime.now().isoformat(),
    "metrics": {"accuracy": 0.92, "f1": 0.91},
    "dataset": "CIFAR-10",
    "hyperparameters": {"lr": 0.001, "epochs": 100}
}
```

### 2.3 Reproducibility

Record:

- PyTorch version
- Random seeds
- Dataset version and splits
- Hyperparameters
- Training environment (GPU, OS)

---

## 3. Production Monitoring

### 3.1 Latency Tracking

Track per-request inference time:

```python
start = time.perf_counter()
output = model(input_tensor)
latency_ms = (time.perf_counter() - start) * 1000
metrics.record_latency(latency_ms)
```

Report percentiles (p50, p95, p99) and mean.

### 3.2 Throughput Monitoring

Measure requests per second:

```python
requests_in_window = count_requests_since(timestamp - 60)
throughput = requests_in_window / 60.0
```

### 3.3 Data Drift Detection

Compare input statistics to a reference (e.g., training data):

```python
reference_mean = torch.mean(reference_data, dim=(0, 2, 3))
reference_std = torch.std(reference_data, dim=(0, 2, 3))

current_mean = torch.mean(batch_data, dim=(0, 2, 3))
current_std = torch.std(batch_data, dim=(0, 2, 3))

mean_drift = torch.mean(torch.abs(current_mean - reference_mean)).item()
std_drift = torch.mean(torch.abs(current_std - reference_std)).item()
overall_drift = (mean_drift + std_drift) / 2
drift_detected = overall_drift > threshold
```

### 3.4 Model Degradation Alerts

Set thresholds and alert when exceeded:

| Metric | Example Threshold |
|--------|-------------------|
| Latency p95 | > 500 ms |
| Error rate | > 1% |
| Throughput | < 10 req/s |
| Confidence | < 0.5 |
| Memory | > 2 GB |

```python
if avg_inference_time_ms > 500:
    alert("HIGH_LATENCY", f"Inference time: {avg_inference_time_ms}ms")
if error_rate > 0.05:
    alert("HIGH_ERROR_RATE", f"Error rate: {error_rate:.2%}")
```

### 3.5 Logging

Structured logs for debugging and auditing:

```python
import logging
logger = logging.getLogger("ModelProduction")
logger.info(f"Prediction - Model: {model_name}, Time: {latency_ms:.2f}ms, Confidence: {confidence:.3f}")
logger.error(f"Error - Model: {model_name}, Error: {str(e)}")
```

---

## 4. A/B Testing

### 4.1 Traffic Splitting

Route a fraction of traffic to the new model:

```python
import random

def get_variant(user_id, split_ratio=0.3):
    if user_id in assignments:
        return assignments[user_id]
    variant = "treatment" if random.random() < split_ratio else "control"
    assignments[user_id] = variant
    return variant
```

### 4.2 Statistical Significance

Use Chi-square for success rates, t-test for continuous metrics:

```python
from scipy import stats

control_successes = int(control_success_rate * control_n)
treatment_successes = int(treatment_success_rate * treatment_n)
chi2, p_value = stats.chi2_contingency([
    [control_successes, control_n - control_successes],
    [treatment_successes, treatment_n - treatment_successes]
])[:2]
is_significant = p_value < 0.05
```

### 4.3 Canary Deployment

Gradual rollout: 5% -> 20% -> 50% -> 100%, with validation at each stage.

### 4.4 Champion/Challenger

- **Champion**: Current production model (e.g., 80% traffic)
- **Challenger**: New model under test (e.g., 20% traffic)
- Promote challenger to champion when it wins on metrics and significance.

### 4.5 Deployment Strategies

| Strategy | Description |
|----------|-------------|
| Canary | Small percentage to new model, increase if metrics good |
| Blue-Green | Full switch after validation |
| Feature flags | Toggle models without redeploy |
| Shadow mode | Run new model in parallel, do not serve its output |

---

## 5. Model Registry with MLflow

### 5.1 Model Registration

```python
import mlflow
import mlflow.pytorch

mlflow.set_experiment("model_registry")
with mlflow.start_run():
    mlflow.log_params({"lr": 0.001, "epochs": 100})
    mlflow.log_metrics({"accuracy": 0.92, "f1": 0.90})
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        registered_model_name="image_classifier"
    )
```

### 5.2 Stage Transitions

MLflow stages: None, Staging, Production, Archived.

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="image_classifier",
    version="2",
    stage="Production",
    archive_existing_versions=True
)
```

### 5.3 Model Serving from Registry

```python
model_uri = "models:/image_classifier/Production"
model = mlflow.pytorch.load_model(model_uri)
```

Load by version:

```python
model_uri = "models:/image_classifier/2"
model = mlflow.pytorch.load_model(model_uri)
```

### 5.4 Registry Operations

```python
client.search_registered_models()
client.get_model_version("image_classifier", "2")
client.delete_model_version("image_classifier", "1")
client.update_model_version("image_classifier", "2", description="Improved model")
```

### 5.5 Integration with Deployment

1. Train and log model to MLflow
2. Transition to Staging for validation
3. Promote to Production
4. Deployment pipeline loads `models:/name/Production`

---

## Summary

| Topic | Key Takeaway |
|-------|--------------|
| Inference optimization | Fuse modules, use channels-last, tune batch size, TorchScript |
| Versioning | Semantic versions, metadata, reproducibility |
| Monitoring | Latency, throughput, drift, alerts, structured logging |
| A/B testing | Traffic split, significance tests, canary, champion/challenger |
| MLflow | Register models, use stages, serve from registry |
