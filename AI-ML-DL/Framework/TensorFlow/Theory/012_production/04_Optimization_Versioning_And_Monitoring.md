# Optimization, Versioning, and Monitoring

## Table of Contents

1. [Model Optimization Overview](#1-model-optimization-overview)
2. [Pruning](#2-pruning)
3. [Weight Clustering](#3-weight-clustering)
4. [Model Versioning](#4-model-versioning)
5. [Production Monitoring](#5-production-monitoring)
6. [A/B Testing and Canary Deployment](#6-ab-testing-and-canary-deployment)

---

## 1. Model Optimization Overview

Model optimization reduces size, latency, and memory footprint for deployment. Common techniques include **pruning**, **clustering**, and **quantization** (covered in TF Lite).

### Optimization Goals

- **Smaller models**: Faster download, less storage
- **Faster inference**: Lower latency, higher throughput
- **Lower memory**: Enable deployment on constrained devices

### TensorFlow Model Optimization Toolkit

The **tensorflow_model_optimization** package provides pruning and clustering APIs that integrate with Keras.

```bash
pip install tensorflow_model_optimization
```

---

## 2. Pruning

**Pruning** removes connections (weights) with small magnitudes. Sparse models can be compressed and accelerated on hardware that supports sparse computation.

### Prune Low Magnitude

```python
import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

model = tf.keras.Sequential([...])
pruned_model = prune_low_magnitude(
    model,
    pruning_schedule=tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,
        begin_step=0,
        end_step=1000
    )
)
```

### PolynomialDecay Schedule

**PolynomialDecay** gradually increases sparsity during training:

- **initial_sparsity**: Starting sparsity (often 0)
- **final_sparsity**: Target sparsity (e.g., 0.5 for 50% zeros)
- **begin_step**: Step to start pruning
- **end_step**: Step to reach final sparsity

Sparsity increases smoothly between these steps.

### Training with Pruning

```python
pruned_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
pruned_model.fit(x_train, y_train, epochs=5, callbacks=callbacks)
```

**UpdatePruningStep** must be called each step to update the pruning mask.

### Strip Pruning

Before deployment, remove pruning wrappers to get a standard model with sparse weights:

```python
final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
```

### Pruning Comparison

| Schedule | Use Case |
|----------|----------|
| ConstantSparsity | Fixed sparsity from start |
| PolynomialDecay | Gradual increase during training |
| PolynomialDecay with fine-tuning | Prune then fine-tune |

---

## 3. Weight Clustering

**Clustering** groups weights into a small number of centroids. Each weight is replaced by its centroid index, reducing the number of unique values and enabling weight sharing.

### Cluster Weights

```python
import tensorflow_model_optimization as tfmot

cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

clustered_model = cluster_weights(
    model,
    number_of_clusters=16,
    cluster_centroids_init=CentroidInitialization.KMEANS_PLUS_PLUS
)
```

### Parameters

- **number_of_clusters**: Number of centroid values (e.g., 16, 32)
- **cluster_centroids_init**: How to initialize centroids (KMEANS_PLUS_PLUS, LINEAR, RANDOM)

### Fine-tuning

Train or fine-tune the clustered model to recover accuracy:

```python
clustered_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
clustered_model.fit(x_train, y_train, epochs=3)
```

### Strip Clustering

```python
final_model = tfmot.clustering.keras.strip_clustering(clustered_model)
```

The stripped model has full-precision weights but with only `number_of_clusters` unique values per layer, enabling efficient compression.

### Pruning vs Clustering

| Technique | Mechanism | Compression | Typical Use |
|-----------|-----------|-------------|-------------|
| Pruning | Set small weights to zero | Sparse storage | General size reduction |
| Clustering | Quantize to centroids | Weight sharing | Fixed-point hardware |

---

## 4. Model Versioning

**Model versioning** tracks which model is in production and enables rollback, A/B testing, and audit trails.

### Directory-Based Versioning

```
models/
  my_model/
    1/          # Version 1
    2/          # Version 2
    20240115/   # Timestamp-based
```

### Metadata

Store metadata alongside each version:

```json
{
  "version": 2,
  "created": "2024-01-15T10:00:00Z",
  "metrics": {"accuracy": 0.92, "auc": 0.95},
  "training_config": {"epochs": 10, "batch_size": 32}
}
```

### Semantic Versioning

Use semantic versions (major.minor.patch) for compatibility signaling:

- **Major**: Breaking changes (e.g., input shape change)
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, no interface change

### Version Selection

- **Latest**: Always serve the newest version
- **Stable**: Serve a version marked as production-ready
- **Canary**: Serve a new version to a fraction of traffic

---

## 5. Production Monitoring

Monitoring ensures models perform correctly in production and helps detect **data drift**, **model drift**, and failures.

### Key Metrics

| Metric | Description |
|--------|-------------|
| **Latency** | Inference time per request (p50, p95, p99) |
| **Throughput** | Requests per second |
| **Error rate** | Failed predictions or timeouts |
| **Input distribution** | Mean, std, min, max of features |
| **Output distribution** | Confidence, entropy, predicted class distribution |

### Data Drift

**Data drift** occurs when the distribution of input data changes over time. Monitor:

- Feature means and standard deviations
- Categorical value frequencies
- Missing value rates

Compare production inputs to a baseline (e.g., training or validation set).

### Model Drift

**Model drift** (concept drift) occurs when the relationship between inputs and outputs changes. Monitor:

- Prediction confidence over time
- Accuracy on labeled samples (if available)
- Business metrics (e.g., conversion rate)

### Logging

Log per-request or batched:

- Request ID, timestamp
- Input hash or summary stats
- Prediction, confidence
- Latency

Use structured logging (JSON) for easy querying.

### Alerting

Set thresholds for:

- Latency above p99
- Error rate above X%
- Input distribution deviation from baseline
- Drop in confidence or accuracy

---

## 6. A/B Testing and Canary Deployment

**A/B testing** compares two or more model versions. **Canary deployment** gradually rolls out a new version to a small fraction of traffic.

### Traffic Splitting

Route a fraction of requests to each model:

| Model | Traffic |
|-------|---------|
| A (current) | 90% |
| B (new) | 10% |

Implement via:

- **Load balancer**: Route by header or random
- **TF Serving**: Version labels; client specifies version
- **Application logic**: Random choice per request

### Canary Stages

1. **5%**: Deploy new model to 5% traffic; monitor for 24–48 hours
2. **25%**: If metrics pass, increase to 25%
3. **50%**: Further increase
4. **100%**: Full rollout; retire old version

### Metrics to Compare

- Latency (p50, p95, p99)
- Error rate
- Business metrics (e.g., CTR, conversion)
- User feedback (if available)

### Rollback

If the new model underperforms:

- Reduce its traffic to 0%
- Route all traffic to the previous version
- Investigate and fix before redeploying

### TF Serving Version Labels

TF Serving supports version labels (e.g., `stable`, `canary`). Update labels to point to different versions without changing client code:

```
PUT /v1/models/my_model/versions/2/labels/canary
```

---

## Summary

- **Pruning** removes small-magnitude weights; use **PolynomialDecay** for gradual sparsity
- **Clustering** quantizes weights to centroids; enables weight sharing
- **Versioning** tracks model versions and metadata for rollback and A/B
- **Monitoring** tracks latency, drift, and errors; use structured logging
- **A/B testing** and **canary deployment** enable safe rollout of new models
