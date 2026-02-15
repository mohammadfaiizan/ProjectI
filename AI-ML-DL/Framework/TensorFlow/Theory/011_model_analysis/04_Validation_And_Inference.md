# Validation and Inference

## Table of Contents

1. [TensorBoard Integration](#1-tensorboard-integration)
2. [Inference Optimization](#2-inference-optimization)
3. [Model Validation Techniques](#3-model-validation-techniques)
4. [Error Analysis Tools](#4-error-analysis-tools)

---

## 1. TensorBoard Integration

**TensorBoard** visualizes training metrics, histograms, images, and model graphs via **tf.summary**.

### SummaryWriter

**tf.summary.create_file_writer** creates a writer for a log directory.

```python
logdir = "/tmp/tensorboard_logs"
writer = tf.summary.create_file_writer(logdir)

with writer.as_default():
    tf.summary.scalar("loss", 0.5, step=0)
    tf.summary.scalar("accuracy", 0.85, step=0)
writer.flush()
```

### tf.summary.scalar

Log scalar values (loss, accuracy, learning rate) over steps.

```python
tf.summary.scalar("loss", loss_value, step=step)
tf.summary.scalar("accuracy", acc_value, step=step)
```

### tf.summary.histogram

Log distribution of tensors (weights, activations, gradients).

```python
tf.summary.histogram("dense_weights", layer.get_weights()[0], step=step)
```

### tf.summary.image

Log images for visualization (inputs, reconstructions, attention maps).

```python
tf.summary.image("sample", img_tensor, step=step, max_outputs=4)
```

### tf.summary.trace (Graph)

Record model graph for visualization in TensorBoard.

```python
tf.summary.trace_on(graph=True, profiler=False)
model(x)
with tf.summary.record_if(True):
    tf.summary.trace_export(name="model_graph", step=0)
```

### Keras Callback

**TensorBoard** callback automates logging during fit.

```python
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
model.fit(x, y, callbacks=[tb_callback])
```

---

## 2. Inference Optimization

Optimize inference for **latency** and **throughput** using **tf.function**, **XLA**, and **batch inference**.

### tf.function

**@tf.function** traces the function to a graph, enabling optimizations (constant folding, op fusion).

```python
@tf.function
def inference(model, x):
    return model(x, training=False)
```

### XLA Compilation

**jit_compile=True** enables XLA (Accelerated Linear Algebra) for further speedups on supported hardware.

```python
@tf.function(jit_compile=True)
def inference_xla(model, x):
    return model(x, training=False)
```

### Batch Inference

Process multiple samples at once for higher throughput. Amortize kernel launch and memory transfer overhead.

```python
# Single-sample: high latency per sample
for x in samples:
    pred = model(x, training=False)

# Batch: lower latency per sample
preds = model(batch_x, training=False)
```

### Best Practices

| Practice | Benefit |
|----------|---------|
| Use tf.function | Graph optimization |
| Batch when possible | Higher throughput |
| XLA on GPU/TPU | Additional speedup |
| Warmup before benchmark | Stable timings |

---

## 3. Model Validation Techniques

**K-fold cross-validation** and **metrics computation** provide robust model evaluation.

### K-Fold Cross-Validation

Split data into K folds; train on K-1, validate on 1; rotate. Report mean and std of metrics.

```python
def kfold_cv(x, y, n_splits=5, epochs=10):
    n = len(x)
    indices = np.random.permutation(n)
    fold_size = n // n_splits
    scores = []
    for k in range(n_splits):
        val_idx = indices[k*fold_size:(k+1)*fold_size]
        train_idx = np.concatenate([indices[:k*fold_size], indices[(k+1)*fold_size:]])
        model = build_model()
        model.fit(x[train_idx], y[train_idx], epochs=epochs)
        _, acc = model.evaluate(x[val_idx], y[val_idx])
        scores.append(acc)
    return np.mean(scores), np.std(scores)
```

### Metrics

**model.compile** accepts multiple metrics. **evaluate** returns them in order.

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)
results = model.evaluate(x_test, y_test)
# results: [loss, accuracy, precision, recall]
```

### Stratified K-Fold

For imbalanced classes, use **StratifiedKFold** to preserve class distribution in each fold.

---

## 4. Error Analysis Tools

**Misclassification analysis** and **per-class performance** help identify model weaknesses.

### Misclassification Identification

Find indices where predictions differ from labels.

```python
y_pred = np.argmax(model.predict(x), axis=1)
misclassified = np.where(y_pred != y_true)[0]
print(f"Misclassified: {len(misclassified)} / {len(y_true)}")
```

### Per-Class Precision and Recall

Compute precision and recall for each class.

```python
def per_class_metrics(y_true, y_pred, num_classes):
    precision, recall = [], []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        pred_c = np.sum(y_pred == c)
        actual_c = np.sum(y_true == c)
        precision.append(tp / pred_c if pred_c > 0 else 0)
        recall.append(tp / actual_c if actual_c > 0 else 0)
    return precision, recall
```

### Confusion Matrix

**tf.math.confusion_matrix** summarizes predictions vs labels.

```python
conf_matrix = tf.math.confusion_matrix(y_true, y_pred)
# Rows = true, cols = predicted
```

### Error Analysis Workflow

1. Compute overall accuracy.
2. Identify worst-performing classes (low recall).
3. Inspect confusion matrix for systematic confusions.
4. Analyze misclassified samples (visualization, feature stats).
5. Iterate on data or model.

---

## Summary

| Topic | Key APIs | Use Case |
|-------|----------|----------|
| TensorBoard | tf.summary, SummaryWriter | Training visualization |
| Inference | tf.function, jit_compile | Latency/throughput |
| Validation | K-fold, metrics | Robust evaluation |
| Error analysis | confusion_matrix, per-class | Model improvement |
