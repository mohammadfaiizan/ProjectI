# Regression, Classification, and Custom Losses

## Table of Contents

1. [Regression Losses](#1-regression-losses)
2. [Classification Losses](#2-classification-losses)
3. [Ranking and Contrastive Losses](#3-ranking-and-contrastive-losses)
4. [Custom Loss Functions](#4-custom-loss-functions)
5. [Reduction Modes](#5-reduction-modes)

---

## 1. Regression Losses

Regression losses measure the discrepancy between continuous predictions and targets. TensorFlow provides several built-in options for different use cases.

### MeanSquaredError (MSE)

**MSE** computes the average of squared differences: `L = mean((y_true - y_pred)^2)`. Sensitive to outliers; penalizes large errors quadratically.

```python
import tensorflow as tf

mse = tf.keras.losses.MeanSquaredError()
y_true = tf.constant([[1.0, 2.0], [3.0, 4.0]])
y_pred = tf.constant([[1.2, 1.8], [2.7, 4.2]])
loss = mse(y_true, y_pred)
```

### MeanAbsoluteError (MAE)

**MAE** uses absolute differences: `L = mean(|y_true - y_pred|)`. More robust to outliers than MSE; gradients are constant.

```python
mae = tf.keras.losses.MeanAbsoluteError()
loss = mae(y_true, y_pred)
```

### Huber Loss

**Huber** combines MSE and MAE: quadratic for small errors, linear for large. Controlled by **delta**; smooth transition at |error| = delta.

```python
huber = tf.keras.losses.Huber(delta=1.0)
loss = huber(y_true, y_pred)
```

### LogCosh

**LogCosh** approximates `log(cosh(x))` for errors. Smooth, twice differentiable; behaves like MSE for small errors and like MAE for large.

```python
logcosh = tf.keras.losses.LogCosh()
loss = logcosh(y_true, y_pred)
```

### MeanSquaredLogarithmicError (MSLE)

**MSLE** uses `(log(1 + y_pred) - log(1 + y_true))^2`. Suitable when targets span orders of magnitude; requires positive values.

```python
msle = tf.keras.losses.MeanSquaredLogarithmicError()
loss = msle(y_true, y_pred)  # y_true, y_pred must be positive
```

### Regression Loss Comparison

| Loss | Outlier Robustness | Gradient Behavior | Use Case |
|------|-------------------|-------------------|----------|
| MSE | Low | Linear in error | Standard regression |
| MAE | High | Constant | Robust regression |
| Huber | Medium | Smooth | General purpose |
| LogCosh | Medium | Smooth | Smooth gradients |
| MSLE | Medium | Scale-invariant | Wide target range |

---

## 2. Classification Losses

Classification losses compare predicted class probabilities (or logits) with true labels.

### CategoricalCrossentropy

**CategoricalCrossentropy** expects one-hot encoded targets and probability outputs. Formula: `-sum(y_true * log(y_pred))`.

```python
cce = tf.keras.losses.CategoricalCrossentropy()
y_true = tf.constant([[0, 1, 0], [1, 0, 0]])  # one-hot
y_pred = tf.constant([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05]])  # probabilities
loss = cce(y_true, y_pred)
```

### SparseCategoricalCrossentropy

**SparseCategoricalCrossentropy** uses integer class indices instead of one-hot. More memory efficient for many classes.

```python
scce = tf.keras.losses.SparseCategoricalCrossentropy()
y_true = tf.constant([1, 0])  # class indices
loss = scce(y_true, y_pred)
```

### BinaryCrossentropy

**BinaryCrossentropy** for binary or multi-label classification. Each output is independent; targets in [0, 1].

```python
bce = tf.keras.losses.BinaryCrossentropy()
y_bin_true = tf.constant([[0, 1], [1, 0]])
y_bin_pred = tf.constant([[0.1, 0.9], [0.85, 0.15]])
loss = bce(y_bin_true, y_bin_pred)
```

### from_logits

When the model outputs **logits** (pre-softmax/sigmoid), set `from_logits=True` for numerical stability. The loss applies softmax/sigmoid internally.

```python
cce_logits = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
y_pred_logits = tf.constant([[0.5, 2.0, 0.3], [2.5, -1.0, -0.5]])
loss = cce_logits(y_true, y_pred_logits)
```

---

## 3. Ranking and Contrastive Losses

These losses learn embeddings or similarity structures rather than direct predictions.

### Triplet Loss

**Triplet loss** uses anchor, positive, and negative samples. Minimizes distance to positive while maximizing distance to negative, with a **margin**.

```python
def triplet_loss(anchor, positive, negative, margin=0.5):
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))
```

### Contrastive Loss

**Contrastive loss** for siamese networks. Same-class pairs should be close; different-class pairs should exceed a margin.

```python
def contrastive_loss(y_true, y_pred, margin=1.0):
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0.0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)
```

### Cosine Similarity Loss

**Cosine similarity loss** encourages predicted and target vectors to align. Uses `1 - mean(cos_sim)`.

```python
def cosine_similarity_loss(y_true, y_pred):
    y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
    y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)
    cos_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
    return 1.0 - tf.reduce_mean(cos_sim)
```

---

## 4. Custom Loss Functions

### Custom Loss as Function

A simple callable taking `(y_true, y_pred)` and returning a scalar tensor.

```python
def custom_mse_weighted(y_true, y_pred, weight_positive=2.0):
    squared = tf.square(y_true - y_pred)
    weights = tf.where(y_true > 0, weight_positive, 1.0)
    return tf.reduce_mean(weights * squared)

model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_mse_weighted(y_true, y_pred, 2.0))
```

### Custom Loss as Class

Subclass `tf.keras.losses.Loss` for serialization and configuration. Override `call()` and optionally `get_config()`.

```python
class WeightedMSELoss(tf.keras.losses.Loss):
    def __init__(self, weight_positive=2.0, name="weighted_mse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.weight_positive = weight_positive

    def call(self, y_true, y_pred):
        squared = tf.square(y_true - y_pred)
        weights = tf.where(y_true > 0, self.weight_positive, 1.0)
        return tf.reduce_mean(weights * squared)

    def get_config(self):
        config = super().get_config()
        config.update({"weight_positive": self.weight_positive})
        return config
```

### Using Custom Loss in Model

```python
model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(4,))])
model.compile(optimizer='adam', loss=WeightedMSELoss(weight_positive=2.0))
model.fit(x_train, y_train, epochs=5)
```

---

## 5. Reduction Modes

Losses support **reduction** to control how per-sample losses are aggregated.

| Reduction | Behavior | Output Shape |
|-----------|----------|--------------|
| **SUM_OVER_BATCH_SIZE** | mean over batch (default) | scalar |
| **SUM** | sum over batch | scalar |
| **NONE** | no reduction | (batch_size,) |

```python
mse_sum = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
mse_none = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

loss_sum = mse_sum(y_true, y_pred)      # scalar
loss_per_sample = mse_none(y_true, y_pred)  # shape (batch_size,)
```

Sample weights can be passed to loss calls for weighted averaging when using `SUM_OVER_BATCH_SIZE` or `SUM`.
