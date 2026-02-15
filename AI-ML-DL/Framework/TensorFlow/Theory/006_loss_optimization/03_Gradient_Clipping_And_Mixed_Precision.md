# Gradient Clipping and Mixed Precision

## Table of Contents

1. [Gradient Clipping Methods](#1-gradient-clipping-methods)
2. [clipnorm and clipvalue](#2-clipnorm-and-clipvalue)
3. [global_clipnorm](#3-global_clipnorm)
4. [Mixed Precision Training](#4-mixed-precision-training)
5. [Policy and LossScaleOptimizer](#5-policy-and-lossscaleoptimizer)

---

## 1. Gradient Clipping Methods

**Gradient clipping** limits gradient magnitude to prevent exploding gradients, which can cause training instability, NaN loss, or divergence. Common in RNNs and Transformers.

### Why Clip Gradients

- **Exploding gradients**: Deep networks can produce very large gradients.
- **Training stability**: Clipping keeps updates bounded.
- **Numerical stability**: Prevents overflow in float16.

### Available Methods

| Method | Scope | Behavior |
|--------|-------|----------|
| **clipnorm** | Per-variable | Scale gradient so L2 norm <= clipnorm |
| **clipvalue** | Per-variable | Clamp each element to [-clipvalue, clipvalue] |
| **global_clipnorm** | All variables | Scale all gradients so global L2 norm <= clipnorm |

---

## 2. clipnorm and clipvalue

### clipnorm

**clipnorm** clips each gradient tensor by its L2 norm. If `||g|| > clipnorm`, scale: `g = g * clipnorm / ||g||`.

```python
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, clipnorm=1.0)
model.compile(optimizer=sgd, loss='mse')
```

Applied per variable independently. Useful when individual parameter gradients can explode.

### clipvalue

**clipvalue** clamps each gradient element to `[-clipvalue, clipvalue]`. Element-wise operation.

```python
adam = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=0.5)
model.compile(optimizer=adam, loss='mse')
```

Can zero out gradients that exceed the threshold; use when you want hard bounds per element.

### Manual Clipping with tf.clip_by_*

```python
grads = tape.gradient(loss, model.trainable_variables)
grads_clipped, _ = tf.clip_by_global_norm(grads, 1.0)
optimizer.apply_gradients(zip(grads_clipped, model.trainable_variables))

# Or per-value:
grads_clipped = [tf.clip_by_value(g, -0.5, 0.5) for g in grads]
```

---

## 3. global_clipnorm

**global_clipnorm** computes the L2 norm over all gradients concatenated, then scales them jointly so the global norm does not exceed the threshold. Ensures the overall update direction is bounded while preserving relative magnitudes.

```python
adam = tf.keras.optimizers.Adam(learning_rate=0.001, global_clipnorm=1.0)
model.compile(optimizer=adam, loss='mse')
```

Preferred when you want a single global bound across all parameters, as in many Transformer training setups.

### When to Use Which

- **clipnorm**: Per-layer or per-tensor control.
- **clipvalue**: When you need hard per-element limits.
- **global_clipnorm**: Standard choice for Transformers and large models; single hyperparameter.

---

## 4. Mixed Precision Training

**Mixed precision** uses float16 (or bfloat16) for most computations and float32 for sensitive operations. Benefits:

- **Speed**: float16 ops are faster on modern GPUs and TPUs.
- **Memory**: Half the memory for activations and gradients.
- **Throughput**: Larger batch sizes or models.

### Potential Issues

- **Underflow**: Small float16 values can become zero.
- **Overflow**: Large values can become inf.
- **Accuracy**: Some operations need float32 for stability.

**Loss scaling** and keeping master weights in float32 address these issues.

---

## 5. Policy and LossScaleOptimizer

### Policy

A **Policy** defines compute and variable dtypes for layers.

```python
policy = tf.keras.mixed_precision.Policy('mixed_float16')
print(policy.compute_dtype)   # float16
print(policy.variable_dtype)   # float32
```

Common policies:

- **float32**: Full precision (default).
- **mixed_float16**: Compute in float16, variables in float32.
- **mixed_bfloat16**: Compute in bfloat16 (TPUs, some GPUs).
- **float16**: All float16 (rare; risk of instability).

### set_global_policy

```python
tf.keras.mixed_precision.set_global_policy('mixed_float16')
current = tf.keras.mixed_precision.global_policy()
# All new layers use mixed_float16
```

### Output Layer Dtype

For classification, keep the output layer in float32 to avoid numerical issues with softmax/cross-entropy:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax', dtype='float32')
])
```

### LossScaleOptimizer

**LossScaleOptimizer** wraps an optimizer and scales the loss before backprop to avoid float16 underflow. Gradients are unscaled before the optimizer step.

```python
opt = tf.keras.optimizers.Adam(0.001)
opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
model.compile(optimizer=opt, loss='mse')
```

With `model.compile`, Keras can apply loss scaling automatically when using mixed_float16. For custom training loops, wrap the optimizer explicitly.

### Custom Training Loop with Mixed Precision

```python
opt = tf.keras.optimizers.Adam(0.001)
opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = loss_fn(y, pred)
        scaled_loss = opt.get_scaled_loss(loss)
    scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
    grads = opt.get_unscaled_gradients(scaled_grads)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss
```

### Resetting Policy

Always reset to float32 when done if you changed the global policy:

```python
tf.keras.mixed_precision.set_global_policy('float32')
```
