# Gradient Control and Clipping

## Table of Contents

1. [Why Gradient Clipping?](#1-why-gradient-clipping)
2. [Gradient Clipping Methods](#2-gradient-clipping-methods)
3. [clip_by_norm](#3-clip_by_norm)
4. [clip_by_value](#4-clip_by_value)
5. [clip_by_global_norm](#5-clip_by_global_norm)
6. [GradientTape with tf.function](#6-gradienttape-with-tffunction)
7. [Gradient Accumulation](#7-gradient-accumulation)
8. [Integration with Optimizers](#8-integration-with-optimizers)

---

## 1. Why Gradient Clipping?

**Exploding gradients** occur when gradients grow very large, causing unstable updates and NaN losses. Common in:
- Recurrent networks (RNNs, LSTMs)
- Deep networks
- Networks with unstable activations

**Gradient clipping** limits the magnitude of gradients before applying them, stabilizing training while preserving direction (for norm-based clipping).

**Key concepts:**
- **Norm clipping:** Scale gradients so their norm does not exceed a threshold
- **Value clipping:** Clamp each gradient element to a range
- **Global norm:** Clip based on the norm of all gradients combined (for multi-parameter models)

---

## 2. Gradient Clipping Methods

| Method | Scope | Effect |
|--------|-------|--------|
| clip_by_norm | Per tensor | Scale tensor so norm <= max_norm |
| clip_by_value | Per element | Clamp each element to [clip_value_min, clip_value_max] |
| clip_by_global_norm | All tensors | Scale all so global norm <= max_norm |

---

## 3. clip_by_norm

**tf.clip_by_norm(t, max_norm)** scales the tensor `t` so that its L2 norm does not exceed `max_norm`. If the norm is already below the threshold, the tensor is unchanged.

```python
grads = tape.gradient(loss, w)
grads_clipped = tf.clip_by_norm(grads, max_norm=1.0)
optimizer.apply_gradients(zip([grads_clipped], [w]))
```

**Formula:** If \(\|g\| > \text{max\_norm}\), then \(g \leftarrow g \cdot \frac{\text{max\_norm}}{\|g\|}\).

**Properties:**
- Preserves gradient direction
- Only scales when norm exceeds threshold
- Applied per tensor (each parameter group separately if you clip each)

---

## 4. clip_by_value

**tf.clip_by_value(t, clip_value_min, clip_value_max)** clamps each element to the specified range.

```python
grads = tape.gradient(loss, w)
grads_clipped = tf.clip_by_value(grads, -0.5, 0.5)
```

**Use case:** When you want hard bounds on gradient magnitudes (e.g., prevent extreme values from a few dimensions). Does **not** preserve direction.

---

## 5. clip_by_global_norm

**tf.clip_by_global_norm(t_list, max_norm)** computes the global L2 norm across all tensors in `t_list`, then scales all tensors by the same factor so the global norm equals `max_norm`.

```python
grads = tape.gradient(loss, [w1, w2, b])
grads_clipped, global_norm = tf.clip_by_global_norm(grads, max_norm=1.0)
optimizer.apply_gradients(zip(grads_clipped, [w1, w2, b]))
```

**Formula:** 
\[
\text{global\_norm} = \sqrt{\sum_i \|g_i\|^2}
\]
If global_norm > max_norm, scale all \(g_i\) by max_norm / global_norm.

**Why global norm?** Ensures balanced updates across all parameters. If one layer has huge gradients and another tiny, per-tensor clipping might still leave the large one dominant. Global norm clipping scales everything proportionally.

---

## 6. GradientTape with tf.function

**@tf.function** traces Python functions to TensorFlow graphs for better performance. GradientTape works inside tf.function; the tape records operations during the traced execution.

```python
@tf.function
def train_step(x, w, optimizer):
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w)
        loss = tf.reduce_mean(y ** 2)
    grads = tape.gradient(loss, w)
    optimizer.apply_gradients(zip([grads], [w]))
    return loss
```

**Tracing behavior:**
- First call: Traces the function (may be slower)
- Subsequent calls: Executes the graph (faster)
- Retracing occurs when input shapes or types change

**Best practices:**
1. Use fixed shapes when possible to avoid retracing
2. Avoid Python side effects (e.g., appending to lists) that depend on values
3. Use `tf.function` for the entire training step, not just gradient computation

**Nested tapes in tf.function:** Fully supported. Second-order derivatives work as in eager mode.

---

## 7. Gradient Accumulation

**Gradient accumulation** simulates a larger batch size by accumulating gradients over several mini-batches before applying an update. Useful when:
- GPU memory limits batch size
- You want effective batch size larger than physical batch size

### Basic Pattern

```python
accumulation_steps = 4
accumulated_grads = [tf.zeros_like(v) for v in model.trainable_variables]

for i, (x, y) in enumerate(dataset):
    with tf.GradientTape() as tape:
        pred = model(x)
        loss = loss_fn(y, pred)
    grads = tape.gradient(loss, model.trainable_variables)
    accumulated_grads = [a + g for a, g in zip(accumulated_grads, grads)]

    if (i + 1) % accumulation_steps == 0:
        avg_grads = [g / accumulation_steps for g in accumulated_grads]
        optimizer.apply_gradients(zip(avg_grads, model.trainable_variables))
        accumulated_grads = [tf.zeros_like(v) for v in model.trainable_variables]
```

**Important:** Divide accumulated gradients by `accumulation_steps` to get the average gradient (equivalent to a larger batch). Alternatively, scale the learning rate.

### Learning Rate Consideration

If you accumulate over \(k\) steps, the effective batch size is \(k \times \text{batch\_size}\). You may need to scale the learning rate (e.g., multiply by \(k\)) when switching from no accumulation to accumulation, depending on your optimization setup.

---

## 8. Integration with Optimizers

### Built-in Clipping

Some optimizers accept `clipnorm` or `clipvalue`:

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
# or
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=0.5)
```

When using `optimizer.apply_gradients()`, gradients are clipped automatically.

### Manual Clipping in Custom Loop

```python
with tf.GradientTape() as tape:
    loss = model(x)
grads = tape.gradient(loss, model.trainable_variables)
grads, _ = tf.clip_by_global_norm(grads, max_norm=1.0)
optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

### Complete Training Step with Clipping

```python
@tf.function
def train_step(x, y, model, optimizer, max_grad_norm=1.0):
    with tf.GradientTape() as tape:
        pred = model(x)
        loss = tf.reduce_mean((y - pred) ** 2)
    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss
```

---

## Summary Table

| Method | Use Case | Preserves Direction? |
|--------|----------|----------------------|
| clip_by_norm | Single tensor, limit magnitude | Yes |
| clip_by_value | Hard bounds per element | No |
| clip_by_global_norm | Multi-parameter models | Yes (proportionally) |

| Concept | Key Point |
|---------|------------|
| tf.function + Tape | Tape records during traced execution; gradients work as in eager |
| Gradient accumulation | Sum gradients over steps, divide by steps, then apply once |
| Optimizer clipping | Use clipnorm/clipvalue in optimizer or clip manually before apply_gradients |
