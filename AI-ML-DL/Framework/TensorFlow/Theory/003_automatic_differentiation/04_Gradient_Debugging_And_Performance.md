# Gradient Debugging and Performance

## Table of Contents

1. [Debugging NaN and Inf Gradients](#1-debugging-nan-and-inf-gradients)
2. [tf.debugging.check_numerics](#2-tfdebuggingcheck_numerics)
3. [Gradient Flow Analysis](#3-gradient-flow-analysis)
4. [Performance Optimization](#4-performance-optimization)
5. [Advanced Gradient Patterns](#5-advanced-gradient-patterns)
6. [Gradient Penalty](#6-gradient-penalty)
7. [Gradient Reversal](#7-gradient-reversal)
8. [Best Practices Summary](#8-best-practices-summary)

---

## 1. Debugging NaN and Inf Gradients

**NaN (Not a Number)** and **Inf** in gradients cause training to diverge. Common causes:

| Cause | Example | Fix |
|-------|---------|-----|
| Division by zero | log(0), 1/0 | Add epsilon, use safe_log |
| Overflow | exp(large) | Clamp inputs, use stable softmax |
| Unstable operations | sqrt(negative) | Clamp before sqrt |
| Learning rate too high | Large updates | Reduce LR, use gradient clipping |

**Debugging strategy:**
1. Enable `tf.debugging.check_numerics` to catch NaNs early
2. Add assertions or prints at key points
3. Use gradient clipping as a safeguard
4. Check for None gradients (unwatched or unconnected sources)

---

## 2. tf.debugging.check_numerics

**tf.debugging.check_numerics(tensor, message)** raises `InvalidArgumentError` if the tensor contains NaN or Inf. Use it to pinpoint where numerical issues arise.

```python
with tf.GradientTape() as tape:
    tape.watch(x)
    y = some_operation(x)

grads = tape.gradient(y, x)
grads_checked = tf.debugging.check_numerics(grads, "gradient check")
```

**Enable globally** (checks all operations):

```python
tf.debugging.enable_check_numerics()
# ... run training ...
tf.debugging.disable_check_numerics()
```

**Warning:** Global check_numerics can significantly slow execution. Use for debugging only.

**Selective checking:**

```python
def safe_gradient(loss, sources):
    with tf.GradientTape() as tape:
        tape.watch(sources)
        loss_val = loss
    grads = tape.gradient(loss_val, sources)
    for i, g in enumerate(grads):
        if g is not None:
            grads[i] = tf.debugging.check_numerics(g, f"grad_{i}")
    return grads
```

---

## 3. Gradient Flow Analysis

**Gradient flow** refers to how gradients propagate backward through the network. Poor flow leads to:
- **Vanishing gradients:** Gradients become near zero in early layers
- **Exploding gradients:** Gradients grow unbounded

### Per-Layer Gradient Norms

```python
with tf.GradientTape() as tape:
    loss = model(x)

grads = tape.gradient(loss, model.trainable_variables)
for i, (g, v) in enumerate(zip(grads, model.trainable_variables)):
    if g is not None:
        norm = tf.norm(g).numpy()
        print(f"Layer {i} ({v.name}): grad norm = {norm:.6f}")
```

### Gradient-to-Parameter Ratio

A useful metric: \(\frac{\|\nabla w\|}{\|w\|}\). If this ratio is very small, the layer receives little learning signal; if very large, updates may be unstable.

```python
for g, w in zip(grads, model.trainable_variables):
    if g is not None:
        ratio = tf.norm(g) / (tf.norm(w) + 1e-8)
        print(f"grad/param ratio: {ratio.numpy():.6f}")
```

### Gradient Statistics

```python
for g in grads:
    if g is not None:
        mean_abs = tf.reduce_mean(tf.abs(g)).numpy()
        max_abs = tf.reduce_max(tf.abs(g)).numpy()
        print(f"mean |grad| = {mean_abs:.6f}, max |grad| = {max_abs:.6f}")
```

---

## 4. Performance Optimization

### Use tf.function

Wrapping gradient computation in `@tf.function` compiles to a graph, reducing Python overhead:

```python
@tf.function
def compute_gradients(x, w):
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w)
        loss = tf.reduce_mean(y ** 2)
    return tape.gradient(loss, w)
```

### Batch Size

Larger batches typically yield better GPU utilization. Balance batch size with memory.

### Avoid Eager-Only Patterns

- Prefer `tf.constant` over Python literals when possible
- Avoid Python control flow that depends on tensor values (use `tf.cond`, `tf.while_loop`)
- Minimize Python callbacks inside the traced region

### Persistent Tape Overhead

Use `persistent=True` only when necessary. Non-persistent tapes free memory immediately after `gradient()`.

---

## 5. Advanced Gradient Patterns

### Gradient Penalty (WGAN-GP)

Used in Wasserstein GANs to enforce a Lipschitz constraint on the critic. Penalize the gradient norm of the critic with respect to interpolated samples:

```python
def gradient_penalty(discriminator, real, fake):
    batch_size = tf.shape(real)[0]
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolates = alpha * real + (1 - alpha) * fake
    with tf.GradientTape() as tape:
        tape.watch(interpolates)
        pred = discriminator(interpolates)
    grads = tape.gradient(pred, interpolates)
    grad_norms = tf.sqrt(tf.reduce_sum(grads ** 2, axis=[1, 2, 3]) + 1e-8)
    penalty = tf.reduce_mean((grad_norms - 1.0) ** 2)
    return penalty
```

### Gradient Reversal (Domain Adaptation)

Reverse gradients in a domain classifier so the feature extractor is encouraged to learn domain-invariant features:

```python
@tf.custom_gradient
def gradient_reversal(x, scale=1.0):
    def grad(dy):
        return -scale * dy
    return x, grad
```

Forward: identity. Backward: negate and scale. Used in DANN (Domain-Adversarial Neural Networks).

---

## 6. Gradient Penalty

**Gradient penalty** regularizes the gradient magnitude. Common in:
- **WGAN-GP:** Penalize \(\|\nabla_{\hat{x}} D(\hat{x})\|\) to be close to 1
- **Spectral normalization:** Alternative way to control Lipschitz constant

The penalty term is added to the loss; gradients flow through the discriminator and the interpolated input.

---

## 7. Gradient Reversal

**Gradient reversal layer (GRL):** Forward pass is identity; backward pass multiplies gradients by \(-\lambda\). Used so that:
- Feature extractor receives gradient that *maximizes* domain classifier loss (fools the classifier)
- Domain classifier receives normal gradient (minimizes its loss)

Implementation via `@tf.custom_gradient` as shown above. The scale \(\lambda\) can be scheduled (e.g., increase over training).

---

## 8. Best Practices Summary

| Area | Practice |
|------|----------|
| Debugging | Use check_numerics during development; disable in production |
| Flow analysis | Log gradient norms per layer; watch for vanishing/exploding |
| Performance | Use tf.function for training steps; prefer larger batches |
| Stability | Add epsilon to log, sqrt; use gradient clipping |
| Advanced | Gradient penalty for GANs; gradient reversal for domain adaptation |

### Checklist for Gradient Issues

1. Are all sources watched (or Variables)?
2. Are gradients None? Check unconnected sources.
3. Are gradients NaN/Inf? Add check_numerics, inspect operations.
4. Are gradients too large? Use clipping.
5. Are gradients too small? Check architecture (skip connections, normalization).

---

## Summary Table

| Topic | Key API / Pattern |
|-------|-------------------|
| NaN detection | tf.debugging.check_numerics |
| Flow analysis | tf.norm(grad), grad/param ratio |
| Performance | @tf.function, larger batches |
| Gradient penalty | tape.gradient(pred, interpolates), norm penalty |
| Gradient reversal | @tf.custom_gradient with negated dy |
