# Gradient Analysis and Debugging

## Table of Contents

1. [Gradient Analysis](#1-gradient-analysis)
2. [Adversarial Examples](#2-adversarial-examples)
3. [TensorFlow Debugging Tools](#3-tensorflow-debugging-tools)

---

## 1. Gradient Analysis

**Gradient flow** analysis helps diagnose **vanishing** and **exploding** gradients in deep networks.

### Gradient Extraction

Use **tf.GradientTape** to compute gradients w.r.t. trainable variables.

```python
with tf.GradientTape() as tape:
    logits = model(x, training=True)
    loss = tf.reduce_mean(loss_fn(y, logits))
grads = tape.gradient(loss, model.trainable_variables)
```

### Per-Layer Gradient Norms

Compute L2 norm of each layer's gradient to identify problematic layers.

```python
for i, (g, v) in enumerate(zip(grads, model.trainable_variables)):
    if g is not None:
        norm = tf.norm(g).numpy()
        print(f"Layer {i} ({v.name}): norm={norm:.6e}")
```

### Vanishing Gradients

Gradients near zero prevent learning. Typical causes: deep networks, saturating activations (sigmoid, tanh), small initialization.

| Symptom | Norm Range | Fix |
|---------|------------|-----|
| Vanishing | < 1e-7 | ReLU, skip connections, batch norm |
| Exploding | > 1e3 | Gradient clipping, lower LR |

### Exploding Gradients

Very large gradients cause instability. Use **gradient clipping**:

```python
grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

### Total Gradient Norm

Monitor overall gradient magnitude during training.

```python
total_norm = tf.sqrt(sum(tf.reduce_sum(tf.square(g)) for g in grads if g is not None))
```

---

## 2. Adversarial Examples

**Adversarial examples** are inputs crafted to fool a model. **FGSM** (Fast Gradient Sign Method) is a simple attack for robustness evaluation.

### FGSM Algorithm

1. Compute gradient of loss w.r.t. input.
2. Add small perturbation in direction of gradient sign.
3. Result: input that maximizes loss (misclassification).

```python
def fgsm_attack(model, x, y, epsilon=0.1):
    x = tf.convert_to_tensor(x)
    with tf.GradientTape() as tape:
        tape.watch(x)
        pred = model(x)
        loss = tf.reduce_mean(loss_fn(y, pred))
    grad = tape.gradient(loss, x)
    signed_grad = tf.sign(grad)
    x_adv = x + epsilon * signed_grad
    return x_adv
```

### Epsilon Parameter

**epsilon** controls perturbation magnitude. Larger epsilon = stronger attack, more visible distortion.

| Epsilon | Typical Effect |
|---------|----------------|
| 0.01 | Subtle, often still correct |
| 0.1 | Noticeable, many misclassified |
| 0.3+ | Strong distortion |

### Robustness Evaluation

Compare accuracy on clean vs adversarial inputs.

```python
orig_acc = evaluate(model, x_clean, y)
x_adv = fgsm_attack(model, x_clean, y, epsilon=0.1)
adv_acc = evaluate(model, x_adv, y)
print(f"Clean: {orig_acc:.2%}, Adversarial: {adv_acc:.2%}")
```

### Defenses

- **Adversarial training**: Include adversarial examples in training.
- **Input preprocessing**: Denoising, JPEG compression.
- **Gradient masking**: Not recommended (false sense of security).

---

## 3. TensorFlow Debugging Tools

**tf.debugging** provides assertions and numerical checks for development and testing.

### assert_equal

Verifies two tensors are element-wise equal.

```python
tf.debugging.assert_equal(a, b)
# Raises InvalidArgumentError if any element differs
```

### assert_positive

Asserts all elements are positive.

```python
tf.debugging.assert_positive(x)
```

### assert_* Family

| Function | Condition |
|----------|-----------|
| assert_equal | a == b |
| assert_none_equal | a != b |
| assert_positive | x > 0 |
| assert_non_negative | x >= 0 |
| assert_less | a < b |
| assert_rank | rank(x) == n |

### check_numerics

Raises **InvalidArgumentError** if tensor contains NaN or Inf. Use to catch numerical instability.

```python
t = tf.constant([1.0, 2.0, np.nan])
tf.debugging.check_numerics(t, "Tensor contains NaN/Inf")  # Raises
```

On valid tensors, returns the tensor unchanged.

```python
t = tf.constant([1.0, 2.0, 3.0])
checked = tf.debugging.check_numerics(t, "Valid tensor")
```

### enable_check_numerics

**enable_check_numerics** enables NaN/Inf checks for all ops within the scope. Useful for tracing the source of numerical errors.

```python
with tf.debugging.experimental.enable_check_numerics():
    z = some_operation(x)  # Raises if any op produces NaN/Inf
```

### Debugging Workflow

1. Use **check_numerics** on loss and key tensors.
2. Use **assert_*** for shape and value invariants.
3. Use **enable_check_numerics** to locate first NaN/Inf.
4. Check gradients with **GradientTape** and norm analysis.

---

## Summary

| Topic | Key APIs | Use Case |
|-------|----------|----------|
| Gradient analysis | GradientTape, tf.norm | Vanishing/exploding detection |
| Adversarial | FGSM, tape.gradient | Robustness evaluation |
| Debugging | assert_*, check_numerics | Numerical stability |
