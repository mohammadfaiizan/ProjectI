# Gradient Tape Fundamentals

## Table of Contents

1. [Introduction to Automatic Differentiation](#1-introduction-to-automatic-differentiation)
2. [GradientTape Basics](#2-gradienttape-basics)
3. [Watching Tensors](#3-watching-tensors)
4. [Computing Gradients](#4-computing-gradients)
5. [Gradients of Nested Operations](#5-gradients-of-nested-operations)
6. [Gradients with Variables](#6-gradients-with-variables)
7. [Multiple Targets and Sources](#7-multiple-targets-and-sources)
8. [Common Pitfalls](#8-common-pitfalls)

---

## 1. Introduction to Automatic Differentiation

**Automatic differentiation** (autodiff) computes derivatives of functions defined by programs. TensorFlow uses **reverse-mode differentiation** (backpropagation) to compute gradients efficiently.

**Key concept:** Unlike symbolic or numerical differentiation, autodiff is exact (up to floating-point) and scales to large computation graphs.

**Use cases:**
- Training neural networks (gradient descent)
- Optimization (gradient-based methods)
- Physics-informed ML
- Sensitivity analysis

---

## 2. GradientTape Basics

**tf.GradientTape** records operations for automatic differentiation. Operations executed inside the tape context are recorded; gradients are computed by calling `tape.gradient()`.

```python
x = tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x * x
dy_dx = tape.gradient(y, x)
print(dy_dx.numpy())
```

**Key concept:** Only operations inside the `with` block are recorded. The tape is single-use by default.

---

## 3. Watching Tensors

**Variables** are automatically watched. **Tensors** (e.g., from `tf.constant`) must be explicitly watched with `tape.watch()`.

```python
x = tf.constant(2.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x ** 2 + tf.sin(x)
grad = tape.gradient(y, x)
```

**Why watch?** Constants are not watched by default to save memory. If you need gradients w.r.t. a constant (e.g., input features), call `tape.watch(x)`.

```python
# Variables are auto-watched
w = tf.Variable(1.0)
with tf.GradientTape() as tape:
    y = w * w
grad = tape.gradient(y, w)
```

---

## 4. Computing Gradients

### tape.gradient(target, source)

Computes the gradient of **target** with respect to **source**. Returns `None` if the gradient cannot be computed (e.g., disconnected graph).

```python
x = tf.constant([1.0, 2.0, 3.0])
with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.reduce_sum(x * x)
grad = tape.gradient(y, x)
print(grad.numpy())
```

**Chain rule:** TensorFlow applies the chain rule automatically. The gradient flows from the target (usually loss) backward to the source.

---

## 5. Gradients of Nested Operations

Complex expressions are differentiated as a whole. All operations in the path from source to target are recorded.

```python
x = tf.constant(2.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    a = x * 2
    b = a + 1
    y = b * b
grad = tape.gradient(y, x)
```

**Key concept:** The gradient `dy/dx` is computed via the chain rule: `dy/dx = dy/db * db/da * da/dx`.

---

## 6. Gradients with Variables

Variables are automatically watched and their gradients are computed. Use `tape.gradient(loss, model.trainable_variables)` for model parameters.

```python
w = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
x = tf.constant([[1.0, 1.0]])
with tf.GradientTape() as tape:
    y = tf.matmul(x, w)
    loss = tf.reduce_mean(y ** 2)
grads = tape.gradient(loss, w)
print(grads.shape)
```

**Optimizer application:**

```python
optimizer = tf.keras.optimizers.SGD(0.01)
grads = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

---

## 7. Multiple Targets and Sources

### Multiple sources

```python
x = tf.constant(1.0)
y = tf.constant(2.0)
with tf.GradientTape() as tape:
    tape.watch([x, y])
    z = x * x + y * y
grads = tape.gradient(z, [x, y])
print(grads[0].numpy(), grads[1].numpy())
```

### Gradient of a list of targets

If the target is a list, the default is to sum the gradients (like a scalar loss).

```python
with tf.GradientTape() as tape:
    loss1 = ...
    loss2 = ...
    total = loss1 + loss2
grad = tape.gradient(total, w)
```

---

## 8. Common Pitfalls

| Pitfall | Cause | Solution |
|---------|-------|----------|
| grad is None | Disconnected graph, wrong dtype | Check tape.watch, ensure float |
| Tape used twice | Default tape is not persistent | Use persistent=True or create new tape |
| In-place ops | Some ops break gradient flow | Avoid in-place updates |
| Control flow | tf.cond/tf.while_recorded | Use tf.cond, tf.while_loop |

**Persistent tape:** For multiple gradient calls (e.g., Hessian), use `persistent=True` and `del tape` when done.

```python
with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    y = x ** 3
dy = tape.gradient(y, x)
d2y = tape.gradient(dy, x)
del tape
```
