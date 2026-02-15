# Custom Gradients and Higher-Order Derivatives

## Table of Contents

1. [Higher-Order Gradients](#1-higher-order-gradients)
2. [Nested GradientTape](#2-nested-gradienttape)
3. [Custom Gradients](#3-custom-gradients)
4. [Stop Gradient](#4-stop-gradient)
5. [Jacobian and Hessian](#5-jacobian-and-hessian)
6. [Practical Applications](#6-practical-applications)

---

## 1. Higher-Order Gradients

**First-order gradients** are derivatives of a scalar loss with respect to parameters: \(\frac{\partial L}{\partial w}\). **Higher-order gradients** include second derivatives (Hessian), third derivatives, and beyond.

**Use cases for higher-order derivatives:**
- **Optimization:** Newton's method, natural gradient
- **Uncertainty:** Laplace approximation, curvature analysis
- **Physics-informed ML:** PDEs with second-order terms
- **Meta-learning:** Gradients of gradients (MAML)

TensorFlow computes higher-order derivatives by **nesting GradientTape** contexts or using **persistent tapes**.

---

## 2. Nested GradientTape

To compute \(\frac{d^2 y}{dx^2}\), use an outer tape that differentiates the result of the inner tape:

```python
x = tf.constant(2.0)
with tf.GradientTape() as tape2:
    tape2.watch(x)
    with tf.GradientTape() as tape1:
        tape1.watch(x)
        y = x ** 3
    dy_dx = tape1.gradient(y, x)
d2y_dx2 = tape2.gradient(dy_dx, x)
# dy_dx = 12.0, d2y_dx2 = 12.0 (since d/dx(3x^2) = 6x, 6*2=12)
```

**Order of nesting:** The innermost tape computes the first derivative; each outer tape adds one more order of differentiation.

**Alternative with persistent tape:**

```python
with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    y = x ** 3
dy_dx = tape.gradient(y, x)
d2y_dx2 = tape.gradient(dy_dx, x)
del tape
```

---

## 3. Custom Gradients

The **@tf.custom_gradient** decorator lets you define a custom forward pass and a custom backward pass (gradient). Use this when:

- The default gradient is numerically unstable
- You want to implement a non-standard or approximate gradient
- You need to clip or modify gradients in the backward pass

### Basic Structure

```python
@tf.custom_gradient
def custom_op(x):
    # Forward pass
    y = ...  # compute output
    def grad(dy):
        # dy is the upstream gradient (d(loss)/dy)
        # Return d(loss)/dx
        return dy * ...  # chain rule
    return y, grad
```

### Example: Custom Square

```python
@tf.custom_gradient
def custom_square(x):
    def grad(dy):
        return 2.0 * x * dy
    return x * x, grad
```

Here the gradient is explicitly \(2x \cdot dy\), which matches the analytic derivative. The decorator is useful when you want to modify this (e.g., gradient clipping, scaling).

### Example: Numerically Stable Log

```python
@tf.custom_gradient
def safe_log(x):
    def grad(dy):
        return dy / tf.maximum(x, 1e-7)
    return tf.math.log(tf.maximum(x, 1e-7)), grad
```

This avoids division by zero and log of non-positive values.

### Multiple Inputs

```python
@tf.custom_gradient
def custom_op(x, y):
    def grad(dz):
        return dz * ..., dz * ...  # gradients for x and y
    return forward_result, grad
```

---

## 4. Stop Gradient

**tf.stop_gradient** blocks gradients from flowing through a tensor. The tensor is used in the forward pass but treated as a constant in the backward pass.

```python
x = tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x ** 2
    z = tf.stop_gradient(y) + x

grad = tape.gradient(z, x)  # Only x contributes: dz/dx = 1
```

**Use cases:**
- **Auxiliary losses:** Prevent an auxiliary loss from affecting shared layers
- **Target networks:** Freeze target values in Q-learning (e.g., DQN)
- **Contrastive learning:** Stop gradient on one branch of siamese networks
- **Gradient surgery:** Implement gradient reversal (combine with custom gradient)

### Gradient Reversal Pattern

```python
@tf.custom_gradient
def gradient_reversal(x, scale=1.0):
    def grad(dy):
        return -scale * dy
    return x, grad
```

Forward: identity. Backward: negate and scale the gradient. Used in domain adaptation (DANN).

---

## 5. Jacobian and Hessian

### Jacobian

The **Jacobian** is the matrix of first partial derivatives: \(J_{ij} = \frac{\partial y_i}{\partial x_j}\). For a vector-valued function \(\mathbf{y} = f(\mathbf{x})\):

```python
x = tf.constant([1.0, 2.0])
with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.stack([x[0]**2, x[0]*x[1], x[1]**2])

jacobian = tape.jacobian(y, x)
# Shape: (3, 2) - 3 outputs, 2 inputs
```

**tape.jacobian(target, source)** returns the full Jacobian matrix. Can be memory-intensive for large outputs.

### Batch Jacobian

For batched inputs, **tape.batch_jacobian** computes a Jacobian per batch element, which is more memory-efficient:

```python
x = tf.constant([[1.0, 2.0], [3.0, 4.0]])  # batch of 2
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x ** 2

batch_jac = tape.batch_jacobian(y, x)
# Shape: (2, 2, 2) - batch, outputs, inputs
```

### Hessian

The **Hessian** is the matrix of second partial derivatives. For a scalar \(y = f(\mathbf{x})\):

```python
x = tf.constant([1.0, 2.0])
with tf.GradientTape() as tape2:
    tape2.watch(x)
    with tf.GradientTape() as tape1:
        tape1.watch(x)
        y = tf.reduce_sum(x ** 3)
    grad = tape1.gradient(y, x)
hessian_diag = tape2.gradient(grad, x)
# For y = x1^3 + x2^3: Hessian is diagonal [6*x1, 6*x2]
```

For a full Hessian matrix, compute the gradient of each component of the gradient with respect to each input (nested loops or jacobian of gradient).

---

## 6. Practical Applications

| Technique | Application |
|-----------|-------------|
| Nested tape | Newton's method, curvature regularization |
| Custom gradient | Stable softmax, safe log, gradient clipping in backward |
| Stop gradient | Target networks, auxiliary heads, contrastive learning |
| Jacobian | Sensitivity analysis, invertible networks, normalizing flows |
| Hessian | Uncertainty estimation, second-order optimization |

### Gradient Penalty (WGAN-GP)

Uses gradients of the discriminator with respect to interpolated samples to enforce Lipschitz constraint:

```python
def gradient_penalty(discriminator, real, fake):
    alpha = tf.random.uniform([batch_size, 1, 1, 1])
    interpolates = alpha * real + (1 - alpha) * fake
    with tf.GradientTape() as tape:
        tape.watch(interpolates)
        pred = discriminator(interpolates)
    grads = tape.gradient(pred, interpolates)
    grad_norms = tf.norm(grads, axis=[1, 2, 3])
    penalty = tf.reduce_mean((grad_norms - 1.0) ** 2)
    return penalty
```

---

## Summary Table

| Concept | API | Purpose |
|---------|-----|---------|
| Second derivative | Nested GradientTape | d2y/dx2 |
| Custom backward | @tf.custom_gradient | Override or stabilize gradients |
| Block gradient | tf.stop_gradient | Exclude from backward pass |
| Full Jacobian | tape.jacobian() | Matrix of partial derivatives |
| Batched Jacobian | tape.batch_jacobian() | Per-sample Jacobians |
| Hessian | gradient of gradient | Second-order derivatives |
