# Multivariable Calculus Fundamentals

## Table of Contents

1. [Introduction](#introduction)
2. [Functions of Several Variables](#functions-of-several-variables)
3. [Partial Derivatives](#partial-derivatives)
4. [Gradients](#gradients)
5. [Directional Derivatives](#directional-derivatives)
6. [Jacobian Matrix](#jacobian-matrix)
7. [Hessian Matrix](#hessian-matrix)
8. [Chain Rule](#chain-rule)
9. [Taylor Series Approximation](#taylor-series-approximation)
10. [Machine Learning Applications](#machine-learning-applications)
11. [Key Takeaways](#key-takeaways)

## Introduction

Multivariable calculus extends single-variable calculus to functions of multiple variables, providing the mathematical foundation for optimization in machine learning. Since ML models typically depend on many parameters (weights, biases), understanding how functions change with respect to multiple variables is essential for gradient-based optimization algorithms that train neural networks and other models.

This document covers partial derivatives, gradients, Jacobians, Hessians, and their applications in ML. We develop formal definitions and computational techniques while emphasizing their role in gradient computation and optimization.

## Functions of Several Variables

### Definition

A **function of several variables** $f: \mathbb{R}^n \to \mathbb{R}$ maps $n$-dimensional vectors to scalars:

$$f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n)$$

**Examples**:
- **Loss function**: $L(\boldsymbol{\theta}) = \frac{1}{2}\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2$ where $\boldsymbol{\theta} \in \mathbb{R}^d$
- **Neural network output**: $f(\mathbf{x}; \mathbf{W}, \mathbf{b})$ where $\mathbf{W}, \mathbf{b}$ are parameters
- **Distance**: $f(x, y, z) = \sqrt{x^2 + y^2 + z^2}$

### Level Sets and Contours

For function $f: \mathbb{R}^2 \to \mathbb{R}$, the **level set** (contour) at value $c$ is:

$$\{(x, y) : f(x, y) = c\}$$

In higher dimensions, these are **level surfaces**. Contours help visualize function behavior.

### Limits and Continuity

Function $f: \mathbb{R}^n \to \mathbb{R}$ is **continuous** at $\mathbf{a}$ if:

$$\lim_{\mathbf{x} \to \mathbf{a}} f(\mathbf{x}) = f(\mathbf{a})$$

This means $f(\mathbf{x})$ approaches $f(\mathbf{a})$ as $\mathbf{x}$ approaches $\mathbf{a}$ along any path.

## Partial Derivatives

### Definition

The **partial derivative** of $f(x_1, \ldots, x_n)$ with respect to $x_i$ at point $\mathbf{a}$ is:

$$\frac{\partial f}{\partial x_i}(\mathbf{a}) = \lim_{h \to 0} \frac{f(a_1, \ldots, a_i + h, \ldots, a_n) - f(a_1, \ldots, a_n)}{h}$$

This measures the rate of change of $f$ with respect to $x_i$ while holding other variables constant.

### Notation

Common notations:
- $\frac{\partial f}{\partial x_i}$
- $\partial_{x_i} f$
- $f_{x_i}$
- $D_i f$

### Computing Partial Derivatives

To compute $\frac{\partial f}{\partial x_i}$, treat all other variables as constants and differentiate with respect to $x_i$ using single-variable calculus rules.

**Example**: For $f(x, y) = x^2y + \sin(xy)$:
- $\frac{\partial f}{\partial x} = 2xy + y\cos(xy)$
- $\frac{\partial f}{\partial y} = x^2 + x\cos(xy)$

### Higher-Order Partial Derivatives

**Second-order partial derivatives**:
$$\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial}{\partial x_i}\left(\frac{\partial f}{\partial x_j}\right)$$

**Clairaut's Theorem**: If $f$ has continuous second partial derivatives, then:

$$\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$$

The order of differentiation doesn't matter for smooth functions.

## Gradients

### Definition

The **gradient** of $f: \mathbb{R}^n \to \mathbb{R}$ is the vector of partial derivatives:

$$\nabla f(\mathbf{x}) = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix}$$

**Notation**: $\nabla f$, $\text{grad } f$, or $\frac{\partial f}{\partial \mathbf{x}}$ (though latter is sometimes used for Jacobian).

### Properties of Gradient

**Linearity**:
$$\nabla(\alpha f + \beta g) = \alpha\nabla f + \beta\nabla g$$

**Product Rule**:
$$\nabla(fg) = f\nabla g + g\nabla f$$

**Chain Rule** (scalar function):
$$\nabla(f \circ g)(\mathbf{x}) = f'(g(\mathbf{x}))\nabla g(\mathbf{x})$$

### Geometric Interpretation

The gradient $\nabla f(\mathbf{a})$ points in the direction of **steepest ascent** of $f$ at point $\mathbf{a}$.

**Magnitude**: $\|\nabla f(\mathbf{a})\|$ gives the rate of increase in that direction.

**Orthogonality**: $\nabla f(\mathbf{a})$ is orthogonal to the level set $\{f = f(\mathbf{a})\}$ at point $\mathbf{a}$.

### Gradient Descent Direction

For minimization, move in direction $-\nabla f(\mathbf{a})$ (steepest descent):

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha\nabla f(\mathbf{x}_k)$$

where $\alpha > 0$ is the learning rate.

## Directional Derivatives

### Definition

The **directional derivative** of $f$ at $\mathbf{a}$ in direction $\mathbf{u}$ (unit vector) is:

$$D_{\mathbf{u}}f(\mathbf{a}) = \lim_{h \to 0} \frac{f(\mathbf{a} + h\mathbf{u}) - f(\mathbf{a})}{h}$$

This measures the rate of change of $f$ in direction $\mathbf{u}$.

### Relationship to Gradient

**Theorem**: For differentiable $f$ and unit vector $\mathbf{u}$:

$$D_{\mathbf{u}}f(\mathbf{a}) = \nabla f(\mathbf{a}) \cdot \mathbf{u}$$

**Proof**: Using chain rule on $g(h) = f(\mathbf{a} + h\mathbf{u})$:
$$g'(0) = \nabla f(\mathbf{a}) \cdot \mathbf{u}$$

**Corollary**: 
- Maximum directional derivative occurs when $\mathbf{u} = \frac{\nabla f(\mathbf{a})}{\|\nabla f(\mathbf{a})\|}$ (direction of gradient)
- Maximum rate of change is $\|\nabla f(\mathbf{a})\|$
- Minimum (most negative) occurs in opposite direction: $-\frac{\nabla f(\mathbf{a})}{\|\nabla f(\mathbf{a})\|}$

## Jacobian Matrix

### Definition

For vector-valued function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$:

$$\mathbf{f}(\mathbf{x}) = \begin{pmatrix} f_1(\mathbf{x}) \\ f_2(\mathbf{x}) \\ \vdots \\ f_m(\mathbf{x}) \end{pmatrix}$$

The **Jacobian matrix** is:

$$\mathbf{J}_{\mathbf{f}}(\mathbf{x}) = \begin{pmatrix} 
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{pmatrix} \in \mathbb{R}^{m \times n}$$

**Notation**: $\mathbf{J}_{\mathbf{f}}$, $\frac{\partial \mathbf{f}}{\partial \mathbf{x}}$, or $D\mathbf{f}$.

### Linear Approximation

For small $\Delta\mathbf{x}$:

$$\mathbf{f}(\mathbf{a} + \Delta\mathbf{x}) \approx \mathbf{f}(\mathbf{a}) + \mathbf{J}_{\mathbf{f}}(\mathbf{a})\Delta\mathbf{x}$$

The Jacobian is the matrix representation of the linear transformation that best approximates $\mathbf{f}$ near $\mathbf{a}$.

### Special Cases

**Scalar function** ($m = 1$): Jacobian is row vector (gradient transpose):
$$\mathbf{J}_f = (\nabla f)^T$$

**Identity function**: $\mathbf{f}(\mathbf{x}) = \mathbf{x}$, then $\mathbf{J}_{\mathbf{f}} = \mathbf{I}$

**Linear function**: $\mathbf{f}(\mathbf{x}) = \mathbf{A}\mathbf{x} + \mathbf{b}$, then $\mathbf{J}_{\mathbf{f}} = \mathbf{A}$

## Hessian Matrix

### Definition

For scalar function $f: \mathbb{R}^n \to \mathbb{R}$, the **Hessian matrix** is:

$$\mathbf{H}_f(\mathbf{x}) = \begin{pmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{pmatrix}$$

**Properties**:
- **Symmetric**: $\mathbf{H}_f = \mathbf{H}_f^T$ (by Clairaut's theorem)
- Contains all second-order partial derivatives
- **Notation**: $\mathbf{H}_f$, $\nabla^2 f$, or $\frac{\partial^2 f}{\partial \mathbf{x}^2}$

### Second-Order Directional Derivative

The second derivative in direction $\mathbf{u}$ (unit vector):

$$D_{\mathbf{u}}^2 f(\mathbf{a}) = \mathbf{u}^T\mathbf{H}_f(\mathbf{a})\mathbf{u}$$

This measures the curvature of $f$ in direction $\mathbf{u}$.

### Eigenvalues of Hessian

Eigenvalues of $\mathbf{H}_f(\mathbf{a})$ indicate:
- **Positive eigenvalues**: Function curves upward in corresponding eigenvector directions
- **Negative eigenvalues**: Function curves downward
- **Zero eigenvalues**: Flat direction (no curvature)

### Positive Definiteness and Convexity

**Theorem**: If $\mathbf{H}_f(\mathbf{x}) \succeq 0$ (positive semidefinite) for all $\mathbf{x}$, then $f$ is **convex**.

**Local minimum**: At critical point $\nabla f(\mathbf{a}) = \mathbf{0}$:
- If $\mathbf{H}_f(\mathbf{a}) \succ 0$ (positive definite), then $\mathbf{a}$ is a local minimum
- If $\mathbf{H}_f(\mathbf{a}) \prec 0$ (negative definite), then $\mathbf{a}$ is a local maximum
- If $\mathbf{H}_f(\mathbf{a})$ has both positive and negative eigenvalues, then $\mathbf{a}$ is a saddle point

## Chain Rule

### Scalar Function Composition

For $f: \mathbb{R}^m \to \mathbb{R}$ and $\mathbf{g}: \mathbb{R}^n \to \mathbb{R}^m$, the composition $h(\mathbf{x}) = f(\mathbf{g}(\mathbf{x}))$ has gradient:

$$\nabla h(\mathbf{x}) = \mathbf{J}_{\mathbf{g}}(\mathbf{x})^T \nabla f(\mathbf{g}(\mathbf{x}))$$

**Component form**:
$$\frac{\partial h}{\partial x_j} = \sum_{i=1}^m \frac{\partial f}{\partial g_i} \frac{\partial g_i}{\partial x_j}$$

### Vector Function Composition

For $\mathbf{f}: \mathbb{R}^m \to \mathbb{R}^p$ and $\mathbf{g}: \mathbb{R}^n \to \mathbb{R}^m$, the composition $\mathbf{h}(\mathbf{x}) = \mathbf{f}(\mathbf{g}(\mathbf{x}))$ has Jacobian:

$$\mathbf{J}_{\mathbf{h}}(\mathbf{x}) = \mathbf{J}_{\mathbf{f}}(\mathbf{g}(\mathbf{x})) \mathbf{J}_{\mathbf{g}}(\mathbf{x})$$

This is matrix multiplication of Jacobians.

### Backpropagation Connection

The chain rule is fundamental to **backpropagation** in neural networks. For loss $L$ depending on parameters $\boldsymbol{\theta}$ through layers:

$$\frac{\partial L}{\partial \theta_i} = \sum_{\text{paths}} \prod_{\text{layers}} \frac{\partial \text{layer}_j}{\partial \text{layer}_{j-1}} \cdot \frac{\partial L}{\partial \text{output}}$$

## Taylor Series Approximation

### First-Order Approximation

For differentiable $f: \mathbb{R}^n \to \mathbb{R}$:

$$f(\mathbf{a} + \Delta\mathbf{x}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a})^T\Delta\mathbf{x}$$

This is the **linear approximation** (tangent plane in 2D).

### Second-Order Approximation

For twice-differentiable $f$:

$$f(\mathbf{a} + \Delta\mathbf{x}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a})^T\Delta\mathbf{x} + \frac{1}{2}\Delta\mathbf{x}^T\mathbf{H}_f(\mathbf{a})\Delta\mathbf{x}$$

This **quadratic approximation** captures curvature information.

### Multivariate Taylor Series

Full Taylor expansion:

$$f(\mathbf{a} + \Delta\mathbf{x}) = \sum_{k=0}^{\infty} \frac{1}{k!} \sum_{i_1,\ldots,i_k} \frac{\partial^k f}{\partial x_{i_1} \cdots \partial x_{i_k}}(\mathbf{a}) \Delta x_{i_1} \cdots \Delta x_{i_k}$$

**Quadratic form**: The second-order term $\frac{1}{2}\Delta\mathbf{x}^T\mathbf{H}_f(\mathbf{a})\Delta\mathbf{x}$ is a quadratic form in $\Delta\mathbf{x}$.

## Machine Learning Applications

### Gradient Computation in Neural Networks

**Forward pass**: Compute $L(\mathbf{y}, \hat{\mathbf{y}}(\mathbf{x}; \boldsymbol{\theta}))$

**Backward pass**: Compute gradients $\frac{\partial L}{\partial \theta_i}$ using chain rule:

$$\frac{\partial L}{\partial \theta_i} = \frac{\partial L}{\partial \hat{\mathbf{y}}} \frac{\partial \hat{\mathbf{y}}}{\partial \theta_i}$$

Each layer contributes a Jacobian matrix to the chain.

### Loss Function Gradients

**Mean Squared Error**:
$$L = \frac{1}{2}\|\mathbf{y} - \hat{\mathbf{y}}\|^2$$

Gradient:
$$\nabla_{\hat{\mathbf{y}}} L = \hat{\mathbf{y}} - \mathbf{y}$$

**Cross-Entropy Loss**:
$$L = -\sum_i y_i \log(\hat{y}_i)$$

Gradient:
$$\frac{\partial L}{\partial \hat{y}_i} = -\frac{y_i}{\hat{y}_i}$$

### Optimization Algorithms

**Gradient Descent**:
$$\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \alpha\nabla L(\boldsymbol{\theta}_k)$$

Uses first-order information (gradient).

**Newton's Method**:
$$\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \mathbf{H}_L^{-1}(\boldsymbol{\theta}_k)\nabla L(\boldsymbol{\theta}_k)$$

Uses second-order information (Hessian), converges faster but computationally expensive.

### Regularization Gradients

**L2 Regularization**:
$$L_{\text{reg}} = L + \lambda\|\boldsymbol{\theta}\|^2$$

Gradient:
$$\nabla L_{\text{reg}} = \nabla L + 2\lambda\boldsymbol{\theta}$$

**L1 Regularization**:
$$L_{\text{reg}} = L + \lambda\|\boldsymbol{\theta}\|_1$$

Subgradient (not differentiable at zero):
$$\frac{\partial L_{\text{reg}}}{\partial \theta_i} = \frac{\partial L}{\partial \theta_i} + \lambda \text{sign}(\theta_i)$$

### Automatic Differentiation

Modern frameworks (PyTorch, TensorFlow) use **automatic differentiation**:
- **Forward mode**: Computes Jacobian-vector products
- **Reverse mode**: Computes vector-Jacobian products (backpropagation)

Both use chain rule systematically.

### Gradient Checking

Numerical gradient approximation for verification:

$$\frac{\partial f}{\partial x_i}(\mathbf{a}) \approx \frac{f(a_1, \ldots, a_i + \epsilon, \ldots) - f(a_1, \ldots, a_i - \epsilon, \ldots)}{2\epsilon}$$

Compare with analytical gradient to catch implementation errors.

### Hessian-Vector Products

For large models, computing full Hessian is expensive. Instead, compute **Hessian-vector products** $\mathbf{H}\mathbf{v}$ efficiently using automatic differentiation, enabling:
- Second-order optimization methods
- Curvature information for adaptive learning rates
- Pruning and compression techniques

### Natural Gradients

**Natural gradient** uses Fisher information matrix $\mathbf{F}$ instead of identity:

$$\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \alpha\mathbf{F}^{-1}\nabla L$$

This accounts for geometry of parameter space, important in policy gradient methods.

### Implicit Differentiation

For problems where $y$ is defined implicitly by $f(\mathbf{x}, y) = 0$:

$$\frac{dy}{dx} = -\frac{\partial f/\partial x}{\partial f/\partial y}$$

Useful in:
- Equilibrium models
- Optimality conditions
- Adversarial training

## Key Takeaways

1. **Partial derivatives** measure rate of change with respect to individual variables, fundamental for understanding multi-parameter functions.

2. **Gradient** is the vector of partial derivatives, pointing in direction of steepest ascent and essential for gradient-based optimization.

3. **Directional derivatives** generalize partial derivatives to arbitrary directions, related to gradient via dot product.

4. **Jacobian matrix** generalizes gradient to vector-valued functions, representing the linear approximation of the function.

5. **Hessian matrix** contains second-order information, crucial for understanding curvature, convexity, and second-order optimization methods.

6. **Chain rule** enables computing derivatives of compositions, fundamental to backpropagation in neural networks.

7. **Taylor approximation** provides local linear or quadratic approximations, useful for optimization and understanding function behavior.

8. **Gradient computation** via backpropagation is the computational engine of deep learning, efficiently computing gradients through automatic differentiation.

9. **Optimization algorithms** use gradients (first-order) and sometimes Hessians (second-order) to find minima of loss functions.

10. **Multivariable calculus** provides the mathematical language for understanding and optimizing functions of many variables, which is the essence of machine learning.
