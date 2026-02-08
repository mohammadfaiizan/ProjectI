# Optimization Theory: Unconstrained Optimization

## Table of Contents

1. [Introduction](#introduction)
2. [Problem Formulation](#problem-formulation)
3. [Gradient Descent](#gradient-descent)
4. [Newton's Method](#newtons-method)
5. [Convergence Analysis](#convergence-analysis)
6. [Convexity](#convexity)
7. [First and Second Order Conditions](#first-and-second-order-conditions)
8. [Learning Rate Analysis](#learning-rate-analysis)
9. [Machine Learning Applications](#machine-learning-applications)
10. [Key Takeaways](#key-takeaways)

## Introduction

Unconstrained optimization is fundamental to machine learning, as most learning algorithms reduce to minimizing a loss function with respect to model parameters. Understanding optimization theory provides the mathematical foundation for designing effective training algorithms, analyzing convergence properties, and diagnosing optimization problems.

An unconstrained optimization problem seeks to find:

$$\mathbf{x}^* = \arg\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x})$$

where $f: \mathbb{R}^n \to \mathbb{R}$ is the objective function. In machine learning, $f$ typically represents a loss function, and $\mathbf{x}$ represents model parameters (weights, biases, etc.).

## Problem Formulation

### General Form

The unconstrained minimization problem is:

$$\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x})$$

where we seek a point $\mathbf{x}^*$ such that $f(\mathbf{x}^*) \leq f(\mathbf{x})$ for all $\mathbf{x} \in \mathbb{R}^n$.

### Local vs Global Optima

- **Global minimum**: $f(\mathbf{x}^*) \leq f(\mathbf{x})$ for all $\mathbf{x}$
- **Local minimum**: There exists $\epsilon > 0$ such that $f(\mathbf{x}^*) \leq f(\mathbf{x})$ for all $\mathbf{x}$ with $||\mathbf{x} - \mathbf{x}^*|| < \epsilon$

For convex functions, any local minimum is also a global minimum.

### Optimality Conditions

A point $\mathbf{x}^*$ is a local minimum only if:
- **First-order condition**: $\nabla f(\mathbf{x}^*) = \mathbf{0}$ (stationary point)
- **Second-order condition**: $\nabla^2 f(\mathbf{x}^*)$ is positive semidefinite

For a strict local minimum, $\nabla^2 f(\mathbf{x}^*)$ must be positive definite.

## Gradient Descent

Gradient descent is the most fundamental optimization algorithm. It iteratively moves in the direction of steepest descent:

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha_k \nabla f(\mathbf{x}_k)$$

where $\alpha_k > 0$ is the step size (learning rate) at iteration $k$.

### Intuition

The gradient $\nabla f(\mathbf{x})$ points in the direction of steepest ascent. Moving in the opposite direction $-\nabla f(\mathbf{x})$ decreases the function value most rapidly.

### Algorithm

```
1. Initialize x_0
2. For k = 0, 1, 2, ...:
   a. Compute gradient: g_k = ∇f(x_k)
   b. Update: x_{k+1} = x_k - α_k * g_k
   c. Check convergence
```

### Step Size Selection

The step size $\alpha_k$ is crucial:

- **Fixed step size**: $\alpha_k = \alpha$ constant
- **Line search**: Choose $\alpha_k$ to minimize $f(\mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k))$
- **Backtracking line search**: Start with large $\alpha$, reduce until sufficient decrease

### Convergence Rate

For a smooth, strongly convex function with condition number $\kappa$, gradient descent with optimal step size achieves:

$$f(\mathbf{x}_k) - f(\mathbf{x}^*) \leq \left(1 - \frac{1}{\kappa}\right)^k (f(\mathbf{x}_0) - f(\mathbf{x}^*))$$

This is linear convergence with rate $\rho = 1 - 1/\kappa$.

## Newton's Method

Newton's method uses second-order information to achieve faster convergence:

$$\mathbf{x}_{k+1} = \mathbf{x}_k - [\nabla^2 f(\mathbf{x}_k)]^{-1} \nabla f(\mathbf{x}_k)$$

### Derivation

Newton's method approximates $f$ by its second-order Taylor expansion:

$$f(\mathbf{x}) \approx f(\mathbf{x}_k) + \nabla f(\mathbf{x}_k)^T(\mathbf{x} - \mathbf{x}_k) + \frac{1}{2}(\mathbf{x} - \mathbf{x}_k)^T\nabla^2 f(\mathbf{x}_k)(\mathbf{x} - \mathbf{x}_k)$$

Minimizing this quadratic approximation yields the Newton update.

### Properties

- **Quadratic convergence**: Under favorable conditions, converges quadratically near the optimum
- **Computational cost**: Requires computing and inverting the Hessian ($O(n^3)$ per iteration)
- **Robustness**: May not converge if started far from optimum or if Hessian is not positive definite

### Quasi-Newton Methods

Quasi-Newton methods approximate the Hessian inverse without computing it explicitly:

- **BFGS**: Updates approximation using gradient differences
- **L-BFGS**: Limited-memory version for large-scale problems

These methods achieve superlinear convergence with lower computational cost than Newton's method.

## Convergence Analysis

### Convergence Definitions

An algorithm converges to $\mathbf{x}^*$ if $\lim_{k \to \infty} \mathbf{x}_k = \mathbf{x}^*$.

**Rate of convergence**:
- **Linear**: $||\mathbf{x}_{k+1} - \mathbf{x}^*|| \leq \rho ||\mathbf{x}_k - \mathbf{x}^*||$ with $\rho \in (0,1)$
- **Superlinear**: $||\mathbf{x}_{k+1} - \mathbf{x}^*|| \leq o(||\mathbf{x}_k - \mathbf{x}^*||)$
- **Quadratic**: $||\mathbf{x}_{k+1} - \mathbf{x}^*|| \leq C ||\mathbf{x}_k - \mathbf{x}^*||^2$

### Conditions for Convergence

**Lipschitz continuity**: The gradient is $L$-Lipschitz continuous if:

$$||\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})|| \leq L ||\mathbf{x} - \mathbf{y}||$$

This implies $f$ is smooth and enables convergence analysis.

**Strong convexity**: A function is $\mu$-strongly convex if:

$$f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T(\mathbf{y} - \mathbf{x}) + \frac{\mu}{2}||\mathbf{y} - \mathbf{x}||^2$$

Strong convexity ensures a unique global minimum and faster convergence.

### Convergence Theorems

**Gradient descent convergence**: If $f$ is $L$-smooth and $\mu$-strongly convex, gradient descent with step size $\alpha \leq 1/L$ converges linearly:

$$||\mathbf{x}_k - \mathbf{x}^*||^2 \leq \left(1 - \frac{\mu}{L}\right)^k ||\mathbf{x}_0 - \mathbf{x}^*||^2$$

The condition number $\kappa = L/\mu$ determines convergence speed.

## Convexity

### Convex Function Definition

A function $f: \mathbb{R}^n \to \mathbb{R}$ is **convex** if for all $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ and $\lambda \in [0,1]$:

$$f(\lambda\mathbf{x} + (1-\lambda)\mathbf{y}) \leq \lambda f(\mathbf{x}) + (1-\lambda)f(\mathbf{y})$$

Geometrically, the line segment between any two points on the graph lies above the graph.

### Strict and Strong Convexity

- **Strictly convex**: Inequality is strict for $\lambda \in (0,1)$
- **$\mu$-strongly convex**: 
  $$f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T(\mathbf{y} - \mathbf{x}) + \frac{\mu}{2}||\mathbf{y} - \mathbf{x}||^2$$

Strong convexity implies a unique global minimum.

### Characterizations

For twice-differentiable functions, convexity is equivalent to:
$$\nabla^2 f(\mathbf{x}) \succeq 0 \quad \forall \mathbf{x}$$

(All eigenvalues of the Hessian are non-negative.)

### Properties

- **Local minimum is global**: For convex functions, any local minimum is a global minimum
- **Sublevel sets are convex**: $\{\mathbf{x} : f(\mathbf{x}) \leq t\}$ is convex
- **Jensen's inequality**: $f(\mathbb{E}[\mathbf{X}]) \leq \mathbb{E}[f(\mathbf{X})]$

## First and Second Order Conditions

### First-Order Necessary Condition

If $\mathbf{x}^*$ is a local minimum and $f$ is differentiable, then:

$$\nabla f(\mathbf{x}^*) = \mathbf{0}$$

This is necessary but not sufficient (consider $f(x) = x^3$ at $x = 0$).

### Second-Order Conditions

**Necessary condition**: If $\mathbf{x}^*$ is a local minimum and $f$ is twice differentiable:

$$\nabla f(\mathbf{x}^*) = \mathbf{0} \quad \text{and} \quad \nabla^2 f(\mathbf{x}^*) \succeq 0$$

**Sufficient condition**: If:

$$\nabla f(\mathbf{x}^*) = \mathbf{0} \quad \text{and} \quad \nabla^2 f(\mathbf{x}^*) \succ 0$$

then $\mathbf{x}^*$ is a strict local minimum.

### Saddle Points

A point $\mathbf{x}$ with $\nabla f(\mathbf{x}) = \mathbf{0}$ but $\nabla^2 f(\mathbf{x})$ having both positive and negative eigenvalues is a **saddle point**. These are common in non-convex optimization (e.g., neural networks).

## Learning Rate Analysis

### Fixed Learning Rate

For $L$-smooth functions, gradient descent converges if:

$$\alpha < \frac{2}{L}$$

The optimal step size is $\alpha = 1/L$, achieving the fastest convergence rate.

### Adaptive Learning Rates

**Momentum**: Adds inertia to gradient updates:

$$\mathbf{v}_{k+1} = \beta \mathbf{v}_k + \nabla f(\mathbf{x}_k)$$
$$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \mathbf{v}_{k+1}$$

where $\beta \in [0,1)$ is the momentum coefficient.

**AdaGrad**: Adapts learning rate per parameter:

$$\mathbf{g}_k = \nabla f(\mathbf{x}_k)$$
$$\mathbf{G}_k = \mathbf{G}_{k-1} + \mathbf{g}_k \odot \mathbf{g}_k$$
$$\mathbf{x}_{k+1} = \mathbf{x}_k - \frac{\alpha}{\sqrt{\mathbf{G}_k + \epsilon}} \odot \mathbf{g}_k$$

**Adam**: Combines momentum and adaptive learning rates:

$$\mathbf{m}_k = \beta_1 \mathbf{m}_{k-1} + (1-\beta_1)\mathbf{g}_k$$
$$\mathbf{v}_k = \beta_2 \mathbf{v}_{k-1} + (1-\beta_2)\mathbf{g}_k \odot \mathbf{g}_k$$
$$\hat{\mathbf{m}}_k = \frac{\mathbf{m}_k}{1-\beta_1^k}, \quad \hat{\mathbf{v}}_k = \frac{\mathbf{v}_k}{1-\beta_2^k}$$
$$\mathbf{x}_{k+1} = \mathbf{x}_k - \frac{\alpha}{\sqrt{\hat{\mathbf{v}}_k} + \epsilon} \odot \hat{\mathbf{m}}_k$$

### Learning Rate Scheduling

Common schedules:
- **Step decay**: Reduce by factor $\gamma$ every $T$ iterations
- **Exponential decay**: $\alpha_k = \alpha_0 \gamma^k$
- **Cosine annealing**: $\alpha_k = \alpha_{\min} + (\alpha_{\max} - \alpha_{\min})(1 + \cos(\pi k/K))/2$

## Machine Learning Applications

### Training Neural Networks

Neural network training minimizes:

$$\min_{\boldsymbol{\theta}} \frac{1}{n}\sum_{i=1}^{n} L(f(\mathbf{x}_i; \boldsymbol{\theta}), y_i) + \lambda R(\boldsymbol{\theta})$$

where:
- $L$ is the loss function (e.g., cross-entropy, MSE)
- $f(\mathbf{x}; \boldsymbol{\theta})$ is the neural network
- $R(\boldsymbol{\theta})$ is regularization (e.g., weight decay)

**Backpropagation** computes gradients efficiently using the chain rule:

$$\frac{\partial L}{\partial \theta_j} = \sum_{i=1}^{n} \frac{\partial L}{\partial f(\mathbf{x}_i)} \frac{\partial f(\mathbf{x}_i)}{\partial \theta_j}$$

### Stochastic Gradient Descent

For large datasets, use stochastic gradient descent (SGD):

$$\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \alpha_k \nabla_{\boldsymbol{\theta}} L(f(\mathbf{x}_{i_k}; \boldsymbol{\theta}_k), y_{i_k})$$

where $i_k$ is randomly sampled. This reduces computational cost per iteration.

**Mini-batch SGD** uses a batch of samples:

$$\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \alpha_k \frac{1}{B}\sum_{i \in \mathcal{B}_k} \nabla_{\boldsymbol{\theta}} L(f(\mathbf{x}_i; \boldsymbol{\theta}_k), y_i)$$

### Loss Minimization

Common loss functions:

**Mean Squared Error**:
$$L(\hat{\mathbf{y}}, \mathbf{y}) = \frac{1}{n}||\hat{\mathbf{y}} - \mathbf{y}||^2$$

**Cross-Entropy** (for classification):
$$L(\hat{\mathbf{y}}, \mathbf{y}) = -\frac{1}{n}\sum_{i=1}^{n} \sum_{c=1}^{C} y_{ic} \log \hat{y}_{ic}$$

**Hinge Loss** (for SVMs):
$$L(\hat{\mathbf{y}}, \mathbf{y}) = \frac{1}{n}\sum_{i=1}^{n} \max(0, 1 - y_i \hat{y}_i)$$

### Learning Rate Analysis in Practice

For neural networks:
- **Too large**: Training diverges or oscillates
- **Too small**: Slow convergence, may get stuck in poor local minima
- **Optimal**: Fast convergence without instability

Common heuristics:
- Start with $\alpha = 0.01$ or $0.001$
- Use learning rate finder: train with exponentially increasing rates, choose rate before loss spikes
- Reduce learning rate when validation loss plateaus

### Non-Convex Optimization

Neural network loss landscapes are non-convex with:
- Many local minima
- Saddle points
- Flat regions

Despite non-convexity, SGD often finds good solutions because:
- Local minima may be equivalent (symmetries)
- Saddle points are usually not problematic
- Flat minima generalize better than sharp minima

## Key Takeaways

1. **Unconstrained optimization** seeks to minimize objective functions without constraints, forming the foundation of machine learning training algorithms.

2. **Gradient descent** iteratively moves in the direction of steepest descent, converging linearly for smooth, strongly convex functions with appropriate step sizes.

3. **Newton's method** uses second-order information for quadratic convergence but requires computing and inverting the Hessian, making it expensive for large problems.

4. **Convergence analysis** depends on smoothness (Lipschitz gradients) and strong convexity, with the condition number $\kappa = L/\mu$ determining convergence speed.

5. **Convexity** ensures that local minima are global, simplifying optimization, though most neural network problems are non-convex.

6. **First-order conditions** ($\nabla f = \mathbf{0}$) are necessary for optimality, while second-order conditions (positive semidefinite Hessian) provide sufficiency.

7. **Learning rate selection** is critical: too large causes divergence, too small slows convergence; adaptive methods like Adam help automate this choice.

8. **Stochastic gradient descent** enables training on large datasets by using random samples, trading off per-iteration accuracy for computational efficiency.

9. **Neural network training** involves non-convex optimization, but SGD with appropriate learning rates and schedules often finds good solutions despite the complexity.

10. **Understanding optimization theory** enables diagnosing training problems, designing better algorithms, and understanding why certain techniques (momentum, adaptive learning rates) work effectively.
