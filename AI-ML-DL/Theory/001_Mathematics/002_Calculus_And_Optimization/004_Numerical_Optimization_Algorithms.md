# Numerical Optimization Algorithms

## Table of Contents

1. [Introduction](#introduction)
2. [First-Order Methods](#first-order-methods)
3. [Quasi-Newton Methods](#quasi-newton-methods)
4. [Stochastic Optimization](#stochastic-optimization)
5. [Adaptive Learning Rates](#adaptive-learning-rates)
6. [Momentum Methods](#momentum-methods)
7. [Learning Rate Schedules](#learning-rate-schedules)
8. [Second-Order Methods](#second-order-methods)
9. [Convergence Analysis](#convergence-analysis)
10. [Machine Learning Applications](#machine-learning-applications)
11. [Key Takeaways](#key-takeaways)

## Introduction

Numerical optimization algorithms are the computational engines that train machine learning models. While gradient descent provides the foundation, modern ML relies on sophisticated algorithms that adapt learning rates, use momentum, approximate second-order information, and handle stochastic gradients efficiently. This document covers quasi-Newton methods (BFGS, L-BFGS), adaptive optimizers (Adam, RMSprop), momentum, and learning rate schedules used in deep learning.

## First-Order Methods

### Gradient Descent Variants

**Batch Gradient Descent**: Uses full dataset:

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha_k \frac{1}{n}\sum_{i=1}^n \nabla f_i(\mathbf{x}_k)$$

**Stochastic Gradient Descent (SGD)**: Uses single example:

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha_k \nabla f_{i_k}(\mathbf{x}_k)$$

where $i_k$ is randomly sampled.

**Mini-Batch SGD**: Uses subset:

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha_k \frac{1}{|\mathcal{B}_k|}\sum_{i \in \mathcal{B}_k} \nabla f_i(\mathbf{x}_k)$$

where $\mathcal{B}_k$ is mini-batch.

### Convergence of SGD

**Theorem**: For convex $f$ with bounded gradients $\|\nabla f_i(\mathbf{x})\| \leq G$ and step size $\alpha_k = \alpha/\sqrt{k}$:

$$\mathbb{E}[f(\bar{\mathbf{x}}_T)] - f^* \leq \frac{G^2(1 + \log T)}{2\alpha\sqrt{T}} + \frac{\alpha R^2}{2\sqrt{T}}$$

where $\bar{\mathbf{x}}_T = \frac{1}{T}\sum_{k=1}^T \mathbf{x}_k$ and $R$ is distance to optimum.

**Rate**: $O(1/\sqrt{T})$ convergence (slower than batch GD's $O(1/T)$ but per-iteration cost is $1/n$ times smaller).

## Quasi-Newton Methods

### Motivation

Newton's method requires computing and inverting Hessian: $O(n^3)$ cost. **Quasi-Newton methods** approximate Hessian using gradient information, achieving superlinear convergence with $O(n^2)$ cost per iteration.

### BFGS Algorithm

**BFGS** (Broyden-Fletcher-Goldfarb-Shanno) maintains approximation $\mathbf{B}_k \approx \mathbf{H}_f(\mathbf{x}_k)$.

**Update**:

$$\mathbf{B}_{k+1} = \mathbf{B}_k + \frac{\mathbf{y}_k\mathbf{y}_k^T}{\mathbf{y}_k^T\mathbf{s}_k} - \frac{\mathbf{B}_k\mathbf{s}_k\mathbf{s}_k^T\mathbf{B}_k}{\mathbf{s}_k^T\mathbf{B}_k\mathbf{s}_k}$$

where:
- $\mathbf{s}_k = \mathbf{x}_{k+1} - \mathbf{x}_k$ (step)
- $\mathbf{y}_k = \nabla f(\mathbf{x}_{k+1}) - \nabla f(\mathbf{x}_k)$ (gradient change)

**Secant condition**: $\mathbf{B}_{k+1}\mathbf{s}_k = \mathbf{y}_k$ (satisfied by update).

**Algorithm**:
1. Compute search direction: $\mathbf{p}_k = -\mathbf{B}_k^{-1}\nabla f(\mathbf{x}_k)$
2. Line search: $\alpha_k = \arg\min_\alpha f(\mathbf{x}_k + \alpha\mathbf{p}_k)$
3. Update: $\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k\mathbf{p}_k$
4. Update $\mathbf{B}_k$ using BFGS formula

### L-BFGS

**Limited-memory BFGS** stores only last $m$ pairs $(\mathbf{s}_i, \mathbf{y}_i)$ instead of full matrix $\mathbf{B}_k$.

**Two-loop recursion** computes $\mathbf{B}_k^{-1}\mathbf{g}_k$ without storing $\mathbf{B}_k$:

1. **Forward loop**: Compute $\mathbf{q}$ using recent pairs
2. **Backward loop**: Compute $\mathbf{p} = \mathbf{B}_0^{-1}\mathbf{q}$ using stored pairs

**Memory**: $O(mn)$ instead of $O(n^2)$.

**Initialization**: $\mathbf{B}_0^{-1} = \gamma_k\mathbf{I}$ where $\gamma_k = \frac{\mathbf{s}_{k-1}^T\mathbf{y}_{k-1}}{\mathbf{y}_{k-1}^T\mathbf{y}_{k-1}}$ (scaling).

### Convergence Properties

**Superlinear convergence**: For smooth strongly convex functions:

$$\lim_{k \to \infty} \frac{\|\mathbf{x}_{k+1} - \mathbf{x}^*\|}{\|\mathbf{x}_k - \mathbf{x}^*\|} = 0$$

**Rate**: Faster than linear but slower than quadratic (Newton).

## Stochastic Optimization

### Stochastic Gradient Descent

**SGD** uses noisy gradient estimates:

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha_k \hat{\nabla} f(\mathbf{x}_k)$$

where $\mathbb{E}[\hat{\nabla} f(\mathbf{x}_k)] = \nabla f(\mathbf{x}_k)$ (unbiased estimate).

**Variance reduction**: Key challenge is reducing variance of gradient estimates.

### SVRG (Stochastic Variance Reduced Gradient)

**Idea**: Periodically compute full gradient $\tilde{\mathbf{g}} = \nabla f(\tilde{\mathbf{x}})$ at snapshot $\tilde{\mathbf{x}}$, then use:

$$\mathbf{v}_k = \nabla f_{i_k}(\mathbf{x}_k) - \nabla f_{i_k}(\tilde{\mathbf{x}}) + \tilde{\mathbf{g}}$$

This is unbiased: $\mathbb{E}[\mathbf{v}_k] = \nabla f(\mathbf{x}_k)$, and has reduced variance if $\mathbf{x}_k \approx \tilde{\mathbf{x}}$.

**Algorithm**:
1. Snapshot: $\tilde{\mathbf{x}} = \mathbf{x}_k$, compute $\tilde{\mathbf{g}} = \nabla f(\tilde{\mathbf{x}})$
2. For $m$ iterations: $\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \mathbf{v}_k$
3. Repeat

### SAGA

**SAGA** stores gradient table $\{\mathbf{g}_i\}$ for each example:

$$\mathbf{v}_k = \nabla f_{i_k}(\mathbf{x}_k) - \mathbf{g}_{i_k} + \frac{1}{n}\sum_{i=1}^n \mathbf{g}_i$$

Update: $\mathbf{g}_{i_k} = \nabla f_{i_k}(\mathbf{x}_k)$.

## Adaptive Learning Rates

### AdaGrad

**AdaGrad** adapts learning rate per parameter:

$$G_{k,ii} = \sum_{j=1}^k (\nabla f(\mathbf{x}_j))_i^2$$
$$\mathbf{x}_{k+1,i} = \mathbf{x}_{k,i} - \frac{\alpha}{\sqrt{G_{k,ii} + \epsilon}} (\nabla f(\mathbf{x}_k))_i$$

**Intuition**: Parameters with large gradients get smaller learning rates.

**Properties**:
- Automatically decreases learning rate
- Good for sparse gradients
- Can decrease too aggressively (learning rate $\to 0$)

### RMSprop

**RMSprop** uses exponential moving average:

$$v_{k,i} = \beta v_{k-1,i} + (1-\beta)(\nabla f(\mathbf{x}_k))_i^2$$
$$\mathbf{x}_{k+1,i} = \mathbf{x}_{k,i} - \frac{\alpha}{\sqrt{v_{k,i} + \epsilon}} (\nabla f(\mathbf{x}_k))_i$$

**Advantage**: Doesn't accumulate all past gradients, adapts to recent gradient magnitudes.

### Adam Optimizer

**Adam** (Adaptive Moment Estimation) combines momentum and adaptive learning rates:

**First moment** (gradient):
$$m_{k,i} = \beta_1 m_{k-1,i} + (1-\beta_1)(\nabla f(\mathbf{x}_k))_i$$

**Second moment** (squared gradient):
$$v_{k,i} = \beta_2 v_{k-1,i} + (1-\beta_2)(\nabla f(\mathbf{x}_k))_i^2$$

**Bias correction**:
$$\hat{m}_{k,i} = \frac{m_{k,i}}{1-\beta_1^k}, \quad \hat{v}_{k,i} = \frac{v_{k,i}}{1-\beta_2^k}$$

**Update**:
$$\mathbf{x}_{k+1,i} = \mathbf{x}_{k,i} - \frac{\alpha}{\sqrt{\hat{v}_{k,i}} + \epsilon} \hat{m}_{k,i}$$

**Default parameters**: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

**Properties**:
- Combines benefits of momentum and adaptive learning rates
- Works well in practice for deep learning
- Requires tuning $\alpha$ but less sensitive than SGD

### AdamW

**AdamW** decouples weight decay from gradient-based updates:

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \left( \frac{\hat{m}_k}{\sqrt{\hat{v}_k} + \epsilon} + \lambda \mathbf{x}_k \right)$$

Separates learning rate adaptation from regularization.

## Momentum Methods

### Momentum

**Momentum** accumulates gradient history:

$$v_k = \beta v_{k-1} + \nabla f(\mathbf{x}_k)$$
$$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha v_k$$

where $\beta \in [0, 1)$ is momentum coefficient.

**Intuition**: Like a ball rolling downhill, accumulates velocity in consistent directions.

**Benefits**:
- Accelerates convergence in consistent directions
- Reduces oscillations
- Helps escape shallow local minima

### Nesterov Accelerated Gradient

**NAG** uses "lookahead" gradient:

$$v_k = \beta v_{k-1} + \nabla f(\mathbf{x}_k - \beta v_{k-1})$$
$$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha v_k$$

Evaluates gradient at "predicted" position $\mathbf{x}_k - \beta v_{k-1}$.

**Convergence**: For convex functions, achieves $O(1/k^2)$ rate vs $O(1/k)$ for gradient descent.

## Learning Rate Schedules

### Fixed Learning Rate

Simplest: $\alpha_k = \alpha$ constant.

**Issues**: May be too large (divergence) or too small (slow convergence).

### Step Decay

$$\alpha_k = \alpha_0 \gamma^{\lfloor k/s \rfloor}$$

Decreases by factor $\gamma$ every $s$ steps.

**Example**: $\alpha_0 = 0.1$, $\gamma = 0.5$, $s = 30$: halves every 30 iterations.

### Exponential Decay

$$\alpha_k = \alpha_0 e^{-\gamma k}$$

Continuous decay.

### Polynomial Decay

$$\alpha_k = \alpha_0 (1 + \gamma k)^{-p}$$

Common: $p = 0.5$ gives $\alpha_k = \alpha_0/\sqrt{k}$.

### Cosine Annealing

$$\alpha_k = \alpha_{\min} + (\alpha_{\max} - \alpha_{\min}) \frac{1 + \cos(\pi k/T)}{2}$$

Smoothly decreases following cosine curve over $T$ iterations.

### Warm Restarts

**SGDR** (Stochastic Gradient Descent with Warm Restarts): Periodically reset learning rate to initial value, enabling escape from local minima.

### One-Cycle Policy

**One-cycle**: Increase learning rate to maximum, then decrease. Often combined with momentum scheduling.

## Second-Order Methods

### Natural Gradient

**Natural gradient** uses Fisher information matrix $\mathbf{F}$:

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \mathbf{F}^{-1}(\mathbf{x}_k)\nabla f(\mathbf{x}_k)$$

Accounts for geometry of parameter space.

**Applications**: Policy gradient methods in reinforcement learning.

### K-FAC

**Kronecker-Factored Approximate Curvature** approximates Fisher information for neural networks using Kronecker products:

$$\mathbf{F} \approx \mathbf{A} \otimes \mathbf{G}$$

where $\mathbf{A}$ and $\mathbf{G}$ are smaller matrices, enabling efficient inversion.

### Hessian-Free Methods

**Hessian-vector products** $\mathbf{H}\mathbf{v}$ computed via automatic differentiation without storing $\mathbf{H}$.

Used in:
- Conjugate gradient methods
- Truncated Newton methods
- Curvature information for adaptive methods

## Convergence Analysis

### Convergence Rates

**Sublinear**: $O(1/\sqrt{k})$ for SGD on convex functions

**Linear**: $O(\rho^k)$ for gradient descent on strongly convex functions, $\rho < 1$

**Superlinear**: Quasi-Newton methods

**Quadratic**: Newton's method

### Conditions for Convergence

**Lipschitz smoothness**: $\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \leq L\|\mathbf{x} - \mathbf{y}\|$

**Strong convexity**: $f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T(\mathbf{y} - \mathbf{x}) + \frac{\mu}{2}\|\mathbf{y} - \mathbf{x}\|^2$

**Condition number**: $\kappa = L/\mu$ affects convergence rate.

### Convergence Guarantees

**Gradient descent** on $L$-smooth $\mu$-strongly convex function:

$$\|\mathbf{x}_k - \mathbf{x}^*\|^2 \leq \left(1 - \frac{\mu}{L}\right)^k \|\mathbf{x}_0 - \mathbf{x}^*\|^2$$

**SGD** with decreasing step size $\alpha_k = \alpha/\sqrt{k}$:

$$\mathbb{E}[f(\bar{\mathbf{x}}_T)] - f^* \leq O(1/\sqrt{T})$$

## Machine Learning Applications

### Training Neural Networks

**Adam** is default choice for many deep learning tasks:
- Adaptive learning rates per parameter
- Momentum for faster convergence
- Works well with default hyperparameters

**SGD with momentum** still competitive:
- Better generalization in some cases
- More predictable behavior
- Used in large-batch training

### Large-Scale Training

**Distributed optimization**:
- Data parallelism: Split data across workers
- Model parallelism: Split model across devices
- Gradient synchronization: Average gradients

**Gradient compression**: Reduce communication in distributed training.

### Transfer Learning

**Fine-tuning** with small learning rate:
$$\alpha_{\text{fine-tune}} = 0.1 \times \alpha_{\text{pretrain}}$$

**Layer-wise learning rates**: Different rates for different layers.

### Hyperparameter Optimization

**Learning rate** is most important hyperparameter:
- Too large: Divergence or instability
- Too small: Slow convergence
- Adaptive methods reduce sensitivity

**Warmup**: Gradually increase learning rate at start of training.

### Regularization Interaction

**Weight decay** interacts with optimizer:
- SGD: $\mathbf{x}_{k+1} = (1-\lambda\alpha)\mathbf{x}_k - \alpha\nabla f(\mathbf{x}_k)$
- Adam: Decoupled weight decay (AdamW) often better

## Key Takeaways

1. **SGD** is foundation but has limitations: slow convergence, requires careful learning rate tuning.

2. **Momentum** accelerates convergence by accumulating gradient history in consistent directions.

3. **Adaptive learning rates** (AdaGrad, RMSprop, Adam) automatically adjust per-parameter learning rates based on gradient history.

4. **Adam** combines momentum and adaptive learning rates, working well out-of-the-box for many deep learning tasks.

5. **Quasi-Newton methods** (BFGS, L-BFGS) approximate second-order information efficiently, achieving superlinear convergence.

6. **Learning rate schedules** are crucial: decay helps convergence, warmup helps stability, restarts can escape local minima.

7. **Stochastic methods** enable training on large datasets by using gradient estimates from subsets.

8. **Variance reduction** (SVRG, SAGA) improves SGD convergence by reducing gradient estimate variance.

9. **Second-order methods** provide faster convergence but are computationally expensive; approximations (K-FAC) help.

10. **Optimizer choice** depends on problem: Adam for general deep learning, SGD+momentum for some cases, L-BFGS for smaller problems.
