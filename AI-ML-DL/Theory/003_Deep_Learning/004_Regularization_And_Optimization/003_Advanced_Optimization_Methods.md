# Advanced Optimization Methods

## Table of Contents

1. [Introduction](#introduction)
2. [Second-Order Methods](#second-order-methods)
3. [Natural Gradient Descent](#natural-gradient-descent)
4. [K-FAC Approximation](#k-fac-approximation)
5. [Sharpness-Aware Minimization](#sharpness-aware-minimization)
6. [Lookahead Optimizer](#lookahead-optimizer)
7. [Other Advanced Methods](#other-advanced-methods)
8. [Theoretical Foundations](#theoretical-foundations)
9. [Practical Considerations](#practical-considerations)
10. [Key Takeaways](#key-takeaways)

## Introduction

Advanced optimization methods go beyond first-order gradient descent, incorporating second-order information, natural gradients, and novel techniques to improve convergence and generalization. These methods address limitations of standard optimizers and enable training of more challenging models.

This chapter covers advanced optimization techniques, from second-order methods to natural gradients, K-FAC, and sharpness-aware minimization.

## Second-Order Methods

### Newton's Method

Uses Hessian matrix for second-order information:

$$\theta_{t+1} = \theta_t - \eta H^{-1}(\theta_t) \nabla_{\theta} \mathcal{L}(\theta_t)$$

where $H(\theta) = \nabla^2_{\theta} \mathcal{L}(\theta)$ is the Hessian.

**Properties**:
- Quadratic convergence (near optimum)
- Uses curvature information
- More accurate steps

**Limitations**:
- Expensive: $O(n^3)$ to invert Hessian
- Memory: $O(n^2)$ for Hessian
- May not be positive definite

### Quasi-Newton Methods

Approximate Hessian without computing it:

**BFGS**: Updates Hessian approximation

$$B_{t+1} = B_t + \frac{\mathbf{y}_t \mathbf{y}_t^T}{\mathbf{y}_t^T \mathbf{s}_t} - \frac{B_t \mathbf{s}_t \mathbf{s}_t^T B_t}{\mathbf{s}_t^T B_t \mathbf{s}_t}$$

where $\mathbf{s}_t = \theta_{t+1} - \theta_t$ and $\mathbf{y}_t = \nabla_{t+1} - \nabla_t$.

**L-BFGS**: Limited memory version
- Stores only recent updates
- More memory efficient
- Still expensive for large models

### Gauss-Newton Method

For least squares problems:

$$H \approx J^T J$$

where $J$ is Jacobian of residuals.

**Levenberg-Marquardt**: Regularized version

$$(J^T J + \lambda I) \Delta \theta = -J^T \mathbf{r}$$

## Natural Gradient Descent

Natural gradient uses geometry of parameter space.

### Motivation

Standard gradient depends on parameterization:
- Different parameterizations → different updates
- Natural gradient is parameterization-invariant

### Fisher Information Matrix

$$F(\theta) = \mathbb{E}_{\mathbf{x} \sim p(\mathbf{x}|\theta)}[\nabla_{\theta} \log p(\mathbf{x}|\theta) \nabla_{\theta} \log p(\mathbf{x}|\theta)^T]$$

### Natural Gradient Update

$$\theta_{t+1} = \theta_t - \eta F^{-1}(\theta_t) \nabla_{\theta} \mathcal{L}(\theta_t)$$

**Properties**:
- Parameterization-invariant
- Respects geometry
- Better convergence

**Challenges**:
- Computing $F$ is expensive
- Inverting $F$ is expensive
- Need approximations

## K-FAC Approximation

Kronecker-Factored Approximate Curvature approximates Fisher information.

### Kronecker Product

For matrices $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{p \times q}$:

$$A \otimes B = \begin{bmatrix} a_{11}B & \cdots & a_{1n}B \\ \vdots & \ddots & \vdots \\ a_{m1}B & \cdots & a_{mn}B \end{bmatrix}$$

### K-FAC Approximation

For linear layer $y = Wx$:

$$F \approx \mathbb{E}[\mathbf{a}\mathbf{a}^T] \otimes \mathbb{E}[\mathbf{g}\mathbf{g}^T]$$

where $\mathbf{a}$ is activation and $\mathbf{g}$ is gradient.

**Benefits**:
- Efficient: Factorize large matrix
- Accurate: Good approximation
- Practical: Can be computed

### K-FAC Update

1. Compute Kronecker factors
2. Invert factors (cheaper)
3. Update parameters

**Complexity**: $O(n^{3/2})$ instead of $O(n^3)$

### Implementation Considerations

- Periodic updates (not every step)
- Momentum
- Damping for stability
- Works well for large models

## Sharpness-Aware Minimization

SAM minimizes loss and sharpness simultaneously.

### Sharpness Definition

For neighborhood $\rho$:

$$\max_{||\epsilon|| \leq \rho} \mathcal{L}(\theta + \epsilon) - \mathcal{L}(\theta)$$

### SAM Objective

$$\min_{\theta} \max_{||\epsilon|| \leq \rho} \mathcal{L}(\theta + \epsilon)$$

Minimize worst-case loss in neighborhood.

### Update Rule

1. **Compute Perturbation**:

$$\epsilon^* = \rho \frac{\nabla_{\theta} \mathcal{L}(\theta)}{||\nabla_{\theta} \mathcal{L}(\theta)||}$$

2. **Compute Gradient at Perturbed Point**:

$$\mathbf{g} = \nabla_{\theta} \mathcal{L}(\theta + \epsilon^*)$$

3. **Update**:

$$\theta_{t+1} = \theta_t - \eta \mathbf{g}$$

### Benefits

- Finds flatter minima
- Better generalization
- More robust

### Computational Cost

- Two forward passes
- Two backward passes
- ~2x computation

### Variants

**ASAM**: Adaptive SAM
**FSAM**: Efficient SAM
**GSAM**: Gradient-based SAM

## Lookahead Optimizer

Lookahead maintains slow and fast weights.

### Algorithm

**Inner Loop** (Fast Weights):
- Update $k$ steps with base optimizer
- Fast weights: $\theta_{\text{fast}}$

**Outer Loop** (Slow Weights):
- Update slow weights: $\theta_{\text{slow}} \leftarrow \theta_{\text{slow}} + \alpha(\theta_{\text{fast}} - \theta_{\text{slow}})$

where $\alpha$ is interpolation factor.

### Properties

- More stable training
- Better generalization
- Works with any base optimizer
- Minimal overhead

### Intuition

- Fast weights: Explore quickly
- Slow weights: Stable average
- Balance exploration and stability

## Other Advanced Methods

### Shampoo

Preconditioned optimizer:
- Uses statistics from gradients
- Efficient for large models
- Good empirical performance

### AdaHessian

Adaptive learning rate with Hessian:
- Estimates diagonal Hessian
- Adaptive per parameter
- More stable than Adam

### LARS

Layer-wise Adaptive Rate Scaling:
- Different learning rates per layer
- Scales by gradient norm
- Enables large batch training

### LAMB

Layer-wise Adaptive Moments:
- Combines Adam with LARS
- Enables very large batches
- Used in BERT training

## Theoretical Foundations

### Convergence Rates

**First-Order**: $O(1/\sqrt{T})$ for convex
**Second-Order**: $O(1/T^2)$ for strongly convex
**Natural Gradient**: Better convergence

### Geometry of Loss Landscape

- Curvature matters
- Flat minima generalize better
- Natural gradient respects geometry

### Generalization

- Sharp minima: Poor generalization
- Flat minima: Better generalization
- SAM finds flatter minima

## Practical Considerations

### When to Use Advanced Methods

**Second-Order**:
- Small to medium models
- When computation affordable
- When convergence critical

**Natural Gradient/K-FAC**:
- Large models
- When geometry matters
- Periodic updates acceptable

**SAM**:
- When generalization critical
- Can afford 2x computation
- Want flatter minima

**Lookahead**:
- With any optimizer
- Want stability
- Minimal overhead acceptable

### Computational Cost

- Second-order: Expensive
- K-FAC: Moderate
- SAM: 2x standard
- Lookahead: Minimal

### Hyperparameter Tuning

- More hyperparameters
- Tune on validation set
- Start with defaults
- Adjust based on results

### Implementation

- Use existing libraries when possible
- Understand trade-offs
- Monitor convergence
- Compare with baselines

## Key Takeaways

1. **Second-Order Methods**: Newton's method and quasi-Newton methods use Hessian information for faster convergence but are computationally expensive.

2. **Natural Gradient**: Uses Fisher information matrix to provide parameterization-invariant updates that respect the geometry of parameter space.

3. **K-FAC**: Kronecker-factored approximation efficiently approximates Fisher information, making natural gradient practical for large models.

4. **Sharpness-Aware Minimization**: Minimizes worst-case loss in neighborhood, finding flatter minima that generalize better, at cost of ~2x computation.

5. **Lookahead**: Maintains slow and fast weights, providing stability and better generalization with minimal overhead over base optimizer.

6. **Other Methods**: Shampoo, AdaHessian, LARS, and LAMB provide various improvements for specific scenarios like large batch training.

7. **Theoretical Foundations**: Advanced methods leverage curvature information, geometry of parameter space, and connection between flat minima and generalization.

8. **Computational Trade-offs**: Advanced methods trade computation for better convergence or generalization, requiring careful consideration of costs and benefits.

9. **Practical Use**: Choose methods based on model size, computational budget, and priorities (convergence speed vs. generalization).

10. **Implementation**: Use existing implementations when possible, tune hyperparameters carefully, and compare with standard optimizers to validate benefits.
