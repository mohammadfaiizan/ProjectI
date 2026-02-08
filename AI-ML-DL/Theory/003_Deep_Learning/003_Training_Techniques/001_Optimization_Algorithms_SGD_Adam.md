# Optimization Algorithms: SGD to Adam

## Table of Contents

1. [Introduction](#introduction)
2. [Gradient Descent Fundamentals](#gradient-descent-fundamentals)
3. [Stochastic Gradient Descent](#stochastic-gradient-descent)
4. [Momentum](#momentum)
5. [Nesterov Accelerated Gradient](#nesterov-accelerated-gradient)
6. [Adaptive Learning Rate Methods](#adaptive-learning-rate-methods)
7. [Adam and AdamW](#adam-and-adamw)
8. [LAMB and Other Advanced Optimizers](#lamb-and-other-advanced-optimizers)
9. [Learning Rate Schedules](#learning-rate-schedules)
10. [Key Takeaways](#key-takeaways)

## Introduction

Optimization algorithms are fundamental to training neural networks, determining how parameters are updated based on gradients. From basic gradient descent to sophisticated adaptive methods like Adam, the choice of optimizer significantly impacts training speed, convergence, and final performance.

This chapter covers optimization algorithms used in deep learning, from stochastic gradient descent to modern adaptive methods, examining their mathematical foundations, properties, and practical considerations.

## Gradient Descent Fundamentals

### Batch Gradient Descent

Update parameters using full dataset:

$$\theta_{t+1} = \theta_t - \eta \nabla_{\theta} \mathcal{L}(\theta_t)$$

where $\eta$ is the learning rate and $\nabla_{\theta} \mathcal{L}$ is the gradient over all training examples.

**Properties**:
- Deterministic (given same initialization)
- Computationally expensive for large datasets
- May get stuck in local minima
- Smooth convergence

### Learning Rate

The learning rate $\eta$ controls step size:
- **Too Small**: Slow convergence, may get stuck
- **Too Large**: Overshoot, oscillations, divergence
- **Optimal**: Balances speed and stability

### Convergence

Under convexity assumptions, gradient descent converges to global minimum with rate $O(1/t)$.

For non-convex problems (neural networks), convergence to local minima or saddle points.

## Stochastic Gradient Descent

### Mini-Batch SGD

Update using subset of data:

$$\theta_{t+1} = \theta_t - \eta \nabla_{\theta} \mathcal{L}_{\mathcal{B}_t}(\theta_t)$$

where $\mathcal{B}_t$ is a mini-batch at iteration $t$.

**Advantages**:
- Faster updates (less computation per iteration)
- Noise helps escape local minima
- Enables online learning
- More frequent updates

**Disadvantages**:
- Noisy gradients
- May oscillate
- Requires learning rate tuning

### Batch Size

**Small Batches** (1-32):
- More noise, better generalization
- More updates per epoch
- Slower (less parallelization)

**Large Batches** (256-1024+):
- Less noise, faster convergence
- Fewer updates per epoch
- Faster (better parallelization)

**Very Large Batches**:
- May hurt generalization
- Require learning rate scaling

### Learning Rate Scaling

For large batches, scale learning rate:

$$\eta_{\text{large}} = \eta_{\text{base}} \times \frac{B_{\text{large}}}{B_{\text{base}}}$$

or use square root scaling:

$$\eta_{\text{large}} = \eta_{\text{base}} \times \sqrt{\frac{B_{\text{large}}}{B_{\text{base}}}}$$

## Momentum

Momentum accumulates gradient history to smooth updates.

### Momentum Update

$$v_{t+1} = \mu v_t + \eta \nabla_{\theta} \mathcal{L}(\theta_t)$$

$$\theta_{t+1} = \theta_t - v_{t+1}$$

where $\mu \in [0,1)$ is the momentum coefficient (typically 0.9).

### Intuition

- **Velocity**: $v_t$ accumulates gradient history
- **Damping**: $\mu$ controls how much history to keep
- **Smoothing**: Reduces oscillations
- **Acceleration**: Helps in consistent directions

### Benefits

1. **Faster Convergence**: Accelerates in consistent directions
2. **Smoother**: Reduces oscillations
3. **Escapes Local Minima**: Momentum helps escape shallow minima
4. **Handles Ravines**: Better navigation of narrow valleys

### Nesterov Formulation

Sometimes written as:

$$v_{t+1} = \mu v_t + \eta \nabla_{\theta} \mathcal{L}(\theta_t - \mu v_t)$$

$$\theta_{t+1} = \theta_t - v_{t+1}$$

## Nesterov Accelerated Gradient

Nesterov momentum looks ahead before computing gradient.

### Nesterov Update

$$v_{t+1} = \mu v_t + \eta \nabla_{\theta} \mathcal{L}(\theta_t - \mu v_t)$$

$$\theta_{t+1} = \theta_t - v_{t+1}$$

### Difference from Momentum

- **Momentum**: Gradient at current position
- **Nesterov**: Gradient at lookahead position ($\theta_t - \mu v_t$)

### Benefits

- **Better Convergence**: Theoretical $O(1/t^2)$ vs $O(1/t)$
- **Less Overshooting**: Corrects before overshooting
- **Faster**: Often converges faster than momentum

### Practical Form

Common implementation:

$$v_{t+1} = \mu v_t + \nabla_{\theta} \mathcal{L}(\theta_t)$$

$$\theta_{t+1} = \theta_t - \eta(v_{t+1} + \mu(v_{t+1} - v_t))$$

## Adaptive Learning Rate Methods

Adaptive methods adjust learning rate per parameter based on gradient history.

### AdaGrad

Accumulates squared gradients:

$$G_t = G_{t-1} + (\nabla_{\theta} \mathcal{L}(\theta_t))^2$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot \nabla_{\theta} \mathcal{L}(\theta_t)$$

**Properties**:
- Learning rate decreases for parameters with large gradients
- Good for sparse gradients
- May decrease too aggressively

**Limitation**: $G_t$ grows monotonically, learning rate decays to zero

### RMSprop

Exponentially decaying average of squared gradients:

$$E[g^2]_t = \rho E[g^2]_{t-1} + (1-\rho) g_t^2$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t$$

where $\rho$ is decay rate (typically 0.9).

**Benefits**:
- Addresses AdaGrad's aggressive decay
- Adapts to recent gradient magnitudes
- Works well in non-stationary settings

### AdaDelta

Extension of RMSprop that also adapts learning rate:

$$E[\Delta\theta^2]_t = \rho E[\Delta\theta^2]_{t-1} + (1-\rho) \Delta\theta_t^2$$

$$\Delta\theta_t = -\frac{\sqrt{E[\Delta\theta^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}} g_t$$

**Properties**:
- No learning rate hyperparameter
- Adapts both numerator and denominator
- More robust to hyperparameters

## Adam and AdamW

### Adam

Adaptive Moment Estimation combines momentum and RMSprop.

**First Moment** (biased):

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

**Second Moment** (biased):

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

**Bias Correction**:

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$$

$$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

**Update**:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**Hyperparameters**:
- $\beta_1 = 0.9$: Momentum decay
- $\beta_2 = 0.999$: Second moment decay
- $\epsilon = 10^{-8}$: Numerical stability

### Properties

1. **Adaptive**: Per-parameter learning rates
2. **Momentum**: Uses first moment (gradient)
3. **Adaptive Scaling**: Uses second moment (squared gradient)
4. **Bias Correction**: Accounts for initialization bias

### Advantages

- Fast convergence
- Robust to hyperparameters
- Works well in practice
- Default choice for many applications

### AdamW

Adam with decoupled weight decay:

**Original Adam** (with L2 regularization):

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} (\hat{m}_t + \lambda \theta_t)$$

**AdamW** (decoupled):

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t - \eta \lambda \theta_t$$

**Benefits**:
- Proper weight decay (not adaptive)
- Better generalization
- More stable training
- Preferred over Adam in many cases

## LAMB and Other Advanced Optimizers

### LAMB

Layer-wise Adaptive Moments for Batch training:

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$$

$$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

$$r_t = \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

$$\theta_{t+1} = \theta_t - \eta \frac{\phi(||\theta_t||)}{||r_t + \lambda \theta_t||} (r_t + \lambda \theta_t)$$

where $\phi$ is a function (typically identity or min).

**Properties**:
- Layer-wise adaptive learning rates
- Enables large batch training
- Used in BERT and other large models

### Lookahead

Wrapper optimizer that maintains slow and fast weights:

1. **Inner Loop**: Update fast weights with base optimizer
2. **Outer Loop**: Update slow weights: $\theta_{\text{slow}} \leftarrow \theta_{\text{slow}} + \alpha(\theta_{\text{fast}} - \theta_{\text{slow}})$

**Benefits**:
- More stable training
- Better generalization
- Works with any base optimizer

### RAdam

Rectified Adam addresses variance issues in early training:

- Uses adaptive learning rate only after sufficient samples
- Before that, uses simpler update
- More stable than Adam in early training

## Learning Rate Schedules

Learning rate schedules adjust learning rate during training.

### Constant

$$\eta_t = \eta_0$$

Simple but may not be optimal.

### Step Decay

$$\eta_t = \eta_0 \times \gamma^{\lfloor t/s \rfloor}$$

where $s$ is step size and $\gamma$ is decay factor.

### Exponential Decay

$$\eta_t = \eta_0 \times \gamma^t$$

Continuous decay.

### Polynomial Decay

$$\eta_t = \eta_0 \times (1 - t/T_{\max})^p$$

where $T_{\max}$ is max iterations and $p$ is power.

### Cosine Annealing

$$\eta_t = \eta_{\min} + (\eta_{\max} - \eta_{\min}) \times \frac{1 + \cos(\pi t/T)}{2}$$

Smooth decay following cosine curve.

### Warmup

Gradually increase learning rate:

$$\eta_t = \eta_{\max} \times \min(1, t/T_{\text{warmup}})$$

Helps with training stability.

### One-Cycle Policy

- Warmup to high learning rate
- Cosine decay
- Often achieves good results quickly

## Key Takeaways

1. **Gradient Descent**: Basic optimization method that updates parameters in direction of negative gradient, with learning rate controlling step size.

2. **SGD**: Stochastic gradient descent uses mini-batches for faster updates and better generalization, with noise helping escape local minima.

3. **Momentum**: Accumulates gradient history to smooth updates and accelerate convergence in consistent directions, reducing oscillations.

4. **Nesterov Momentum**: Looks ahead before computing gradient, providing better convergence rates and less overshooting than standard momentum.

5. **Adaptive Methods**: AdaGrad, RMSprop, and AdaDelta adapt learning rates per parameter based on gradient history, improving convergence for sparse or non-stationary gradients.

6. **Adam**: Combines momentum and adaptive learning rates with bias correction, providing fast convergence and robustness, making it a popular default choice.

7. **AdamW**: Decouples weight decay from adaptive learning rate, providing better generalization and more stable training than Adam.

8. **Advanced Optimizers**: LAMB enables large batch training, Lookahead provides stability, and RAdam addresses early training variance.

9. **Learning Rate Schedules**: Step decay, cosine annealing, and warmup strategies adjust learning rate during training to improve convergence and final performance.

10. **Practical Considerations**: Choice of optimizer depends on problem characteristics, with Adam/AdamW being good defaults, and learning rate schedules crucial for optimal performance.
