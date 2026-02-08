# Weight Initialization Strategies

## Table of Contents

1. [Introduction](#introduction)
2. [The Importance of Initialization](#the-importance-of-initialization)
3. [Random Initialization Basics](#random-initialization-basics)
4. [Xavier/Glorot Initialization](#xavierglorot-initialization)
5. [He/Kaiming Initialization](#hekaiming-initialization)
6. [Orthogonal Initialization](#orthogonal-initialization)
7. [Layer-Sequential Unit-Variance (LSUV)](#layer-sequential-unit-variance-lsuv)
8. [Initialization for Different Architectures](#initialization-for-different-architectures)
9. [Impact on Training Dynamics](#impact-on-training-dynamics)
10. [Key Takeaways](#key-takeaways)

## Introduction

Weight initialization is a critical but often overlooked aspect of neural network training. Poor initialization can lead to vanishing or exploding gradients, slow convergence, or complete training failure. Good initialization sets the network in a favorable region of the loss landscape, enabling effective gradient-based optimization.

This chapter explores the theoretical foundations and practical strategies for weight initialization, from basic random initialization to sophisticated methods like Xavier, He, and orthogonal initialization, examining how they affect training dynamics and model performance.

## The Importance of Initialization

### Why Initialization Matters

Initial weights determine:
1. **Starting Point**: Where optimization begins in parameter space
2. **Gradient Magnitudes**: Affects gradient flow through the network
3. **Activation Statistics**: Influences activation distributions and saturation
4. **Symmetry Breaking**: Ensures neurons learn different features

### Problems with Poor Initialization

**Too Small Weights**:
- Activations become too small
- Gradients vanish (especially with sigmoid/tanh)
- Slow learning or no learning

**Too Large Weights**:
- Activations saturate (sigmoid/tanh) or explode (ReLU)
- Gradients explode or vanish
- Training instability

**Zero Initialization**:
- Symmetry problem: All neurons compute the same function
- No gradient diversity
- Network cannot learn

**Same Initialization**:
- Symmetry: Neurons remain identical
- Reduces effective capacity
- Poor feature diversity

### Desirable Properties

Good initialization should:
1. **Break Symmetry**: Different neurons start with different weights
2. **Preserve Variance**: Maintain reasonable activation variance across layers
3. **Enable Gradient Flow**: Allow gradients to flow backward effectively
4. **Avoid Saturation**: Keep activations in active region of activation functions
5. **Scale Appropriately**: Account for layer width and activation function

## Random Initialization Basics

### Uniform Random Initialization

Simple uniform distribution:

$$W_{ij} \sim \mathcal{U}(-a, a)$$

where $a$ is typically a small constant (e.g., $0.1$).

**Problems**:
- No theoretical justification
- Doesn't scale with layer size
- Often too small or too large

### Gaussian Random Initialization

$$W_{ij} \sim \mathcal{N}(0, \sigma^2)$$

where $\sigma$ is the standard deviation.

**Naive Approach**:
- Fixed $\sigma$ (e.g., $0.01$) regardless of layer size
- Doesn't account for network architecture
- Can cause vanishing or exploding activations

### Scaling Considerations

For a layer with $n_{\text{in}}$ inputs, if weights have variance $\sigma^2$, the variance of the output (assuming zero-mean inputs) is:

$$\text{Var}(z) = n_{\text{in}} \sigma^2 \text{Var}(x)$$

This shows that variance scales with the number of inputs, requiring careful weight scaling.

## Xavier/Glorot Initialization

Xavier initialization (also called Glorot initialization) was proposed to maintain activation and gradient variances across layers.

### Theoretical Foundation

For a linear layer $z = W\mathbf{x} + b$ with:
- Inputs: $\mathbf{x} \in \mathbb{R}^{n_{\text{in}}}$ with variance $\text{Var}(x_i) = 1$
- Weights: $W_{ij} \sim \mathcal{N}(0, \sigma^2)$
- Output: $z_j = \sum_{i=1}^{n_{\text{in}}} W_{ji} x_i$

Assuming independence:

$$\text{Var}(z_j) = n_{\text{in}} \sigma^2$$

To maintain variance, we want $\text{Var}(z_j) = 1$, so:

$$\sigma^2 = \frac{1}{n_{\text{in}}}$$

### Forward Pass Variance

For forward propagation, maintaining variance requires:

$$\sigma^2 = \frac{1}{n_{\text{in}}}$$

### Backward Pass Variance

For backpropagation, gradients flow backward. If $\delta^{(l)}$ is the error signal at layer $l$, and we want $\text{Var}(\delta^{(l)}) = \text{Var}(\delta^{(l+1)})$, we need:

$$\sigma^2 = \frac{1}{n_{\text{out}}}$$

where $n_{\text{out}}$ is the number of outputs.

### Compromise: Average

Xavier initialization uses the average:

$$\sigma^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}$$

### Xavier Normal Initialization

$$W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$$

### Xavier Uniform Initialization

$$W_{ij} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)$$

The uniform version uses range $[-a, a]$ where variance is $\frac{a^2}{3}$, so:

$$\frac{a^2}{3} = \frac{2}{n_{\text{in}} + n_{\text{out}}} \implies a = \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}$$

### Assumptions

Xavier initialization assumes:
1. Linear activations (or activations near linear region)
2. Zero-mean inputs
3. Symmetric activation function (tanh, symmetric sigmoid)
4. Weights and inputs are independent

### Limitations

- **ReLU Networks**: Doesn't account for ReLU's zeroing of half the activations
- **Deep Networks**: May not work well for very deep networks
- **Non-Linear Activations**: Assumes activations don't change variance significantly

### Implementation

```python
import numpy as np

def xavier_normal(shape):
    """Xavier normal initialization."""
    n_in, n_out = shape[0], shape[1]
    std = np.sqrt(2.0 / (n_in + n_out))
    return np.random.normal(0, std, shape)

def xavier_uniform(shape):
    """Xavier uniform initialization."""
    n_in, n_out = shape[0], shape[1]
    limit = np.sqrt(6.0 / (n_in + n_out))
    return np.random.uniform(-limit, limit, shape)
```

## He/Kaiming Initialization

He initialization (also called Kaiming initialization) was designed specifically for ReLU networks, accounting for ReLU's effect on variance.

### ReLU Variance Analysis

For ReLU activation $\phi(z) = \max(0, z)$:

If $z \sim \mathcal{N}(0, \sigma^2)$, then:

$$\text{Var}(\phi(z)) = \frac{1}{2} \text{Var}(z)$$

This is because ReLU zeros out half the distribution, halving the variance.

### Forward Pass

To maintain variance through forward pass with ReLU:

$$\text{Var}(z^{(l)}) = n_{\text{in}} \sigma^2 \text{Var}(a^{(l-1)})$$

With ReLU: $\text{Var}(a^{(l)}) = \frac{1}{2} \text{Var}(z^{(l)})$

To maintain $\text{Var}(a^{(l)}) = \text{Var}(a^{(l-1)})$:

$$\text{Var}(a^{(l-1)}) = \frac{1}{2} n_{\text{in}} \sigma^2 \text{Var}(a^{(l-1)})$$

This requires:

$$\sigma^2 = \frac{2}{n_{\text{in}}}$$

### Backward Pass

For backward pass with ReLU, similar analysis gives:

$$\sigma^2 = \frac{2}{n_{\text{out}}}$$

### He Normal Initialization

$$W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)$$

### He Uniform Initialization

$$W_{ij} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}}}}, \sqrt{\frac{6}{n_{\text{in}}}}\right)$$

### Leaky ReLU Variant

For Leaky ReLU with negative slope $\alpha$:

$$\text{Var}(\phi(z)) = \frac{1 + \alpha^2}{2} \text{Var}(z)$$

He initialization becomes:

$$\sigma^2 = \frac{2}{(1 + \alpha^2) n_{\text{in}}}$$

### Advantages

- **ReLU-Optimized**: Specifically designed for ReLU networks
- **Better for Deep Networks**: Works well even in very deep networks
- **Empirically Superior**: Often outperforms Xavier for ReLU networks

### Implementation

```python
def he_normal(shape):
    """He normal initialization for ReLU."""
    n_in = shape[0]
    std = np.sqrt(2.0 / n_in)
    return np.random.normal(0, std, shape)

def he_uniform(shape):
    """He uniform initialization for ReLU."""
    n_in = shape[0]
    limit = np.sqrt(6.0 / n_in)
    return np.random.uniform(-limit, limit, shape)

def he_normal_leaky_relu(shape, alpha=0.01):
    """He initialization for Leaky ReLU."""
    n_in = shape[0]
    std = np.sqrt(2.0 / ((1 + alpha**2) * n_in))
    return np.random.normal(0, std, shape)
```

## Orthogonal Initialization

Orthogonal initialization initializes weight matrices as orthogonal matrices, preserving norms and enabling better gradient flow.

### Definition

A matrix $W$ is orthogonal if $W^T W = I$ (or $W W^T = I$ for non-square matrices).

### Properties

- **Norm Preservation**: $||W\mathbf{x}||_2 = ||\mathbf{x}||_2$ for orthogonal $W$
- **Condition Number**: Condition number equals 1 (optimal)
- **Gradient Flow**: Better gradient flow through orthogonal transformations

### Initialization Method

1. Generate random matrix $A$ with appropriate shape
2. Compute QR decomposition: $A = QR$ where $Q$ is orthogonal
3. Use $Q$ (or scaled $Q$) as weight matrix

For non-square matrices, use:
- $W \in \mathbb{R}^{m \times n}$ with $m \geq n$: $W = Q$ from $QR$ decomposition
- $W \in \mathbb{R}^{m \times n}$ with $m < n$: $W = Q^T$ from $QR$ decomposition of $A^T$

### Scaling

Orthogonal matrices preserve norms, but we may want to scale:

$$W = \gamma Q$$

where $\gamma$ is a scaling factor. Common choices:
- $\gamma = 1$: Preserves input norm exactly
- $\gamma = \sqrt{2/n_{\text{in}}}$: Similar to He initialization scale

### Advantages

- **Stable Gradients**: Excellent gradient flow properties
- **No Vanishing/Exploding**: Maintains gradient norms
- **Theoretical Guarantees**: Strong theoretical properties

### Disadvantages

- **Computational Cost**: QR decomposition is more expensive
- **Not Always Better**: May not outperform He/Xavier in practice
- **Limited to Square Layers**: Less natural for non-square weight matrices

### Implementation

```python
from scipy.linalg import qr

def orthogonal_init(shape, gain=1.0):
    """Orthogonal initialization."""
    if len(shape) < 2:
        raise ValueError("Orthogonal init requires at least 2D tensor")
    
    # Flatten all but last dimension
    flat_shape = (np.prod(shape[:-1]), shape[-1])
    
    # Generate random matrix
    a = np.random.normal(0, 1, flat_shape)
    
    # QR decomposition
    q, r = qr(a, mode='economic')
    
    # Make Q have positive diagonal (for uniqueness)
    q *= np.sign(np.diag(r))
    
    # Reshape and scale
    w = q.reshape(shape) * gain
    
    return w
```

## Layer-Sequential Unit-Variance (LSUV)

LSUV is a data-dependent initialization method that sets weights to achieve unit variance activations.

### Algorithm

1. Initialize weights with small random values (e.g., orthonormal)
2. Forward pass a small batch through the network
3. For each layer:
   - Measure variance of pre-activations
   - Scale weights to achieve unit variance: $W \leftarrow W / \sqrt{\text{Var}(z)}$
4. Repeat until convergence or maximum iterations

### Advantages

- **Data-Dependent**: Adapts to actual data distribution
- **Unit Variance**: Ensures activations have desired variance
- **Works with Any Architecture**: Not limited to specific activation functions

### Disadvantages

- **Requires Data**: Needs a forward pass with real data
- **Computational Cost**: Additional forward passes during initialization
- **May Overfit**: Could overfit to initialization batch

### Implementation

```python
def lsuv_init(model, data_loader, target_var=1.0, tol=0.1, max_iter=10):
    """LSUV initialization."""
    # Small random initialization
    for param in model.parameters():
        if len(param.shape) >= 2:
            nn.init.orthogonal_(param, gain=0.1)
    
    # Get a batch
    batch = next(iter(data_loader))
    x = batch[0]
    
    # Forward pass and adjust each layer
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Forward through this layer
                x = module(x)
                
                # Measure variance
                var = x.var().item()
                
                # Adjust weights
                if abs(var - target_var) > tol:
                    scale = np.sqrt(target_var / var)
                    module.weight.data *= scale
                    
                    # Re-forward to check
                    x = module(x)
                    var = x.var().item()
    
    return model
```

## Initialization for Different Architectures

### Convolutional Networks

For convolutional layers with kernel size $k \times k$, input channels $C_{\text{in}}$, output channels $C_{\text{out}}$:

**Xavier**:
$$\sigma^2 = \frac{2}{k^2 C_{\text{in}} + k^2 C_{\text{out}}}$$

**He**:
$$\sigma^2 = \frac{2}{k^2 C_{\text{in}}}$$

### Recurrent Networks

**LSTM/GRU**: Often use orthogonal initialization for recurrent weights to prevent vanishing gradients.

**Small Random**: Input-to-hidden weights often use small random initialization.

### Transformer Networks

**Xavier/He**: Standard for feedforward layers.

**Small Values**: Attention weights often initialized to small values.

**Positional Embeddings**: Often use learned embeddings or sinusoidal initialization.

### Residual Networks

**He Initialization**: Standard for ResNet blocks.

**Zero Initialization**: Final layer of residual block sometimes initialized to zero to start as identity.

### Batch Normalization Networks

With batch normalization, initialization is less critical as BN normalizes activations. However, He/Xavier still commonly used.

## Impact on Training Dynamics

### Convergence Speed

Good initialization:
- Starts in favorable region
- Faster convergence
- Fewer iterations to reach good solution

Poor initialization:
- May require many iterations to escape poor region
- Slower convergence
- May never converge

### Gradient Flow

**Well-Initialized Networks**:
- Gradients flow smoothly
- No vanishing or exploding gradients
- Stable training

**Poorly Initialized Networks**:
- Gradients may vanish or explode
- Unstable training
- May require gradient clipping

### Activation Statistics

Good initialization maintains:
- Reasonable activation magnitudes
- Avoids saturation
- Preserves information flow

### Generalization

Initialization can affect:
- Final solution quality
- Generalization performance
- Robustness to hyperparameters

### Empirical Observations

- **He > Xavier** for ReLU networks
- **Orthogonal** helpful for RNNs
- **LSUV** can improve convergence
- **Too small** worse than too large (for ReLU)

## Key Takeaways

1. **Critical for Training**: Poor initialization can prevent networks from learning, while good initialization enables effective training.

2. **Xavier/Glorot**: Designed for tanh/sigmoid networks, maintains variance assuming linear activations. Uses $\sigma^2 = 2/(n_{\text{in}} + n_{\text{out}})$.

3. **He/Kaiming**: Specifically designed for ReLU networks, accounts for ReLU's variance reduction. Uses $\sigma^2 = 2/n_{\text{in}}$.

4. **Orthogonal Initialization**: Preserves norms and enables excellent gradient flow, particularly useful for RNNs and deep networks.

5. **LSUV**: Data-dependent method that achieves unit variance activations through iterative adjustment, works with any architecture.

6. **Architecture-Specific**: Different architectures benefit from different initialization strategies (CNNs, RNNs, Transformers).

7. **Variance Preservation**: Good initialization maintains reasonable activation and gradient variances across layers.

8. **Symmetry Breaking**: Random initialization ensures neurons learn different features, breaking symmetry.

9. **Training Dynamics**: Initialization significantly affects convergence speed, gradient flow, and final model quality.

10. **Empirical Validation**: While theory guides selection, empirical performance should validate initialization choices for specific architectures and datasets.
