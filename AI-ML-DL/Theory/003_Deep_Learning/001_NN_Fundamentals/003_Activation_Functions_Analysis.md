# Activation Functions Analysis

## Table of Contents

1. [Introduction](#introduction)
2. [Role of Activation Functions](#role-of-activation-functions)
3. [Sigmoid and Tanh](#sigmoid-and-tanh)
4. [Rectified Linear Unit (ReLU)](#rectified-linear-unit-relu)
5. [Leaky ReLU and Variants](#leaky-relu-and-variants)
6. [Swish and GELU](#swish-and-gelu)
7. [Mish and Other Modern Activations](#mish-and-other-modern-activations)
8. [Activation Function Selection](#activation-function-selection)
9. [Dying ReLU Problem](#dying-relu-problem)
10. [Key Takeaways](#key-takeaways)

## Introduction

Activation functions are crucial components of neural networks that introduce non-linearity, enabling networks to learn complex patterns and approximate non-linear functions. Without activation functions, even deep networks would collapse to linear transformations, losing their expressive power. The choice of activation function significantly impacts training dynamics, gradient flow, and model performance.

This chapter provides a comprehensive analysis of activation functions used in modern deep learning, examining their mathematical properties, derivatives, advantages, disadvantages, and practical considerations for selection and implementation.

## Role of Activation Functions

Activation functions transform the weighted sum of inputs (pre-activation) into an output signal (activation) that serves as input to the next layer.

### Mathematical Formulation

For a neuron with input $\mathbf{x}$, weights $\mathbf{w}$, and bias $b$:

$$z = \mathbf{w}^T \mathbf{x} + b$$

$$a = \phi(z)$$

where $\phi$ is the activation function.

### Why Non-Linearity is Essential

Without activation functions, a multi-layer network would be equivalent to a single linear layer:

$$W_2(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = W_2 W_1 \mathbf{x} + W_2 \mathbf{b}_1 + \mathbf{b}_2 = W' \mathbf{x} + \mathbf{b}'$$

Non-linear activations enable:
- Learning complex decision boundaries
- Approximating non-linear functions
- Creating hierarchical feature representations

### Desirable Properties

1. **Non-Linearity**: Enables learning complex patterns
2. **Differentiability**: Required for gradient-based optimization (almost everywhere)
3. **Monotonicity**: Often desirable for stable training
4. **Bounded Output**: Can prevent activation explosion
5. **Zero-Centered**: Helps with gradient flow (tanh better than sigmoid)
6. **Computational Efficiency**: Fast to compute and differentiate

## Sigmoid and Tanh

### Sigmoid Function

The sigmoid (logistic) function is defined as:

$$\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}$$

**Properties**:
- Range: $(0, 1)$
- Smooth and differentiable
- Monotonic increasing
- S-shaped curve

**Derivative**:

$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

The derivative is maximum at $z = 0$ (value $0.25$) and approaches zero as $|z|$ increases.

**Advantages**:
- Smooth gradient
- Bounded output
- Interpretable as probability

**Disadvantages**:
- **Saturation**: Gradients vanish for large $|z|$
- **Not Zero-Centered**: Output always positive, causing zigzag updates
- **Slow Convergence**: Due to vanishing gradients
- **Computational Cost**: Exponential function is expensive

### Hyperbolic Tangent (Tanh)

The tanh function is defined as:

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = \frac{\sinh(z)}{\cosh(z)} = 2\sigma(2z) - 1$$

**Properties**:
- Range: $(-1, 1)$
- Zero-centered
- Smooth and differentiable
- Monotonic increasing

**Derivative**:

$$\tanh'(z) = 1 - \tanh^2(z) = \text{sech}^2(z)$$

The derivative ranges from 0 to 1, with maximum at $z = 0$.

**Advantages**:
- Zero-centered output (better than sigmoid)
- Stronger gradients than sigmoid (derivative can reach 1)
- Bounded output

**Disadvantages**:
- Still suffers from saturation
- Vanishing gradients for large $|z|$
- More computationally expensive than ReLU

### Comparison

| Property | Sigmoid | Tanh |
|----------|---------|------|
| Range | $(0, 1)$ | $(-1, 1)$ |
| Zero-Centered | No | Yes |
| Max Derivative | $0.25$ | $1.0$ |
| Saturation | Yes | Yes |
| Common Use | Output layer (binary) | Hidden layers (historical) |

## Rectified Linear Unit (ReLU)

The Rectified Linear Unit (ReLU) is the most widely used activation function in modern deep learning:

$$\text{ReLU}(z) = \max(0, z) = \begin{cases}
z & \text{if } z > 0 \\
0 & \text{if } z \leq 0
\end{cases}$$

### Properties

- **Range**: $[0, \infty)$
- **Non-linear**: Introduces non-linearity despite piecewise linear form
- **Computationally Efficient**: Simple max operation
- **Sparse Activations**: Produces zeros for negative inputs

### Derivative

$$\text{ReLU}'(z) = \begin{cases}
1 & \text{if } z > 0 \\
0 & \text{if } z < 0
\end{cases}$$

The derivative is undefined at $z = 0$, but in practice, $0$ or $1$ is used (subgradient).

### Advantages

1. **No Saturation**: Gradient is constant (1) for positive inputs
2. **Computational Efficiency**: Fast forward and backward passes
3. **Sparsity**: Naturally produces sparse representations
4. **Faster Convergence**: Compared to sigmoid/tanh
5. **Biological Plausibility**: Similar to neuron firing threshold

### Disadvantages

1. **Not Zero-Centered**: Output always non-negative
2. **Dying ReLU Problem**: Neurons can become permanently inactive
3. **Unbounded**: Output can grow very large
4. **Non-Differentiable at Zero**: Though subgradient is used

### Dying ReLU Problem

If a neuron's weights are updated such that it always outputs negative values, the gradient becomes zero and the neuron "dies":

- Pre-activation $z < 0$ → Activation $a = 0$
- Gradient $\frac{\partial \mathcal{L}}{\partial z} = 0$ (if using standard backprop)
- No weight updates → Neuron remains inactive

This is particularly problematic with:
- High learning rates
- Poor initialization
- Large negative biases

## Leaky ReLU and Variants

### Leaky ReLU

Addresses the dying ReLU problem by allowing small negative gradients:

$$\text{LeakyReLU}(z) = \max(\alpha z, z) = \begin{cases}
z & \text{if } z > 0 \\
\alpha z & \text{if } z \leq 0
\end{cases}$$

where $\alpha$ is a small positive constant (typically $0.01$).

**Derivative**:

$$\text{LeakyReLU}'(z) = \begin{cases}
1 & \text{if } z > 0 \\
\alpha & \text{if } z \leq 0
\end{cases}$$

**Advantages**:
- Prevents dying ReLU problem
- Maintains computational efficiency
- Allows gradient flow for negative inputs

**Disadvantages**:
- Requires tuning $\alpha$ (though $0.01$ works well)
- Still not zero-centered

### Parametric ReLU (PReLU)

Makes the negative slope learnable:

$$\text{PReLU}(z) = \max(\alpha z, z)$$

where $\alpha$ is a learnable parameter initialized to $0.25$ or $0.01$.

**Advantages**:
- Adapts to data
- Can learn optimal negative slope

**Disadvantages**:
- Additional parameter per neuron
- Requires more memory and computation

### Randomized Leaky ReLU (RReLU)

Uses a random $\alpha$ during training (sampled from uniform distribution), fixed during testing:

$$\alpha \sim U(l, u)$$

where typically $l = \frac{1}{8}$ and $u = \frac{1}{3}$.

**Advantages**:
- Regularization effect
- Reduces overfitting

### Exponential Linear Unit (ELU)

$$\text{ELU}(z) = \begin{cases}
z & \text{if } z > 0 \\
\alpha(e^z - 1) & \text{if } z \leq 0
\end{cases}$$

where $\alpha$ is typically $1.0$.

**Properties**:
- Smooth for negative values
- Approaches $-\alpha$ as $z \to -\infty$
- Zero-centered (approximately)

**Derivative**:

$$\text{ELU}'(z) = \begin{cases}
1 & \text{if } z > 0 \\
\text{ELU}(z) + \alpha & \text{if } z \leq 0
\end{cases}$$

**Advantages**:
- No dying ReLU problem
- Smooth and differentiable everywhere
- Negative outputs can help with zero-centered property

**Disadvantages**:
- More computationally expensive (exponential)
- Requires more memory

### Scaled Exponential Linear Unit (SELU)

Self-normalizing activation function:

$$\text{SELU}(z) = \lambda \begin{cases}
z & \text{if } z > 0 \\
\alpha(e^z - 1) & \text{if } z \leq 0
\end{cases}$$

where $\lambda \approx 1.0507$ and $\alpha \approx 1.6733$.

**Properties**:
- Enables self-normalizing networks
- Maintains mean and variance through layers
- Requires specific initialization (LeCun normal)

## Swish and GELU

### Swish

Discovered through automated search:

$$\text{Swish}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$

**Properties**:
- Smooth and non-monotonic
- Bounded below, unbounded above
- Self-gated (input gates itself)

**Derivative**:

$$\text{Swish}'(z) = \sigma(z) + z \cdot \sigma(z)(1 - \sigma(z)) = \sigma(z)(1 + z(1 - \sigma(z)))$$

**Advantages**:
- Often outperforms ReLU
- Smooth gradients
- Better for deeper networks

**Disadvantages**:
- More computationally expensive
- Requires sigmoid computation

### Gaussian Error Linear Unit (GELU)

$$\text{GELU}(z) = z \Phi(z)$$

where $\Phi(z)$ is the cumulative distribution function of the standard normal distribution:

$$\Phi(z) = \frac{1}{2}\left(1 + \text{erf}\left(\frac{z}{\sqrt{2}}\right)\right)$$

**Approximation**:

$$\text{GELU}(z) \approx 0.5z\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(z + 0.044715z^3)\right)\right)$$

**Properties**:
- Smooth activation
- Used in BERT and GPT models
- Probabilistic interpretation

**Advantages**:
- Smooth and differentiable
- Good empirical performance
- Used in state-of-the-art transformers

**Disadvantages**:
- More expensive than ReLU
- Requires approximation for efficiency

## Mish and Other Modern Activations

### Mish

$$\text{Mish}(z) = z \cdot \tanh(\text{softplus}(z)) = z \cdot \tanh(\ln(1 + e^z))$$

**Properties**:
- Smooth, non-monotonic
- Self-regularized
- Unbounded above, bounded below (approximately $-0.31$)

**Derivative**:

$$\text{Mish}'(z) = \frac{e^z(4(z+1) + 4e^{2z} + e^{3z} + e^z(4z+6))}{(2e^z + e^{2z} + 2)^2}$$

**Advantages**:
- Often outperforms Swish
- Smooth gradients
- Better accuracy in many tasks

**Disadvantages**:
- Computationally expensive
- Complex derivative

### Hard Swish

Hardware-friendly approximation of Swish:

$$\text{HardSwish}(z) = z \cdot \frac{\text{ReLU6}(z+3)}{6}$$

where $\text{ReLU6}(z) = \min(\max(0, z), 6)$.

**Advantages**:
- Efficient on mobile devices
- No exponential operations
- Good approximation of Swish

### Hard Sigmoid

Piecewise linear approximation of sigmoid:

$$\text{HardSigmoid}(z) = \max(0, \min(1, \frac{z+3}{6}))$$

**Advantages**:
- Very efficient
- Quantization-friendly

**Disadvantages**:
- Less smooth than sigmoid
- Limited expressiveness

## Activation Function Selection

### Guidelines by Layer Type

**Input Layer**:
- Usually no activation (or normalization)
- Raw features passed through

**Hidden Layers**:
- **ReLU**: Default choice for most CNNs and feedforward networks
- **Leaky ReLU/PReLU**: When dying ReLU is a concern
- **Swish/GELU**: For transformers and modern architectures
- **Tanh**: Rarely used in modern networks

**Output Layer**:
- **Sigmoid**: Binary classification
- **Softmax**: Multi-class classification
- **Linear**: Regression
- **Tanh**: Bounded regression

### Task-Specific Recommendations

**Computer Vision (CNNs)**:
- ReLU or Leaky ReLU for hidden layers
- Softmax for classification output

**Natural Language Processing**:
- GELU or Swish for transformers
- Softmax for language modeling

**Recurrent Networks**:
- Tanh or ReLU for hidden states
- Sigmoid for gates (LSTM, GRU)

**Generative Models**:
- ReLU/Leaky ReLU for generators
- Sigmoid for discriminator output

### Empirical Comparison

| Activation | Convergence | Accuracy | Efficiency | Use Case |
|------------|-------------|----------|------------|----------|
| ReLU | Fast | Good | High | General purpose |
| Leaky ReLU | Fast | Good | High | When ReLU dies |
| Swish | Medium | Excellent | Medium | Deep networks |
| GELU | Medium | Excellent | Medium | Transformers |
| Mish | Slow | Excellent | Low | Research |
| Tanh | Slow | Moderate | Medium | RNNs (historical) |
| Sigmoid | Slow | Moderate | Medium | Output only |

## Dying ReLU Problem

The dying ReLU problem occurs when neurons become permanently inactive, outputting zero for all inputs.

### Causes

1. **Large Negative Bias**: Initialization or updates push pre-activations negative
2. **High Learning Rate**: Large updates can push weights into negative region
3. **Poor Initialization**: Weights initialized too small or incorrectly
4. **Gradient Flow**: Once gradient is zero, no recovery mechanism

### Mathematical Analysis

For a ReLU neuron to die:
- Pre-activation: $z = \mathbf{w}^T \mathbf{x} + b < 0$ for all training examples
- Activation: $a = 0$
- Gradient: $\frac{\partial \mathcal{L}}{\partial z} = 0$ (if no error propagates)
- Weight update: $\Delta \mathbf{w} = 0$ → No recovery

### Solutions

1. **Leaky ReLU**: Allows small negative gradients
2. **Proper Initialization**: He initialization for ReLU networks
3. **Lower Learning Rate**: Prevents large updates
4. **Batch Normalization**: Normalizes inputs, reducing dead neuron probability
5. **Gradient Clipping**: Prevents extreme updates
6. **Monitoring**: Track percentage of dead neurons

### Detection

```python
def count_dead_neurons(activations, threshold=1e-6):
    """Count neurons that are always inactive."""
    dead_count = 0
    for layer_activations in activations:
        # Check if any neuron is always zero
        neuron_sums = np.sum(np.abs(layer_activations), axis=0)
        dead_count += np.sum(neuron_sums < threshold)
    return dead_count
```

### Prevention Strategies

1. **He Initialization**: 
   $$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)$$

2. **Small Initial Bias**: Initialize biases to small positive values

3. **Learning Rate Scheduling**: Start with smaller learning rate

4. **Regularization**: L2 regularization can prevent extreme weights

## Key Takeaways

1. **Essential Non-Linearity**: Activation functions enable neural networks to learn complex, non-linear patterns that linear transformations cannot capture.

2. **ReLU Dominance**: ReLU and its variants (Leaky ReLU, PReLU) are the default choice for most modern architectures due to computational efficiency and good performance.

3. **Sigmoid/Tanh Limitations**: While historically important, sigmoid and tanh suffer from vanishing gradients and are rarely used in hidden layers today.

4. **Modern Activations**: Swish, GELU, and Mish offer improved performance in certain contexts (especially transformers) but at higher computational cost.

5. **Dying ReLU Problem**: A significant issue with ReLU that can be mitigated through Leaky ReLU, proper initialization, or other techniques.

6. **Layer-Specific Selection**: Different layers benefit from different activations—ReLU for hidden layers, sigmoid/softmax for output layers.

7. **Task-Dependent Choice**: Activation selection should consider the specific task, architecture, and computational constraints.

8. **Gradient Flow**: Activation functions significantly impact gradient flow during backpropagation, affecting training dynamics and convergence.

9. **Computational Trade-offs**: More sophisticated activations (Swish, Mish) may improve accuracy but increase computational cost—important for deployment.

10. **Empirical Validation**: While theoretical properties matter, empirical performance on validation data should guide final activation selection.
