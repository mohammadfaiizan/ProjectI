# Batch Normalization and Layer Normalization

## Table of Contents

1. [Introduction](#introduction)
2. [Internal Covariate Shift](#internal-covariate-shift)
3. [Batch Normalization](#batch-normalization)
4. [Layer Normalization](#layer-normalization)
5. [Group Normalization](#group-normalization)
6. [Instance Normalization](#instance-normalization)
7. [Normalization in Different Architectures](#normalization-in-different-architectures)
8. [Theoretical Analysis](#theoretical-analysis)
9. [Practical Considerations](#practical-considerations)
10. [Key Takeaways](#key-takeaways)

## Introduction

Normalization techniques stabilize and accelerate training by normalizing activations or inputs. Batch Normalization revolutionized deep learning by enabling training of deeper networks, while Layer Normalization and variants address limitations and extend applicability to different architectures.

This chapter covers normalization techniques, from Batch Normalization to Layer, Group, and Instance Normalization, examining their mechanisms, benefits, and applications.

## Internal Covariate Shift

### Definition

Internal covariate shift refers to the change in distribution of layer inputs during training:

- Parameters change → Input distributions shift
- Later layers must adapt to shifting distributions
- Slows down training

### Problem

As network trains:
- Earlier layer parameters update
- Their output distributions change
- Later layers see different input distributions
- Must continuously adapt

### Impact

- Slower convergence
- Requires smaller learning rates
- Harder to train deep networks
- Sensitive to initialization

## Batch Normalization

Batch Normalization (BatchNorm) normalizes activations across the batch dimension.

### Algorithm

**Training**:

For each feature dimension:

$$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$$

$$y_i = \gamma \hat{x}_i + \beta$$

where:
- $\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} x_i$: Batch mean
- $\sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2$: Batch variance
- $\gamma, \beta$: Learnable scale and shift
- $\epsilon$: Small constant for numerical stability

**Inference**:

Use running statistics:

$$\mu_{\text{running}} = \text{momentum} \times \mu_{\text{running}} + (1-\text{momentum}) \times \mu_{\mathcal{B}}$$

$$\sigma_{\text{running}}^2 = \text{momentum} \times \sigma_{\text{running}}^2 + (1-\text{momentum}) \times \sigma_{\mathcal{B}}^2$$

$$y_i = \gamma \frac{x_i - \mu_{\text{running}}}{\sqrt{\sigma_{\text{running}}^2 + \epsilon}} + \beta$$

### Properties

1. **Reduces Internal Covariate Shift**: Normalizes inputs to each layer
2. **Enables Higher Learning Rates**: More stable gradients
3. **Regularization Effect**: Noise from batch statistics
4. **Less Sensitive to Initialization**: Normalized inputs

### Benefits

- Faster convergence
- Deeper networks trainable
- Higher learning rates possible
- Better generalization (regularization)
- Less sensitive to initialization

### Placement

Common placements:
- **After Convolution/Linear**: Before activation
- **Before Activation**: Most common
- **After Activation**: Less common

**Recommended**: Before activation (ReLU)

### Learnable Parameters

- **$\gamma$ (scale)**: Allows network to learn identity if beneficial
- **$\beta$ (shift)**: Allows network to learn optimal mean

If normalization hurts, network can learn $\gamma = \sigma_{\mathcal{B}}$ and $\beta = \mu_{\mathcal{B}}$ to recover original.

### Implementation

```python
class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x):
        if self.training:
            # Compute batch statistics
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            var = x.var(dim=[0, 2, 3], keepdim=True, unbiased=False)
            
            # Normalize
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
            
            # Update running statistics
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + \
                                   self.momentum * mean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + \
                                  self.momentum * var.squeeze()
        else:
            # Use running statistics
            mean = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        return self.gamma.view(1, -1, 1, 1) * x_norm + self.beta.view(1, -1, 1, 1)
```

### Limitations

1. **Batch Size Dependency**: Small batches have noisy statistics
2. **Batch Statistics**: Different behavior at train vs. test
3. **Sequence Models**: Doesn't work well with variable-length sequences
4. **Online Learning**: Requires batches

## Layer Normalization

Layer Normalization normalizes across features for each example.

### Algorithm

For input $\mathbf{x} \in \mathbb{R}^d$:

$$\mu = \frac{1}{d} \sum_{i=1}^{d} x_i$$

$$\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$$

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

$$y_i = \gamma \hat{x}_i + \beta$$

### Properties

- **Example-Wise**: Normalizes each example independently
- **No Batch Dependency**: Works with batch size 1
- **Sequence Models**: Natural fit for RNNs/Transformers
- **Consistent**: Same normalization at train and test

### Comparison with BatchNorm

| Property | BatchNorm | LayerNorm |
|----------|-----------|-----------|
| Normalization | Across batch | Across features |
| Batch Size | Requires batches | Works with 1 |
| RNNs | Difficult | Natural fit |
| CNNs | Common | Less common |

### Use Cases

- **RNNs**: Natural normalization for sequences
- **Transformers**: Standard in transformer blocks
- **Small Batches**: When batch size is small
- **Online Learning**: When batches aren't available

## Group Normalization

Group Normalization divides channels into groups and normalizes within groups.

### Algorithm

For feature map with $C$ channels:
1. Divide into $G$ groups (each with $C/G$ channels)
2. Normalize within each group
3. Normalize across spatial dimensions and group channels

### Formulation

For group $g$:

$$\mu_g = \frac{1}{|\mathcal{G}_g|} \sum_{i \in \mathcal{G}_g} x_i$$

$$\sigma_g^2 = \frac{1}{|\mathcal{G}_g|} \sum_{i \in \mathcal{G}_g} (x_i - \mu_g)^2$$

where $\mathcal{G}_g$ contains indices in group $g$.

### Properties

- **Group-Wise**: Normalizes within groups
- **No Batch Dependency**: Independent of batch size
- **CNNs**: Good alternative to BatchNorm in CNNs
- **Flexible**: Can adjust number of groups

### Typical Configuration

- **$G = 32$**: Common for ResNet
- **$G = C$**: Equivalent to Instance Normalization
- **$G = 1$**: Equivalent to Layer Normalization

## Instance Normalization

Instance Normalization normalizes each channel independently for each example.

### Algorithm

For each channel $c$ and example:

$$\mu_c = \frac{1}{HW} \sum_{h,w} x_{c,h,w}$$

$$\sigma_c^2 = \frac{1}{HW} \sum_{h,w} (x_{c,h,w} - \mu_c)^2$$

$$\hat{x}_{c,h,w} = \frac{x_{c,h,w} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}$$

### Properties

- **Channel-Wise**: Each channel normalized independently
- **Style Transfer**: Removes instance-specific contrast
- **Generative Models**: Common in GANs
- **No Batch Dependency**: Works with any batch size

### Use Cases

- **Style Transfer**: Removes style information
- **Image Generation**: Normalizes instance statistics
- **Domain Adaptation**: Reduces domain-specific statistics

## Normalization in Different Architectures

### Convolutional Networks

**BatchNorm**: Standard choice
- Normalizes across batch and spatial dimensions
- Per-channel normalization
- Works well with large batches

**GroupNorm**: Alternative
- When batch size is small
- More stable statistics
- Better for detection/segmentation

### Recurrent Networks

**LayerNorm**: Standard choice
- Normalizes across features
- Works with variable-length sequences
- Consistent at train/test

**BatchNorm**: Less common
- Can be applied but tricky
- Requires fixed-length sequences
- Less natural fit

### Transformers

**LayerNorm**: Standard
- Applied before attention and FFN
- Pre-norm or post-norm
- Essential for training

### Generative Models

**InstanceNorm**: Common
- Removes instance statistics
- Better for style transfer
- Used in GANs

## Theoretical Analysis

### Why Normalization Works

1. **Gradient Flow**: Normalized inputs have better gradient properties
2. **Smoother Loss Landscape**: Easier optimization
3. **Regularization**: Noise from batch statistics
4. **Initialization Independence**: Less sensitive to initialization

### Whitening vs. Standardization

- **Full Whitening**: Decorrelate features (expensive)
- **Standardization**: Normalize mean/variance (BatchNorm)
- **Simpler**: Often sufficient

### Gradient Analysis

Normalization affects gradients:

$$\frac{\partial \mathcal{L}}{\partial \gamma} = \sum_{i} \frac{\partial \mathcal{L}}{\partial y_i} \hat{x}_i$$

$$\frac{\partial \mathcal{L}}{\partial \beta} = \sum_{i} \frac{\partial \mathcal{L}}{\partial y_i}$$

Gradients flow through normalization, enabling learning of scale/shift.

## Practical Considerations

### When to Use BatchNorm

- Large batch sizes
- CNNs
- When faster convergence needed
- When regularization beneficial

### When to Use LayerNorm

- RNNs/Transformers
- Small batch sizes
- Sequence models
- When consistent train/test needed

### When to Use GroupNorm

- Small batch sizes in CNNs
- Detection/segmentation
- When BatchNorm statistics unreliable

### Hyperparameters

- **Momentum**: 0.1 typical for BatchNorm
- **Epsilon**: $10^{-5}$ typical
- **Groups**: 32 typical for GroupNorm
- **Placement**: Before activation recommended

### Common Pitfalls

1. **Batch Size**: BatchNorm needs reasonable batch size
2. **Inference Mode**: Remember to set eval mode
3. **Frozen BN**: Consider freezing in fine-tuning
4. **Synchronization**: Multi-GPU requires sync

## Key Takeaways

1. **Internal Covariate Shift**: The change in input distributions during training slows convergence and makes deep networks harder to train.

2. **Batch Normalization**: Normalizes activations across batch dimension, reducing internal covariate shift and enabling faster training of deeper networks.

3. **Layer Normalization**: Normalizes across features for each example, making it suitable for RNNs and Transformers and eliminating batch dependency.

4. **Group Normalization**: Normalizes within groups of channels, providing BatchNorm alternative for small batches and detection tasks.

5. **Instance Normalization**: Normalizes each channel independently per example, useful for style transfer and generative models.

6. **Architecture-Specific**: BatchNorm for CNNs, LayerNorm for RNNs/Transformers, GroupNorm for small-batch CNNs, InstanceNorm for style transfer.

7. **Benefits**: Normalization enables higher learning rates, faster convergence, better generalization, and reduced sensitivity to initialization.

8. **Learnable Parameters**: Scale ($\gamma$) and shift ($\beta$) allow networks to learn optimal normalization or recover original if needed.

9. **Theoretical Understanding**: Normalization improves gradient flow, smooths loss landscape, and provides implicit regularization through batch statistics noise.

10. **Practical Considerations**: Choice depends on architecture, batch size, and task requirements, with proper placement (before activation) and hyperparameter tuning crucial for effectiveness.
