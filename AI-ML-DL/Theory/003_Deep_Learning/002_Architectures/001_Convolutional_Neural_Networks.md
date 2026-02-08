# Convolutional Neural Networks

## Table of Contents

1. [Introduction](#introduction)
2. [The Convolution Operation](#the-convolution-operation)
3. [Pooling Layers](#pooling-layers)
4. [Stride and Padding](#stride-and-padding)
5. [Parameter Sharing and Locality](#parameter-sharing-and-locality)
6. [CNN Architectures: LeNet to EfficientNet](#cnn-architectures-lenet-to-efficientnet)
7. [Modern CNN Design Principles](#modern-cnn-design-principles)
8. [Depthwise and Pointwise Convolutions](#depthwise-and-pointwise-convolutions)
9. [Implementation Considerations](#implementation-considerations)
10. [Key Takeaways](#key-takeaways)

## Introduction

Convolutional Neural Networks (CNNs) revolutionized computer vision and remain fundamental to modern deep learning. Unlike fully connected networks that treat input as a flat vector, CNNs exploit spatial structure in images through convolutional operations, parameter sharing, and hierarchical feature learning.

This chapter covers the mathematical foundations of CNNs, from basic convolution operations to advanced architectures like ResNet, DenseNet, and EfficientNet, examining how these designs enable effective visual pattern recognition.

## The Convolution Operation

### Mathematical Definition

Convolution is a mathematical operation that combines two functions. In the context of CNNs, we perform discrete convolution:

$$(f * g)[i, j] = \sum_{m} \sum_{n} f[m, n] \cdot g[i-m, j-n]$$

For image processing, this becomes:

$$(I * K)[i, j] = \sum_{m} \sum_{n} I[i+m, j+n] \cdot K[m, n]$$

where $I$ is the input image and $K$ is the kernel (filter).

### Cross-Correlation

In practice, CNNs use cross-correlation (which is convolution without flipping the kernel):

$$(I \star K)[i, j] = \sum_{m} \sum_{n} I[i+m, j+n] \cdot K[m, n]$$

This is computationally equivalent and more intuitive.

### Convolutional Layer

A convolutional layer applies multiple filters to produce feature maps:

$$\mathbf{Z}_{i,j,k} = \sum_{c=0}^{C_{\text{in}}-1} \sum_{u=0}^{H_k-1} \sum_{v=0}^{W_k-1} \mathbf{X}_{i+u, j+v, c} \cdot \mathbf{W}_{u,v,c,k} + b_k$$

where:
- $\mathbf{X}$ is input of shape $(H_{\text{in}}, W_{\text{in}}, C_{\text{in}})$
- $\mathbf{W}$ is kernel of shape $(H_k, W_k, C_{\text{in}}, C_{\text{out}})$
- $\mathbf{Z}$ is output of shape $(H_{\text{out}}, W_{\text{out}}, C_{\text{out}})$
- $b_k$ is bias for filter $k$

### Feature Maps

Each filter produces a feature map detecting specific patterns:
- **Edge Detectors**: Horizontal, vertical, diagonal edges
- **Texture Detectors**: Various textures and patterns
- **Object Parts**: Higher-level features in deeper layers

### Convolution as Matrix Multiplication

Convolution can be implemented as matrix multiplication using the im2col operation, which unrolls image patches into columns:

$$\mathbf{Z} = \text{im2col}(\mathbf{X}) \cdot \mathbf{W}_{\text{flattened}}$$

This enables efficient implementation using optimized matrix multiplication libraries.

## Pooling Layers

Pooling layers reduce spatial dimensions while retaining important information.

### Max Pooling

Selects the maximum value in each pooling region:

$$\text{MaxPool}(\mathbf{X})_{i,j} = \max_{u,v \in \mathcal{R}_{i,j}} \mathbf{X}_{u,v}$$

where $\mathcal{R}_{i,j}$ is the pooling region.

**Properties**:
- Translation invariance
- Reduces computational cost
- Reduces overfitting
- Preserves strongest activations

### Average Pooling

Computes the average value in each pooling region:

$$\text{AvgPool}(\mathbf{X})_{i,j} = \frac{1}{|\mathcal{R}_{i,j}|} \sum_{u,v \in \mathcal{R}_{i,j}} \mathbf{X}_{u,v}$$

**Properties**:
- Smooths activations
- Less aggressive than max pooling
- Sometimes used in final layers

### Global Pooling

Pools over entire spatial dimensions:

- **Global Max Pooling**: $\max_{i,j} \mathbf{X}_{i,j}$
- **Global Average Pooling**: $\frac{1}{HW} \sum_{i,j} \mathbf{X}_{i,j}$

Often used before classification layers to reduce parameters.

### Adaptive Pooling

Pools to a fixed output size regardless of input size:

- **Adaptive Max Pooling**: Output size $(H_{\text{out}}, W_{\text{out}})$
- **Adaptive Average Pooling**: Output size $(H_{\text{out}}, W_{\text{out}})$

Useful for handling variable input sizes.

## Stride and Padding

### Stride

Stride controls how much the filter moves:

For stride $s$, output size is:

$$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} - H_k + 2P}{s} \right\rfloor + 1$$

$$W_{\text{out}} = \left\lfloor \frac{W_{\text{in}} - W_k + 2P}{s} \right\rfloor + 1$$

where $P$ is padding.

**Stride > 1**: Reduces spatial dimensions (alternative to pooling)

### Padding

Padding adds zeros around the input:

- **Valid Padding**: No padding ($P = 0$)
- **Same Padding**: Padding to preserve size ($P = \frac{K-1}{2}$ for odd $K$)

**Same Padding Formula**:
$$P = \left\lfloor \frac{K-1}{2} \right\rfloor$$

For stride $s=1$ and same padding, output size equals input size.

### Dilation

Dilated convolution increases receptive field without increasing parameters:

$$(I \star_d K)[i, j] = \sum_{m} \sum_{n} I[i+dm, j+dn] \cdot K[m, n]$$

where $d$ is the dilation rate.

**Benefits**:
- Larger receptive field
- Same number of parameters
- Useful for semantic segmentation

## Parameter Sharing and Locality

### Parameter Sharing

Unlike fully connected layers, convolutional layers share parameters across spatial locations:

- **Fully Connected**: $O(H \times W \times C_{\text{in}} \times C_{\text{out}})$ parameters
- **Convolutional**: $O(H_k \times W_k \times C_{\text{in}} \times C_{\text{out}})$ parameters

This dramatically reduces parameters and enables translation equivariance.

### Locality

Convolutional layers exploit local connectivity:
- Each neuron connects only to a local region
- Mimics biological visual processing
- Enables hierarchical feature learning

### Translation Equivariance

Convolution is translation equivariant:

$$\text{Conv}(\text{Translate}(I)) = \text{Translate}(\text{Conv}(I))$$

If input shifts, output shifts by the same amount (up to boundary effects).

### Translation Invariance

Pooling provides translation invariance:
- Small translations don't affect pooled output
- Important for object recognition

## CNN Architectures: LeNet to EfficientNet

### LeNet-5 (1998)

Pioneering CNN architecture:
- Convolutional layers: 2
- Pooling layers: 2
- Fully connected layers: 2
- Designed for digit recognition

**Architecture**:
1. Conv (6 filters, 5x5) → Pool (2x2)
2. Conv (16 filters, 5x5) → Pool (2x2)
3. FC (120) → FC (84) → FC (10)

### AlexNet (2012)

Breakthrough architecture winning ImageNet:
- 8 layers (5 conv + 3 FC)
- ReLU activation
- Dropout regularization
- Data augmentation

**Key Innovations**:
- ReLU instead of tanh
- Dropout (0.5)
- Local response normalization
- Overlapping pooling

### VGG (2014)

Very deep network with small filters:
- 3x3 convolutions throughout
- 11-19 layers
- Simpler architecture than AlexNet

**Design Principles**:
- Small filters (3x3) instead of large (5x5, 7x7)
- Deeper networks
- More non-linearities

**VGG-16 Architecture**:
- 13 conv layers + 3 FC layers
- All 3x3 convolutions
- Max pooling (2x2)

### GoogLeNet / Inception (2014)

Inception modules with parallel convolutions:
- Multiple filter sizes in parallel
- 1x1 convolutions for dimensionality reduction
- 22 layers but efficient

**Inception Module**:
- Parallel: 1x1, 3x3, 5x5, max pooling
- Concatenate outputs
- 1x1 convs reduce channels before 3x3/5x5

### ResNet (2015)

Residual learning with skip connections:
- 152 layers (deeper than VGG)
- Residual blocks
- Batch normalization

**Residual Block**:
$$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$

Enables training of very deep networks.

**Variants**:
- ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152

### DenseNet (2017)

Dense connectivity:
- Each layer connects to all subsequent layers
- Feature reuse
- Fewer parameters

**Dense Block**:
$$\mathbf{x}_l = H_l([\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_{l-1}])$$

### MobileNet (2017)

Efficient architecture for mobile:
- Depthwise separable convolution
- Width multiplier
- Resolution multiplier

**Depthwise Separable Convolution**:
1. Depthwise conv: Filter each channel separately
2. Pointwise conv: 1x1 convolution to combine channels

### EfficientNet (2019)

Compound scaling:
- Scales depth, width, and resolution together
- Better accuracy-efficiency trade-off

**Scaling**:
- Depth: $d = \alpha^\phi$
- Width: $w = \beta^\phi$
- Resolution: $r = \gamma^\phi$

where $\alpha \beta^2 \gamma^2 \approx 2$ and $\phi$ is the compound coefficient.

## Modern CNN Design Principles

### Depth

Deeper networks can learn more complex features:
- But diminishing returns
- Require skip connections (ResNet)
- Need proper initialization

### Width

Wider networks have more capacity:
- But more parameters
- Can overfit more easily
- Often combined with depth

### Resolution

Higher resolution inputs:
- More spatial information
- But more computation
- Important for fine-grained tasks

### Efficiency

Modern designs focus on:
- **FLOPs**: Floating point operations
- **Parameters**: Model size
- **Latency**: Inference time
- **Memory**: Activation memory

### Design Patterns

1. **Bottleneck Design**: Reduce channels before expensive operations
2. **Grouped Convolutions**: Split channels into groups
3. **Squeeze-and-Excitation**: Channel attention
4. **Spatial Attention**: Spatial feature selection

## Depthwise and Pointwise Convolutions

### Standard Convolution

Standard convolution:
- Input: $(H, W, C_{\text{in}})$
- Kernel: $(K, K, C_{\text{in}}, C_{\text{out}})$
- Parameters: $K \times K \times C_{\text{in}} \times C_{\text{out}}$

### Depthwise Separable Convolution

Two-step process:

**1. Depthwise Convolution**:
- One filter per input channel
- Parameters: $K \times K \times C_{\text{in}}$

**2. Pointwise Convolution**:
- 1x1 convolution to combine channels
- Parameters: $1 \times 1 \times C_{\text{in}} \times C_{\text{out}}$

**Total Parameters**: $K \times K \times C_{\text{in}} + C_{\text{in}} \times C_{\text{out}}$

**Reduction Factor**:
$$\frac{K \times K \times C_{\text{in}} + C_{\text{in}} \times C_{\text{out}}}{K \times K \times C_{\text{in}} \times C_{\text{out}}} \approx \frac{1}{C_{\text{out}}} + \frac{1}{K^2}$$

For $K=3$ and typical $C_{\text{out}}$, reduction is ~8-9x.

### Grouped Convolution

Splits input channels into groups:

- **Standard**: All channels → All channels
- **Grouped**: Groups of channels → Groups of channels

**Parameters**: Reduced by group size $G$

Used in ResNeXt, ShuffleNet.

## Implementation Considerations

### Memory Efficiency

- **Gradient Checkpointing**: Trade computation for memory
- **Mixed Precision**: FP16 instead of FP32
- **In-place Operations**: When safe

### Computational Efficiency

- **Winograd Convolution**: Faster for small kernels
- **FFT Convolution**: Efficient for large kernels
- **Sparse Convolutions**: Skip zero activations

### Hardware Optimization

- **Tensor Cores**: NVIDIA GPUs for mixed precision
- **SIMD**: Vectorized operations
- **Cache Optimization**: Data layout matters

### Framework Implementation

```python
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn(self.conv2(out))
        out += residual
        return self.relu(out)
```

## Key Takeaways

1. **Convolution Operation**: The fundamental operation of CNNs, exploiting spatial locality and enabling translation equivariance through parameter sharing.

2. **Pooling Layers**: Reduce spatial dimensions while retaining important information, providing translation invariance and reducing computational cost.

3. **Stride and Padding**: Control output size and receptive field. Stride reduces dimensions; padding preserves size or controls boundary effects.

4. **Parameter Sharing**: Dramatically reduces parameters compared to fully connected layers while enabling translation equivariance.

5. **Architectural Evolution**: From LeNet to EfficientNet, architectures evolved to be deeper, more efficient, and better at learning hierarchical features.

6. **ResNet**: Residual connections enable training of very deep networks (100+ layers) by providing direct gradient paths.

7. **Efficiency**: Modern designs (MobileNet, EfficientNet) optimize for accuracy-efficiency trade-offs using depthwise separable convolutions and compound scaling.

8. **Depthwise Separable Convolution**: Reduces parameters and computation by separating spatial and channel-wise operations, enabling efficient mobile architectures.

9. **Design Principles**: Modern CNNs balance depth, width, and resolution, with attention to efficiency metrics (FLOPs, parameters, latency).

10. **Implementation**: Efficient implementations leverage optimized libraries, hardware acceleration, and memory optimization techniques for practical deployment.
