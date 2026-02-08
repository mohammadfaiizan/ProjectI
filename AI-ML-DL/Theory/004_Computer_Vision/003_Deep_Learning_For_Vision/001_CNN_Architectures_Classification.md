# CNN Architectures for Classification

## Table of Contents

1. [Introduction](#introduction)
2. [LeNet and Early CNNs](#lenet-and-early-cnns)
3. [AlexNet and Deep Learning Revolution](#alexnet-and-deep-learning-revolution)
4. [VGG Networks](#vgg-networks)
5. [GoogLeNet and Inception](#googlenet-and-inception)
6. [ResNet and Residual Learning](#resnet-and-residual-learning)
7. [DenseNet](#densenet)
8. [MobileNet and Efficient Architectures](#mobilenet-and-efficient-architectures)
9. [EfficientNet](#efficientnet)
10. [Vision Transformer](#vision-transformer)
11. [Key Takeaways](#key-takeaways)

## Introduction

Convolutional Neural Networks (CNNs) have revolutionized computer vision, achieving superhuman performance on image classification tasks. The evolution from LeNet to modern architectures like Vision Transformers represents decades of architectural innovation, addressing challenges including depth, efficiency, and representational capacity.

Each architecture introduces key innovations: LeNet established the CNN paradigm, AlexNet demonstrated deep learning's potential, VGG showed the power of depth, GoogLeNet introduced efficient building blocks, ResNet enabled very deep networks, and Vision Transformers brought attention mechanisms to vision. Understanding these architectures provides insight into modern deep learning design principles.

## LeNet and Early CNNs

LeNet-5, developed by Yann LeCun in 1998, established the fundamental CNN architecture still used today.

### LeNet-5 Architecture

LeNet-5 consists of:
1. **Convolutional layer**: 6 filters, 5×5, stride 1
2. **Subsampling layer**: 2×2 average pooling, stride 2
3. **Convolutional layer**: 16 filters, 5×5, stride 1
4. **Subsampling layer**: 2×2 average pooling, stride 2
5. **Fully connected layers**: 120, 84, 10 neurons

Key innovations:
- **Convolution**: Local connectivity and weight sharing
- **Subsampling**: Spatial dimension reduction
- **End-to-end learning**: Backpropagation through entire network

### Convolutional Layer

A convolutional layer applies learnable filters to input:

$$y_{i,j} = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} w_{m,n} \cdot x_{i+m, j+n} + b$$

where:
- $x$: Input feature map
- $w$: Filter weights
- $b$: Bias term
- $y$: Output feature map

Multiple filters produce multiple output channels.

### Pooling Layer

Pooling reduces spatial dimensions:

**Max pooling**: $y_{i,j} = \max_{m,n \in W} x_{i+m, j+n}$
**Average pooling**: $y_{i,j} = \frac{1}{|W|} \sum_{m,n \in W} x_{i+m, j+n}$

where $W$ is the pooling window (typically 2×2).

### LeNet Impact

LeNet demonstrated:
- CNNs can learn hierarchical features automatically
- End-to-end training is feasible
- Local connectivity reduces parameters significantly

However, limited by:
- Shallow depth (5 layers)
- Small datasets
- Computational constraints

## AlexNet and Deep Learning Revolution

AlexNet, winner of ImageNet 2012, ignited the deep learning revolution by demonstrating the power of deep CNNs.

### AlexNet Architecture

AlexNet consists of:
1. **Conv1**: 96 filters, 11×11, stride 4, ReLU
2. **MaxPool1**: 3×3, stride 2
3. **Conv2**: 256 filters, 5×5, ReLU
4. **MaxPool2**: 3×3, stride 2
5. **Conv3-5**: 384, 384, 256 filters, 3×3, ReLU
6. **MaxPool3**: 3×3, stride 2
7. **FC6-7**: 4096 neurons each
8. **FC8**: 1000 neurons (ImageNet classes)

### Key Innovations

**ReLU Activation**: Replaced tanh/sigmoid
$$f(x) = \max(0, x)$$

Benefits:
- Faster training (no saturation)
- Sparse activations
- Better gradient flow

**Dropout**: Randomly set neurons to zero during training
$$h_i = \begin{cases} 0 & \text{with probability } p \\ \frac{x_i}{1-p} & \text{otherwise} \end{cases}$$

Prevents overfitting in fully connected layers.

**Data Augmentation**: 
- Random crops
- Horizontal flips
- Color jittering

**Multi-GPU Training**: Split model across 2 GPUs for parallelization.

### Impact

AlexNet achieved:
- Top-5 error: 15.3% (vs 26% previous best)
- Demonstrated scalability to large datasets
- Established deep learning as viable approach

## VGG Networks

VGG (Visual Geometry Group) networks demonstrated that depth is crucial for performance, using only 3×3 convolutions.

### VGG Architecture Principles

**Small filters**: Stack 3×3 convolutions instead of larger filters
- Two 3×3 convs ≈ one 5×5 conv (same receptive field, fewer parameters)
- Three 3×3 convs ≈ one 7×7 conv

**Depth**: Networks from 11 to 19 layers (VGG-11, VGG-13, VGG-16, VGG-19)

**Architecture pattern**:
- Multiple conv layers (with ReLU)
- Max pooling (2×2, stride 2)
- Fully connected layers at end

### VGG-16 Structure

1. **Block 1**: 2× conv3-64, MaxPool
2. **Block 2**: 2× conv3-128, MaxPool
3. **Block 3**: 3× conv3-256, MaxPool
4. **Block 4**: 3× conv3-512, MaxPool
5. **Block 5**: 3× conv3-512, MaxPool
6. **FC**: 4096, 4096, 1000

### Key Insights

- **Depth matters**: Deeper networks learn better representations
- **Small filters**: 3×3 convolutions are building blocks
- **Receptive field**: Stacking small filters increases receptive field efficiently

### Limitations

- **Many parameters**: Fully connected layers have millions of parameters
- **Memory intensive**: Large feature maps in early layers
- **Slow training**: Deep networks require long training

## GoogLeNet and Inception

GoogLeNet (Inception v1) introduced the Inception module, enabling wider and more efficient networks.

### Inception Module

The Inception module applies multiple filter sizes in parallel:

$$\text{Inception}(x) = \text{Concat}[\text{Conv1×1}(x), \text{Conv3×3}(x), \text{Conv5×5}(x), \text{MaxPool3×3}(x)]$$

Benefits:
- **Multi-scale features**: Captures features at different scales
- **Sparse connections**: More efficient than dense connections
- **Wider network**: Increases capacity without excessive depth

### 1×1 Convolutions

1×1 convolutions (pointwise convolutions) reduce dimensionality:

$$y_{i,j,c'} = \sum_{c=1}^{C} w_{c',c} \cdot x_{i,j,c} + b_{c'}$$

Benefits:
- **Dimensionality reduction**: Reduce channels before expensive operations
- **Non-linearity**: Add ReLU for additional non-linearity
- **Parameter efficiency**: Fewer parameters than larger convolutions

### Inception v1 Architecture

- **Stem**: Initial conv and pooling layers
- **Inception modules**: 9 Inception modules in 3 groups
- **Auxiliary classifiers**: Two auxiliary outputs for training
- **Average pooling**: Replace fully connected layers

### Inception Variants

**Inception v2/v3**:
- Factorize 5×5 into two 3×3 convolutions
- Factorize 7×7 into multiple 3×3 convolutions
- Batch normalization

**Inception v4**:
- Unified Inception blocks
- Residual connections (Inception-ResNet)

### Impact

- Reduced parameters compared to VGG
- Better accuracy with efficient architecture
- Established design pattern for efficient CNNs

## ResNet and Residual Learning

ResNet (Residual Networks) enabled training of very deep networks (100+ layers) through residual connections.

### Residual Block

A residual block learns the residual (difference) rather than the mapping directly:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$

where:
- $\mathbf{x}$: Input
- $\mathcal{F}(\mathbf{x})$: Residual function (conv layers)
- $\mathbf{y}$: Output

If identity mapping is optimal, $\mathcal{F}(\mathbf{x}) = 0$ is easier to learn than $\mathcal{F}(\mathbf{x}) = \mathbf{x}$.

### Residual Learning Motivation

For very deep networks, learning identity mapping becomes difficult:
- Gradients vanish in deep networks
- Optimization becomes harder

Residual learning addresses this:
- If identity is optimal, residual is zero (easy to learn)
- Gradients flow through skip connections
- Enables training of 100+ layer networks

### ResNet Architecture

**Basic block** (for shallow ResNets):
- Two 3×3 convolutions
- Skip connection (identity if same dimensions)

**Bottleneck block** (for deep ResNets):
- 1×1 conv (reduce), 3×3 conv, 1×1 conv (expand)
- More efficient for deep networks

**Architecture**:
- Initial conv + pooling
- 4 groups of residual blocks
- Global average pooling
- Fully connected layer

### ResNet Variants

- **ResNet-18, -34**: Basic blocks
- **ResNet-50, -101, -152**: Bottleneck blocks
- **ResNet-152**: 152 layers, won ImageNet 2015

### Batch Normalization

ResNet uses batch normalization:
$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
$$y = \gamma \hat{x} + \beta$$

Benefits:
- Faster training
- Higher learning rates
- Regularization effect

## DenseNet

DenseNet (Densely Connected Convolutional Networks) connects each layer to all subsequent layers, maximizing information flow.

### Dense Block

In a dense block, each layer receives feature maps from all previous layers:

$$\mathbf{x}_l = H_l([\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_{l-1}])$$

where $H_l$ is a composite function (BN, ReLU, Conv) and $[\cdot]$ denotes concatenation.

### Architecture

**Dense block**: Multiple layers with dense connections
**Transition layer**: Between dense blocks
- 1×1 conv (bottleneck)
- 2×2 average pooling

**Growth rate**: $k$ new feature maps per layer
- Small $k$ (e.g., 12) keeps model compact
- Total features: $k_0 + k(l-1)$ for $l$ layers

### Advantages

- **Feature reuse**: All previous features available
- **Parameter efficiency**: Fewer parameters than ResNet
- **Gradient flow**: Direct paths for gradients
- **Regularization**: Dense connections act as regularization

### Comparison with ResNet

**ResNet**: $\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$ (addition)
**DenseNet**: $\mathbf{y} = H([\mathbf{x}_0, \ldots, \mathbf{x}_{l-1}])$ (concatenation)

DenseNet uses concatenation, requiring more memory but enabling better feature reuse.

## MobileNet and Efficient Architectures

MobileNet addresses efficiency for mobile and embedded devices through depthwise separable convolutions.

### Depthwise Separable Convolution

Standard convolution:
$$y_{i,j,k} = \sum_{c=1}^{C} \sum_{m,n} w_{k,c,m,n} \cdot x_{i+m, j+n, c}$$

Depthwise separable convolution splits into two operations:

**Depthwise convolution**: Filter each input channel separately
$$y'_{i,j,c} = \sum_{m,n} w_{c,m,n} \cdot x_{i+m, j+n, c}$$

**Pointwise convolution**: 1×1 convolution across channels
$$y_{i,j,k} = \sum_{c=1}^{C} w_{k,c} \cdot y'_{i,j,c}$$

### Computational Reduction

Standard conv: $D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F$
Depthwise separable: $D_K \cdot D_K \cdot M \cdot D_F \cdot D_F + M \cdot N \cdot D_F \cdot D_F$

Reduction ratio: $\frac{1}{N} + \frac{1}{D_K^2}$ (typically 8-9× for 3×3 filters)

### MobileNet Architecture

- **Initial conv**: Standard 3×3, stride 2
- **13 depthwise separable blocks**: Each with depthwise + pointwise conv
- **Global average pooling**: Replace FC layers
- **Width multiplier**: $\alpha \in \{0.25, 0.5, 0.75, 1.0\}$ scales channels
- **Resolution multiplier**: $\rho$ scales input resolution

### MobileNet v2

Adds inverted residuals and linear bottlenecks:
- **Expansion**: 1×1 conv expands channels
- **Depthwise**: 3×3 depthwise conv
- **Projection**: 1×1 conv projects back
- **Skip connection**: Only if same dimensions

Improves accuracy while maintaining efficiency.

## EfficientNet

EfficientNet uses compound scaling to balance depth, width, and resolution optimally.

### Compound Scaling

Instead of scaling one dimension, scale all three:

**Depth**: $d = \alpha^\phi$
**Width**: $w = \beta^\phi$
**Resolution**: $r = \gamma^\phi$

where $\alpha \beta^2 \gamma^2 \approx 2$ and $\phi$ is the compound coefficient.

### EfficientNet Architecture

**Mobile inverted bottleneck (MBConv)**:
- Expansion → Depthwise → SE → Projection
- Similar to MobileNet v2 with squeeze-and-excitation

**Stages**:
- Initial conv + pooling
- 7 MBConv blocks with different configurations
- Final conv + global pooling + FC

### Scaling Strategy

1. **Baseline**: EfficientNet-B0 (small, fast)
2. **Compound scaling**: Scale B0 to B1-B7
3. **Optimal**: Found via neural architecture search

### Performance

EfficientNet-B7 achieves:
- ImageNet top-1: 84.4% accuracy
- 8.4× smaller, 6.1× faster than best previous model

## Vision Transformer

Vision Transformer (ViT) applies Transformer architecture to images, treating patches as sequences.

### Image Patches

Divide image into patches:
$$\mathbf{x}_p \in \mathbb{R}^{P^2 \times C}$$

where $P$ is patch size (e.g., 16×16) and $C$ is channels.

### Patch Embeddings

Linear projection to embedding dimension:
$$\mathbf{z}_0 = [\mathbf{x}_{class}; \mathbf{x}_p^1 \mathbf{E}; \mathbf{x}_p^2 \mathbf{E}; \ldots; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{pos}$$

where:
- $\mathbf{E}$: Patch embedding matrix
- $\mathbf{E}_{pos}$: Position embeddings
- $\mathbf{x}_{class}$: Learnable class token

### Transformer Encoder

Standard Transformer encoder:
$$\mathbf{z}'_l = \text{MSA}(\text{LN}(\mathbf{z}_{l-1})) + \mathbf{z}_{l-1}$$
$$\mathbf{z}_l = \text{MLP}(\text{LN}(\mathbf{z}'_l)) + \mathbf{z}'_l$$

where:
- MSA: Multi-head self-attention
- LN: Layer normalization
- MLP: Multi-layer perceptron

### Self-Attention

Self-attention computes relationships between patches:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

Multi-head attention uses $h$ parallel attention heads.

### ViT Variants

- **ViT-Base**: 12 layers, 768 dim, 12 heads
- **ViT-Large**: 24 layers, 1024 dim, 16 heads
- **ViT-Huge**: 32 layers, 1280 dim, 16 heads

### Pre-training

ViT requires large-scale pre-training:
- **ImageNet-21k**: 14M images, 21k classes
- **JFT-300M**: 300M images (Google internal)

Transfer learning to downstream tasks.

### Hybrid Approaches

**CNN + ViT**: Use CNN feature maps as patches
**DeiT**: Data-efficient image transformers (train on ImageNet only)
**Swin Transformer**: Hierarchical vision transformer with shifted windows

## Key Takeaways

1. LeNet established the CNN paradigm with convolutional layers, pooling, and end-to-end learning, demonstrating automatic feature learning.

2. AlexNet demonstrated deep learning's potential with ReLU activations, dropout, and data augmentation, achieving breakthrough ImageNet performance.

3. VGG showed that depth is crucial, using only 3×3 convolutions to build deeper networks efficiently.

4. GoogLeNet introduced Inception modules with parallel convolutions at multiple scales, enabling wider and more efficient networks.

5. ResNet enabled very deep networks (100+ layers) through residual learning, where networks learn residuals rather than direct mappings.

6. DenseNet maximizes information flow by connecting each layer to all subsequent layers, improving parameter efficiency and gradient flow.

7. MobileNet uses depthwise separable convolutions for mobile efficiency, achieving good accuracy with minimal computation.

8. EfficientNet uses compound scaling to balance depth, width, and resolution optimally, achieving state-of-the-art efficiency.

9. Vision Transformer applies Transformer architecture to images by treating patches as sequences, achieving competitive performance with large-scale pre-training.

10. Understanding CNN evolution from LeNet to Vision Transformers provides insight into architectural design principles: depth, efficiency, residual learning, attention, and scaling strategies.
