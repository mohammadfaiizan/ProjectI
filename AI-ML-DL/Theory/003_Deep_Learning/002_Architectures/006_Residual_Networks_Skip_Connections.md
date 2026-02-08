# Residual Networks and Skip Connections

## Table of Contents

1. [Introduction](#introduction)
2. [The Degradation Problem](#the-degradation-problem)
3. [Residual Learning](#residual-learning)
4. [ResNet Architecture](#resnet-architecture)
5. [Gradient Flow Through Skip Connections](#gradient-flow-through-skip-connections)
6. [DenseNet: Dense Connections](#densenet-dense-connections)
7. [Highway Networks](#highway-networks)
8. [Skip Connection Variants](#skip-connection-variants)
9. [Design Principles](#design-principles)
10. [Key Takeaways](#key-takeaways)

## Introduction

Residual networks (ResNets) revolutionized deep learning by enabling training of networks with hundreds of layers through skip connections. These connections provide direct paths for gradient flow and enable residual learning, where networks learn to make incremental improvements rather than complete transformations.

This chapter covers the theory and design of residual networks, examining how skip connections solve the degradation problem, enable gradient flow, and facilitate training of very deep networks.

## The Degradation Problem

### Observation

As network depth increases:
- Training error increases (not just test error)
- Not due to overfitting
- Suggests optimization difficulty

### Hypothesis

Deeper networks should perform at least as well as shallower networks:
- Can learn identity mapping in extra layers
- Should not have higher training error

### Why Deeper Networks Fail

Possible reasons:
1. **Vanishing Gradients**: Gradients become too small
2. **Optimization Difficulty**: Harder to optimize deeper networks
3. **Degradation**: Layers become less useful

### Experimental Evidence

ResNet paper showed:
- 56-layer network has higher training error than 20-layer
- Not due to overfitting
- Suggests fundamental optimization problem

## Residual Learning

### Residual Block

Instead of learning $H(\mathbf{x})$, learn residual $F(\mathbf{x}) = H(\mathbf{x}) - \mathbf{x}$:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

where:
- $\mathbf{x}$: Input
- $\mathcal{F}(\mathbf{x}, \{W_i\})$: Residual mapping
- $\mathbf{y}$: Output

### Why Residual Learning Works

1. **Identity Mapping**: If optimal mapping is identity, residual is zero (easy to learn)
2. **Incremental Updates**: Learn small changes rather than complete transformation
3. **Gradient Flow**: Direct path through skip connection
4. **Easier Optimization**: Residuals are typically smaller

### Mathematical Formulation

For a residual block:

$$\mathbf{y} = \sigma(\mathcal{F}(\mathbf{x}) + \mathbf{x})$$

where $\sigma$ is activation (ReLU), applied after addition.

**If $\mathcal{F}(\mathbf{x}) = \mathbf{0}$**: Block becomes identity
**If $\mathcal{F}(\mathbf{x}) \neq \mathbf{0}$**: Block makes incremental change

### Projection Shortcuts

When dimensions don't match, use projection:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}) + W_s \mathbf{x}$$

where $W_s$ projects $\mathbf{x}$ to match dimensions.

## ResNet Architecture

### Basic Building Block

**Two-Layer Block**:
1. Conv (3x3) → BN → ReLU
2. Conv (3x3) → BN
3. Add shortcut
4. ReLU

**Three-Layer Block** (Bottleneck):
1. Conv (1x1) → BN → ReLU (reduce channels)
2. Conv (3x3) → BN → ReLU
3. Conv (1x1) → BN (restore channels)
4. Add shortcut
5. ReLU

### ResNet Variants

**ResNet-18/34**: Use basic blocks
**ResNet-50/101/152**: Use bottleneck blocks

### Architecture Design

1. **Initial Layer**: 7x7 conv, stride 2
2. **Max Pooling**: 3x3, stride 2
3. **Residual Blocks**: Multiple blocks per stage
4. **Stride**: Reduce spatial size in first block of each stage
5. **Global Average Pooling**: Before final FC layer
6. **Classification**: Final FC layer

### Implementation Example

```python
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out
```

### Pre-Activation vs. Post-Activation

**Original ResNet** (Post-activation):
- Conv → BN → ReLU → Conv → BN → Add → ReLU

**Pre-Activation ResNet**:
- BN → ReLU → Conv → BN → ReLU → Conv → Add

**Benefits of Pre-Activation**:
- Better gradient flow
- More stable training
- Simpler identity mapping

## Gradient Flow Through Skip Connections

### Direct Gradient Path

Skip connection provides direct gradient path:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \left(1 + \frac{\partial \mathcal{F}}{\partial \mathbf{x}}\right)$$

Even if $\frac{\partial \mathcal{F}}{\partial \mathbf{x}} \approx 0$, gradient flows through identity path.

### Preventing Vanishing Gradients

In deep networks without skip connections:
- Gradients multiplied many times
- Can vanish exponentially

With skip connections:
- Direct path maintains gradient magnitude
- Enables training of 100+ layer networks

### Experimental Validation

ResNet-152 (152 layers) trains successfully:
- Lower training error than ResNet-34
- Better generalization
- Demonstrates effectiveness of skip connections

### Identity Mapping Initialization

Initializing residual blocks near identity helps:
- Start with identity mapping
- Learn incremental improvements
- Faster convergence

## DenseNet: Dense Connections

DenseNet connects each layer to all subsequent layers.

### Dense Block

Each layer receives concatenated features from all previous layers:

$$\mathbf{x}_l = H_l([\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_{l-1}])$$

where $[\cdot]$ denotes concatenation.

### Architecture

**Dense Block**:
- Multiple layers
- Each layer's output concatenated to all previous
- Growth rate $k$: channels added per layer

**Transition Layer**:
- Between dense blocks
- 1x1 conv (reduce channels) + 2x2 avg pool

### Benefits

1. **Feature Reuse**: All previous features available
2. **Parameter Efficiency**: Fewer parameters than ResNet
3. **Gradient Flow**: Many paths for gradients
4. **Regularization**: Implicit regularization through feature reuse

### Comparison with ResNet

| Property | ResNet | DenseNet |
|----------|--------|----------|
| Connection | Additive | Concatenative |
| Feature Reuse | Limited | Extensive |
| Parameters | More | Fewer |
| Memory | Less | More |

## Highway Networks

Highway networks use gated skip connections.

### Highway Layer

$$\mathbf{y} = T(\mathbf{x}) \odot H(\mathbf{x}) + (1 - T(\mathbf{x})) \odot \mathbf{x}$$

where:
- $T(\mathbf{x})$: Transform gate (sigmoid)
- $H(\mathbf{x})$: Transform function
- $(1-T(\mathbf{x}))$: Carry gate

### Properties

- **Gated**: Learns when to use transformation vs. identity
- **Flexible**: Can learn to pass through or transform
- **Predecessor**: Inspired ResNet design

### Comparison

**Highway**: Gated, learnable
**ResNet**: Fixed, always additive
**DenseNet**: Concatenative, all-to-all

## Skip Connection Variants

### Pre-Activation ResNet

Apply activation before convolution:
- Better gradient flow
- Simpler identity mapping
- More stable training

### Stochastic Depth

Randomly skip layers during training:
- Regularization effect
- Faster training
- Better generalization

### FractalNet

Recursive architecture with skip connections:
- Multiple paths of different lengths
- Better feature learning

### ResNeXt

ResNet with grouped convolutions:
- Increases capacity
- More efficient than increasing width

### Wide ResNet

Increases width instead of depth:
- Fewer layers, more channels
- Faster training
- Good performance

## Design Principles

### When to Use Skip Connections

1. **Deep Networks**: Essential for 50+ layers
2. **Gradient Flow**: When gradients vanish
3. **Identity Mapping**: When identity is reasonable
4. **Residual Learning**: When incremental updates make sense

### Skip Connection Placement

1. **After Addition**: Original ResNet
2. **Before Addition**: Pre-activation (better)
3. **Multiple Skips**: DenseNet (all-to-all)

### Dimension Matching

When dimensions don't match:
1. **Projection**: Learnable linear layer
2. **Zero Padding**: Add zeros (simpler, may hurt performance)
3. **Stride**: Reduce spatial size

### Initialization

1. **Near Identity**: Initialize residual blocks near identity
2. **Small Residuals**: Encourage small initial residuals
3. **Batch Normalization**: Helps with initialization

## Key Takeaways

1. **Degradation Problem**: Deeper networks can have higher training error, suggesting optimization difficulty rather than capacity issues.

2. **Residual Learning**: Learning residuals $F(\mathbf{x}) = H(\mathbf{x}) - \mathbf{x}$ instead of direct mapping $H(\mathbf{x})$ enables easier optimization and identity mapping.

3. **Skip Connections**: Provide direct gradient paths, preventing vanishing gradients and enabling training of 100+ layer networks.

4. **ResNet Architecture**: Uses residual blocks with additive skip connections, enabling successful training of very deep networks (ResNet-152).

5. **Gradient Flow**: Skip connections maintain gradient magnitude through identity path, even when transformation path has small gradients.

6. **DenseNet**: Connects each layer to all subsequent layers via concatenation, enabling extensive feature reuse and parameter efficiency.

7. **Highway Networks**: Use gated skip connections that learn when to transform vs. pass through, inspiring ResNet design.

8. **Pre-Activation**: Applying activation before convolution improves gradient flow and simplifies identity mapping compared to post-activation.

9. **Design Principles**: Skip connections are essential for deep networks, with proper dimension matching and initialization crucial for success.

10. **Impact**: Residual learning and skip connections enabled the training of very deep networks, revolutionizing computer vision and deep learning.
