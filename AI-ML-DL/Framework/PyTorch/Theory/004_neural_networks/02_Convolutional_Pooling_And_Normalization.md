# Convolutional, Pooling, and Normalization Layers

## Table of Contents

- [Convolutional Layers](#convolutional-layers)
- [Transposed Convolution](#transposed-convolution)
- [Pooling Layers](#pooling-layers)
- [Normalization Layers](#normalization-layers)

---

## Convolutional Layers

Convolutional layers provide **local feature detection**, **translation equivariance**, and **parameter sharing** across spatial dimensions. PyTorch offers Conv1d, Conv2d, and Conv3d for 1D (sequences), 2D (images), and 3D (video/volumetric) data.

### Conv1d, Conv2d, Conv3d

| Layer | Input Shape | Use Case |
|-------|-------------|----------|
| Conv1d | (batch, channels, length) | Sequences, time series |
| Conv2d | (batch, channels, height, width) | Images |
| Conv3d | (batch, channels, depth, height, width) | Video, volumetric |

### Key Parameters

- **kernel_size**: Size of the convolving kernel
- **stride**: Step size of the convolution (default: 1)
- **padding**: Zero-padding added to input (default: 0)
- **dilation**: Spacing between kernel elements (atrous convolution)
- **groups**: Number of blocked connections (groups=in_channels for depthwise)

### Conv1d

```python
import torch
import torch.nn as nn

conv1d = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=5, stride=2, padding=2)
x = torch.randn(3, 4, 20)
out = conv1d(x)
print(out.shape)
```

### Conv2d

```python
conv2d = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
x = torch.randn(4, 3, 32, 32)
out = conv2d(x)
print(out.shape)
```

### Padding Types

```python
conv_no_pad = nn.Conv2d(3, 16, kernel_size=3, padding=0)
conv_same_pad = nn.Conv2d(3, 16, kernel_size=3, padding=1)
conv_asym = nn.Conv2d(3, 16, kernel_size=3, padding=(1, 2))
```

### Stride and Dilation

```python
conv_stride = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
conv_dilated = nn.Conv2d(3, 16, kernel_size=3, padding=2, dilation=2)
```

### Grouped and Depthwise Convolution

```python
conv_grouped = nn.Conv2d(8, 16, kernel_size=3, padding=1, groups=2)
conv_depthwise = nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8)
```

### Output Size Formula

For convolution:
```
out = (in + 2*padding - dilation*(kernel_size-1) - 1) // stride + 1
```

### Conv3d

```python
conv3d = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3)
x = torch.randn(2, 1, 16, 32, 32)
out = conv3d(x)
```

### Standard Conv Block

```python
class ConvBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
```

### Depthwise Separable Convolution

```python
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        return self.pointwise(x)
```

---

## Transposed Convolution

**ConvTranspose1d/2d/3d** perform upsampling by applying a transposed convolution. Used in autoencoders, segmentation, and generative models.

```python
conv_transpose = nn.ConvTranspose2d(
    in_channels=16,
    out_channels=3,
    kernel_size=3,
    stride=2,
    padding=1,
    output_padding=1
)
x = torch.randn(2, 16, 16, 16)
out = conv_transpose(x)
```

### Output Size Formula

```
out = (in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
```

---

## Pooling Layers

Pooling reduces spatial dimensions, provides **translation invariance**, and reduces computation.

### MaxPool and AvgPool

```python
maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
avgpool2d = nn.AvgPool2d(kernel_size=2, stride=2)

x = torch.randn(4, 3, 32, 32)
out_max = maxpool2d(x)
out_avg = avgpool2d(x)
```

### Adaptive Pooling

**AdaptiveMaxPool** and **AdaptiveAvgPool** produce a fixed output size regardless of input dimensions. **Global average pooling** uses `output_size=(1, 1)`.

```python
adaptive_max = nn.AdaptiveMaxPool2d(output_size=(7, 7))
adaptive_avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))

out = adaptive_avg(x)
print(out.shape)
```

### 1D and 3D Pooling

```python
maxpool1d = nn.MaxPool1d(kernel_size=3, stride=2)
maxpool3d = nn.MaxPool3d(kernel_size=2)
```

### MaxUnpool

Reverses max pooling using stored indices.

```python
maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
maxunpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

pooled, indices = maxpool(x)
unpooled = maxunpool(pooled, indices)
```

### Functional Pooling

```python
import torch.nn.functional as F

out = F.max_pool2d(x, kernel_size=2, stride=2)
out, indices = F.max_pool2d(x, 2, 2, return_indices=True)
out = F.adaptive_avg_pool2d(x, (1, 1))
```

### Output Size Formula

```
out = (in + 2*padding - dilation*(kernel_size-1) - 1) // stride + 1
```

---

## Normalization Layers

Normalization stabilizes training by controlling the distribution of activations. Different layers normalize over different dimensions.

### BatchNorm

**BatchNorm** normalizes over the batch dimension. Uses running mean and variance at inference. Best for CNNs with sufficient batch size.

```python
bn1d = nn.BatchNorm1d(64)
bn2d = nn.BatchNorm2d(32)
bn3d = nn.BatchNorm3d(16)

x_1d = torch.randn(32, 64)
x_2d = torch.randn(16, 32, 28, 28)

out_1d = bn1d(x_1d)
out_2d = bn2d(x_2d)
```

| Attribute | Description |
|-----------|-------------|
| running_mean | Exponential moving average of batch mean |
| running_var | Exponential moving average of batch variance |
| weight, bias | Learnable scale and shift (affine) |

### BatchNorm Parameters

```python
bn = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
bn_no_affine = nn.BatchNorm2d(64, affine=False)
```

### LayerNorm

**LayerNorm** normalizes across the feature dimension for each sample. Preferred for transformers and RNNs. Independent of batch size.

```python
ln = nn.LayerNorm(64)
x = torch.randn(32, 64)
out = ln(x)

ln_2d = nn.LayerNorm([32, 32])
x_2d = torch.randn(16, 64, 32, 32)
out_2d = ln_2d(x_2d)
```

### GroupNorm

**GroupNorm** divides channels into groups and normalizes within each group. Good when batch size is small.

```python
gn = nn.GroupNorm(num_groups=8, num_channels=32)
x = torch.randn(16, 32, 28, 28)
out = gn(x)
```

### InstanceNorm

**InstanceNorm** normalizes each channel independently for each sample. Used in style transfer and GANs.

```python
in2d = nn.InstanceNorm2d(32)
x = torch.randn(8, 32, 28, 28)
out = in2d(x)
```

### Normalization Comparison

| Layer | Normalizes Over | Use Case |
|-------|-----------------|----------|
| BatchNorm | Batch + spatial | CNNs, large batches |
| LayerNorm | Features | Transformers, RNNs |
| GroupNorm | Groups of channels | Small batches |
| InstanceNorm | Per sample, per channel | Style transfer, GANs |

### Functional Normalization

```python
import torch.nn.functional as F

out_ln = F.layer_norm(x, x.shape[1:])
out_gn = F.group_norm(x, num_groups=8)
```

### Conv Block with Normalization

```python
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='batch'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=(norm_type == 'none'))
        if norm_type == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == 'group':
            self.norm = nn.GroupNorm(8, out_channels)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels)
        else:
            self.norm = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))
```
