# Normalization, Activations, and Initialization

## Table of Contents

- [Normalization Operations](#normalization-operations)
  - [Batch Normalization](#batch-normalization)
  - [Layer Normalization](#layer-normalization)
  - [Instance Normalization](#instance-normalization)
  - [Group Normalization](#group-normalization)
  - [Local Response Normalization](#local-response-normalization)
  - [Functional and Custom Normalization](#functional-and-custom-normalization)
- [Activation Functions](#activation-functions)
  - [Basic Activations](#basic-activations)
  - [Modern Activations](#modern-activations)
  - [Softmax and Output Activations](#softmax-and-output-activations)
  - [Activation Selection Guidelines](#activation-selection-guidelines)
- [Weight Initialization](#weight-initialization)
  - [Uniform and Normal Initialization](#uniform-and-normal-initialization)
  - [Xavier and Kaiming Initialization](#xavier-and-kaiming-initialization)
  - [Specialized Initialization](#specialized-initialization)
  - [Model-Wide Initialization](#model-wide-initialization)

---

## Normalization Operations

**Normalization** stabilizes training by controlling the distribution of activations. Different normalization techniques operate over different dimensions and are suited to different architectures.

### Batch Normalization

**BatchNorm** normalizes across the batch dimension for each channel. It computes mean and variance over the batch and applies affine transformation with learnable scale and shift.

| Variant | Input Shape | Normalization Axis |
|---------|------------|-------------------|
| BatchNorm1d | (N, C, L) | Batch + spatial |
| BatchNorm2d | (N, C, H, W) | Batch + spatial |
| BatchNorm3d | (N, C, D, H, W) | Batch + spatial |

```python
import torch
import torch.nn as nn

batch_norm_1d = nn.BatchNorm1d(num_features=128)
input_1d = torch.randn(32, 128)
output_1d = batch_norm_1d(input_1d)

batch_norm_2d = nn.BatchNorm2d(num_features=64)
input_2d = torch.randn(16, 64, 32, 32)
output_2d = batch_norm_2d(input_2d)

bn_momentum = nn.BatchNorm2d(32, momentum=0.1)
bn_no_track = nn.BatchNorm2d(32, track_running_stats=False)
```

Key parameters: **affine** (learnable gamma/beta), **momentum** (running stats update), **track_running_stats** (use batch vs running stats in eval).

---

### Layer Normalization

**LayerNorm** normalizes across the last specified dimensions, independent of batch size. Ideal for RNNs, Transformers, and small batches.

```python
layer_norm = nn.LayerNorm(normalized_shape=128)
input_ln = torch.randn(32, 10, 128)
output_ln = layer_norm(input_ln)

layer_norm_2d = nn.LayerNorm([64, 32])
input_ln_2d = torch.randn(16, 64, 32)
output_ln_2d = layer_norm_2d(input_ln_2d)
```

---

### Instance Normalization

**InstanceNorm** normalizes each sample independently for each channel. Common in style transfer and GANs.

```python
instance_norm_2d = nn.InstanceNorm2d(num_features=32)
input_in = torch.randn(4, 32, 64, 64)
output_in = instance_norm_2d(input_in)
```

---

### Group Normalization

**GroupNorm** divides channels into groups and normalizes within each group. Robust to batch size; use when batch size is small or varies.

```python
group_norm = nn.GroupNorm(num_groups=8, num_channels=32)
input_gn = torch.randn(4, 32, 64, 64)
output_gn = group_norm(input_gn)
```

---

### Local Response Normalization

**LRN** implements lateral inhibition across channels at each spatial location. Historically used in early CNNs.

```python
lrn = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
output_lrn = lrn(input_lrn)
```

---

### Functional and Custom Normalization

```python
import torch.nn.functional as F

output_func_bn = F.batch_norm(input, running_mean, running_var, weight, bias, training=True)
output_func_ln = F.layer_norm(input, normalized_shape=[128])
output_func_gn = F.group_norm(input, num_groups=8)

def manual_batch_norm(x, eps=1e-5):
    mean = x.mean(dim=0, keepdim=True)
    var = x.var(dim=0, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps)

def manual_layer_norm(x, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps)
```

**Weight Standardization** and **Spectral Normalization** are advanced techniques:

```python
spectral_norm_conv = nn.utils.spectral_norm(nn.Conv2d(32, 64, 3))
```

| Normalization | Use Case |
|---------------|----------|
| BatchNorm | CNNs with sufficient batch size |
| LayerNorm | RNNs, Transformers, small batches |
| InstanceNorm | Style transfer, GANs |
| GroupNorm | Small or variable batch sizes |

---

## Activation Functions

**Activation functions** introduce non-linearity, enabling networks to learn complex patterns.

### Basic Activations

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| ReLU | max(0, x) | [0, inf) | Default hidden layers |
| LeakyReLU | max(ax, x), a=0.01 | (-inf, inf) | Avoid dying ReLU |
| PReLU | max(ax, x), a learnable | (-inf, inf) | Per-channel slope |
| Sigmoid | 1/(1+e^-x) | (0, 1) | Output probability |
| Tanh | (e^x - e^-x)/(e^x + e^-x) | (-1, 1) | RNNs, zero-centered |

```python
import torch.nn as nn
import torch.nn.functional as F

relu = nn.ReLU()
relu_output = relu(x)

leaky_relu = nn.LeakyReLU(negative_slope=0.01)
prelu = nn.PReLU(num_parameters=100)

sigmoid = nn.Sigmoid()
tanh = nn.Tanh()

hardsigmoid = nn.Hardsigmoid()
hardtanh = nn.Hardtanh(min_val=-1, max_val=1)
```

---

### Modern Activations

| Function | Notes |
|----------|-------|
| GELU | Gaussian Error Linear Unit; common in Transformers |
| SiLU/Swish | x * sigmoid(x); smooth, self-gated |
| Mish | x * tanh(softplus(x)); self-regularizing |
| ELU | Smooth negative region; good for deep nets |
| SELU | Scaled ELU; self-normalizing networks |

```python
gelu = nn.GELU()
silu = nn.SiLU()
mish = nn.Mish()
elu = nn.ELU(alpha=1.0)
selu = nn.SELU()

gelu_output = F.gelu(x)
swish_output = F.silu(x)
```

---

### Softmax and Output Activations

```python
softmax = nn.Softmax(dim=1)
logsoftmax = nn.LogSoftmax(dim=1)

softmax_output = F.softmax(logits, dim=1)
logsoftmax_output = F.log_softmax(logits, dim=1)

gumbel_softmax = F.gumbel_softmax(logits, tau=1.0, hard=False, dim=1)
```

For classification: use **logits** with CrossEntropyLoss (includes LogSoftmax). Apply Softmax only when probabilities are needed for inference.

---

### Activation Selection Guidelines

| Layer Type | Recommended |
|------------|-------------|
| Hidden (CNN/MLP) | ReLU, GELU, SiLU |
| Hidden (Transformer) | GELU, SiLU |
| RNN | Tanh, GELU |
| Output (classification) | None (logits) or Softmax |
| Output (binary) | Sigmoid or BCEWithLogitsLoss |
| Output (regression) | None or Sigmoid/Tanh for bounded |

```python
class MLPWithActivations(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        return self.fc2(x)
```

---

## Weight Initialization

Proper **initialization** prevents vanishing/exploding gradients and speeds convergence.

### Uniform and Normal Initialization

```python
import torch.nn.init as init

init.uniform_(tensor, a=-0.1, b=0.1)
init.normal_(tensor, mean=0.0, std=0.02)
init.constant_(tensor, 0.1)
init.zeros_(tensor)
init.ones_(tensor)
init.eye_(tensor)
```

---

### Xavier and Kaiming Initialization

**Xavier (Glorot)** maintains variance across layers for tanh/sigmoid. **Kaiming (He)** is designed for ReLU.

| Method | Formula | Best For |
|--------|---------|----------|
| Xavier uniform | U(-a, a), a = sqrt(6/(fan_in+fan_out)) | Tanh, Sigmoid |
| Xavier normal | N(0, sqrt(2/(fan_in+fan_out))) | Tanh, Sigmoid |
| Kaiming uniform | U(-a, a), a = sqrt(6/fan_in) | ReLU |
| Kaiming normal | N(0, sqrt(2/fan_in)) | ReLU |

```python
init.xavier_uniform_(tensor, gain=1.0)
init.xavier_normal_(tensor, gain=1.0)
init.kaiming_uniform_(tensor, mode='fan_in', nonlinearity='relu')
init.kaiming_normal_(tensor, mode='fan_out', nonlinearity='relu')
```

Activation-specific gains:

```python
gains = {'linear': 1.0, 'sigmoid': 1.0, 'tanh': 5/3, 'relu': 2**0.5, 'selu': 3/4}
init.xavier_uniform_(tensor, gain=gains['relu'])
```

---

### Specialized Initialization

```python
init.orthogonal_(tensor, gain=1.0)
init.dirac_(conv_weight)

def init_lstm(lstm_layer):
    for name, param in lstm_layer.named_parameters():
        if 'weight_ih' in name:
            init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            init.orthogonal_(param)
        elif 'bias' in name:
            param.data.fill_(0)
            n = param.size(0)
            param.data[n//4:n//2].fill_(1)
```

---

### Model-Wide Initialization

```python
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        init.constant_(m.bias, 0)

model.apply(init_weights)
```

| Layer Type | Initialization |
|------------|----------------|
| Conv | Kaiming normal, ReLU |
| Linear | Xavier or Kaiming based on activation |
| LSTM input | Xavier uniform |
| LSTM recurrent | Orthogonal |
| BatchNorm | weight=1, bias=0 |
| Forget gate bias | 1 |
