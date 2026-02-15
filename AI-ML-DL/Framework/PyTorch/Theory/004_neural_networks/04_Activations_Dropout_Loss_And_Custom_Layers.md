# Activations, Dropout, Loss, and Custom Layers

## Table of Contents

- [Activation Functions](#activation-functions)
- [Dropout Regularization](#dropout-regularization)
- [Loss Modules](#loss-modules)
- [Model Composition](#model-composition)
- [Writing Custom Layers](#writing-custom-layers)

---

## Activation Functions

Activation functions introduce **non-linearity** into neural networks. PyTorch provides both module (`nn.ReLU`) and functional (`F.relu`) interfaces.

### ReLU and Variants

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

relu = nn.ReLU()
relu_inplace = nn.ReLU(inplace=True)
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(relu(x))
print(F.relu(x))

leaky_relu = nn.LeakyReLU(negative_slope=0.01)
prelu = nn.PReLU(num_parameters=1)
relu6 = nn.ReLU6()
```

### ELU and SELU

```python
elu = nn.ELU(alpha=1.0)
selu = nn.SELU()
```

### GELU and SiLU (Swish)

```python
gelu = nn.GELU()
gelu_approx = nn.GELU(approximate='tanh')
silu = nn.SiLU()
```

### Sigmoid and Tanh

```python
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
hardtanh = nn.Hardtanh(min_val=-1.0, max_val=1.0)
softsign = nn.Softsign()
```

### Softmax and LogSoftmax

```python
softmax = nn.Softmax(dim=-1)
log_softmax = nn.LogSoftmax(dim=-1)
softmax_2d = nn.Softmax2d()
```

### Activation Summary

| Activation | Output Range | Use Case |
|------------|--------------|----------|
| ReLU | [0, inf) | Default for CNNs |
| LeakyReLU | (-inf, inf) | Avoids dead neurons |
| GELU | (-inf, inf) | Transformers |
| SiLU/Swish | (-inf, inf) | Modern architectures |
| Sigmoid | (0, 1) | Output, gating |
| Tanh | (-1, 1) | Centered activations |
| Softmax | (0, 1), sum=1 | Classification output |

---

## Dropout Regularization

Dropout randomly zeros activations during training to prevent overfitting. It is **inactive during evaluation**.

### nn.Dropout

Standard dropout for fully connected layers. Drops individual elements.

```python
dropout = nn.Dropout(p=0.5)
x = torch.randn(4, 10)
dropout.train()
out_train = dropout(x)
dropout.eval()
out_eval = dropout(x)
```

### nn.Dropout2d

**Spatial dropout** for convolutional feature maps. Drops entire channels.

```python
dropout2d = nn.Dropout2d(p=0.3)
x = torch.randn(2, 3, 5, 5)
dropout2d.train()
out = dropout2d(x)
```

### nn.Dropout3d

Drops entire 3D feature maps (channel-wise for volumetric data).

```python
dropout3d = nn.Dropout3d(p=0.2)
x = torch.randn(1, 2, 3, 4, 4)
out = dropout3d(x)
```

### nn.AlphaDropout

For **SELU** networks. Preserves self-normalizing properties (mean and variance).

```python
alpha_dropout = nn.AlphaDropout(p=0.1)
x = F.selu(torch.randn(4, 10))
out = alpha_dropout(x)
```

### Functional Dropout

```python
out = F.dropout(x, p=0.5, training=True)
out = F.dropout2d(x, p=0.3, training=True)
```

### Dropout in Networks

```python
class DropoutNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
```

---

## Loss Modules

Loss modules compute the objective for training. Use reduction modes: `'mean'`, `'sum'`, or `'none'`.

### Regression Losses

```python
mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()
smooth_l1 = nn.SmoothL1Loss()
huber_loss = nn.HuberLoss(delta=1.0)

pred = torch.randn(10, 1)
target = torch.randn(10, 1)
loss = mse_loss(pred, target)
```

### Classification Losses

```python
ce_loss = nn.CrossEntropyLoss()
nll_loss = nn.NLLLoss()
bce_loss = nn.BCELoss()
bce_logits_loss = nn.BCEWithLogitsLoss()

logits = torch.randn(4, 10)
targets = torch.randint(0, 10, (4,))
loss_ce = ce_loss(logits, targets)

log_probs = F.log_softmax(logits, dim=1)
loss_nll = nll_loss(log_probs, targets)

probs = torch.sigmoid(torch.randn(8, 1))
binary_targets = torch.randint(0, 2, (8, 1)).float()
loss_bce = bce_loss(probs, binary_targets)
```

### Reduction Modes

```python
loss_mean = nn.CrossEntropyLoss(reduction='mean')
loss_sum = nn.CrossEntropyLoss(reduction='sum')
loss_none = nn.CrossEntropyLoss(reduction='none')
```

### Advanced Losses

```python
kl_loss = nn.KLDivLoss(reduction='batchmean')
cosine_loss = nn.CosineEmbeddingLoss()
triplet_loss = nn.TripletMarginLoss(margin=1.0)
```

### Custom Loss: Focal Loss

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal
```

### Custom Loss: Dice Loss

```python
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice
```

---

## Model Composition

Model composition builds complex architectures from reusable blocks. Use **submodules**, **nesting**, and **multi-branch** designs.

### Building Blocks

```python
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)
```

### Composed Model

```python
class SimpleComposedModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = ConvBlock(3, 64, kernel_size=7)
        self.pool = nn.MaxPool2d(2)
        self.res_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(self.stem(x))
        x = self.res_blocks(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```

### Multi-Branch Architecture

```python
class MultiBranchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.branch_a = nn.Sequential(nn.Flatten(), nn.Linear(128 * 16, 10))
        self.branch_b = nn.Sequential(nn.Flatten(), nn.Linear(128 * 16, 1))

    def forward(self, x, branches=['a', 'b']):
        features = self.backbone(x)
        outputs = {}
        if 'a' in branches:
            outputs['classification'] = self.branch_a(features)
        if 'b' in branches:
            outputs['regression'] = self.branch_b(features)
        return outputs
```

### Model Ensembling

```python
class ModelEnsemble(nn.Module):
    def __init__(self, models, method='average'):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.method = method
        if method == 'weighted':
            self.weights = nn.Parameter(torch.ones(len(models)) / len(models))

    def forward(self, x):
        preds = [m(x) for m in self.models]
        if self.method == 'average':
            return torch.stack(preds).mean(dim=0)
        elif self.method == 'weighted':
            w = F.softmax(self.weights, dim=0)
            return sum(p * w[i] for i, p in enumerate(preds))
        return torch.stack(preds).mean(dim=0)
```

---

## Writing Custom Layers

Custom layers extend PyTorch by inheriting from **nn.Module**, registering **nn.Parameter** and **register_buffer**, and implementing **forward**.

### Basic Custom Layer

```python
class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'
```

### Custom Layer with Buffers

```python
class RunningStatsLayer(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(features, features))
        self.register_buffer('running_mean', torch.zeros(features))
        self.register_buffer('running_var', torch.ones(features))

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            momentum = 0.1
            self.running_mean.mul_(1 - momentum).add_(batch_mean, alpha=momentum)
            self.running_var.mul_(1 - momentum).add_(batch_var, alpha=momentum)
        normalized = (x - self.running_mean) / torch.sqrt(self.running_var + 1e-5)
        return torch.matmul(normalized, self.weight)
```

### Depthwise Separable Convolution

```python
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        return self.pointwise(x)
```

### Adaptive Convolution

```python
class AdaptiveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, max_kernel_size=7):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, k, padding=k//2)
            for k in range(1, max_kernel_size + 1, 2)
        ])
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, len(self.convs)),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        weights = self.gate(x)
        out = 0
        for i, conv in enumerate(self.convs):
            out = out + weights[:, i:i+1, None, None] * conv(x)
        return out
```

### Best Practices for Custom Layers

1. Always call `super().__init__()` first
2. Use `nn.Parameter` for learnable parameters
3. Use `register_buffer` for non-learnable state
4. Implement `reset_parameters()` for initialization
5. Implement `extra_repr()` for readable `str()`
6. Handle `self.training` when behavior differs in train/eval
7. Ensure gradients flow correctly through `forward`
