# Regularization, Gradients, and Batch Operations

## Table of Contents
1. [Regularization Overview](#regularization-overview)
2. [Dropout](#dropout)
3. [Weight Decay and Explicit Penalties](#weight-decay-and-explicit-penalties)
4. [Label Smoothing](#label-smoothing)
5. [Early Stopping](#early-stopping)
6. [Batch Operations](#batch-operations)
7. [Gradient Operations](#gradient-operations)
8. [Gradient Accumulation](#gradient-accumulation)
9. [Gradient Clipping](#gradient-clipping)
10. [Gradient Hooks](#gradient-hooks)

---

## Regularization Overview

**Regularization** constrains model capacity to prevent overfitting. It biases the model toward simpler solutions that generalize better to unseen data. PyTorch provides multiple mechanisms: dropout, weight decay, explicit penalties, label smoothing, and data augmentation.

---

## Dropout

### How Dropout Works

During training, dropout randomly sets a fraction `p` of input units to zero at each forward pass. This prevents co-adaptation of neurons and acts as an implicit ensemble. During evaluation, dropout is disabled and outputs are scaled by `(1 - p)` to maintain expected values (inverted dropout).

### Dropout Variants

| Module | Input Shape | Drops | Use Case |
|--------|-------------|-------|----------|
| `nn.Dropout(p)` | Any | Individual elements | Fully connected layers |
| `nn.Dropout1d(p)` | (N, C, L) | Entire channels | 1D convolutions |
| `nn.Dropout2d(p)` | (N, C, H, W) | Entire channels | 2D convolutions |
| `nn.Dropout3d(p)` | (N, C, D, H, W) | Entire channels | 3D convolutions |
| `nn.AlphaDropout(p)` | Any | Elements (preserves mean/var) | SELU networks |

```python
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 128, 3, padding=1)
        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.25)

    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.conv(x))))
```

### Train vs Eval Behavior

```python
model.train()   # dropout active
model.eval()    # dropout disabled
```

### Functional API

```python
import torch.nn.functional as F
output = F.dropout(input, p=0.5, training=self.training)
output = F.dropout2d(input, p=0.25, training=self.training)
```

---

## Weight Decay and Explicit Penalties

### Weight Decay via Optimizer

Weight decay adds a penalty proportional to the L2 norm of parameters directly in the optimizer update step. In `AdamW`, this is decoupled from the gradient.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)
```

### Explicit L2 Regularization

Manually add the L2 penalty to the loss. Equivalent to weight decay for SGD, but differs for adaptive optimizers.

```python
def l2_penalty(model, lambda_reg=1e-4):
    penalty = 0.0
    for param in model.parameters():
        if param.requires_grad:
            penalty += torch.norm(param, p=2) ** 2
    return lambda_reg * penalty

loss = criterion(output, target) + l2_penalty(model)
loss.backward()
```

### Explicit L1 Regularization

Encourages sparsity in parameters.

```python
def l1_penalty(model, lambda_reg=1e-5):
    penalty = 0.0
    for param in model.parameters():
        if param.requires_grad:
            penalty += torch.norm(param, p=1)
    return lambda_reg * penalty
```

### Elastic Net (L1 + L2)

```python
def elastic_net(model, l1_lambda=1e-5, l2_lambda=1e-4):
    l1 = sum(torch.norm(p, 1) for p in model.parameters() if p.requires_grad)
    l2 = sum(torch.norm(p, 2) ** 2 for p in model.parameters() if p.requires_grad)
    return l1_lambda * l1 + l2_lambda * l2
```

### Selective Weight Decay

Exclude bias and normalization parameters from weight decay:

```python
decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if 'bias' in name or 'norm' in name or 'bn' in name:
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = torch.optim.AdamW([
    {'params': decay_params, 'weight_decay': 0.01},
    {'params': no_decay_params, 'weight_decay': 0.0}
], lr=1e-3)
```

---

## Label Smoothing

Replaces hard one-hot targets with soft targets. Instead of target = [0, 0, 1, 0], use target = [0.033, 0.033, 0.9, 0.033] with smoothing=0.1.

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### Manual Implementation

```python
def label_smoothed_loss(logits, targets, num_classes, smoothing=0.1):
    confidence = 1.0 - smoothing
    smooth_value = smoothing / (num_classes - 1)
    one_hot = torch.full_like(logits, smooth_value)
    one_hot.scatter_(1, targets.unsqueeze(1), confidence)
    log_probs = F.log_softmax(logits, dim=1)
    return -(one_hot * log_probs).sum(dim=1).mean()
```

---

## Early Stopping

Monitors a validation metric and stops training when it stops improving.

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
```

---

## Batch Operations

### Batch Dimension Convention

PyTorch modules expect the **first dimension** to be the batch dimension.

| Data Type | Shape Convention |
|-----------|-----------------|
| Tabular | (N, features) |
| Images | (N, C, H, W) |
| Sequences | (N, seq_len, features) or (seq_len, N, features) |
| 3D data | (N, C, D, H, W) |

### Batch Statistics

```python
batch = torch.randn(32, 784)

batch_mean = batch.mean(dim=0)          # per-feature mean, shape (784,)
batch_std = batch.std(dim=0)            # per-feature std
sample_means = batch.mean(dim=1)        # per-sample mean, shape (32,)
```

### Constructing Batches

```python
samples = [torch.randn(3, 224, 224) for _ in range(32)]
batch = torch.stack(samples, dim=0)                    # (32, 3, 224, 224)

single = torch.randn(3, 224, 224)
batched = single.unsqueeze(0)                           # (1, 3, 224, 224)
```

### Batch Matrix Operations

```python
batch_A = torch.randn(10, 3, 4)
batch_B = torch.randn(10, 4, 5)
result = torch.bmm(batch_A, batch_B)                   # (10, 3, 5)
```

---

## Gradient Operations

### Basic Training Step

```python
optimizer.zero_grad()               # clear gradients
output = model(input_data)          # forward pass
loss = criterion(output, targets)   # compute loss
loss.backward()                     # backward pass (compute gradients)
optimizer.step()                    # update parameters
```

### Accessing Gradients

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm().item():.4f}")
```

### Zeroing Gradients

Gradients accumulate by default. Two ways to zero:

```python
optimizer.zero_grad()
optimizer.zero_grad(set_to_none=True)    # faster, sets .grad to None instead of zero tensor
```

### Disabling Gradients

```python
with torch.no_grad():
    output = model(input)            # no graph built, saves memory

with torch.inference_mode():
    output = model(input)            # stronger optimization, no grad tracking
```

### Selective Gradient Computation

```python
for param in model.backbone.parameters():
    param.requires_grad = False

x = torch.randn(5, requires_grad=True)
y = x.detach()                       # y has no gradient connection to x
```

---

## Gradient Accumulation

Simulates larger batch sizes by accumulating gradients over multiple forward-backward passes before stepping.

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, (data, target) in enumerate(dataloader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Effective batch size** = `actual_batch_size * accumulation_steps`. The loss is divided by `accumulation_steps` so the gradient magnitude matches that of a single large batch.

---

## Gradient Clipping

Prevents exploding gradients by limiting gradient magnitude.

### Clip by Norm

Scales all parameter gradients so their combined norm does not exceed `max_norm`.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Returns the total gradient norm before clipping, useful for monitoring.

```python
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
if total_norm > 1.0:
    print(f"Gradients clipped: {total_norm:.4f}")
```

### Clip by Value

Clamps each gradient element to [-clip_value, clip_value].

```python
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

### Usage in Training Loop

```python
for data, target in dataloader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

---

## Gradient Hooks

Hooks allow inspecting or modifying gradients during backpropagation.

### Tensor Hooks

```python
def print_grad(grad):
    print(f"Gradient shape: {grad.shape}, norm: {grad.norm():.4f}")
    return grad

x = torch.randn(3, 4, requires_grad=True)
handle = x.register_hook(print_grad)

y = (x ** 2).sum()
y.backward()

handle.remove()
```

### Modifying Gradients

Return a modified tensor from the hook to change the gradient.

```python
def scale_grad(grad):
    return grad * 0.1

handle = param.register_hook(scale_grad)
```

### Module Hooks

```python
def backward_hook(module, grad_input, grad_output):
    print(f"{module.__class__.__name__}: grad_output norm = {grad_output[0].norm():.4f}")

handle = model.layer.register_full_backward_hook(backward_hook)
```
