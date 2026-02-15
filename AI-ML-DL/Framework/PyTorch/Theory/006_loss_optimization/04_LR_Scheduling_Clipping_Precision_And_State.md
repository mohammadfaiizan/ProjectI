# LR Scheduling, Clipping, Precision, and State

## Table of Contents
1. [Overview](#overview)
2. [Learning Rate Schedulers](#learning-rate-schedulers)
3. [Custom and Chained Schedulers](#custom-and-chained-schedulers)
4. [Gradient Clipping](#gradient-clipping)
5. [Mixed Precision Training](#mixed-precision-training)
6. [Optimizer State Save and Load](#optimizer-state-save-and-load)
7. [Multi-Optimizer Training](#multi-optimizer-training)
8. [Debugging Optimization](#debugging-optimization)

---

## Overview

Learning rate scheduling, gradient clipping, and mixed precision training improve convergence and efficiency. Optimizer state management enables checkpointing and transfer learning. Multi-optimizer setups support GANs and multi-task models.

```python
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict
```

---

## Learning Rate Schedulers

### StepLR

Reduces LR by `gamma` every `step_size` epochs.

```python
optimizer = optim.Adam(model.parameters(), lr=0.1)
step_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
for epoch in range(100):
    train_one_epoch()
    step_scheduler.step()
```

### MultiStepLR

Reduces LR at specified milestones.

```python
multistep_scheduler = lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[30, 60, 90],
    gamma=0.2
)
```

### ExponentialLR

Exponential decay: \( \eta_t = \eta_0 \gamma^t \)

```python
exp_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
```

### CosineAnnealingLR

Smooth cosine decay to `eta_min`.

```python
cosine_scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=0.001
)
```

### CosineAnnealingWarmRestarts (SGDR)

Cosine annealing with periodic restarts.

```python
sgdr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,
    T_mult=2,
    eta_min=0.001
)
```

### ReduceLROnPlateau

Reduces LR when metric plateaus.

```python
plateau_scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    threshold=0.01,
    min_lr=0.001
)
for epoch in range(100):
    train_one_epoch()
    val_loss = validate()
    plateau_scheduler.step(val_loss)
```

### OneCycleLR

One cycle policy: warmup then cosine decay.

```python
onecycle_scheduler = lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,
    steps_per_epoch=len(train_loader),
    epochs=10,
    pct_start=0.3,
    anneal_strategy='cos',
    div_factor=25.0,
    final_div_factor=10000.0
)
for batch in train_loader:
    train_step()
    onecycle_scheduler.step()
```

### LinearLR

Linear decay from `start_factor` to `end_factor`.

```python
linear_scheduler = lr_scheduler.LinearLR(
    optimizer,
    start_factor=1.0,
    end_factor=0.1,
    total_iters=50
)
```

### SequentialLR

Switches between schedulers at milestones.

```python
constant_lr = lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=5)
exp_lr = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
step_lr = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
sequential_scheduler = lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[constant_lr, exp_lr, step_lr],
    milestones=[5, 15]
)
```

### LambdaLR

Custom function-based scheduling.

```python
def lr_lambda(epoch):
    if epoch < 10:
        return 1.0
    elif epoch < 20:
        return 0.1
    return 0.01

lambda_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
```

### Scheduler Summary

| Scheduler | Step Frequency | Metric-Based |
|-----------|----------------|--------------|
| StepLR | Epoch | No |
| MultiStepLR | Epoch | No |
| ExponentialLR | Epoch | No |
| CosineAnnealingLR | Epoch | No |
| CosineAnnealingWarmRestarts | Epoch | No |
| ReduceLROnPlateau | Epoch | Yes |
| OneCycleLR | Batch | No |
| LinearLR | Epoch | No |
| SequentialLR | Epoch | No |
| LambdaLR | Epoch | No |

---

## Custom and Chained Schedulers

### Custom Warmup + Cosine

```python
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=0.0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_epoch += 1
```

### ChainedScheduler

```python
linear_warmup = lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
cosine_annealing = lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=0.001)
chained_scheduler = lr_scheduler.ChainedScheduler([linear_warmup, cosine_annealing])
```

---

## Gradient Clipping

### clip_grad_norm_

Clips gradient norm (L2 by default). Preserves direction.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
```

| Parameter | Typical | Purpose |
|-----------|---------|---------|
| `max_norm` | 1.0 | Maximum gradient norm |
| `norm_type` | 2 | 1, 2, or inf |

```python
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### clip_grad_value_

Clips gradient values element-wise. Can change direction.

```python
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

### Adaptive Clipping

```python
class AdaptiveGradientClipper:
    def __init__(self, percentile=95, history_size=100):
        self.percentile = percentile
        self.history_size = history_size
        self.gradient_norms = []

    def clip_gradients(self, model):
        current_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                current_norm += param.grad.data.norm(2).item() ** 2
        current_norm = current_norm ** 0.5
        self.gradient_norms.append(current_norm)
        if len(self.gradient_norms) > self.history_size:
            self.gradient_norms.pop(0)
        if len(self.gradient_norms) >= 10:
            threshold = torch.tensor(self.gradient_norms).quantile(self.percentile / 100.0)
            if current_norm > threshold:
                clip_factor = threshold / current_norm
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data.mul_(clip_factor)
```

### When to Use Clipping

| Scenario | Method | Typical max_norm |
|----------|--------|------------------|
| RNNs/LSTMs | clip_grad_norm_ | 1.0-5.0 |
| Transformers | clip_grad_norm_ | 1.0-2.0 |
| GANs | clip_grad_norm_ | 0.1-1.0 |
| Exploding gradients | clip_grad_norm_ | 1.0 |

---

## Mixed Precision Training

### autocast and GradScaler

**autocast** selects FP16 for supported ops; **GradScaler** scales loss to avoid underflow.

```python
scaler = GradScaler()

def train_step(model, optimizer, input_data, targets, loss_fn):
    optimizer.zero_grad()
    with autocast():
        outputs = model(input_data)
        loss = loss_fn(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()
```

### GradScaler Configuration

```python
scaler = GradScaler(
    init_scale=2**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
    enabled=True
)
```

### autocast Contexts

```python
with autocast():
    output = model(input_data)

with autocast(dtype=torch.float16):
    output = model(input_data)

with autocast(enabled=False):
    output = model(input_data)
```

### Gradient Clipping with AMP

Unscale before clipping when using GradScaler.

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

### bfloat16

For newer GPUs (e.g., A100):

```python
with autocast(dtype=torch.bfloat16):
    output = model(input_data)
```

### Operations Under autocast

| Operation | Typical dtype |
|-----------|---------------|
| Linear, Conv2d | FP16 |
| BatchNorm, LayerNorm | FP32 |
| Softmax | FP32 |
| Loss (CrossEntropy, etc.) | FP32 |

---

## Optimizer State Save and Load

### Full Checkpoint

```python
def save_checkpoint(model, optimizer, epoch, loss, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
```

### Optimizer State Structure

```python
state_dict = optimizer.state_dict()
state_dict.keys()
state_dict['param_groups']
state_dict['state']
```

### State Surgery

```python
def modify_optimizer_lr(optimizer, new_lr):
    state_dict = optimizer.state_dict()
    for group in state_dict['param_groups']:
        group['lr'] = new_lr
    optimizer.load_state_dict(state_dict)

def reset_optimizer_state(optimizer, keep_param_groups=True):
    state_dict = optimizer.state_dict()
    state_dict['state'] = {}
    optimizer.load_state_dict(state_dict)
```

---

## Multi-Optimizer Training

### GAN-Style (Generator vs Discriminator)

```python
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

def train_discriminator(real_data):
    optimizer_d.zero_grad()
    real_loss = criterion(discriminator(real_data), real_labels)
    fake_data = generator(noise).detach()
    fake_loss = criterion(discriminator(fake_data), fake_labels)
    d_loss = real_loss + fake_loss
    d_loss.backward()
    optimizer_d.step()
    return d_loss.item()

def train_generator(batch_size):
    optimizer_g.zero_grad()
    fake_data = generator(noise)
    g_loss = criterion(discriminator(fake_data), real_labels)
    g_loss.backward()
    optimizer_g.step()
    return g_loss.item()
```

### Multi-Task

```python
param_groups = {
    'shared': list(model.shared_backbone.parameters()),
    'task_a': list(model.task_a_head.parameters()),
    'task_b': list(model.task_b_head.parameters())
}
optimizer = optim.Adam([
    {'params': param_groups['shared'], 'lr': 0.001},
    {'params': param_groups['task_a'], 'lr': 0.01},
    {'params': param_groups['task_b'], 'lr': 0.01}
])
```

### Multi-Optimizer Manager

```python
class MultiOptimizerManager:
    def __init__(self):
        self.optimizers = {}

    def add_optimizer(self, name, optimizer):
        self.optimizers[name] = optimizer

    def zero_grad(self):
        for opt in self.optimizers.values():
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers.values():
            opt.step()

    def save_state(self, filepath):
        state = {name: opt.state_dict() for name, opt in self.optimizers.items()}
        torch.save(state, filepath)
```

---

## Debugging Optimization

### Loss Landscapes

Monitor loss over training. Sudden spikes suggest LR too high or gradient explosion.

### Gradient Statistics

```python
def get_gradient_norm(model, norm_type=2):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    return total_norm ** (1.0 / norm_type)

def check_gradient_flow(model, input_data, target, loss_fn):
    model.zero_grad()
    output = model(input_data)
    loss = loss_fn(output, target)
    loss.backward()
    gradient_info = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradient_info[name] = {
                'norm': param.grad.norm().item(),
                'mean': param.grad.mean().item(),
                'std': param.grad.std().item()
            }
    return gradient_info
```

### Learning Rate Finder

```python
def find_optimal_learning_rate(model, train_loader_fn, loss_fn, min_lr=1e-6, max_lr=1e-1, num_iter=100):
    model_copy = type(model)()
    model_copy.load_state_dict(model.state_dict())
    lrs = torch.logspace(math.log10(min_lr), math.log10(max_lr), num_iter)
    losses = []
    optimizer = optim.SGD(model_copy.parameters(), lr=min_lr)
    for lr in lrs:
        for group in optimizer.param_groups:
            group['lr'] = lr.item()
        input_data, target = train_loader_fn()
        optimizer.zero_grad()
        loss = loss_fn(model_copy(input_data), target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if len(losses) > 10 and loss.item() > 4 * min(losses):
            break
    gradients = np.gradient(losses)
    best_idx = np.argmin(gradients)
    return lrs[best_idx].item(), losses
```

### Optimization Debugger

```python
class OptimizationDebugger:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_history = []
        self.gradient_norms = defaultdict(list)

    def record_step(self, loss):
        self.loss_history.append(loss)
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.gradient_norms[name].append(param.grad.norm().item())

    def diagnose(self):
        if any(np.isnan(l) or np.isinf(l) for l in self.loss_history[-10:]):
            print("CRITICAL: NaN or Inf in loss")
        if self.loss_history[-1] > self.loss_history[0] * 1.1:
            print("WARNING: Loss increasing")
        for name, norms in self.gradient_norms.items():
            if norms and norms[-1] > 10:
                print(f"WARNING: Large gradient in {name}: {norms[-1]}")
            if norms and norms[-1] < 1e-7:
                print(f"WARNING: Vanishing gradient in {name}: {norms[-1]}")
```

### Common Issues and Solutions

| Issue | Possible Cause | Solution |
|-------|----------------|----------|
| Loss not decreasing | LR too low, bad init | Increase LR, check init |
| Loss exploding | LR too high, gradient explosion | Reduce LR, clip gradients |
| NaN loss | Numerical instability | Check data, reduce LR, use AMP carefully |
| Oscillating loss | LR too high | Reduce LR, add momentum |
| Slow convergence | LR too low | Increase LR, try different optimizer |
