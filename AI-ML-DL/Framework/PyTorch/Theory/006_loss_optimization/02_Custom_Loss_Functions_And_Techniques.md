# Custom Loss Functions and Optimization Techniques

## Table of Contents
1. [Overview](#overview)
2. [Writing Custom Losses as Functions](#writing-custom-losses-as-functions)
3. [Custom Losses as nn.Module Subclasses](#custom-losses-as-nnmodule-subclasses)
4. [FocalLoss and DiceLoss](#focalloss-and-diceloss)
5. [Combined and Weighted Losses](#combined-and-weighted-losses)
6. [Loss Balancing Strategies](#loss-balancing-strategies)
7. [Optimization Techniques: Warmup and Cyclical Rates](#optimization-techniques-warmup-and-cyclical-rates)
8. [Stochastic Weight Averaging and EMA](#stochastic-weight-averaging-and-ema)
9. [Lookahead Optimizer Concepts](#lookahead-optimizer-concepts)

---

## Overview

Custom loss functions allow domain-specific optimization objectives. Optimization techniques like warmup, cyclical learning rates, and weight averaging improve convergence and generalization.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
```

---

## Writing Custom Losses as Functions

Function-based losses are simple and stateless. Ensure differentiability for gradient-based optimization.

```python
def custom_mae_loss(predictions, targets, reduction='mean'):
    loss = torch.abs(predictions - targets)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

def wing_loss(predictions, targets, omega=10.0, epsilon=2.0, reduction='mean'):
    diff = torch.abs(predictions - targets)
    c = omega - omega * math.log(1 + omega / epsilon)
    loss = torch.where(
        diff < omega,
        omega * torch.log(1 + diff / epsilon),
        diff - c
    )
    return loss.mean() if reduction == 'mean' else loss.sum()
```

---

## Custom Losses as nn.Module Subclasses

**nn.Module** subclasses support learnable parameters and integration with standard training loops.

```python
class BasicCustomLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, predictions, targets):
        loss = (predictions - targets) ** 2
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class WeightedMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, predictions, targets, weights=None):
        loss = (predictions - targets) ** 2
        if weights is not None:
            loss = loss * weights
        if self.reduction == 'mean':
            return loss.sum() / weights.sum() if weights is not None else loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
```

---

## FocalLoss and DiceLoss

### FocalLoss

**FocalLoss** down-weights easy examples to focus on hard ones. Formula: \( FL(p_t) = -\alpha(1-p_t)^\gamma \log(p_t) \)

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
```

### DiceLoss

**DiceLoss** for segmentation: \( D = 1 - \frac{2|X \cap Y|}{|X| + |Y|} \)

```python
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice
```

---

## Combined and Weighted Losses

### Multi-Component Loss

```python
class MultiComponentLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights or {'mse': 1.0, 'mae': 0.5, 'smooth_l1': 0.3}
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.smooth_l1_loss = nn.SmoothL1Loss()

    def forward(self, predictions, targets):
        mse = self.mse_loss(predictions, targets)
        mae = self.mae_loss(predictions, targets)
        smooth_l1 = self.smooth_l1_loss(predictions, targets)
        total_loss = (self.weights['mse'] * mse +
                     self.weights['mae'] * mae +
                     self.weights['smooth_l1'] * smooth_l1)
        return total_loss, {'mse': mse, 'mae': mae, 'smooth_l1': smooth_l1, 'total': total_loss}
```

### Curriculum and Self-Paced Loss

```python
class CurriculumLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.current_epoch = 0
        self.total_epochs = 100

    def set_epoch(self, epoch, total_epochs=100):
        self.current_epoch = epoch
        self.total_epochs = total_epochs

    def forward(self, predictions, targets):
        progress = min(self.current_epoch / self.total_epochs, 1.0)
        mae_loss = F.l1_loss(predictions, targets)
        mse_loss = F.mse_loss(predictions, targets)
        return (1 - progress) * mae_loss + progress * mse_loss
```

---

## Loss Balancing Strategies

### Uncertainty-Weighted Loss

Learnable precision (inverse variance) per loss component:

```python
class BalancedLoss(nn.Module):
    def __init__(self, num_losses, alpha=0.1):
        super().__init__()
        self.num_losses = num_losses
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, loss_components):
        balanced_loss = 0
        for i, loss in enumerate(loss_components):
            precision = torch.exp(-self.log_vars[i])
            balanced_loss += precision * loss + self.log_vars[i]
        return balanced_loss
```

### Pareto Loss

```python
class ParetoLoss(nn.Module):
    def __init__(self, objectives, alpha=0.5):
        super().__init__()
        self.objectives = objectives
        self.alpha = alpha

    def forward(self, predictions, targets):
        losses = [obj(predictions, targets) for obj in self.objectives]
        normalized = [l / (l.detach() + 1e-8) for l in losses]
        if len(normalized) == 2:
            return self.alpha * normalized[0] + (1 - self.alpha) * normalized[1]
        return sum(normalized) / len(normalized)
```

---

## Optimization Techniques: Warmup and Cyclical Rates

### Linear Warmup

```python
class LinearWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, target_lr=None):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.target_lrs = target_lr or self.base_lrs
        for group in optimizer.param_groups:
            group['lr'] = 0.0
        self.current_step = 0

    def step(self):
        if self.current_step < self.warmup_steps:
            for i, group in enumerate(self.optimizer.param_groups):
                lr = self.target_lrs[i] * (self.current_step + 1) / self.warmup_steps
                group['lr'] = lr
        self.current_step += 1
```

### Cosine Warmup

```python
class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, max_lr, min_lr=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        if self.current_step < self.warmup_steps:
            progress = self.current_step / self.warmup_steps
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 - math.cos(math.pi * progress))
            for group in self.optimizer.param_groups:
                group['lr'] = lr
        self.current_step += 1
```

### Cyclical Learning Rates

```python
cyclic_scheduler = optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=0.001,
    max_lr=0.01,
    step_size_up=5,
    mode='triangular',
    cycle_momentum=False
)
```

### One Cycle Policy

```python
onecycle_scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,
    steps_per_epoch=10,
    epochs=10,
    pct_start=0.3,
    anneal_strategy='cos',
    div_factor=25.0,
    final_div_factor=10000.0
)
```

---

## Stochastic Weight Averaging and EMA

### Stochastic Weight Averaging (SWA)

**SWA** averages model parameters over the final phase of training for better generalization.

```python
class SWAOptimizer:
    def __init__(self, optimizer, swa_start=5, swa_freq=1, swa_lr=0.01):
        self.optimizer = optimizer
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_lr = swa_lr
        self.step_count = 0
        self.swa_state = {}
        self.n_averaged = 0

    def step(self):
        self.optimizer.step()
        self.step_count += 1
        if self.step_count >= self.swa_start and (self.step_count - self.swa_start) % self.swa_freq == 0:
            self.update_swa()

    def update_swa(self):
        if not self.swa_state:
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    self.swa_state[p] = p.data.clone()
            self.n_averaged = 1
        else:
            self.n_averaged += 1
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    self.swa_state[p] = (self.swa_state[p] * (self.n_averaged - 1) + p.data) / self.n_averaged

    def swap_swa_sgd(self):
        if self.swa_state:
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    tmp = p.data.clone()
                    p.data = self.swa_state[p]
                    self.swa_state[p] = tmp
```

### Exponential Moving Average (EMA)

```python
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])
```

---

## Lookahead Optimizer Concepts

**Lookahead** maintains slow and fast weights. Every k steps, slow weights move toward fast weights.

```python
class LookaheadOptimizer:
    def __init__(self, base_optimizer, k=5, alpha=0.5):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        self.slow_weights = {}
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                self.slow_weights[p] = p.data.clone()

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def step(self):
        self.base_optimizer.step()
        self.step_count += 1
        if self.step_count % self.k == 0:
            self.update_slow_weights()

    def update_slow_weights(self):
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                self.slow_weights[p] = self.slow_weights[p] + self.alpha * (p.data - self.slow_weights[p])
                p.data = self.slow_weights[p]
```

### Technique Comparison

| Technique | When to Use |
|-----------|-------------|
| Warmup | Large batch training, high LR, transformers |
| Cyclical LR | Finding optimal LR, escaping local minima |
| One Cycle | Fast convergence, super-convergence |
| SWA | Final phase of training, better generalization |
| EMA | Stable evaluation, denoised parameters |
| Lookahead | Combine with any base optimizer for stability |
