# Loss Functions and Optimizers

## Table of Contents
1. [Loss Functions Overview](#loss-functions-overview)
2. [Classification Losses](#classification-losses)
3. [Regression Losses](#regression-losses)
4. [Ranking and Metric Learning Losses](#ranking-and-metric-learning-losses)
5. [Custom Loss Functions](#custom-loss-functions)
6. [Loss Reduction and Weighting](#loss-reduction-and-weighting)
7. [Optimizers Overview](#optimizers-overview)
8. [SGD and Variants](#sgd-and-variants)
9. [Adam Family](#adam-family)
10. [Other Optimizers](#other-optimizers)
11. [Parameter Groups](#parameter-groups)
12. [Learning Rate Scheduling](#learning-rate-scheduling)
13. [Optimizer State Management](#optimizer-state-management)

---

## Loss Functions Overview

A loss function (criterion) measures the discrepancy between model predictions and ground truth targets. The loss value is a scalar tensor that enables backpropagation through the computation graph.

All PyTorch loss functions live in `torch.nn` and have functional counterparts in `torch.nn.functional`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

---

## Classification Losses

### CrossEntropyLoss

Combines `LogSoftmax` and `NLLLoss` in one class. Expects **raw logits** (not softmax outputs) and integer class indices.

```python
criterion = nn.CrossEntropyLoss()
logits = torch.randn(32, 10)
targets = torch.randint(0, 10, (32,))
loss = criterion(logits, targets)
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `weight` | None | Per-class weights for imbalanced data |
| `ignore_index` | -100 | Index to exclude from loss computation |
| `reduction` | 'mean' | 'none', 'mean', or 'sum' |
| `label_smoothing` | 0.0 | Smoothing factor (0 to 1) |

```python
class_weights = torch.tensor([1.0, 2.0, 1.5, 1.0, 3.0])
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
```

### NLLLoss

Expects **log-probabilities** (output of `LogSoftmax`). Rarely used directly; prefer `CrossEntropyLoss`.

```python
log_probs = F.log_softmax(logits, dim=1)
loss = F.nll_loss(log_probs, targets)
```

### BCEWithLogitsLoss

Binary cross-entropy with built-in sigmoid. Numerically more stable than applying sigmoid separately.

```python
criterion = nn.BCEWithLogitsLoss()
logits = torch.randn(32, 1)
targets = torch.randint(0, 2, (32, 1)).float()
loss = criterion(logits, targets)
```

Supports `pos_weight` for imbalanced binary classification:

```python
pos_weight = torch.tensor([3.0])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### BCELoss

Requires inputs to be probabilities in [0, 1]. Use `BCEWithLogitsLoss` instead for stability.

```python
criterion = nn.BCELoss()
probs = torch.sigmoid(logits)
loss = criterion(probs, targets)
```

### Multi-Label Classification

For multi-label problems (multiple classes per sample), use `BCEWithLogitsLoss` with shape (N, C):

```python
criterion = nn.BCEWithLogitsLoss()
logits = torch.randn(32, 5)
targets = torch.tensor([[1, 0, 1, 0, 0]] * 32, dtype=torch.float)
loss = criterion(logits, targets)
```

---

## Regression Losses

### MSELoss (L2 Loss)

Mean squared error. Sensitive to outliers due to squaring.

```python
criterion = nn.MSELoss()
predictions = torch.randn(32, 1)
targets = torch.randn(32, 1)
loss = criterion(predictions, targets)
```

### L1Loss (MAE)

Mean absolute error. More robust to outliers than MSE.

```python
criterion = nn.L1Loss()
loss = criterion(predictions, targets)
```

### SmoothL1Loss (Huber Loss)

Behaves as L2 near zero and L1 for large errors. Combines benefits of both.

```python
criterion = nn.SmoothL1Loss(beta=1.0)
loss = criterion(predictions, targets)

loss_functional = F.huber_loss(predictions, targets, delta=1.0)
```

### Comparison

| Loss | Formula (per element) | Behavior near 0 | Behavior far from 0 | Outlier Sensitivity |
|------|----------------------|-----------------|---------------------|-------------------|
| MSELoss | (y - y_hat)^2 | Smooth, small gradient | Large gradient | High |
| L1Loss | \|y - y_hat\| | Non-smooth | Constant gradient | Low |
| SmoothL1 | Hybrid | Smooth (L2) | Linear (L1) | Low |

---

## Ranking and Metric Learning Losses

### TripletMarginLoss

Pushes anchor closer to positive and away from negative by at least `margin`.

```python
criterion = nn.TripletMarginLoss(margin=1.0, p=2)
anchor = torch.randn(32, 128)
positive = torch.randn(32, 128)
negative = torch.randn(32, 128)
loss = criterion(anchor, positive, negative)
```

### CosineEmbeddingLoss

Measures cosine similarity. Target is +1 (similar) or -1 (dissimilar).

```python
criterion = nn.CosineEmbeddingLoss(margin=0.0)
x1 = torch.randn(32, 128)
x2 = torch.randn(32, 128)
target = torch.ones(32)
loss = criterion(x1, x2, target)
```

### MarginRankingLoss

```python
criterion = nn.MarginRankingLoss(margin=0.0)
x1 = torch.randn(32)
x2 = torch.randn(32)
target = torch.ones(32)
loss = criterion(x1, x2, target)
```

### KLDivLoss

Kullback-Leibler divergence. Input must be **log-probabilities**, target must be probabilities.

```python
criterion = nn.KLDivLoss(reduction='batchmean')
log_probs = F.log_softmax(logits, dim=1)
target_probs = F.softmax(teacher_logits, dim=1)
loss = criterion(log_probs, target_probs)
```

---

## Custom Loss Functions

### As a Function

```python
def focal_loss(logits, targets, alpha=1.0, gamma=2.0):
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    loss = alpha * (1 - pt) ** gamma * ce_loss
    return loss.mean()
```

### As an nn.Module

```python
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        probs = torch.sigmoid(predictions)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice
```

### Combining Losses

```python
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.alpha = alpha

    def forward(self, predictions, targets):
        return self.alpha * self.ce(predictions, targets) + (1 - self.alpha) * self.dice(predictions, targets)
```

---

## Loss Reduction and Weighting

### Reduction Modes

| Mode | Behavior |
|------|----------|
| `'none'` | Returns per-element loss tensor |
| `'mean'` | Returns mean of all element losses |
| `'sum'` | Returns sum of all element losses |

```python
per_sample_loss = nn.MSELoss(reduction='none')(predictions, targets)
weighted_loss = (per_sample_loss * sample_weights).mean()
```

### Ignoring Indices

```python
criterion = nn.CrossEntropyLoss(ignore_index=-1)
```

---

## Optimizers Overview

Optimizers update model parameters using computed gradients. All optimizers share a common interface:

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)

optimizer.zero_grad()       # clear previous gradients
loss.backward()             # compute gradients
optimizer.step()            # update parameters
```

**zero_grad** must be called before each backward pass to prevent gradient accumulation across iterations (unless intentional).

---

## SGD and Variants

### Vanilla SGD

```python
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

### SGD with Momentum

Momentum accumulates past gradients to smooth updates and accelerate convergence.

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

### Nesterov Momentum

Computes gradient at the look-ahead position, typically converging faster.

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
```

### SGD with Weight Decay

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
```

---

## Adam Family

### Adam

Adaptive learning rates using first and second moment estimates. Good default choice.

```python
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
```

### AdamW

**Decoupled weight decay**. Correctly separates weight decay from the adaptive learning rate. Preferred over Adam when using weight decay.

```python
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

### Other Adam Variants

```python
optim.Adamax(model.parameters(), lr=0.002)
optim.NAdam(model.parameters(), lr=0.002)
optim.RAdam(model.parameters(), lr=0.001)
```

| Variant | Key Difference |
|---------|---------------|
| Adam | Original adaptive optimizer |
| AdamW | Decoupled weight decay |
| Adamax | Uses infinity norm instead of L2 |
| NAdam | Adam + Nesterov momentum |
| RAdam | Rectified Adam, auto-warmup |

---

## Other Optimizers

### RMSprop

Divides learning rate by running average of gradient magnitudes. Good for RNNs.

```python
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, momentum=0.0)
```

### Adagrad

Adapts learning rate per parameter based on cumulative squared gradients. Learning rate monotonically decreases.

```python
optimizer = optim.Adagrad(model.parameters(), lr=0.01)
```

### Adadelta

Extension of Adagrad that limits the window of accumulated gradients.

```python
optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9)
```

### LBFGS

Quasi-Newton method. Requires a closure that re-evaluates the model.

```python
optimizer = optim.LBFGS(model.parameters(), lr=1.0)

def closure():
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    return loss

optimizer.step(closure)
```

---

## Parameter Groups

Different parameters can have different learning rates and hyperparameters.

```python
optimizer = optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-4},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
], weight_decay=1e-5)
```

### Freezing with Parameter Groups

```python
for param in model.backbone.parameters():
    param.requires_grad = False

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
```

---

## Learning Rate Scheduling

Schedulers adjust the learning rate during training.

### Step-Based Schedulers

```python
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR

step_lr = StepLR(optimizer, step_size=30, gamma=0.1)
multi_step = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
exp_lr = ExponentialLR(optimizer, gamma=0.95)
```

### Cosine Annealing

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

cosine = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
warm_restarts = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
```

### Plateau-Based

Reduces LR when a metric stops improving.

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-7)

for epoch in range(num_epochs):
    train(...)
    val_loss = validate(...)
    plateau.step(val_loss)
```

### Linear Warmup + Decay

```python
from torch.optim.lr_scheduler import LinearLR, SequentialLR

warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)
decay = CosineAnnealingLR(optimizer, T_max=95)
scheduler = SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[5])
```

### OneCycleLR

Implements the 1cycle policy with warmup and annealing in a single scheduler.

```python
from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(optimizer, max_lr=0.01, total_steps=total_steps)

for batch in dataloader:
    train_step(...)
    scheduler.step()
```

### Usage Pattern

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()
    scheduler.step()
```

---

## Optimizer State Management

### Save and Load

```python
torch.save(optimizer.state_dict(), 'optimizer.pt')
optimizer.load_state_dict(torch.load('optimizer.pt'))
```

### Full Training Checkpoint

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pt')

checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
```

### Practical Guidelines

| Scenario | Recommended Optimizer | Typical LR |
|----------|----------------------|------------|
| Default starting point | AdamW | 1e-3 |
| Fine-tuning pretrained | AdamW + low LR | 1e-5 to 1e-4 |
| Large-scale vision | SGD + momentum + cosine | 0.1 |
| RNNs / LSTMs | RMSprop or Adam | 1e-3 |
| Transformers | AdamW + warmup + cosine | 1e-4 to 3e-4 |
