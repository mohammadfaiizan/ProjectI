# SGD, Adam, and Advanced Optimizers

## Table of Contents
1. [Overview](#overview)
2. [SGD: Vanilla, Momentum, Nesterov, Weight Decay](#sgd-vanilla-momentum-nesterov-weight-decay)
3. [Adam Family: Adam, AdamW, Adamax, NAdam, RAdam](#adam-family-adam-adamw-adamax-nadam-radam)
4. [Adam Family Comparison Table](#adam-family-comparison-table)
5. [RMSprop, Adagrad, Adadelta, LBFGS](#rmsprop-adagrad-adadelta-lbfgs)
6. [Writing Custom Optimizers](#writing-custom-optimizers)
7. [Parameter Groups and Differential Learning Rates](#parameter-groups-and-differential-learning-rates)

---

## Overview

Optimizers update model parameters using gradients. PyTorch provides first-order (SGD, Adam) and second-order (LBFGS) methods. Choice depends on problem structure, data scale, and hardware.

```python
import torch
import torch.nn as nn
import torch.optim as optim
```

---

## SGD: Vanilla, Momentum, Nesterov, Weight Decay

### Basic SGD

**Vanilla SGD** update: \( \theta_{t+1} = \theta_t - \eta \nabla L(\theta_t) \)

```python
basic_sgd = optim.SGD(model.parameters(), lr=0.01)
```

### SGD with Momentum

**Momentum** accumulates gradient history: \( v_t = \mu v_{t-1} + g_t \), \( \theta_{t+1} = \theta_t - \eta v_t \)

```python
momentum_sgd = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)
```

| Parameter | Typical | Purpose |
|-----------|---------|---------|
| `lr` | 0.01-0.1 | Step size |
| `momentum` | 0.9 | Dampening for oscillations |

### Nesterov Accelerated Gradient (NAG)

**Nesterov** uses look-ahead gradient: update uses gradient at \( \theta_t + \mu v_{t-1} \).

```python
nesterov_sgd = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    nesterov=True
)
```

### SGD with Weight Decay

**Weight decay** adds L2 regularization: \( g_t = g_t + \lambda \theta_t \)

```python
weight_decay_sgd = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4
)
```

| Parameter | Typical | Purpose |
|-----------|---------|---------|
| `weight_decay` | 1e-4 to 1e-2 | L2 regularization strength |

---

## Adam Family: Adam, AdamW, Adamax, NAdam, RAdam

### Adam (Adaptive Moment Estimation)

**Adam** maintains first and second moment estimates with bias correction.

```python
adam_optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0
)
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `lr` | 0.001 | Step size |
| `betas` | (0.9, 0.999) | Decay for first and second moments |
| `eps` | 1e-8 | Numerical stability |
| `weight_decay` | 0 | L2 (coupled with gradient in Adam) |
| `amsgrad` | False | Use max of second moment |

### AdamW (Decoupled Weight Decay)

**AdamW** applies weight decay separately from gradient scaling (decoupled).

```python
adamw_optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    amsgrad=False
)
```

### Adamax

**Adamax** uses infinity norm for second moment. Often allows higher learning rates.

```python
adamax_optimizer = optim.Adamax(
    model.parameters(),
    lr=0.002,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0
)
```

### NAdam (Nesterov Adam)

**NAdam** incorporates Nesterov momentum into Adam.

```python
nadam_optimizer = optim.NAdam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0
)
```

### RAdam (Rectified Adam)

**RAdam** rectifies variance of adaptive learning rate in early training.

```python
radam_optimizer = optim.RAdam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0
)
```

---

## Adam Family Comparison Table

| Optimizer | Weight Decay | Warmup | Best For |
|-----------|--------------|--------|----------|
| Adam | Coupled | Often needed | General purpose |
| AdamW | Decoupled | Less critical | Transformers, when WD matters |
| Adamax | Optional | - | Sparse, high-dim |
| NAdam | Optional | - | Faster convergence |
| RAdam | Optional | Built-in | Early training stability |

---

## RMSprop, Adagrad, Adadelta, LBFGS

### RMSprop

**RMSprop** divides gradient by root mean square of recent gradients.

```python
rmsprop_optimizer = optim.RMSprop(
    model.parameters(),
    lr=0.01,
    alpha=0.99,
    eps=1e-8,
    weight_decay=0,
    momentum=0,
    centered=False
)
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `lr` | 0.01 | Step size |
| `alpha` | 0.99 | Decay for squared gradient |
| `momentum` | 0 | Momentum |
| `centered` | False | Use centered variance |

### Adagrad

**Adagrad** adapts learning rate per parameter based on gradient history.

```python
adagrad_optimizer = optim.Adagrad(
    model.parameters(),
    lr=0.01,
    lr_decay=0,
    weight_decay=0,
    eps=1e-10
)
```

### Adadelta

**Adadelta** extends Adagrad without explicit learning rate decay.

```python
adadelta_optimizer = optim.Adadelta(
    model.parameters(),
    lr=1.0,
    rho=0.9,
    eps=1e-6,
    weight_decay=0
)
```

### LBFGS

**LBFGS** is a quasi-Newton method. Requires a closure that returns the loss.

```python
lbfgs_optimizer = optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=20,
    max_eval=None,
    tolerance_grad=1e-7,
    tolerance_change=1e-9,
    history_size=100,
    line_search_fn='strong_wolfe'
)

def closure():
    lbfgs_optimizer.zero_grad()
    output = model(input_data)
    loss = loss_fn(output, target)
    loss.backward()
    return loss

lbfgs_optimizer.step(closure)
```

| Parameter | Typical | Purpose |
|-----------|---------|---------|
| `lr` | 1.0 | Step size (line search may override) |
| `max_iter` | 20 | Iterations per step |
| `line_search_fn` | 'strong_wolfe' | Line search strategy |

---

## Writing Custom Optimizers

Subclass `torch.optim.Optimizer` and implement `step()`.

### Basic Template

```python
from torch.optim.optimizer import Optimizer

class BasicOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, **kwargs):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, **kwargs)
        super(BasicOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                p.data.add_(grad, alpha=-group['lr'])
        return loss
```

### Custom SGD with Momentum

```python
class CustomSGDMomentum(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                       dampening=0, nesterov=nesterov)
        super(CustomSGDMomentum, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            nesterov = group['nesterov']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(grad)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(grad)
                    grad = grad.add(buf, alpha=momentum) if nesterov else buf
                p.data.add_(grad, alpha=-group['lr'])
        return loss
```

### Custom Adam

```python
class CustomAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(CustomAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients')
                amsgrad = group['amsgrad']
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    torch.maximum(state['max_exp_avg_sq'], exp_avg_sq, out=state['max_exp_avg_sq'])
                    denom = (state['max_exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        return loss
```

---

## Parameter Groups and Differential Learning Rates

**Parameter groups** allow different hyperparameters (e.g., learning rate) per layer.

```python
param_groups = [
    {'params': model.linear1.parameters(), 'lr': 0.01, 'weight_decay': 1e-4},
    {'params': model.linear2.parameters(), 'lr': 0.001, 'weight_decay': 1e-3}
]
optimizer = optim.Adam(param_groups)
```

### Transfer Learning Example

```python
def create_param_groups(model):
    return [
        {'params': model.backbone.parameters(), 'lr': 0.0001},
        {'params': model.head.parameters(), 'lr': 0.001}
    ]
optimizer = optim.AdamW(create_param_groups(model), weight_decay=0.01)
```

### Per-Layer Configuration

```python
param_groups = []
for name, param in model.named_parameters():
    if 'linear1' in name:
        param_groups.append({'params': [param], 'lr': 0.001})
    elif 'linear2' in name:
        param_groups.append({'params': [param], 'lr': 0.0005})
    else:
        param_groups.append({'params': [param], 'lr': 0.0001})
optimizer = optim.Adam(param_groups)
```
