# Gradient Analysis and Debugging

## Table of Contents

- [Gradient Analysis](#gradient-analysis)
- [Model Debugging](#model-debugging)
- [Adversarial Examples](#adversarial-examples)

---

## Gradient Analysis

**Gradient analysis** helps diagnose training issues. Key techniques include **gradient flow visualization**, **vanishing/exploding gradient detection**, **gradient statistics per layer**, and **gradient histograms**.

### Gradient Flow Visualization

Track gradient magnitudes as they flow backward through the network:

```python
import torch
import torch.nn as nn
import numpy as np

layer_gradients = []
layer_names = []

def make_backward_hook(name):
    def hook(module, grad_input, grad_output):
        if grad_output[0] is not None:
            avg_grad = grad_output[0].abs().mean().item()
            layer_gradients.append(avg_grad)
            layer_names.append(name)
    return hook

for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
        module.register_backward_hook(make_backward_hook(name))

outputs = model(data)
loss = criterion(outputs, targets)
loss.backward()

import matplotlib.pyplot as plt
plt.bar(range(len(layer_gradients)), layer_gradients)
plt.yscale('log')
plt.xlabel('Layer (output to input)')
plt.ylabel('Average Gradient Magnitude')
plt.title('Gradient Flow')
plt.show()
```

### Vanishing and Exploding Gradient Detection

```python
def analyze_gradients(model, threshold_vanishing=1e-7, threshold_exploding=10.0):
    vanishing = []
    exploding = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            if norm < threshold_vanishing:
                vanishing.append((name, norm))
            elif norm > threshold_exploding:
                exploding.append((name, norm))
    return vanishing, exploding
```

### Gradient Statistics Per Layer

```python
def gradient_stats(model):
    stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.view(-1)
            stats[name] = {
                'norm': grad.norm().item(),
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'max': grad.max().item(),
                'min': grad.min().item(),
                'sparsity': (grad == 0).float().mean().item()
            }
    return stats
```

### Gradient Histograms

Collect gradients over multiple batches and plot distributions:

```python
gradient_history = {}

def collect_gradients(model, data_loader, criterion, num_batches=10):
    model.train()
    for name, param in model.named_parameters():
        gradient_history[name] = []

    for batch_idx, (data, targets) in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
        model.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradient_history[name].append(param.grad.clone().detach().cpu())
```

---

## Model Debugging

**Model debugging** involves **anomaly detection**, **NaN/Inf tracking**, **shape mismatch debugging**, **hook-based inspection**, and common error fixes.

### Anomaly Detection

**torch.autograd.detect_anomaly** raises an error when NaN or Inf is produced during backward:

```python
torch.autograd.set_detect_anomaly(True)

try:
    outputs = model(data)
    loss = criterion(outputs, targets)
    loss.backward()
except RuntimeError as e:
    print(f"Anomaly detected: {e}")

torch.autograd.set_detect_anomaly(False)
```

### NaN/Inf Tracking

```python
def check_nan_inf(tensor, name=""):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"NaN/Inf in {name}: nan={torch.isnan(tensor).any()}, inf={torch.isinf(tensor).any()}")

def debug_forward(model, input_tensor):
    for name, module in model.named_modules():
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                check_nan_inf(output, name)
        module.register_forward_hook(hook)
    _ = model(input_tensor)
```

### Shape Mismatch Debugging

Test model with various input shapes to isolate dimension errors:

```python
def test_input_shapes(model, input_shapes):
    model.eval()
    results = {}
    for shape in input_shapes:
        try:
            x = torch.randn(1, *shape)
            with torch.no_grad():
                out = model(x)
            results[str(shape)] = {'success': True, 'output_shape': list(out.shape)}
        except Exception as e:
            results[str(shape)] = {'success': False, 'error': str(e)}
    return results
```

### Hook-based Inspection

Store intermediate outputs for debugging:

```python
class DebugModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.debug_outputs = {}

    def forward(self, x):
        self.debug_outputs['input'] = x.shape
        for name, module in self.base_model.named_modules():
            if len(list(module.children())) == 0:
                def make_hook(n):
                    def hook(module, input, output):
                        self.debug_outputs[n] = output.shape if isinstance(output, torch.Tensor) else type(output)
                    return hook
                module.register_forward_hook(make_hook(name))
        out = self.base_model(x)
        self.debug_outputs['output'] = out.shape
        return out
```

### Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| size mismatch | Wrong layer dimensions | Use `nn.AdaptiveAvgPool2d(1)` before Linear, or compute spatial size correctly |
| CUDA out of memory | Batch too large | Reduce batch size, use gradient accumulation, `torch.cuda.empty_cache()` |
| element 0 does not require grad | Loss without grad | Ensure `loss.requires_grad` or use `loss.backward()` on scalar loss |
| Expected batch_size to match | Data/target mismatch | Check DataLoader returns same batch size for data and targets |
| NaN in loss | Numerical instability | Lower learning rate, gradient clipping, check for log(0) or div by zero |

---

## Adversarial Examples

**Adversarial examples** are inputs crafted to fool the model. Key methods include **FGSM**, **PGD**, **model robustness evaluation**, and **adversarial accuracy**.

### FGSM Attack

**Fast Gradient Sign Method** adds a small perturbation in the direction of the gradient sign:

```python
import torch.nn.functional as F

def fgsm_attack(model, images, labels, epsilon=0.03):
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    data_grad = images.grad.data
    sign_grad = data_grad.sign()
    perturbed = images + epsilon * sign_grad
    perturbed = torch.clamp(perturbed, 0, 1)
    return perturbed.detach()
```

### PGD (Projected Gradient Descent)

Iterative FGSM with projection to epsilon ball:

```python
def pgd_attack(model, images, labels, epsilon=0.03, alpha=0.01, num_iter=10):
    images = images.clone().detach()
    delta = torch.zeros_like(images).uniform_(-epsilon, epsilon)
    delta.requires_grad_(True)

    for _ in range(num_iter):
        outputs = model(images + delta)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        with torch.no_grad():
            delta.data = delta.data + alpha * delta.grad.sign()
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            delta.data = torch.clamp(images + delta.data, 0, 1) - images
        delta.grad.zero_()

    return (images + delta).detach()
```

### Model Robustness Evaluation

Evaluate accuracy under adversarial attacks at different epsilon values:

```python
def evaluate_robustness(model, data_loader, epsilons=[0.01, 0.03, 0.05, 0.1], num_batches=10):
    model.eval()
    results = {}
    for eps in epsilons:
        correct = 0
        total = 0
        for batch_idx, (data, targets) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            data, targets = data.cuda(), targets.cuda()
            adv_data = fgsm_attack(model, data, targets, epsilon=eps)
            with torch.no_grad():
                outputs = model(adv_data)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        results[eps] = correct / total

    return results
```

### Adversarial Accuracy

Report clean accuracy vs adversarial accuracy:

```python
def adversarial_accuracy(model, data_loader, epsilon=0.03):
    model.eval()
    clean_correct = 0
    adv_correct = 0
    total = 0

    for data, targets in data_loader:
        data, targets = data.cuda(), targets.cuda()
        with torch.no_grad():
            clean_preds = model(data).argmax(dim=1)
            clean_correct += (clean_preds == targets).sum().item()

        adv_data = fgsm_attack(model, data, targets, epsilon=epsilon)
        with torch.no_grad():
            adv_preds = model(adv_data).argmax(dim=1)
            adv_correct += (adv_preds == targets).sum().item()

        total += targets.size(0)

    return clean_correct / total, adv_correct / total
```

### Adversarial Training

Train on a mix of clean and adversarial examples to improve robustness:

```python
def adversarial_train_step(model, images, labels, optimizer, criterion, epsilon=0.03):
    model.train()
    adv_images = fgsm_attack(model, images, labels, epsilon=epsilon)
    combined_images = torch.cat([images, adv_images], dim=0)
    combined_labels = torch.cat([labels, labels], dim=0)

    optimizer.zero_grad()
    outputs = model(combined_images)
    loss = criterion(outputs, combined_labels)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        clean_acc = (model(images).argmax(1) == labels).float().mean().item()
        adv_acc = (model(adv_images).argmax(1) == labels).float().mean().item()

    return loss.item(), clean_acc, adv_acc
```
