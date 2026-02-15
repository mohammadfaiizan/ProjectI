# Visualization, Saliency, and Interpretability

## Table of Contents

- [Activation Visualization](#activation-visualization)
- [Feature Visualization](#feature-visualization)
- [Saliency Maps](#saliency-maps)
- [Model Interpretability](#model-interpretability)

---

## Activation Visualization

**Activation visualization** extracts and inspects intermediate layer outputs using **hook-based extraction**. Track **activation statistics** (mean, std, sparsity) and detect **dead neurons** (channels with near-zero activation).

### Hook-based Extraction

Register forward hooks to capture activations from any layer:

```python
import torch
import torch.nn as nn

activations = {}

def make_hook(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
        module.register_forward_hook(make_hook(name))

model.eval()
with torch.no_grad():
    _ = model(input_tensor)

for name, act in activations.items():
    print(f"{name}: shape={act.shape}, mean={act.mean():.4f}")
```

### Activation Statistics

Compute per-layer statistics to understand activation behavior:

```python
def activation_stats(activation):
    if len(activation.shape) == 4:
        act_flat = activation.view(activation.size(0), activation.size(1), -1).mean(dim=2)
    else:
        act_flat = activation
    return {
        'mean': act_flat.mean().item(),
        'std': act_flat.std().item(),
        'max': act_flat.max().item(),
        'sparsity': (act_flat == 0).float().mean().item()
    }
```

### Dead Neuron Detection

Identify channels with consistently zero or near-zero activation:

```python
def detect_dead_neurons(activations, threshold=1e-6):
    dead = {}
    for name, act in activations.items():
        if len(act.shape) >= 2:
            channel_means = act.mean(dim=[0] + list(range(2, act.dim())))
            dead_channels = (channel_means.abs() < threshold).sum().item()
            if dead_channels > 0:
                dead[name] = dead_channels
    return dead
```

### Feature Map Visualization

Visualize convolutional feature maps as a grid:

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_feature_maps(activation, num_maps=16):
    if len(activation.shape) == 4:
        maps = activation[0].cpu()
        num_maps = min(num_maps, maps.size(0))
        cols = int(np.ceil(np.sqrt(num_maps)))
        rows = int(np.ceil(num_maps / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
        for i in range(num_maps):
            row, col = i // cols, i % cols
            axes[row, col].imshow(maps[i], cmap='viridis')
            axes[row, col].axis('off')
        plt.tight_layout()
        plt.show()
```

---

## Feature Visualization

**Feature visualization** finds inputs that **maximize activation** of specific neurons or channels. Concepts include **DeepDream-style** optimization and **filter visualization** of learned weights.

### Maximizing Activation

Optimize a random input to maximize a target channel's activation:

```python
def maximize_activation(model, layer_name, channel_idx, input_shape, num_iter=200, lr=0.1):
    model.eval()
    input_tensor = torch.randn(1, *input_shape, requires_grad=True)
    optimizer = torch.optim.Adam([input_tensor], lr=lr)

    target_act = None
    def hook(module, input, output):
        nonlocal target_act
        if len(output.shape) == 4:
            target_act = output[:, channel_idx].mean()
        else:
            target_act = output[:, channel_idx].mean()

    handle = dict(model.named_modules())[layer_name].register_forward_hook(hook)

    for i in range(num_iter):
        optimizer.zero_grad()
        _ = model(input_tensor)
        loss = -target_act
        loss.backward()
        optimizer.step()

    handle.remove()
    return input_tensor.detach()
```

### DeepDream Concepts

Apply gradient ascent on the input with respect to a layer's activation:

```python
def deepdream_step(model, input_tensor, layer_name, lr=0.01):
    input_tensor = input_tensor.clone().requires_grad_(True)
    target_act = None

    def hook(module, input, output):
        nonlocal target_act
        target_act = output

    handle = dict(model.named_modules())[layer_name].register_forward_hook(hook)
    _ = model(input_tensor)
    target_act.mean().backward()
    handle.remove()

    with torch.no_grad():
        input_tensor.data = input_tensor.data + lr * input_tensor.grad.sign()
    return input_tensor.detach()
```

### Filter Visualization

Visualize learned convolutional filters (weights):

```python
def visualize_filters(layer, num_filters=16):
    weights = layer.weight.data.cpu()
    weights_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    num_filters = min(num_filters, weights.size(0))
    cols = int(np.ceil(np.sqrt(num_filters)))
    rows = int(np.ceil(num_filters / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    for i in range(num_filters):
        row, col = i // cols, i % cols
        if weights.size(1) == 3:
            filter_img = weights_norm[i].permute(1, 2, 0)
        else:
            filter_img = weights_norm[i].mean(0)
        axes[row, col].imshow(filter_img, cmap='viridis')
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.show()
```

---

## Saliency Maps

**Saliency maps** highlight input regions that influence the model's prediction. Methods include **vanilla gradients**, **Grad-CAM**, **Guided Backpropagation**, and **Integrated Gradients**.

### Vanilla Gradients

Gradient of the target class score with respect to the input:

```python
def vanilla_saliency(model, input_tensor, target_class=None):
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    model.eval()
    output = model(input_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, target_class].backward()
    saliency = input_tensor.grad.data.abs()
    saliency, _ = torch.max(saliency, dim=1)
    return saliency.squeeze().cpu()
```

### Grad-CAM

**Gradient-weighted Class Activation Mapping** combines gradients with activations for class-discriminative localization:

```python
import torch.nn.functional as F

def gradcam(model, input_tensor, target_layer, target_class=None):
    gradients = None
    activations = None

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    target_module = dict(model.named_modules())[target_layer]
    h1 = target_module.register_forward_hook(forward_hook)
    h2 = target_module.register_backward_hook(backward_hook)

    output = model(input_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, target_class].backward()

    weights = gradients.mean(dim=[2, 3], keepdim=True)
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam / cam.max()

    h1.remove()
    h2.remove()
    return cam.squeeze().detach().cpu()
```

### Guided Backpropagation

Modify ReLU backward to only pass positive gradients, producing cleaner visualizations:

```python
def guided_backprop(model, input_tensor, target_class=None):
    def relu_hook(module, grad_in, grad_out):
        return (torch.clamp(grad_in[0], min=0.0),)

    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.register_backward_hook(relu_hook)

    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    output = model(input_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, target_class].backward()
    return input_tensor.grad.data.squeeze().cpu()
```

### Integrated Gradients

Integrate gradients along a path from baseline to input for attribution:

```python
def integrated_gradients(model, input_tensor, target_class=None, baseline=None, steps=50):
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    model.eval()
    alphas = torch.linspace(0, 1, steps).to(input_tensor.device)
    integrated_grads = torch.zeros_like(input_tensor)

    for alpha in alphas:
        interpolated = baseline + alpha * (input_tensor - baseline)
        interpolated.requires_grad_(True)
        output = model(interpolated)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        model.zero_grad()
        output[0, target_class].backward()
        integrated_grads += interpolated.grad.data

    integrated_grads /= steps
    integrated_grads *= (input_tensor - baseline)
    return integrated_grads.squeeze().cpu()
```

| Method | Pros | Cons |
|--------|------|------|
| Vanilla | Simple, fast | Noisy, less discriminative |
| Grad-CAM | Class-specific, smooth | Requires conv layer choice |
| Guided Backprop | Sharp edges | Can highlight irrelevant pixels |
| Integrated Gradients | Axiomatically grounded | Computationally expensive |

---

## Model Interpretability

**Model interpretability** aims to explain why a model made a prediction. Concepts include **SHAP** (Shapley values), **LIME** (local linear approximations), **attention weight analysis**, and **feature importance**.

### SHAP Concepts

**SHAP** (SHapley Additive exPlanations) assigns each feature a contribution to the prediction based on game-theoretic Shapley values. Use a background dataset and compute marginal contributions:

```python
def shap_sampling_approximation(model, input_tensor, background, target_class, n_samples=100):
    model.eval()
    baseline = background.mean(dim=0, keepdim=True)
    shap_values = torch.zeros_like(input_tensor)

    for _ in range(n_samples):
        mask = torch.rand_like(input_tensor) > 0.5
        masked_input = input_tensor * mask + baseline * (1 - mask)
        masked_input.requires_grad_(True)
        output = model(masked_input)
        model.zero_grad()
        output[0, target_class].backward()
        shap_values += (input_tensor - baseline) * masked_input.grad

    return (shap_values / n_samples).squeeze().cpu()
```

### LIME Concepts

**LIME** (Local Interpretable Model-agnostic Explanations) fits a simple linear model to perturbed samples around the instance:

```python
from sklearn.linear_model import LinearRegression

def lime_explain(model, image, num_samples=1000, num_features=100):
    model.eval()
    batch_size, channels, height, width = image.shape
    patch_h = height // int(np.sqrt(num_features))
    patch_w = width // int(np.sqrt(num_features))

    perturbations = []
    masks = []
    for _ in range(num_samples):
        mask = torch.randint(0, 2, (int(np.sqrt(num_features)), int(np.sqrt(num_features))))
        mask_resized = F.interpolate(
            mask.float().unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode='nearest'
        ).squeeze()
        perturbed = image.clone()
        for c in range(channels):
            perturbed[0, c] *= mask_resized
        perturbations.append(perturbed)
        masks.append(mask.flatten())

    perturbations = torch.stack(perturbations)
    masks = torch.stack(masks).numpy()

    with torch.no_grad():
        target_class = model(image).argmax(dim=1).item()
        preds = model(perturbations)
        preds = F.softmax(preds, dim=1)[:, target_class].cpu().numpy()

    lr = LinearRegression()
    lr.fit(masks, preds)
    importance = lr.coef_.reshape(int(np.sqrt(num_features)), int(np.sqrt(num_features)))
    return importance
```

### Attention Weight Analysis

For attention-based models, visualize attention weights to see what the model attends to:

```python
def extract_attention_weights(model, input_tensor):
    attention_maps = []
    def hook(module, input, output):
        if hasattr(output, 'attn_weights'):
            attention_maps.append(output.attn_weights.detach())
        elif isinstance(output, tuple) and len(output) > 1:
            attention_maps.append(output[1].detach())

    for name, module in model.named_modules():
        if 'attention' in name.lower():
            module.register_forward_hook(hook)

    _ = model(input_tensor)
    return attention_maps
```

### Feature Importance

Permutation importance and occlusion-based importance:

```python
def occlusion_importance(model, input_tensor, target_class, window_size=8):
    model.eval()
    with torch.no_grad():
        baseline = model(input_tensor)[0, target_class].item()

    _, _, height, width = input_tensor.shape
    importance = torch.zeros(height, width)

    for i in range(0, height - window_size + 1, window_size // 2):
        for j in range(0, width - window_size + 1, window_size // 2):
            occluded = input_tensor.clone()
            occluded[:, :, i:i+window_size, j:j+window_size] = 0
            with torch.no_grad():
                score = model(occluded)[0, target_class].item()
            importance[i:i+window_size, j:j+window_size] += baseline - score

    return importance.cpu()
```
