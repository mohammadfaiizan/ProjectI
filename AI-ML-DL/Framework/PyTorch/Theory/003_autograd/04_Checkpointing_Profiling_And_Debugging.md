# Checkpointing, Profiling, and Debugging

## Table of Contents

- [Gradient Checkpointing for Memory Savings](#gradient-checkpointing-for-memory-savings)
- [Autograd Profiler](#autograd-profiler)
- [Anomaly Detection (detect_anomaly)](#anomaly-detection-detect_anomaly)
- [Gradient Flow Visualization](#gradient-flow-visualization)
- [Vanishing/Exploding Gradient Detection](#vanishingexploding-gradient-detection)
- [Autograd Performance Tips](#autograd-performance-tips)

---

## Gradient Checkpointing for Memory Savings

**Gradient checkpointing** trades computation for memory. Instead of storing all intermediate activations for backward, activations are recomputed during the backward pass. Use **torch.utils.checkpoint** for memory-efficient training of deep networks.

```python
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

def basic_function(x):
    h1 = torch.relu(x)
    h2 = torch.tanh(h1)
    h3 = torch.sigmoid(h2)
    return h3

x = torch.randn(1000, 500, requires_grad=True)
y = checkpoint.checkpoint(basic_function, x)
loss = y.sum()
loss.backward()
```

**In neural network blocks:**

```python
class CheckpointedBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU()
        )

    def forward(self, x):
        return checkpoint.checkpoint(self.layers, x)
```

**Trade-offs:**
- Memory: Significant reduction (often 30-50%)
- Computation: Forward pass runs twice (once in forward, once in backward)
- Use for: Deep networks, limited GPU memory, large batch sizes

---

## Autograd Profiler

The **torch.profiler** provides detailed analysis of gradient computation time, memory usage, and bottlenecks.

```python
from torch.profiler import profile, record_function, ProfilerActivity

def profile_gradient_computation(model, input_data, target):
    criterion = nn.MSELoss()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True
    ) as prof:
        with record_function("forward_pass"):
            output = model(input_data)
            loss = criterion(output, target)
        with record_function("backward_pass"):
            loss.backward()
    return prof

prof = profile_gradient_computation(model, input_data, target)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

**Memory profiling:**

```python
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    output = model(input_data)
    loss = criterion(output, target)
    forward_memory = torch.cuda.memory_allocated() / 1e6
    loss.backward()
    peak_memory = torch.cuda.max_memory_allocated() / 1e6
```

---

## Anomaly Detection (detect_anomaly)

**torch.autograd.detect_anomaly()** helps debug NaN/Inf gradients by raising an error when an anomalous operation is detected during backward.

```python
torch.autograd.set_detect_anomaly(True)

try:
    x = torch.tensor([1.0, 0.0], requires_grad=True)
    y = torch.log(x)
    loss = y.sum()
    loss.backward()
except RuntimeError as e:
    print(f"Anomaly detected: {e}")

torch.autograd.set_detect_anomaly(False)
```

**Context manager (recommended):**

```python
with torch.autograd.detect_anomaly():
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
```

**Common anomaly sources:**
- Division by zero
- log of non-positive values
- sqrt of negative values
- Exponential overflow
- Poor weight initialization

---

## Gradient Flow Visualization

**Layer-wise gradient tracking:**

```python
class GradientFlowTracker:
    def __init__(self):
        self.gradient_history = {}

    def track_model(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.gradient_history[name] = []

    def record_gradients(self, model):
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.gradient_history[name].append(param.grad.norm().item())
```

**Plotting gradient flow:**

```python
import matplotlib.pyplot as plt

def plot_gradient_flow(gradient_history):
    plt.figure(figsize=(12, 8))
    for name, norms in gradient_history.items():
        if norms:
            plt.plot(norms, label=name)
    plt.xlabel('Training Step')
    plt.ylabel('Gradient Norm')
    plt.yscale('log')
    plt.legend()
    plt.show()
```

---

## Vanishing/Exploding Gradient Detection

**Vanishing gradients** - Gradients become very small in deep networks:

```python
def detect_vanishing_gradients(model, threshold=1e-6):
    vanishing_layers = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm < threshold:
                vanishing_layers.append((name, grad_norm))
    return vanishing_layers
```

**Exploding gradients** - Gradients become very large:

```python
def detect_exploding_gradients(model, threshold=10.0):
    exploding_layers = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > threshold:
                exploding_layers.append((name, grad_norm))
    return exploding_layers
```

**Total gradient norm:**

```python
total_norm = torch.norm(torch.stack([
    torch.norm(p.grad) for p in model.parameters() if p.grad is not None
]))
```

**Diagnostic thresholds:**

| Condition | Gradient Norm |
|-----------|---------------|
| Vanishing | < 1e-6 |
| Exploding | > 100 |
| Healthy | 1e-4 to 10 |

---

## Autograd Performance Tips

**Memory optimization:**
- Use `torch.no_grad()` for inference
- Use gradient checkpointing for deep networks
- Clear intermediate variables with `del`
- Call `optimizer.zero_grad()` promptly

**Computation optimization:**
- Minimize graph depth and complexity
- Use in-place operations where safe (avoid on `requires_grad` tensors)
- Ensure tensors are contiguous
- Use efficient activation functions (ReLU over Sigmoid for deep nets)

**Graph optimization:**

```python
def efficient_computation(x):
    result = x
    for _ in range(10):
        sincos = torch.sin(result) + torch.cos(result)
        result = sincos * 0.1 + result * 0.9
    return result
```

**Batch size optimization:** Find optimal batch size for memory constraints:

```python
def find_optimal_batch_size(model, sample_input, max_memory_mb):
    batch_size = 1
    while True:
        try:
            torch.cuda.reset_peak_memory_stats()
            batch = sample_input.unsqueeze(0).repeat(batch_size, 1)
            with torch.no_grad():
                model(batch)
            if torch.cuda.max_memory_allocated() / 1e6 < max_memory_mb:
                batch_size *= 2
            else:
                return batch_size // 2
        except RuntimeError:
            return batch_size // 2
```

**Best practices:**
- Profile before optimizing
- Use `torch.cuda.synchronize()` for accurate GPU timing
- Warm up before benchmarking
- Combine checkpointing with mixed precision for maximum memory savings
