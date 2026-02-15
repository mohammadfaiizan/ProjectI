# Context Managers and Gradient Control

## Table of Contents

- [torch.no_grad](#torchno_grad)
- [torch.enable_grad](#torchenable_grad)
- [torch.set_grad_enabled](#torchset_grad_enabled)
- [torch.inference_mode](#torchinference_mode)
- [Gradient Accumulation Across Batches](#gradient-accumulation-across-batches)
- [Gradient Clipping](#gradient-clipping)

---

## torch.no_grad

The **torch.no_grad()** context manager disables gradient computation. All operations inside the block produce tensors with `requires_grad=False`, saving memory and computation.

**Use cases:** Inference, validation, evaluation when gradients are not needed.

```python
import torch
import torch.nn as nn

x = torch.tensor([1.0, 2.0], requires_grad=True)

y_normal = x**2
print(y_normal.requires_grad)

with torch.no_grad():
    y_no_grad = x**2
    print(y_no_grad.requires_grad)
```

**Model inference example:**

```python
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)

model.eval()
with torch.no_grad():
    output = model(input_data)
    print(output.requires_grad)
```

---

## torch.enable_grad

The **torch.enable_grad()** context manager re-enables gradient computation. It is useful when nested inside a `torch.no_grad()` block.

```python
with torch.no_grad():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = x**2
    print(y.requires_grad)

    with torch.enable_grad():
        z = x**3
        print(z.requires_grad)
        z.sum().backward()
        print(x.grad)
```

---

## torch.set_grad_enabled

The **torch.set_grad_enabled(mode)** context manager conditionally enables or disables gradients based on a boolean. Useful for conditional logic.

```python
def conditional_computation(x, training=True):
    with torch.set_grad_enabled(training):
        y = x**2 + 2*x + 1
        return y

x = torch.tensor([1.0, 2.0], requires_grad=True)
y_training = conditional_computation(x, training=True)
y_inference = conditional_computation(x, training=False)
```

**Training/validation pattern:**

```python
def validate_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    return total_loss / len(dataloader)
```

---

## torch.inference_mode

The **torch.inference_mode()** context manager provides more aggressive inference optimization than `no_grad()` (PyTorch 1.9+). It disables view tracking and other autograd overhead.

**Important:** Unlike `no_grad()`, you cannot enable gradients inside `inference_mode()`.

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)

with torch.inference_mode():
    y = x**2
    print(y.requires_grad)
```

**Context manager nesting:**

| Outer | Inner | Result |
|-------|-------|--------|
| no_grad | enable_grad | Gradients enabled |
| no_grad | no_grad | Gradients disabled |
| enable_grad | no_grad | Gradients disabled |
| inference_mode | enable_grad | Error |

---

## Gradient Accumulation Across Batches

**Gradient accumulation** allows simulating larger batch sizes by accumulating gradients over multiple small batches before updating weights.

**Key concepts:**
- Scale loss by `accumulation_steps` to maintain gradient magnitude
- Call `backward()` without `optimizer.step()` for each micro-batch
- Call `optimizer.step()` and `optimizer.zero_grad()` only after accumulation

```python
def train_with_accumulation(model, data_loader, optimizer, criterion, accumulation_steps=4):
    model.train()
    optimizer.zero_grad()

    for batch_idx, (data, target) in enumerate(data_loader):
        output = model(data)
        loss = criterion(output, target) / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

    if (batch_idx + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Memory usage:** Smaller batches reduce peak memory; larger effective batch size improves training stability.

```python
accumulation_steps = 4
batch_size = 8

optimizer.zero_grad()
for step in range(accumulation_steps):
    output = model(small_batch_data)
    loss = criterion(output, small_batch_target) / accumulation_steps
    loss.backward()

optimizer.step()
optimizer.zero_grad()
```

---

## Gradient Clipping

Gradient clipping prevents **exploding gradients** and training instability. Two main methods are available.

### clip_grad_norm_

**clip_grad_norm_** scales gradients when the total norm exceeds a threshold so the norm equals the target.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Returns** the total gradient norm before clipping. Useful for monitoring.

```python
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"Total gradient norm: {total_norm}")
optimizer.step()
```

**Common thresholds:**

| Architecture | Typical max_norm |
|--------------|------------------|
| Transformers | 1.0 - 5.0 |
| RNNs/LSTMs | 5.0 - 10.0 |
| CNNs | 1.0 - 2.0 |
| GANs (discriminator) | 0.01 - 0.1 |

### clip_grad_value_

**clip_grad_value_** clips element-wise gradient values to a range [-clip_value, clip_value].

```python
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

**Training loop with clipping:**

```python
def train_with_clipping(model, data_loader, optimizer, criterion, max_norm=1.0):
    model.train()
    for data, target in data_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
```

**Best practices:**
- Use `torch.no_grad()` for inference and validation
- Use `torch.inference_mode()` for pure inference when possible
- Scale loss by accumulation steps when using gradient accumulation
- Apply gradient clipping after accumulation, before `optimizer.step()`
- Use `clip_grad_norm_` for most cases; `clip_grad_value_` for element-wise control
