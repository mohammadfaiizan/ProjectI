# Model Saving, Scripting, and Quantization

## Table of Contents

1. [Model Saving](#1-model-saving)
2. [TorchScript](#2-torchscript)
3. [Quantization](#3-quantization)

---

## 1. Model Saving

### 1.1 State Dict vs Full Model

PyTorch offers two primary approaches for persisting models: **state_dict** and **complete model** serialization.

**State Dict** saves only the learnable parameters (weights and biases) as a dictionary. This approach requires the model architecture to be available when loading. It provides flexibility for loading weights into different architectures and is the recommended approach for production.

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

torch.save(model.state_dict(), 'model_state.pth')

loaded_model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
loaded_model.load_state_dict(torch.load('model_state.pth'))
```

**Complete Model** serialization saves both architecture and weights using Python's pickle. This approach is simpler but ties the saved file to the exact class definition and can cause compatibility issues across PyTorch versions.

```python
torch.save(model, 'model_complete.pth')
loaded_model = torch.load('model_complete.pth')
```

| Approach | Pros | Cons |
|----------|------|------|
| state_dict | Flexible, version-independent, smaller files | Requires model class at load time |
| Complete model | Simple, self-contained | Pickle security risks, version coupling |

### 1.2 Checkpoint Format

Training checkpoints extend the state_dict format to include optimizer state, epoch, loss, and scheduler for resuming training.

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'scheduler_state_dict': scheduler.state_dict() if scheduler else None
}
torch.save(checkpoint, 'checkpoint_epoch_50.pth')

checkpoint = torch.load('checkpoint_epoch_50.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

### 1.3 map_location and Device Handling

The **map_location** parameter controls where tensors are loaded when the save and load environments differ (e.g., model trained on GPU, loaded on CPU).

```python
model = torch.load('model.pth', map_location='cpu')
model = torch.load('model.pth', map_location=torch.device('cuda:0'))
model = torch.load('model.pth', map_location={'cuda:0': 'cuda:1'})
```

### 1.4 Safe Loading and Weights Only

For loading only weights from a checkpoint (ignoring optimizer and other keys):

```python
checkpoint = torch.load('checkpoint.pth', map_location='cpu', weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

Use **weights_only=True** (PyTorch 1.13+) to avoid arbitrary code execution from untrusted pickle files.

### 1.5 Metadata and Checksums

Including metadata and checksums improves traceability and integrity verification:

```python
import hashlib
import datetime

save_data = {
    'state_dict': model.state_dict(),
    'model_class': model.__class__.__name__,
    'timestamp': datetime.datetime.now().isoformat(),
    'num_classes': getattr(model, 'num_classes', None),
    'checksum': hashlib.md5(str(model.state_dict()).encode()).hexdigest()
}
torch.save(save_data, 'model_with_metadata.pth')
```

---

## 2. TorchScript

TorchScript is PyTorch's solution for deploying models in non-Python environments (C++, mobile, production servers) by compiling models to an intermediate representation.

### 2.1 torch.jit.trace vs torch.jit.script

**torch.jit.trace** records the operations executed during a forward pass with example inputs. It produces a fixed computational graph and is ideal for models without control flow.

```python
model.eval()
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)
traced_model.save('traced_model.pt')
```

**torch.jit.script** compiles the model by analyzing the Python source code. It supports control flow (if/else, loops) and dynamic behavior.

```python
scripted_model = torch.jit.script(model)
scripted_model.save('scripted_model.pt')
```

| Method | Control Flow | Dynamic Shapes | Use Case |
|--------|--------------|----------------|----------|
| trace | No | Fixed by example | Static models, CNNs |
| script | Yes | Limited | RNNs, conditionals |

### 2.2 Scripting Modules with Control Flow

Models with conditionals require scripting:

```python
class ConditionalModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.classifier = nn.Linear(64, num_classes)
        self.aux_classifier = nn.Linear(64, num_classes)

    def forward(self, x, use_aux=False):
        features = self.backbone(x)
        if use_aux:
            return self.aux_classifier(features)
        return self.classifier(features)

scripted = torch.jit.script(model)
```

### 2.3 Saving and Loading ScriptModules

```python
scripted_model.save('model.pt')
loaded_model = torch.jit.load('model.pt', map_location='cpu')
loaded_model.eval()
```

### 2.4 TorchScript Limitations

- Python built-ins like `len()`, `range()` on tensors require `torch.jit.annotate` or specific patterns
- Some NumPy operations are not supported
- Dynamic shapes may require `torch.jit.script_if_tracing` for hybrid scenarios
- Mutable Python containers in forward() can cause issues

### 2.5 Hybrid Compilation and Optimization

```python
try:
    scripted_model = torch.jit.trace(model, example_input)
except Exception:
    scripted_model = torch.jit.script(model)

scripted_model = torch.jit.freeze(scripted_model)
scripted_model = torch.jit.optimize_for_inference(scripted_model)
mobile_model = torch.utils.mobile_optimizer.optimize_for_mobile(scripted_model)
```

---

## 3. Quantization

Quantization reduces model size and accelerates inference by using lower-precision arithmetic (typically INT8 instead of FP32).

### 3.1 Dynamic Quantization

**Dynamic quantization** quantizes weights at load time and computes activations in floating point. No calibration data is required. Best for Linear and LSTM layers.

```python
import torch.quantization as quantization

model.eval()
quantized_model = quantization.quantize_dynamic(
    model,
    {nn.Linear, nn.Conv2d},
    dtype=torch.qint8
)
```

| Parameter | Description |
|-----------|-------------|
| model | Model to quantize |
| qconfig_spec | Set of module types to quantize |
| dtype | torch.qint8 or torch.float16 |

### 3.2 Static Quantization with Calibration

**Static quantization** uses calibration data to determine activation scale and zero-point, yielding better accuracy. Requires representative data.

```python
torch.backends.quantized.engine = 'fbgemm'
model.qconfig = quantization.get_default_qconfig('fbgemm')
prepared_model = quantization.prepare(model)

with torch.no_grad():
    for data, _ in calibration_loader:
        _ = prepared_model(data)

quantized_model = quantization.convert(prepared_model)
```

### 3.3 Quantization-Aware Training (QAT)

**QAT** simulates quantization during training so the model learns to compensate for quantization error. Best accuracy when post-training quantization degrades performance.

```python
model.train()
model.qconfig = quantization.get_default_qat_qconfig('fbgemm')
prepared_model = quantization.prepare_qat(model)

for epoch in range(epochs):
    for data, target in train_loader:
        output = prepared_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

model.eval()
quantized_model = quantization.convert(prepared_model)
```

### 3.4 Per-Tensor vs Per-Channel Quantization

- **Per-tensor**: One scale and zero-point per tensor. Simpler, less accurate.
- **Per-channel**: One scale and zero-point per channel (e.g., per output channel in Conv2d). Better accuracy, more computation.

```python
from torch.quantization import PerChannelMinMaxObserver, default_per_channel_weight_observer

qconfig = torch.quantization.QConfig(
    activation=torch.quantization.default_observer,
    weight=default_per_channel_weight_observer
)
```

### 3.5 Quantized Operators and Module Fusion

Fusing Conv-BN-ReLU before quantization improves performance:

```python
modules_to_fuse = [['conv', 'bn', 'relu']]
fused_model = quantization.fuse_modules(model, modules_to_fuse)
```

Quantizable modules include: `nn.Linear`, `nn.Conv2d`, `nn.Conv3d`, `nn.LSTM`, `nn.GRU`. Use **QuantStub** and **DeQuantStub** to define quantization boundaries:

```python
class QuantizableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()
        self.conv = nn.Conv2d(3, 64, 3)

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.dequant(x)
        return x
```

### 3.6 Supported Backends: fbgemm and qnnpack

| Backend | Platform | Use Case |
|---------|----------|----------|
| fbgemm | x86 CPU | Server, desktop |
| qnnpack | ARM CPU | Mobile, edge |

```python
torch.backends.quantized.engine = 'fbgemm'
torch.backends.quantized.engine = 'qnnpack'
```

---

## Summary

| Topic | Key Takeaway |
|-------|--------------|
| Model Saving | Prefer state_dict for flexibility; use map_location for device portability |
| TorchScript | Use trace for static models, script for control flow; freeze and optimize for inference |
| Quantization | Dynamic for quick wins; static with calibration for accuracy; QAT when accuracy matters most |
