# ONNX, Mobile, and Edge Deployment

## Table of Contents

1. [ONNX Export](#1-onnx-export)
2. [ONNX Runtime Inference](#2-onnx-runtime-inference)
3. [Mobile Deployment](#3-mobile-deployment)
4. [Edge Deployment](#4-edge-deployment)

---

## 1. ONNX Export

### 1.1 torch.onnx.export

The **torch.onnx.export** function converts PyTorch models to the Open Neural Network Exchange (ONNX) format for cross-platform deployment.

```python
import torch

model.eval()
example_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    example_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes=None
)
```

| Parameter | Description |
|-----------|-------------|
| export_params | Include model parameters in the file |
| opset_version | ONNX opset version (higher = more ops, check compatibility) |
| do_constant_folding | Optimize constant subgraphs |
| input_names | Names for input tensors |
| output_names | Names for output tensors |
| dynamic_axes | Allow dynamic dimensions (batch, sequence length) |

### 1.2 Dynamic Axes

For variable batch size or spatial dimensions:

```python
dynamic_axes = {
    'input': {0: 'batch_size', 2: 'height', 3: 'width'},
    'output': {0: 'batch_size'}
}

torch.onnx.export(
    model,
    example_input,
    "model_dynamic.onnx",
    dynamic_axes=dynamic_axes
)
```

### 1.3 Opset Version

Different opset versions support different operators. Common choices:

| Opset | PyTorch | Notes |
|-------|---------|-------|
| 11 | 1.6+ | Good baseline |
| 13 | 1.10+ | More operators |
| 17 | 2.0+ | Latest features |

### 1.4 Input and Output Names

Named inputs and outputs simplify inference and integration:

```python
torch.onnx.export(
    model,
    example_input,
    "model.onnx",
    input_names=['image'],
    output_names=['logits', 'aux_output']
)
```

### 1.5 Operator Support

Common supported operations: Conv2d, Linear, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Dropout, Flatten, Add, Mul, Concat, Reshape, Transpose, Softmax, Sigmoid, Tanh, GELU. LSTM and GRU have limitations. Custom or unsupported ops may require alternative implementations or custom ONNX operators.

### 1.6 Traced Model Export

Tracing before export can improve compatibility:

```python
traced_model = torch.jit.trace(model, example_input)
torch.onnx.export(traced_model, example_input, "model.onnx")
```

---

## 2. ONNX Runtime Inference

### 2.1 Loading and Running ONNX Models

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]

input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
outputs = session.run(output_names, {input_name: input_data})
```

### 2.2 Execution Providers

```python
session = ort.InferenceSession(
    "model.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

| Provider | Use Case |
|----------|----------|
| CPUExecutionProvider | CPU inference |
| CUDAExecutionProvider | NVIDIA GPU |
| TensorrtExecutionProvider | TensorRT on NVIDIA |
| CoreMLExecutionProvider | Apple devices |

### 2.3 Validation

Compare PyTorch and ONNX outputs:

```python
model.eval()
with torch.no_grad():
    pytorch_output = model(example_input).numpy()

onnx_output = session.run(None, {input_name: example_input.numpy()})[0]
max_diff = np.max(np.abs(pytorch_output - onnx_output))
assert max_diff < 1e-5
```

---

## 3. Mobile Deployment

### 3.1 PyTorch Mobile and ExecuTorch

**PyTorch Mobile** provides a lightweight runtime for Android and iOS. **ExecuTorch** is the next-generation mobile runtime with improved portability.

### 3.2 Model Optimization for Mobile

```python
model.eval()
scripted_model = torch.jit.trace(model, example_input)
mobile_model = torch.utils.mobile_optimizer.optimize_for_mobile(
    scripted_model,
    optimization_blocklist={"remove_dropout", "fuse_add_relu"}
)
mobile_model.save("model.ptl")
```

### 3.3 Lite Interpreter

The **lite interpreter** reduces binary size and startup time. Models saved with `_save_for_lite_interpreter` use this format:

```python
scripted_model._save_for_lite_interpreter("model.ptl")
```

### 3.4 Operator Selection

Mobile builds can exclude unused operators to reduce binary size. Use `optimization_blocklist` to avoid optimizations that cause issues on specific devices.

### 3.5 Mobile-Friendly Architectures

- Depthwise separable convolutions
- ReLU6 (better quantization on mobile)
- Efficient backbones (MobileNet, EfficientNet)
- Avoid: heavy RNNs, large attention, unsupported ops

```python
class MobileCompatibleCNN(nn.Module):
    def _make_depthwise_block(self, in_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True)
        )
```

### 3.6 Android and iOS Integration

**Android (Java):**

```java
import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;

Module model = LiteModuleLoader.load(modelPath);
Tensor inputTensor = Tensor.fromBlob(inputArray, inputShape);
IValue output = model.forward(IValue.from(inputTensor));
```

**iOS (Swift):** Use LibTorch-Lite and load the model from the app bundle.

---

## 4. Edge Deployment

### 4.1 Model Compression for Edge

Edge devices (Raspberry Pi, Jetson Nano, microcontrollers) have limited compute and memory. Compression techniques:

| Technique | Description |
|-----------|-------------|
| Quantization | INT8 weights and activations |
| Pruning | Remove redundant weights |
| Knowledge distillation | Train smaller student from teacher |
| Architecture search | Efficient networks (MobileNet, etc.) |

### 4.2 INT8 Quantization for Edge

```python
torch.backends.quantized.engine = 'qnnpack'
model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
prepared = torch.quantization.prepare(model)
# Calibrate with representative data
quantized = torch.quantization.convert(prepared)
```

### 4.3 Pruning for Edge

**Unstructured pruning** zeros individual weights:

```python
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        weights = module.weight.data.abs()
        threshold = torch.quantile(weights, sparsity)
        module.weight.data *= (weights > threshold).float()
```

**Structured pruning** removes entire channels or filters for real speedups.

### 4.4 Hardware-Specific Optimization

| Device | Strategy |
|--------|----------|
| Raspberry Pi 4 | Dynamic quantization, depthwise separable, <100MB model |
| Jetson Nano | GPU optimization, TensorRT, <200MB model |
| Mobile phone | qnnpack, mobile optimizer, <50MB model |
| Microcontroller | INT8, extreme pruning, <500KB model |

### 4.5 Latency Constraints

Target latency depends on the use case:

| Use Case | Target Latency |
|----------|-----------------|
| Real-time video | <33 ms (30 FPS) |
| Interactive UI | <100 ms |
| Batch processing | Seconds acceptable |

### 4.6 Edge Device Targets

```python
DEVICE_SPECS = {
    'raspberry_pi_4': {
        'memory_mb': 4096,
        'max_model_size_mb': 100,
        'target_inference_ms': 500
    },
    'jetson_nano': {
        'memory_mb': 4096,
        'gpu_memory_mb': 2048,
        'max_model_size_mb': 200,
        'target_inference_ms': 100
    },
    'mobile_phone': {
        'max_model_size_mb': 50,
        'target_inference_ms': 200
    }
}
```

### 4.7 Knowledge Distillation

Train a small student to mimic a large teacher:

```python
teacher.eval()
student.train()
for data, targets in train_loader:
    with torch.no_grad():
        teacher_logits = teacher(data)
    student_logits = student(data)
    soft_loss = KL_div(F.log_softmax(student_logits/T), F.softmax(teacher_logits/T))
    hard_loss = CrossEntropy(student_logits, targets)
    loss = alpha * soft_loss + (1 - alpha) * hard_loss
    loss.backward()
```

---

## Summary

| Topic | Key Takeaway |
|-------|--------------|
| ONNX | Use dynamic_axes for variable shapes; validate outputs; choose opset for target runtime |
| Mobile | TorchScript + mobile optimizer; use ReLU6 and efficient architectures; keep model <50MB |
| Edge | Quantization, pruning, distillation; match model size and latency to device specs |
