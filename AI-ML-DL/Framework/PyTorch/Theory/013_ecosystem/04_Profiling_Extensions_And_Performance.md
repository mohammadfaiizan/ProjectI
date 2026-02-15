# Profiling, Extensions, and Performance

## Table of Contents

1. [TensorBoard Advanced](#1-tensorboard-advanced)
2. [PyTorch Profiler Advanced](#2-pytorch-profiler-advanced)
3. [Custom C++ Extensions](#3-custom-c-extensions)
4. [Performance Optimization Tips](#4-performance-optimization-tips)

---

## 1. TensorBoard Advanced

TensorBoard provides rich visualization for training metrics, model structure, and embeddings.

### 1.1 Custom Scalars

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='./runs')

writer.add_scalar('loss/train', train_loss, step)
writer.add_scalar('loss/val', val_loss, step)

writer.add_scalars('loss/comparison', {
    'train': train_loss,
    'validation': val_loss
}, step)
```

### 1.2 Embedding Projector

Visualize high-dimensional embeddings in 2D/3D:

```python
writer.add_embedding(
    features,
    metadata=class_labels,
    label_img=images,
    tag='embeddings'
)
```

### 1.3 Mesh Visualization

```python
writer.add_mesh('mesh', vertices=vertices, faces=faces, global_step=step)
```

### 1.4 Hyperparameter Tuning Visualization

```python
writer.add_hparams(
    {'lr': 0.001, 'batch_size': 32},
    {'accuracy': 0.95, 'loss': 0.1}
)
```

### 1.5 Plugin System

TensorBoard supports custom plugins. The PyTorch Profiler plugin integrates trace visualization:

```bash
tensorboard --logdir=./profiling_logs
```

Navigate to the PyTorch Profiler tab for trace analysis.

### 1.6 Additional Logging Methods

```python
writer.add_histogram('weights/layer1', model.layer1.weight, step)
writer.add_image('samples', image_tensor, step, dataformats='CHW')
writer.add_images('batch', image_batch, step, dataformats='NCHW')
writer.add_figure('confusion_matrix', fig, step)
writer.add_text('summary', 'Training completed', step)
writer.add_pr_curve('pr_curve', labels, predictions, step)
writer.add_audio('audio', audio_tensor, step, sample_rate=22050)
writer.add_graph(model, input_sample)
```

---

## 2. PyTorch Profiler Advanced

The PyTorch Profiler identifies performance bottlenecks in training and inference.

### 2.1 Trace Analysis

```python
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True,
    on_trace_ready=tensorboard_trace_handler('./profiling_logs')
) as prof:
    for batch in dataloader:
        with record_function("forward"):
            output = model(batch)
        with record_function("backward"):
            loss.backward()
        optimizer.step()
```

### 2.2 Memory Profiling

```python
with profile(profile_memory=True) as prof:
    output = model(input_tensor)

print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))
print(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
```

### 2.3 GPU Kernel Analysis

Enable CUDA profiling to analyze kernel execution:

```python
with torch.profiler.profile(
    activities=[ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    _ = model(input_tensor)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### 2.4 Bottleneck Identification

```python
key_averages = prof.key_averages()
cpu_sorted = sorted(key_averages, key=lambda x: x.cpu_time_total, reverse=True)

for event in cpu_sorted[:10]:
    print(f"{event.key}: {event.cpu_time_total/1000:.2f}ms")
```

### 2.5 Flame Graphs

Export traces for flame graph visualization:

```python
prof.export_chrome_trace("trace.json")
```

Open in Chrome's `chrome://tracing` or use TensorBoard's Profiler plugin.

### 2.6 Scheduled Profiling

```python
from torch.profiler import schedule

my_schedule = schedule(
    wait=1,
    warmup=2,
    active=3,
    repeat=2
)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=my_schedule,
    on_trace_ready=tensorboard_trace_handler('./logs')
) as prof:
    for step, batch in enumerate(dataloader):
        train_step(batch)
        prof.step()
```

---

## 3. Custom C++ Extensions

PyTorch allows writing custom operators in C++ and CUDA for performance-critical code.

### 3.1 torch.utils.cpp_extension

**load** compiles and loads extensions at runtime (JIT):

```python
from torch.utils.cpp_extension import load

module = load(
    name='custom_ops',
    sources=['custom_ops.cpp'],
    extra_cflags=['-O3'],
    verbose=True
)

result = module.custom_add(tensor_a, tensor_b)
```

### 3.2 CppExtension

For build-time compilation:

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='custom_ops',
    ext_modules=[
        CppExtension(
            name='custom_ops',
            sources=['custom_ops.cpp'],
            extra_compile_args=['-O3']
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

### 3.3 CUDAExtension

```python
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_ops',
    ext_modules=[
        CUDAExtension(
            name='cuda_ops',
            sources=['cuda_ops.cpp', 'cuda_ops_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

### 3.4 Writing Custom Operators

**C++ with pybind11:**

```cpp
#include <torch/extension.h>

torch::Tensor custom_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.dtype() == b.dtype(), "Tensors must have same dtype");
    TORCH_CHECK(a.device() == b.device(), "Tensors must be on same device");
    return a + b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_add", &custom_add, "Custom addition");
}
```

### 3.5 JIT Compilation of Extensions

```python
from torch.utils.cpp_extension import load_inline

cpp_src = """
torch::Tensor add_forward(torch::Tensor a, torch::Tensor b) {
    return a + b;
}
"""

module = load_inline(
    name='inline_ops',
    cpp_sources=[cpp_src],
    functions=['add_forward']
)
```

### 3.6 pybind11 Bindings

```cpp
#include <pybind11/pybind11.h>
#include <torch/extension.h>

torch::Tensor relu_forward(torch::Tensor input) {
    return torch::clamp_min(input, 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu", &relu_forward, "ReLU activation");
}
```

---

## 4. Performance Optimization Tips

### 4.1 Memory Optimization

| Technique | Description |
|-----------|-------------|
| Gradient checkpointing | Trade compute for memory by recomputing activations |
| Mixed precision (AMP) | Use fp16 for forward/backward, fp32 for optimizer |
| Gradient accumulation | Simulate larger batch sizes with smaller memory |
| In-place operations | Use `relu_(x)` instead of `x = relu(x)` where safe |
| Empty cache | Call `torch.cuda.empty_cache()` between phases |

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 4.2 Computation Optimization

| Technique | Description |
|-----------|-------------|
| torch.compile | JIT compile models (PyTorch 2.0+) |
| TorchScript | Script or trace models for deployment |
| cuDNN benchmark | `torch.backends.cudnn.benchmark = True` |
| Fused operations | Use fused kernels (e.g., FusedAdam) |
| Vectorization | Avoid Python loops; use tensor ops |

```python
model = torch.jit.script(model)
model = torch.compile(model, mode="reduce-overhead")
```

### 4.3 I/O Optimization

| Technique | Description |
|-----------|-------------|
| num_workers | Use 4-8 DataLoader workers |
| pin_memory | Enable for GPU training |
| persistent_workers | Avoid worker restart overhead |
| prefetch_factor | Preload batches (default 2) |
| Efficient formats | HDF5, LMDB, WebDataset |

```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
    prefetch_factor=2
)
```

### 4.4 Distributed Optimization

| Technique | Description |
|-----------|-------------|
| Gradient compression | Reduce communication in distributed training |
| Overlap communication | Pipeline all-reduce with computation |
| Local SGD | Accumulate gradients before sync |
| Mixed precision | Reduce bandwidth with fp16 gradients |

### 4.5 Profiling-Guided Optimization

1. **Profile first**: Use PyTorch Profiler to identify bottlenecks
2. **Prioritize**: Focus on operations consuming most time
3. **Optimize**: Apply targeted optimizations
4. **Measure**: Verify improvements with benchmarks
5. **Iterate**: Repeat until targets are met

```python
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    train_one_epoch()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

### 4.6 Quick Optimization Checklist

- Use DataLoader with optimal `num_workers` and `pin_memory`
- Enable mixed precision training (AMP)
- Set `torch.backends.cudnn.benchmark = True` for fixed input sizes
- Use `torch.compile()` or TorchScript for inference
- Profile to identify and eliminate bottlenecks
- Use gradient accumulation for larger effective batch sizes
- Consider distributed training for multi-GPU setups
