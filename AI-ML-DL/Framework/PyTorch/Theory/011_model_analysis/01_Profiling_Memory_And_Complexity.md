# Profiling, Memory, and Complexity Analysis

## Table of Contents

- [torch.profiler for Model Profiling](#torchprofiler-for-model-profiling)
- [Memory Profiling with CUDA APIs](#memory-profiling-with-cuda-apis)
- [Model Complexity Analysis](#model-complexity-analysis)
- [Performance Benchmarking](#performance-benchmarking)

---

## torch.profiler for Model Profiling

**torch.profiler** provides comprehensive profiling of PyTorch models for both CPU and CUDA operations. Use **ProfilerActivity** to specify which activities to profile, **schedule** for controlled profiling during training, and **key_averages** for aggregated statistics.

### ProfilerActivity and Basic Profiling

```python
import torch
import torch.nn as nn
import torch.profiler

model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
).cuda()

input_tensor = torch.randn(32, 784).cuda()

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
) as prof:
    model.eval()
    with torch.no_grad():
        for _ in range(100):
            _ = model(input_tensor)
```

### Schedule for Training Profiling

The **schedule** parameter controls when profiling is active during iterative execution. Use `wait`, `warmup`, `active`, and `repeat` to avoid profiling cold starts and focus on steady-state behavior.

```python
def trace_handler(prof):
    prof.export_chrome_trace("trace.json")
    prof.export_stacks("profiler_stacks.txt", "self_cuda_time_total")

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=3,
        repeat=2
    ),
    on_trace_ready=trace_handler,
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as profiler:
    for step, (data, targets) in enumerate(data_loader):
        if step >= 10:
            break
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        profiler.step()
```

### TensorBoard Trace Export

Export profiler output for visualization in TensorBoard:

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    _ = model(input_tensor)

prof.export_chrome_trace("inference_trace.json")
```

### key_averages and Sorting

**key_averages** aggregates events by operation name and returns a table. Sort by `cuda_time_total`, `cpu_time_total`, or `self_cuda_time_total` to find bottlenecks.

```python
print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

| sort_by Value | Description |
|---------------|-------------|
| cuda_time_total | Total CUDA time including children |
| self_cuda_time_total | CUDA time excluding children |
| cpu_time_total | Total CPU time |
| self_cpu_time_total | CPU time excluding children |

---

## Memory Profiling with CUDA APIs

Use **torch.cuda** memory APIs to monitor GPU memory usage during model execution. Key functions include **memory_allocated**, **memory_reserved**, **memory_summary**, **max_memory_allocated**, and **memory snapshots**.

### memory_allocated and memory_reserved

**memory_allocated** returns currently allocated memory. **memory_reserved** returns memory reserved by the caching allocator.

```python
if torch.cuda.is_available():
    allocated_mb = torch.cuda.memory_allocated() / 1024**2
    reserved_mb = torch.cuda.memory_reserved() / 1024**2
    print(f"Allocated: {allocated_mb:.1f} MB, Reserved: {reserved_mb:.1f} MB")
```

### max_memory_allocated and Peak Statistics

**max_memory_allocated** returns the peak allocated memory since the last reset. Call **reset_peak_memory_stats** before a run to measure a specific operation.

```python
torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    _ = model(input_tensor)

peak_mb = torch.cuda.max_memory_allocated() / 1024**2
print(f"Peak memory: {peak_mb:.1f} MB")
```

### memory_summary

**memory_summary** returns a human-readable string of memory usage:

```python
print(torch.cuda.memory_summary(abbreviated=True))
```

### Memory Snapshots and Layer-wise Analysis

Use forward hooks to capture memory at each layer:

```python
layer_memory = {}

def forward_hook(name):
    def hook(module, input, output):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            layer_memory[name] = allocated
    return hook

for name, module in model.named_modules():
    if len(list(module.children())) == 0:
        module.register_forward_hook(forward_hook(name))

with torch.no_grad():
    _ = model(input_tensor)
```

### Memory Leak Detection

Monitor memory over iterations to detect leaks:

```python
import gc

gc.collect()
torch.cuda.empty_cache()
baseline = torch.cuda.memory_allocated()

for i in range(100):
    outputs = model(data)
    loss = criterion(outputs, targets)
    loss.backward()
    model.zero_grad()
    gc.collect()
    torch.cuda.empty_cache()

    current = torch.cuda.memory_allocated()
    increase_mb = (current - baseline) / 1024**2
    if increase_mb > 10:
        print(f"Potential leak at iteration {i}: +{increase_mb:.1f} MB")
```

---

## Model Complexity Analysis

**Model complexity** is quantified by parameter count, FLOPs (floating-point operations), MACs (multiply-accumulate operations), and layer-by-layer breakdowns.

### Parameter Count

```python
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total_params:,}, Trainable: {trainable_params:,}")
```

### FLOPs Estimation

FLOPs for convolution: `2 * batch * out_h * out_w * in_c * out_c * k_h * k_w / groups`

```python
def conv_flops(input_shape, output_shape, kernel_size, groups=1):
    batch, in_c, in_h, in_w = input_shape
    _, out_c, out_h, out_w = output_shape
    k_h, k_w = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    conv_per_pos = (k_h * k_w * in_c // groups) * (out_c // groups)
    return batch * out_h * out_w * conv_per_pos * 2

def linear_flops(in_features, out_features, batch_size):
    return batch_size * in_features * out_features * 2
```

### MACs (Multiply-Accumulate Operations)

MACs are approximately half of FLOPs for typical operations. One MAC = one multiply and one add.

### Layer-by-Layer Analysis

```python
layer_info = {}
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        params = module.weight.numel() + (module.bias.numel() if module.bias else 0)
        layer_info[name] = {'type': 'Conv2d', 'params': params}
    elif isinstance(module, nn.Linear):
        params = module.in_features * module.out_features + module.out_features
        layer_info[name] = {'type': 'Linear', 'params': params}
```

### Memory Usage Estimation

```python
param_memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
activation_memory = 0

def memory_hook(module, input, output):
    global activation_memory
    if isinstance(output, torch.Tensor):
        activation_memory += output.numel() * output.element_size()

for module in model.modules():
    module.register_forward_hook(memory_hook)

with torch.no_grad():
    _ = model(input_tensor)

activation_memory_mb = activation_memory / 1024**2
print(f"Parameters: {param_memory_mb:.2f} MB, Activations: {activation_memory_mb:.2f} MB")
```

---

## Performance Benchmarking

Use **torch.utils.benchmark.Timer** and **Compare** for reliable latency and throughput measurements. Always include **warm-up runs** to avoid cold-start bias.

### torch.utils.benchmark.Timer

```python
from torch.utils.benchmark import Timer

model.eval()
input_tensor = torch.randn(32, 3, 224, 224).cuda()

t = Timer(
    stmt="model(input_tensor)",
    globals={"model": model, "input_tensor": input_tensor},
    num_threads=1,
)

result = t.blocked_autorange()
print(f"Mean: {result.mean * 1000:.2f} ms")
print(f"Std: {result.std * 1000:.2f} ms")
```

### Compare for Model Comparison

```python
from torch.utils.benchmark import Compare

results = []
for name, model in models.items():
    model.eval()
    t = Timer(
        stmt="model(x)",
        globals={"model": model, "x": input_tensor},
    )
    results.append(t.blocked_autorange())

print(Compare(results))
```

### Latency and Throughput Measurement

```python
def benchmark_inference(model, input_tensor, num_runs=100, warmup=10):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        times = []
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(input_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    mean_ms = np.mean(times)
    throughput = input_tensor.size(0) / (mean_ms / 1000)
    return {'latency_ms': mean_ms, 'throughput_per_sec': throughput}
```

### Warm-up Runs

Warm-up runs ensure CUDA kernels are compiled and caches are populated before measurement:

```python
with torch.no_grad():
    for _ in range(10):
        _ = model(input_tensor)
if torch.cuda.is_available():
    torch.cuda.synchronize()
```

### Batch Size Scaling

Measure throughput across batch sizes to find the optimal operating point:

```python
batch_sizes = [1, 2, 4, 8, 16, 32]
results = {}
for bs in batch_sizes:
    try:
        x = torch.randn(bs, 3, 224, 224).cuda()
        stats = benchmark_inference(model, x)
        results[bs] = stats
    except RuntimeError as e:
        if "out of memory" in str(e):
            break
```
