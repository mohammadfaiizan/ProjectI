# Profiling, Memory, and Complexity

## Table of Contents

1. [TensorFlow Profiler](#1-tensorflow-profiler)
2. [Memory Profiling](#2-memory-profiling)
3. [Model Complexity Analysis](#3-model-complexity-analysis)
4. [Performance Benchmarking](#4-performance-benchmarking)

---

## 1. TensorFlow Profiler

The **TensorFlow Profiler** provides detailed performance analysis of training and inference workloads. It helps identify bottlenecks in GPU/CPU utilization, memory usage, and op execution.

### Profiler Options

**ProfilerOptions** controls the tracing level for different components:

| Option | Levels | Description |
|--------|--------|-------------|
| host_tracer_level | 0-3 | CPU host tracing depth |
| python_tracer_level | 0-1 | Python call stack tracing |
| device_tracer_level | 0-1 | GPU/TPU device tracing |

```python
options = tf.profiler.experimental.ProfilerOptions(
    host_tracer_level=3,
    python_tracer_level=1,
    device_tracer_level=1
)
tf.profiler.experimental.start(logdir, options=options)
# ... training ...
tf.profiler.experimental.stop()
```

### Profiling Training Steps

Wrap training loops with profiler start/stop to capture execution traces. View results in TensorBoard via `tensorboard --logdir=<logdir>`.

```python
logdir = "/tmp/profiler_log"
tf.profiler.experimental.start(logdir)
model.fit(dataset, epochs=5)
tf.profiler.experimental.stop()
```

### Key Metrics

- **Step time**: Time per training step
- **Op time**: Per-operation breakdown
- **Memory**: Peak and current GPU memory
- **Trace viewer**: Timeline of op execution

---

## 2. Memory Profiling

**Memory growth** and **memory stats** help control and monitor GPU memory usage during training.

### tf.config.experimental.set_memory_growth

By default, TensorFlow allocates all GPU memory. **set_memory_growth** enables dynamic allocation so memory grows as needed.

```python
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### Memory Stats

**get_memory_info** returns current and peak memory usage for a device.

```python
mem = tf.config.experimental.get_memory_info('GPU:0')
print(f"Current: {mem['current'] / 1e6:.2f} MB")
print(f"Peak: {mem['peak'] / 1e6:.2f} MB")
```

### reset_memory_stats

Reset memory counters before a run for accurate measurements.

```python
tf.config.experimental.reset_memory_stats('GPU:0')
```

### Memory Best Practices

| Practice | Purpose |
|----------|---------|
| Enable memory growth | Avoid OOM on multi-process setups |
| Use mixed precision | Reduce memory footprint |
| Batch size tuning | Balance memory vs throughput |
| Clear large tensors | Free memory when no longer needed |

---

## 3. Model Complexity Analysis

Understanding **parameter count**, **FLOPs**, and **layer-by-layer** breakdown helps with model design and deployment decisions.

### Parameter Count

**model.count_params()** returns total trainable and non-trainable parameters.

```python
total = model.count_params()
print(f"Total parameters: {total:,}")
```

Per-layer counts via `layer.count_params()`.

### FLOPs Estimation

**FLOPs** (floating-point operations) estimate computational cost. For Conv2D:

```
FLOPs = 2 * H * W * K * K * C_in * C_out
```

Where H, W = spatial size, K = kernel size, C_in/C_out = channels.

```python
def conv_flops(layer, input_shape):
    _, h, w, c_in = input_shape
    k = layer.kernel_size[0]
    c_out = layer.filters
    return 2 * h * w * k * k * c_in * c_out
```

### Layer-by-Layer Analysis

Iterate over model layers to report params and FLOPs per layer.

```python
for layer in model.layers:
    params = layer.count_params()
    # Compute FLOPs based on layer type and input shape
    print(f"{layer.name}: {params:,} params")
```

### model.summary()

Keras **summary()** provides a compact table of layers, output shapes, and parameter counts.

---

## 4. Performance Benchmarking

**tf.test.Benchmark** and manual timing enable reproducible latency and throughput measurements.

### tf.test.Benchmark

Subclass **Benchmark** and use **run_op_benchmark** to report metrics.

```python
class ModelBenchmark(tf.test.Benchmark):
    def benchmark_inference(self):
        model = build_model()
        x = tf.random.normal((32, 128))
        result = model(x, training=False)
        self.run_op_benchmark(iters=100, op=result)
```

### Latency Measurement

Measure single-sample and batch inference latency with warmup to avoid cold-start bias.

```python
# Warmup
for _ in range(50):
    _ = model(x, training=False)

start = time.perf_counter()
for _ in range(200):
    _ = model(x, training=False)
latency_ms = (time.perf_counter() - start) / 200 * 1000
```

### Throughput Measurement

Throughput = samples per second. For batch inference:

```python
throughput = batch_size * 1000 / batch_latency_ms  # samples/sec
```

### Benchmarking Best Practices

| Practice | Purpose |
|----------|---------|
| Warmup iterations | Stabilize GPU/CPU state |
| Multiple runs | Report mean/median |
| Fixed input size | Reproducible comparison |
| Disable training | Ensure inference mode |

---

## Summary

| Topic | Key APIs | Use Case |
|-------|----------|----------|
| Profiler | tf.profiler.experimental | Training step analysis |
| Memory | tf.config.experimental | GPU memory tracking |
| Complexity | count_params, FLOPs | Model design |
| Benchmark | tf.test.Benchmark | Latency/throughput |
