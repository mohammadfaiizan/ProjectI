# Profiling, Extensions, and Performance

## Table of Contents

1. [TensorFlow Profiler](#1-tensorflow-profiler)
2. [Memory and Timeline Analysis](#2-memory-and-timeline-analysis)
3. [TF Addons and Extensions](#3-tf-addons-and-extensions)
4. [Custom Ops and Kernels](#4-custom-ops-and-kernels)
5. [Performance Optimization](#5-performance-optimization)
6. [Mixed Precision Training](#6-mixed-precision-training)
7. [XLA Compilation](#7-xla-compilation)
8. [Best Practices](#8-best-practices)

---

## 1. TensorFlow Profiler

**TensorFlow Profiler** analyzes model performance: GPU utilization, kernel execution, memory usage.

### Enable Profiling

```python
from tensorflow.python.eager import profiler

# Option 1: Programmatic
profiler.start()
# ... training step ...
profiler.stop()
profiler.profile(options=profiler.ProfilerOptions(host_tracer_level=2))

# Option 2: TensorBoard callback
tf.keras.callbacks.TensorBoard(
    log_dir='logs',
    profile_batch='10,20'
)
```

### Key Metrics

- **Step time:** Time per training step.
- **GPU utilization:** Fraction of time GPU is busy.
- **Memory:** Peak and fragmentation.

---

## 2. Memory and Timeline Analysis

### Memory Profiler

```python
from tensorflow.python.profiler import profiler_v2 as profiler

with profiler.Profile('logs'):
    model.fit(train_ds, epochs=1)
```

### Timeline

Export a trace for Chrome's trace viewer:

```python
options = tf.profiler.experimental.ProfilerOptions(
    host_tracer_level=3,
    python_tracer_level=1
)
tf.profiler.experimental.start('logdir', options=options)
# ... run ...
tf.profiler.experimental.stop()
```

---

## 3. TF Addons and Extensions

**TensorFlow Addons (TFA)** provides extra layers, losses, and optimizers not in core TF.

```python
import tensorflow_addons as tfa

# Layer Normalization
layer_norm = tfa.layers.GroupNormalization(groups=8)

# Optimizer
optimizer = tfa.optimizers.AdamW(weight_decay=0.01)

# Loss
loss = tfa.losses.TripletSemiHardLoss()
```

**Note:** TFA is in maintenance mode. Some functionality is migrating to Keras core.

---

## 4. Custom Ops and Kernels

### Custom Layer (Python)

```python
class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='glorot_uniform')
        self.b = self.add_weight(shape=(self.units,), initializer='zeros')
        super().build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

### Custom Op (C++)

For performance-critical code, implement a C++ op and register with TensorFlow. Use `tf.load_op_library` to load.

---

## 5. Performance Optimization

### tf.function

Wrap training step in **tf.function** for graph execution and optimization:

```python
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        pred = model(x)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, pred)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss
```

### Data Pipeline

- **prefetch:** Overlap data loading with compute.
- **cache:** Avoid recomputing preprocessing.
- **num_parallel_calls:** Parallelize map.

```python
ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
```

---

## 6. Mixed Precision Training

**Mixed precision** uses float16 for compute and float32 for master weights. Reduces memory and speeds up on compatible GPUs.

```python
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

model = build_model()
# Loss scaling for float16
optimizer = tf.keras.optimizers.Adam()
optimizer = mixed_precision.LossScaleOptimizer(optimizer)
```

**Key concept:** Output layer should use float32 (e.g., `dtype='float32'`) for numerical stability.

---

## 7. XLA Compilation

**XLA (Accelerated Linear Algebra)** compiles the TensorFlow graph for faster execution.

```python
@tf.function(jit_compile=True)
def train_step(x, y):
    ...
```

Or set globally:

```python
tf.config.optimizer.set_jit(True)
```

**Note:** XLA may increase compile time. Best for stable model architectures and repeated execution.

---

## 8. Best Practices

| Practice | Description |
|----------|-------------|
| Profile before optimizing | Identify bottlenecks first |
| Use tf.function for hot paths | Training step, inference |
| Optimize data pipeline | prefetch, cache, AUTOTUNE |
| Consider mixed precision | On Volta+ GPUs |
| Use XLA for stable graphs | After model is fixed |
