# Distributed Training Strategies

## Table of Contents

1. [Introduction to Distributed Training](#1-introduction-to-distributed-training)
2. [MirroredStrategy](#2-mirroredstrategy)
3. [MultiWorkerMirroredStrategy](#3-multiworkermirroredstrategy)
4. [ParameterServerStrategy](#4-parameterserverstrategy)
5. [CentralStorageStrategy](#5-centralstoragestrategy)
6. [Data Parallelism](#6-data-parallelism)
7. [Model Parallelism](#7-model-parallelism)
8. [Best Practices](#8-best-practices)

---

## 1. Introduction to Distributed Training

**Distributed training** spreads computation across multiple devices (GPUs, TPUs, machines) to reduce training time and handle larger models or datasets.

**Key concepts:**
- **Data parallelism:** Same model on each device; different data shards; gradients aggregated.
- **Model parallelism:** Different parts of the model on different devices.
- **Synchronous vs asynchronous:** Gradients applied together (sync) or independently (async).

---

## 2. MirroredStrategy

**MirroredStrategy** replicates the model on each GPU and uses **AllReduce** to synchronize gradients. Best for single-machine multi-GPU.

```python
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, epochs=5)
```

**Key concept:** Create the model inside `strategy.scope()` so variables are mirrored. The dataset is automatically sharded.

---

## 3. MultiWorkerMirroredStrategy

**MultiWorkerMirroredStrategy** extends MirroredStrategy to multiple machines. Each worker has one or more GPUs.

```python
strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    model = build_model()
    model.compile(...)
model.fit(train_dataset, epochs=5)
```

**Cluster configuration:** Set `TF_CONFIG` environment variable with cluster spec and task info.

```python
import json
import os
tf_config = {
    'cluster': {
        'worker': ['worker0:port', 'worker1:port']
    },
    'task': {'type': 'worker', 'index': 0}
}
os.environ['TF_CONFIG'] = json.dumps(tf_config)
```

---

## 4. ParameterServerStrategy

**ParameterServerStrategy** uses parameter server (PS) workers to store variables. Workers compute gradients and push/pull from PS. Supports async updates.

```python
strategy = tf.distribute.experimental.ParameterServerStrategy(
    cluster_resolver
)
with strategy.scope():
    model = build_model()
```

**Use case:** Large-scale training with many workers; async can improve throughput when workers have varying speeds.

---

## 5. CentralStorageStrategy

**CentralStorageStrategy** keeps variables on CPU and replicates compute on GPUs. Useful when model is large and does not fit in GPU memory.

```python
strategy = tf.distribute.experimental.CentralStorageStrategy()
with strategy.scope():
    model = build_model()
```

---

## 6. Data Parallelism

**Data parallelism** is the default with MirroredStrategy. The global batch size is split across devices.

```python
# Global batch = 64, 4 GPUs -> 16 per GPU
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
dataset = dataset.with_options(options)
model.fit(dataset, epochs=5)
```

**Gradient accumulation:** Simulate larger batches by accumulating gradients over multiple steps before applying.

---

## 7. Model Parallelism

**Model parallelism** splits layers across devices. TensorFlow supports this via manual placement or custom training loops.

```python
with tf.device('/GPU:0'):
    layer1 = ...
with tf.device('/GPU:1'):
    layer2 = ...
```

For very large models, use **tf.distribute.TPUStrategy** or custom pipelines.

---

## 8. Best Practices

| Practice | Description |
|----------|-------------|
| Use MirroredStrategy for single-node multi-GPU | Simple, synchronous, good performance |
| Scale batch size with devices | Global batch = per_device_batch * num_devices |
| Use tf.data with AUTOTUNE | Overlap data loading with compute |
| Set TF_CONFIG for multi-worker | Required for MultiWorkerMirroredStrategy |
| Prefer sync over async | More stable convergence |
