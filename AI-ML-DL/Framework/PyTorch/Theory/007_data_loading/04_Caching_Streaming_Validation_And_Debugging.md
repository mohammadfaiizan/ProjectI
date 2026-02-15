# Caching, Streaming, Validation, and Debugging

## Table of Contents

- [Memory-Mapped Datasets](#memory-mapped-datasets)
- [Caching Strategies](#caching-strategies)
- [Streaming and Iterable Datasets](#streaming-and-iterable-datasets)
- [Data Validation Pipelines](#data-validation-pipelines)
- [Data Loading Debugging](#data-loading-debugging)

---

## Memory-Mapped Datasets

Memory mapping allows accessing large files without loading them entirely into RAM. The operating system maps file regions to virtual memory and loads pages on demand.

### NumPy Memory-Mapped Arrays

**np.load** with `mmap_mode='r'` creates a memory-mapped array. Data stays on disk; only accessed regions are loaded into RAM.

```python
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

class NumpyMemMapDataset(Dataset):
    def __init__(self, data_file, labels_file):
        self.data_mmap = np.load(data_file, mmap_mode='r')
        self.labels_mmap = np.load(labels_file, mmap_mode='r')

    def __len__(self):
        return len(self.data_mmap)

    def __getitem__(self, idx):
        data = torch.from_numpy(self.data_mmap[idx].copy())
        label = torch.tensor(self.labels_mmap[idx], dtype=torch.long)
        return data, label

num_samples = 10000
feature_dim = 128
data = np.random.randn(num_samples, feature_dim).astype(np.float32)
labels = np.random.randint(0, 10, num_samples, dtype=np.int32)
np.save('data.npy', data)
np.save('labels.npy', labels)

dataset = NumpyMemMapDataset('data.npy', 'labels.npy')
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### HDF5 Datasets

**HDF5** supports hierarchical storage, compression, and chunked access. Use **h5py** for efficient random access to large arrays.

```python
import h5py

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file, data_key='data', labels_key='labels'):
        self.h5_file = h5py.File(hdf5_file, 'r')
        self.data = self.h5_file[data_key]
        self.labels = self.h5_file[labels_key]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return data, label

    def get_slice(self, start, end):
        return torch.from_numpy(self.data[start:end]), torch.from_numpy(self.labels[start:end])

    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()
```

Creating HDF5 with compression and chunking:

```python
with h5py.File('dataset.h5', 'w') as f:
    data_ds = f.create_dataset(
        'data',
        shape=(5000, 256),
        dtype=np.float32,
        compression='gzip',
        compression_opts=9,
        chunks=(1000, 256),
        shuffle=True
    )
    labels_ds = f.create_dataset('labels', shape=(5000,), dtype=np.int32, compression='gzip')
    for i in range(0, 5000, 1000):
        data_ds[i:i+1000] = np.random.randn(1000, 256).astype(np.float32)
        labels_ds[i:i+1000] = np.random.randint(0, 5, 1000, dtype=np.int32)
```

### Custom Binary Format with mmap

For variable-length or custom formats, use **mmap** with an index file for random access.

```python
import mmap
import struct
import pickle

class BinaryMemMapDataset(Dataset):
    def __init__(self, binary_file, index_file):
        self.index = self._load_index(index_file)
        self.file_handle = open(binary_file, 'rb')
        self.mmap = mmap.mmap(self.file_handle.fileno(), 0, access=mmap.ACCESS_READ)

    def _load_index(self, index_file):
        with open(index_file, 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        info = self.index[idx]
        self.mmap.seek(info['offset'])
        feature_dim = struct.unpack('I', self.mmap.read(4))[0]
        data = np.frombuffer(self.mmap.read(feature_dim * 4), dtype=np.float32)
        label = struct.unpack('i', self.mmap.read(4))[0]
        return torch.from_numpy(data), torch.tensor(label, dtype=torch.long)

    def __del__(self):
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'file_handle'):
            self.file_handle.close()
```

---

## Caching Strategies

### In-Memory LRU Cache

**LRU (Least Recently Used)** evicts the least recently accessed item when the cache is full. Use for frequently accessed samples.

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value

class InMemoryCachedDataset(Dataset):
    def __init__(self, base_dataset, cache_size=1000):
        self.base_dataset = base_dataset
        self.cache = LRUCache(cache_size)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        cached = self.cache.get(idx)
        if cached is not None:
            return cached
        item = self.base_dataset[idx]
        self.cache.put(idx, item)
        return item
```

### Disk Cache

For datasets that exceed RAM, cache preprocessed samples to disk.

```python
import hashlib
import pickle

class DiskCache:
    def __init__(self, cache_dir, max_size_mb=100):
        self.cache_dir = cache_dir
        self.max_size_bytes = max_size_mb * 1024 * 1024
        os.makedirs(cache_dir, exist_ok=True)
        self.index = {}
        self.access_order = []

    def _get_path(self, key):
        key_hash = hashlib.md5(str(key).encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.cache")

    def get(self, key):
        if key in self.index:
            path = self._get_path(key)
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                return data
        return None

    def put(self, key, data):
        path = self._get_path(key)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        self.index[key] = len(pickle.dumps(data))
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
```

### Multi-Level Cache

Combine in-memory (L1) and disk (L2) caches. Check L1 first, then L2, then load from source. Promote L2 hits to L1.

```python
class MultiLevelCache:
    def __init__(self, l1_capacity, l2_cache_dir, l2_size_mb=100):
        self.l1 = LRUCache(l1_capacity)
        self.l2 = DiskCache(l2_cache_dir, l2_size_mb)

    def get(self, key):
        item = self.l1.get(key)
        if item is not None:
            return item
        item = self.l2.get(str(key))
        if item is not None:
            self.l1.put(key, item)
            return item
        return None

    def put(self, key, value):
        self.l1.put(key, value)
        self.l2.put(str(key), value)
```

### Memory-Aware Caching

Adapt cache size based on available system memory using **psutil**.

```python
import psutil

class MemoryAwareCache:
    def __init__(self, max_memory_percent=10.0):
        self.max_memory_percent = max_memory_percent
        self.cache = OrderedDict()
        self.memory_usage = 0
        self.process = psutil.Process()

    def _get_memory_limit(self):
        available = psutil.virtual_memory().available
        return min(available * 0.8, psutil.virtual_memory().total * self.max_memory_percent / 100)

    def _estimate_size(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.numel() * obj.element_size()
        return 1000

    def _cleanup(self):
        limit = self._get_memory_limit()
        while self.memory_usage > limit and self.cache:
            key, value = self.cache.popitem(last=False)
            self.memory_usage -= self._estimate_size(value)

    def put(self, key, value):
        size = self._estimate_size(value)
        if key in self.cache:
            self.memory_usage -= self._estimate_size(self.cache[key])
        self.cache[key] = value
        self.memory_usage += size
        self._cleanup()
```

---

## Streaming and Iterable Datasets

### IterableDataset

**IterableDataset** does not support `__len__` or indexing. It implements `__iter__` to yield samples. Use for unbounded streams, large files read sequentially, or distributed streaming.

```python
from torch.utils.data import IterableDataset

class StreamingDataset(IterableDataset):
    def __init__(self, stream_generator):
        self.stream_generator = stream_generator

    def __iter__(self):
        return iter(self.stream_generator())

def synthetic_stream():
    while True:
        x = torch.randn(10)
        y = torch.randint(0, 2, (1,)).float()
        yield x, y

streaming_ds = StreamingDataset(synthetic_stream)
loader = DataLoader(streaming_ds, batch_size=32, num_workers=0)
```

### Buffered Streaming

Use a buffer to smooth production-consumption rates.

```python
from collections import deque

class BufferedStreamingDataset(IterableDataset):
    def __init__(self, stream_source, buffer_size=10000, batch_size=32):
        self.stream_source = stream_source
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)

    def __iter__(self):
        stream_iter = iter(self.stream_source())
        for data in stream_iter:
            self.buffer.append(data)
            if len(self.buffer) >= self.batch_size:
                batch = [self.buffer.popleft() for _ in range(self.batch_size)]
                xs, ys = zip(*batch)
                yield torch.stack(xs), torch.stack(ys)
```

### WebDataset Concepts

**WebDataset**-style loading reads from sharded tar archives. Each shard contains multiple samples. Workers read different shards in parallel. Conceptually:

```python
class ShardedStreamingDataset(IterableDataset):
    def __init__(self, shard_patterns, worker_info=None):
        self.shard_patterns = shard_patterns
        self.worker_info = worker_info

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            shards = [s for i, s in enumerate(self.shard_patterns) if i % num_workers == worker_id]
        else:
            shards = self.shard_patterns
        for shard_path in shards:
            for sample in self._read_shard(shard_path):
                yield sample
```

### Online Learning Dataset

For online learning with concept drift, generate samples that evolve over time.

```python
class OnlineLearningDataset(IterableDataset):
    def __init__(self, drift_rate=0.001, noise_level=0.1):
        self.drift_rate = drift_rate
        self.noise_level = noise_level
        self.time_step = 0

    def __iter__(self):
        while True:
            drift_factor = np.sin(self.time_step * self.drift_rate) * 0.5
            noise = np.random.normal(0, self.noise_level)
            x = torch.randn(5) + drift_factor + noise
            label_prob = torch.sigmoid(x.sum() + drift_factor)
            y = torch.bernoulli(label_prob).long()
            self.time_step += 1
            yield x, y
```

---

## Data Validation Pipelines

### Validation Rules

Define validation rules that check shape, dtype, value range, and NaN/Inf.

```python
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

class ValidationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    passed: bool
    severity: ValidationSeverity
    message: str
    details: dict = None

class ValidationRule(ABC):
    def __init__(self, name, severity=ValidationSeverity.ERROR):
        self.name = name
        self.severity = severity

    @abstractmethod
    def validate(self, data):
        pass

class TensorShapeRule(ValidationRule):
    def __init__(self, expected_shape, name="shape_check"):
        super().__init__(name)
        self.expected_shape = expected_shape

    def validate(self, data):
        if not isinstance(data, torch.Tensor):
            return ValidationResult(False, self.severity, f"Expected tensor, got {type(data)}")
        if data.shape != self.expected_shape:
            return ValidationResult(False, self.severity, f"Shape mismatch: expected {self.expected_shape}, got {data.shape}")
        return ValidationResult(True, ValidationSeverity.INFO, "Shape validation passed")

class TensorDtypeRule(ValidationRule):
    def __init__(self, expected_dtype, name="dtype_check"):
        super().__init__(name)
        self.expected_dtype = expected_dtype

    def validate(self, data):
        if not isinstance(data, torch.Tensor):
            return ValidationResult(False, self.severity, f"Expected tensor, got {type(data)}")
        if data.dtype != self.expected_dtype:
            return ValidationResult(False, self.severity, f"Dtype mismatch: expected {self.expected_dtype}, got {data.dtype}")
        return ValidationResult(True, ValidationSeverity.INFO, "Dtype validation passed")

class TensorNaNRule(ValidationRule):
    def __init__(self, name="nan_check"):
        super().__init__(name, ValidationSeverity.ERROR)

    def validate(self, data):
        if not isinstance(data, torch.Tensor):
            return ValidationResult(False, self.severity, f"Expected tensor, got {type(data)}")
        if torch.isnan(data).any():
            nan_count = torch.isnan(data).sum().item()
            return ValidationResult(False, self.severity, f"Found {nan_count} NaN values")
        return ValidationResult(True, ValidationSeverity.INFO, "No NaN values found")
```

### Validation Pipeline

Chain multiple rules and aggregate results.

```python
class ValidationPipeline:
    def __init__(self, rules, stop_on_error=True):
        self.rules = rules
        self.stop_on_error = stop_on_error

    def validate(self, data):
        results = []
        for rule in self.rules:
            result = rule.validate(data)
            results.append(result)
            if not result.passed and self.stop_on_error and result.severity == ValidationSeverity.ERROR:
                break
        passed = all(r.passed for r in results)
        return {'passed': passed, 'results': results, 'total_rules': len(self.rules), 'passed_rules': sum(1 for r in results if r.passed)}

pipeline = ValidationPipeline([
    TensorShapeRule((3, 224, 224)),
    TensorDtypeRule(torch.float32),
    TensorNaNRule(),
])
result = pipeline.validate(torch.randn(3, 224, 224))
```

### ValidatedDataset

Wrap a dataset to validate each sample on access.

```python
class ValidatedDataset(Dataset):
    def __init__(self, base_dataset, input_pipeline=None, target_pipeline=None, validation_mode='strict'):
        self.base_dataset = base_dataset
        self.input_pipeline = input_pipeline
        self.target_pipeline = target_pipeline
        self.validation_mode = validation_mode

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        if isinstance(data, tuple) and len(data) == 2:
            inputs, targets = data
            if self.input_pipeline:
                r = self.input_pipeline.validate(inputs)
                if not r['passed'] and self.validation_mode == 'strict':
                    raise ValueError(str([x.message for x in r['results']]))
            if self.target_pipeline:
                r = self.target_pipeline.validate(targets)
                if not r['passed'] and self.validation_mode == 'strict':
                    raise ValueError(str([x.message for x in r['results']]))
            return inputs, targets
        return data
```

### Outlier Detection

Use z-scores to flag outliers.

```python
class OutlierDetectionRule(ValidationRule):
    def __init__(self, z_threshold=3.0, max_outliers=None, name="outlier_check"):
        super().__init__(name, ValidationSeverity.WARNING)
        self.z_threshold = z_threshold
        self.max_outliers = max_outliers

    def validate(self, data):
        if not isinstance(data, torch.Tensor):
            return ValidationResult(False, self.severity, f"Expected tensor, got {type(data)}")
        mean, std = data.mean(), data.std()
        if std == 0:
            return ValidationResult(True, ValidationSeverity.INFO, "No variance")
        z_scores = torch.abs((data - mean) / std)
        outlier_count = (z_scores > self.z_threshold).sum().item()
        if self.max_outliers is not None and outlier_count > self.max_outliers:
            return ValidationResult(False, self.severity, f"Too many outliers: {outlier_count}")
        return ValidationResult(True, ValidationSeverity.INFO, f"Found {outlier_count} outliers", {'outlier_count': outlier_count})
```

---

## Data Loading Debugging

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Index out of range | IndexError in __getitem__ | Validate idx; check __len__ |
| Wrong return type | Collate/stack errors | Return tensors, not numpy/list |
| Slow loading | GPU idle, low utilization | Profile; increase num_workers; cache |
| Worker hang | Timeout, no progress | Use num_workers=0 to debug; check worker_init_fn |
| Memory leak | RSS grows over epochs | Avoid retaining refs in __getitem__; clear caches |

### Timing and Bottleneck Identification

Profile batch loading time to distinguish data loading from training.

```python
import time
import numpy as np

class DataLoaderProfiler:
    def __init__(self, loader):
        self.loader = loader
        self.batch_times = []

    def profile(self, num_batches=10):
        start = time.time()
        for i, batch in enumerate(self.loader):
            if i >= num_batches:
                break
            batch_end = time.time()
            self.batch_times.append(batch_end - start)
            start = batch_end
        return {
            'avg_batch_time_ms': np.mean(self.batch_times) * 1000,
            'min_batch_time_ms': np.min(self.batch_times) * 1000,
            'max_batch_time_ms': np.max(self.batch_times) * 1000,
            'throughput': num_batches / sum(self.batch_times)
        }

profiler = DataLoaderProfiler(loader)
stats = profiler.profile(num_batches=20)
print(stats)
```

### Debug Dataset

Add logging and statistics to trace access patterns.

```python
class DebugDataset(Dataset):
    def __init__(self, base_dataset, debug=True):
        self.base_dataset = base_dataset
        self.debug = debug
        self.access_count = 0
        self.access_times = []

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        start = time.time()
        self.access_count += 1
        if self.debug and self.access_count % 10 == 0:
            print(f"Accessing item {idx}, total accesses: {self.access_count}")
        result = self.base_dataset[idx]
        self.access_times.append(time.time() - start)
        return result

    def get_stats(self):
        return {
            'total_accesses': self.access_count,
            'avg_access_time': np.mean(self.access_times) if self.access_times else 0,
            'max_access_time': np.max(self.access_times) if self.access_times else 0
        }
```

### Memory Monitoring

Use **psutil** to track memory during loading.

```python
import psutil

def monitor_memory():
    return psutil.Process().memory_info().rss / 1024 / 1024

initial_mb = monitor_memory()
for i, batch in enumerate(loader):
    current_mb = monitor_memory()
    print(f"Batch {i}: {current_mb:.1f} MB (delta: {current_mb - initial_mb:.1f} MB)")
    if i >= 5:
        break
```

### Debugging Checklist

1. **Dataset**: `__len__` correct; `__getitem__` handles all valid indices; consistent return types; no memory leaks.
2. **DataLoader**: Appropriate batch_size; num_workers <= CPU cores; pin_memory for GPU; drop_last for consistent batches.
3. **Multiprocessing**: worker_init_fn for setup; avoid shared mutable state; handle timeouts; use num_workers=0 when debugging.
4. **Performance**: Pre-load or cache when possible; profile different configurations; monitor data loading vs training time ratio.
