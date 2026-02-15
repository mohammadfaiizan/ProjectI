# Samplers and Distributed Loading

## Table of Contents

- [Sampler Fundamentals](#sampler-fundamentals)
- [BatchSampler and Custom Samplers](#batchsampler-and-custom-samplers)
- [Multiprocessing Data Loading](#multiprocessing-data-loading)
- [DistributedSampler for DDP Training](#distributedsampler-for-ddp-training)
- [Data Sharding Across Processes](#data-sharding-across-processes)

---

## Sampler Fundamentals

A **Sampler** defines the order in which indices are drawn from a dataset. DataLoader uses a sampler (or shuffle) to determine which samples form each batch. When a **sampler** is provided, **shuffle** must be False.

### SequentialSampler

**SequentialSampler** yields indices in order: 0, 1, 2, ..., n-1.

```python
from torch.utils.data import SequentialSampler, DataLoader

class SequentialSampler:
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)

dataset = torch.utils.data.TensorDataset(torch.randn(100, 10), torch.randint(0, 5, (100,)))
sampler = SequentialSampler(dataset)
loader = DataLoader(dataset, batch_size=16, sampler=sampler)
```

### RandomSampler

**RandomSampler** yields indices in random order. Supports sampling with or without replacement.

```python
from torch.utils.data import RandomSampler

sampler = RandomSampler(dataset)
loader = DataLoader(dataset, batch_size=16, sampler=sampler)
```

### SubsetRandomSampler

**SubsetRandomSampler** samples only from a subset of indices, in random order. Useful for validation splits or debugging.

```python
from torch.utils.data import SubsetRandomSampler

subset_indices = list(range(0, len(dataset), 2))
sampler = SubsetRandomSampler(subset_indices)
loader = DataLoader(dataset, batch_size=16, sampler=sampler)
```

### WeightedRandomSampler

**WeightedRandomSampler** samples indices with replacement according to given weights. Used for imbalanced datasets to oversample minority classes.

```python
from torch.utils.data import WeightedRandomSampler

labels = dataset.tensors[1]
class_counts = torch.bincount(labels)
weights = 1.0 / class_counts[labels]
sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
loader = DataLoader(dataset, batch_size=16, sampler=sampler)
```

| Sampler | Order | Replacement | Use Case |
|---------|-------|-------------|----------|
| SequentialSampler | Sequential | N/A | Validation, inference |
| RandomSampler | Random | Optional | Training |
| SubsetRandomSampler | Random | No | Subset validation |
| WeightedRandomSampler | Weighted random | Yes | Imbalanced data |

---

## BatchSampler and Custom Samplers

### BatchSampler

**BatchSampler** wraps another sampler and yields batches of indices. When using `batch_sampler`, do not specify `batch_size` or `sampler` in DataLoader.

```python
from torch.utils.data import BatchSampler

base_sampler = RandomSampler(dataset)
batch_sampler = BatchSampler(base_sampler, batch_size=16, drop_last=False)
loader = DataLoader(dataset, batch_sampler=batch_sampler)
```

### Custom Sampler Implementation

A custom sampler must implement `__iter__` (yielding indices) and `__len__` (returning the number of samples to yield).

```python
from torch.utils.data import Sampler

class CustomRandomSampler(Sampler):
    def __init__(self, data_source, seed=None):
        self.data_source = data_source
        self.seed = seed

    def __iter__(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        indices = torch.randperm(len(self.data_source)).tolist()
        return iter(indices)

    def __len__(self):
        return len(self.data_source)

class ReverseSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source) - 1, -1, -1))

    def __len__(self):
        return len(self.data_source)
```

### BalancedSampler

For imbalanced datasets, sample with inverse class frequency to balance batches.

```python
class BalancedSampler(Sampler):
    def __init__(self, dataset, num_samples=None):
        self.dataset = dataset
        self.num_samples = num_samples or len(dataset)
        self.labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])
        self.class_counts = torch.bincount(self.labels)
        self.weights = torch.zeros(len(dataset))
        for i, label in enumerate(self.labels):
            self.weights[i] = 1.0 / self.class_counts[label].float()

    def __iter__(self):
        indices = torch.multinomial(self.weights, self.num_samples, replacement=True)
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples
```

### BalancedBatchSampler

Ensures each batch contains roughly equal samples per class.

```python
import random
from collections import defaultdict

class BalancedBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, samples_per_class=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_indices[label.item()].append(idx)
        self.num_classes = len(self.class_indices)
        self.samples_per_class = samples_per_class or (batch_size // self.num_classes)
        for indices in self.class_indices.values():
            random.shuffle(indices)

    def __iter__(self):
        class_iters = {c: iter(indices * 100) for c, indices in self.class_indices.items()}
        while True:
            batch = []
            for c in self.class_indices:
                for _ in range(self.samples_per_class):
                    try:
                        batch.append(next(class_iters[c]))
                    except StopIteration:
                        if batch:
                            random.shuffle(batch)
                            yield batch[:self.batch_size]
                        return
            random.shuffle(batch)
            yield batch[:self.batch_size]

    def __len__(self):
        min_size = min(len(indices) for indices in self.class_indices.values())
        return min_size // self.samples_per_class
```

---

## Multiprocessing Data Loading

### num_workers

**num_workers** specifies how many subprocesses to use for data loading. Workers load batches in parallel while the main process consumes them. Set to 0 for single-process loading (useful for debugging).

```python
loader = DataLoader(dataset, batch_size=32, num_workers=4)
```

### worker_init_fn

**worker_init_fn** is called once per worker at startup. Use it to set worker-specific random seeds and avoid duplicate augmentation.

```python
def worker_init_fn(worker_id):
    np.random.seed(worker_id)
    torch.manual_seed(worker_id)

loader = DataLoader(dataset, batch_size=32, num_workers=4, worker_init_fn=worker_init_fn)
```

### get_worker_info

Inside `__getitem__`, call **torch.utils.data.get_worker_info()** to obtain worker ID and dataset replica info. Returns None in the main process.

```python
def __getitem__(self, idx):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        worker_id = worker_info.id
        seed = idx + worker_id * 1000
        torch.manual_seed(seed)
    return self.data[idx], self.labels[idx]
```

### fork vs spawn

On **Linux**, the default multiprocessing start method is **fork**. On **Windows** and **macOS** (with CUDA), **spawn** is used. Fork shares memory; spawn starts fresh processes. With CUDA, spawn is required to avoid issues. Use `mp.set_start_method('spawn')` if needed.

### Shared Memory

For large datasets, use **torch.Tensor.share_memory_()** to share tensors across workers without copying.

```python
class SharedMemoryDataset(Dataset):
    def __init__(self, size=500, feature_dim=50):
        self.shared_data = torch.randn(size, feature_dim).share_memory_()
        self.shared_labels = torch.randint(0, 10, (size,)).share_memory_()

    def __len__(self):
        return len(self.shared_data)

    def __getitem__(self, idx):
        return self.shared_data[idx], self.shared_labels[idx]
```

### persistent_workers and prefetch_factor

**persistent_workers=True** keeps workers alive between epochs, avoiding restart overhead. **prefetch_factor** (when num_workers > 0) controls how many batches each worker prefetches.

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=2,
    pin_memory=torch.cuda.is_available()
)
```

---

## DistributedSampler for DDP Training

### DistributedSampler Basics

**DistributedSampler** partitions dataset indices across processes so each rank sees a disjoint subset. For shuffling, call **set_epoch(epoch)** at the start of each epoch to ensure different shuffles across epochs.

```python
from torch.utils.data import DistributedSampler

sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    seed=42
)

loader = DataLoader(
    dataset,
    batch_size=32,
    sampler=sampler,
    num_workers=4,
    pin_memory=True
)

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    for batch in loader:
        pass
```

### Key Rules for DDP

1. Use **DistributedSampler**; do not use `shuffle=True` (sampler handles shuffling).
2. **batch_size** is per-device, not global.
3. Set **drop_last=True** for consistent batch sizes across ranks.
4. Call **sampler.set_epoch(epoch)** each epoch for proper shuffling.

### Custom Distributed Samplers

For imbalanced data, extend DistributedSampler to balance class distribution across ranks.

```python
class BalancedDistributedSampler(DistributedSampler):
    def __init__(self, dataset, labels, num_replicas=None, rank=None, shuffle=True, seed=0):
        super().__init__(dataset, num_replicas, rank, shuffle, seed)
        self.labels = labels
        self.class_indices = {}
        for idx, label in enumerate(labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

    def __iter__(self):
        if self.shuffle:
            for indices in self.class_indices.values():
                random.seed(self.seed + self.epoch)
                random.shuffle(indices)
        distributed_indices = []
        for indices in self.class_indices.values():
            per_rank = len(indices) // self.num_replicas
            remainder = len(indices) % self.num_replicas
            start = self.rank * per_rank + min(self.rank, remainder)
            end = start + per_rank + (1 if self.rank < remainder else 0)
            distributed_indices.extend(indices[start:end])
        if self.shuffle:
            random.seed(self.seed + self.epoch + self.rank)
            random.shuffle(distributed_indices)
        return iter(distributed_indices)
```

---

## Data Sharding Across Processes

### Sharding Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| Round-robin | Assign index i to rank i % world_size | Uniform data |
| Contiguous | Assign contiguous blocks to each rank | Sequential access |
| Hash-based | Assign by hash(i) % world_size | Load balancing |

### ShardedDataset

Implement sharding at the dataset level for very large or distributed storage.

```python
class ShardedDataset(Dataset):
    def __init__(self, base_dataset, world_size=1, rank=0, shard_strategy='round_robin'):
        self.base_dataset = base_dataset
        self.world_size = world_size
        self.rank = rank
        self.shard_strategy = shard_strategy
        self.local_indices = self._create_shard_mapping()

    def _create_shard_mapping(self):
        total = len(self.base_dataset)
        if self.shard_strategy == 'round_robin':
            return [i for i in range(total) if i % self.world_size == self.rank]
        elif self.shard_strategy == 'contiguous':
            per_shard = total // self.world_size
            remainder = total % self.world_size
            start = self.rank * per_shard + min(self.rank, remainder)
            end = start + per_shard + (1 if self.rank < remainder else 0)
            return list(range(start, end))
        raise ValueError(f"Unknown strategy: {self.shard_strategy}")

    def __len__(self):
        return len(self.local_indices)

    def __getitem__(self, idx):
        global_idx = self.local_indices[idx]
        return self.base_dataset[global_idx]
```

### Load Balancing

For variable-cost samples, assign samples to ranks to balance total work.

```python
class LoadBalancedDataset(Dataset):
    def __init__(self, base_dataset, world_size, rank, sample_complexities):
        self.base_dataset = base_dataset
        self.world_size = world_size
        self.rank = rank
        self.sample_complexities = sample_complexities
        self.local_indices = self._balance_load()

    def _balance_load(self):
        order = np.argsort(self.sample_complexities)
        worker_loads = [0.0] * self.world_size
        worker_assignments = [[] for _ in range(self.world_size)]
        for idx in order:
            min_worker = np.argmin(worker_loads)
            worker_assignments[min_worker].append(idx)
            worker_loads[min_worker] += self.sample_complexities[idx]
        return worker_assignments[self.rank]

    def __len__(self):
        return len(self.local_indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.local_indices[idx]]
```

### Fault Tolerance

Implement retry logic and checkpointing for long-running distributed jobs.

```python
class FaultTolerantDataLoader:
    def __init__(self, dataset, batch_size, world_size, rank, max_retries=3):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        self.max_retries = max_retries
        self.failed_samples = set()

    def __iter__(self):
        batch_indices = []
        for idx in self.sampler:
            if idx in self.failed_samples:
                continue
            batch_indices.append(idx)
            if len(batch_indices) >= self.batch_size:
                yield self._load_batch_with_retry(batch_indices)
                batch_indices = []
        if batch_indices:
            yield self._load_batch_with_retry(batch_indices)

    def _load_batch_with_retry(self, indices):
        batch_data, batch_labels = [], []
        for idx in indices:
            for attempt in range(self.max_retries):
                try:
                    d, l = self.dataset[idx]
                    batch_data.append(d)
                    batch_labels.append(l)
                    break
                except Exception:
                    if attempt == self.max_retries - 1:
                        self.failed_samples.add(idx)
        return torch.stack(batch_data), torch.tensor(batch_labels)
```
