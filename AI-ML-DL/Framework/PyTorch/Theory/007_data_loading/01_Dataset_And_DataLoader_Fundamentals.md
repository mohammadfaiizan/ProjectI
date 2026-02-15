# Dataset and DataLoader Fundamentals

## Table of Contents

- [torch.utils.data.Dataset](#torchutilsdatadataset)
- [TensorDataset, ConcatDataset, Subset, and random_split](#tensordataset-concatdataset-subset-and-random_split)
- [DataLoader Parameters and Configuration](#dataloader-parameters-and-configuration)
- [Custom Dataset Classes](#custom-dataset-classes)
- [Custom Collate Functions](#custom-collate-functions)

---

## torch.utils.data.Dataset

The **Dataset** class is the foundational abstraction for representing a collection of data in PyTorch. Any map-style dataset must implement two core methods: `__len__` and `__getitem__`. These methods enable indexing, iteration, and integration with the DataLoader.

### Abstract Methods

| Method | Purpose |
|--------|---------|
| `__len__(self)` | Returns the total number of samples in the dataset |
| `__getitem__(self, idx)` | Returns the sample at index `idx` |

The **index** passed to `__getitem__` can be an integer, a slice, a list, or a tensor. PyTorch's DataLoader typically passes integers; custom indexing patterns require explicit handling.

```python
from torch.utils.data import Dataset
import torch

class BasicDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        if not isinstance(self.data, torch.Tensor):
            self.data = torch.tensor(self.data, dtype=torch.float32)
        if not isinstance(self.labels, torch.Tensor):
            self.labels = torch.tensor(self.labels, dtype=torch.long)
        assert len(self.data) == len(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx], self.labels[idx]

sample_data = torch.randn(100, 10)
sample_labels = torch.randint(0, 3, (100,))
dataset = BasicDataset(sample_data, sample_labels)
print(len(dataset))
print(dataset[0][0].shape)
```

### Data Indexing and Access Patterns

Datasets support multiple indexing patterns: single index, slice, list of indices, and tensor indices. Handling these correctly ensures compatibility with various use cases.

```python
class IndexedDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 3, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.data[idx], self.labels[idx]
        elif isinstance(idx, (list, tuple)):
            return self.data[idx], self.labels[idx]
        elif isinstance(idx, torch.Tensor):
            if idx.dim() == 0:
                return self.data[idx.item()], self.labels[idx.item()]
            return self.data[idx.tolist()], self.labels[idx.tolist()]
        return self.data[idx], self.labels[idx]
```

### Memory vs On-Demand Loading

**Eager loading** loads all data into memory at initialization. **Lazy loading** loads samples on demand in `__getitem__`. **Memory-mapped loading** uses memory-mapped files for large datasets that exceed RAM. The choice depends on dataset size, access patterns, and available memory.

---

## TensorDataset, ConcatDataset, Subset, and random_split

PyTorch provides built-in utilities for composing and splitting datasets.

### TensorDataset

**TensorDataset** wraps one or more tensors into a map-style dataset. Each tensor's first dimension is treated as the sample dimension.

```python
from torch.utils.data import TensorDataset, DataLoader

data = torch.randn(100, 20)
labels = torch.randint(0, 5, (100,))
tensor_dataset = TensorDataset(data, labels)
loader = DataLoader(tensor_dataset, batch_size=16, shuffle=True)
batch_data, batch_labels = next(iter(loader))
```

### ConcatDataset

**ConcatDataset** concatenates multiple datasets into a single logical dataset. Indices are mapped sequentially across datasets.

```python
from torch.utils.data import ConcatDataset

dataset1 = BasicDataset(torch.randn(50, 5), torch.randint(0, 2, (50,)))
dataset2 = BasicDataset(torch.randn(30, 5), torch.randint(0, 2, (30,)))
concat_dataset = ConcatDataset(dataset1, dataset2)
print(len(concat_dataset))
sample = concat_dataset[60]
```

### Subset

**Subset** creates a view over a dataset restricted to specified indices.

```python
from torch.utils.data import Subset

subset_indices = list(range(0, len(concat_dataset), 2))
subset_dataset = Subset(concat_dataset, subset_indices)
print(len(subset_dataset))
```

### random_split

**random_split** splits a dataset into non-overlapping subsets with specified lengths. Uses a generator for reproducibility.

```python
from torch.utils.data import random_split

full_dataset = BasicDataset(torch.randn(100, 10), torch.randint(0, 3, (100,)))
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                         generator=torch.Generator().manual_seed(42))
```

---

## DataLoader Parameters and Configuration

The **DataLoader** batches samples from a dataset, supports shuffling, multiprocessing, and custom collation.

### Core Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset` | Dataset | The dataset to load from |
| `batch_size` | int | Number of samples per batch |
| `shuffle` | bool | Shuffle indices each epoch |
| `num_workers` | int | Number of worker processes |
| `pin_memory` | bool | Pin memory for faster GPU transfer |
| `drop_last` | bool | Drop last incomplete batch |
| `collate_fn` | callable | Custom batch collation function |
| `prefetch_factor` | int | Batches to prefetch per worker |
| `persistent_workers` | bool | Keep workers alive between epochs |

### Basic DataLoader Usage

```python
from torch.utils.data import DataLoader

dataset = BasicDataset(torch.randn(100, 10), torch.randint(0, 3, (100,)))

loader = DataLoader(dataset, batch_size=16, shuffle=True)
for batch_data, batch_labels in loader:
    pass

loader_no_shuffle = DataLoader(dataset, batch_size=16, shuffle=False)
loader_drop_last = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
```

### Multiprocessing and Memory

When `num_workers > 0`, DataLoader spawns worker processes. **pin_memory** allocates pinned (page-locked) memory for faster CPU-to-GPU transfers. **persistent_workers** avoids worker restart overhead between epochs. **prefetch_factor** controls how many batches each worker prefetches.

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
    prefetch_factor=2
)
```

### worker_init_fn

Use **worker_init_fn** to set per-worker random seeds and perform worker-specific initialization.

```python
def worker_init_fn(worker_id):
    import numpy as np
    np.random.seed(worker_id)
    torch.manual_seed(worker_id)

loader = DataLoader(dataset, batch_size=16, num_workers=2, worker_init_fn=worker_init_fn)
```

---

## Custom Dataset Classes

Custom datasets adapt various data sources to the PyTorch interface.

### Image Dataset

For image classification, load images from disk, apply transforms, and return tensors with labels.

```python
from pathlib import Path
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, annotations_file=None, transform=None, target_transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.samples = self._load_samples()
        self.classes = sorted(set(s[1] for s in self.samples))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def _load_samples(self):
        samples = []
        for ext in ['.jpg', '.jpeg', '.png']:
            for img_path in self.root_dir.rglob(f'*{ext}'):
                class_name = img_path.parent.name
                samples.append((str(img_path), class_name))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        label = self.class_to_idx[class_name]
        if self.transform:
            image_tensor = self.transform(image_tensor)
        if self.target_transform:
            label = self.target_transform(label)
        return image_tensor, label
```

### CSV/Tabular Dataset

For tabular data, load CSV, handle categorical encoding, normalize numerical features, and return feature tensors with targets.

```python
import pandas as pd

class TabularDataset(Dataset):
    def __init__(self, csv_file, target_column, feature_columns=None, categorical_columns=None, normalize=True):
        df = pd.read_csv(csv_file)
        self.feature_columns = feature_columns or [c for c in df.columns if c != target_column]
        X = df[self.feature_columns].copy()
        y = df[target_column].copy()
        self.categorical_mappings = {}
        for col in (categorical_columns or []):
            if col in X.columns:
                mapping = {v: i for i, v in enumerate(X[col].unique())}
                self.categorical_mappings[col] = mapping
                X[col] = X[col].map(mapping)
        X = X.fillna(X.mean())
        if normalize:
            self.feature_means = X.select_dtypes(include=[np.number]).mean()
            self.feature_stds = X.select_dtypes(include=[np.number]).std()
            X = (X - self.feature_means) / (self.feature_stds + 1e-8)
        self.features = torch.tensor(X.values, dtype=torch.float32)
        self.targets = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
```

### Text Dataset with Vocabulary

For NLP, build a vocabulary, tokenize text, pad/truncate sequences, and return encoded tensors.

```python
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab_size=10000, max_length=128):
        self.vocab = self._build_vocabulary(texts, vocab_size)
        self.label_to_idx = {l: i for i, l in enumerate(set(labels))}
        self.encoded_data = []
        for text, label in zip(texts, labels):
            tokens = text.lower().split()
            ids = [self.vocab.get(t, self.vocab['<UNK>']) for t in tokens]
            ids = ids[:max_length] or ids + [self.vocab['<PAD>']] * (max_length - len(ids))
            self.encoded_data.append((
                torch.tensor(ids, dtype=torch.long),
                torch.tensor(self.label_to_idx[label], dtype=torch.long)
            ))

    def _build_vocabulary(self, texts, vocab_size):
        from collections import Counter
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.lower().split())
        vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        for word, _ in word_counts.most_common(vocab_size - 4):
            vocab[word] = len(vocab)
        return vocab

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]
```

### Multi-Modal Dataset

For multi-modal data (e.g., image + text), load and return dictionaries or tuples of tensors.

```python
class MultiModalDataset(Dataset):
    def __init__(self, image_paths, texts, labels, image_transform=None):
        self.image_paths = image_paths
        self.texts = texts
        self.labels = labels
        self.image_transform = image_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        if self.image_transform:
            image_tensor = self.image_transform(image_tensor)
        return {
            'image': image_tensor,
            'text': self.texts[idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
```

---

## Custom Collate Functions

The default **collate_fn** stacks tensors along a new batch dimension. For variable-length sequences or custom structures, provide a custom collate function.

### Default Collation

The default collate stacks `(data, label)` tuples into `(batch_data, batch_labels)` tensors. It fails when samples have different shapes.

### Padding Collate for Variable-Length Sequences

```python
def padding_collate_fn(batch):
    data, labels = zip(*batch)
    lengths = [d.size(0) for d in data]
    max_length = max(lengths)
    padded_data = []
    for d in data:
        if d.size(0) < max_length:
            padding = torch.zeros(max_length - d.size(0), d.size(1))
            d = torch.cat([d, padding], dim=0)
        padded_data.append(d)
    return {
        'data': torch.stack(padded_data),
        'labels': torch.stack(labels),
        'lengths': torch.tensor(lengths)
    }

class VariableLengthDataset(Dataset):
    def __init__(self, size=50):
        self.data = [torch.randn(np.random.randint(10, 50), 5) for _ in range(size)]
        self.labels = torch.randint(0, 3, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

var_dataset = VariableLengthDataset(size=20)
loader = DataLoader(var_dataset, batch_size=4, collate_fn=padding_collate_fn)
batch = next(iter(loader))
print(batch['data'].shape, batch['lengths'])
```

### Custom Collate with Batch Statistics

```python
def custom_collate_fn(batch):
    data, labels = zip(*batch)
    data = torch.stack(data, dim=0)
    labels = torch.stack(labels, dim=0)
    batch_stats = {'mean': data.mean(), 'std': data.std(), 'size': len(batch)}
    return {'data': data, 'labels': labels, 'stats': batch_stats}
```

### Handling None or Invalid Samples

```python
def robust_collate_fn(batch):
    valid_batch = [(d, l) for d, l in batch if d is not None]
    if not valid_batch:
        return torch.empty(0, 10), torch.empty(0, dtype=torch.long)
    data_list, label_list = zip(*valid_batch)
    return torch.stack(data_list), torch.tensor(label_list)
```

---

## Error Handling and Best Practices

**Validate data** during initialization. Handle missing files, corrupted samples, and NaN values. Use `handle_errors` strategies: `raise`, `skip`, or `default`. Keep `__getitem__` fast; avoid expensive I/O or computation. Use appropriate dtypes (float32, long) for model compatibility. Consider lazy loading and caching for large datasets.
