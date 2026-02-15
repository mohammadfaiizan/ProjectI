# Statistics, Sampling, Splitting, and Efficiency

## Table of Contents

- [Batch Processing Patterns](#batch-processing-patterns)
- [Data Splitting](#data-splitting)
- [Tensor Statistics](#tensor-statistics)
- [Sampling Methods](#sampling-methods)
- [Data Transformation Operations](#data-transformation-operations)
- [Memory-Efficient Data Operations](#memory-efficient-data-operations)

---

## Batch Processing Patterns

**Batch processing** improves GPU utilization, stabilizes gradients, and enables vectorized operations. Key patterns include batch normalization, batch matrix operations, and variable-length sequence handling.

### Basic Batch Operations

```python
import torch
import torch.nn.functional as F
import math

batch_samples = torch.randn(16, 3, 32, 32)
batch_mean = batch_samples.mean(dim=0)
batch_std = batch_samples.std(dim=0)
per_sample_mean = batch_samples.mean(dim=[1, 2, 3])
per_sample_norm = batch_samples.norm(dim=[1, 2, 3])
```

### Batch Matrix Operations

**torch.bmm** performs batch matrix multiplication. **torch.einsum** expresses complex batch operations concisely.

```python
batch_size = 32
matrix_a = torch.randn(batch_size, 10, 20)
matrix_b = torch.randn(batch_size, 20, 15)
batch_result = torch.bmm(matrix_a, matrix_b)
einsum_result = torch.einsum('bik,bkj->bij', matrix_a, matrix_b)
```

### Batch Normalization

```python
def batch_normalize_manual(x, eps=1e-5):
    batch_mean = x.mean(dim=0, keepdim=True)
    batch_var = x.var(dim=0, keepdim=True, unbiased=False)
    normalized = (x - batch_mean) / torch.sqrt(batch_var + eps)
    return normalized, batch_mean, batch_var
```

### Batch Attention

```python
def scaled_dot_product_attention_batch(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores.masked_fill_(mask == 0, -1e9)
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights
```

### Variable-Length Sequence Padding

```python
def pad_batch_sequences(sequences, pad_value=0):
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = []
    masks = []
    for seq in sequences:
        pad_length = max_length - len(seq)
        if pad_length > 0:
            padding = torch.full((pad_length,) + seq.shape[1:], pad_value, dtype=seq.dtype)
            padded_seq = torch.cat([seq, padding], dim=0)
        else:
            padded_seq = seq
        mask = torch.zeros(max_length, dtype=torch.bool)
        mask[:len(seq)] = True
        padded_sequences.append(padded_seq)
        masks.append(mask)
    return torch.stack(padded_sequences), torch.stack(masks)
```

### Chunked Processing for Large Data

```python
def process_in_chunks(data, chunk_size, process_func):
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        chunk_result = process_func(chunk)
        results.append(chunk_result)
    return torch.cat(results, dim=0)
```

---

## Data Splitting

**Data splitting** partitions datasets into train, validation, and test sets. The strategy depends on data characteristics: i.i.d., temporal, or grouped.

### Random Splitting

```python
def random_split(dataset_size, ratios=(0.7, 0.2, 0.1), seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    indices = torch.randperm(dataset_size)
    train_size = int(dataset_size * ratios[0])
    val_size = int(dataset_size * ratios[1])
    test_size = dataset_size - train_size - val_size
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    return train_indices, val_indices, test_indices

def simple_train_test_split(data, test_ratio=0.2, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    dataset_size = len(data)
    test_size = int(dataset_size * test_ratio)
    indices = torch.randperm(dataset_size)
    train_indices = indices[:dataset_size - test_size]
    test_indices = indices[dataset_size - test_size:]
    return data[train_indices], data[test_indices], train_indices, test_indices
```

### Stratified Splitting

**Stratified splitting** preserves class distribution across splits. Essential for imbalanced datasets.

```python
def stratified_split(labels, ratios=(0.7, 0.2, 0.1), seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    unique_classes = torch.unique(labels)
    class_indices = {cls.item(): torch.where(labels == cls)[0] for cls in unique_classes}
    train_indices, val_indices, test_indices = [], [], []
    for cls, indices in class_indices.items():
        class_size = len(indices)
        train_size = int(class_size * ratios[0])
        val_size = int(class_size * ratios[1])
        perm_indices = indices[torch.randperm(len(indices))]
        train_indices.extend(perm_indices[:train_size].tolist())
        val_indices.extend(perm_indices[train_size:train_size + val_size].tolist())
        test_indices.extend(perm_indices[train_size + val_size:].tolist())
    train_indices = torch.tensor(train_indices)[torch.randperm(len(train_indices))]
    val_indices = torch.tensor(val_indices)[torch.randperm(len(val_indices))]
    test_indices = torch.tensor(test_indices)[torch.randperm(len(test_indices))]
    return train_indices, val_indices, test_indices
```

### Temporal Splitting

For time series, split chronologically to avoid future information leakage.

```python
def temporal_split(timestamps, ratios=(0.7, 0.2, 0.1)):
    sorted_indices = torch.argsort(timestamps)
    dataset_size = len(timestamps)
    train_size = int(dataset_size * ratios[0])
    val_size = int(dataset_size * ratios[1])
    train_indices = sorted_indices[:train_size]
    val_indices = sorted_indices[train_size:train_size + val_size]
    test_indices = sorted_indices[train_size + val_size:]
    return train_indices, val_indices, test_indices
```

### K-Fold Cross-Validation

```python
def kfold_split(dataset_size, k=5, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    indices = torch.randperm(dataset_size)
    fold_size = dataset_size // k
    folds = []
    for i in range(k):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < k - 1 else dataset_size
        test_indices = indices[start_idx:end_idx]
        train_indices = torch.cat([indices[:start_idx], indices[end_idx:]])
        folds.append((train_indices, test_indices))
    return folds

def stratified_kfold_split(labels, k=5, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    unique_classes = torch.unique(labels)
    class_indices = {cls.item(): torch.where(labels == cls)[0][torch.randperm((labels == cls).sum().item())] for cls in unique_classes}
    folds = [[] for _ in range(k)]
    for cls, indices in class_indices.items():
        class_size = len(indices)
        fold_size = class_size // k
        for i in range(k):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < k - 1 else class_size
            folds[i].extend(indices[start_idx:end_idx].tolist())
    cv_splits = []
    for i in range(k):
        test_indices = torch.tensor(folds[i])
        train_indices = torch.cat([torch.tensor(folds[j]) for j in range(k) if j != i])
        cv_splits.append((train_indices, test_indices))
    return cv_splits
```

### Group-Based Splitting

Ensures samples from the same group (e.g., user, subject) stay in the same split.

```python
def group_split(groups, ratios=(0.7, 0.2, 0.1), seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    unique_groups = torch.unique(groups)
    shuffled_groups = unique_groups[torch.randperm(len(unique_groups))]
    train_groups = int(len(unique_groups) * ratios[0])
    val_groups = int(len(unique_groups) * ratios[1])
    train_group_set = set(shuffled_groups[:train_groups].tolist())
    val_group_set = set(shuffled_groups[train_groups:train_groups + val_groups].tolist())
    test_group_set = set(shuffled_groups[train_groups + val_groups:].tolist())
    train_indices = torch.where(torch.isin(groups, torch.tensor(list(train_group_set))))[0]
    val_indices = torch.where(torch.isin(groups, torch.tensor(list(val_group_set))))[0]
    test_indices = torch.where(torch.isin(groups, torch.tensor(list(test_group_set))))[0]
    return train_indices, val_indices, test_indices
```

---

## Tensor Statistics

**Tensor statistics** summarize central tendency, dispersion, correlation, and distribution shape. PyTorch provides built-in reduction operations and supports custom statistical computations.

### Central Tendency and Dispersion

```python
data = torch.randn(1000)
mean_val = data.mean()
median_val = data.median()
mode_val = data.mode().values
variance = data.var()
std_dev = data.std()
quantiles = torch.quantile(data, torch.tensor([0.25, 0.5, 0.75]))
q1, q2, q3 = quantiles
iqr = q3 - q1
```

### Multi-Dimensional Statistics

```python
matrix_data = torch.randn(100, 5)
row_means = matrix_data.mean(dim=1)
col_means = matrix_data.mean(dim=0)
col_mins, col_min_indices = matrix_data.min(dim=0)
col_maxs, col_max_indices = matrix_data.max(dim=0)
```

### Skewness and Kurtosis

```python
def compute_skewness(tensor):
    mean_val = tensor.mean()
    std_val = tensor.std()
    return ((tensor - mean_val) / std_val).pow(3).mean()

def compute_kurtosis(tensor, fisher=True):
    mean_val = tensor.mean()
    std_val = tensor.std()
    kurtosis = ((tensor - mean_val) / std_val).pow(4).mean()
    return kurtosis - 3 if fisher else kurtosis
```

### Correlation and Covariance

```python
def compute_covariance_matrix(matrix):
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    return torch.mm(centered.T, centered) / (matrix.shape[0] - 1)

def compute_correlation_matrix(matrix):
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    standardized = centered / matrix.std(dim=0, keepdim=True)
    return torch.mm(standardized.T, standardized) / (matrix.shape[0] - 1)

def pearson_correlation(x, y):
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    numerator = (x_centered * y_centered).sum()
    denominator = torch.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum())
    return numerator / denominator
```

### Histograms and Empirical CDF

```python
def histogram_counts(tensor, bins=10, range_vals=None):
    if range_vals is None:
        range_vals = (tensor.min().item(), tensor.max().item())
    min_val, max_val = range_vals
    bin_edges = torch.linspace(min_val, max_val, bins + 1)
    counts = torch.zeros(bins)
    for i in range(bins):
        mask = (tensor >= bin_edges[i]) & (tensor < bin_edges[i + 1]) if i < bins - 1 else (tensor >= bin_edges[i]) & (tensor <= bin_edges[i + 1])
        counts[i] = mask.sum()
    return counts, bin_edges

def empirical_cdf(tensor, x_values=None):
    if x_values is None:
        x_values = torch.linspace(tensor.min(), tensor.max(), 100)
    n = len(tensor)
    cdf_values = torch.tensor([(tensor <= x).float().sum() / n for x in x_values])
    return x_values, cdf_values
```

### Robust Statistics

```python
def mad_statistic(tensor):
    return torch.median(torch.abs(tensor - tensor.median()))

def trimmed_mean(tensor, trim_fraction=0.1):
    n = len(tensor)
    trim_count = int(n * trim_fraction)
    sorted_tensor, _ = torch.sort(tensor)
    return sorted_tensor[trim_count:n-trim_count].mean()

def winsorized_mean(tensor, winsor_fraction=0.05):
    n = len(tensor)
    winsor_count = int(n * winsor_fraction)
    sorted_tensor, _ = torch.sort(tensor)
    lower = sorted_tensor[winsor_count]
    upper = sorted_tensor[n - winsor_count - 1]
    winsorized = tensor.clone()
    winsorized[winsorized < lower] = lower
    winsorized[winsorized > upper] = upper
    return winsorized.mean()
```

### Batch Statistics

```python
def batch_statistics(batch_tensor, dim=0):
    return {
        'mean': batch_tensor.mean(dim=dim),
        'std': batch_tensor.std(dim=dim),
        'var': batch_tensor.var(dim=dim),
        'min': batch_tensor.min(dim=dim)[0],
        'max': batch_tensor.max(dim=dim)[0],
        'median': batch_tensor.median(dim=dim)[0],
        'q25': torch.quantile(batch_tensor, 0.25, dim=dim),
        'q75': torch.quantile(batch_tensor, 0.75, dim=dim)
    }
```

---

## Sampling Methods

**Sampling methods** select subsets of data for training, validation, or analysis. Choice depends on data structure and goals.

### Random Sampling

```python
def uniform_sampling(data, n_samples, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    if n_samples > len(data):
        raise ValueError("Cannot sample more than available")
    indices = torch.randperm(len(data))[:n_samples]
    return data[indices], indices

def uniform_sampling_with_replacement(data, n_samples, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    indices = torch.randint(0, len(data), (n_samples,))
    return data[indices], indices

def weighted_sampling(data, weights, n_samples, replacement=True, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    weights = weights / weights.sum()
    indices = torch.multinomial(weights, n_samples, replacement=replacement)
    return data[indices], indices
```

### Stratified Sampling

```python
def stratified_sampling(data, labels, n_samples_per_class=None, proportional=True, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    unique_classes = torch.unique(labels)
    sampled_indices = []
    if proportional and n_samples_per_class is None:
        class_counts = [(labels == cls).sum().item() for cls in unique_classes]
        total_count = sum(class_counts)
        target_total = len(data) // 10
        n_samples_per_class = [max(1, int(c * target_total / total_count)) for c in class_counts]
    elif n_samples_per_class is None:
        n_samples_per_class = [50] * len(unique_classes)
    elif isinstance(n_samples_per_class, int):
        n_samples_per_class = [n_samples_per_class] * len(unique_classes)
    for i, cls in enumerate(unique_classes):
        class_indices = torch.where(labels == cls)[0]
        n_class = min(n_samples_per_class[i], len(class_indices))
        if n_class > 0:
            perm = torch.randperm(len(class_indices))[:n_class]
            sampled_indices.extend(class_indices[perm].tolist())
    return data[torch.tensor(sampled_indices)], labels[torch.tensor(sampled_indices)], torch.tensor(sampled_indices)
```

### Bootstrap Sampling

```python
def bootstrap_sample(data, n_bootstrap=1000, sample_size=None, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    n = len(data)
    sample_size = sample_size or n
    return [data[torch.randint(0, n, (sample_size,))] for _ in range(n_bootstrap)]

def bootstrap_confidence_interval(data, statistic_fn, confidence=0.95, n_bootstrap=1000, seed=None):
    bootstrap_samples = bootstrap_sample(data, n_bootstrap, seed=seed)
    bootstrap_stats = []
    for sample in bootstrap_samples:
        stat = statistic_fn(sample)
        bootstrap_stats.append(stat.item() if torch.is_tensor(stat) else stat)
    bootstrap_stats = torch.tensor(bootstrap_stats)
    alpha = 1 - confidence
    ci_lower = torch.quantile(bootstrap_stats, alpha / 2)
    ci_upper = torch.quantile(bootstrap_stats, 1 - alpha / 2)
    return ci_lower.item(), ci_upper.item(), bootstrap_stats
```

### Reservoir Sampling

For streaming data of unknown size, **reservoir sampling** maintains a uniform random sample.

```python
import random

def reservoir_sampling(data_stream, k, seed=None):
    if seed is not None:
        random.seed(seed)
    reservoir = []
    for i, item in enumerate(data_stream):
        if i < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    return torch.stack(reservoir) if reservoir else torch.empty(0)
```

---

## Data Transformation Operations

**Data transformation operations** modify values or structure: mathematical transforms, activations, spatial filters, and frequency-domain operations.

### Mathematical Transformations

```python
def log_transform(tensor, base='e', offset=1e-8):
    if base == 'e':
        return torch.log(tensor + offset)
    elif base == 10:
        return torch.log10(tensor + offset)
    elif base == 2:
        return torch.log2(tensor + offset)
    return torch.log(tensor + offset) / math.log(base)

def power_transform(tensor, power=0.5):
    if power == 0:
        return torch.log(tensor + 1e-8)
    return torch.sign(tensor) * torch.pow(torch.abs(tensor), power)

def sqrt_transform(tensor):
    return torch.sqrt(torch.abs(tensor)) * torch.sign(tensor)
```

### Cyclical Encoding

For periodic features (hours, angles), **sin/cos encoding** preserves continuity at boundaries.

```python
def cyclical_encoding(values, period):
    normalized = 2 * math.pi * values / period
    sin_encoded = torch.sin(normalized)
    cos_encoded = torch.cos(normalized)
    return torch.stack([sin_encoded, cos_encoded], dim=-1)
```

### Activation Function Transformations

```python
def apply_activation(tensor, activation='relu'):
    if activation == 'relu':
        return F.relu(tensor)
    elif activation == 'sigmoid':
        return torch.sigmoid(tensor)
    elif activation == 'tanh':
        return torch.tanh(tensor)
    elif activation == 'softmax':
        return F.softmax(tensor, dim=-1)
    elif activation == 'gelu':
        return F.gelu(tensor)
    elif activation == 'swish':
        return tensor * torch.sigmoid(tensor)
    elif activation == 'mish':
        return tensor * torch.tanh(F.softplus(tensor))
    raise ValueError(f"Unknown activation: {activation}")
```

### Frequency Domain Transformations

```python
def fft_transform(signal, dim=-1):
    fft_result = torch.fft.fft(signal, dim=dim)
    magnitude = torch.abs(fft_result)
    phase = torch.angle(fft_result)
    return fft_result, magnitude, phase

def spectrogram(signal, window_size=64, hop_length=32):
    n_frames = (len(signal) - window_size) // hop_length + 1
    spectrogram_data = torch.zeros(n_frames, window_size // 2 + 1)
    window = torch.hann_window(window_size)
    for i in range(n_frames):
        start = i * hop_length
        frame = signal[start:start + window_size] * window
        fft_frame = torch.fft.fft(frame)
        spectrogram_data[i] = torch.abs(fft_frame[:window_size // 2 + 1])
    return spectrogram_data
```

### Rank and Quantile Transforms

```python
def rank_transform(tensor, dim=None):
    if dim is None:
        flat_tensor = tensor.flatten()
        sorted_vals, sorted_indices = torch.sort(flat_tensor)
        ranks = torch.zeros_like(flat_tensor)
        ranks[sorted_indices] = torch.arange(len(flat_tensor), dtype=torch.float32)
        return ranks.reshape(tensor.shape)
    sorted_vals, sorted_indices = torch.sort(tensor, dim=dim)
    ranks = torch.zeros_like(tensor)
    for i in range(tensor.shape[dim]):
        indices = sorted_indices.select(dim, i)
        ranks.scatter_(dim, indices.unsqueeze(dim), torch.tensor(float(i)).expand_as(indices.unsqueeze(dim)))
    return ranks
```

---

## Memory-Efficient Data Operations

**Memory efficiency** is critical for large datasets and limited GPU memory. Strategies include in-place operations, vectorization, chunking, and sparse representations.

### In-Place Operations

In-place operations (suffix `_`) avoid allocating new tensors but can break autograd if the tensor requires gradients.

```python
data = torch.randn(10000, 1000)
data_inplace = data.clone()
data_inplace.mul_(2.0).add_(1.0)
```

### Vectorization vs Loops

**Vectorized operations** leverage optimized kernels and avoid Python loop overhead.

```python
data = torch.randn(10000, 100)
weights = torch.randn(100)
result_vectorized = torch.mv(data, weights)
result_broadcast = (data * weights).sum(dim=1)
```

### Chunked Processing

Process large tensors in chunks to limit peak memory usage.

```python
def chunked_processing(data, chunk_size, process_func):
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:min(i + chunk_size, len(data))]
        results.append(process_func(chunk))
    return torch.cat(results, dim=0)

def chunked_matrix_multiply(A, B, chunk_size=1000):
    m, k = A.shape
    k2, n = B.shape
    assert k == k2
    results = []
    for i in range(0, m, chunk_size):
        end_idx = min(i + chunk_size, m)
        results.append(torch.mm(A[i:end_idx], B))
    return torch.cat(results, dim=0)
```

### Memory-Efficient Dataset

Generate or load data on demand instead of holding everything in memory.

```python
class MemoryEfficientDataset:
    def __init__(self, data_size, feature_dim):
        self.data_size = data_size
        self.feature_dim = feature_dim
        self.metadata = {'mean': torch.zeros(feature_dim), 'std': torch.ones(feature_dim), 'seed': 42}
    
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        torch.manual_seed(self.metadata['seed'] + idx)
        return torch.randn(self.feature_dim) * self.metadata['std'] + self.metadata['mean']
```

### Sparse Tensor Operations

For sparse data, use **torch.sparse** to avoid storing zeros.

```python
def create_sparse_tensor(indices, values, size):
    return torch.sparse_coo_tensor(indices, values, size)

sparse_tensor = torch.sparse_coo_tensor(
    torch.tensor([[0, 1, 2], [2, 0, 1]]),
    torch.tensor([1.0, 2.0, 3.0]),
    (3, 3)
)
dense_result = torch.sparse.mm(sparse_tensor, dense_matrix)
```

### GPU Memory Management

```python
def clear_cache():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if torch.cuda.is_available():
    gpu_tensor = torch.randn(1000, 1000, device='cuda')
    del gpu_tensor
    torch.cuda.empty_cache()
```

### Pre-Allocation and Out Parameters

Avoid repeated allocations by pre-allocating output tensors.

```python
if torch.cuda.is_available():
    a = torch.empty(1000, 1000, device='cuda')
    b = torch.empty(1000, 1000, device='cuda')
    result = torch.empty(1000, 1000, device='cuda')
    a.normal_()
    b.normal_()
    torch.mm(a, b, out=result)
```

### Best Practices Summary

| Category | Recommendation |
|----------|----------------|
| Memory | Use in-place ops, chunk processing, appropriate dtypes |
| Compute | Vectorize, use broadcasting, batch operations |
| GPU | Keep data on GPU, pre-allocate, use mixed precision |
| Data loading | Lazy loading, memory-mapped files, efficient samplers |
