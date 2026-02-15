# Validation, Missing Data, and Outliers

## Table of Contents

- [Handling Missing Values](#handling-missing-values)
- [Data Validation Checks](#data-validation-checks)
- [Outlier Detection with Tensors](#outlier-detection-with-tensors)

---

## Handling Missing Values

**Missing values** appear as NaN (Not a Number) or infinite values. Detection, imputation, and masking are essential before training.

### NaN and Inf Detection

```python
import torch

data = torch.tensor([1.0, 2.0, float('nan'), 4.0, float('inf'), -float('inf'), 7.0])

nan_mask = torch.isnan(data)
inf_mask = torch.isinf(data)
finite_mask = torch.isfinite(data)
invalid_mask = ~finite_mask
```

### Removal and Replacement

**Removal** drops samples or features with missing values. **Replacement** fills with a constant, mean, median, or mode.

```python
def remove_nan_samples(tensor, dim=0):
    if dim == 0:
        valid_mask = ~torch.isnan(tensor).any(dim=1)
        return tensor[valid_mask]
    valid_mask = ~torch.isnan(tensor).any(dim=0)
    return tensor[:, valid_mask]

def replace_nan_with_value(tensor, fill_value=0.0):
    return torch.where(torch.isnan(tensor), fill_value, tensor)

def replace_nan_with_mean(tensor, dim=None):
    if dim is None:
        valid_values = tensor[torch.isfinite(tensor)]
        mean_value = valid_values.mean() if len(valid_values) > 0 else 0.0
        return torch.where(torch.isnan(tensor), mean_value, tensor)
    result = tensor.clone()
    nan_mask = torch.isnan(tensor)
    if dim == 0:
        for col in range(tensor.shape[1]):
            col_data = tensor[:, col]
            valid_values = col_data[torch.isfinite(col_data)]
            if len(valid_values) > 0:
                result[nan_mask[:, col], col] = valid_values.mean()
    return result
```

### Handling Infinite Values

```python
def handle_inf_values(tensor, method='clamp', clamp_value=1e6):
    if method == 'remove':
        return torch.where(torch.isinf(tensor), float('nan'), tensor)
    elif method == 'clamp':
        return torch.clamp(tensor, -clamp_value, clamp_value)
    elif method == 'replace':
        result = tensor.clone()
        result[tensor == float('inf')] = clamp_value
        result[tensor == -float('inf')] = -clamp_value
        return result
    raise ValueError(f"Unknown method: {method}")
```

### Interpolation Methods

For ordered data (e.g., time series), **linear interpolation**, **forward fill**, and **backward fill** preserve local structure.

```python
def linear_interpolate_1d(tensor):
    result = tensor.clone()
    nan_mask = torch.isnan(tensor)
    if not nan_mask.any():
        return result
    valid_indices = torch.where(~nan_mask)[0]
    for i in torch.where(nan_mask)[0]:
        left_idx = valid_indices[valid_indices < i]
        right_idx = valid_indices[valid_indices > i]
        if len(left_idx) > 0 and len(right_idx) > 0:
            left, right = left_idx[-1].item(), right_idx[0].item()
            alpha = (i - left) / (right - left)
            result[i] = (1 - alpha) * tensor[left] + alpha * tensor[right]
        elif len(left_idx) > 0:
            result[i] = tensor[left_idx[-1]]
        elif len(right_idx) > 0:
            result[i] = tensor[right_idx[0]]
    return result

def forward_fill(tensor):
    result = tensor.clone()
    last_valid = None
    for i in range(len(tensor)):
        if torch.isnan(tensor[i]) and last_valid is not None:
            result[i] = last_valid
        else:
            last_valid = tensor[i]
    return result
```

### Statistical Imputation

```python
def statistical_imputation(tensor, method='mean', dim=None):
    result = tensor.clone()
    nan_mask = torch.isnan(tensor)
    if method == 'mean':
        if dim is None:
            valid_values = tensor[torch.isfinite(tensor)]
            fill_value = valid_values.mean() if len(valid_values) > 0 else 0.0
        else:
            fill_value = torch.nanmean(tensor, dim=dim, keepdim=True)
    elif method == 'median':
        valid_values = tensor[torch.isfinite(tensor)]
        fill_value = valid_values.median() if len(valid_values) > 0 else 0.0
    if isinstance(fill_value, torch.Tensor):
        result = torch.where(nan_mask, fill_value.expand_as(tensor), result)
    else:
        result = torch.where(nan_mask, fill_value, result)
    return result
```

### Masking for Sequences

For sequence data, create **attention masks** or **valid token masks** to exclude padded or missing positions from loss computation.

```python
def create_attention_mask(tensor, pad_token=0):
    return (tensor != pad_token).float()

def mask_missing_in_sequences(sequences, missing_value=float('nan')):
    special_token = -999
    masked_sequences = torch.where(torch.isnan(sequences), special_token, sequences)
    valid_mask = ~torch.isnan(sequences)
    return masked_sequences, valid_mask
```

### Fit-Transform Handler

A reusable handler fits on training data and applies the same imputation to validation and test data.

```python
class MissingDataHandler:
    def __init__(self, strategy='mean', fill_value=0.0):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics = {}
    
    def fit(self, tensor, dim=None):
        if self.strategy == 'mean':
            if dim is None:
                valid_values = tensor[torch.isfinite(tensor)]
                self.statistics['fill_value'] = valid_values.mean() if len(valid_values) > 0 else self.fill_value
            else:
                self.statistics['fill_value'] = torch.nanmean(tensor, dim=dim)
        elif self.strategy == 'constant':
            self.statistics['fill_value'] = self.fill_value
    
    def transform(self, tensor):
        result = tensor.clone()
        missing_mask = torch.isnan(tensor) | torch.isinf(tensor)
        fill_value = self.statistics['fill_value']
        if isinstance(fill_value, torch.Tensor):
            fill_value = fill_value.unsqueeze(0).expand_as(tensor) if fill_value.dim() == 1 else fill_value
        result = torch.where(missing_mask, fill_value, result)
        return result
```

---

## Data Validation Checks

**Data validation** ensures tensors meet shape, dtype, range, and statistical assumptions before training or inference.

### Shape and Dimension Validation

```python
def validate_shape(tensor, expected_shape, name="tensor"):
    if tensor.shape != expected_shape:
        raise ValueError(f"{name} shape {tensor.shape} doesn't match expected {expected_shape}")
    return True

def validate_ndim(tensor, expected_ndim, name="tensor"):
    if tensor.ndim != expected_ndim:
        raise ValueError(f"{name} has {tensor.ndim} dimensions, expected {expected_ndim}")
    return True

def validate_min_shape(tensor, min_shape, name="tensor"):
    for i, (actual, minimum) in enumerate(zip(tensor.shape, min_shape)):
        if actual < minimum:
            raise ValueError(f"{name} dimension {i} is {actual}, minimum required is {minimum}")
    return True
```

### Data Type and Range Validation

```python
def validate_dtype(tensor, expected_dtype, name="tensor"):
    if tensor.dtype != expected_dtype:
        raise ValueError(f"{name} dtype {tensor.dtype} doesn't match expected {expected_dtype}")
    return True

def validate_range(tensor, min_val=None, max_val=None, name="tensor"):
    if min_val is not None and tensor.min() < min_val:
        raise ValueError(f"{name} contains values below minimum {min_val}")
    if max_val is not None and tensor.max() > max_val:
        raise ValueError(f"{name} contains values above maximum {max_val}")
    return True

def validate_probability_distribution(tensor, dim=-1, tolerance=1e-6, name="tensor"):
    if (tensor < 0).any() or (tensor > 1).any():
        raise ValueError(f"{name} contains values outside [0, 1]")
    sums = tensor.sum(dim=dim)
    if not torch.allclose(sums, torch.ones_like(sums), atol=tolerance):
        raise ValueError(f"{name} doesn't sum to 1 along dimension {dim}")
    return True
```

### Statistical Validation

```python
def validate_finite(tensor, name="tensor"):
    if not torch.isfinite(tensor).all():
        nan_count = torch.isnan(tensor).sum()
        inf_count = torch.isinf(tensor).sum()
        raise ValueError(f"{name} contains {nan_count} NaN and {inf_count} infinite values")
    return True

def validate_mean_range(tensor, min_mean=None, max_mean=None, name="tensor"):
    mean_val = tensor.mean().item()
    if min_mean is not None and mean_val < min_mean:
        raise ValueError(f"{name} mean {mean_val:.6f} is below minimum {min_mean}")
    if max_mean is not None and mean_val > max_mean:
        raise ValueError(f"{name} mean {mean_val:.6f} is above maximum {max_mean}")
    return True

def validate_no_constant(tensor, name="tensor"):
    if tensor.std() < 1e-8:
        raise ValueError(f"{name} appears to be constant")
    return True
```

### Missing Data Validation

```python
def validate_no_missing(tensor, name="tensor"):
    if torch.isnan(tensor).any():
        raise ValueError(f"{name} contains {torch.isnan(tensor).sum()} NaN values")
    if torch.isinf(tensor).any():
        raise ValueError(f"{name} contains {torch.isinf(tensor).sum()} infinite values")
    return True

def validate_missing_rate(tensor, max_missing_rate=0.1, name="tensor"):
    missing_rate = (~torch.isfinite(tensor)).sum().item() / tensor.numel()
    if missing_rate > max_missing_rate:
        raise ValueError(f"{name} missing rate {missing_rate:.2%} exceeds maximum {max_missing_rate:.2%}")
    return True
```

### Consistency Validation

```python
def validate_batch_consistency(tensors, name_prefix="tensor"):
    reference_dtype = tensors[0].dtype
    reference_shape = tensors[0].shape[1:]
    for i, tensor in enumerate(tensors[1:], 1):
        if tensor.dtype != reference_dtype:
            raise ValueError(f"{name_prefix}[{i}] dtype differs")
        if tensor.shape[1:] != reference_shape:
            raise ValueError(f"{name_prefix}[{i}] shape differs")
    return True

def validate_paired_tensors(tensor1, tensor2, name1="tensor1", name2="tensor2"):
    if tensor1.shape[0] != tensor2.shape[0]:
        raise ValueError(f"{name1} and {name2} have different sample counts")
    if tensor1.device != tensor2.device:
        raise ValueError(f"{name1} and {name2} on different devices")
    return True
```

### Comprehensive Validator

```python
class DataValidator:
    def __init__(self, strict=True):
        self.strict = strict
        self.validation_results = {}
    
    def validate_tensor(self, tensor, name="tensor", **kwargs):
        results = {}
        try:
            validate_finite(tensor, name)
            results['finite'] = True
        except ValueError as e:
            results['finite'] = False
            results['finite_error'] = str(e)
        if 'expected_shape' in kwargs:
            try:
                validate_shape(tensor, kwargs['expected_shape'], name)
                results['shape'] = True
            except ValueError as e:
                results['shape'] = False
                results['shape_error'] = str(e)
        if 'expected_dtype' in kwargs:
            try:
                validate_dtype(tensor, kwargs['expected_dtype'], name)
                results['dtype'] = True
            except ValueError as e:
                results['dtype'] = False
                results['dtype_error'] = str(e)
        self.validation_results[name] = results
        if self.strict and any(not v for k, v in results.items() if not k.endswith('_error')):
            for k, v in results.items():
                if k.endswith('_error'):
                    raise ValueError(v)
        return results
```

---

## Outlier Detection with Tensors

**Outlier detection** identifies values that deviate significantly from the majority. Methods range from simple statistical rules to distance-based and ensemble approaches.

### Z-Score Method

Assumes normal distribution. Points with |z| > threshold are outliers.

```python
def zscore_outliers(tensor, threshold=3.0):
    mean = tensor.mean()
    std = tensor.std()
    if std == 0:
        return torch.zeros_like(tensor, dtype=torch.bool), torch.zeros_like(tensor)
    z_scores = torch.abs((tensor - mean) / std)
    outliers = z_scores > threshold
    return outliers, z_scores
```

### Modified Z-Score (MAD)

Uses **median** and **median absolute deviation** for robustness to outliers.

```python
def modified_zscore_outliers(tensor, threshold=3.5):
    median = tensor.median()
    mad = torch.median(torch.abs(tensor - median))
    if mad == 0:
        return torch.zeros_like(tensor, dtype=torch.bool), torch.zeros_like(tensor)
    modified_z_scores = 0.6745 * (tensor - median) / mad
    outliers = torch.abs(modified_z_scores) > threshold
    return outliers, modified_z_scores
```

### IQR Method

**Interquartile range** defines a non-parametric fence. Values outside [Q1 - k*IQR, Q3 + k*IQR] are outliers.

```python
def iqr_outliers(tensor, multiplier=1.5):
    q1 = torch.quantile(tensor, 0.25)
    q3 = torch.quantile(tensor, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    outliers = (tensor < lower_bound) | (tensor > upper_bound)
    return outliers, (lower_bound, upper_bound)
```

### Distance-Based Methods

**K-nearest neighbor distance** and **Mahalanobis distance** extend outlier detection to multivariate data.

```python
def euclidean_distance_outliers(data, k=5, threshold_percentile=95):
    distances = torch.cdist(data, data, p=2)
    distances.fill_diagonal_(float('inf'))
    knn_distances, _ = torch.topk(distances, k, largest=False, dim=1)
    outlier_scores = knn_distances[:, -1]
    threshold = torch.quantile(outlier_scores, threshold_percentile / 100.0)
    outliers = outlier_scores > threshold
    return outliers, outlier_scores

def mahalanobis_distance_outliers(data, threshold_percentile=95):
    mean = data.mean(dim=0)
    centered_data = data - mean
    cov_matrix = torch.mm(centered_data.T, centered_data) / (data.shape[0] - 1)
    cov_matrix += torch.eye(cov_matrix.shape[0]) * 1e-6
    cov_inv = torch.inverse(cov_matrix)
    distances = torch.zeros(data.shape[0])
    for i in range(data.shape[0]):
        diff = centered_data[i:i+1]
        distances[i] = torch.sqrt(torch.mm(torch.mm(diff, cov_inv), diff.T)).item()
    threshold = torch.quantile(distances, threshold_percentile / 100.0)
    outliers = distances > threshold
    return outliers, distances
```

### Isolation Forest (Simplified)

**Isolation Forest** isolates anomalies by random splits. Anomalies require fewer splits and thus shorter path lengths.

```python
class SimpleIsolationTree:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.size = 0
        self.depth = 0
    
    def fit(self, data, depth=0):
        self.size = data.shape[0]
        self.depth = depth
        if depth >= self.max_depth or data.shape[0] <= 1:
            return self
        self.split_feature = torch.randint(0, data.shape[1], (1,)).item()
        feature_values = data[:, self.split_feature]
        min_val, max_val = feature_values.min(), feature_values.max()
        if min_val == max_val:
            return self
        self.split_value = torch.rand(1, device=data.device) * (max_val - min_val) + min_val
        left_mask = feature_values < self.split_value
        right_mask = ~left_mask
        if left_mask.any():
            self.left = SimpleIsolationTree(self.max_depth)
            self.left.fit(data[left_mask], depth + 1)
        if right_mask.any():
            self.right = SimpleIsolationTree(self.max_depth)
            self.right.fit(data[right_mask], depth + 1)
        return self
    
    def path_length(self, sample):
        if self.split_feature is None or self.size <= 1:
            return self.depth + 2 * (torch.log(torch.tensor(self.size - 1.0)) + 0.5772156649) - 2 * (self.size - 1) / self.size
        if sample[self.split_feature] < self.split_value:
            return self.left.path_length(sample) if self.left else self.depth
        return self.right.path_length(sample) if self.right else self.depth

def isolation_forest_outliers(data, n_trees=100, threshold_percentile=10):
    trees = []
    n_samples = data.shape[0]
    subsample_size = min(256, n_samples)
    for _ in range(n_trees):
        indices = torch.randperm(n_samples)[:subsample_size]
        tree = SimpleIsolationTree(max_depth=int(torch.log2(torch.tensor(subsample_size)).item()))
        tree.fit(data[indices])
        trees.append(tree)
    scores = torch.tensor([sum(t.path_length(data[i]) for t in trees) / n_trees for i in range(n_samples)])
    c = 2 * (torch.log(torch.tensor(subsample_size - 1.0)) + 0.5772156649) - 2 * (subsample_size - 1) / subsample_size
    anomaly_scores = torch.pow(2, -scores / c)
    threshold = torch.quantile(anomaly_scores, 1 - threshold_percentile / 100.0)
    outliers = anomaly_scores > threshold
    return outliers, anomaly_scores
```

### Time Series Outlier Detection

Rolling statistics or seasonal decomposition can detect anomalies in temporal data.

```python
def time_series_outliers(ts_data, window_size=10, threshold=3.0):
    n_points = len(ts_data)
    outliers = torch.zeros(n_points, dtype=torch.bool)
    scores = torch.zeros(n_points)
    for i in range(window_size, n_points):
        window = ts_data[i-window_size:i]
        window_mean = window.mean()
        window_std = window.std()
        if window_std > 0:
            z_score = abs((ts_data[i] - window_mean) / window_std)
            scores[i] = z_score
            outliers[i] = z_score > threshold
    return outliers, scores
```

### Ensemble Outlier Detection

Combining multiple methods via voting reduces false positives and improves robustness.

```python
class OutlierEnsemble:
    def __init__(self, methods=None, voting='majority'):
        self.methods = methods or ['zscore', 'iqr']
        self.voting = voting
    
    def detect_outliers(self, data, **kwargs):
        outlier_masks = {}
        flat_data = data.flatten() if data.dim() > 1 else data
        if 'zscore' in self.methods:
            mask, _ = zscore_outliers(flat_data, kwargs.get('zscore_threshold', 3.0))
            outlier_masks['zscore'] = mask.view(data.shape[0], -1).any(dim=1) if data.dim() > 1 else mask
        if 'iqr' in self.methods:
            mask, _ = iqr_outliers(flat_data, kwargs.get('iqr_multiplier', 1.5))
            outlier_masks['iqr'] = mask.view(data.shape[0], -1).any(dim=1) if data.dim() > 1 else mask
        votes = torch.stack(list(outlier_masks.values()), dim=0)
        if self.voting == 'majority':
            ensemble_outliers = votes.float().mean(dim=0) > 0.5
        elif self.voting == 'unanimous':
            ensemble_outliers = votes.all(dim=0)
        else:
            ensemble_outliers = votes.any(dim=0)
        return ensemble_outliers, outlier_masks
```
