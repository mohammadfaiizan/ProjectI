# Preprocessing, Normalization, and Scaling

## Table of Contents

- [Basic Tensor Preprocessing](#basic-tensor-preprocessing)
- [Normalization Methods](#normalization-methods)
- [Standardization](#standardization)
- [Feature Scaling](#feature-scaling)

---

## Basic Tensor Preprocessing

**Basic tensor preprocessing** encompasses cleaning, type preparation, and initial transformations applied to raw data before model training. Common operations include type conversions, device management, shape transformations, value range normalization, and data quality checks.

### Type Conversions and Value Range Preparation

Raw data often arrives in integer formats (e.g., uint8 for images). Converting to float and normalizing to a standard range is essential for neural network compatibility.

```python
import torch
import torch.nn.functional as F

raw_data = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
normalized_data = raw_data.float() / 255.0

mean = normalized_data.mean()
std = normalized_data.std()
standardized_data = (normalized_data - mean) / std
```

### Batch Preprocessing

For batch processing, **per-sample normalization** computes statistics across each sample independently, while **batch-wise normalization** computes statistics across the batch dimension.

```python
batch_size = 8
batch_data = torch.randint(0, 256, (batch_size, 3, 32, 32), dtype=torch.uint8)
batch_normalized = batch_data.float() / 255.0

per_sample_mean = batch_normalized.view(batch_size, -1).mean(dim=1, keepdim=True)
per_sample_std = batch_normalized.view(batch_size, -1).std(dim=1, keepdim=True)
batch_standardized = (batch_normalized.view(batch_size, -1) - per_sample_mean) / (per_sample_std + 1e-8)
batch_standardized = batch_standardized.view(batch_size, 3, 32, 32)
```

### Image Preprocessing Pipeline

A complete image preprocessing pipeline typically includes dtype conversion, pixel normalization, resizing, and channel-wise standardization using dataset-specific statistics (e.g., ImageNet).

```python
def preprocess_image_tensor(tensor, target_size=(224, 224), normalize=True):
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    
    if tensor.shape[-2:] != target_size:
        tensor = F.interpolate(tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
        tensor = tensor.squeeze(0)
    
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
    
    return tensor
```

### Tabular Data Preprocessing

**Continuous features** require scaling; **categorical features** require encoding (e.g., one-hot encoding). Combining both yields a unified feature representation.

```python
def scale_features(features, method='standard'):
    if method == 'standard':
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)
        return (features - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_vals = features.min(dim=0, keepdim=True)[0]
        max_vals = features.max(dim=0, keepdim=True)[0]
        return (features - min_vals) / (max_vals - min_vals + 1e-8)
    elif method == 'robust':
        median = features.median(dim=0, keepdim=True)[0]
        mad = torch.median(torch.abs(features - median), dim=0, keepdim=True)[0]
        return (features - median) / (mad + 1e-8)

categorical_onehot = F.one_hot(categorical_features, num_classes=5).float()
combined_features = torch.cat([scaled_continuous, categorical_onehot.view(num_samples, -1)], dim=1)
```

---

## Normalization Methods

**Normalization** rescales data to a standard range or distribution. The choice depends on data characteristics and downstream algorithms.

### Min-Max Normalization

**Min-max normalization** scales values to a specified range (typically [0, 1] or [-1, 1]). It is sensitive to outliers because min and max define the scaling.

```python
def min_max_normalize(tensor, dim=0, feature_range=(0, 1)):
    min_vals = tensor.min(dim=dim, keepdim=True)[0]
    max_vals = tensor.max(dim=dim, keepdim=True)[0]
    range_vals = max_vals - min_vals
    range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)
    
    normalized = (tensor - min_vals) / range_vals
    min_target, max_target = feature_range
    normalized = normalized * (max_target - min_target) + min_target
    
    return normalized, (min_vals, max_vals)
```

### Z-Score Standardization

**Z-score standardization** transforms data to zero mean and unit variance. It assumes approximate normality and is widely used for neural networks.

```python
def z_score_standardize(tensor, dim=0, eps=1e-8):
    mean = tensor.mean(dim=dim, keepdim=True)
    std = tensor.std(dim=dim, keepdim=True, unbiased=False)
    std = torch.where(std < eps, torch.ones_like(std), std)
    standardized = (tensor - mean) / std
    return standardized, (mean, std)
```

### Robust Scaling

**Robust scaling** uses median and IQR (interquartile range), making it resistant to outliers compared to mean and standard deviation.

```python
def robust_scale(tensor, dim=0, eps=1e-8):
    median = tensor.median(dim=dim, keepdim=True)[0]
    q25 = torch.quantile(tensor, 0.25, dim=dim, keepdim=True)
    q75 = torch.quantile(tensor, 0.75, dim=dim, keepdim=True)
    iqr = q75 - q25
    iqr = torch.where(iqr < eps, torch.ones_like(iqr), iqr)
    scaled = (tensor - median) / iqr
    return scaled, (median, iqr)
```

### L2 Normalization (Unit Vector Scaling)

**L2 normalization** scales each sample to unit norm. Useful for similarity and distance-based algorithms.

```python
def l2_normalize(tensor, dim=-1, eps=1e-8):
    norm = tensor.norm(dim=dim, keepdim=True)
    norm = torch.where(norm < eps, torch.ones_like(norm), norm)
    return tensor / norm
```

### Layer and Instance Normalization

**Layer normalization** normalizes over the last dimensions (e.g., feature dimension in sequences). **Instance normalization** normalizes each sample and channel independently, common in style transfer.

```python
def layer_normalize(tensor, normalized_shape, eps=1e-5):
    dims_to_normalize = list(range(-len(normalized_shape), 0))
    mean = tensor.mean(dim=dims_to_normalize, keepdim=True)
    var = tensor.var(dim=dims_to_normalize, keepdim=True, unbiased=False)
    return (tensor - mean) / torch.sqrt(var + eps)

def instance_normalize(tensor, eps=1e-5):
    dims = [2, 3] if tensor.dim() == 4 else [2] if tensor.dim() == 3 else [1]
    mean = tensor.mean(dim=dims, keepdim=True)
    var = tensor.var(dim=dims, keepdim=True, unbiased=False)
    return (tensor - mean) / torch.sqrt(var + eps)
```

---

## Standardization

**Standardization** refers specifically to z-score transformation: centering at zero and scaling by standard deviation. It is the most common preprocessing for neural networks.

| Method | Formula | Use Case |
|--------|---------|----------|
| Z-Score | (x - mean) / std | Normally distributed data |
| Per-channel | (x - mean_c) / std_c | Image data (NCHW) |
| Per-sample | (x - mean_n) / std_n | Variable-length sequences |

```python
def normalize_image_batch(images, mean=None, std=None):
    if mean is None:
        mean = images.mean(dim=[0, 2, 3], keepdim=True)
    if std is None:
        std = images.std(dim=[0, 2, 3], keepdim=True)
    return (images - mean) / (std + 1e-8)
```

---

## Feature Scaling

**Feature scaling** in PyTorch follows patterns similar to scikit-learn scalers: fit on training data, transform training and inference data with the same parameters.

### StandardScaler-like Pattern

```python
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False
    
    def fit(self, X):
        self.mean = X.mean(dim=0, keepdim=True)
        self.std = X.std(dim=0, keepdim=True, unbiased=False)
        self.std = torch.where(self.std < 1e-8, torch.ones_like(self.std), self.std)
        self.fitted = True
        return self
    
    def transform(self, X):
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        return (X - self.mean) / self.std
    
    def inverse_transform(self, X_scaled):
        return X_scaled * self.std + self.mean
```

### MinMaxScaler-like Pattern

```python
class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_vals = None
        self.max_vals = None
        self.scale_range = None
        self.fitted = False
    
    def fit(self, X):
        self.min_vals = X.min(dim=0, keepdim=True)[0]
        self.max_vals = X.max(dim=0, keepdim=True)[0]
        self.scale_range = self.max_vals - self.min_vals
        self.scale_range = torch.where(self.scale_range < 1e-8, torch.ones_like(self.scale_range), self.scale_range)
        self.fitted = True
        return self
    
    def transform(self, X):
        scaled = (X - self.min_vals) / self.scale_range
        min_target, max_target = self.feature_range
        return scaled * (max_target - min_target) + min_target
```

### RobustScaler-like Pattern

```python
class RobustScaler:
    def __init__(self):
        self.median = None
        self.iqr = None
        self.fitted = False
    
    def fit(self, X):
        self.median = X.median(dim=0, keepdim=True)[0]
        q25 = torch.quantile(X, 0.25, dim=0, keepdim=True)
        q75 = torch.quantile(X, 0.75, dim=0, keepdim=True)
        self.iqr = q75 - q25
        self.iqr = torch.where(self.iqr < 1e-8, torch.ones_like(self.iqr), self.iqr)
        self.fitted = True
        return self
    
    def transform(self, X):
        return (X - self.median) / self.iqr
```

### Power Transformations

**Box-Cox** and **Yeo-Johnson** transformations reduce skewness. Box-Cox requires positive values; Yeo-Johnson handles negative values.

```python
def box_cox_transform(tensor, lambda_param):
    if lambda_param == 0:
        return torch.log(tensor)
    return (torch.pow(tensor, lambda_param) - 1) / lambda_param

def yeo_johnson_transform(tensor, lambda_param):
    result = torch.zeros_like(tensor)
    mask1 = (tensor >= 0) & (lambda_param != 0)
    result[mask1] = (torch.pow(tensor[mask1] + 1, lambda_param) - 1) / lambda_param
    mask2 = (tensor >= 0) & (lambda_param == 0)
    result[mask2] = torch.log(tensor[mask2] + 1)
    mask3 = (tensor < 0) & (lambda_param != 2)
    result[mask3] = -(torch.pow(-tensor[mask3] + 1, 2 - lambda_param) - 1) / (2 - lambda_param)
    mask4 = (tensor < 0) & (lambda_param == 2)
    result[mask4] = -torch.log(-tensor[mask4] + 1)
    return result
```

### Best Practices

- **Fit scalers on training data only** to avoid data leakage
- **Apply the same transformation** to validation and test data
- **Save scaler parameters** for inference deployment
- **Handle zero variance** with epsilon values to avoid division by zero
- **Choose scaling method** based on data distribution and algorithm sensitivity
