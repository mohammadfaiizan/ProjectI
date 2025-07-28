#!/usr/bin/env python3
"""PyTorch Normalization & Standardization - Data normalization techniques"""

import torch
import torch.nn.functional as F
import numpy as np

print("=== Normalization & Standardization Overview ===")

print("Normalization techniques:")
print("1. Min-Max Normalization (Scale to [0,1])")
print("2. Z-Score Standardization (Zero mean, unit variance)")
print("3. Robust Scaling (Using median and IQR)")
print("4. Unit Vector Scaling (L2 normalization)")
print("5. Quantile Normalization")
print("6. Power Transformations")

print("\n=== Min-Max Normalization ===")

# Sample data with different ranges
data = torch.tensor([
    [1.0, 10.0, 100.0],
    [2.0, 20.0, 200.0],
    [3.0, 30.0, 300.0],
    [4.0, 40.0, 400.0],
    [5.0, 50.0, 500.0]
])

print(f"Original data:\n{data}")
print(f"Original ranges: {data.min(dim=0)[0]} to {data.max(dim=0)[0]}")

# Min-Max normalization to [0, 1]
def min_max_normalize(tensor, dim=0, feature_range=(0, 1)):
    """Min-max normalization"""
    min_vals = tensor.min(dim=dim, keepdim=True)[0]
    max_vals = tensor.max(dim=dim, keepdim=True)[0]
    range_vals = max_vals - min_vals
    
    # Avoid division by zero
    range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)
    
    normalized = (tensor - min_vals) / range_vals
    
    # Scale to desired range
    min_target, max_target = feature_range
    normalized = normalized * (max_target - min_target) + min_target
    
    return normalized, (min_vals, max_vals)

minmax_normalized, minmax_params = min_max_normalize(data)
print(f"\nMin-Max normalized:\n{minmax_normalized}")
print(f"New ranges: {minmax_normalized.min(dim=0)[0]} to {minmax_normalized.max(dim=0)[0]}")

# Min-Max to different range [-1, 1]
minmax_neg1_1, _ = min_max_normalize(data, feature_range=(-1, 1))
print(f"\nMin-Max to [-1, 1]:\n{minmax_neg1_1}")

print("\n=== Z-Score Standardization ===")

def z_score_standardize(tensor, dim=0, eps=1e-8):
    """Z-score standardization (zero mean, unit variance)"""
    mean = tensor.mean(dim=dim, keepdim=True)
    std = tensor.std(dim=dim, keepdim=True, unbiased=False)
    
    # Avoid division by zero
    std = torch.where(std < eps, torch.ones_like(std), std)
    
    standardized = (tensor - mean) / std
    return standardized, (mean, std)

zscore_standardized, zscore_params = z_score_standardize(data)
print(f"Z-score standardized:\n{zscore_standardized}")
print(f"Means: {zscore_standardized.mean(dim=0)}")
print(f"Stds: {zscore_standardized.std(dim=0, unbiased=False)}")

print("\n=== Robust Scaling ===")

def robust_scale(tensor, dim=0, eps=1e-8):
    """Robust scaling using median and IQR"""
    median = tensor.median(dim=dim, keepdim=True)[0]
    
    # Calculate IQR (Interquartile Range)
    q25 = torch.quantile(tensor, 0.25, dim=dim, keepdim=True)
    q75 = torch.quantile(tensor, 0.75, dim=dim, keepdim=True)
    iqr = q75 - q25
    
    # Avoid division by zero
    iqr = torch.where(iqr < eps, torch.ones_like(iqr), iqr)
    
    scaled = (tensor - median) / iqr
    return scaled, (median, iqr)

robust_scaled, robust_params = robust_scale(data)
print(f"Robust scaled:\n{robust_scaled}")

# Test with outliers
data_with_outliers = data.clone()
data_with_outliers[0, 2] = 10000  # Add outlier

print(f"\nData with outlier:\n{data_with_outliers}")

zscore_with_outlier, _ = z_score_standardize(data_with_outliers)
robust_with_outlier, _ = robust_scale(data_with_outliers)

print(f"Z-score with outlier (affected):\n{zscore_with_outlier}")
print(f"Robust scale with outlier (robust):\n{robust_with_outlier}")

print("\n=== Unit Vector Scaling (L2 Normalization) ===")

def l2_normalize(tensor, dim=-1, eps=1e-8):
    """L2 normalization (unit vectors)"""
    norm = tensor.norm(dim=dim, keepdim=True)
    norm = torch.where(norm < eps, torch.ones_like(norm), norm)
    return tensor / norm

# Sample vectors
vectors = torch.tensor([
    [3.0, 4.0],
    [1.0, 1.0],
    [5.0, 12.0],
    [8.0, 15.0]
])

l2_normalized = l2_normalize(vectors)
print(f"Original vectors:\n{vectors}")
print(f"L2 normalized:\n{l2_normalized}")
print(f"L2 norms after normalization: {l2_normalized.norm(dim=1)}")

print("\n=== Batch Normalization Implementation ===")

class BatchNorm1d(torch.nn.Module):
    """Custom Batch Normalization implementation"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.weight = torch.nn.Parameter(torch.ones(num_features))
        self.bias = torch.nn.Parameter(torch.zeros(num_features))
        
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
    
    def forward(self, x):
        if self.training:
            # Compute batch statistics
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            
            # Update running statistics
            self.num_batches_tracked += 1
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            # Use batch statistics
            mean = batch_mean
            var = batch_var
        else:
            # Use running statistics
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        return self.weight * x_normalized + self.bias

# Test custom batch norm
batch_data = torch.randn(32, 10)  # batch_size=32, features=10
custom_bn = BatchNorm1d(10)

custom_bn.train()
output_train = custom_bn(batch_data)

custom_bn.eval()
output_eval = custom_bn(batch_data)

print(f"Input batch shape: {batch_data.shape}")
print(f"Output shape: {output_train.shape}")
print(f"Training mode - batch mean: {output_train.mean(dim=0)}")
print(f"Training mode - batch std: {output_train.std(dim=0, unbiased=False)}")

print("\n=== Layer Normalization ===")

def layer_normalize(tensor, normalized_shape, weight=None, bias=None, eps=1e-5):
    """Layer normalization implementation"""
    # Calculate statistics over the last len(normalized_shape) dimensions
    dims_to_normalize = list(range(-len(normalized_shape), 0))
    
    mean = tensor.mean(dim=dims_to_normalize, keepdim=True)
    var = tensor.var(dim=dims_to_normalize, keepdim=True, unbiased=False)
    
    # Normalize
    normalized = (tensor - mean) / torch.sqrt(var + eps)
    
    # Apply learnable parameters if provided
    if weight is not None:
        normalized = normalized * weight
    if bias is not None:
        normalized = normalized + bias
    
    return normalized

# Test layer normalization
sequence_data = torch.randn(4, 10, 128)  # batch, sequence, features
layer_norm_output = layer_normalize(sequence_data, (128,))

print(f"Sequence data shape: {sequence_data.shape}")
print(f"Layer norm output shape: {layer_norm_output.shape}")
print(f"Mean along feature dim: {layer_norm_output.mean(dim=-1)[0]}")  # Should be ~0
print(f"Std along feature dim: {layer_norm_output.std(dim=-1, unbiased=False)[0]}")  # Should be ~1

print("\n=== Instance Normalization ===")

def instance_normalize(tensor, eps=1e-5):
    """Instance normalization for each sample independently"""
    # For 4D tensors (N, C, H, W), normalize over H, W for each sample and channel
    if tensor.dim() == 4:
        dims = [2, 3]
    elif tensor.dim() == 3:
        dims = [2]
    else:
        dims = [1]
    
    mean = tensor.mean(dim=dims, keepdim=True)
    var = tensor.var(dim=dims, keepdim=True, unbiased=False)
    
    return (tensor - mean) / torch.sqrt(var + eps)

# Test instance normalization
image_batch = torch.randn(2, 3, 8, 8)  # batch, channels, height, width
instance_norm_output = instance_normalize(image_batch)

print(f"Image batch shape: {image_batch.shape}")
print(f"Instance norm shape: {instance_norm_output.shape}")

print("\n=== Group Normalization ===")

def group_normalize(tensor, num_groups, eps=1e-5):
    """Group normalization"""
    N, C, H, W = tensor.shape
    assert C % num_groups == 0, f"Channels {C} must be divisible by num_groups {num_groups}"
    
    # Reshape to (N, num_groups, C//num_groups, H, W)
    tensor = tensor.view(N, num_groups, C // num_groups, H, W)
    
    # Normalize over group dimensions
    mean = tensor.mean(dim=[2, 3, 4], keepdim=True)
    var = tensor.var(dim=[2, 3, 4], keepdim=True, unbiased=False)
    normalized = (tensor - mean) / torch.sqrt(var + eps)
    
    # Reshape back to original
    return normalized.view(N, C, H, W)

# Test group normalization
group_norm_output = group_normalize(image_batch, num_groups=1)  # num_groups=1 is equivalent to layer norm
print(f"Group norm shape: {group_norm_output.shape}")

print("\n=== Quantile Normalization ===")

def quantile_normalize(tensor, dim=0):
    """Quantile normalization - makes distributions identical"""
    # Sort along the specified dimension
    sorted_tensor, sort_indices = tensor.sort(dim=dim)
    
    # Calculate ranks
    ranks = torch.argsort(sort_indices, dim=dim).float()
    ranks = ranks / (tensor.shape[dim] - 1)  # Normalize ranks to [0, 1]
    
    # Calculate target quantiles (mean of all sorted columns)
    target_quantiles = sorted_tensor.mean(dim=1-dim, keepdim=True)
    
    # Interpolate to get normalized values
    # For simplicity, we'll use the ranks to index into target quantiles
    normalized = torch.zeros_like(tensor)
    for i in range(tensor.shape[dim]):
        if dim == 0:
            normalized[i] = target_quantiles[int(ranks[i, 0] * (target_quantiles.shape[0] - 1))]
        else:
            normalized[:, i] = target_quantiles[int(ranks[0, i] * (target_quantiles.shape[0] - 1))]
    
    return normalized

# Test quantile normalization (simplified version)
small_data = torch.tensor([[1., 5., 3.], [2., 8., 1.], [3., 2., 4.]])
print(f"Original data for quantile norm:\n{small_data}")

print("\n=== Power Transformations ===")

def box_cox_transform(tensor, lambda_param):
    """Box-Cox transformation"""
    if lambda_param == 0:
        return torch.log(tensor)
    else:
        return (torch.pow(tensor, lambda_param) - 1) / lambda_param

def yeo_johnson_transform(tensor, lambda_param):
    """Yeo-Johnson transformation (handles negative values)"""
    result = torch.zeros_like(tensor)
    
    # Case 1: x >= 0 and lambda != 0
    mask1 = (tensor >= 0) & (lambda_param != 0)
    result[mask1] = (torch.pow(tensor[mask1] + 1, lambda_param) - 1) / lambda_param
    
    # Case 2: x >= 0 and lambda == 0
    mask2 = (tensor >= 0) & (lambda_param == 0)
    result[mask2] = torch.log(tensor[mask2] + 1)
    
    # Case 3: x < 0 and lambda != 2
    mask3 = (tensor < 0) & (lambda_param != 2)
    result[mask3] = -(torch.pow(-tensor[mask3] + 1, 2 - lambda_param) - 1) / (2 - lambda_param)
    
    # Case 4: x < 0 and lambda == 2
    mask4 = (tensor < 0) & (lambda_param == 2)
    result[mask4] = -torch.log(-tensor[mask4] + 1)
    
    return result

# Test power transformations
positive_data = torch.tensor([1., 4., 9., 16., 25.])
mixed_data = torch.tensor([-2., -1., 0., 1., 2., 4., 9.])

box_cox_result = box_cox_transform(positive_data, lambda_param=0.5)
yeo_johnson_result = yeo_johnson_transform(mixed_data, lambda_param=0.5)

print(f"Original positive data: {positive_data}")
print(f"Box-Cox (λ=0.5): {box_cox_result}")
print(f"Original mixed data: {mixed_data}")
print(f"Yeo-Johnson (λ=0.5): {yeo_johnson_result}")

print("\n=== Normalization for Different Data Types ===")

# Image normalization (per-channel)
def normalize_image_batch(images, mean=None, std=None):
    """Normalize image batch with channel-wise statistics"""
    if mean is None:
        mean = images.mean(dim=[0, 2, 3], keepdim=True)
    if std is None:
        std = images.std(dim=[0, 2, 3], keepdim=True)
    
    return (images - mean) / (std + 1e-8)

# Time series normalization
def normalize_time_series(ts, method='z_score', window_size=None):
    """Normalize time series data"""
    if method == 'z_score':
        return (ts - ts.mean()) / ts.std()
    elif method == 'minmax':
        return (ts - ts.min()) / (ts.max() - ts.min())
    elif method == 'rolling' and window_size:
        # Rolling normalization
        normalized = torch.zeros_like(ts)
        for i in range(len(ts)):
            start = max(0, i - window_size + 1)
            window = ts[start:i+1]
            normalized[i] = (ts[i] - window.mean()) / window.std()
        return normalized
    
    return ts

# Test time series normalization
time_series = torch.randn(100) + torch.sin(torch.arange(100) * 0.1)
ts_normalized = normalize_time_series(time_series, method='z_score')
ts_rolling = normalize_time_series(time_series, method='rolling', window_size=10)

print(f"Time series shape: {time_series.shape}")
print(f"Original mean: {time_series.mean():.3f}, std: {time_series.std():.3f}")
print(f"Z-score mean: {ts_normalized.mean():.6f}, std: {ts_normalized.std():.3f}")

print("\n=== Normalization Best Practices ===")

print("Choosing the right normalization:")
print("1. Min-Max: When you need specific range [0,1] or [-1,1]")
print("2. Z-Score: When data is normally distributed")
print("3. Robust: When data has outliers")
print("4. L2: For similarity/distance calculations")
print("5. Batch Norm: During training for internal layers")
print("6. Layer Norm: For sequences/transformers")
print("7. Instance Norm: For style transfer/image generation")

print("\nImplementation Tips:")
print("- Always save normalization parameters for inverse transform")
print("- Apply same normalization to train/val/test splits")
print("- Consider per-feature vs global normalization")
print("- Handle edge cases (zero variance, missing values)")
print("- Use appropriate epsilon values to avoid division by zero")

print("\n=== Normalization & Standardization Complete ===") 