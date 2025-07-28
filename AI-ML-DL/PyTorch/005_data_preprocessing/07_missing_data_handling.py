#!/usr/bin/env python3
"""PyTorch Missing Data Handling - Handling NaN, inf values"""

import torch
import torch.nn.functional as F
import math

print("=== Missing Data Handling Overview ===")

print("Types of missing/invalid data:")
print("1. NaN (Not a Number) values")
print("2. Infinite values (positive and negative)")
print("3. Missing entries in sequences")
print("4. Corrupted data points")
print("5. Out-of-range values")

print("\n=== Detecting Missing Data ===")

# Create data with missing values
data = torch.tensor([1.0, 2.0, float('nan'), 4.0, float('inf'), -float('inf'), 7.0])
print(f"Data with missing values: {data}")

# Detection methods
nan_mask = torch.isnan(data)
inf_mask = torch.isinf(data)
finite_mask = torch.isfinite(data)

print(f"NaN mask: {nan_mask}")
print(f"Inf mask: {inf_mask}")
print(f"Finite mask: {finite_mask}")

# Combined detection
invalid_mask = ~finite_mask
print(f"Invalid data mask: {invalid_mask}")
print(f"Number of invalid values: {invalid_mask.sum().item()}")

print("\n=== Handling NaN Values ===")

def remove_nan_samples(tensor, dim=0):
    """Remove samples containing NaN values"""
    if dim == 0:
        # Remove rows with any NaN
        valid_mask = ~torch.isnan(tensor).any(dim=1)
        return tensor[valid_mask]
    else:
        # Remove columns with any NaN
        valid_mask = ~torch.isnan(tensor).any(dim=0)
        return tensor[:, valid_mask]

def replace_nan_with_value(tensor, fill_value=0.0):
    """Replace NaN values with a specific value"""
    return torch.where(torch.isnan(tensor), fill_value, tensor)

def replace_nan_with_mean(tensor, dim=None):
    """Replace NaN values with mean of valid values"""
    if dim is None:
        # Global mean
        valid_values = tensor[torch.isfinite(tensor)]
        if len(valid_values) == 0:
            return tensor
        mean_value = valid_values.mean()
        return torch.where(torch.isnan(tensor), mean_value, tensor)
    else:
        # Mean along specific dimension
        result = tensor.clone()
        nan_mask = torch.isnan(tensor)
        
        if dim == 0:
            # Column-wise mean
            for col in range(tensor.shape[1]):
                col_data = tensor[:, col]
                valid_values = col_data[torch.isfinite(col_data)]
                if len(valid_values) > 0:
                    col_mean = valid_values.mean()
                    result[nan_mask[:, col], col] = col_mean
        elif dim == 1:
            # Row-wise mean
            for row in range(tensor.shape[0]):
                row_data = tensor[row, :]
                valid_values = row_data[torch.isfinite(row_data)]
                if len(valid_values) > 0:
                    row_mean = valid_values.mean()
                    result[row, nan_mask[row, :]] = row_mean
        
        return result

# Test NaN handling
matrix_with_nan = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, float('nan'), 6.0],
    [7.0, 8.0, float('nan')],
    [10.0, 11.0, 12.0]
])

print(f"Original matrix:\n{matrix_with_nan}")

# Remove samples with NaN
clean_samples = remove_nan_samples(matrix_with_nan)
print(f"Samples without NaN:\n{clean_samples}")

# Replace with zero
zero_filled = replace_nan_with_value(matrix_with_nan, fill_value=0.0)
print(f"NaN replaced with 0:\n{zero_filled}")

# Replace with mean
mean_filled = replace_nan_with_mean(matrix_with_nan)
print(f"NaN replaced with global mean:\n{mean_filled}")

# Replace with column mean
col_mean_filled = replace_nan_with_mean(matrix_with_nan, dim=0)
print(f"NaN replaced with column mean:\n{col_mean_filled}")

print("\n=== Handling Infinite Values ===")

def handle_inf_values(tensor, method='clamp', clamp_value=1e6):
    """Handle infinite values in tensor"""
    if method == 'remove':
        # Remove infinite values (replace with NaN for removal)
        return torch.where(torch.isinf(tensor), float('nan'), tensor)
    elif method == 'clamp':
        # Clamp infinite values
        return torch.clamp(tensor, -clamp_value, clamp_value)
    elif method == 'replace':
        # Replace with specific value
        pos_inf_mask = tensor == float('inf')
        neg_inf_mask = tensor == -float('inf')
        result = tensor.clone()
        result[pos_inf_mask] = clamp_value
        result[neg_inf_mask] = -clamp_value
        return result
    else:
        raise ValueError(f"Unknown method: {method}")

# Test infinite value handling
data_with_inf = torch.tensor([1.0, 2.0, float('inf'), 4.0, -float('inf'), 6.0])
print(f"Data with inf: {data_with_inf}")

clamped = handle_inf_values(data_with_inf, method='clamp', clamp_value=100.0)
replaced = handle_inf_values(data_with_inf, method='replace', clamp_value=999.0)

print(f"Clamped inf values: {clamped}")
print(f"Replaced inf values: {replaced}")

print("\n=== Interpolation Methods ===")

def linear_interpolate_1d(tensor):
    """Linear interpolation for 1D tensor with NaN values"""
    result = tensor.clone()
    nan_mask = torch.isnan(tensor)
    
    if not nan_mask.any():
        return result
    
    # Find valid indices
    valid_indices = torch.where(~nan_mask)[0]
    
    if len(valid_indices) < 2:
        # Not enough valid points for interpolation
        return result
    
    # Interpolate NaN values
    for i in torch.where(nan_mask)[0]:
        # Find nearest valid points
        left_idx = valid_indices[valid_indices < i]
        right_idx = valid_indices[valid_indices > i]
        
        if len(left_idx) > 0 and len(right_idx) > 0:
            # Interpolate between left and right
            left = left_idx[-1].item()
            right = right_idx[0].item()
            
            # Linear interpolation
            alpha = (i - left) / (right - left)
            interpolated = (1 - alpha) * tensor[left] + alpha * tensor[right]
            result[i] = interpolated
        elif len(left_idx) > 0:
            # Use last valid value (forward fill)
            result[i] = tensor[left_idx[-1]]
        elif len(right_idx) > 0:
            # Use next valid value (backward fill)
            result[i] = tensor[right_idx[0]]
    
    return result

def forward_fill(tensor):
    """Forward fill missing values"""
    result = tensor.clone()
    last_valid = None
    
    for i in range(len(tensor)):
        if torch.isnan(tensor[i]):
            if last_valid is not None:
                result[i] = last_valid
        else:
            last_valid = tensor[i]
    
    return result

def backward_fill(tensor):
    """Backward fill missing values"""
    result = tensor.clone()
    next_valid = None
    
    for i in range(len(tensor) - 1, -1, -1):
        if torch.isnan(tensor[i]):
            if next_valid is not None:
                result[i] = next_valid
        else:
            next_valid = tensor[i]
    
    return result

# Test interpolation
sequence_with_gaps = torch.tensor([1.0, 2.0, float('nan'), float('nan'), 5.0, float('nan'), 7.0])
print(f"Sequence with gaps: {sequence_with_gaps}")

interpolated = linear_interpolate_1d(sequence_with_gaps)
forward_filled = forward_fill(sequence_with_gaps)
backward_filled = backward_fill(sequence_with_gaps)

print(f"Linear interpolated: {interpolated}")
print(f"Forward filled: {forward_filled}")
print(f"Backward filled: {backward_filled}")

print("\n=== Missing Data in Sequences ===")

def create_attention_mask(tensor, pad_token=0):
    """Create attention mask for padded sequences"""
    return (tensor != pad_token).float()

def mask_missing_in_sequences(sequences, missing_value=float('nan')):
    """Handle missing values in sequence data"""
    # Replace missing values with a special token
    special_token = -999  # Choose a value not in your vocabulary
    masked_sequences = torch.where(torch.isnan(sequences), special_token, sequences)
    
    # Create mask for valid tokens
    valid_mask = ~torch.isnan(sequences)
    
    return masked_sequences, valid_mask

# Test sequence masking
batch_sequences = torch.tensor([
    [1.0, 2.0, 3.0, float('nan'), 5.0],
    [6.0, float('nan'), 8.0, 9.0, 10.0],
    [11.0, 12.0, float('nan'), float('nan'), 15.0]
])

masked_seqs, valid_masks = mask_missing_in_sequences(batch_sequences)
print(f"Original sequences:\n{batch_sequences}")
print(f"Masked sequences:\n{masked_seqs}")
print(f"Valid masks:\n{valid_masks}")

print("\n=== Statistical Imputation ===")

def statistical_imputation(tensor, method='mean', dim=None):
    """Impute missing values using statistical methods"""
    result = tensor.clone()
    nan_mask = torch.isnan(tensor)
    
    if method == 'mean':
        if dim is None:
            valid_values = tensor[torch.isfinite(tensor)]
            fill_value = valid_values.mean() if len(valid_values) > 0 else 0.0
        else:
            fill_value = torch.nanmean(tensor, dim=dim, keepdim=True)
    
    elif method == 'median':
        if dim is None:
            valid_values = tensor[torch.isfinite(tensor)]
            fill_value = valid_values.median() if len(valid_values) > 0 else 0.0
        else:
            # PyTorch doesn't have nanmedian, so we implement it
            fill_value = tensor.clone()
            if dim == 0:
                for col in range(tensor.shape[1]):
                    col_data = tensor[:, col]
                    valid_col = col_data[torch.isfinite(col_data)]
                    if len(valid_col) > 0:
                        fill_value[:, col] = valid_col.median()
            # Similar for other dimensions
    
    elif method == 'mode':
        # Simple mode implementation
        if dim is None:
            valid_values = tensor[torch.isfinite(tensor)]
            if len(valid_values) > 0:
                fill_value = valid_values.mode().values
            else:
                fill_value = 0.0
        else:
            fill_value = tensor.mode(dim=dim, keepdim=True).values
    
    # Fill missing values
    if isinstance(fill_value, torch.Tensor):
        result = torch.where(nan_mask, fill_value.expand_as(tensor), result)
    else:
        result = torch.where(nan_mask, fill_value, result)
    
    return result

# Test statistical imputation
data_for_imputation = torch.tensor([
    [1.0, 4.0, 7.0],
    [2.0, float('nan'), 8.0],
    [3.0, 6.0, float('nan')],
    [float('nan'), 5.0, 9.0]
])

mean_imputed = statistical_imputation(data_for_imputation, method='mean')
print(f"Original data:\n{data_for_imputation}")
print(f"Mean imputed:\n{mean_imputed}")

print("\n=== Advanced Missing Data Handling ===")

class MissingDataHandler:
    """Comprehensive missing data handler"""
    
    def __init__(self, strategy='mean', fill_value=0.0):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics = {}
    
    def fit(self, tensor, dim=None):
        """Fit the handler on training data"""
        if self.strategy == 'mean':
            if dim is None:
                valid_values = tensor[torch.isfinite(tensor)]
                self.statistics['fill_value'] = valid_values.mean() if len(valid_values) > 0 else self.fill_value
            else:
                # Compute mean along specified dimension
                self.statistics['fill_value'] = torch.nanmean(tensor, dim=dim)
        
        elif self.strategy == 'median':
            if dim is None:
                valid_values = tensor[torch.isfinite(tensor)]
                self.statistics['fill_value'] = valid_values.median() if len(valid_values) > 0 else self.fill_value
            else:
                # Implement per-feature median
                if dim == 0:
                    medians = []
                    for col in range(tensor.shape[1]):
                        col_data = tensor[:, col]
                        valid_col = col_data[torch.isfinite(col_data)]
                        if len(valid_col) > 0:
                            medians.append(valid_col.median())
                        else:
                            medians.append(self.fill_value)
                    self.statistics['fill_value'] = torch.tensor(medians)
        
        elif self.strategy == 'constant':
            self.statistics['fill_value'] = self.fill_value
    
    def transform(self, tensor):
        """Transform tensor by filling missing values"""
        if 'fill_value' not in self.statistics:
            raise ValueError("Handler must be fitted before transform")
        
        result = tensor.clone()
        nan_mask = torch.isnan(tensor)
        inf_mask = torch.isinf(tensor)
        missing_mask = nan_mask | inf_mask
        
        fill_value = self.statistics['fill_value']
        
        if isinstance(fill_value, torch.Tensor):
            if fill_value.dim() == 1 and tensor.dim() == 2:
                # Broadcast feature-wise fill values
                fill_value = fill_value.unsqueeze(0).expand_as(tensor)
            result = torch.where(missing_mask, fill_value, result)
        else:
            result = torch.where(missing_mask, fill_value, result)
        
        return result
    
    def fit_transform(self, tensor, dim=None):
        """Fit and transform in one step"""
        self.fit(tensor, dim)
        return self.transform(tensor)

# Test advanced handler
handler = MissingDataHandler(strategy='mean')
train_data = torch.tensor([
    [1.0, 4.0, 7.0, 10.0],
    [2.0, 5.0, 8.0, 11.0],
    [3.0, 6.0, 9.0, 12.0]
])

test_data = torch.tensor([
    [float('nan'), 4.5, 7.5, float('inf')],
    [2.5, float('nan'), 8.5, 11.5]
])

# Fit on training data
handler.fit(train_data, dim=0)
print(f"Fitted statistics: {handler.statistics}")

# Transform test data
transformed_test = handler.transform(test_data)
print(f"Test data with missing:\n{test_data}")
print(f"Transformed test data:\n{transformed_test}")

print("\n=== Missing Data Validation ===")

def validate_no_missing(tensor, name="tensor"):
    """Validate that tensor has no missing values"""
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()
    
    if nan_count > 0:
        print(f"Warning: {name} contains {nan_count} NaN values")
    if inf_count > 0:
        print(f"Warning: {name} contains {inf_count} infinite values")
    
    return nan_count == 0 and inf_count == 0

def missing_data_report(tensor):
    """Generate comprehensive missing data report"""
    total_elements = tensor.numel()
    nan_count = torch.isnan(tensor).sum().item()
    pos_inf_count = (tensor == float('inf')).sum().item()
    neg_inf_count = (tensor == -float('inf')).sum().item()
    valid_count = torch.isfinite(tensor).sum().item()
    
    report = {
        'total_elements': total_elements,
        'valid_elements': valid_count,
        'nan_count': nan_count,
        'positive_inf': pos_inf_count,
        'negative_inf': neg_inf_count,
        'missing_percentage': (total_elements - valid_count) / total_elements * 100
    }
    
    return report

# Test validation
problematic_data = torch.tensor([
    [1.0, 2.0, float('nan')],
    [4.0, float('inf'), 6.0],
    [7.0, 8.0, -float('inf')]
])

is_clean = validate_no_missing(problematic_data, "problematic_data")
report = missing_data_report(problematic_data)

print(f"Data is clean: {is_clean}")
print("Missing data report:")
for key, value in report.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.2f}")
    else:
        print(f"  {key}: {value}")

print("\n=== Missing Data Best Practices ===")

print("Missing Data Guidelines:")
print("1. Always check for missing data before training")
print("2. Understand the missingness pattern (random vs systematic)")
print("3. Choose appropriate imputation strategy for your domain")
print("4. Consider the impact of missing data on model performance")
print("5. Document your missing data handling approach")
print("6. Validate that missing data handling preserves data distribution")
print("7. Consider using masks instead of imputation when appropriate")

print("\nImputation Strategy Selection:")
print("- Mean/Median: For numerical features with random missingness")
print("- Mode: For categorical features")
print("- Forward/Backward fill: For time series data")
print("- Interpolation: For smooth time series")
print("- Domain-specific: Use domain knowledge when available")
print("- Multiple imputation: For high missing data rates")

print("\n=== Missing Data Handling Complete ===") 