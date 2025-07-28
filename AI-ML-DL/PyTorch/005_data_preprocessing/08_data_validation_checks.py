#!/usr/bin/env python3
"""PyTorch Data Validation Checks - Validating tensor data"""

import torch
import warnings

print("=== Data Validation Overview ===")

print("Validation categories:")
print("1. Shape and dimension validation")
print("2. Data type and range validation")
print("3. Statistical validation")
print("4. Missing data validation")
print("5. Distribution validation")
print("6. Consistency checks")

print("\n=== Shape and Dimension Validation ===")

def validate_shape(tensor, expected_shape, name="tensor"):
    """Validate tensor shape matches expected shape"""
    if tensor.shape != expected_shape:
        raise ValueError(f"{name} shape {tensor.shape} doesn't match expected {expected_shape}")
    return True

def validate_ndim(tensor, expected_ndim, name="tensor"):
    """Validate tensor has expected number of dimensions"""
    if tensor.ndim != expected_ndim:
        raise ValueError(f"{name} has {tensor.ndim} dimensions, expected {expected_ndim}")
    return True

def validate_min_shape(tensor, min_shape, name="tensor"):
    """Validate tensor shape is at least min_shape"""
    if len(tensor.shape) < len(min_shape):
        raise ValueError(f"{name} has fewer dimensions than required")
    
    for i, (actual, minimum) in enumerate(zip(tensor.shape, min_shape)):
        if actual < minimum:
            raise ValueError(f"{name} dimension {i} is {actual}, minimum required is {minimum}")
    return True

# Test shape validation
try:
    test_tensor = torch.randn(3, 4, 5)
    validate_shape(test_tensor, (3, 4, 5), "test_tensor")
    validate_ndim(test_tensor, 3, "test_tensor")
    validate_min_shape(test_tensor, (2, 3, 4), "test_tensor")
    print("✓ Shape validation passed")
except ValueError as e:
    print(f"✗ Shape validation failed: {e}")

print("\n=== Data Type and Range Validation ===")

def validate_dtype(tensor, expected_dtype, name="tensor"):
    """Validate tensor data type"""
    if tensor.dtype != expected_dtype:
        raise ValueError(f"{name} dtype {tensor.dtype} doesn't match expected {expected_dtype}")
    return True

def validate_range(tensor, min_val=None, max_val=None, name="tensor"):
    """Validate tensor values are within specified range"""
    if min_val is not None and tensor.min() < min_val:
        raise ValueError(f"{name} contains values below minimum {min_val}: {tensor.min()}")
    
    if max_val is not None and tensor.max() > max_val:
        raise ValueError(f"{name} contains values above maximum {max_val}: {tensor.max()}")
    
    return True

def validate_positive(tensor, name="tensor"):
    """Validate all tensor values are positive"""
    if tensor.min() <= 0:
        raise ValueError(f"{name} contains non-positive values: minimum is {tensor.min()}")
    return True

def validate_probability(tensor, name="tensor"):
    """Validate tensor represents valid probabilities"""
    validate_range(tensor, 0.0, 1.0, name)
    return True

def validate_probability_distribution(tensor, dim=-1, tolerance=1e-6, name="tensor"):
    """Validate tensor represents probability distributions (sums to 1)"""
    validate_probability(tensor, name)
    
    sums = tensor.sum(dim=dim)
    if not torch.allclose(sums, torch.ones_like(sums), atol=tolerance):
        raise ValueError(f"{name} doesn't sum to 1 along dimension {dim}")
    
    return True

# Test data type and range validation
try:
    float_tensor = torch.randn(10).abs()
    validate_dtype(float_tensor, torch.float32)
    validate_positive(float_tensor)
    
    prob_dist = torch.softmax(torch.randn(5, 10), dim=1)
    validate_probability_distribution(prob_dist, dim=1)
    print("✓ Data type and range validation passed")
except ValueError as e:
    print(f"✗ Validation failed: {e}")

print("\n=== Statistical Validation ===")

def validate_finite(tensor, name="tensor"):
    """Validate tensor contains only finite values"""
    if not torch.isfinite(tensor).all():
        nan_count = torch.isnan(tensor).sum()
        inf_count = torch.isinf(tensor).sum()
        raise ValueError(f"{name} contains {nan_count} NaN and {inf_count} infinite values")
    return True

def validate_mean_range(tensor, min_mean=None, max_mean=None, name="tensor"):
    """Validate tensor mean is within expected range"""
    mean_val = tensor.mean().item()
    
    if min_mean is not None and mean_val < min_mean:
        raise ValueError(f"{name} mean {mean_val:.6f} is below minimum {min_mean}")
    
    if max_mean is not None and mean_val > max_mean:
        raise ValueError(f"{name} mean {mean_val:.6f} is above maximum {max_mean}")
    
    return True

def validate_std_range(tensor, min_std=None, max_std=None, name="tensor"):
    """Validate tensor standard deviation is within expected range"""
    std_val = tensor.std().item()
    
    if min_std is not None and std_val < min_std:
        raise ValueError(f"{name} std {std_val:.6f} is below minimum {min_std}")
    
    if max_std is not None and std_val > max_std:
        raise ValueError(f"{name} std {std_val:.6f} is above maximum {max_std}")
    
    return True

def validate_no_constant(tensor, name="tensor"):
    """Validate tensor is not constant (has variation)"""
    if tensor.std() < 1e-8:
        raise ValueError(f"{name} appears to be constant (std = {tensor.std():.2e})")
    return True

# Test statistical validation
try:
    stats_tensor = torch.randn(1000)
    validate_finite(stats_tensor)
    validate_mean_range(stats_tensor, -0.5, 0.5)
    validate_std_range(stats_tensor, 0.5, 1.5)
    validate_no_constant(stats_tensor)
    print("✓ Statistical validation passed")
except ValueError as e:
    print(f"✗ Statistical validation failed: {e}")

print("\n=== Missing Data Validation ===")

def validate_no_missing(tensor, name="tensor"):
    """Validate tensor has no missing values (NaN or inf)"""
    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isinf(tensor)
    
    if nan_mask.any():
        raise ValueError(f"{name} contains {nan_mask.sum()} NaN values")
    
    if inf_mask.any():
        raise ValueError(f"{name} contains {inf_mask.sum()} infinite values")
    
    return True

def validate_missing_rate(tensor, max_missing_rate=0.1, name="tensor"):
    """Validate missing data rate is below threshold"""
    total_elements = tensor.numel()
    missing_elements = (~torch.isfinite(tensor)).sum().item()
    missing_rate = missing_elements / total_elements
    
    if missing_rate > max_missing_rate:
        raise ValueError(f"{name} missing rate {missing_rate:.2%} exceeds maximum {max_missing_rate:.2%}")
    
    return True

# Test missing data validation
try:
    clean_tensor = torch.randn(100)
    validate_no_missing(clean_tensor)
    
    sparse_missing_tensor = torch.randn(100)
    sparse_missing_tensor[torch.randperm(100)[:5]] = float('nan')  # 5% missing
    validate_missing_rate(sparse_missing_tensor, max_missing_rate=0.1)
    print("✓ Missing data validation passed")
except ValueError as e:
    print(f"✗ Missing data validation failed: {e}")

print("\n=== Distribution Validation ===")

def validate_approximately_normal(tensor, significance_level=0.05, name="tensor"):
    """Validate tensor is approximately normally distributed (simplified test)"""
    # Simple normality check using skewness and kurtosis
    mean = tensor.mean()
    std = tensor.std()
    
    if std == 0:
        warnings.warn(f"{name} has zero variance, cannot test normality")
        return True
    
    # Standardize
    standardized = (tensor - mean) / std
    
    # Check skewness (should be close to 0 for normal distribution)
    skewness = ((standardized ** 3).mean()).abs()
    if skewness > 2:  # Rule of thumb
        warnings.warn(f"{name} may not be normal (high skewness: {skewness:.3f})")
    
    # Check kurtosis (should be close to 3 for normal distribution)
    kurtosis = (standardized ** 4).mean()
    if abs(kurtosis - 3) > 3:  # Rule of thumb
        warnings.warn(f"{name} may not be normal (kurtosis: {kurtosis:.3f})")
    
    return True

def validate_approximately_uniform(tensor, tolerance=0.1, name="tensor"):
    """Validate tensor is approximately uniformly distributed"""
    # Simple uniformity check using range and std
    min_val, max_val = tensor.min(), tensor.max()
    range_val = max_val - min_val
    
    if range_val == 0:
        raise ValueError(f"{name} has zero range, cannot be uniform")
    
    # For uniform distribution, std = range / sqrt(12)
    expected_std = range_val / (12 ** 0.5)
    actual_std = tensor.std()
    
    if abs(actual_std - expected_std) / expected_std > tolerance:
        warnings.warn(f"{name} may not be uniform (expected std: {expected_std:.3f}, actual: {actual_std:.3f})")
    
    return True

# Test distribution validation
normal_tensor = torch.randn(10000)
uniform_tensor = torch.rand(10000)

validate_approximately_normal(normal_tensor, name="normal_tensor")
validate_approximately_uniform(uniform_tensor, name="uniform_tensor")
print("✓ Distribution validation completed")

print("\n=== Consistency Validation ===")

def validate_batch_consistency(tensors, name_prefix="tensor"):
    """Validate batch of tensors have consistent properties"""
    if not tensors:
        raise ValueError("Empty tensor list provided")
    
    # Check dtype consistency
    reference_dtype = tensors[0].dtype
    for i, tensor in enumerate(tensors[1:], 1):
        if tensor.dtype != reference_dtype:
            raise ValueError(f"{name_prefix}[{i}] dtype {tensor.dtype} differs from {name_prefix}[0] dtype {reference_dtype}")
    
    # Check shape consistency (except batch dimension)
    reference_shape = tensors[0].shape[1:]  # Exclude first dimension (batch)
    for i, tensor in enumerate(tensors[1:], 1):
        if tensor.shape[1:] != reference_shape:
            raise ValueError(f"{name_prefix}[{i}] shape {tensor.shape[1:]} differs from expected {reference_shape}")
    
    # Check device consistency
    reference_device = tensors[0].device
    for i, tensor in enumerate(tensors[1:], 1):
        if tensor.device != reference_device:
            raise ValueError(f"{name_prefix}[{i}] device {tensor.device} differs from {name_prefix}[0] device {reference_device}")
    
    return True

def validate_paired_tensors(tensor1, tensor2, name1="tensor1", name2="tensor2"):
    """Validate two tensors are compatible for operations"""
    # Check same number of samples
    if tensor1.shape[0] != tensor2.shape[0]:
        raise ValueError(f"{name1} has {tensor1.shape[0]} samples, {name2} has {tensor2.shape[0]} samples")
    
    # Check device compatibility
    if tensor1.device != tensor2.device:
        raise ValueError(f"{name1} on {tensor1.device}, {name2} on {tensor2.device}")
    
    return True

# Test consistency validation
try:
    batch_tensors = [torch.randn(32, 10, 20) for _ in range(5)]
    validate_batch_consistency(batch_tensors, "batch_tensor")
    
    features = torch.randn(100, 50)
    labels = torch.randint(0, 10, (100,))
    validate_paired_tensors(features, labels, "features", "labels")
    print("✓ Consistency validation passed")
except ValueError as e:
    print(f"✗ Consistency validation failed: {e}")

print("\n=== Comprehensive Validation Suite ===")

class DataValidator:
    """Comprehensive data validation suite"""
    
    def __init__(self, strict=True):
        self.strict = strict
        self.validation_results = {}
    
    def validate_tensor(self, tensor, name="tensor", **kwargs):
        """Run comprehensive validation on a tensor"""
        results = {}
        
        # Basic validations
        try:
            validate_finite(tensor, name)
            results['finite'] = True
        except ValueError as e:
            results['finite'] = False
            results['finite_error'] = str(e)
        
        # Shape validation
        if 'expected_shape' in kwargs:
            try:
                validate_shape(tensor, kwargs['expected_shape'], name)
                results['shape'] = True
            except ValueError as e:
                results['shape'] = False
                results['shape_error'] = str(e)
        
        # Data type validation
        if 'expected_dtype' in kwargs:
            try:
                validate_dtype(tensor, kwargs['expected_dtype'], name)
                results['dtype'] = True
            except ValueError as e:
                results['dtype'] = False
                results['dtype_error'] = str(e)
        
        # Range validation
        if 'min_val' in kwargs or 'max_val' in kwargs:
            try:
                validate_range(tensor, kwargs.get('min_val'), kwargs.get('max_val'), name)
                results['range'] = True
            except ValueError as e:
                results['range'] = False
                results['range_error'] = str(e)
        
        # Statistical validation
        if 'min_mean' in kwargs or 'max_mean' in kwargs:
            try:
                validate_mean_range(tensor, kwargs.get('min_mean'), kwargs.get('max_mean'), name)
                results['mean_range'] = True
            except ValueError as e:
                results['mean_range'] = False
                results['mean_error'] = str(e)
        
        self.validation_results[name] = results
        
        # Handle strict mode
        if self.strict:
            for key, value in results.items():
                if key.endswith('_error'):
                    continue
                if not value:
                    error_key = key + '_error'
                    if error_key in results:
                        raise ValueError(results[error_key])
        
        return results
    
    def validate_dataset(self, dataset_dict, requirements):
        """Validate a complete dataset"""
        all_results = {}
        
        for tensor_name, tensor in dataset_dict.items():
            if tensor_name in requirements:
                reqs = requirements[tensor_name]
                results = self.validate_tensor(tensor, tensor_name, **reqs)
                all_results[tensor_name] = results
        
        return all_results
    
    def get_validation_summary(self):
        """Get summary of validation results"""
        summary = {
            'total_tensors': len(self.validation_results),
            'passed_tensors': 0,
            'failed_tensors': 0,
            'failures': []
        }
        
        for tensor_name, results in self.validation_results.items():
            has_failure = any(key.endswith('_error') for key in results.keys())
            if has_failure:
                summary['failed_tensors'] += 1
                failures = [key for key in results.keys() if key.endswith('_error')]
                summary['failures'].append((tensor_name, failures))
            else:
                summary['passed_tensors'] += 1
        
        return summary

# Test comprehensive validation
validator = DataValidator(strict=False)

# Sample dataset
dataset = {
    'train_images': torch.rand(1000, 3, 32, 32),
    'train_labels': torch.randint(0, 10, (1000,)),
    'val_images': torch.rand(200, 3, 32, 32),
    'val_labels': torch.randint(0, 10, (200,))
}

# Requirements specification
requirements = {
    'train_images': {
        'expected_shape': (1000, 3, 32, 32),
        'expected_dtype': torch.float32,
        'min_val': 0.0,
        'max_val': 1.0
    },
    'train_labels': {
        'expected_dtype': torch.int64,
        'min_val': 0,
        'max_val': 9
    },
    'val_images': {
        'expected_dtype': torch.float32,
        'min_val': 0.0,
        'max_val': 1.0
    },
    'val_labels': {
        'expected_dtype': torch.int64,
        'min_val': 0,
        'max_val': 9
    }
}

validation_results = validator.validate_dataset(dataset, requirements)
summary = validator.get_validation_summary()

print("Validation Summary:")
print(f"  Total tensors: {summary['total_tensors']}")
print(f"  Passed: {summary['passed_tensors']}")
print(f"  Failed: {summary['failed_tensors']}")

if summary['failures']:
    print("  Failures:")
    for tensor_name, failures in summary['failures']:
        print(f"    {tensor_name}: {failures}")

print("\n=== Data Validation Best Practices ===")

print("Validation Guidelines:")
print("1. Validate data early in the pipeline (fail fast)")
print("2. Use appropriate tolerance levels for floating point comparisons")
print("3. Provide clear error messages with context")
print("4. Log validation results for debugging")
print("5. Consider performance impact of validation in production")
print("6. Validate both training and inference data")
print("7. Use different validation levels (strict vs warnings)")

print("\nValidation Checklist:")
print("□ Shape and dimensions match expectations")
print("□ Data types are correct")
print("□ Value ranges are valid")
print("□ No missing or invalid values")
print("□ Statistical properties are reasonable")
print("□ Batch consistency is maintained")
print("□ Device placement is correct")
print("□ Memory usage is within limits")

print("\n=== Data Validation Complete ===") 