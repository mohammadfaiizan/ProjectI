"""
PyTorch Data Validation Pipeline - Data Quality and Integrity
Comprehensive guide to validating data integrity and quality in PyTorch pipelines
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional, Callable, Union
import warnings
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import json
import time

print("=== DATA VALIDATION PIPELINE ===")

# 1. VALIDATION SCHEMA DEFINITION
print("\n1. VALIDATION SCHEMA DEFINITION")

class ValidationSeverity(Enum):
    """Validation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Result of a validation check"""
    passed: bool
    severity: ValidationSeverity
    message: str
    details: Optional[Dict[str, Any]] = None

class ValidationRule(ABC):
    """Abstract base class for validation rules"""
    
    def __init__(self, name: str, severity: ValidationSeverity = ValidationSeverity.ERROR):
        self.name = name
        self.severity = severity
        
    @abstractmethod
    def validate(self, data: Any) -> ValidationResult:
        """Validate data against this rule"""
        pass

# 2. TENSOR VALIDATION RULES
print("\n2. TENSOR VALIDATION RULES")

class TensorShapeRule(ValidationRule):
    """Validate tensor shape"""
    
    def __init__(self, expected_shape: Tuple[int, ...], name: str = "shape_check"):
        super().__init__(name)
        self.expected_shape = expected_shape
        
    def validate(self, data: torch.Tensor) -> ValidationResult:
        if not isinstance(data, torch.Tensor):
            return ValidationResult(
                False, self.severity, 
                f"Expected tensor, got {type(data)}"
            )
            
        if data.shape != self.expected_shape:
            return ValidationResult(
                False, self.severity,
                f"Shape mismatch: expected {self.expected_shape}, got {data.shape}"
            )
            
        return ValidationResult(True, ValidationSeverity.INFO, "Shape validation passed")

class TensorRangeRule(ValidationRule):
    """Validate tensor value range"""
    
    def __init__(self, min_val: float = None, max_val: float = None, name: str = "range_check"):
        super().__init__(name)
        self.min_val = min_val
        self.max_val = max_val
        
    def validate(self, data: torch.Tensor) -> ValidationResult:
        if not isinstance(data, torch.Tensor):
            return ValidationResult(
                False, self.severity,
                f"Expected tensor, got {type(data)}"
            )
            
        if self.min_val is not None and data.min().item() < self.min_val:
            return ValidationResult(
                False, self.severity,
                f"Value below minimum: {data.min().item()} < {self.min_val}"
            )
            
        if self.max_val is not None and data.max().item() > self.max_val:
            return ValidationResult(
                False, self.severity,
                f"Value above maximum: {data.max().item()} > {self.max_val}"
            )
            
        return ValidationResult(True, ValidationSeverity.INFO, "Range validation passed")

class TensorDtypeRule(ValidationRule):
    """Validate tensor data type"""
    
    def __init__(self, expected_dtype: torch.dtype, name: str = "dtype_check"):
        super().__init__(name)
        self.expected_dtype = expected_dtype
        
    def validate(self, data: torch.Tensor) -> ValidationResult:
        if not isinstance(data, torch.Tensor):
            return ValidationResult(
                False, self.severity,
                f"Expected tensor, got {type(data)}"
            )
            
        if data.dtype != self.expected_dtype:
            return ValidationResult(
                False, self.severity,
                f"Dtype mismatch: expected {self.expected_dtype}, got {data.dtype}"
            )
            
        return ValidationResult(True, ValidationSeverity.INFO, "Dtype validation passed")

class TensorNaNRule(ValidationRule):
    """Check for NaN values in tensor"""
    
    def __init__(self, name: str = "nan_check"):
        super().__init__(name, ValidationSeverity.ERROR)
        
    def validate(self, data: torch.Tensor) -> ValidationResult:
        if not isinstance(data, torch.Tensor):
            return ValidationResult(
                False, self.severity,
                f"Expected tensor, got {type(data)}"
            )
            
        if torch.isnan(data).any():
            nan_count = torch.isnan(data).sum().item()
            return ValidationResult(
                False, self.severity,
                f"Found {nan_count} NaN values in tensor"
            )
            
        return ValidationResult(True, ValidationSeverity.INFO, "No NaN values found")

# Test tensor validation rules
print("Testing tensor validation rules...")
valid_tensor = torch.randn(32, 3, 224, 224)
invalid_tensor = torch.tensor([1.0, 2.0, float('nan'), 4.0])

shape_rule = TensorShapeRule((32, 3, 224, 224))
range_rule = TensorRangeRule(-5.0, 5.0)
dtype_rule = TensorDtypeRule(torch.float32)
nan_rule = TensorNaNRule()

print(f"Shape validation: {shape_rule.validate(valid_tensor).passed}")
print(f"NaN validation on valid tensor: {nan_rule.validate(valid_tensor).passed}")
print(f"NaN validation on invalid tensor: {nan_rule.validate(invalid_tensor).passed}")

# 3. STATISTICAL VALIDATION RULES
print("\n3. STATISTICAL VALIDATION RULES")

class StatisticalDistributionRule(ValidationRule):
    """Validate statistical properties of data"""
    
    def __init__(self, expected_mean: float = None, expected_std: float = None, 
                 tolerance: float = 0.1, name: str = "stats_check"):
        super().__init__(name)
        self.expected_mean = expected_mean
        self.expected_std = expected_std
        self.tolerance = tolerance
        
    def validate(self, data: torch.Tensor) -> ValidationResult:
        if not isinstance(data, torch.Tensor):
            return ValidationResult(
                False, self.severity,
                f"Expected tensor, got {type(data)}"
            )
            
        actual_mean = data.mean().item()
        actual_std = data.std().item()
        
        details = {"actual_mean": actual_mean, "actual_std": actual_std}
        
        if self.expected_mean is not None:
            if abs(actual_mean - self.expected_mean) > self.tolerance:
                return ValidationResult(
                    False, self.severity,
                    f"Mean outside tolerance: {actual_mean} vs {self.expected_mean}",
                    details
                )
                
        if self.expected_std is not None:
            if abs(actual_std - self.expected_std) > self.tolerance:
                return ValidationResult(
                    False, self.severity,
                    f"Std outside tolerance: {actual_std} vs {self.expected_std}",
                    details
                )
                
        return ValidationResult(True, ValidationSeverity.INFO, "Statistical validation passed", details)

class OutlierDetectionRule(ValidationRule):
    """Detect outliers using z-score"""
    
    def __init__(self, z_threshold: float = 3.0, max_outliers: int = None, name: str = "outlier_check"):
        super().__init__(name, ValidationSeverity.WARNING)
        self.z_threshold = z_threshold
        self.max_outliers = max_outliers
        
    def validate(self, data: torch.Tensor) -> ValidationResult:
        if not isinstance(data, torch.Tensor):
            return ValidationResult(
                False, self.severity,
                f"Expected tensor, got {type(data)}"
            )
            
        # Calculate z-scores
        mean = data.mean()
        std = data.std()
        
        if std == 0:
            return ValidationResult(True, ValidationSeverity.INFO, "No variance, no outliers")
            
        z_scores = torch.abs((data - mean) / std)
        outliers = z_scores > self.z_threshold
        outlier_count = outliers.sum().item()
        
        details = {"outlier_count": outlier_count, "total_samples": data.numel()}
        
        if self.max_outliers is not None and outlier_count > self.max_outliers:
            return ValidationResult(
                False, self.severity,
                f"Too many outliers: {outlier_count} > {self.max_outliers}",
                details
            )
            
        return ValidationResult(True, ValidationSeverity.INFO, f"Found {outlier_count} outliers", details)

# Test statistical validation
print("Testing statistical validation...")
normal_data = torch.randn(1000)
stats_rule = StatisticalDistributionRule(expected_mean=0.0, expected_std=1.0, tolerance=0.2)
outlier_rule = OutlierDetectionRule(z_threshold=2.0, max_outliers=50)

print(f"Statistical validation: {stats_rule.validate(normal_data).passed}")
print(f"Outlier detection: {outlier_rule.validate(normal_data).passed}")

# 4. CUSTOM VALIDATION RULES
print("\n4. CUSTOM VALIDATION RULES")

class ImageValidationRule(ValidationRule):
    """Validate image-specific properties"""
    
    def __init__(self, min_channels: int = 1, max_channels: int = 4,
                 min_size: Tuple[int, int] = (32, 32),
                 max_size: Tuple[int, int] = (2048, 2048),
                 name: str = "image_check"):
        super().__init__(name)
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.min_size = min_size
        self.max_size = max_size
        
    def validate(self, data: torch.Tensor) -> ValidationResult:
        if not isinstance(data, torch.Tensor):
            return ValidationResult(
                False, self.severity,
                f"Expected tensor, got {type(data)}"
            )
            
        if len(data.shape) != 3:  # C, H, W
            return ValidationResult(
                False, self.severity,
                f"Expected 3D tensor (C, H, W), got {len(data.shape)}D"
            )
            
        channels, height, width = data.shape
        
        # Check channels
        if not (self.min_channels <= channels <= self.max_channels):
            return ValidationResult(
                False, self.severity,
                f"Invalid channels: {channels} not in [{self.min_channels}, {self.max_channels}]"
            )
            
        # Check size
        if height < self.min_size[0] or width < self.min_size[1]:
            return ValidationResult(
                False, self.severity,
                f"Image too small: {(height, width)} < {self.min_size}"
            )
            
        if height > self.max_size[0] or width > self.max_size[1]:
            return ValidationResult(
                False, self.severity,
                f"Image too large: {(height, width)} > {self.max_size}"
            )
            
        return ValidationResult(True, ValidationSeverity.INFO, "Image validation passed")

class LabelValidationRule(ValidationRule):
    """Validate classification labels"""
    
    def __init__(self, num_classes: int, allow_negative: bool = False, name: str = "label_check"):
        super().__init__(name)
        self.num_classes = num_classes
        self.allow_negative = allow_negative
        
    def validate(self, data: torch.Tensor) -> ValidationResult:
        if not isinstance(data, torch.Tensor):
            return ValidationResult(
                False, self.severity,
                f"Expected tensor, got {type(data)}"
            )
            
        # Check for integer type
        if not data.dtype.is_integer:
            return ValidationResult(
                False, ValidationSeverity.WARNING,
                f"Labels should be integers, got {data.dtype}"
            )
            
        min_label = data.min().item()
        max_label = data.max().item()
        
        # Check range
        expected_min = -1 if self.allow_negative else 0
        expected_max = self.num_classes - 1
        
        if min_label < expected_min:
            return ValidationResult(
                False, self.severity,
                f"Label below minimum: {min_label} < {expected_min}"
            )
            
        if max_label > expected_max:
            return ValidationResult(
                False, self.severity,
                f"Label above maximum: {max_label} > {expected_max}"
            )
            
        return ValidationResult(True, ValidationSeverity.INFO, "Label validation passed")

# Test custom validation rules
print("Testing custom validation...")
image_tensor = torch.randn(3, 224, 224)
label_tensor = torch.randint(0, 10, (32,))

image_rule = ImageValidationRule(min_channels=1, max_channels=3)
label_rule = LabelValidationRule(num_classes=10)

print(f"Image validation: {image_rule.validate(image_tensor).passed}")
print(f"Label validation: {label_rule.validate(label_tensor).passed}")

# 5. VALIDATION PIPELINE
print("\n5. VALIDATION PIPELINE")

class ValidationPipeline:
    """Pipeline for running multiple validation rules"""
    
    def __init__(self, rules: List[ValidationRule], stop_on_error: bool = True):
        self.rules = rules
        self.stop_on_error = stop_on_error
        self.results = []
        
    def add_rule(self, rule: ValidationRule):
        """Add validation rule to pipeline"""
        self.rules.append(rule)
        
    def validate(self, data: Any) -> Dict[str, Any]:
        """Run all validation rules on data"""
        self.results = []
        passed_count = 0
        
        for rule in self.rules:
            try:
                result = rule.validate(data)
                self.results.append(result)
                
                if result.passed:
                    passed_count += 1
                elif self.stop_on_error and result.severity == ValidationSeverity.ERROR:
                    break
                    
            except Exception as e:
                error_result = ValidationResult(
                    False, ValidationSeverity.CRITICAL,
                    f"Validation rule {rule.name} failed: {str(e)}"
                )
                self.results.append(error_result)
                
                if self.stop_on_error:
                    break
                    
        return {
            'passed': passed_count == len(self.results) and all(r.passed for r in self.results),
            'total_rules': len(self.rules),
            'passed_rules': passed_count,
            'results': self.results
        }
        
    def get_summary(self) -> str:
        """Get validation summary"""
        if not self.results:
            return "No validation results available"
            
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        
        summary = f"Validation Summary: {passed}/{total} rules passed\n"
        
        for result in self.results:
            status = "✓" if result.passed else "✗"
            summary += f"{status} {result.message}\n"
            
        return summary

# Create validation pipeline
validation_pipeline = ValidationPipeline([
    TensorShapeRule((3, 224, 224)),
    TensorRangeRule(-3.0, 3.0),
    TensorDtypeRule(torch.float32),
    TensorNaNRule(),
    ImageValidationRule(),
    StatisticalDistributionRule(expected_mean=0.0, expected_std=1.0, tolerance=0.5)
])

# Test pipeline
test_image = torch.randn(3, 224, 224)
pipeline_result = validation_pipeline.validate(test_image)
print(f"Pipeline validation passed: {pipeline_result['passed']}")
print(f"Summary:\n{validation_pipeline.get_summary()}")

# 6. DATASET VALIDATION
print("\n6. DATASET VALIDATION")

class ValidatedDataset(Dataset):
    """Dataset with built-in validation"""
    
    def __init__(self, base_dataset: Dataset, 
                 input_pipeline: ValidationPipeline = None,
                 target_pipeline: ValidationPipeline = None,
                 validation_mode: str = "strict"):
        
        self.base_dataset = base_dataset
        self.input_pipeline = input_pipeline
        self.target_pipeline = target_pipeline
        self.validation_mode = validation_mode  # strict, warn, ignore
        self.validation_stats = {
            'total_samples': 0,
            'validation_failures': 0,
            'validation_warnings': 0
        }
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        try:
            data = self.base_dataset[idx]
            
            if isinstance(data, tuple) and len(data) == 2:
                inputs, targets = data
                
                # Validate inputs
                if self.input_pipeline:
                    input_result = self.input_pipeline.validate(inputs)
                    self._handle_validation_result(input_result, f"Input validation failed for sample {idx}")
                    
                # Validate targets
                if self.target_pipeline:
                    target_result = self.target_pipeline.validate(targets)
                    self._handle_validation_result(target_result, f"Target validation failed for sample {idx}")
                    
                self.validation_stats['total_samples'] += 1
                return inputs, targets
            else:
                # Single item validation
                if self.input_pipeline:
                    result = self.input_pipeline.validate(data)
                    self._handle_validation_result(result, f"Validation failed for sample {idx}")
                    
                self.validation_stats['total_samples'] += 1
                return data
                
        except Exception as e:
            if self.validation_mode == "strict":
                raise
            else:
                warnings.warn(f"Validation error for sample {idx}: {str(e)}")
                self.validation_stats['validation_failures'] += 1
                return self.base_dataset[idx]  # Return original data
                
    def _handle_validation_result(self, result: Dict[str, Any], error_msg: str):
        """Handle validation result based on mode"""
        if not result['passed']:
            self.validation_stats['validation_failures'] += 1
            
            if self.validation_mode == "strict":
                raise ValueError(error_msg)
            elif self.validation_mode == "warn":
                warnings.warn(error_msg)
                self.validation_stats['validation_warnings'] += 1
                
    def get_validation_stats(self):
        """Get validation statistics"""
        return self.validation_stats.copy()

# Example usage with dummy dataset
class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Occasionally return invalid data for testing
        if idx % 50 == 0:
            return torch.randn(3, 128, 128), torch.randint(0, 10, (1,))  # Wrong size
        return torch.randn(3, 224, 224), torch.randint(0, 10, (1,))

dummy_ds = DummyDataset(100)

# Create pipelines for input and target validation
input_validation = ValidationPipeline([
    TensorShapeRule((3, 224, 224)),
    TensorDtypeRule(torch.float32),
    ImageValidationRule()
], stop_on_error=False)

target_validation = ValidationPipeline([
    LabelValidationRule(num_classes=10)
], stop_on_error=False)

# Test different validation modes
print("\nTesting validation modes...")
for mode in ["ignore", "warn", "strict"]:
    print(f"\nTesting {mode} mode:")
    validated_ds = ValidatedDataset(
        dummy_ds, input_validation, target_validation, validation_mode=mode
    )
    
    try:
        # Test a few samples
        for i in range(3):
            _ = validated_ds[i * 25]  # Test samples that might fail
        print(f"✓ {mode} mode completed successfully")
        print(f"Stats: {validated_ds.get_validation_stats()}")
    except Exception as e:
        print(f"✗ {mode} mode failed: {str(e)}")

# 7. BATCH VALIDATION
print("\n7. BATCH VALIDATION")

class BatchValidator:
    """Validator for data batches"""
    
    def __init__(self, validation_pipeline: ValidationPipeline):
        self.validation_pipeline = validation_pipeline
        self.batch_stats = {
            'total_batches': 0,
            'failed_batches': 0,
            'total_samples': 0,
            'failed_samples': 0
        }
        
    def validate_batch(self, batch: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> Dict[str, Any]:
        """Validate a batch of data"""
        self.batch_stats['total_batches'] += 1
        
        if isinstance(batch, tuple):
            inputs, targets = batch
            batch_size = inputs.size(0)
            
            input_results = []
            target_results = []
            
            # Validate each sample in batch
            for i in range(batch_size):
                # Validate input
                input_result = self.validation_pipeline.validate(inputs[i])
                input_results.append(input_result)
                
                if not input_result['passed']:
                    self.batch_stats['failed_samples'] += 1
                    
            self.batch_stats['total_samples'] += batch_size
            
            batch_passed = all(r['passed'] for r in input_results)
            if not batch_passed:
                self.batch_stats['failed_batches'] += 1
                
            return {
                'batch_passed': batch_passed,
                'batch_size': batch_size,
                'input_results': input_results,
                'failed_samples': sum(1 for r in input_results if not r['passed'])
            }
        else:
            # Single tensor batch
            batch_size = batch.size(0)
            results = []
            
            for i in range(batch_size):
                result = self.validation_pipeline.validate(batch[i])
                results.append(result)
                
                if not result['passed']:
                    self.batch_stats['failed_samples'] += 1
                    
            self.batch_stats['total_samples'] += batch_size
            
            batch_passed = all(r['passed'] for r in results)
            if not batch_passed:
                self.batch_stats['failed_batches'] += 1
                
            return {
                'batch_passed': batch_passed,
                'batch_size': batch_size,
                'results': results,
                'failed_samples': sum(1 for r in results if not r['passed'])
            }
            
    def get_stats(self):
        """Get batch validation statistics"""
        return self.batch_stats.copy()

# Test batch validation
batch_validator = BatchValidator(input_validation)
test_batch = torch.randn(32, 3, 224, 224)
batch_result = batch_validator.validate_batch(test_batch)
print(f"Batch validation passed: {batch_result['batch_passed']}")
print(f"Failed samples in batch: {batch_result['failed_samples']}")

print("\n=== DATA VALIDATION PIPELINE COMPLETE ===")
print("Key concepts covered:")
print("- Validation rule framework")
print("- Tensor validation (shape, range, dtype, NaN)")
print("- Statistical validation")
print("- Custom validation rules")
print("- Validation pipelines")
print("- Dataset validation")
print("- Batch validation")
print("- Error handling strategies")