#!/usr/bin/env python3
"""PyTorch Dataset Fundamentals - Dataset class fundamentals"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import pickle
import json
from typing import Any, Tuple, List, Optional, Union
from pathlib import Path

print("=== Dataset Fundamentals Overview ===")

print("Dataset concepts covered:")
print("1. PyTorch Dataset base class")
print("2. Abstract methods (__len__, __getitem__)")
print("3. Data indexing and access patterns")
print("4. Memory vs on-demand loading")
print("5. Data type handling and conversions")
print("6. Dataset composition and chaining")
print("7. Error handling in datasets")
print("8. Performance considerations")

print("\n=== Basic Dataset Implementation ===")

class BasicDataset(Dataset):
    """Basic dataset implementation example"""
    
    def __init__(self, data, labels):
        """
        Initialize dataset with data and labels
        
        Args:
            data: Input data (tensor, list, numpy array)
            labels: Target labels
        """
        self.data = data
        self.labels = labels
        
        # Ensure data is tensor
        if not isinstance(self.data, torch.Tensor):
            self.data = torch.tensor(self.data, dtype=torch.float32)
        
        if not isinstance(self.labels, torch.Tensor):
            self.labels = torch.tensor(self.labels, dtype=torch.long)
        
        # Validate data consistency
        assert len(self.data) == len(self.labels), "Data and labels must have same length"
    
    def __len__(self) -> int:
        """Return the total number of samples"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample by index
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (data, label)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample_data = self.data[idx]
        sample_label = self.labels[idx]
        
        return sample_data, sample_label

# Test basic dataset
print("Testing basic dataset:")
sample_data = torch.randn(100, 10)
sample_labels = torch.randint(0, 3, (100,))

basic_dataset = BasicDataset(sample_data, sample_labels)
print(f"  Dataset length: {len(basic_dataset)}")
print(f"  First sample shape: {basic_dataset[0][0].shape}")
print(f"  First sample label: {basic_dataset[0][1].item()}")

# Test with DataLoader
dataloader = DataLoader(basic_dataset, batch_size=16, shuffle=True)
batch_data, batch_labels = next(iter(dataloader))
print(f"  Batch data shape: {batch_data.shape}")
print(f"  Batch labels shape: {batch_labels.shape}")

print("\n=== Advanced Dataset Features ===")

class AdvancedDataset(Dataset):
    """Advanced dataset with additional features"""
    
    def __init__(self, data_path: str, transform=None, target_transform=None, 
                 cache_data: bool = False, validate_data: bool = True):
        """
        Advanced dataset initialization
        
        Args:
            data_path: Path to data file
            transform: Optional transform to be applied on samples
            target_transform: Optional transform to be applied on targets
            cache_data: Whether to cache data in memory
            validate_data: Whether to validate data integrity
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.target_transform = target_transform
        self.cache_data = cache_data
        self.validate_data = validate_data
        
        # Load data
        self._load_data()
        
        # Validate if requested
        if self.validate_data:
            self._validate_data()
        
        # Cache data if requested
        if self.cache_data:
            self._cache_all_data()
        
        self._cached_items = {} if not cache_data else None
    
    def _load_data(self):
        """Load data from file"""
        if not self.data_path.exists():
            # Create sample data if file doesn't exist
            print(f"  Creating sample data at {self.data_path}")
            sample_data = {
                'features': np.random.randn(1000, 20).astype(np.float32),
                'labels': np.random.randint(0, 5, 1000),
                'metadata': {'num_classes': 5, 'feature_dim': 20}
            }
            
            # Ensure directory exists
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as numpy file
            np.savez(self.data_path, **sample_data)
        
        # Load data
        data = np.load(self.data_path, allow_pickle=True)
        self.features = data['features']
        self.labels = data['labels']
        self.metadata = data['metadata'].item() if 'metadata' in data else {}
        
        print(f"  Loaded {len(self.features)} samples from {self.data_path}")
    
    def _validate_data(self):
        """Validate data integrity"""
        assert len(self.features) == len(self.labels), "Features and labels length mismatch"
        assert len(self.features) > 0, "Empty dataset"
        assert self.features.ndim >= 2, "Features should be at least 2D"
        
        # Check for NaN values
        if np.isnan(self.features).any():
            print("  WARNING: NaN values found in features")
        
        if np.isnan(self.labels).any():
            print("  WARNING: NaN values found in labels")
        
        print("  Data validation passed")
    
    def _cache_all_data(self):
        """Cache all data in memory"""
        print("  Caching all data in memory...")
        self.cached_features = torch.tensor(self.features, dtype=torch.float32)
        self.cached_labels = torch.tensor(self.labels, dtype=torch.long)
    
    def __len__(self) -> int:
        """Return the total number of samples"""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample by index with caching and transforms"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Check cache first
        if not self.cache_data and idx in self._cached_items:
            sample, label = self._cached_items[idx]
        else:
            # Load data
            if self.cache_data:
                sample = self.cached_features[idx]
                label = self.cached_labels[idx]
            else:
                sample = torch.tensor(self.features[idx], dtype=torch.float32)
                label = torch.tensor(self.labels[idx], dtype=torch.long)
            
            # Cache item if not full caching
            if not self.cache_data:
                self._cached_items[idx] = (sample.clone(), label.clone())
        
        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return sample, label
    
    def get_metadata(self) -> dict:
        """Get dataset metadata"""
        return self.metadata
    
    def get_class_counts(self) -> dict:
        """Get count of samples per class"""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))

# Test advanced dataset
print("Testing advanced dataset:")
advanced_dataset = AdvancedDataset("temp_data.npz", cache_data=True)
print(f"  Dataset metadata: {advanced_dataset.get_metadata()}")
print(f"  Class distribution: {advanced_dataset.get_class_counts()}")

# Clean up
if os.path.exists("temp_data.npz"):
    os.remove("temp_data.npz")

print("\n=== Dataset Indexing Patterns ===")

class IndexedDataset(Dataset):
    """Dataset demonstrating different indexing patterns"""
    
    def __init__(self, size=1000):
        self.size = size
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 3, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        """Support multiple indexing patterns"""
        
        # Handle different index types
        if isinstance(idx, slice):
            # Slice indexing
            return self._get_slice(idx)
        elif isinstance(idx, (list, tuple, np.ndarray)):
            # Multiple index selection
            return self._get_multiple(idx)
        elif isinstance(idx, torch.Tensor):
            # Tensor indexing
            if idx.dim() == 0:  # Single element tensor
                idx = idx.item()
                return self.data[idx], self.labels[idx]
            else:  # Multiple element tensor
                return self._get_multiple(idx.tolist())
        else:
            # Single index
            return self.data[idx], self.labels[idx]
    
    def _get_slice(self, slice_obj):
        """Handle slice indexing"""
        data_slice = self.data[slice_obj]
        labels_slice = self.labels[slice_obj]
        return data_slice, labels_slice
    
    def _get_multiple(self, indices):
        """Handle multiple index selection"""
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        
        selected_data = self.data[indices]
        selected_labels = self.labels[indices]
        return selected_data, selected_labels

# Test indexing patterns
print("Testing dataset indexing patterns:")
indexed_dataset = IndexedDataset(size=50)

# Single index
sample = indexed_dataset[5]
print(f"  Single index [5]: data shape {sample[0].shape}, label {sample[1].item()}")

# Slice index
slice_sample = indexed_dataset[10:15]
print(f"  Slice [10:15]: data shape {slice_sample[0].shape}, labels shape {slice_sample[1].shape}")

# Multiple indices
multi_sample = indexed_dataset[[1, 5, 10, 20]]
print(f"  Multiple indices [1,5,10,20]: data shape {multi_sample[0].shape}")

# Tensor index
tensor_idx = torch.tensor([2, 7, 12])
tensor_sample = indexed_dataset[tensor_idx]
print(f"  Tensor indices: data shape {tensor_sample[0].shape}")

print("\n=== Dataset Composition and Chaining ===")

class ConcatDataset(Dataset):
    """Concatenate multiple datasets"""
    
    def __init__(self, *datasets):
        self.datasets = datasets
        self.cumulative_sizes = self._get_cumulative_sizes()
    
    def _get_cumulative_sizes(self):
        """Calculate cumulative sizes for indexing"""
        cumulative_sizes = []
        cumsum = 0
        for dataset in self.datasets:
            cumsum += len(dataset)
            cumulative_sizes.append(cumsum)
        return cumulative_sizes
    
    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0
    
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("Absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        
        # Find which dataset the index belongs to
        dataset_idx = next(i for i, size in enumerate(self.cumulative_sizes) if idx < size)
        
        # Calculate local index within the dataset
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        
        return self.datasets[dataset_idx][sample_idx]

class SubsetDataset(Dataset):
    """Create a subset of another dataset"""
    
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

class TransformDataset(Dataset):
    """Apply transforms to another dataset"""
    
    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        
        if self.transform:
            data = self.transform(data)
        
        if self.target_transform:
            target = self.target_transform(target)
        
        return data, target

# Test dataset composition
print("Testing dataset composition:")

# Create base datasets
dataset1 = BasicDataset(torch.randn(50, 5), torch.randint(0, 2, (50,)))
dataset2 = BasicDataset(torch.randn(30, 5), torch.randint(0, 2, (30,)))

# Concatenate datasets
concat_dataset = ConcatDataset(dataset1, dataset2)
print(f"  Concatenated dataset length: {len(concat_dataset)}")
print(f"  Sample from first dataset: {concat_dataset[10][1].item()}")
print(f"  Sample from second dataset: {concat_dataset[60][1].item()}")

# Create subset
subset_indices = list(range(0, len(concat_dataset), 2))  # Every other sample
subset_dataset = SubsetDataset(concat_dataset, subset_indices)
print(f"  Subset dataset length: {len(subset_dataset)}")

# Apply transforms
def normalize_transform(x):
    return (x - x.mean()) / (x.std() + 1e-8)

def label_transform(y):
    return y + 10  # Shift labels

transform_dataset = TransformDataset(subset_dataset, 
                                   transform=normalize_transform,
                                   target_transform=label_transform)

sample_data, sample_label = transform_dataset[0]
print(f"  Transformed sample: mean={sample_data.mean():.4f}, std={sample_data.std():.4f}")
print(f"  Transformed label: {sample_label.item()}")

print("\n=== Error Handling in Datasets ===")

class RobustDataset(Dataset):
    """Dataset with comprehensive error handling"""
    
    def __init__(self, data, labels, handle_errors='raise'):
        """
        Initialize robust dataset
        
        Args:
            data: Input data
            labels: Target labels
            handle_errors: How to handle errors ('raise', 'skip', 'default')
        """
        self.data = data
        self.labels = labels
        self.handle_errors = handle_errors
        self.error_count = 0
        self.valid_indices = list(range(len(data)))
        
        if handle_errors == 'skip':
            self._validate_all_samples()
    
    def _validate_all_samples(self):
        """Pre-validate all samples and create valid indices list"""
        valid_indices = []
        for idx in range(len(self.data)):
            try:
                self._validate_sample(idx)
                valid_indices.append(idx)
            except Exception as e:
                self.error_count += 1
                print(f"  Skipping invalid sample at index {idx}: {e}")
        
        self.valid_indices = valid_indices
        print(f"  Found {len(valid_indices)} valid samples out of {len(self.data)}")
    
    def _validate_sample(self, idx):
        """Validate a single sample"""
        data_sample = self.data[idx]
        label_sample = self.labels[idx]
        
        # Check for NaN values
        if torch.isnan(data_sample).any():
            raise ValueError("NaN values in data")
        
        if torch.isnan(label_sample):
            raise ValueError("NaN value in label")
        
        # Check for infinite values
        if torch.isinf(data_sample).any():
            raise ValueError("Infinite values in data")
        
        # Check data shape
        if data_sample.numel() == 0:
            raise ValueError("Empty data sample")
    
    def __len__(self):
        if self.handle_errors == 'skip':
            return len(self.valid_indices)
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            if self.handle_errors == 'skip':
                # Use valid indices mapping
                actual_idx = self.valid_indices[idx]
            else:
                actual_idx = idx
            
            # Validate sample
            self._validate_sample(actual_idx)
            
            data_sample = self.data[actual_idx]
            label_sample = self.labels[actual_idx]
            
            return data_sample, label_sample
            
        except Exception as e:
            self.error_count += 1
            
            if self.handle_errors == 'raise':
                raise e
            elif self.handle_errors == 'default':
                # Return default values
                default_data = torch.zeros_like(self.data[0])
                default_label = torch.tensor(0, dtype=self.labels.dtype)
                print(f"  Returning default for index {idx}: {e}")
                return default_data, default_label
            else:
                raise e

# Test error handling
print("Testing dataset error handling:")

# Create data with some corrupted samples
data_with_errors = torch.randn(20, 5)
labels_with_errors = torch.randint(0, 3, (20,))

# Introduce some errors
data_with_errors[5] = float('nan')  # NaN values
data_with_errors[10] = float('inf')  # Infinite values
labels_with_errors[15] = float('nan')  # NaN label

# Test different error handling strategies
strategies = ['skip', 'default', 'raise']

for strategy in strategies[:2]:  # Skip 'raise' to avoid exception
    print(f"\n  Testing '{strategy}' strategy:")
    try:
        robust_dataset = RobustDataset(data_with_errors, labels_with_errors, 
                                     handle_errors=strategy)
        print(f"    Dataset length: {len(robust_dataset)}")
        print(f"    Error count: {robust_dataset.error_count}")
        
        # Try to get a few samples
        for i in range(min(3, len(robust_dataset))):
            sample = robust_dataset[i]
            print(f"    Sample {i}: data shape {sample[0].shape}, label {sample[1].item()}")
    
    except Exception as e:
        print(f"    Error with {strategy} strategy: {e}")

print("\n=== Dataset Performance Considerations ===")

import time

class PerformanceDataset(Dataset):
    """Dataset to demonstrate performance considerations"""
    
    def __init__(self, size=1000, load_mode='lazy', use_cache=False):
        """
        Initialize performance dataset
        
        Args:
            size: Dataset size
            load_mode: 'lazy', 'eager', 'memory_mapped'
            use_cache: Whether to use caching
        """
        self.size = size
        self.load_mode = load_mode
        self.use_cache = use_cache
        self.cache = {} if use_cache else None
        
        if load_mode == 'eager':
            # Load all data into memory
            print(f"  Loading all {size} samples into memory...")
            self.data = torch.randn(size, 100)
            self.labels = torch.randint(0, 10, (size,))
        elif load_mode == 'memory_mapped':
            # Simulate memory-mapped data
            self.data_file = 'temp_mmap.npy'
            data = np.random.randn(size, 100).astype(np.float32)
            np.save(self.data_file, data)
            self.data = np.load(self.data_file, mmap_mode='r')
            self.labels = torch.randint(0, 10, (size,))
        else:
            # Lazy loading - generate on demand
            self.data = None
            self.labels = None
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Check cache first
        if self.use_cache and idx in self.cache:
            return self.cache[idx]
        
        if self.load_mode == 'lazy':
            # Generate data on demand
            torch.manual_seed(idx)  # Consistent data generation
            data = torch.randn(100)
            label = torch.randint(0, 10, (1,)).item()
        elif self.load_mode == 'eager':
            # Data already in memory
            data = self.data[idx]
            label = self.labels[idx]
        elif self.load_mode == 'memory_mapped':
            # Load from memory-mapped file
            data = torch.tensor(self.data[idx], dtype=torch.float32)
            label = self.labels[idx]
        
        sample = (data, torch.tensor(label, dtype=torch.long))
        
        # Cache if enabled
        if self.use_cache:
            self.cache[idx] = sample
        
        return sample
    
    def __del__(self):
        # Clean up memory-mapped file
        if hasattr(self, 'data_file') and os.path.exists(self.data_file):
            os.remove(self.data_file)

# Test performance
print("Testing dataset performance:")

def benchmark_dataset(dataset, num_samples=100):
    """Benchmark dataset access time"""
    start_time = time.time()
    
    for i in range(num_samples):
        _ = dataset[i % len(dataset)]
    
    end_time = time.time()
    return (end_time - start_time) / num_samples

# Test different loading modes
modes = ['lazy', 'eager', 'memory_mapped']
dataset_size = 500
num_benchmark_samples = 100

for mode in modes:
    print(f"\n  Testing {mode} mode:")
    dataset = PerformanceDataset(size=dataset_size, load_mode=mode, use_cache=False)
    
    # First access (cold)
    cold_time = benchmark_dataset(dataset, num_benchmark_samples)
    print(f"    Cold access time: {cold_time*1000:.3f} ms per sample")
    
    # Second access (warm)
    warm_time = benchmark_dataset(dataset, num_benchmark_samples)
    print(f"    Warm access time: {warm_time*1000:.3f} ms per sample")
    
    # With caching
    cached_dataset = PerformanceDataset(size=dataset_size, load_mode=mode, use_cache=True)
    cached_time = benchmark_dataset(cached_dataset, num_benchmark_samples)
    print(f"    With caching: {cached_time*1000:.3f} ms per sample")

print("\n=== Dataset Best Practices ===")

print("Design Principles:")
print("1. Implement __len__ and __getitem__ methods")
print("2. Handle indexing edge cases (negative indices, bounds)")
print("3. Use appropriate data types (torch.Tensor)")
print("4. Consider memory usage vs access speed trade-offs")
print("5. Implement proper error handling")

print("\nPerformance Tips:")
print("1. Use lazy loading for large datasets")
print("2. Implement caching for frequently accessed items")
print("3. Consider memory-mapped files for huge datasets")
print("4. Avoid expensive operations in __getitem__")
print("5. Pre-compute and cache transforms when possible")

print("\nMemory Management:")
print("1. Don't load entire dataset if it doesn't fit in memory")
print("2. Use appropriate data types (float32 vs float64)")
print("3. Release unused data references")
print("4. Consider using generators for streaming data")
print("5. Monitor memory usage during development")

print("\nError Handling:")
print("1. Validate data integrity during initialization")
print("2. Handle missing or corrupted files gracefully")
print("3. Provide meaningful error messages")
print("4. Consider fallback strategies for corrupted data")
print("5. Log errors for debugging purposes")

print("\nTesting Strategies:")
print("1. Test with different dataset sizes")
print("2. Verify indexing edge cases")
print("3. Test with DataLoader (shuffling, batching)")
print("4. Validate data types and shapes")
print("5. Performance test with realistic data sizes")

print("\n=== Dataset Fundamentals Complete ===")

# Memory cleanup
torch.cuda.empty_cache() if torch.cuda.is_available() else None