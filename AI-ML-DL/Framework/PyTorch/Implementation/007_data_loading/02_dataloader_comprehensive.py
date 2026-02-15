#!/usr/bin/env python3
"""PyTorch DataLoader Comprehensive - DataLoader all parameters and usage"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
from torch.utils.data import RandomSampler, SequentialSampler, SubsetRandomSampler
import numpy as np
import time
import multiprocessing
from typing import Iterator, List, Optional, Sized
import warnings

print("=== DataLoader Comprehensive Overview ===")

print("DataLoader parameters covered:")
print("1. Dataset and batch_size")
print("2. Shuffle and sampling")
print("3. Batch sampling strategies")
print("4. Collate functions")
print("5. Multiprocessing (num_workers)")
print("6. Memory pinning and prefetching")
print("7. Timeout and error handling")
print("8. Advanced DataLoader configurations")

print("\n=== Basic DataLoader Usage ===")

class SimpleDataset(Dataset):
    """Simple dataset for DataLoader examples"""
    
    def __init__(self, size=1000, feature_dim=20):
        self.size = size
        self.feature_dim = feature_dim
        self.data = torch.randn(size, feature_dim)
        self.labels = torch.randint(0, 5, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create sample dataset
dataset = SimpleDataset(size=100)
print(f"Dataset size: {len(dataset)}")

# Basic DataLoader configurations
print("\nBasic DataLoader configurations:")

configs = [
    {"batch_size": 16, "shuffle": False, "description": "Sequential batching"},
    {"batch_size": 16, "shuffle": True, "description": "Random batching"},
    {"batch_size": 32, "shuffle": True, "description": "Larger batch size"},
    {"batch_size": 1, "shuffle": False, "description": "Single sample batches"},
]

for config in configs:
    desc = config.pop("description")
    dataloader = DataLoader(dataset, **config)
    
    # Get first batch
    batch_data, batch_labels = next(iter(dataloader))
    print(f"  {desc}:")
    print(f"    Batch shape: {batch_data.shape}")
    print(f"    Labels shape: {batch_labels.shape}")
    print(f"    Batches per epoch: {len(dataloader)}")

print("\n=== Sampling Strategies ===")

# Different sampling strategies
samplers = {
    "Random": RandomSampler(dataset),
    "Sequential": SequentialSampler(dataset),
    "Subset Random": SubsetRandomSampler(range(0, len(dataset), 2)),  # Every other sample
}

for name, sampler in samplers.items():
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)
    
    # Get indices from first batch
    batch_data, batch_labels = next(iter(dataloader))
    print(f"  {name} Sampler:")
    print(f"    First batch labels: {batch_labels[:5].tolist()}")
    print(f"    Total batches: {len(dataloader)}")

print("\n=== Custom Samplers ===")

class WeightedRandomSampler(Sampler):
    """Custom weighted random sampler"""
    
    def __init__(self, weights, num_samples=None, replacement=True):
        self.weights = torch.tensor(weights, dtype=torch.float)
        self.num_samples = num_samples if num_samples else len(weights)
        self.replacement = replacement
    
    def __iter__(self) -> Iterator[int]:
        if self.replacement:
            # Sample with replacement
            indices = torch.multinomial(self.weights, self.num_samples, replacement=True)
        else:
            # Sample without replacement (not implemented for simplicity)
            indices = torch.multinomial(self.weights, self.num_samples, replacement=False)
        return iter(indices.tolist())
    
    def __len__(self):
        return self.num_samples

class BalancedBatchSampler(BatchSampler):
    """Custom batch sampler for balanced batches"""
    
    def __init__(self, dataset, batch_size, samples_per_class=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class or batch_size // 5
        
        # Group indices by class
        self.class_indices = {}
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            label = label.item()
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        
        self.num_classes = len(self.class_indices)
        
    def __iter__(self):
        # Create balanced batches
        while True:
            batch = []
            for class_label in self.class_indices:
                class_samples = np.random.choice(
                    self.class_indices[class_label], 
                    size=min(self.samples_per_class, len(self.class_indices[class_label])),
                    replace=True
                )
                batch.extend(class_samples.tolist())
            
            # Ensure batch size
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
            else:
                # Pad batch if needed
                remaining = self.batch_size - len(batch)
                all_indices = [idx for indices in self.class_indices.values() for idx in indices]
                extra_samples = np.random.choice(all_indices, size=remaining, replace=True)
                batch.extend(extra_samples.tolist())
                yield batch
    
    def __len__(self):
        # Approximate number of batches
        return len(self.dataset) // self.batch_size

# Test custom samplers
print("Testing custom samplers:")

# Weighted sampler (bias towards certain samples)
weights = [1.0 if i % 10 == 0 else 0.1 for i in range(len(dataset))]
weighted_sampler = WeightedRandomSampler(weights, num_samples=50)
weighted_dataloader = DataLoader(dataset, batch_size=10, sampler=weighted_sampler)

batch_data, batch_labels = next(iter(weighted_dataloader))
print(f"  Weighted sampler batch labels: {batch_labels.tolist()}")

# Balanced batch sampler
balanced_sampler = BalancedBatchSampler(dataset, batch_size=20, samples_per_class=4)
balanced_dataloader = DataLoader(dataset, batch_sampler=balanced_sampler)

batch_data, batch_labels = next(iter(balanced_dataloader))
print(f"  Balanced batch labels: {sorted(batch_labels.tolist())}")

print("\n=== Collate Functions ===")

def custom_collate_fn(batch):
    """Custom collate function to handle batching"""
    data, labels = zip(*batch)
    
    # Stack data
    data = torch.stack(data, dim=0)
    labels = torch.stack(labels, dim=0)
    
    # Add batch statistics
    batch_stats = {
        'mean': data.mean(),
        'std': data.std(),
        'size': len(batch)
    }
    
    return {
        'data': data,
        'labels': labels,
        'stats': batch_stats
    }

def padding_collate_fn(batch):
    """Collate function with padding for variable-length sequences"""
    data, labels = zip(*batch)
    
    # Find max length (simulate variable length data)
    lengths = [d.size(0) for d in data]
    max_length = max(lengths)
    
    # Pad sequences
    padded_data = []
    for d in data:
        if d.size(0) < max_length:
            padding = torch.zeros(max_length - d.size(0), d.size(1))
            padded_d = torch.cat([d, padding], dim=0)
        else:
            padded_d = d
        padded_data.append(padded_d)
    
    return {
        'data': torch.stack(padded_data),
        'labels': torch.stack(labels),
        'lengths': torch.tensor(lengths)
    }

# Test collate functions
print("Testing collate functions:")

# Custom collate function
custom_dataloader = DataLoader(dataset, batch_size=8, collate_fn=custom_collate_fn, shuffle=True)
custom_batch = next(iter(custom_dataloader))
print(f"  Custom collate - Data shape: {custom_batch['data'].shape}")
print(f"  Custom collate - Batch stats: mean={custom_batch['stats']['mean']:.4f}, std={custom_batch['stats']['std']:.4f}")

# Variable length dataset for padding example
class VariableLengthDataset(Dataset):
    def __init__(self, size=50):
        self.size = size
        # Create variable length sequences
        self.data = [torch.randn(np.random.randint(10, 50), 5) for _ in range(size)]
        self.labels = torch.randint(0, 3, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

var_dataset = VariableLengthDataset(size=20)
padding_dataloader = DataLoader(var_dataset, batch_size=4, collate_fn=padding_collate_fn)
padding_batch = next(iter(padding_dataloader))
print(f"  Padding collate - Data shape: {padding_batch['data'].shape}")
print(f"  Padding collate - Lengths: {padding_batch['lengths'].tolist()}")

print("\n=== Multiprocessing DataLoader ===")

class SlowDataset(Dataset):
    """Dataset that simulates slow data loading"""
    
    def __init__(self, size=100, delay=0.01):
        self.size = size
        self.delay = delay
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Simulate slow data loading
        time.sleep(self.delay)
        return torch.randn(10), torch.randint(0, 2, (1,)).item()

# Test different numbers of workers
print("Testing multiprocessing performance:")

slow_dataset = SlowDataset(size=50, delay=0.001)  # Reduced delay for testing
worker_configs = [0, 1, 2]

for num_workers in worker_configs:
    dataloader = DataLoader(slow_dataset, batch_size=8, num_workers=num_workers, 
                          persistent_workers=(num_workers > 0))
    
    start_time = time.time()
    
    # Time one epoch
    for batch_idx, (data, labels) in enumerate(dataloader):
        if batch_idx >= 5:  # Only test first 5 batches
            break
    
    end_time = time.time()
    
    print(f"  {num_workers} workers: {(end_time - start_time)*1000:.2f} ms for 5 batches")

print("\n=== Memory Pinning and Prefetching ===")

def test_memory_pinning(use_pin_memory=False, prefetch_factor=2):
    """Test memory pinning performance"""
    dataset = SimpleDataset(size=200)
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        pin_memory=use_pin_memory,
        num_workers=1 if use_pin_memory else 0,
        prefetch_factor=prefetch_factor if use_pin_memory else 2
    )
    
    start_time = time.time()
    
    for batch_idx, (data, labels) in enumerate(dataloader):
        # Simulate GPU transfer
        if torch.cuda.is_available() and use_pin_memory:
            data = data.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        
        if batch_idx >= 5:  # Test first 5 batches
            break
    
    end_time = time.time()
    return end_time - start_time

print("Testing memory pinning:")

if torch.cuda.is_available():
    time_without_pinning = test_memory_pinning(use_pin_memory=False)
    time_with_pinning = test_memory_pinning(use_pin_memory=True)
    
    print(f"  Without pin_memory: {time_without_pinning*1000:.2f} ms")
    print(f"  With pin_memory: {time_with_pinning*1000:.2f} ms")
    print(f"  Speedup: {time_without_pinning/time_with_pinning:.2f}x")
else:
    print("  CUDA not available - skipping memory pinning test")

print("\n=== Error Handling and Timeouts ===")

class ProblematicDataset(Dataset):
    """Dataset that occasionally fails"""
    
    def __init__(self, size=50, error_probability=0.1):
        self.size = size
        self.error_probability = error_probability
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Randomly fail
        if np.random.random() < self.error_probability:
            raise RuntimeError(f"Simulated error at index {idx}")
        
        # Occasionally very slow
        if np.random.random() < 0.05:
            time.sleep(0.1)
        
        return torch.randn(5), torch.randint(0, 2, (1,)).item()

# Error handling function
def worker_init_fn(worker_id):
    """Initialize worker with different random seed"""
    np.random.seed(worker_id)

print("Testing error handling:")

problematic_dataset = ProblematicDataset(size=30, error_probability=0.05)
error_dataloader = DataLoader(
    problematic_dataset,
    batch_size=4,
    num_workers=1,
    timeout=1.0,  # 1 second timeout
    worker_init_fn=worker_init_fn
)

successful_batches = 0
failed_batches = 0

for batch_idx, batch in enumerate(error_dataloader):
    try:
        data, labels = batch
        successful_batches += 1
        if batch_idx >= 5:  # Test first 5 batches
            break
    except Exception as e:
        failed_batches += 1
        print(f"  Batch {batch_idx} failed: {e}")

print(f"  Successful batches: {successful_batches}")
print(f"  Failed batches: {failed_batches}")

print("\n=== Advanced DataLoader Configurations ===")

class AdvancedDataLoader:
    """Wrapper for advanced DataLoader configurations"""
    
    def __init__(self, dataset, config_name="default"):
        self.dataset = dataset
        self.config_name = config_name
        self.configs = self._get_configs()
        
    def _get_configs(self):
        """Define different DataLoader configurations"""
        return {
            "default": {
                "batch_size": 32,
                "shuffle": True,
                "num_workers": 0,
                "pin_memory": False
            },
            "high_throughput": {
                "batch_size": 64,
                "shuffle": True,
                "num_workers": min(4, multiprocessing.cpu_count()),
                "pin_memory": torch.cuda.is_available(),
                "persistent_workers": True,
                "prefetch_factor": 4
            },
            "memory_efficient": {
                "batch_size": 16,
                "shuffle": True,
                "num_workers": 1,
                "pin_memory": False,
                "drop_last": True
            },
            "debugging": {
                "batch_size": 4,
                "shuffle": False,
                "num_workers": 0,
                "drop_last": False,
                "timeout": 10.0
            },
            "distributed": {
                "batch_size": 32,
                "shuffle": False,  # Handled by DistributedSampler
                "num_workers": 2,
                "pin_memory": True,
                "drop_last": True
            }
        }
    
    def get_dataloader(self, config_name=None):
        """Get DataLoader with specified configuration"""
        config_name = config_name or self.config_name
        config = self.configs[config_name].copy()
        
        # Filter out None values and unsupported parameters
        filtered_config = {k: v for k, v in config.items() 
                          if v is not None and k in DataLoader.__init__.__code__.co_varnames}
        
        return DataLoader(self.dataset, **filtered_config)
    
    def compare_configs(self, config_names=None, num_batches=3):
        """Compare different configurations"""
        config_names = config_names or list(self.configs.keys())
        
        for config_name in config_names:
            print(f"  Testing {config_name} configuration:")
            
            try:
                dataloader = self.get_dataloader(config_name)
                
                start_time = time.time()
                batch_count = 0
                
                for batch_idx, (data, labels) in enumerate(dataloader):
                    batch_count += 1
                    if batch_idx >= num_batches - 1:
                        break
                
                end_time = time.time()
                
                avg_time = (end_time - start_time) / batch_count * 1000
                print(f"    Avg time per batch: {avg_time:.2f} ms")
                print(f"    Batches per epoch: {len(dataloader)}")
                
            except Exception as e:
                print(f"    Error: {e}")

# Test advanced configurations
print("Testing advanced DataLoader configurations:")

advanced_loader = AdvancedDataLoader(dataset)
test_configs = ["default", "memory_efficient", "debugging"]
advanced_loader.compare_configs(test_configs)

print("\n=== DataLoader Iteration Patterns ===")

def demonstrate_iteration_patterns():
    """Demonstrate different DataLoader iteration patterns"""
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    print("  Standard iteration:")
    batch_count = 0
    for batch_idx, (data, labels) in enumerate(dataloader):
        batch_count += 1
        if batch_idx >= 2:
            break
    print(f"    Processed {batch_count} batches")
    
    print("  Iterator pattern:")
    dataloader_iter = iter(dataloader)
    try:
        for i in range(3):
            batch = next(dataloader_iter)
            print(f"    Batch {i}: data shape {batch[0].shape}")
    except StopIteration:
        print("    Iterator exhausted")
    
    print("  Infinite iteration (for training loops):")
    def infinite_dataloader(dataloader):
        while True:
            for batch in dataloader:
                yield batch
    
    infinite_iter = infinite_dataloader(dataloader)
    for i in range(5):
        batch = next(infinite_iter)
        print(f"    Infinite batch {i}: shape {batch[0].shape}")

demonstrate_iteration_patterns()

print("\n=== DataLoader Performance Monitoring ===")

class DataLoaderProfiler:
    """Profile DataLoader performance"""
    
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.stats = {
            'batch_times': [],
            'data_loading_times': [],
            'total_time': 0,
            'num_batches': 0
        }
    
    def profile(self, num_epochs=1):
        """Profile DataLoader for specified epochs"""
        
        overall_start = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            for batch_idx, batch in enumerate(self.dataloader):
                batch_end = time.time()
                
                # Record batch time
                if batch_idx == 0:
                    batch_time = batch_end - epoch_start
                else:
                    batch_time = batch_end - prev_batch_end
                
                self.stats['batch_times'].append(batch_time)
                self.stats['num_batches'] += 1
                
                prev_batch_end = batch_end
        
        self.stats['total_time'] = time.time() - overall_start
        
        return self.stats
    
    def print_stats(self):
        """Print profiling statistics"""
        if not self.stats['batch_times']:
            print("    No profiling data available")
            return
        
        batch_times = np.array(self.stats['batch_times'])
        
        print(f"    Total time: {self.stats['total_time']:.3f} seconds")
        print(f"    Total batches: {self.stats['num_batches']}")
        print(f"    Avg batch time: {batch_times.mean()*1000:.2f} ms")
        print(f"    Min batch time: {batch_times.min()*1000:.2f} ms")
        print(f"    Max batch time: {batch_times.max()*1000:.2f} ms")
        print(f"    Std batch time: {batch_times.std()*1000:.2f} ms")

# Profile DataLoader performance
print("Profiling DataLoader performance:")

test_dataset = SimpleDataset(size=100)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=0)

profiler = DataLoaderProfiler(test_dataloader)
stats = profiler.profile(num_epochs=1)
profiler.print_stats()

print("\n=== DataLoader Best Practices ===")

print("Performance Optimization:")
print("1. Use appropriate batch_size (powers of 2 often work well)")
print("2. Enable pin_memory when using GPU")
print("3. Use multiple workers for I/O bound datasets")
print("4. Consider persistent_workers for faster worker initialization")
print("5. Optimize collate_fn for custom data structures")

print("\nMemory Management:")
print("1. Use drop_last=True to avoid small final batches")
print("2. Monitor memory usage with different num_workers")
print("3. Consider prefetch_factor for worker memory usage")
print("4. Use appropriate data types (float32 vs float64)")
print("5. Be careful with worker memory leaks")

print("\nDebugging Tips:")
print("1. Start with num_workers=0 for debugging")
print("2. Use small batch_size for initial testing")
print("3. Set timeout for hanging workers")
print("4. Implement proper error handling in datasets")
print("5. Use worker_init_fn for worker-specific initialization")

print("\nDistributed Training:")
print("1. Use DistributedSampler instead of shuffle=True")
print("2. Ensure batch_size is per-device, not global")
print("3. Set drop_last=True for consistent batch sizes")
print("4. Coordinate random seeds across workers")
print("5. Consider data loading imbalance across ranks")

print("\n=== DataLoader Comprehensive Complete ===")

# Memory cleanup
torch.cuda.empty_cache() if torch.cuda.is_available() else None