#!/usr/bin/env python3
"""PyTorch Distributed Data Loading - Data loading for distributed training"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import os
import time
import random
from typing import List, Dict, Any, Optional, Tuple
import warnings

print("=== Distributed Data Loading Overview ===")

print("Distributed data loading topics:")
print("1. DistributedSampler basics")
print("2. Multi-GPU data loading strategies")
print("3. Custom distributed samplers")
print("4. Data sharding and partitioning")
print("5. Load balancing across workers")
print("6. Fault tolerance and recovery")
print("7. Performance optimization")
print("8. Best practices for different scenarios")

print("\n=== DistributedSampler Basics ===")

class SimpleDataset(Dataset):
    """Simple dataset for distributed training examples"""
    
    def __init__(self, size=1000, feature_dim=20):
        self.size = size
        self.feature_dim = feature_dim
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate deterministic data based on index
        torch.manual_seed(idx)
        data = torch.randn(self.feature_dim)
        label = torch.randint(0, 5, (1,)).item()
        return data, label

def demonstrate_distributed_sampler():
    """Demonstrate basic DistributedSampler usage"""
    
    print("Basic DistributedSampler demonstration:")
    
    dataset = SimpleDataset(size=100)
    
    # Simulate different numbers of processes
    world_sizes = [1, 2, 4]
    
    for world_size in world_sizes:
        print(f"\n  World size: {world_size}")
        
        for rank in range(world_size):
            # Create distributed sampler for this rank
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=42
            )
            
            # Create DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=8,
                sampler=sampler,
                num_workers=0
            )
            
            # Show sample distribution
            indices = list(sampler)
            print(f"    Rank {rank}: {len(indices)} samples, indices[0:5] = {indices[:5]}")

# Test basic distributed sampler
demonstrate_distributed_sampler()

print("\n=== Multi-GPU Data Loading Strategies ===")

class DistributedDatasetWrapper:
    """Wrapper for managing distributed dataset loading"""
    
    def __init__(self, dataset, world_size=None, rank=None):
        self.dataset = dataset
        self.world_size = world_size or 1
        self.rank = rank or 0
        
        # Calculate data distribution
        self.total_samples = len(dataset)
        self.samples_per_rank = self.total_samples // self.world_size
        self.remainder = self.total_samples % self.world_size
        
        # Determine this rank's data range
        self.start_idx = self.rank * self.samples_per_rank
        if self.rank < self.remainder:
            self.start_idx += self.rank
            self.end_idx = self.start_idx + self.samples_per_rank + 1
        else:
            self.start_idx += self.remainder
            self.end_idx = self.start_idx + self.samples_per_rank
        
        self.local_samples = self.end_idx - self.start_idx
        
        print(f"  Rank {self.rank}/{self.world_size}: samples {self.start_idx}-{self.end_idx-1} "
              f"({self.local_samples} samples)")
    
    def get_dataloader(self, batch_size=32, num_workers=0, shuffle=False):
        """Get DataLoader for this rank's data"""
        
        if shuffle:
            # Use DistributedSampler for shuffling
            sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
            
            return DataLoader(
                self.dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
        else:
            # Use subset for deterministic splitting
            from torch.utils.data import Subset
            subset_indices = list(range(self.start_idx, self.end_idx))
            subset = Subset(self.dataset, subset_indices)
            
            return DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
    
    def get_statistics(self):
        """Get distribution statistics"""
        return {
            'total_samples': self.total_samples,
            'world_size': self.world_size,
            'rank': self.rank,
            'local_samples': self.local_samples,
            'load_balance': self.local_samples / (self.total_samples / self.world_size)
        }

# Test multi-GPU data loading strategies
print("Testing multi-GPU data loading strategies:")

dataset = SimpleDataset(size=1000)

# Test different world sizes
for world_size in [2, 4, 8]:
    print(f"\nWorld size {world_size}:")
    
    wrappers = []
    for rank in range(world_size):
        wrapper = DistributedDatasetWrapper(dataset, world_size=world_size, rank=rank)
        wrappers.append(wrapper)
    
    # Show load balancing
    total_local_samples = sum(w.local_samples for w in wrappers)
    print(f"  Total samples distributed: {total_local_samples}")
    print(f"  Load balance factors: {[w.get_statistics()['load_balance'] for w in wrappers]}")

print("\n=== Custom Distributed Samplers ===")

class BalancedDistributedSampler(DistributedSampler):
    """Custom sampler that ensures balanced class distribution across ranks"""
    
    def __init__(self, dataset, labels, num_replicas=None, rank=None, shuffle=True, seed=0):
        super().__init__(dataset, num_replicas, rank, shuffle, seed)
        self.labels = labels
        self.class_indices = self._group_by_class()
    
    def _group_by_class(self):
        """Group sample indices by class"""
        class_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices
    
    def __iter__(self):
        """Generate balanced samples for this rank"""
        if self.shuffle:
            # Shuffle within each class
            for class_label in self.class_indices:
                random.seed(self.seed + self.epoch)
                random.shuffle(self.class_indices[class_label])
        
        # Distribute samples across ranks while maintaining class balance
        distributed_indices = []
        
        # Round-robin distribution within each class
        for class_label, indices in self.class_indices.items():
            class_samples_per_rank = len(indices) // self.num_replicas
            remainder = len(indices) % self.num_replicas
            
            # Calculate this rank's portion
            start_idx = self.rank * class_samples_per_rank
            if self.rank < remainder:
                start_idx += self.rank
                end_idx = start_idx + class_samples_per_rank + 1
            else:
                start_idx += remainder
                end_idx = start_idx + class_samples_per_rank
            
            rank_class_indices = indices[start_idx:end_idx]
            distributed_indices.extend(rank_class_indices)
        
        # Shuffle final list if needed
        if self.shuffle:
            random.seed(self.seed + self.epoch + self.rank)
            random.shuffle(distributed_indices)
        
        return iter(distributed_indices)

class WeightedDistributedSampler(DistributedSampler):
    """Distributed sampler with weighted sampling for imbalanced datasets"""
    
    def __init__(self, dataset, weights, num_replicas=None, rank=None, 
                 replacement=True, num_samples=None):
        super().__init__(dataset, num_replicas, rank, shuffle=False)
        
        self.weights = torch.tensor(weights, dtype=torch.float)
        self.replacement = replacement
        self.num_samples = num_samples or len(dataset)
    
    def __iter__(self):
        """Generate weighted samples for this rank"""
        # Calculate samples per rank
        samples_per_rank = self.num_samples // self.num_replicas
        
        # Generate weighted samples for this rank
        rank_samples = torch.multinomial(
            self.weights, 
            samples_per_rank, 
            replacement=self.replacement
        )
        
        return iter(rank_samples.tolist())
    
    def __len__(self):
        return self.num_samples // self.num_replicas

# Test custom distributed samplers
print("Testing custom distributed samplers:")

# Create imbalanced dataset
dataset_size = 1000
labels = np.concatenate([
    np.full(500, 0),  # Class 0: 500 samples
    np.full(300, 1),  # Class 1: 300 samples  
    np.full(150, 2),  # Class 2: 150 samples
    np.full(50, 3)    # Class 3: 50 samples
])
np.random.shuffle(labels)

imbalanced_dataset = SimpleDataset(size=dataset_size)

print("Testing balanced distributed sampler:")
for rank in range(2):
    balanced_sampler = BalancedDistributedSampler(
        imbalanced_dataset, 
        labels, 
        num_replicas=2, 
        rank=rank,
        shuffle=True
    )
    
    # Count class distribution
    rank_indices = list(balanced_sampler)
    rank_labels = [labels[i] for i in rank_indices]
    
    class_counts = {}
    for label in rank_labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print(f"  Rank {rank}: {len(rank_indices)} samples, class distribution: {class_counts}")

# Test weighted sampler
print("\nTesting weighted distributed sampler:")

# Create weights inversely proportional to class frequency
class_counts = np.bincount(labels)
weights = 1.0 / class_counts[labels]

for rank in range(2):
    weighted_sampler = WeightedDistributedSampler(
        imbalanced_dataset,
        weights,
        num_replicas=2,
        rank=rank,
        num_samples=200
    )
    
    rank_indices = list(weighted_sampler)
    rank_labels = [labels[i] for i in rank_indices]
    
    weighted_class_counts = {}
    for label in rank_labels:
        weighted_class_counts[label] = weighted_class_counts.get(label, 0) + 1
    
    print(f"  Rank {rank}: {len(rank_indices)} samples, weighted distribution: {weighted_class_counts}")

print("\n=== Data Sharding and Partitioning ===")

class ShardedDataset(Dataset):
    """Dataset that implements sharding at the dataset level"""
    
    def __init__(self, base_dataset, world_size=1, rank=0, shard_strategy='round_robin'):
        self.base_dataset = base_dataset
        self.world_size = world_size
        self.rank = rank
        self.shard_strategy = shard_strategy
        
        # Create sharding mapping
        self.local_indices = self._create_shard_mapping()
        
        print(f"  Shard {rank}/{world_size}: {len(self.local_indices)} samples "
              f"(strategy: {shard_strategy})")
    
    def _create_shard_mapping(self):
        """Create mapping of global indices to local shard"""
        total_samples = len(self.base_dataset)
        
        if self.shard_strategy == 'round_robin':
            # Round-robin assignment
            return [i for i in range(total_samples) if i % self.world_size == self.rank]
        
        elif self.shard_strategy == 'contiguous':
            # Contiguous blocks
            samples_per_shard = total_samples // self.world_size
            remainder = total_samples % self.world_size
            
            start_idx = self.rank * samples_per_shard
            if self.rank < remainder:
                start_idx += self.rank
                end_idx = start_idx + samples_per_shard + 1
            else:
                start_idx += remainder
                end_idx = start_idx + samples_per_shard
            
            return list(range(start_idx, end_idx))
        
        elif self.shard_strategy == 'hash_based':
            # Hash-based sharding for better load balancing
            return [i for i in range(total_samples) if hash(i) % self.world_size == self.rank]
        
        else:
            raise ValueError(f"Unknown shard strategy: {self.shard_strategy}")
    
    def __len__(self):
        return len(self.local_indices)
    
    def __getitem__(self, idx):
        # Map local index to global index
        global_idx = self.local_indices[idx]
        return self.base_dataset[global_idx]
    
    def get_global_index(self, local_idx):
        """Convert local index to global index"""
        return self.local_indices[local_idx]

# Test data sharding strategies
print("Testing data sharding strategies:")

base_dataset = SimpleDataset(size=100)
strategies = ['round_robin', 'contiguous', 'hash_based']

for strategy in strategies:
    print(f"\nStrategy: {strategy}")
    
    shards = []
    for rank in range(4):
        shard = ShardedDataset(base_dataset, world_size=4, rank=rank, shard_strategy=strategy)
        shards.append(shard)
    
    # Verify no overlap and complete coverage
    all_indices = set()
    for shard in shards:
        shard_indices = set(shard.local_indices)
        overlap = all_indices.intersection(shard_indices)
        if overlap:
            print(f"  WARNING: Overlap detected: {overlap}")
        all_indices.update(shard_indices)
    
    coverage = len(all_indices) / len(base_dataset)
    print(f"  Coverage: {coverage:.2%} ({len(all_indices)}/{len(base_dataset)} samples)")

print("\n=== Load Balancing Across Workers ===")

class LoadBalancedDataset(Dataset):
    """Dataset that monitors and balances load across workers"""
    
    def __init__(self, base_dataset, world_size=1, rank=0):
        self.base_dataset = base_dataset
        self.world_size = world_size
        self.rank = rank
        
        # Load balancing metrics
        self.access_count = 0
        self.access_times = []
        self.sample_complexities = self._compute_sample_complexities()
        
        # Dynamic load balancing
        self.local_indices = self._balance_load()
    
    def _compute_sample_complexities(self):
        """Compute relative complexity of each sample"""
        # In real scenarios, this could be based on:
        # - File sizes, processing time, network latency, etc.
        # For demo, use random complexities
        np.random.seed(42)
        return np.random.exponential(1.0, len(self.base_dataset))
    
    def _balance_load(self):
        """Balance load across workers based on sample complexities"""
        total_samples = len(self.base_dataset)
        
        # Sort samples by complexity
        complexity_order = np.argsort(self.sample_complexities)
        
        # Distribute samples to minimize load imbalance
        worker_loads = [0.0] * self.world_size
        worker_assignments = [[] for _ in range(self.world_size)]
        
        # Greedy assignment - assign each sample to least loaded worker
        for sample_idx in complexity_order:
            min_load_worker = np.argmin(worker_loads)
            worker_assignments[min_load_worker].append(sample_idx)
            worker_loads[min_load_worker] += self.sample_complexities[sample_idx]
        
        # Return assignments for this worker
        local_indices = worker_assignments[self.rank]
        total_complexity = sum(self.sample_complexities[i] for i in local_indices)
        
        print(f"  Worker {self.rank}: {len(local_indices)} samples, "
              f"total complexity: {total_complexity:.2f}")
        
        return local_indices
    
    def __len__(self):
        return len(self.local_indices)
    
    def __getitem__(self, idx):
        start_time = time.time()
        
        global_idx = self.local_indices[idx]
        result = self.base_dataset[global_idx]
        
        # Simulate variable processing time based on complexity
        complexity = self.sample_complexities[global_idx]
        time.sleep(complexity * 0.001)  # Scale down for demo
        
        # Track metrics
        self.access_count += 1
        self.access_times.append(time.time() - start_time)
        
        return result
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.access_times:
            return {}
        
        return {
            'access_count': self.access_count,
            'avg_access_time': np.mean(self.access_times),
            'total_access_time': sum(self.access_times),
            'min_access_time': min(self.access_times),
            'max_access_time': max(self.access_times)
        }

# Test load balancing
print("Testing load balancing:")

base_dataset = SimpleDataset(size=200)

# Create load-balanced workers
workers = []
for rank in range(4):
    worker = LoadBalancedDataset(base_dataset, world_size=4, rank=rank)
    workers.append(worker)

# Simulate access patterns
print("\nSimulating data access:")
for worker in workers:
    # Access some samples to generate statistics
    for i in range(min(20, len(worker))):
        _ = worker[i]
    
    stats = worker.get_performance_stats()
    print(f"  Worker {worker.rank}: {stats['access_count']} accesses, "
          f"avg time: {stats['avg_access_time']*1000:.2f}ms")

print("\n=== Fault Tolerance and Recovery ===")

class FaultTolerantDataLoader:
    """DataLoader with fault tolerance and recovery mechanisms"""
    
    def __init__(self, dataset, batch_size=32, world_size=1, rank=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        
        # Fault tolerance state
        self.checkpoint_state = {}
        self.failed_samples = set()
        self.retry_count = {}
        self.max_retries = 3
        
        # Create reliable sampler
        self.sampler = self._create_reliable_sampler()
    
    def _create_reliable_sampler(self):
        """Create sampler with fault tolerance"""
        # Start with standard distributed sampler
        base_sampler = DistributedSampler(
            self.dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        
        return base_sampler
    
    def __iter__(self):
        """Iterate with fault tolerance"""
        sampler_iter = iter(self.sampler)
        batch_indices = []
        
        while True:
            try:
                # Get next index
                idx = next(sampler_iter)
                
                # Skip failed samples that exceeded retry limit
                if idx in self.failed_samples:
                    continue
                
                batch_indices.append(idx)
                
                # Yield batch when ready
                if len(batch_indices) >= self.batch_size:
                    yield self._load_batch_with_retry(batch_indices)
                    batch_indices = []
                    
            except StopIteration:
                # Yield final partial batch if exists
                if batch_indices:
                    yield self._load_batch_with_retry(batch_indices)
                break
    
    def _load_batch_with_retry(self, indices):
        """Load batch with retry logic"""
        batch_data = []
        batch_labels = []
        
        for idx in indices:
            try:
                data, label = self._load_sample_with_retry(idx)
                batch_data.append(data)
                batch_labels.append(label)
                
            except Exception as e:
                print(f"    Failed to load sample {idx} after retries: {e}")
                continue
        
        if not batch_data:
            # Return empty batch if all samples failed
            return torch.empty(0, 20), torch.empty(0, dtype=torch.long)
        
        return torch.stack(batch_data), torch.tensor(batch_labels)
    
    def _load_sample_with_retry(self, idx):
        """Load single sample with retry logic"""
        for attempt in range(self.max_retries):
            try:
                # Simulate occasional failures
                if random.random() < 0.05:  # 5% failure rate
                    raise RuntimeError(f"Simulated failure for sample {idx}")
                
                return self.dataset[idx]
                
            except Exception as e:
                self.retry_count[idx] = self.retry_count.get(idx, 0) + 1
                
                if attempt == self.max_retries - 1:
                    # Mark as permanently failed
                    self.failed_samples.add(idx)
                    raise e
                
                # Wait before retry
                time.sleep(0.001 * (attempt + 1))
        
        raise RuntimeError(f"Failed to load sample {idx}")
    
    def save_checkpoint(self, filepath):
        """Save fault tolerance state"""
        state = {
            'failed_samples': list(self.failed_samples),
            'retry_count': self.retry_count,
            'sampler_state': getattr(self.sampler, 'epoch', 0)
        }
        
        torch.save(state, filepath)
    
    def load_checkpoint(self, filepath):
        """Load fault tolerance state"""
        if os.path.exists(filepath):
            state = torch.load(filepath)
            self.failed_samples = set(state['failed_samples'])
            self.retry_count = state['retry_count']
            
            if hasattr(self.sampler, 'set_epoch'):
                self.sampler.set_epoch(state['sampler_state'])
    
    def get_fault_statistics(self):
        """Get fault tolerance statistics"""
        return {
            'failed_samples': len(self.failed_samples),
            'total_retries': sum(self.retry_count.values()),
            'samples_with_retries': len(self.retry_count),
            'max_retries_per_sample': max(self.retry_count.values()) if self.retry_count else 0
        }

# Test fault tolerance
print("Testing fault tolerance:")

fault_dataset = SimpleDataset(size=100)
fault_loader = FaultTolerantDataLoader(fault_dataset, batch_size=16, world_size=2, rank=0)

print("Processing batches with simulated failures:")
batch_count = 0
for batch_data, batch_labels in fault_loader:
    batch_count += 1
    print(f"  Batch {batch_count}: {len(batch_data)} samples")
    
    if batch_count >= 5:
        break

fault_stats = fault_loader.get_fault_statistics()
print(f"  Fault statistics: {fault_stats}")

print("\n=== Performance Optimization for Distributed Loading ===")

def benchmark_distributed_performance():
    """Benchmark different distributed loading strategies"""
    
    print("Benchmarking distributed loading performance:")
    
    dataset = SimpleDataset(size=1000)
    batch_size = 32
    num_batches = 20
    
    strategies = [
        ('Standard DistributedSampler', lambda: DistributedSampler(dataset, num_replicas=2, rank=0)),
        ('Balanced Sampler', lambda: BalancedDistributedSampler(dataset, [i%5 for i in range(len(dataset))], num_replicas=2, rank=0)),
    ]
    
    for name, sampler_fn in strategies:
        sampler = sampler_fn()
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=False
        )
        
        # Benchmark
        start_time = time.time()
        
        for batch_idx, (data, labels) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            # Simulate processing
            _ = data.mean()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"  {name:25}: {duration:.3f} seconds ({duration/num_batches*1000:.1f} ms/batch)")

benchmark_distributed_performance()

print("\n=== Distributed Loading Best Practices ===")

print("Sampler Selection:")
print("1. Use DistributedSampler for standard distributed training")
print("2. Implement custom samplers for imbalanced datasets")
print("3. Consider weighted sampling for rare classes")
print("4. Ensure reproducible sampling with proper seeding")
print("5. Set sampler.set_epoch() for proper shuffling")

print("\nData Distribution:")
print("1. Minimize data skew across workers")
print("2. Consider sample complexity for load balancing")
print("3. Use appropriate sharding strategies")
print("4. Monitor worker utilization during training")
print("5. Account for variable processing times")

print("\nPerformance Optimization:")
print("1. Use pin_memory=True for GPU training")
print("2. Optimize num_workers based on I/O characteristics")
print("3. Consider prefetch_factor for memory/speed trade-off")
print("4. Use persistent_workers to reduce overhead")
print("5. Profile data loading vs training time ratio")

print("\nFault Tolerance:")
print("1. Implement retry mechanisms for failed samples")
print("2. Save/restore sampler state for recovery")
print("3. Monitor and log data loading errors")
print("4. Use redundant data sources when possible")
print("5. Design graceful degradation strategies")

print("\nScaling Considerations:")
print("1. Test performance with different world sizes")
print("2. Monitor memory usage per worker")
print("3. Consider network bandwidth for shared storage")
print("4. Account for storage I/O contention")
print("5. Benchmark with realistic data sizes")

print("\nDebugging Tips:")
print("1. Start with single-process debugging")
print("2. Verify data distribution across workers")
print("3. Check for synchronization issues")
print("4. Monitor resource utilization")
print("5. Use deterministic seeds for reproducibility")

print("\n=== Distributed Data Loading Complete ===")

# Memory cleanup
torch.cuda.empty_cache() if torch.cuda.is_available() else None