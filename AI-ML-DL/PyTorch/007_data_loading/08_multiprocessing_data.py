#!/usr/bin/env python3
"""PyTorch Multiprocessing Data - Multi-worker data loading"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import numpy as np
import time
import os
import pickle
import random
from multiprocessing import Queue, Process, Manager, Pool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any, Optional, Callable, Tuple
import threading
import queue
import warnings

print("=== Multiprocessing Data Loading Overview ===")

print("Multiprocessing topics covered:")
print("1. Basic DataLoader multiprocessing")
print("2. Worker initialization and management")
print("3. Shared memory and inter-process communication")
print("4. Custom multiprocessing strategies")
print("5. Memory-efficient data loading")
print("6. Error handling and debugging")
print("7. Performance optimization")
print("8. Distributed data loading patterns")

print("\n=== Basic DataLoader Multiprocessing ===")

class SimpleDataset(Dataset):
    """Simple dataset for multiprocessing examples"""
    
    def __init__(self, size=1000, feature_dim=20, simulate_io_delay=False):
        self.size = size
        self.feature_dim = feature_dim
        self.simulate_io_delay = simulate_io_delay
        
        # Generate data indices
        self.data_indices = list(range(size))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Simulate I/O delay for realistic testing
        if self.simulate_io_delay:
            time.sleep(0.001)  # 1ms delay
        
        # Generate deterministic data based on index
        torch.manual_seed(idx)
        data = torch.randn(self.feature_dim)
        label = torch.randint(0, 5, (1,)).item()
        
        return data, label

# Test basic multiprocessing performance
print("Testing basic DataLoader multiprocessing:")

def benchmark_dataloader(dataset, num_workers, batch_size=32, num_batches=50):
    """Benchmark DataLoader with different worker configurations"""
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0)
    )
    
    start_time = time.time()
    
    for batch_idx, (data, labels) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        # Simulate some processing
        _ = data.mean()
    
    end_time = time.time()
    return end_time - start_time

# Test with different worker counts
test_dataset = SimpleDataset(size=500, simulate_io_delay=True)
worker_counts = [0, 1, 2, 4]

print("DataLoader performance comparison:")
for num_workers in worker_counts:
    try:
        duration = benchmark_dataloader(test_dataset, num_workers, num_batches=20)
        print(f"  {num_workers} workers: {duration:.3f} seconds")
    except Exception as e:
        print(f"  {num_workers} workers: Error - {e}")

print("\n=== Worker Initialization and Management ===")

# Global variables for worker initialization
worker_info = {}

def worker_init_fn(worker_id):
    """Initialize worker with specific configuration"""
    global worker_info
    
    # Set different random seeds for each worker
    np.random.seed(worker_id)
    torch.manual_seed(worker_id)
    
    # Store worker information
    worker_info[worker_id] = {
        'process_id': os.getpid(),
        'initialization_time': time.time(),
        'data_processed': 0
    }
    
    print(f"  Worker {worker_id} initialized (PID: {os.getpid()})")

class WorkerAwareDataset(Dataset):
    """Dataset that interacts with worker information"""
    
    def __init__(self, size=200):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Get worker info if available
        worker_info_obj = torch.utils.data.get_worker_info()
        
        if worker_info_obj is not None:
            worker_id = worker_info_obj.id
            # Update worker statistics
            if worker_id in worker_info:
                worker_info[worker_id]['data_processed'] += 1
        else:
            worker_id = -1  # Main process
        
        # Generate data with worker-specific seed
        seed = idx + (worker_id * 1000) if worker_id >= 0 else idx
        torch.manual_seed(seed)
        
        data = torch.randn(10)
        label = torch.randint(0, 3, (1,)).item()
        
        return data, label, worker_id

# Test worker initialization
print("Testing worker initialization:")

worker_dataset = WorkerAwareDataset(size=100)
worker_dataloader = DataLoader(
    worker_dataset,
    batch_size=16,
    num_workers=2,
    worker_init_fn=worker_init_fn
)

print("Processing batches with worker tracking:")
batch_count = 0
for data, labels, worker_ids in worker_dataloader:
    if batch_count >= 5:
        break
    
    unique_workers = torch.unique(worker_ids)
    print(f"  Batch {batch_count}: processed by workers {unique_workers.tolist()}")
    batch_count += 1

print(f"Final worker statistics: {worker_info}")

print("\n=== Shared Memory and IPC ===")

class SharedMemoryDataset(Dataset):
    """Dataset using shared memory for efficient data sharing"""
    
    def __init__(self, size=500, feature_dim=50):
        self.size = size
        self.feature_dim = feature_dim
        
        # Create shared memory tensor
        self.shared_data = torch.randn(size, feature_dim).share_memory_()
        self.shared_labels = torch.randint(0, 10, (size,)).share_memory_()
        
        print(f"  Created shared memory dataset: {self.shared_data.shape}")
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Access shared memory directly - no copying needed
        return self.shared_data[idx], self.shared_labels[idx]

class MPManagerDataset(Dataset):
    """Dataset using multiprocessing Manager for shared state"""
    
    def __init__(self, size=200):
        self.size = size
        
        # Create managed shared state
        manager = Manager()
        self.shared_cache = manager.dict()
        self.access_stats = manager.list()
        
        # Pre-populate some cache entries
        for i in range(0, size, 10):
            self.shared_cache[i] = {
                'data': torch.randn(15).tolist(),
                'label': random.randint(0, 5)
            }
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Record access
        self.access_stats.append(idx)
        
        # Check cache first
        if idx in self.shared_cache:
            cached = self.shared_cache[idx]
            return torch.tensor(cached['data']), cached['label']
        
        # Generate new data
        torch.manual_seed(idx)
        data = torch.randn(15)
        label = torch.randint(0, 5, (1,)).item()
        
        # Update cache
        self.shared_cache[idx] = {
            'data': data.tolist(),
            'label': label
        }
        
        return data, label

# Test shared memory datasets
print("Testing shared memory datasets:")

# Test shared memory tensor dataset
shared_dataset = SharedMemoryDataset(size=100, feature_dim=20)
shared_dataloader = DataLoader(shared_dataset, batch_size=16, num_workers=2)

start_time = time.time()
for batch_idx, (data, labels) in enumerate(shared_dataloader):
    if batch_idx >= 5:
        break

shared_time = time.time() - start_time
print(f"  Shared memory dataset: {shared_time:.3f} seconds for 5 batches")

# Test manager dataset (commented out to avoid hanging)
print("  Manager dataset: (skipped to avoid potential hanging)")

print("\n=== Custom Multiprocessing Strategies ===")

class AsyncDataLoader:
    """Custom asynchronous data loader"""
    
    def __init__(self, dataset, batch_size=32, num_workers=2, buffer_size=10):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        
        # Create data queue
        self.data_queue = queue.Queue(maxsize=buffer_size)
        self.workers = []
        self.stop_event = threading.Event()
        
        # Start worker threads
        self._start_workers()
    
    def _start_workers(self):
        """Start worker threads"""
        indices = list(range(len(self.dataset)))
        
        # Split indices among workers
        indices_per_worker = len(indices) // self.num_workers
        
        for worker_id in range(self.num_workers):
            start_idx = worker_id * indices_per_worker
            end_idx = start_idx + indices_per_worker if worker_id < self.num_workers - 1 else len(indices)
            worker_indices = indices[start_idx:end_idx]
            
            worker = threading.Thread(
                target=self._worker_loop,
                args=(worker_id, worker_indices)
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def _worker_loop(self, worker_id, indices):
        """Worker loop for loading data"""
        batch_indices = []
        
        for idx in indices:
            if self.stop_event.is_set():
                break
            
            batch_indices.append(idx)
            
            if len(batch_indices) >= self.batch_size:
                # Create batch
                batch_data = []
                batch_labels = []
                
                for batch_idx in batch_indices:
                    data, label = self.dataset[batch_idx]
                    batch_data.append(data)
                    batch_labels.append(label)
                
                # Stack into tensors
                batch_data = torch.stack(batch_data)
                batch_labels = torch.tensor(batch_labels)
                
                # Add to queue
                try:
                    self.data_queue.put((batch_data, batch_labels), timeout=1.0)
                except queue.Full:
                    pass  # Skip if queue is full
                
                batch_indices = []
    
    def __iter__(self):
        """Iterator interface"""
        return self
    
    def __next__(self):
        """Get next batch"""
        try:
            return self.data_queue.get(timeout=5.0)
        except queue.Empty:
            self.stop()
            raise StopIteration
    
    def stop(self):
        """Stop all workers"""
        self.stop_event.set()
        
        # Clear queue
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break

class PipelineDataLoader:
    """Data loader with processing pipeline"""
    
    def __init__(self, dataset, transforms=None, batch_size=32, num_workers=2):
        self.dataset = dataset
        self.transforms = transforms or []
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def _process_batch(self, batch_data):
        """Apply transforms to batch"""
        for transform in self.transforms:
            batch_data = transform(batch_data)
        return batch_data
    
    def load_parallel(self, num_batches=10):
        """Load data in parallel with processing pipeline"""
        
        # Create batch indices
        all_indices = list(range(len(self.dataset)))
        random.shuffle(all_indices)
        
        batches = []
        for i in range(0, min(len(all_indices), num_batches * self.batch_size), self.batch_size):
            batch_indices = all_indices[i:i + self.batch_size]
            batches.append(batch_indices)
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            for batch_indices in batches:
                future = executor.submit(self._load_and_process_batch, batch_indices)
                futures.append(future)
            
            # Collect results
            results = []
            for future in futures:
                try:
                    batch_data, batch_labels = future.result(timeout=10.0)
                    results.append((batch_data, batch_labels))
                except Exception as e:
                    print(f"    Batch processing error: {e}")
            
            return results
    
    def _load_and_process_batch(self, indices):
        """Load and process a single batch"""
        batch_data = []
        batch_labels = []
        
        for idx in indices:
            data, label = self.dataset[idx]
            batch_data.append(data)
            batch_labels.append(label)
        
        # Stack and process
        batch_data = torch.stack(batch_data)
        batch_labels = torch.tensor(batch_labels)
        
        # Apply transforms
        batch_data = self._process_batch(batch_data)
        
        return batch_data, batch_labels

# Test custom multiprocessing strategies
print("Testing custom multiprocessing strategies:")

test_dataset_small = SimpleDataset(size=200, simulate_io_delay=False)

# Test async data loader
print("Testing async data loader:")
async_loader = AsyncDataLoader(test_dataset_small, batch_size=16, num_workers=2)

try:
    batch_count = 0
    start_time = time.time()
    
    for batch_data, batch_labels in async_loader:
        batch_count += 1
        if batch_count >= 5:
            break
    
    async_time = time.time() - start_time
    print(f"  Async loader: {async_time:.3f} seconds for {batch_count} batches")
    
finally:
    async_loader.stop()

# Test pipeline data loader
print("Testing pipeline data loader:")

# Define simple transforms
def normalize_transform(batch):
    return (batch - batch.mean()) / (batch.std() + 1e-8)

def noise_transform(batch):
    return batch + torch.randn_like(batch) * 0.01

pipeline_loader = PipelineDataLoader(
    test_dataset_small,
    transforms=[normalize_transform, noise_transform],
    batch_size=16,
    num_workers=2
)

start_time = time.time()
results = pipeline_loader.load_parallel(num_batches=5)
pipeline_time = time.time() - start_time

print(f"  Pipeline loader: {pipeline_time:.3f} seconds for {len(results)} batches")

print("\n=== Memory-Efficient Data Loading ===")

class MemoryEfficientDataset(Dataset):
    """Dataset with memory-efficient loading strategies"""
    
    def __init__(self, size=1000, chunk_size=100):
        self.size = size
        self.chunk_size = chunk_size
        self.current_chunk = None
        self.current_chunk_start = -1
        
        # Metadata only - no actual data stored
        self.data_metadata = [
            {'mean': random.uniform(-1, 1), 'std': random.uniform(0.5, 2.0)}
            for _ in range(size)
        ]
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Check if we need to load a new chunk
        chunk_start = (idx // self.chunk_size) * self.chunk_size
        
        if self.current_chunk is None or chunk_start != self.current_chunk_start:
            self._load_chunk(chunk_start)
        
        # Get data from current chunk
        local_idx = idx - self.current_chunk_start
        return self.current_chunk[local_idx]
    
    def _load_chunk(self, chunk_start):
        """Load a chunk of data into memory"""
        chunk_end = min(chunk_start + self.chunk_size, self.size)
        chunk_size = chunk_end - chunk_start
        
        # Generate chunk data
        chunk_data = []
        for i in range(chunk_size):
            global_idx = chunk_start + i
            metadata = self.data_metadata[global_idx]
            
            # Generate data based on metadata
            data = torch.randn(20) * metadata['std'] + metadata['mean']
            label = torch.randint(0, 5, (1,)).item()
            chunk_data.append((data, label))
        
        self.current_chunk = chunk_data
        self.current_chunk_start = chunk_start
        
        print(f"    Loaded chunk {chunk_start}-{chunk_end-1}")

class LazyLoadDataset(Dataset):
    """Dataset that loads data on-demand with caching"""
    
    def __init__(self, size=500, cache_size=50):
        self.size = size
        self.cache_size = cache_size
        self.cache = {}
        self.access_order = []
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Check cache first
        if idx in self.cache:
            # Move to end of access order
            self.access_order.remove(idx)
            self.access_order.append(idx)
            return self.cache[idx]
        
        # Load data
        data = self._load_data(idx)
        
        # Add to cache
        self._add_to_cache(idx, data)
        
        return data
    
    def _load_data(self, idx):
        """Simulate loading data from disk/network"""
        time.sleep(0.001)  # Simulate I/O delay
        
        torch.manual_seed(idx)
        data = torch.randn(25)
        label = torch.randint(0, 8, (1,)).item()
        
        return data, label
    
    def _add_to_cache(self, idx, data):
        """Add data to cache with LRU eviction"""
        if len(self.cache) >= self.cache_size:
            # Remove least recently used
            lru_idx = self.access_order.pop(0)
            del self.cache[lru_idx]
        
        self.cache[idx] = data
        self.access_order.append(idx)

# Test memory-efficient datasets
print("Testing memory-efficient data loading:")

# Test chunked loading
print("Testing chunked dataset:")
chunked_dataset = MemoryEfficientDataset(size=50, chunk_size=10)
chunked_loader = DataLoader(chunked_dataset, batch_size=8, shuffle=False)

chunk_batches = 0
for batch in chunked_loader:
    chunk_batches += 1
    if chunk_batches >= 3:
        break

print(f"  Processed {chunk_batches} batches with chunked loading")

# Test lazy loading with cache
print("Testing lazy loading with cache:")
lazy_dataset = LazyLoadDataset(size=100, cache_size=20)
lazy_loader = DataLoader(lazy_dataset, batch_size=16, shuffle=True)

start_time = time.time()
lazy_batches = 0
for batch in lazy_loader:
    lazy_batches += 1
    if lazy_batches >= 5:
        break

lazy_time = time.time() - start_time
print(f"  Lazy loading: {lazy_time:.3f} seconds for {lazy_batches} batches")
print(f"  Cache statistics: {len(lazy_dataset.cache)} items cached")

print("\n=== Error Handling and Debugging ===")

class ProblematicDataset(Dataset):
    """Dataset that occasionally fails for testing error handling"""
    
    def __init__(self, size=100, error_probability=0.1):
        self.size = size
        self.error_probability = error_probability
        self.error_count = 0
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Randomly fail
        if random.random() < self.error_probability:
            self.error_count += 1
            raise RuntimeError(f"Simulated error at index {idx}")
        
        # Occasionally return invalid data
        if random.random() < 0.05:
            return None, None  # Invalid data
        
        # Normal data
        torch.manual_seed(idx)
        data = torch.randn(10)
        label = torch.randint(0, 3, (1,)).item()
        
        return data, label

def robust_collate_fn(batch):
    """Robust collate function that handles None values"""
    # Filter out None values
    valid_batch = [(data, label) for data, label in batch if data is not None]
    
    if not valid_batch:
        # Return empty batch if all samples are invalid
        return torch.empty(0, 10), torch.empty(0, dtype=torch.long)
    
    # Standard collation for valid samples
    data_list, label_list = zip(*valid_batch)
    return torch.stack(data_list), torch.tensor(label_list)

class ErrorHandlingDataLoader:
    """Custom data loader with comprehensive error handling"""
    
    def __init__(self, dataset, batch_size=16, max_retries=3):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.error_stats = {'total_errors': 0, 'error_types': {}}
    
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            for retry in range(self.max_retries):
                try:
                    batch = self._load_batch(batch_indices)
                    yield batch
                    break
                except Exception as e:
                    self._record_error(e)
                    if retry == self.max_retries - 1:
                        print(f"    Failed to load batch after {self.max_retries} retries: {e}")
                        # Yield empty batch as fallback
                        yield torch.empty(0, 10), torch.empty(0, dtype=torch.long)
    
    def _load_batch(self, indices):
        """Load a batch with error handling"""
        batch_data = []
        batch_labels = []
        
        for idx in indices:
            try:
                data, label = self.dataset[idx]
                if data is not None:
                    batch_data.append(data)
                    batch_labels.append(label)
            except Exception as e:
                # Skip problematic sample
                self._record_error(e)
                continue
        
        if not batch_data:
            raise RuntimeError("No valid samples in batch")
        
        return torch.stack(batch_data), torch.tensor(batch_labels)
    
    def _record_error(self, error):
        """Record error statistics"""
        self.error_stats['total_errors'] += 1
        error_type = type(error).__name__
        self.error_stats['error_types'][error_type] = \
            self.error_stats['error_types'].get(error_type, 0) + 1

# Test error handling
print("Testing error handling and debugging:")

# Test problematic dataset with standard DataLoader
problematic_dataset = ProblematicDataset(size=80, error_probability=0.1)

print("Standard DataLoader with robust collate function:")
try:
    problem_loader = DataLoader(
        problematic_dataset,
        batch_size=16,
        collate_fn=robust_collate_fn,
        num_workers=0  # Single worker to avoid multiprocessing issues
    )
    
    batches_processed = 0
    for batch_data, batch_labels in problem_loader:
        batches_processed += 1
        print(f"  Batch {batches_processed}: {len(batch_data)} samples")
        if batches_processed >= 3:
            break
    
    print(f"  Dataset errors encountered: {problematic_dataset.error_count}")

except Exception as e:
    print(f"  DataLoader error: {e}")

# Test custom error handling data loader
print("Custom error-handling DataLoader:")
error_loader = ErrorHandlingDataLoader(problematic_dataset, batch_size=12)

batches_processed = 0
for batch_data, batch_labels in error_loader:
    batches_processed += 1
    print(f"  Batch {batches_processed}: {len(batch_data)} samples")
    if batches_processed >= 3:
        break

print(f"  Error statistics: {error_loader.error_stats}")

print("\n=== Performance Optimization Tips ===")

class OptimizedDataset(Dataset):
    """Dataset with various performance optimizations"""
    
    def __init__(self, size=1000):
        self.size = size
        
        # Pre-compute expensive operations
        self.precomputed_seeds = [hash(i) % 2**31 for i in range(size)]
        
        # Use efficient data structures
        self.metadata = np.array([(i % 10, i % 5) for i in range(size)], 
                                dtype=[('category', 'i4'), ('subcategory', 'i4')])
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Use pre-computed values
        seed = self.precomputed_seeds[idx]
        category = self.metadata[idx]['category']
        subcategory = self.metadata[idx]['subcategory']
        
        # Efficient data generation
        np.random.seed(seed)
        data = torch.from_numpy(np.random.randn(15).astype(np.float32))
        label = category
        
        return data, label

def optimize_dataloader_settings():
    """Demonstrate optimal DataLoader settings"""
    
    settings = {
        'batch_size': 64,  # Power of 2, GPU-friendly
        'num_workers': min(4, mp.cpu_count()),  # Don't exceed CPU cores
        'pin_memory': torch.cuda.is_available(),  # Enable for GPU
        'persistent_workers': True,  # Avoid worker restart overhead
        'prefetch_factor': 2,  # Balance memory vs speed
        'drop_last': True,  # Consistent batch sizes
    }
    
    return settings

# Test performance optimizations
print("Testing performance optimizations:")

optimized_dataset = OptimizedDataset(size=500)
optimal_settings = optimize_dataloader_settings()

print("Optimal DataLoader settings:")
for key, value in optimal_settings.items():
    print(f"  {key}: {value}")

# Benchmark optimized vs standard
standard_loader = DataLoader(optimized_dataset, batch_size=32, num_workers=0)
optimized_loader = DataLoader(optimized_dataset, **optimal_settings)

def benchmark_loader(loader, name, num_batches=20):
    start_time = time.time()
    for batch_idx, (data, labels) in enumerate(loader):
        if batch_idx >= num_batches:
            break
        # Simulate processing
        _ = data.mean()
    return time.time() - start_time

try:
    standard_time = benchmark_loader(standard_loader, "Standard", 10)
    optimized_time = benchmark_loader(optimized_loader, "Optimized", 10)
    
    print(f"\nPerformance comparison:")
    print(f"  Standard loader: {standard_time:.3f} seconds")
    print(f"  Optimized loader: {optimized_time:.3f} seconds")
    if standard_time > 0:
        speedup = standard_time / optimized_time
        print(f"  Speedup: {speedup:.2f}x")

except Exception as e:
    print(f"  Benchmark error: {e}")

print("\n=== Multiprocessing Best Practices ===")

print("Worker Configuration:")
print("1. Start with num_workers = min(4, cpu_count())")
print("2. Increase gradually while monitoring performance")
print("3. Consider I/O vs CPU bound workloads")
print("4. Use persistent_workers for faster initialization")
print("5. Monitor memory usage with multiple workers")

print("\nMemory Management:")
print("1. Enable pin_memory for GPU training")
print("2. Use shared memory tensors when possible")
print("3. Be careful with large datasets and multiple workers")
print("4. Consider prefetch_factor for memory/speed balance")
print("5. Use drop_last for consistent memory usage")

print("\nDebugging Tips:")
print("1. Start debugging with num_workers=0")
print("2. Use worker_init_fn for worker-specific setup")
print("3. Handle worker timeouts gracefully")
print("4. Monitor worker process health")
print("5. Use logging instead of print in workers")

print("\nCommon Pitfalls:")
print("1. Too many workers can hurt performance")
print("2. Shared state between workers can cause issues")
print("3. Large objects in dataset can cause memory issues")
print("4. Forgetting to handle worker crashes")
print("5. Not considering NUMA topology on large systems")

print("\nPlatform Considerations:")
print("1. Windows: Limited multiprocessing support")
print("2. macOS: spawn context required for CUDA")
print("3. Linux: Best multiprocessing performance")
print("4. Docker: Consider container CPU limits")
print("5. Cloud: Monitor network I/O with remote storage")

print("\n=== Multiprocessing Data Loading Complete ===")

# Cleanup
mp.set_sharing_strategy('file_system')  # Reset to default
torch.cuda.empty_cache() if torch.cuda.is_available() else None