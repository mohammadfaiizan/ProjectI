#!/usr/bin/env python3
"""PyTorch Data Caching Strategies - Caching and performance optimization"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
import pickle
import tempfile
import hashlib
import threading
from collections import OrderedDict, defaultdict
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import weakref
import gc
import psutil
import warnings

print("=== Data Caching Strategies Overview ===")

print("Caching strategy topics:")
print("1. In-memory caching strategies")
print("2. Disk-based caching systems")
print("3. Multi-level caching hierarchies")
print("4. Cache replacement policies")
print("5. Distributed caching")
print("6. Performance monitoring and optimization")
print("7. Memory-aware caching")
print("8. Cache coherence and invalidation")

print("\n=== In-Memory Caching Strategies ===")

class LRUCache:
    """Least Recently Used cache implementation"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: Any) -> Optional[Any]:
        """Get item from cache"""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        else:
            self.misses += 1
            return None
    
    def put(self, key: Any, value: Any) -> None:
        """Put item into cache"""
        if key in self.cache:
            # Update existing key
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            # Remove least recently used item
            self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache),
            'capacity': self.capacity
        }
    
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

class InMemoryCachedDataset(Dataset):
    """Dataset with in-memory LRU caching"""
    
    def __init__(self, base_dataset: Dataset, cache_size: int = 1000):
        self.base_dataset = base_dataset
        self.cache = LRUCache(cache_size)
        self.load_times = []
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Any:
        # Check cache first
        cached_item = self.cache.get(idx)
        if cached_item is not None:
            return cached_item
        
        # Load from base dataset
        start_time = time.time()
        
        # Simulate expensive loading operation
        time.sleep(0.001)  # 1ms delay
        item = self.base_dataset[idx]
        
        load_time = time.time() - start_time
        self.load_times.append(load_time)
        
        # Cache the item
        self.cache.put(idx, item)
        
        return item
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching statistics"""
        stats = self.cache.get_stats()
        if self.load_times:
            stats['avg_load_time'] = np.mean(self.load_times)
            stats['total_load_time'] = sum(self.load_times)
        return stats
    
    def clear_cache(self) -> None:
        """Clear cache and reset statistics"""
        self.cache.clear()
        self.load_times = []

# Test in-memory caching
print("Testing in-memory caching:")

class SimpleDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate deterministic data
        torch.manual_seed(idx)
        return torch.randn(50), torch.randint(0, 10, (1,)).item()

# Create cached dataset
base_dataset = SimpleDataset(size=500)
cached_dataset = InMemoryCachedDataset(base_dataset, cache_size=100)

# Test caching performance
print("Testing cache performance:")

# First pass - cold cache
start_time = time.time()
for i in range(150):  # Access more than cache size
    _ = cached_dataset[i % 500]
cold_time = time.time() - start_time

# Second pass - warm cache (repeat recent accesses)
start_time = time.time()
for i in range(50):  # Access within cache size
    _ = cached_dataset[i]
warm_time = time.time() - start_time

print(f"  Cold cache (150 accesses): {cold_time:.3f} seconds")
print(f"  Warm cache (50 accesses): {warm_time:.3f} seconds")
print(f"  Cache statistics: {cached_dataset.get_cache_stats()}")

print("\n=== Disk-Based Caching Systems ===")

class DiskCache:
    """Disk-based cache with configurable policies"""
    
    def __init__(self, cache_dir: str, max_size_mb: int = 100):
        self.cache_dir = cache_dir
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.index_file = os.path.join(cache_dir, "cache_index.pkl")
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing index or create new one
        self.index = self._load_index()
        self.access_order = []
        
        # Statistics
        self.hits = 0
        self.misses = 0
    
    def _load_index(self) -> Dict[str, Dict]:
        """Load cache index from disk"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def _save_index(self) -> None:
        """Save cache index to disk"""
        with open(self.index_file, 'wb') as f:
            pickle.dump(self.index, f)
    
    def _get_cache_file(self, key: str) -> str:
        """Get cache file path for key"""
        key_hash = hashlib.md5(str(key).encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.cache")
    
    def _get_current_size(self) -> int:
        """Get current cache size in bytes"""
        total_size = 0
        for key, info in self.index.items():
            cache_file = self._get_cache_file(key)
            if os.path.exists(cache_file):
                total_size += info.get('size', 0)
        return total_size
    
    def _cleanup_old_entries(self) -> None:
        """Remove old entries to stay within size limit"""
        while self._get_current_size() > self.max_size_bytes and self.access_order:
            # Remove least recently accessed
            old_key = self.access_order.pop(0)
            if old_key in self.index:
                cache_file = self._get_cache_file(old_key)
                try:
                    os.remove(cache_file)
                    del self.index[old_key]
                except OSError:
                    pass
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from disk cache"""
        if key in self.index:
            cache_file = self._get_cache_file(key)
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Update access order
                    if key in self.access_order:
                        self.access_order.remove(key)
                    self.access_order.append(key)
                    
                    self.hits += 1
                    return data
                except:
                    # Remove corrupted entry
                    if key in self.index:
                        del self.index[key]
        
        self.misses += 1
        return None
    
    def put(self, key: str, data: Any) -> bool:
        """Put item into disk cache"""
        try:
            cache_file = self._get_cache_file(key)
            
            # Serialize data to get size
            serialized_data = pickle.dumps(data)
            data_size = len(serialized_data)
            
            # Check if data is too large
            if data_size > self.max_size_bytes:
                return False
            
            # Cleanup if necessary
            self._cleanup_old_entries()
            
            # Write to cache
            with open(cache_file, 'wb') as f:
                f.write(serialized_data)
            
            # Update index
            self.index[key] = {
                'size': data_size,
                'timestamp': time.time()
            }
            
            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            self._save_index()
            return True
            
        except Exception as e:
            print(f"Cache write error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'entries': len(self.index),
            'size_mb': self._get_current_size() / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024)
        }
    
    def clear(self) -> None:
        """Clear cache"""
        for key in list(self.index.keys()):
            cache_file = self._get_cache_file(key)
            try:
                os.remove(cache_file)
            except OSError:
                pass
        
        self.index.clear()
        self.access_order.clear()
        self.hits = 0
        self.misses = 0
        self._save_index()

class DiskCachedDataset(Dataset):
    """Dataset with disk-based caching"""
    
    def __init__(self, base_dataset: Dataset, cache_dir: str, cache_size_mb: int = 50):
        self.base_dataset = base_dataset
        self.disk_cache = DiskCache(cache_dir, max_size_mb=cache_size_mb)
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Any:
        # Try disk cache first
        cache_key = f"item_{idx}"
        cached_item = self.disk_cache.get(cache_key)
        
        if cached_item is not None:
            return cached_item
        
        # Load from base dataset
        item = self.base_dataset[idx]
        
        # Cache to disk
        self.disk_cache.put(cache_key, item)
        
        return item
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.disk_cache.get_stats()

# Test disk-based caching
print("Testing disk-based caching:")

temp_cache_dir = tempfile.mkdtemp(prefix="pytorch_cache_")
disk_cached_dataset = DiskCachedDataset(base_dataset, temp_cache_dir, cache_size_mb=5)

print("Testing disk cache performance:")

# Test with repeated access patterns
start_time = time.time()
for i in range(100):
    _ = disk_cached_dataset[i % 20]  # Repeated access to 20 items
disk_cache_time = time.time() - start_time

print(f"  Disk cache (100 accesses): {disk_cache_time:.3f} seconds")
print(f"  Disk cache statistics: {disk_cached_dataset.get_cache_stats()}")

print("\n=== Multi-Level Caching Hierarchy ===")

class MultiLevelCache:
    """Multi-level cache with L1 (memory) and L2 (disk) tiers"""
    
    def __init__(self, l1_capacity: int, l2_cache_dir: str, l2_size_mb: int = 100):
        self.l1_cache = LRUCache(l1_capacity)
        self.l2_cache = DiskCache(l2_cache_dir, max_size_mb=l2_size_mb)
        
        # Statistics
        self.l1_hits = 0
        self.l2_hits = 0
        self.misses = 0
    
    def get(self, key: Any) -> Optional[Any]:
        """Get item from multi-level cache"""
        # Try L1 first
        item = self.l1_cache.get(key)
        if item is not None:
            self.l1_hits += 1
            return item
        
        # Try L2
        item = self.l2_cache.get(str(key))
        if item is not None:
            # Promote to L1
            self.l1_cache.put(key, item)
            self.l2_hits += 1
            return item
        
        # Cache miss
        self.misses += 1
        return None
    
    def put(self, key: Any, value: Any) -> None:
        """Put item into multi-level cache"""
        # Store in both L1 and L2
        self.l1_cache.put(key, value)
        self.l2_cache.put(str(key), value)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        total_requests = self.l1_hits + self.l2_hits + self.misses
        
        stats = {
            'l1_hits': self.l1_hits,
            'l2_hits': self.l2_hits,
            'misses': self.misses,
            'total_requests': total_requests,
            'l1_hit_rate': self.l1_hits / total_requests if total_requests > 0 else 0,
            'l2_hit_rate': self.l2_hits / total_requests if total_requests > 0 else 0,
            'overall_hit_rate': (self.l1_hits + self.l2_hits) / total_requests if total_requests > 0 else 0,
            'l1_stats': self.l1_cache.get_stats(),
            'l2_stats': self.l2_cache.get_stats()
        }
        
        return stats

class MultiLevelCachedDataset(Dataset):
    """Dataset with multi-level caching"""
    
    def __init__(self, base_dataset: Dataset, l1_size: int, l2_cache_dir: str, l2_size_mb: int = 50):
        self.base_dataset = base_dataset
        self.cache = MultiLevelCache(l1_size, l2_cache_dir, l2_size_mb)
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Any:
        # Try multi-level cache
        cached_item = self.cache.get(idx)
        if cached_item is not None:
            return cached_item
        
        # Load from base dataset
        item = self.base_dataset[idx]
        
        # Cache at all levels
        self.cache.put(idx, item)
        
        return item
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()

# Test multi-level caching
print("Testing multi-level caching:")

temp_l2_dir = tempfile.mkdtemp(prefix="pytorch_l2_cache_")
multilevel_dataset = MultiLevelCachedDataset(base_dataset, l1_size=50, l2_cache_dir=temp_l2_dir, l2_size_mb=10)

print("Testing multi-level cache performance:")

# Test access pattern that benefits from multi-level caching
access_pattern = []
# Recent items (L1 hits)
access_pattern.extend([i for i in range(20)] * 3)
# Older items (L2 hits)
access_pattern.extend([i for i in range(20, 80)] * 2)
# New items (misses)
access_pattern.extend([i for i in range(80, 120)])

start_time = time.time()
for idx in access_pattern:
    _ = multilevel_dataset[idx]
multilevel_time = time.time() - start_time

print(f"  Multi-level cache ({len(access_pattern)} accesses): {multilevel_time:.3f} seconds")
stats = multilevel_dataset.get_cache_stats()
print(f"  L1 hit rate: {stats['l1_hit_rate']:.2%}")
print(f"  L2 hit rate: {stats['l2_hit_rate']:.2%}")
print(f"  Overall hit rate: {stats['overall_hit_rate']:.2%}")

print("\n=== Memory-Aware Caching ===")

class MemoryAwareCache:
    """Cache that adapts based on available system memory"""
    
    def __init__(self, max_memory_percent: float = 10.0):
        self.max_memory_percent = max_memory_percent
        self.cache = OrderedDict()
        self.memory_usage = 0
        self.hits = 0
        self.misses = 0
        
        # Get initial memory info
        self.process = psutil.Process()
        self.system_memory = psutil.virtual_memory().total
        
    def _get_memory_limit(self) -> int:
        """Calculate current memory limit based on system state"""
        available_memory = psutil.virtual_memory().available
        max_allowed = self.system_memory * (self.max_memory_percent / 100)
        
        # Use conservative limit based on available memory
        return min(max_allowed, available_memory * 0.8)
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object"""
        try:
            import sys
            return sys.getsizeof(obj)
        except:
            # Fallback estimation
            if isinstance(obj, torch.Tensor):
                return obj.numel() * obj.element_size()
            else:
                return 1000  # Default estimate
    
    def _cleanup_memory(self) -> None:
        """Clean up cache to fit memory constraints"""
        memory_limit = self._get_memory_limit()
        
        while self.memory_usage > memory_limit and self.cache:
            # Remove least recently used item
            key, value = self.cache.popitem(last=False)
            self.memory_usage -= self._estimate_size(value)
            
            # Trigger garbage collection periodically
            if len(self.cache) % 100 == 0:
                gc.collect()
    
    def get(self, key: Any) -> Optional[Any]:
        """Get item from memory-aware cache"""
        if key in self.cache:
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            self.hits += 1
            return value
        else:
            self.misses += 1
            return None
    
    def put(self, key: Any, value: Any) -> bool:
        """Put item into memory-aware cache"""
        item_size = self._estimate_size(value)
        memory_limit = self._get_memory_limit()
        
        # Don't cache if item is too large
        if item_size > memory_limit * 0.5:
            return False
        
        # Remove existing key if present
        if key in self.cache:
            old_value = self.cache.pop(key)
            self.memory_usage -= self._estimate_size(old_value)
        
        # Add new item
        self.cache[key] = value
        self.memory_usage += item_size
        
        # Cleanup if necessary
        self._cleanup_memory()
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache and memory statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        memory_info = psutil.virtual_memory()
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'entries': len(self.cache),
            'memory_usage_mb': self.memory_usage / (1024 * 1024),
            'memory_limit_mb': self._get_memory_limit() / (1024 * 1024),
            'system_memory_usage_percent': memory_info.percent,
            'available_memory_mb': memory_info.available / (1024 * 1024)
        }

class MemoryAwareCachedDataset(Dataset):
    """Dataset with memory-aware caching"""
    
    def __init__(self, base_dataset: Dataset, max_memory_percent: float = 5.0):
        self.base_dataset = base_dataset
        self.cache = MemoryAwareCache(max_memory_percent)
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Any:
        # Try cache first
        cached_item = self.cache.get(idx)
        if cached_item is not None:
            return cached_item
        
        # Load from base dataset
        item = self.base_dataset[idx]
        
        # Cache if memory allows
        self.cache.put(idx, item)
        
        return item
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()

# Test memory-aware caching
print("Testing memory-aware caching:")

memory_aware_dataset = MemoryAwareCachedDataset(base_dataset, max_memory_percent=1.0)

print("Testing memory-aware cache:")

# Access pattern that would normally exceed memory
for i in range(200):
    _ = memory_aware_dataset[i]

stats = memory_aware_dataset.get_cache_stats()
print(f"  Memory-aware cache stats:")
print(f"    Hit rate: {stats['hit_rate']:.2%}")
print(f"    Memory usage: {stats['memory_usage_mb']:.1f} MB")
print(f"    Memory limit: {stats['memory_limit_mb']:.1f} MB")
print(f"    System memory usage: {stats['system_memory_usage_percent']:.1f}%")

print("\n=== Performance Monitoring and Optimization ===")

class CachePerformanceMonitor:
    """Monitor and optimize cache performance"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()
    
    def record_access(self, cache_name: str, hit: bool, latency: float, memory_usage: int = 0) -> None:
        """Record cache access metrics"""
        self.metrics[cache_name].append({
            'timestamp': time.time() - self.start_time,
            'hit': hit,
            'latency': latency,
            'memory_usage': memory_usage
        })
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {}
        
        for cache_name, metrics in self.metrics.items():
            if not metrics:
                continue
            
            hits = sum(1 for m in metrics if m['hit'])
            total = len(metrics)
            hit_rate = hits / total if total > 0 else 0
            
            latencies = [m['latency'] for m in metrics]
            avg_latency = np.mean(latencies) if latencies else 0
            
            cache_report = {
                'total_accesses': total,
                'hits': hits,
                'misses': total - hits,
                'hit_rate': hit_rate,
                'avg_latency_ms': avg_latency * 1000,
                'p95_latency_ms': np.percentile(latencies, 95) * 1000 if latencies else 0,
                'p99_latency_ms': np.percentile(latencies, 99) * 1000 if latencies else 0
            }
            
            # Memory usage stats if available
            memory_usages = [m['memory_usage'] for m in metrics if m['memory_usage'] > 0]
            if memory_usages:
                cache_report['avg_memory_mb'] = np.mean(memory_usages) / (1024 * 1024)
                cache_report['max_memory_mb'] = max(memory_usages) / (1024 * 1024)
            
            report[cache_name] = cache_report
        
        return report
    
    def optimize_cache_size(self, cache_name: str, target_hit_rate: float = 0.8) -> Dict[str, Any]:
        """Suggest optimal cache size based on access patterns"""
        metrics = self.metrics.get(cache_name, [])
        if not metrics:
            return {}
        
        # Analyze access patterns
        unique_keys = set()
        sliding_window_hits = []
        window_size = 100
        
        for i, metric in enumerate(metrics):
            if i >= window_size:
                # Calculate hit rate in sliding window
                window_metrics = metrics[i-window_size:i]
                window_hits = sum(1 for m in window_metrics if m['hit'])
                sliding_window_hits.append(window_hits / window_size)
        
        # Estimate working set size
        working_set_estimates = []
        for window_start in range(0, len(metrics) - window_size, window_size):
            window_metrics = metrics[window_start:window_start + window_size]
            # Estimate unique accesses in window (simplified)
            working_set_estimates.append(len(set(range(window_start, window_start + window_size))))
        
        avg_working_set = np.mean(working_set_estimates) if working_set_estimates else 0
        
        return {
            'current_hit_rate': np.mean(sliding_window_hits) if sliding_window_hits else 0,
            'target_hit_rate': target_hit_rate,
            'estimated_working_set': avg_working_set,
            'recommended_cache_size': int(avg_working_set * 1.5),  # 50% buffer
            'performance_trend': 'improving' if len(sliding_window_hits) > 1 and 
                               sliding_window_hits[-1] > sliding_window_hits[0] else 'stable'
        }

# Test performance monitoring
print("Testing performance monitoring:")

monitor = CachePerformanceMonitor()

# Simulate cache accesses with monitoring
class MonitoredCachedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, cache_size: int = 100):
        self.base_dataset = base_dataset
        self.cache = LRUCache(cache_size)
        self.monitor = monitor
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        start_time = time.time()
        
        # Try cache
        cached_item = self.cache.get(idx)
        hit = cached_item is not None
        
        if not hit:
            cached_item = self.base_dataset[idx]
            self.cache.put(idx, cached_item)
        
        latency = time.time() - start_time
        
        # Record metrics
        self.monitor.record_access('lru_cache', hit, latency)
        
        return cached_item

monitored_dataset = MonitoredCachedDataset(base_dataset, cache_size=50)

# Generate access pattern for monitoring
for i in range(300):
    _ = monitored_dataset[i % 100]  # Some repeated accesses

# Generate performance report
performance_report = monitor.get_performance_report()
optimization_suggestions = monitor.optimize_cache_size('lru_cache')

print("Performance report:")
for cache_name, stats in performance_report.items():
    print(f"  {cache_name}:")
    print(f"    Hit rate: {stats['hit_rate']:.2%}")
    print(f"    Average latency: {stats['avg_latency_ms']:.2f} ms")
    print(f"    P95 latency: {stats['p95_latency_ms']:.2f} ms")

print("\nOptimization suggestions:")
for key, value in optimization_suggestions.items():
    print(f"  {key}: {value}")

print("\n=== Caching Best Practices ===")

print("Cache Design Principles:")
print("1. Choose appropriate cache size based on working set")
print("2. Use multi-level caching for diverse access patterns")
print("3. Monitor hit rates and adjust cache policies")
print("4. Consider memory constraints in cache sizing")
print("5. Implement proper cache invalidation strategies")

print("\nPerformance Optimization:")
print("1. Profile cache access patterns regularly")
print("2. Use memory-aware caching for dynamic environments")
print("3. Consider disk caching for large datasets")
print("4. Implement efficient serialization for disk caches")
print("5. Monitor system resources during caching")

print("\nCommon Pitfalls:")
print("1. Cache sizes that don't match working set")
print("2. Ignoring memory pressure on system")
print("3. Poor cache key design leading to low hit rates")
print("4. Not considering cache coherence in multi-process scenarios")
print("5. Over-caching small, fast-to-compute items")

print("\nTroubleshooting Guide:")
print("1. Low hit rates: Analyze access patterns, increase cache size")
print("2. High memory usage: Implement memory-aware policies")
print("3. Slow cache operations: Optimize serialization, use faster storage")
print("4. Cache inconsistency: Implement proper invalidation")
print("5. Poor performance: Profile and benchmark different strategies")

print("\n=== Data Caching Strategies Complete ===")

# Cleanup temporary directories
import shutil
for temp_dir in [temp_cache_dir, temp_l2_dir]:
    try:
        shutil.rmtree(temp_dir)
    except:
        pass

# Memory cleanup
torch.cuda.empty_cache() if torch.cuda.is_available() else None