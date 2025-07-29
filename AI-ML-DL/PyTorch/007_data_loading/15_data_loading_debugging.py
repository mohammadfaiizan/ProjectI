"""
PyTorch Data Loading Debugging - Troubleshooting Data Pipeline Issues
Comprehensive guide to debugging and solving data loading problems in PyTorch
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import traceback
import sys
import os
from typing import Any, Dict, List, Optional
import psutil
import threading
import multiprocessing as mp

print("=== DATA LOADING DEBUGGING ===")

# 1. COMMON DATA LOADING ISSUES
print("\n1. COMMON DATA LOADING ISSUES")

class ProblematicDataset(Dataset):
    """Dataset that demonstrates common issues"""
    
    def __init__(self, size=100, error_type="none"):
        self.size = size
        self.error_type = error_type
        self.data = torch.randn(size, 10)
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Simulate various issues
        if self.error_type == "index_error" and idx >= self.size // 2:
            raise IndexError(f"Index {idx} out of range")
        elif self.error_type == "memory_leak":
            # Simulate memory leak
            large_tensor = torch.randn(1000, 1000)
            return self.data[idx], large_tensor.sum()
        elif self.error_type == "slow_loading":
            # Simulate slow data loading
            time.sleep(0.1)
            return self.data[idx], torch.tensor(0)
        elif self.error_type == "random_failure":
            # Random failures
            if np.random.rand() < 0.1:
                raise RuntimeError("Random failure occurred")
            return self.data[idx], torch.tensor(0)
        elif self.error_type == "wrong_type":
            # Return wrong data type
            return self.data[idx].numpy(), "wrong_label"
        else:
            return self.data[idx], torch.tensor(0)

# Test different error types
error_types = ["none", "index_error", "slow_loading", "random_failure", "wrong_type"]
for error_type in error_types[:3]:  # Test first 3
    print(f"Testing {error_type}...")
    try:
        dataset = ProblematicDataset(50, error_type)
        loader = DataLoader(dataset, batch_size=8, num_workers=0)
        batch = next(iter(loader))
        print(f"✓ {error_type}: Success")
    except Exception as e:
        print(f"✗ {error_type}: {type(e).__name__}: {str(e)}")

# 2. DEBUGGING TECHNIQUES
print("\n2. DEBUGGING TECHNIQUES")

class DebugDataset(Dataset):
    """Dataset with extensive debugging capabilities"""
    
    def __init__(self, size=100, debug=True):
        self.size = size
        self.debug = debug
        self.access_count = 0
        self.error_count = 0
        self.access_times = []
        
    def __len__(self):
        if self.debug:
            print(f"Dataset length requested: {self.size}")
        return self.size
        
    def __getitem__(self, idx):
        start_time = time.time()
        self.access_count += 1
        
        try:
            if self.debug and self.access_count % 10 == 0:
                print(f"Accessing item {idx}, total accesses: {self.access_count}")
                
            # Simulate data loading
            data = torch.randn(10)
            label = torch.randint(0, 5, (1,)).item()
            
            end_time = time.time()
            self.access_times.append(end_time - start_time)
            
            return data, label
            
        except Exception as e:
            self.error_count += 1
            if self.debug:
                print(f"Error accessing item {idx}: {e}")
            raise
            
    def get_stats(self):
        """Get debugging statistics"""
        return {
            'total_accesses': self.access_count,
            'error_count': self.error_count,
            'avg_access_time': np.mean(self.access_times) if self.access_times else 0,
            'max_access_time': np.max(self.access_times) if self.access_times else 0
        }

debug_dataset = DebugDataset(50, debug=True)
debug_loader = DataLoader(debug_dataset, batch_size=8, num_workers=0)

# Test with debug info
batch = next(iter(debug_loader))
print(f"Debug stats: {debug_dataset.get_stats()}")

# 3. MEMORY DEBUGGING
print("\n3. MEMORY DEBUGGING")

def monitor_memory():
    """Monitor memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

class MemoryMonitoredDataset(Dataset):
    """Dataset that monitors memory usage"""
    
    def __init__(self, size=100):
        self.size = size
        self.memory_usage = []
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        memory_before = monitor_memory()
        
        # Simulate data loading
        data = torch.randn(100, 100)  # Larger tensor
        label = torch.tensor(idx % 5)
        
        memory_after = monitor_memory()
        self.memory_usage.append(memory_after - memory_before)
        
        return data, label
        
    def get_memory_stats(self):
        """Get memory usage statistics"""
        if not self.memory_usage:
            return {}
        return {
            'avg_memory_delta': np.mean(self.memory_usage),
            'max_memory_delta': np.max(self.memory_usage),
            'total_memory_used': sum(self.memory_usage)
        }

memory_dataset = MemoryMonitoredDataset(20)
memory_loader = DataLoader(memory_dataset, batch_size=4, num_workers=0)

initial_memory = monitor_memory()
for i, batch in enumerate(memory_loader):
    if i >= 2:  # Just a few batches
        break
    current_memory = monitor_memory()
    print(f"Batch {i}: Memory usage = {current_memory:.1f} MB")

print(f"Memory stats: {memory_dataset.get_memory_stats()}")

# 4. MULTIPROCESSING DEBUGGING
print("\n4. MULTIPROCESSING DEBUGGING")

class MultiprocessingDebugDataset(Dataset):
    """Dataset for debugging multiprocessing issues"""
    
    def __init__(self, size=100):
        self.size = size
        self.worker_info = {}
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Get worker information
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            if worker_id not in self.worker_info:
                self.worker_info[worker_id] = {'count': 0, 'pid': os.getpid()}
            self.worker_info[worker_id]['count'] += 1
        
        # Simulate work
        data = torch.randn(10)
        return data, torch.tensor(idx % 3)

def worker_init_fn(worker_id):
    """Initialize worker process"""
    print(f"Worker {worker_id} initialized with PID {os.getpid()}")

mp_dataset = MultiprocessingDebugDataset(40)

# Test with different number of workers
for num_workers in [0, 2]:
    print(f"\nTesting with {num_workers} workers...")
    mp_loader = DataLoader(
        mp_dataset, 
        batch_size=8, 
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None
    )
    
    try:
        for i, batch in enumerate(mp_loader):
            if i >= 2:
                break
        print(f"✓ Multiprocessing with {num_workers} workers successful")
    except Exception as e:
        print(f"✗ Multiprocessing failed: {e}")

# 5. PERFORMANCE DEBUGGING
print("\n5. PERFORMANCE DEBUGGING")

class PerformanceProfiler:
    """Profile data loading performance"""
    
    def __init__(self):
        self.batch_times = []
        self.loading_times = []
        
    def profile_loader(self, loader, num_batches=5):
        """Profile DataLoader performance"""
        total_start = time.time()
        
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
                
            batch_start = time.time()
            # Simulate processing time
            time.sleep(0.001)
            batch_end = time.time()
            
            self.batch_times.append(batch_end - batch_start)
            
        total_end = time.time()
        self.loading_times.append(total_end - total_start)
        
    def get_report(self):
        """Generate performance report"""
        if not self.batch_times:
            return "No profiling data available"
            
        return {
            'avg_batch_time': np.mean(self.batch_times),
            'max_batch_time': np.max(self.batch_times),
            'total_time': sum(self.loading_times),
            'throughput': len(self.batch_times) / sum(self.loading_times)
        }

# Profile different configurations
profiler = PerformanceProfiler()
configurations = [
    {"batch_size": 16, "num_workers": 0},
    {"batch_size": 32, "num_workers": 0},
    {"batch_size": 16, "num_workers": 2}
]

for config in configurations:
    print(f"\nProfiling configuration: {config}")
    test_dataset = DebugDataset(100, debug=False)
    test_loader = DataLoader(test_dataset, **config)
    profiler.profile_loader(test_loader, num_batches=3)
    print(f"Results: {profiler.get_report()}")

# 6. ERROR HANDLING AND RECOVERY
print("\n6. ERROR HANDLING AND RECOVERY")

class RobustDataset(Dataset):
    """Dataset with built-in error handling"""
    
    def __init__(self, size=100, failure_rate=0.1):
        self.size = size
        self.failure_rate = failure_rate
        self.fallback_data = torch.zeros(10)
        self.error_log = []
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        try:
            # Simulate random failures
            if np.random.rand() < self.failure_rate:
                raise RuntimeError(f"Simulated failure for index {idx}")
                
            data = torch.randn(10)
            return data, torch.tensor(idx % 5)
            
        except Exception as e:
            # Log error and return fallback data
            self.error_log.append({'idx': idx, 'error': str(e)})
            print(f"Error at index {idx}, using fallback data")
            return self.fallback_data, torch.tensor(-1)  # Error label
            
    def get_error_summary(self):
        """Get error summary"""
        return {
            'total_errors': len(self.error_log),
            'error_rate': len(self.error_log) / self.size,
            'recent_errors': self.error_log[-5:] if self.error_log else []
        }

robust_dataset = RobustDataset(50, failure_rate=0.2)
robust_loader = DataLoader(robust_dataset, batch_size=8, num_workers=0)

# Test robust loading
for i, batch in enumerate(robust_loader):
    if i >= 3:
        break
    error_labels = (batch[1] == -1).sum().item()
    print(f"Batch {i}: {error_labels} error samples out of {len(batch[1])}")

print(f"Error summary: {robust_dataset.get_error_summary()}")

# 7. DEBUGGING TOOLS AND UTILITIES
print("\n7. DEBUGGING TOOLS AND UTILITIES")

class DataLoaderDebugger:
    """Comprehensive debugging utility for DataLoader"""
    
    def __init__(self, dataset, loader_configs):
        self.dataset = dataset
        self.loader_configs = loader_configs
        self.results = {}
        
    def run_diagnostics(self):
        """Run comprehensive diagnostics"""
        print("Running DataLoader diagnostics...")
        
        for name, config in self.loader_configs.items():
            print(f"\nTesting configuration: {name}")
            try:
                loader = DataLoader(self.dataset, **config)
                result = self._test_loader(loader, name)
                self.results[name] = result
                print(f"✓ {name}: {result['status']}")
            except Exception as e:
                self.results[name] = {'status': 'failed', 'error': str(e)}
                print(f"✗ {name}: Failed - {e}")
                
    def _test_loader(self, loader, name):
        """Test individual loader configuration"""
        start_time = time.time()
        batch_count = 0
        error_count = 0
        
        try:
            for batch in loader:
                batch_count += 1
                if batch_count >= 3:  # Test first 3 batches
                    break
        except Exception as e:
            error_count += 1
            
        end_time = time.time()
        
        return {
            'status': 'success' if error_count == 0 else 'partial',
            'batch_count': batch_count,
            'error_count': error_count,
            'time_taken': end_time - start_time
        }
        
    def generate_report(self):
        """Generate diagnostic report"""
        if not self.results:
            return "No diagnostic results available"
            
        report = "DataLoader Diagnostic Report\n" + "="*40 + "\n"
        
        for name, result in self.results.items():
            report += f"\nConfiguration: {name}\n"
            report += f"Status: {result['status']}\n"
            if 'batch_count' in result:
                report += f"Batches processed: {result['batch_count']}\n"
                report += f"Time taken: {result['time_taken']:.3f}s\n"
            if 'error' in result:
                report += f"Error: {result['error']}\n"
                
        return report

# Run comprehensive diagnostics
test_dataset = DebugDataset(100, debug=False)
configs = {
    'basic': {'batch_size': 16, 'num_workers': 0},
    'multiprocess': {'batch_size': 16, 'num_workers': 2},
    'large_batch': {'batch_size': 64, 'num_workers': 0}
}

debugger = DataLoaderDebugger(test_dataset, configs)
debugger.run_diagnostics()
print(f"\n{debugger.generate_report()}")

# 8. COMMON SOLUTIONS AND BEST PRACTICES
print("\n8. COMMON SOLUTIONS AND BEST PRACTICES")

def debug_checklist():
    """Debugging checklist for data loading issues"""
    checklist = {
        "Dataset Implementation": [
            "✓ __len__ returns correct size",
            "✓ __getitem__ handles all valid indices",
            "✓ Consistent return types",
            "✓ No memory leaks in data loading"
        ],
        "DataLoader Configuration": [
            "✓ Appropriate batch_size for memory",
            "✓ num_workers <= CPU cores",
            "✓ pin_memory for GPU training",
            "✓ drop_last for consistent batches"
        ],
        "Multiprocessing": [
            "✓ worker_init_fn for worker setup",
            "✓ Avoid shared state between workers",
            "✓ Handle worker timeouts",
            "✓ Proper cleanup on exit"
        ],
        "Performance": [
            "✓ Pre-load data when possible",
            "✓ Use appropriate transforms",
            "✓ Monitor memory usage",
            "✓ Profile different configurations"
        ]
    }
    
    print("Data Loading Debugging Checklist:")
    for category, items in checklist.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  {item}")

debug_checklist()

print("\n=== DATA LOADING DEBUGGING COMPLETE ===")
print("Key debugging techniques covered:")
print("- Common data loading issues identification")
print("- Debug dataset implementation")
print("- Memory usage monitoring")
print("- Multiprocessing debugging")
print("- Performance profiling")
print("- Error handling and recovery")
print("- Comprehensive debugging tools")
print("- Best practices checklist")