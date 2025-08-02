import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import os
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from pathlib import Path
from contextlib import contextmanager
import threading
import psutil
import gc

# Note: PyTorch Profiler operations require torch >= 1.8.0
# Some features require additional packages

try:
    from torch.profiler import profile, record_function, ProfilerActivity
    from torch.profiler import schedule as profiler_schedule
    from torch.profiler import tensorboard_trace_handler
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False
    print("Warning: PyTorch Profiler not available. Upgrade PyTorch to >= 1.8.0")

# Advanced PyTorch Profiling
class PyTorchProfiler:
    """Advanced profiling utilities for PyTorch applications"""
    
    def __init__(self, log_dir: str = "./profiling_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.profiler = None
        
        # Performance tracking
        self.performance_data = {}
        self.memory_snapshots = []
        self.timing_data = {}
    
    @contextmanager
    def profile_context(self, 
                       activities: List[str] = ['cpu', 'cuda'],
                       record_shapes: bool = True,
                       profile_memory: bool = True,
                       with_stack: bool = True,
                       with_flops: bool = True,
                       use_tensorboard: bool = True):
        """Context manager for profiling with comprehensive options"""
        
        if not PROFILER_AVAILABLE:
            print("Profiler not available - using timing fallback")
            yield None
            return
        
        # Convert activity names to ProfilerActivity
        profiler_activities = []
        if 'cpu' in activities:
            profiler_activities.append(ProfilerActivity.CPU)
        if 'cuda' in activities and torch.cuda.is_available():
            profiler_activities.append(ProfilerActivity.CUDA)
        
        # Setup profiler
        if use_tensorboard:
            trace_handler = tensorboard_trace_handler(str(self.log_dir))
        else:
            trace_handler = None
        
        profiler = profile(
            activities=profiler_activities,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
            on_trace_ready=trace_handler
        )
        
        try:
            profiler.start()
            yield profiler
        finally:
            profiler.stop()
            self.profiler = profiler
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
        """Profile a single function call"""
        
        if not PROFILER_AVAILABLE:
            # Fallback timing
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            return result, {"execution_time": end_time - start_time}
        
        with self.profile_context() as profiler:
            with record_function("function_execution"):
                result = func(*args, **kwargs)
        
        # Extract timing information
        timing_info = self._extract_timing_info(profiler)
        
        return result, timing_info
    
    def profile_model_forward(self, model: nn.Module, input_tensor: torch.Tensor,
                             num_runs: int = 100, warmup_runs: int = 10) -> Dict[str, Any]:
        """Profile model forward pass with detailed analysis"""
        
        model.eval()
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_tensor)
        
        # Synchronize CUDA if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Profile forward pass
        forward_times = []
        memory_usage = []
        
        with self.profile_context(profile_memory=True) as profiler:
            for i in range(num_runs):
                with record_function(f"forward_pass_{i}"):
                    start_time = time.time()
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    forward_times.append(end_time - start_time)
                    
                    # Memory usage
                    if torch.cuda.is_available():
                        memory_usage.append(torch.cuda.memory_allocated())
        
        # Calculate statistics
        forward_times = np.array(forward_times)
        
        profile_results = {
            "mean_time": float(np.mean(forward_times)),
            "std_time": float(np.std(forward_times)),
            "min_time": float(np.min(forward_times)),
            "max_time": float(np.max(forward_times)),
            "median_time": float(np.median(forward_times)),
            "num_runs": num_runs,
            "input_shape": list(input_tensor.shape),
            "output_shape": list(output.shape) if output is not None else None,
            "device": str(input_tensor.device)
        }
        
        if memory_usage:
            profile_results.update({
                "memory_usage_bytes": memory_usage,
                "peak_memory_bytes": max(memory_usage),
                "avg_memory_bytes": np.mean(memory_usage)
            })
        
        # Model complexity analysis
        profile_results.update(self._analyze_model_complexity(model, input_tensor))
        
        return profile_results
    
    def profile_training_step(self, model: nn.Module, data_loader: DataLoader,
                             optimizer: torch.optim.Optimizer, criterion: nn.Module,
                             num_steps: int = 10) -> Dict[str, Any]:
        """Profile training steps with detailed breakdown"""
        
        model.train()
        
        step_times = []
        forward_times = []
        backward_times = []
        optimizer_times = []
        memory_snapshots = []
        
        with self.profile_context() as profiler:
            data_iter = iter(data_loader)
            
            for step in range(num_steps):
                try:
                    data, target = next(data_iter)
                except StopIteration:
                    break
                
                with record_function(f"training_step_{step}"):
                    step_start = time.time()
                    
                    # Forward pass
                    with record_function("forward"):
                        forward_start = time.time()
                        output = model(data)
                        loss = criterion(output, target)
                        forward_end = time.time()
                        forward_times.append(forward_end - forward_start)
                    
                    # Backward pass
                    with record_function("backward"):
                        backward_start = time.time()
                        optimizer.zero_grad()
                        loss.backward()
                        backward_end = time.time()
                        backward_times.append(backward_end - backward_start)
                    
                    # Optimizer step
                    with record_function("optimizer"):
                        opt_start = time.time()
                        optimizer.step()
                        opt_end = time.time()
                        optimizer_times.append(opt_end - opt_start)
                    
                    step_end = time.time()
                    step_times.append(step_end - step_start)
                    
                    # Memory snapshot
                    if torch.cuda.is_available():
                        memory_snapshots.append({
                            "step": step,
                            "allocated": torch.cuda.memory_allocated(),
                            "cached": torch.cuda.memory_reserved()
                        })
        
        return {
            "step_times": step_times,
            "forward_times": forward_times,
            "backward_times": backward_times,
            "optimizer_times": optimizer_times,
            "memory_snapshots": memory_snapshots,
            "mean_step_time": np.mean(step_times),
            "mean_forward_time": np.mean(forward_times),
            "mean_backward_time": np.mean(backward_times),
            "mean_optimizer_time": np.mean(optimizer_times)
        }
    
    def _analyze_model_complexity(self, model: nn.Module, 
                                 input_tensor: torch.Tensor) -> Dict[str, Any]:
        """Analyze model complexity"""
        
        complexity_info = {}
        
        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        complexity_info.update({
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
            "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
        })
        
        # Estimate FLOPs (simplified)
        try:
            flops = self._estimate_flops(model, input_tensor)
            complexity_info["estimated_flops"] = flops
        except Exception:
            complexity_info["estimated_flops"] = "unavailable"
        
        return complexity_info
    
    def _estimate_flops(self, model: nn.Module, input_tensor: torch.Tensor) -> int:
        """Estimate FLOPs for the model (simplified implementation)"""
        
        flops = 0
        
        def flop_hook(module, input, output):
            nonlocal flops
            
            if isinstance(module, nn.Linear):
                flops += module.in_features * module.out_features
            elif isinstance(module, nn.Conv2d):
                output_dims = output.shape[2:]
                kernel_dims = module.kernel_size
                in_channels = module.in_channels
                out_channels = module.out_channels
                groups = module.groups
                
                filters_per_channel = out_channels // groups
                conv_per_position_flops = int(np.prod(kernel_dims)) * in_channels // groups
                active_elements_count = int(np.prod(output_dims))
                flops += conv_per_position_flops * active_elements_count * filters_per_channel
        
        # Register hooks
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hooks.append(module.register_forward_hook(flop_hook))
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return flops
    
    def _extract_timing_info(self, profiler) -> Dict[str, float]:
        """Extract timing information from profiler"""
        
        if not profiler:
            return {}
        
        timing_info = {}
        
        try:
            # Get CPU events
            events = profiler.events()
            cpu_events = [e for e in events if e.device_type == torch.autograd.DeviceType.CPU]
            
            if cpu_events:
                total_time = sum(e.cpu_time_total for e in cpu_events)
                timing_info["total_cpu_time"] = total_time / 1000.0  # Convert to ms
            
            # Get CUDA events if available
            if torch.cuda.is_available():
                cuda_events = [e for e in events if e.device_type == torch.autograd.DeviceType.CUDA]
                if cuda_events:
                    total_cuda_time = sum(e.cuda_time_total for e in cuda_events)
                    timing_info["total_cuda_time"] = total_cuda_time / 1000.0
        
        except Exception as e:
            print(f"Warning: Could not extract timing info: {e}")
        
        return timing_info
    
    def generate_profiling_report(self, save_path: Optional[str] = None) -> str:
        """Generate comprehensive profiling report"""
        
        if not self.profiler:
            return "No profiling data available"
        
        report = []
        report.append("PyTorch Profiling Report")
        report.append("=" * 30)
        report.append("")
        
        try:
            # Key averages
            key_averages = self.profiler.key_averages()
            
            report.append("Top Operations by CPU Time:")
            report.append("-" * 30)
            
            # Sort by CPU time
            cpu_sorted = sorted(key_averages, key=lambda x: x.cpu_time_total, reverse=True)
            
            for i, event in enumerate(cpu_sorted[:10]):  # Top 10
                report.append(f"{i+1:2d}. {event.key:30s} {event.cpu_time_total/1000:.2f}ms")
            
            report.append("")
            
            if torch.cuda.is_available():
                report.append("Top Operations by CUDA Time:")
                report.append("-" * 30)
                
                # Sort by CUDA time
                cuda_sorted = sorted(key_averages, key=lambda x: x.cuda_time_total, reverse=True)
                
                for i, event in enumerate(cuda_sorted[:10]):  # Top 10
                    if event.cuda_time_total > 0:
                        report.append(f"{i+1:2d}. {event.key:30s} {event.cuda_time_total/1000:.2f}ms")
            
            # Memory usage
            report.append("")
            report.append("Memory Usage:")
            report.append("-" * 15)
            
            if torch.cuda.is_available():
                report.append(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
                report.append(f"Current CUDA memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        except Exception as e:
            report.append(f"Error generating detailed report: {e}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"✓ Profiling report saved to {save_path}")
        
        return report_text

class MemoryProfiler:
    """Advanced memory profiling utilities"""
    
    def __init__(self):
        self.snapshots = []
        self.peak_memory = 0
    
    @contextmanager
    def memory_snapshot(self, name: str):
        """Context manager for memory profiling"""
        
        # Initial memory
        initial_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            # Final memory
            final_memory = self._get_memory_usage()
            
            snapshot = {
                "name": name,
                "initial_memory": initial_memory,
                "final_memory": final_memory,
                "memory_delta": final_memory["total"] - initial_memory["total"],
                "timestamp": time.time()
            }
            
            self.snapshots.append(snapshot)
            
            if final_memory["total"] > self.peak_memory:
                self.peak_memory = final_memory["total"]
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        
        memory_info = {
            "cpu_memory_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "total": 0
        }
        
        if torch.cuda.is_available():
            memory_info.update({
                "cuda_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "cuda_cached_mb": torch.cuda.memory_reserved() / 1024 / 1024
            })
            memory_info["total"] = memory_info["cpu_memory_mb"] + memory_info["cuda_allocated_mb"]
        else:
            memory_info["total"] = memory_info["cpu_memory_mb"]
        
        return memory_info
    
    def profile_memory_usage(self, model: nn.Module, input_tensor: torch.Tensor,
                           batch_sizes: List[int] = [1, 8, 16, 32, 64]) -> Dict[str, Any]:
        """Profile memory usage across different batch sizes"""
        
        memory_profile = {}
        
        model.eval()
        
        for batch_size in batch_sizes:
            # Create batch
            if len(input_tensor.shape) == 1:
                batch_input = input_tensor.unsqueeze(0).repeat(batch_size, 1)
            else:
                batch_input = input_tensor.unsqueeze(0).repeat(batch_size, *[1] * (len(input_tensor.shape)))
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            gc.collect()
            
            with self.memory_snapshot(f"batch_size_{batch_size}"):
                with torch.no_grad():
                    output = model(batch_input)
            
            # Get latest snapshot
            latest_snapshot = self.snapshots[-1]
            memory_profile[batch_size] = {
                "input_shape": list(batch_input.shape),
                "output_shape": list(output.shape),
                "memory_usage_mb": latest_snapshot["final_memory"]["total"],
                "memory_delta_mb": latest_snapshot["memory_delta"]
            }
        
        return memory_profile
    
    def analyze_memory_leaks(self, operations: List[Callable], 
                           num_iterations: int = 100) -> Dict[str, Any]:
        """Analyze potential memory leaks"""
        
        leak_analysis = {}
        
        for i, operation in enumerate(operations):
            operation_name = getattr(operation, '__name__', f'operation_{i}')
            
            initial_memory = self._get_memory_usage()["total"]
            memory_trend = []
            
            for iteration in range(num_iterations):
                operation()
                
                if iteration % 10 == 0:  # Sample every 10 iterations
                    current_memory = self._get_memory_usage()["total"]
                    memory_trend.append(current_memory - initial_memory)
                
                # Force garbage collection
                if iteration % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Analyze trend
            if len(memory_trend) > 1:
                memory_slope = np.polyfit(range(len(memory_trend)), memory_trend, 1)[0]
                
                leak_analysis[operation_name] = {
                    "memory_trend": memory_trend,
                    "memory_slope_mb_per_iteration": memory_slope * 10,  # Adjust for sampling
                    "potential_leak": memory_slope > 0.1,  # Threshold for leak detection
                    "final_memory_delta": memory_trend[-1]
                }
        
        return leak_analysis
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of memory profiling"""
        
        if not self.snapshots:
            return {"error": "No memory snapshots available"}
        
        total_allocations = sum(s["memory_delta"] for s in self.snapshots if s["memory_delta"] > 0)
        total_deallocations = sum(abs(s["memory_delta"]) for s in self.snapshots if s["memory_delta"] < 0)
        
        return {
            "total_snapshots": len(self.snapshots),
            "peak_memory_mb": self.peak_memory,
            "total_allocations_mb": total_allocations,
            "total_deallocations_mb": total_deallocations,
            "net_memory_change_mb": total_allocations - total_deallocations,
            "snapshots": self.snapshots
        }

class PerformanceBenchmark:
    """Comprehensive performance benchmarking"""
    
    def __init__(self):
        self.benchmark_results = {}
    
    def benchmark_operations(self, operations: Dict[str, Callable],
                           input_data: torch.Tensor, num_runs: int = 1000) -> Dict[str, Any]:
        """Benchmark different operations"""
        
        results = {}
        
        for name, operation in operations.items():
            times = []
            
            # Warmup
            for _ in range(10):
                _ = operation(input_data)
            
            # Synchronize if CUDA
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Benchmark
            for _ in range(num_runs):
                start_time = time.time()
                result = operation(input_data)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
            
            times = np.array(times)
            
            results[name] = {
                "mean_time": float(np.mean(times)),
                "std_time": float(np.std(times)),
                "min_time": float(np.min(times)),
                "max_time": float(np.max(times)),
                "median_time": float(np.median(times)),
                "throughput_ops_per_sec": 1.0 / np.mean(times),
                "num_runs": num_runs
            }
        
        return results
    
    def benchmark_model_sizes(self, model_factory: Callable, 
                             sizes: List[int], input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Benchmark models of different sizes"""
        
        size_results = {}
        
        for size in sizes:
            model = model_factory(size)
            input_tensor = torch.randn(*input_shape)
            
            # Profile forward pass
            profiler = PyTorchProfiler()
            forward_results = profiler.profile_model_forward(model, input_tensor, num_runs=50)
            
            size_results[size] = {
                "model_size": size,
                "parameters": sum(p.numel() for p in model.parameters()),
                "forward_time": forward_results["mean_time"],
                "memory_usage": forward_results.get("avg_memory_bytes", 0)
            }
        
        return size_results
    
    def compare_devices(self, model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """Compare performance across different devices"""
        
        device_results = {}
        
        # CPU
        cpu_model = model.cpu()
        cpu_input = input_tensor.cpu()
        
        profiler = PyTorchProfiler()
        cpu_results = profiler.profile_model_forward(cpu_model, cpu_input, num_runs=50)
        device_results["cpu"] = cpu_results
        
        # CUDA (if available)
        if torch.cuda.is_available():
            cuda_model = model.cuda()
            cuda_input = input_tensor.cuda()
            
            cuda_results = profiler.profile_model_forward(cuda_model, cuda_input, num_runs=50)
            device_results["cuda"] = cuda_results
            
            # Calculate speedup
            if cpu_results["mean_time"] > 0:
                device_results["cuda_speedup"] = cpu_results["mean_time"] / cuda_results["mean_time"]
        
        return device_results
    
    def profile_data_loading(self, data_loader: DataLoader, 
                           num_batches: int = 100) -> Dict[str, Any]:
        """Profile data loading performance"""
        
        loading_times = []
        batch_sizes = []
        
        start_time = time.time()
        
        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break
            
            batch_start = time.time()
            
            # Simulate processing
            if isinstance(batch, (list, tuple)):
                data = batch[0]
            else:
                data = batch
            
            batch_end = time.time()
            
            loading_times.append(batch_end - batch_start)
            batch_sizes.append(len(data))
        
        end_time = time.time()
        
        loading_times = np.array(loading_times)
        
        return {
            "total_time": end_time - start_time,
            "mean_batch_time": float(np.mean(loading_times)),
            "std_batch_time": float(np.std(loading_times)),
            "throughput_batches_per_sec": len(loading_times) / (end_time - start_time),
            "throughput_samples_per_sec": np.sum(batch_sizes) / (end_time - start_time),
            "mean_batch_size": float(np.mean(batch_sizes)),
            "num_batches_profiled": len(loading_times)
        }

# Demo Models and Data
class ProfilerDemoModel(nn.Module):
    """Demo model for profiling"""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

class ProfilerDemoDataset(Dataset):
    """Demo dataset for profiling"""
    
    def __init__(self, size: int, input_dim: int, num_classes: int):
        self.data = torch.randn(size, input_dim)
        self.labels = torch.randint(0, num_classes, (size,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

if __name__ == "__main__":
    print("Advanced PyTorch Profiling")
    print("=" * 30)
    
    if not PROFILER_AVAILABLE:
        print("PyTorch Profiler not available. Upgrade PyTorch to >= 1.8.0")
        print("Showing fallback profiling techniques...")
    
    print("\n1. Basic Profiling Setup")
    print("-" * 26)
    
    # Initialize profiler
    profiler = PyTorchProfiler()
    
    # Create demo model and data
    model = ProfilerDemoModel(input_size=784, hidden_size=256, num_classes=10)
    input_tensor = torch.randn(32, 784)
    
    print(f"✓ Demo model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    print("\n2. Model Forward Pass Profiling")
    print("-" * 34)
    
    # Profile forward pass
    forward_results = profiler.profile_model_forward(model, input_tensor, num_runs=100)
    
    print("Forward Pass Profile Results:")
    print(f"  Mean time: {forward_results['mean_time']*1000:.2f} ms")
    print(f"  Std time: {forward_results['std_time']*1000:.2f} ms")
    print(f"  Min time: {forward_results['min_time']*1000:.2f} ms")
    print(f"  Max time: {forward_results['max_time']*1000:.2f} ms")
    print(f"  Total parameters: {forward_results['total_parameters']:,}")
    print(f"  Model size: {forward_results['model_size_mb']:.2f} MB")
    
    if 'estimated_flops' in forward_results:
        flops = forward_results['estimated_flops']
        if isinstance(flops, int):
            print(f"  Estimated FLOPs: {flops:,}")
    
    print("\n3. Training Step Profiling")
    print("-" * 28)
    
    # Create training components
    dataset = ProfilerDemoDataset(1000, 784, 10)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Profile training steps
    training_results = profiler.profile_training_step(
        model, data_loader, optimizer, criterion, num_steps=5
    )
    
    print("Training Step Profile Results:")
    print(f"  Mean step time: {training_results['mean_step_time']*1000:.2f} ms")
    print(f"  Mean forward time: {training_results['mean_forward_time']*1000:.2f} ms")
    print(f"  Mean backward time: {training_results['mean_backward_time']*1000:.2f} ms")
    print(f"  Mean optimizer time: {training_results['mean_optimizer_time']*1000:.2f} ms")
    
    print("\n4. Memory Profiling")
    print("-" * 20)
    
    # Initialize memory profiler
    memory_profiler = MemoryProfiler()
    
    # Profile memory usage across batch sizes
    memory_results = memory_profiler.profile_memory_usage(
        model, torch.randn(784), batch_sizes=[1, 8, 16, 32]
    )
    
    print("Memory Usage by Batch Size:")
    for batch_size, results in memory_results.items():
        print(f"  Batch {batch_size:2d}: {results['memory_usage_mb']:.2f} MB "
              f"(Δ {results['memory_delta_mb']:.2f} MB)")
    
    # Memory leak analysis
    def dummy_operation():
        x = torch.randn(100, 100)
        y = torch.randn(100, 100)
        z = torch.mm(x, y)
        return z
    
    leak_analysis = memory_profiler.analyze_memory_leaks([dummy_operation], num_iterations=50)
    
    for op_name, analysis in leak_analysis.items():
        print(f"\nMemory Leak Analysis - {op_name}:")
        print(f"  Potential leak: {analysis['potential_leak']}")
        print(f"  Memory slope: {analysis['memory_slope_mb_per_iteration']:.4f} MB/iter")
    
    print("\n5. Performance Benchmarking")
    print("-" * 30)
    
    # Initialize benchmark
    benchmark = PerformanceBenchmark()
    
    # Benchmark different operations
    operations = {
        "relu": lambda x: F.relu(x),
        "gelu": lambda x: F.gelu(x),
        "tanh": lambda x: torch.tanh(x),
        "sigmoid": lambda x: torch.sigmoid(x)
    }
    
    op_results = benchmark.benchmark_operations(operations, torch.randn(1000, 1000))
    
    print("Operation Benchmarks (1000x1000 tensor):")
    for op_name, results in op_results.items():
        print(f"  {op_name:8s}: {results['mean_time']*1000:.3f} ms "
              f"({results['throughput_ops_per_sec']:.1f} ops/sec)")
    
    # Device comparison
    if torch.cuda.is_available():
        device_results = benchmark.compare_devices(model, input_tensor)
        
        print("\nDevice Comparison:")
        for device, results in device_results.items():
            if device != "cuda_speedup":
                print(f"  {device.upper():4s}: {results['mean_time']*1000:.2f} ms")
        
        if "cuda_speedup" in device_results:
            print(f"  CUDA Speedup: {device_results['cuda_speedup']:.2f}x")
    
    # Data loading benchmark
    data_loading_results = benchmark.profile_data_loading(data_loader, num_batches=50)
    
    print(f"\nData Loading Performance:")
    print(f"  Throughput: {data_loading_results['throughput_samples_per_sec']:.1f} samples/sec")
    print(f"  Mean batch time: {data_loading_results['mean_batch_time']*1000:.2f} ms")
    
    print("\n6. Profiling Best Practices")
    print("-" * 31)
    
    best_practices = [
        "Use warmup runs before actual profiling",
        "Profile with representative data and batch sizes",
        "Include both CPU and CUDA profiling when available",
        "Profile memory usage to identify bottlenecks",
        "Use record_function for custom profiling regions",
        "Analyze both forward and backward pass separately",
        "Profile data loading pipeline independently",
        "Monitor for memory leaks in long-running processes",
        "Compare performance across different devices",
        "Profile with different precision modes (fp16, fp32)",
        "Use TensorBoard integration for visualization",
        "Profile inference and training separately"
    ]
    
    print("PyTorch Profiling Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\n7. Profiling Tools Comparison")
    print("-" * 33)
    
    profiling_tools = {
        "PyTorch Profiler": "Built-in, comprehensive, TensorBoard integration",
        "cProfile": "Python standard library, function-level profiling",
        "line_profiler": "Line-by-line profiling, detailed analysis",
        "memory_profiler": "Memory usage tracking and visualization",
        "py-spy": "Sampling profiler, minimal overhead",
        "NVIDIA Nsight": "CUDA kernel profiling, GPU analysis",
        "Intel VTune": "CPU optimization and vectorization analysis",
        "perf": "Linux system profiler, hardware counters"
    }
    
    print("Profiling Tools Comparison:")
    for tool, description in profiling_tools.items():
        print(f"  {tool}: {description}")
    
    print("\n8. Performance Optimization Workflow")
    print("-" * 38)
    
    optimization_workflow = [
        "1. Profile to identify bottlenecks",
        "2. Analyze hotspots and memory usage",
        "3. Optimize data loading pipeline",
        "4. Optimize model architecture",
        "5. Use mixed precision training",
        "6. Apply model compilation (TorchScript/TorchDynamo)",
        "7. Optimize memory allocation patterns",
        "8. Use efficient CUDA operations",
        "9. Profile again to measure improvements",
        "10. Repeat until performance targets are met"
    ]
    
    print("Performance Optimization Workflow:")
    for step in optimization_workflow:
        print(f"  {step}")
    
    print("\n9. Common Performance Issues")
    print("-" * 33)
    
    common_issues = {
        "Small batch sizes": "Underutilizes GPU parallelism",
        "Data loading bottleneck": "CPU-bound data preprocessing",
        "Memory allocation": "Frequent malloc/free operations",
        "Device transfers": "Unnecessary CPU-GPU data movement",
        "Gradient accumulation": "Large memory usage in backward pass",
        "Model compilation": "Python overhead in model execution",
        "Inefficient operations": "Using loops instead of vectorized ops",
        "Memory fragmentation": "Poor memory allocation patterns"
    }
    
    print("Common Performance Issues:")
    for issue, description in common_issues.items():
        print(f"  {issue}: {description}")
    
    # Generate profiling report
    print("\n10. Profiling Report Generation")
    print("-" * 35)
    
    if PROFILER_AVAILABLE and profiler.profiler:
        report = profiler.generate_profiling_report("./profiling_report.txt")
        print("Profiling report preview:")
        print(report[:500] + "..." if len(report) > 500 else report)
    else:
        print("Profiling report generation requires PyTorch Profiler")
    
    print("\nAdvanced PyTorch profiling demonstration completed!")
    print("Key components covered:")
    print("  - Model forward pass and training step profiling")
    print("  - Memory usage analysis and leak detection")
    print("  - Performance benchmarking across operations and devices")
    print("  - Data loading pipeline profiling")
    print("  - Best practices and optimization workflows")
    
    print("\nProfiling enables:")
    print("  - Identification of performance bottlenecks")
    print("  - Memory usage optimization")
    print("  - Data-driven performance improvements")
    print("  - Systematic performance analysis")
    print("  - Production readiness assessment")
    
    if PROFILER_AVAILABLE:
        print(f"\n🔍 View detailed profiling results with:")
        print(f"   tensorboard --logdir={profiler.log_dir}")
        print(f"   Then navigate to the PyTorch Profiler tab")