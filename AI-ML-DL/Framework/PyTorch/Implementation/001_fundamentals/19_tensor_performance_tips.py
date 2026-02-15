#!/usr/bin/env python3
"""PyTorch Tensor Performance Tips and Optimization Techniques"""

import torch
import time
import tracemalloc

print("=== Memory Layout Optimization ===")

# Contiguous vs non-contiguous tensors
def benchmark_contiguous():
    # Create tensors
    tensor = torch.randn(1000, 1000)
    non_contiguous = tensor.t()  # Transpose makes non-contiguous
    
    print(f"Original contiguous: {tensor.is_contiguous()}")
    print(f"Transposed contiguous: {non_contiguous.is_contiguous()}")
    
    # Benchmark operations
    num_iterations = 100
    
    # Contiguous tensor operation
    start_time = time.time()
    for _ in range(num_iterations):
        result = tensor.sum()
    contiguous_time = time.time() - start_time
    
    # Non-contiguous tensor operation
    start_time = time.time()
    for _ in range(num_iterations):
        result = non_contiguous.sum()
    non_contiguous_time = time.time() - start_time
    
    # Make contiguous and benchmark
    made_contiguous = non_contiguous.contiguous()
    start_time = time.time()
    for _ in range(num_iterations):
        result = made_contiguous.sum()
    made_contiguous_time = time.time() - start_time
    
    print(f"Contiguous operation time: {contiguous_time:.6f}s")
    print(f"Non-contiguous operation time: {non_contiguous_time:.6f}s")
    print(f"Made contiguous operation time: {made_contiguous_time:.6f}s")
    print(f"Speedup after making contiguous: {non_contiguous_time / made_contiguous_time:.2f}x")

benchmark_contiguous()

print("\n=== In-Place Operations for Memory Efficiency ===")

def benchmark_inplace_operations():
    # Memory usage comparison
    tracemalloc.start()
    
    # Out-of-place operations
    tensor = torch.randn(1000, 1000)
    snapshot1 = tracemalloc.take_snapshot()
    
    # Multiple out-of-place operations
    result = tensor + 1
    result = result * 2
    result = result / 3
    result = torch.sqrt(result.abs())
    
    snapshot2 = tracemalloc.take_snapshot()
    
    # In-place operations
    tensor_inplace = torch.randn(1000, 1000)
    snapshot3 = tracemalloc.take_snapshot()
    
    tensor_inplace.add_(1)
    tensor_inplace.mul_(2)
    tensor_inplace.div_(3)
    tensor_inplace.abs_().sqrt_()
    
    snapshot4 = tracemalloc.take_snapshot()
    
    # Calculate memory differences
    top_stats2 = snapshot2.compare_to(snapshot1, 'lineno')
    top_stats4 = snapshot4.compare_to(snapshot3, 'lineno')
    
    print("Out-of-place operations memory usage:")
    for stat in top_stats2[:3]:
        print(stat)
    
    print("\nIn-place operations memory usage:")
    for stat in top_stats4[:3]:
        print(stat)
    
    tracemalloc.stop()

# Note: Running this in production code
# benchmark_inplace_operations()
print("In-place operations save memory by modifying tensors directly")

print("\n=== Efficient Broadcasting ===")

def demonstrate_broadcasting_efficiency():
    # Inefficient: explicit expansion
    a = torch.randn(1000, 1)
    b = torch.randn(1, 1000)
    
    # Method 1: Explicit expansion (memory intensive)
    start_time = time.time()
    a_expanded = a.expand(1000, 1000)
    b_expanded = b.expand(1000, 1000)
    result1 = a_expanded + b_expanded
    explicit_time = time.time() - start_time
    
    # Method 2: Let PyTorch handle broadcasting (efficient)
    start_time = time.time()
    result2 = a + b
    broadcast_time = time.time() - start_time
    
    print(f"Explicit expansion time: {explicit_time:.6f}s")
    print(f"Broadcasting time: {broadcast_time:.6f}s")
    print(f"Broadcasting speedup: {explicit_time / broadcast_time:.2f}x")
    print(f"Results equal: {torch.equal(result1, result2)}")

demonstrate_broadcasting_efficiency()

print("\n=== Vectorization vs Loops ===")

def compare_vectorization():
    # Create data
    data = torch.randn(10000)
    threshold = 0.5
    
    # Method 1: Python loop (slow)
    start_time = time.time()
    result_loop = torch.zeros_like(data)
    for i in range(len(data)):
        if data[i] > threshold:
            result_loop[i] = data[i] ** 2
        else:
            result_loop[i] = 0
    loop_time = time.time() - start_time
    
    # Method 2: Vectorized operations (fast)
    start_time = time.time()
    mask = data > threshold
    result_vectorized = torch.where(mask, data ** 2, torch.tensor(0.0))
    vectorized_time = time.time() - start_time
    
    print(f"Python loop time: {loop_time:.6f}s")
    print(f"Vectorized time: {vectorized_time:.6f}s")
    print(f"Vectorization speedup: {loop_time / vectorized_time:.2f}x")
    print(f"Results equal: {torch.allclose(result_loop, result_vectorized)}")

compare_vectorization()

print("\n=== Batch Processing Optimization ===")

def optimize_batch_processing():
    # Simulate processing many small operations vs batch operation
    num_samples = 1000
    sample_size = 100
    
    # Method 1: Process one by one
    samples = [torch.randn(sample_size, sample_size) for _ in range(num_samples)]
    
    start_time = time.time()
    results_individual = []
    for sample in samples[:100]:  # Use subset for timing
        result = torch.mm(sample, sample.t())
        results_individual.append(result)
    individual_time = time.time() - start_time
    
    # Method 2: Batch processing
    batch_samples = torch.stack(samples[:100])
    start_time = time.time()
    batch_result = torch.bmm(batch_samples, batch_samples.transpose(-2, -1))
    batch_time = time.time() - start_time
    
    print(f"Individual processing time: {individual_time:.6f}s")
    print(f"Batch processing time: {batch_time:.6f}s")
    print(f"Batch speedup: {individual_time / batch_time:.2f}x")

optimize_batch_processing()

print("\n=== GPU Optimization ===")

def gpu_optimization_tips():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU optimization")
        return
    
    # Efficient data transfer
    cpu_data = torch.randn(1000, 1000)
    
    # Method 1: Transfer then operate
    start_time = time.time()
    gpu_data1 = cpu_data.cuda()
    result1 = torch.mm(gpu_data1, gpu_data1.t())
    transfer_then_compute_time = time.time() - start_time
    
    # Method 2: Create directly on GPU
    start_time = time.time()
    gpu_data2 = torch.randn(1000, 1000, device='cuda')
    result2 = torch.mm(gpu_data2, gpu_data2.t())
    direct_gpu_time = time.time() - start_time
    
    print(f"Transfer then compute time: {transfer_then_compute_time:.6f}s")
    print(f"Direct GPU creation time: {direct_gpu_time:.6f}s")
    
    # Pinned memory for faster transfers
    pinned_data = torch.randn(1000, 1000).pin_memory()
    start_time = time.time()
    gpu_pinned = pinned_data.cuda(non_blocking=True)
    torch.cuda.synchronize()
    pinned_transfer_time = time.time() - start_time
    
    regular_data = torch.randn(1000, 1000)
    start_time = time.time()
    gpu_regular = regular_data.cuda()
    regular_transfer_time = time.time() - start_time
    
    print(f"Regular transfer time: {regular_transfer_time:.6f}s")
    print(f"Pinned memory transfer time: {pinned_transfer_time:.6f}s")

gpu_optimization_tips()

print("\n=== Memory Pool Management ===")

def memory_management_tips():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory management")
        return
    
    # Monitor memory usage
    print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    
    # Create and delete tensors
    large_tensors = []
    for i in range(10):
        tensor = torch.randn(100, 100, device='cuda')
        large_tensors.append(tensor)
    
    print(f"After creating tensors: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    
    # Delete tensors
    del large_tensors
    print(f"After deleting tensors: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    
    # Clear cache
    torch.cuda.empty_cache()
    print(f"After clearing cache: {torch.cuda.memory_reserved() / 1e6:.2f} MB")

memory_management_tips()

print("\n=== Data Type Optimization ===")

def dtype_optimization():
    # Compare different data types
    size = (1000, 1000)
    
    # Create tensors with different dtypes
    float32_tensor = torch.randn(size, dtype=torch.float32)
    float16_tensor = torch.randn(size, dtype=torch.float16)
    
    # Memory usage
    float32_memory = float32_tensor.numel() * float32_tensor.element_size()
    float16_memory = float16_tensor.numel() * float16_tensor.element_size()
    
    print(f"Float32 memory: {float32_memory / 1e6:.2f} MB")
    print(f"Float16 memory: {float16_memory / 1e6:.2f} MB")
    print(f"Memory savings with float16: {float32_memory / float16_memory:.2f}x")
    
    # Performance comparison
    num_iterations = 100
    
    # Float32 operations
    start_time = time.time()
    for _ in range(num_iterations):
        result = torch.mm(float32_tensor, float32_tensor.t())
    float32_time = time.time() - start_time
    
    # Float16 operations
    start_time = time.time()
    for _ in range(num_iterations):
        result = torch.mm(float16_tensor, float16_tensor.t())
    float16_time = time.time() - start_time
    
    print(f"Float32 operation time: {float32_time:.6f}s")
    print(f"Float16 operation time: {float16_time:.6f}s")
    print(f"Float16 speedup: {float32_time / float16_time:.2f}x")

dtype_optimization()

print("\n=== Efficient Tensor Creation ===")

def efficient_tensor_creation():
    size = (1000, 1000)
    num_iterations = 100
    
    # Method 1: torch.zeros + fill
    start_time = time.time()
    for _ in range(num_iterations):
        tensor = torch.zeros(size)
        tensor.fill_(5.0)
    zeros_fill_time = time.time() - start_time
    
    # Method 2: torch.full
    start_time = time.time()
    for _ in range(num_iterations):
        tensor = torch.full(size, 5.0)
    full_time = time.time() - start_time
    
    # Method 3: torch.empty + fill_ (fastest for large tensors)
    start_time = time.time()
    for _ in range(num_iterations):
        tensor = torch.empty(size)
        tensor.fill_(5.0)
    empty_fill_time = time.time() - start_time
    
    print(f"torch.zeros + fill time: {zeros_fill_time:.6f}s")
    print(f"torch.full time: {full_time:.6f}s")
    print(f"torch.empty + fill_ time: {empty_fill_time:.6f}s")
    print(f"Fastest method speedup: {max(zeros_fill_time, full_time) / empty_fill_time:.2f}x")

efficient_tensor_creation()

print("\n=== Operation Fusion ===")

def operation_fusion():
    # Multiple separate operations vs fused operations
    x = torch.randn(1000, 1000)
    
    # Separate operations
    start_time = time.time()
    y1 = x + 1
    y2 = y1 * 2
    y3 = torch.relu(y2)
    result1 = y3.sum()
    separate_time = time.time() - start_time
    
    # Fused operations
    start_time = time.time()
    result2 = torch.relu((x + 1) * 2).sum()
    fused_time = time.time() - start_time
    
    print(f"Separate operations time: {separate_time:.6f}s")
    print(f"Fused operations time: {fused_time:.6f}s")
    print(f"Fusion speedup: {separate_time / fused_time:.2f}x")
    print(f"Results equal: {torch.allclose(result1, result2)}")

operation_fusion()

print("\n=== Avoiding Python Loops ===")

def avoid_python_loops():
    # Create sample data
    tensor = torch.randn(10000)
    
    # Bad: Python loop
    start_time = time.time()
    result_bad = []
    for i in range(len(tensor)):
        if tensor[i] > 0:
            result_bad.append(tensor[i] ** 2)
        else:
            result_bad.append(0)
    result_bad = torch.tensor(result_bad)
    python_loop_time = time.time() - start_time
    
    # Good: Vectorized operations
    start_time = time.time()
    mask = tensor > 0
    result_good = torch.where(mask, tensor ** 2, torch.tensor(0.0))
    vectorized_time = time.time() - start_time
    
    # Better: Using built-in functions
    start_time = time.time()
    result_better = torch.clamp(tensor, min=0) ** 2
    builtin_time = time.time() - start_time
    
    print(f"Python loop time: {python_loop_time:.6f}s")
    print(f"Vectorized time: {vectorized_time:.6f}s")
    print(f"Built-in function time: {builtin_time:.6f}s")
    print(f"Vectorization speedup: {python_loop_time / vectorized_time:.2f}x")
    print(f"Built-in speedup: {python_loop_time / builtin_time:.2f}x")

avoid_python_loops()

print("\n=== Memory Access Patterns ===")

def memory_access_patterns():
    # Row-major vs column-major access
    matrix = torch.randn(1000, 1000)
    
    # Row-wise access (efficient - follows memory layout)
    start_time = time.time()
    row_sum = 0
    for i in range(matrix.size(0)):
        row_sum += matrix[i].sum()
    row_time = time.time() - start_time
    
    # Column-wise access (less efficient)
    start_time = time.time()
    col_sum = 0
    for j in range(matrix.size(1)):
        col_sum += matrix[:, j].sum()
    col_time = time.time() - start_time
    
    # Vectorized (most efficient)
    start_time = time.time()
    vec_sum = matrix.sum()
    vec_time = time.time() - start_time
    
    print(f"Row-wise access time: {row_time:.6f}s")
    print(f"Column-wise access time: {col_time:.6f}s")
    print(f"Vectorized access time: {vec_time:.6f}s")
    print(f"Vectorized speedup over row-wise: {row_time / vec_time:.2f}x")
    print(f"Vectorized speedup over column-wise: {col_time / vec_time:.2f}x")

memory_access_patterns()

print("\n=== Gradient Computation Optimization ===")

def gradient_optimization():
    # Demonstrate torch.no_grad() for inference
    model_weights = torch.randn(1000, 1000, requires_grad=True)
    input_data = torch.randn(1000, 1000)
    
    # With gradient tracking (slower, more memory)
    start_time = time.time()
    with torch.enable_grad():
        output = torch.mm(model_weights, input_data)
        result = output.sum()
    grad_time = time.time() - start_time
    
    # Without gradient tracking (faster, less memory)
    start_time = time.time()
    with torch.no_grad():
        output = torch.mm(model_weights, input_data)
        result = output.sum()
    no_grad_time = time.time() - start_time
    
    print(f"With gradient tracking time: {grad_time:.6f}s")
    print(f"Without gradient tracking time: {no_grad_time:.6f}s")
    print(f"No grad speedup: {grad_time / no_grad_time:.2f}x")

gradient_optimization()

print("\n=== Compilation and JIT Optimization ===")

def jit_optimization():
    # Define a simple function
    def simple_function(x, y):
        return torch.relu(x + y) * 2
    
    # JIT compile the function
    jit_function = torch.jit.script(simple_function)
    
    # Prepare data
    x = torch.randn(1000, 1000)
    y = torch.randn(1000, 1000)
    
    # Warm up
    for _ in range(10):
        _ = simple_function(x, y)
        _ = jit_function(x, y)
    
    # Benchmark regular function
    start_time = time.time()
    for _ in range(100):
        result1 = simple_function(x, y)
    regular_time = time.time() - start_time
    
    # Benchmark JIT compiled function
    start_time = time.time()
    for _ in range(100):
        result2 = jit_function(x, y)
    jit_time = time.time() - start_time
    
    print(f"Regular function time: {regular_time:.6f}s")
    print(f"JIT compiled function time: {jit_time:.6f}s")
    print(f"JIT speedup: {regular_time / jit_time:.2f}x")
    print(f"Results equal: {torch.allclose(result1, result2)}")

jit_optimization()

print("\n=== Performance Profiling ===")

def profiling_example():
    """Example of using PyTorch profiler"""
    print("Performance Profiling Tips:")
    print("1. Use torch.profiler for detailed performance analysis")
    print("2. Profile both CPU and GPU operations")
    print("3. Look for memory bottlenecks")
    print("4. Identify kernel fusion opportunities")
    print("5. Monitor data transfer overhead")
    
    # Simple profiling example
    def model_step(x):
        return torch.mm(x, x.t()).sum()
    
    x = torch.randn(500, 500)
    
    # Basic timing
    start_time = time.time()
    for _ in range(10):
        result = model_step(x)
    basic_time = time.time() - start_time
    
    print(f"Basic timing for 10 iterations: {basic_time:.6f}s")

profiling_example()

print("\n=== Performance Best Practices Summary ===")

print("\nPerformance Optimization Checklist:")
print("1. ✓ Keep tensors contiguous in memory")
print("2. ✓ Use in-place operations when possible")
print("3. ✓ Leverage broadcasting instead of explicit expansion")
print("4. ✓ Vectorize operations, avoid Python loops")
print("5. ✓ Use batch processing for multiple samples")
print("6. ✓ Optimize GPU memory transfers with pinned memory")
print("7. ✓ Choose appropriate data types (float16 vs float32)")
print("8. ✓ Use efficient tensor creation methods")
print("9. ✓ Fuse operations when possible")
print("10. ✓ Use torch.no_grad() for inference")
print("11. ✓ Consider JIT compilation for performance-critical code")
print("12. ✓ Profile your code to identify bottlenecks")
print("13. ✓ Manage GPU memory carefully")
print("14. ✓ Use appropriate memory access patterns")
print("15. ✓ Minimize data transfers between CPU and GPU")

print("\nAnti-patterns to Avoid:")
print("❌ Converting tensors to Python lists unnecessarily")
print("❌ Using .item() in loops")
print("❌ Creating tensors inside loops")
print("❌ Unnecessary .cpu() and .cuda() calls")
print("❌ Not reusing tensor allocations")
print("❌ Ignoring memory fragmentation")
print("❌ Mixing data types unnecessarily")
print("❌ Not using appropriate batch sizes")

print("\n=== Performance Optimization Complete ===")

# Performance monitoring utilities
class PerformanceMonitor:
    """Simple performance monitoring utility"""
    
    def __init__(self):
        self.start_time = None
        self.timings = {}
    
    def start(self, name="default"):
        self.start_time = time.time()
        self.current_name = name
    
    def end(self):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            self.timings[self.current_name] = elapsed
            return elapsed
        return None
    
    def report(self):
        print("\nPerformance Report:")
        for name, timing in self.timings.items():
            print(f"{name}: {timing:.6f}s")

# Example usage
monitor = PerformanceMonitor()
monitor.start("tensor_creation")
large_tensor = torch.randn(1000, 1000)
monitor.end()

monitor.start("matrix_multiplication")
result = torch.mm(large_tensor, large_tensor.t())
monitor.end()

monitor.report() 