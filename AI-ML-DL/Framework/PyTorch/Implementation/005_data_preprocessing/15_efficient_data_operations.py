#!/usr/bin/env python3
"""PyTorch Efficient Data Operations - Memory-efficient data operations"""

import torch
import gc
import time
import psutil
import os

print("=== Efficient Data Operations Overview ===")

print("Efficiency considerations:")
print("1. Memory management and optimization")
print("2. In-place operations vs memory allocation")
print("3. Vectorization and broadcasting")
print("4. Chunked processing for large datasets")
print("5. GPU memory management")
print("6. Lazy evaluation and streaming")
print("7. Parallel processing")

print("\n=== Memory Management ===")

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB

def memory_profile(func):
    """Decorator to profile memory usage"""
    def wrapper(*args, **kwargs):
        mem_before = get_memory_usage()
        result = func(*args, **kwargs)
        mem_after = get_memory_usage()
        print(f"{func.__name__}: Memory usage: {mem_after - mem_before:.2f} MB")
        return result
    return wrapper

def clear_cache():
    """Clear Python and CUDA cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Test memory profiling
@memory_profile
def create_large_tensor(size):
    return torch.randn(size, size)

@memory_profile
def create_large_tensor_inplace(size):
    tensor = torch.empty(size, size)
    tensor.normal_()
    return tensor

print("Memory usage comparison:")
tensor1 = create_large_tensor(1000)
clear_cache()
tensor2 = create_large_tensor_inplace(1000)

del tensor1, tensor2
clear_cache()

print("\n=== In-place Operations ===")

def compare_inplace_vs_copy():
    """Compare in-place vs copy operations"""
    data = torch.randn(10000, 1000)
    
    # Copy operation
    start_time = time.time()
    result_copy = data * 2.0 + 1.0
    copy_time = time.time() - start_time
    
    # In-place operations
    data_inplace = data.clone()
    start_time = time.time()
    data_inplace.mul_(2.0).add_(1.0)
    inplace_time = time.time() - start_time
    
    return copy_time, inplace_time

copy_time, inplace_time = compare_inplace_vs_copy()
print(f"Copy operations time: {copy_time:.6f} seconds")
print(f"In-place operations time: {inplace_time:.6f} seconds")
print(f"Speedup: {copy_time / inplace_time:.2f}x")

print("\n=== Vectorization and Broadcasting ===")

def vectorized_operations_demo():
    """Demonstrate vectorized operations vs loops"""
    data = torch.randn(10000, 100)
    weights = torch.randn(100)
    
    # Slow: Loop-based approach
    start_time = time.time()
    result_loop = torch.zeros(10000)
    for i in range(10000):
        result_loop[i] = torch.dot(data[i], weights)
    loop_time = time.time() - start_time
    
    # Fast: Vectorized approach
    start_time = time.time()
    result_vectorized = torch.mv(data, weights)
    vectorized_time = time.time() - start_time
    
    # Fast: Broadcasting approach
    start_time = time.time()
    result_broadcast = (data * weights).sum(dim=1)
    broadcast_time = time.time() - start_time
    
    return loop_time, vectorized_time, broadcast_time

loop_time, vec_time, broad_time = vectorized_operations_demo()
print(f"Loop-based time: {loop_time:.6f} seconds")
print(f"Vectorized time: {vec_time:.6f} seconds")
print(f"Broadcasting time: {broad_time:.6f} seconds")
print(f"Vectorization speedup: {loop_time / vec_time:.1f}x")

print("\n=== Chunked Processing ===")

def chunked_processing(data, chunk_size, process_func):
    """Process large data in chunks"""
    n_samples = len(data)
    results = []
    
    for i in range(0, n_samples, chunk_size):
        end_idx = min(i + chunk_size, n_samples)
        chunk = data[i:end_idx]
        
        chunk_result = process_func(chunk)
        results.append(chunk_result)
        
        # Optional: Clear intermediate results
        del chunk_result
    
    return torch.cat(results, dim=0)

def chunked_matrix_multiply(A, B, chunk_size=1000):
    """Memory-efficient matrix multiplication for large matrices"""
    m, k = A.shape
    k2, n = B.shape
    assert k == k2, "Matrix dimensions must match"
    
    # Process in chunks along the m dimension
    results = []
    for i in range(0, m, chunk_size):
        end_idx = min(i + chunk_size, m)
        chunk_A = A[i:end_idx]
        chunk_result = torch.mm(chunk_A, B)
        results.append(chunk_result)
    
    return torch.cat(results, dim=0)

# Test chunked processing
large_data = torch.randn(10000, 50)

def simple_process(chunk):
    return torch.sum(chunk, dim=1, keepdim=True)

# Process in chunks vs all at once
start_time = time.time()
chunked_result = chunked_processing(large_data, chunk_size=1000, process_func=simple_process)
chunked_time = time.time() - start_time

start_time = time.time()
direct_result = simple_process(large_data)
direct_time = time.time() - start_time

print(f"Chunked processing time: {chunked_time:.6f} seconds")
print(f"Direct processing time: {direct_time:.6f} seconds")
print(f"Results match: {torch.allclose(chunked_result, direct_result)}")

print("\n=== Memory-Efficient Data Structures ===")

class MemoryEfficientDataset:
    """Memory-efficient dataset that loads data on demand"""
    
    def __init__(self, data_size, feature_dim):
        self.data_size = data_size
        self.feature_dim = feature_dim
        # Store metadata instead of actual data
        self.metadata = {
            'mean': torch.zeros(feature_dim),
            'std': torch.ones(feature_dim),
            'seed': 42
        }
    
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        # Generate data on-the-fly
        torch.manual_seed(self.metadata['seed'] + idx)
        sample = torch.randn(self.feature_dim)
        sample = sample * self.metadata['std'] + self.metadata['mean']
        return sample
    
    def get_batch(self, indices):
        """Efficiently get multiple samples"""
        batch = torch.zeros(len(indices), self.feature_dim)
        for i, idx in enumerate(indices):
            batch[i] = self[idx]
        return batch

class SparseDataOperations:
    """Efficient operations for sparse data"""
    
    @staticmethod
    def sparse_dense_multiply(sparse_tensor, dense_tensor):
        """Efficient sparse-dense multiplication"""
        return torch.sparse.mm(sparse_tensor, dense_tensor)
    
    @staticmethod
    def create_sparse_tensor(indices, values, size):
        """Create sparse tensor efficiently"""
        return torch.sparse_coo_tensor(indices, values, size)
    
    @staticmethod
    def sparse_to_dense_efficient(sparse_tensor, chunk_size=1000):
        """Convert sparse to dense in chunks"""
        indices = sparse_tensor.indices()
        values = sparse_tensor.values()
        shape = sparse_tensor.shape
        
        dense_tensor = torch.zeros(shape)
        
        # Process in chunks to avoid memory issues
        n_values = len(values)
        for i in range(0, n_values, chunk_size):
            end_idx = min(i + chunk_size, n_values)
            chunk_indices = indices[:, i:end_idx]
            chunk_values = values[i:end_idx]
            
            # Set values
            if chunk_indices.shape[0] == 2:  # 2D case
                dense_tensor[chunk_indices[0], chunk_indices[1]] = chunk_values
            elif chunk_indices.shape[0] == 1:  # 1D case
                dense_tensor[chunk_indices[0]] = chunk_values
        
        return dense_tensor

# Test memory-efficient dataset
efficient_dataset = MemoryEfficientDataset(10000, 100)
print(f"Dataset size: {len(efficient_dataset)}")

# Sample some data
sample_indices = torch.randint(0, len(efficient_dataset), (5,))
batch_data = efficient_dataset.get_batch(sample_indices)
print(f"Batch shape: {batch_data.shape}")

# Test sparse operations
sparse_indices = torch.tensor([[0, 1, 2], [2, 0, 1]])
sparse_values = torch.tensor([1.0, 2.0, 3.0])
sparse_tensor = SparseDataOperations.create_sparse_tensor(
    sparse_indices, sparse_values, (3, 3)
)
print(f"Sparse tensor: {sparse_tensor}")

print("\n=== GPU Memory Management ===")

def gpu_memory_management():
    """Demonstrate GPU memory management techniques"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU tests")
        return
    
    device = torch.device('cuda')
    
    print("GPU Memory management:")
    
    # Check initial memory
    print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1e6:.2f} MB")
    
    # Create tensor on GPU
    gpu_tensor = torch.randn(1000, 1000, device=device)
    print(f"After creating tensor: {torch.cuda.memory_allocated()/1e6:.2f} MB")
    
    # Delete tensor and clear cache
    del gpu_tensor
    torch.cuda.empty_cache()
    print(f"After cleanup: {torch.cuda.memory_allocated()/1e6:.2f} MB")

def efficient_gpu_operations():
    """Efficient GPU operations"""
    if not torch.cuda.is_available():
        return
    
    device = torch.device('cuda')
    
    # Pre-allocate tensors
    a = torch.empty(1000, 1000, device=device)
    b = torch.empty(1000, 1000, device=device)
    result = torch.empty(1000, 1000, device=device)
    
    # Fill tensors
    a.normal_()
    b.normal_()
    
    # In-place operations to avoid memory allocation
    torch.mm(a, b, out=result)
    
    return result

gpu_memory_management()

print("\n=== Lazy Evaluation and Streaming ===")

class LazyOperation:
    """Lazy evaluation for chained operations"""
    
    def __init__(self, data):
        self.data = data
        self.operations = []
    
    def add_operation(self, op_func, *args, **kwargs):
        """Add operation to chain"""
        self.operations.append((op_func, args, kwargs))
        return self
    
    def execute(self):
        """Execute all operations"""
        result = self.data
        for op_func, args, kwargs in self.operations:
            result = op_func(result, *args, **kwargs)
        return result
    
    def execute_chunked(self, chunk_size=1000):
        """Execute operations in chunks"""
        n_samples = len(self.data)
        results = []
        
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            chunk = self.data[i:end_idx]
            
            # Apply all operations to chunk
            for op_func, args, kwargs in self.operations:
                chunk = op_func(chunk, *args, **kwargs)
            
            results.append(chunk)
        
        return torch.cat(results, dim=0)

class StreamingProcessor:
    """Process data in streaming fashion"""
    
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.buffer = []
    
    def add_sample(self, sample):
        """Add sample to buffer"""
        self.buffer.append(sample)
        
        if len(self.buffer) >= self.batch_size:
            return self.process_batch()
        return None
    
    def process_batch(self):
        """Process accumulated batch"""
        if not self.buffer:
            return None
        
        batch = torch.stack(self.buffer)
        self.buffer.clear()
        
        # Process batch (example: normalize)
        normalized_batch = (batch - batch.mean()) / (batch.std() + 1e-8)
        return normalized_batch
    
    def flush(self):
        """Process remaining samples"""
        if self.buffer:
            return self.process_batch()
        return None

# Test lazy evaluation
lazy_data = torch.randn(5000, 10)
lazy_op = LazyOperation(lazy_data)
lazy_op.add_operation(torch.abs)
lazy_op.add_operation(torch.log)
lazy_op.add_operation(lambda x: x + 1)

lazy_result = lazy_op.execute_chunked(chunk_size=1000)
print(f"Lazy evaluation result shape: {lazy_result.shape}")

# Test streaming
streaming_proc = StreamingProcessor(batch_size=16)
processed_batches = []

for i in range(50):
    sample = torch.randn(10)
    batch_result = streaming_proc.add_sample(sample)
    if batch_result is not None:
        processed_batches.append(batch_result)

# Process remaining
final_batch = streaming_proc.flush()
if final_batch is not None:
    processed_batches.append(final_batch)

print(f"Processed {len(processed_batches)} batches via streaming")

print("\n=== Advanced Memory Optimization ===")

def memory_mapped_operations():
    """Demonstrate memory-mapped file operations"""
    # Create a temporary file for demonstration
    temp_data = torch.randn(1000, 100)
    
    # Save to file
    torch.save(temp_data, 'temp_data.pt')
    
    # Load without fully reading into memory (conceptual)
    # In practice, you'd use memory-mapped files for very large datasets
    loaded_data = torch.load('temp_data.pt')
    
    # Clean up
    os.remove('temp_data.pt')
    
    return loaded_data.shape

def gradient_checkpointing_demo():
    """Demonstrate gradient checkpointing concept"""
    
    class CheckpointedFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, weight):
            # Save only what's needed for backward
            ctx.save_for_backward(input_tensor, weight)
            # Don't save intermediate activations
            return torch.mm(input_tensor, weight)
        
        @staticmethod
        def backward(ctx, grad_output):
            input_tensor, weight = ctx.saved_tensors
            # Recompute forward pass if needed
            grad_input = torch.mm(grad_output, weight.t())
            grad_weight = torch.mm(input_tensor.t(), grad_output)
            return grad_input, grad_weight
    
    # Use the checkpointed function
    input_data = torch.randn(100, 50, requires_grad=True)
    weight = torch.randn(50, 25, requires_grad=True)
    
    output = CheckpointedFunction.apply(input_data, weight)
    return output

print("Memory-mapped operations shape:", memory_mapped_operations())

print("\n=== Performance Optimization Tips ===")

def benchmark_operation(operation, *args, num_runs=100):
    """Benchmark an operation"""
    # Warm up
    for _ in range(10):
        operation(*args)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    start_time = time.time()
    for _ in range(num_runs):
        operation(*args)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time

def optimized_batch_norm(input_tensor, eps=1e-5):
    """Optimized batch normalization"""
    # Use torch.var with unbiased=False for efficiency
    mean = input_tensor.mean(dim=0, keepdim=True)
    var = input_tensor.var(dim=0, keepdim=True, unbiased=False)
    
    # In-place operations where possible
    normalized = input_tensor.sub(mean).div(torch.sqrt(var.add_(eps)))
    return normalized

def naive_batch_norm(input_tensor, eps=1e-5):
    """Naive batch normalization"""
    mean = input_tensor.mean(dim=0, keepdim=True)
    var = input_tensor.var(dim=0, keepdim=True, unbiased=False)
    normalized = (input_tensor - mean) / torch.sqrt(var + eps)
    return normalized

# Benchmark different implementations
test_input = torch.randn(1000, 128)

optimized_time = benchmark_operation(optimized_batch_norm, test_input)
naive_time = benchmark_operation(naive_batch_norm, test_input)

print(f"Optimized batch norm: {optimized_time*1000:.6f} ms")
print(f"Naive batch norm: {naive_time*1000:.6f} ms")
print(f"Speedup: {naive_time / optimized_time:.2f}x")

print("\n=== Efficiency Best Practices ===")

print("Memory Efficiency Guidelines:")
print("1. Use in-place operations when possible (add_, mul_, etc.)")
print("2. Delete unused tensors and call gc.collect()")
print("3. Use appropriate data types (float16 vs float32)")
print("4. Process data in chunks for large datasets")
print("5. Use sparse tensors for sparse data")
print("6. Implement lazy evaluation for complex pipelines")
print("7. Monitor memory usage during development")

print("\nComputational Efficiency:")
print("1. Vectorize operations instead of using loops")
print("2. Use broadcasting for element-wise operations")
print("3. Choose efficient algorithms and implementations")
print("4. Use torch.compile() for JIT compilation (PyTorch 2.0+)")
print("5. Batch operations when possible")
print("6. Use appropriate tensor layouts (contiguous memory)")
print("7. Profile your code to identify bottlenecks")

print("\nGPU Optimization:")
print("1. Keep data on GPU to avoid transfer overhead")
print("2. Use appropriate batch sizes for GPU utilization")
print("3. Use torch.cuda.amp for mixed precision training")
print("4. Pre-allocate tensors when possible")
print("5. Use asynchronous operations when appropriate")
print("6. Monitor GPU memory usage")

print("\nDevelopment Tips:")
print("1. Profile memory and compute usage regularly")
print("2. Test with different data sizes")
print("3. Use memory and time benchmarks")
print("4. Consider trade-offs between memory and computation")
print("5. Document performance characteristics")
print("6. Use efficient data loading strategies")

print("\n=== Efficient Data Operations Complete ===")

# Final cleanup
clear_cache() 