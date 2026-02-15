#!/usr/bin/env python3
"""PyTorch Tensor Memory Management - Memory optimization and in-place operations"""

import torch
import gc

print("=== Memory Management Basics ===")

# Check current memory usage
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")

# Create tensor and check memory
large_tensor = torch.randn(1000, 1000)
print(f"Created tensor shape: {large_tensor.shape}")
print(f"Tensor memory usage: {large_tensor.numel() * large_tensor.element_size() / 1e6:.2f} MB")

print("\n=== In-place Operations ===")

# In-place vs out-of-place operations
original_tensor = torch.randn(1000, 1000)
original_id = id(original_tensor.data_ptr())

print(f"Original tensor ID: {original_id}")

# Out-of-place operation (creates new tensor)
new_tensor = original_tensor + 1
new_id = id(new_tensor.data_ptr())
print(f"Out-of-place result ID: {new_id}")
print(f"Different memory location: {original_id != new_id}")

# In-place operation (modifies existing tensor)
original_tensor.add_(1)
inplace_id = id(original_tensor.data_ptr())
print(f"In-place result ID: {inplace_id}")
print(f"Same memory location: {original_id == inplace_id}")

print("\n=== In-place Operation Methods ===")

# Various in-place operations
inplace_demo = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(f"Original: {inplace_demo}")

# Arithmetic in-place operations
inplace_demo.add_(2)  # Add 2 in-place
print(f"After add_(2): {inplace_demo}")

inplace_demo.mul_(0.5)  # Multiply by 0.5 in-place
print(f"After mul_(0.5): {inplace_demo}")

inplace_demo.div_(2)  # Divide by 2 in-place
print(f"After div_(2): {inplace_demo}")

inplace_demo.pow_(2)  # Square in-place
print(f"After pow_(2): {inplace_demo}")

inplace_demo.sqrt_()  # Square root in-place
print(f"After sqrt_(): {inplace_demo}")

# Mathematical function in-place operations
math_tensor = torch.tensor([0.1, 0.5, 1.0, 2.0])
print(f"Math tensor: {math_tensor}")

math_tensor.exp_()  # Exponential in-place
print(f"After exp_(): {math_tensor}")

math_tensor.log_()  # Natural log in-place
print(f"After log_(): {math_tensor}")

print("\n=== Memory Views and Sharing ===")

# Views share memory with original tensor
base_tensor = torch.arange(12)
reshaped_view = base_tensor.view(3, 4)
transposed_view = reshaped_view.t()

print(f"Base tensor: {base_tensor}")
print(f"Reshaped view:\n{reshaped_view}")
print(f"Share memory: {base_tensor.data_ptr() == reshaped_view.data_ptr()}")

# Modifying view affects original
reshaped_view[0, 0] = 999
print(f"After modifying view, base tensor: {base_tensor}")

# Clone creates independent copy
cloned_tensor = base_tensor.clone()
cloned_tensor[0] = 777
print(f"After modifying clone, base tensor: {base_tensor}")
print(f"Cloned tensor: {cloned_tensor}")

print("\n=== Contiguous Memory Layout ===")

# Contiguous vs non-contiguous tensors
contiguous_tensor = torch.randn(4, 4)
non_contiguous = contiguous_tensor.transpose(0, 1)

print(f"Contiguous tensor is_contiguous: {contiguous_tensor.is_contiguous()}")
print(f"Transposed tensor is_contiguous: {non_contiguous.is_contiguous()}")

# Strides show memory layout
print(f"Contiguous strides: {contiguous_tensor.stride()}")
print(f"Non-contiguous strides: {non_contiguous.stride()}")

# Make contiguous for better performance
made_contiguous = non_contiguous.contiguous()
print(f"Made contiguous is_contiguous: {made_contiguous.is_contiguous()}")
print(f"Made contiguous strides: {made_contiguous.stride()}")

print("\n=== Memory-Efficient Operations ===")

# Using torch.no_grad() to save memory
def memory_intensive_function(x):
    return x.pow(2).sum()

input_tensor = torch.randn(1000, 1000, requires_grad=True)

# With gradients (uses more memory)
if torch.cuda.is_available():
    initial_memory = torch.cuda.memory_allocated()

result_with_grad = memory_intensive_function(input_tensor)
if torch.cuda.is_available():
    memory_with_grad = torch.cuda.memory_allocated() - initial_memory
    print(f"Memory with gradients: {memory_with_grad / 1e6:.2f} MB")

# Without gradients (saves memory)
with torch.no_grad():
    if torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated()
    result_no_grad = memory_intensive_function(input_tensor)
    if torch.cuda.is_available():
        memory_no_grad = torch.cuda.memory_allocated() - initial_memory
        print(f"Memory without gradients: {memory_no_grad / 1e6:.2f} MB")

print("\n=== Memory Deallocation ===")

# Explicit memory deallocation
temp_tensor = torch.randn(500, 500)
if torch.cuda.is_available():
    temp_gpu_tensor = torch.randn(500, 500, device='cuda')
    print(f"Memory before deletion: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    
    del temp_gpu_tensor
    print(f"Memory after deletion: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    
    # Clear cache
    torch.cuda.empty_cache()
    print(f"Memory after cache clear: {torch.cuda.memory_reserved() / 1e6:.2f} MB")

# Python garbage collection
del temp_tensor
gc.collect()  # Force garbage collection

print("\n=== Memory Pool Management ===")

if torch.cuda.is_available():
    # Memory pool statistics
    memory_stats = torch.cuda.memory_stats()
    print("GPU Memory Statistics:")
    print(f"  Allocated: {memory_stats['allocated_bytes.all.current'] / 1e6:.2f} MB")
    print(f"  Reserved: {memory_stats['reserved_bytes.all.current'] / 1e6:.2f} MB")
    print(f"  Active allocs: {memory_stats['active.all.current']}")
    
    # Memory snapshot for debugging
    torch.cuda.memory_snapshot()  # For profiling tools

print("\n=== Storage and Data Pointers ===")

# Understanding tensor storage
tensor_a = torch.randn(3, 4)
tensor_b = tensor_a.view(2, 6)

print(f"Tensor A storage: {tensor_a.storage()}")
print(f"Tensor B storage: {tensor_b.storage()}")
print(f"Same storage: {tensor_a.storage().data_ptr() == tensor_b.storage().data_ptr()}")

# Data pointer comparison
print(f"Data pointer A: {tensor_a.data_ptr()}")
print(f"Data pointer B: {tensor_b.data_ptr()}")

print("\n=== Memory Mapping ===")

# Memory mapping for large files (simulated)
large_data = torch.randn(1000, 1000)

# Save tensor
torch.save(large_data, 'large_tensor.pt')

# Load with memory mapping (for large files)
loaded_tensor = torch.load('large_tensor.pt', map_location='cpu')
print(f"Loaded tensor shape: {loaded_tensor.shape}")

# Memory-mapped storage (conceptual)
print("Memory mapping useful for:")
print("- Large datasets that don't fit in RAM")
print("- Shared memory between processes")
print("- Faster loading of large tensors")

print("\n=== Pinned Memory ===")

if torch.cuda.is_available():
    # Pinned memory for faster CPU-GPU transfer
    regular_tensor = torch.randn(1000, 1000)
    pinned_tensor = torch.randn(1000, 1000).pin_memory()
    
    print(f"Regular tensor pinned: {regular_tensor.is_pinned()}")
    print(f"Pinned tensor pinned: {pinned_tensor.is_pinned()}")
    
    # Asynchronous transfer with pinned memory
    gpu_tensor = pinned_tensor.cuda(non_blocking=True)
    print("Asynchronous transfer initiated")

print("\n=== Memory-Efficient Patterns ===")

# Pattern 1: In-place operations for memory efficiency
def efficient_normalization(tensor):
    """Normalize tensor in-place to save memory"""
    mean_val = tensor.mean()
    std_val = tensor.std()
    tensor.sub_(mean_val).div_(std_val)
    return tensor

# Pattern 2: Using detach() to break gradient computation
def detached_operation(tensor):
    """Perform operation on detached tensor"""
    detached = tensor.detach()  # Remove from computation graph
    result = detached.pow(2).sum()
    return result

# Pattern 3: Gradient checkpointing for memory efficiency
def memory_efficient_forward(x):
    """Example of memory-efficient forward pass"""
    with torch.no_grad():
        intermediate = x.relu()
    
    # Re-enable gradients only when needed
    intermediate.requires_grad_(True)
    output = intermediate.pow(2)
    return output

# Pattern 4: Chunked processing for large tensors
def chunked_operation(large_tensor, chunk_size=1000):
    """Process large tensor in chunks"""
    results = []
    for i in range(0, large_tensor.size(0), chunk_size):
        chunk = large_tensor[i:i+chunk_size]
        chunk_result = chunk.pow(2).sum()
        results.append(chunk_result)
    return torch.stack(results)

print("\n=== Memory Profiling Utilities ===")

# Memory profiling context manager
class MemoryProfiler:
    def __init__(self):
        self.start_memory = 0
        
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_memory = torch.cuda.memory_allocated()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_memory = torch.cuda.memory_allocated()
            memory_used = (end_memory - self.start_memory) / 1e6
            print(f"Memory used: {memory_used:.2f} MB")

# Example usage
if torch.cuda.is_available():
    with MemoryProfiler():
        temp_large = torch.randn(500, 500, device='cuda')
        temp_result = temp_large @ temp_large.t()
        del temp_large, temp_result

print("\n=== Memory Best Practices ===")

print("Memory optimization tips:")
print("1. Use in-place operations when possible")
print("2. Delete large tensors explicitly with del")
print("3. Use torch.no_grad() for inference")
print("4. Clear CUDA cache periodically")
print("5. Use pinned memory for CPU-GPU transfers")
print("6. Process large datasets in chunks")
print("7. Use views instead of copies when possible")
print("8. Monitor memory usage in training loops")

# Cleanup
import os
if os.path.exists('large_tensor.pt'):
    os.remove('large_tensor.pt')

print("\n=== Memory Management Complete ===") 