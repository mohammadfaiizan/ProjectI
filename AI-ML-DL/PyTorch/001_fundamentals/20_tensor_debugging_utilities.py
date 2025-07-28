#!/usr/bin/env python3
"""PyTorch Tensor Debugging Utilities and Tools"""

import torch
import sys
import traceback

print("=== Tensor Information Inspection ===")

# Create sample tensors for debugging
debug_tensor = torch.randn(3, 4, 5)
complex_tensor = torch.tensor([1+2j, 3+4j], dtype=torch.complex64)
bool_tensor = torch.tensor([True, False, True])

def inspect_tensor(tensor, name="tensor"):
    """Comprehensive tensor inspection function"""
    print(f"\n=== {name.upper()} INSPECTION ===")
    print(f"Shape: {tensor.shape}")
    print(f"Size: {tensor.size()}")
    print(f"Number of dimensions: {tensor.ndim}")
    print(f"Number of elements: {tensor.numel()}")
    print(f"Data type: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    print(f"Layout: {tensor.layout}")
    print(f"Memory format: {tensor.memory_format if hasattr(tensor, 'memory_format') else 'N/A'}")
    print(f"Requires grad: {tensor.requires_grad}")
    print(f"Is leaf: {tensor.is_leaf if tensor.requires_grad else 'N/A'}")
    print(f"Grad function: {tensor.grad_fn if tensor.requires_grad else 'N/A'}")
    print(f"Is contiguous: {tensor.is_contiguous()}")
    print(f"Storage offset: {tensor.storage_offset()}")
    print(f"Stride: {tensor.stride()}")
    print(f"Element size: {tensor.element_size()} bytes")
    print(f"Storage size: {tensor.storage().size()}")
    print(f"Memory usage: {tensor.numel() * tensor.element_size() / 1024:.2f} KB")
    
    if tensor.is_complex():
        print(f"Is complex: True")
        print(f"Real part shape: {tensor.real.shape}")
        print(f"Imaginary part shape: {tensor.imag.shape}")
    
    if tensor.is_floating_point():
        print(f"Is floating point: True")
        if tensor.numel() > 0:
            print(f"Min value: {tensor.min().item()}")
            print(f"Max value: {tensor.max().item()}")
            print(f"Mean: {tensor.mean().item():.6f}")
            print(f"Std: {tensor.std().item():.6f}")
    
    if tensor.dtype == torch.bool:
        print(f"True count: {tensor.sum().item()}")
        print(f"False count: {(~tensor).sum().item()}")
    
    print(f"Has NaN: {torch.isnan(tensor).any().item() if tensor.is_floating_point() else False}")
    print(f"Has Inf: {torch.isinf(tensor).any().item() if tensor.is_floating_point() else False}")

# Inspect sample tensors
inspect_tensor(debug_tensor, "debug_tensor")
inspect_tensor(complex_tensor, "complex_tensor")
inspect_tensor(bool_tensor, "bool_tensor")

print("\n=== Memory and Storage Debugging ===")

def debug_memory_layout(tensor, name="tensor"):
    """Debug tensor memory layout and storage"""
    print(f"\n--- {name} Memory Layout ---")
    print(f"Data pointer: {tensor.data_ptr()}")
    print(f"Storage data pointer: {tensor.storage().data_ptr()}")
    print(f"Same data pointer: {tensor.data_ptr() == tensor.storage().data_ptr()}")
    print(f"Storage size: {tensor.storage().size()}")
    print(f"Storage element size: {tensor.storage().element_size()}")
    print(f"Tensor elements: {tensor.numel()}")
    print(f"Storage utilization: {tensor.numel() / tensor.storage().size() * 100:.1f}%")

# Memory layout examples
original = torch.arange(12).reshape(3, 4)
view_tensor = original.view(2, 6)
transposed = original.t()

debug_memory_layout(original, "original")
debug_memory_layout(view_tensor, "view")
debug_memory_layout(transposed, "transposed")

print("\n=== Gradient Debugging ===")

# Gradient debugging utilities
def debug_gradients(tensor, name="tensor"):
    """Debug gradient information"""
    print(f"\n--- {name} Gradient Info ---")
    print(f"Requires grad: {tensor.requires_grad}")
    if tensor.requires_grad:
        print(f"Is leaf: {tensor.is_leaf}")
        print(f"Grad function: {tensor.grad_fn}")
        print(f"Gradient: {tensor.grad}")
        if tensor.grad is not None:
            print(f"Grad shape: {tensor.grad.shape}")
            print(f"Grad dtype: {tensor.grad.dtype}")
    else:
        print("No gradient tracking")

# Gradient debugging example
x = torch.randn(3, 4, requires_grad=True)
y = x.pow(2).sum()
y.backward()

debug_gradients(x, "input_tensor")
debug_gradients(y, "output_tensor")

print("\n=== NaN and Infinity Detection ===")

def find_problematic_values(tensor, name="tensor"):
    """Find and locate NaN and infinity values"""
    print(f"\n--- {name} Problematic Values ---")
    
    if not tensor.is_floating_point():
        print("Not a floating point tensor")
        return
    
    # Check for NaN
    nan_mask = torch.isnan(tensor)
    nan_count = nan_mask.sum().item()
    
    # Check for infinity
    inf_mask = torch.isinf(tensor)
    inf_count = inf_mask.sum().item()
    
    # Check for finite values
    finite_mask = torch.isfinite(tensor)
    finite_count = finite_mask.sum().item()
    
    print(f"Total elements: {tensor.numel()}")
    print(f"NaN count: {nan_count}")
    print(f"Infinity count: {inf_count}")
    print(f"Finite count: {finite_count}")
    
    if nan_count > 0:
        nan_indices = torch.nonzero(nan_mask)
        print(f"NaN locations (first 5): {nan_indices[:5].tolist()}")
    
    if inf_count > 0:
        inf_indices = torch.nonzero(inf_mask)
        print(f"Infinity locations (first 5): {inf_indices[:5].tolist()}")

# Test with problematic values
problematic = torch.tensor([1.0, float('nan'), 3.0, float('inf'), 5.0, float('-inf')])
find_problematic_values(problematic, "problematic_tensor")

print("\n=== Shape and Dimension Debugging ===")

def debug_tensor_shapes(*tensors, operation="unknown"):
    """Debug shape compatibility for operations"""
    print(f"\n--- Shape Debugging for {operation} ---")
    
    for i, tensor in enumerate(tensors):
        print(f"Tensor {i}: shape {tensor.shape}, dtype {tensor.dtype}")
    
    # Check broadcasting compatibility
    if len(tensors) == 2:
        try:
            broadcast_shape = torch.broadcast_shapes(tensors[0].shape, tensors[1].shape)
            print(f"Broadcast compatible: {broadcast_shape}")
        except RuntimeError as e:
            print(f"Broadcast error: {e}")
    
    # Check matrix multiplication compatibility
    if len(tensors) == 2 and len(tensors[0].shape) >= 2 and len(tensors[1].shape) >= 2:
        a_shape = tensors[0].shape
        b_shape = tensors[1].shape
        if a_shape[-1] == b_shape[-2]:
            result_shape = a_shape[:-1] + b_shape[-1:]
            print(f"Matmul compatible: {result_shape}")
        else:
            print(f"Matmul incompatible: {a_shape[-1]} != {b_shape[-2]}")

# Shape debugging examples
tensor_a = torch.randn(3, 4)
tensor_b = torch.randn(4, 5)
tensor_c = torch.randn(3, 5)

debug_tensor_shapes(tensor_a, tensor_b, operation="matrix multiplication")
debug_tensor_shapes(tensor_a, tensor_c, operation="element-wise addition")

print("\n=== Performance Debugging ===")

def profile_tensor_operation(operation, *args, num_runs=100, name="operation"):
    """Profile tensor operation performance"""
    import time
    
    print(f"\n--- Profiling {name} ---")
    
    # Warm up
    for _ in range(10):
        result = operation(*args)
    
    # GPU synchronization if needed
    if torch.cuda.is_available() and any(arg.is_cuda for arg in args if torch.is_tensor(arg)):
        torch.cuda.synchronize()
    
    # Time the operation
    start_time = time.time()
    for _ in range(num_runs):
        result = operation(*args)
    
    if torch.cuda.is_available() and any(arg.is_cuda for arg in args if torch.is_tensor(arg)):
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    print(f"Average time per operation: {avg_time*1000:.4f} ms")
    print(f"Operations per second: {1/avg_time:.0f}")
    
    return result, avg_time

# Profile example operations
large_a = torch.randn(1000, 1000)
large_b = torch.randn(1000, 1000)

profile_tensor_operation(torch.add, large_a, large_b, name="addition")
profile_tensor_operation(torch.mm, large_a, large_b, name="matrix multiplication")

print("\n=== Error Context Debugging ===")

class TensorDebugContext:
    """Context manager for tensor debugging"""
    
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.original_settings = {}
    
    def __enter__(self):
        if self.enabled:
            # Enable anomaly detection
            self.original_settings['anomaly'] = torch.is_anomaly_enabled()
            torch.autograd.set_detect_anomaly(True)
            
            # Set print options for better debugging
            self.original_settings['precision'] = torch.get_printoptions()['precision']
            torch.set_printoptions(precision=8, sci_mode=False)
            
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            # Restore settings
            torch.autograd.set_detect_anomaly(self.original_settings['anomaly'])
            torch.set_printoptions(precision=self.original_settings['precision'])
            
        if exc_type is not None:
            print(f"\nException occurred: {exc_type.__name__}: {exc_val}")
            print("Tensor debugging context active")

# Example usage
with TensorDebugContext():
    try:
        # This might cause issues
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = x / 0  # Division by zero
    except Exception as e:
        print(f"Caught exception: {e}")

print("\n=== Custom Debugging Utilities ===")

def tensor_summary_table(*tensors, names=None):
    """Create a summary table of tensor properties"""
    if names is None:
        names = [f"tensor_{i}" for i in range(len(tensors))]
    
    print("\n" + "="*80)
    print(f"{'Name':<15} {'Shape':<15} {'Dtype':<10} {'Device':<8} {'Grad':<5} {'Memory(KB)':<12}")
    print("="*80)
    
    for tensor, name in zip(tensors, names):
        memory_kb = tensor.numel() * tensor.element_size() / 1024
        grad_status = "Yes" if tensor.requires_grad else "No"
        
        print(f"{name:<15} {str(tensor.shape):<15} {str(tensor.dtype):<10} "
              f"{str(tensor.device):<8} {grad_status:<5} {memory_kb:<12.2f}")

# Summary table example
t1 = torch.randn(100, 100)
t2 = torch.randn(50, 50, device='cpu', requires_grad=True)
t3 = torch.zeros(200, 200, dtype=torch.int32)

tensor_summary_table(t1, t2, t3, names=["random_tensor", "grad_tensor", "zero_tensor"])

print("\n=== Debugging Best Practices ===")

print("PyTorch Debugging Tips:")
print("1. Use torch.autograd.set_detect_anomaly(True) for gradient issues")
print("2. Check tensor shapes before operations")
print("3. Verify data types are compatible")
print("4. Monitor memory usage with torch.cuda.memory_allocated()")
print("5. Use tensor.is_contiguous() for performance issues")
print("6. Check for NaN/Inf with torch.isnan() and torch.isinf()")
print("7. Use tensor.grad_fn to trace computation graph")
print("8. Profile operations to identify bottlenecks")
print("9. Use descriptive variable names for easier debugging")
print("10. Implement custom debugging contexts for complex operations")

print("\n=== Debugging Utilities Complete ===") 