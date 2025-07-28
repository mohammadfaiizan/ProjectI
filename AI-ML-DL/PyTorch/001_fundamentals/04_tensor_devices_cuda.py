#!/usr/bin/env python3
"""PyTorch Tensor Devices and CUDA Operations"""

import torch

print("=== Device Detection ===")

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"Device {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"  Compute capability: {props.major}.{props.minor}")

# Check MPS (Apple Silicon) availability
mps_available = torch.backends.mps.is_available()
print(f"MPS (Apple Silicon) available: {mps_available}")

print("\n=== Device Objects ===")

# CPU device
cpu_device = torch.device('cpu')
print(f"CPU device: {cpu_device}")

# CUDA devices
if cuda_available:
    cuda_device = torch.device('cuda')
    cuda_device_0 = torch.device('cuda:0')
    if torch.cuda.device_count() > 1:
        cuda_device_1 = torch.device('cuda:1')
        print(f"CUDA device: {cuda_device}")
        print(f"CUDA device 0: {cuda_device_0}")
        print(f"CUDA device 1: {cuda_device_1}")

# MPS device
if mps_available:
    mps_device = torch.device('mps')
    print(f"MPS device: {mps_device}")

print("\n=== Creating Tensors on Devices ===")

# CPU tensors
cpu_tensor = torch.randn(3, 4)
print(f"CPU tensor device: {cpu_tensor.device}")

# Direct device creation
if cuda_available:
    cuda_tensor = torch.randn(3, 4, device='cuda')
    print(f"CUDA tensor device: {cuda_tensor.device}")

# Using device object
cpu_zeros = torch.zeros(2, 3, device=cpu_device)
print(f"CPU zeros device: {cpu_zeros.device}")

if cuda_available:
    cuda_ones = torch.ones(2, 3, device=cuda_device)
    print(f"CUDA ones device: {cuda_ones.device}")

print("\n=== Moving Tensors Between Devices ===")

# Create tensor on CPU
tensor_cpu = torch.randn(4, 4)
print(f"Original device: {tensor_cpu.device}")

if cuda_available:
    # Method 1: .to() method
    tensor_gpu1 = tensor_cpu.to('cuda')
    print(f"Moved to CUDA (.to): {tensor_gpu1.device}")
    
    # Method 2: .cuda() method
    tensor_gpu2 = tensor_cpu.cuda()
    print(f"Moved to CUDA (.cuda): {tensor_gpu2.device}")
    
    # Method 3: .to(device) with device object
    tensor_gpu3 = tensor_cpu.to(cuda_device)
    print(f"Moved to CUDA (.to device): {tensor_gpu3.device}")
    
    # Move back to CPU
    tensor_cpu_back1 = tensor_gpu1.to('cpu')
    tensor_cpu_back2 = tensor_gpu1.cpu()
    print(f"Back to CPU (.to): {tensor_cpu_back1.device}")
    print(f"Back to CPU (.cpu): {tensor_cpu_back2.device}")

print("\n=== Device Context Managers ===")

if cuda_available:
    # Set default device
    with torch.cuda.device(0):
        tensor_default = torch.randn(2, 2)
        print(f"Tensor in context: {tensor_default.device}")
    
    # Multiple device context
    if torch.cuda.device_count() > 1:
        with torch.cuda.device(1):
            tensor_dev1 = torch.randn(2, 2)
            print(f"Tensor on device 1: {tensor_dev1.device}")

print("\n=== Memory Management ===")

if cuda_available:
    # Check memory usage
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
    
    # Create large tensor
    large_tensor = torch.randn(1000, 1000, device='cuda')
    print(f"After large tensor - Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    
    # Delete tensor
    del large_tensor
    print(f"After deletion - Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    
    # Clear cache
    torch.cuda.empty_cache()
    print(f"After cache clear - Cached: {torch.cuda.memory_reserved() / 1e6:.2f} MB")

print("\n=== Cross-Device Operations ===")

if cuda_available:
    # Tensors on different devices
    cpu_tensor = torch.randn(3, 3)
    gpu_tensor = torch.randn(3, 3, device='cuda')
    
    print(f"CPU tensor: {cpu_tensor.device}")
    print(f"GPU tensor: {gpu_tensor.device}")
    
    # This would cause error - tensors on different devices
    try:
        result = cpu_tensor + gpu_tensor
    except RuntimeError as e:
        print(f"Cross-device error: {str(e)[:50]}...")
    
    # Correct way - move to same device
    result = cpu_tensor.cuda() + gpu_tensor
    print(f"Correct operation result device: {result.device}")

print("\n=== Device-Specific Operations ===")

if cuda_available:
    # CUDA-specific functions
    tensor_cuda = torch.randn(1000, 1000, device='cuda')
    
    # Synchronize CUDA operations
    torch.cuda.synchronize()
    
    # Get current device
    current_device = torch.cuda.current_device()
    print(f"Current CUDA device: {current_device}")
    
    # Set device
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(1)
        print(f"Set device to: {torch.cuda.current_device()}")
        torch.cuda.set_device(0)  # Reset

print("\n=== Performance Comparison ===")

import time

# CPU performance
cpu_a = torch.randn(1000, 1000)
cpu_b = torch.randn(1000, 1000)

start_time = time.time()
cpu_result = torch.mm(cpu_a, cpu_b)
cpu_time = time.time() - start_time
print(f"CPU matrix multiplication: {cpu_time:.4f} seconds")

if cuda_available:
    # GPU performance
    gpu_a = torch.randn(1000, 1000, device='cuda')
    gpu_b = torch.randn(1000, 1000, device='cuda')
    
    # Warm up GPU
    _ = torch.mm(gpu_a, gpu_b)
    torch.cuda.synchronize()
    
    start_time = time.time()
    gpu_result = torch.mm(gpu_a, gpu_b)
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    print(f"GPU matrix multiplication: {gpu_time:.4f} seconds")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")

print("\n=== Data Transfer Timing ===")

if cuda_available:
    # Time data transfer
    large_cpu_tensor = torch.randn(5000, 5000)
    
    # CPU to GPU transfer
    start_time = time.time()
    large_gpu_tensor = large_cpu_tensor.cuda()
    transfer_time = time.time() - start_time
    print(f"CPU to GPU transfer: {transfer_time:.4f} seconds")
    
    # GPU to CPU transfer
    start_time = time.time()
    back_to_cpu = large_gpu_tensor.cpu()
    back_transfer_time = time.time() - start_time
    print(f"GPU to CPU transfer: {back_transfer_time:.4f} seconds")

print("\n=== Pinned Memory ===")

if cuda_available:
    # Regular tensor transfer
    regular_tensor = torch.randn(1000, 1000)
    start_time = time.time()
    regular_gpu = regular_tensor.cuda()
    regular_time = time.time() - start_time
    
    # Pinned memory tensor transfer
    pinned_tensor = torch.randn(1000, 1000).pin_memory()
    start_time = time.time()
    pinned_gpu = pinned_tensor.cuda(non_blocking=True)
    torch.cuda.synchronize()
    pinned_time = time.time() - start_time
    
    print(f"Regular transfer: {regular_time:.4f} seconds")
    print(f"Pinned transfer: {pinned_time:.4f} seconds")
    print(f"Speedup with pinned memory: {regular_time / pinned_time:.2f}x")

print("\n=== Error Handling ===")

# Common device-related errors
try:
    # Trying to use CUDA when not available
    if not cuda_available:
        torch.randn(2, 2, device='cuda')
except RuntimeError as e:
    print(f"CUDA not available error: {str(e)[:50]}...")

try:
    # Invalid device index
    torch.randn(2, 2, device=f'cuda:{torch.cuda.device_count()}')
except RuntimeError as e:
    print(f"Invalid device error: {str(e)[:50]}...")

print("\n=== Device Operations Complete ===") 