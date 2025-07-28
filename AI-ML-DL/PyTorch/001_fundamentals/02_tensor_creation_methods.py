#!/usr/bin/env python3
"""PyTorch Tensor Creation Methods - All tensor creation syntax"""

import torch
import numpy as np

print("=== Basic Tensor Creation ===")

# From Python lists
list_1d = [1, 2, 3, 4, 5]
tensor_1d = torch.tensor(list_1d)
print(f"From 1D list: {tensor_1d}")

list_2d = [[1, 2, 3], [4, 5, 6]]
tensor_2d = torch.tensor(list_2d)
print(f"From 2D list: {tensor_2d}")

# From NumPy arrays
np_array = np.array([1.0, 2.0, 3.0])
tensor_from_numpy = torch.from_numpy(np_array)
print(f"From NumPy: {tensor_from_numpy}")

# From other tensors
tensor_copy = torch.tensor(tensor_1d)
tensor_clone = tensor_1d.clone()
print(f"Copy: {tensor_copy}")
print(f"Clone: {tensor_clone}")

print("\n=== Zero and One Tensors ===")

# Zeros
zeros_1d = torch.zeros(5)
zeros_2d = torch.zeros(3, 4)
zeros_3d = torch.zeros(2, 3, 4)
print(f"Zeros 1D: {zeros_1d}")
print(f"Zeros 2D shape: {zeros_2d.shape}")
print(f"Zeros 3D shape: {zeros_3d.shape}")

# Ones
ones_1d = torch.ones(5)
ones_2d = torch.ones(3, 4)
print(f"Ones 1D: {ones_1d}")
print(f"Ones 2D shape: {ones_2d.shape}")

# Zeros/ones like existing tensor
sample_tensor = torch.randn(2, 3)
zeros_like = torch.zeros_like(sample_tensor)
ones_like = torch.ones_like(sample_tensor)
print(f"Zeros like shape: {zeros_like.shape}")
print(f"Ones like shape: {ones_like.shape}")

print("\n=== Full and Empty Tensors ===")

# Full with specific value
full_tensor = torch.full((3, 4), 7.5)
print(f"Full tensor (7.5): {full_tensor}")

# Empty (uninitialized)
empty_tensor = torch.empty(2, 3)
print(f"Empty tensor shape: {empty_tensor.shape}")

# Full like existing tensor
full_like = torch.full_like(sample_tensor, 3.14)
print(f"Full like (3.14): {full_like}")

print("\n=== Identity and Diagonal Tensors ===")

# Identity matrix
eye_3x3 = torch.eye(3)
eye_4x6 = torch.eye(4, 6)
print(f"Identity 3x3:\n{eye_3x3}")
print(f"Identity 4x6 shape: {eye_4x6.shape}")

# Diagonal matrix
diag_values = torch.tensor([1, 2, 3, 4])
diag_matrix = torch.diag(diag_values)
print(f"Diagonal matrix:\n{diag_matrix}")

print("\n=== Range and Sequence Tensors ===")

# Arange
arange_basic = torch.arange(10)
arange_start_stop = torch.arange(2, 10)
arange_step = torch.arange(0, 10, 2)
arange_float = torch.arange(0.0, 5.0, 0.5)
print(f"Arange basic: {arange_basic}")
print(f"Arange start-stop: {arange_start_stop}")
print(f"Arange with step: {arange_step}")
print(f"Arange float: {arange_float}")

# Linspace
linspace_basic = torch.linspace(0, 10, 5)
linspace_detailed = torch.linspace(-1, 1, 11)
print(f"Linspace basic: {linspace_basic}")
print(f"Linspace detailed: {linspace_detailed}")

# Logspace
logspace_tensor = torch.logspace(0, 3, 4)
print(f"Logspace: {logspace_tensor}")

print("\n=== Random Tensors ===")

# Random uniform [0, 1)
rand_uniform = torch.rand(2, 3)
print(f"Random uniform shape: {rand_uniform.shape}")

# Random normal (mean=0, std=1)
randn_normal = torch.randn(2, 3)
print(f"Random normal shape: {randn_normal.shape}")

# Random integers
randint_tensor = torch.randint(0, 10, (3, 4))
print(f"Random int [0,10): {randint_tensor}")

# Random permutation
randperm_tensor = torch.randperm(10)
print(f"Random permutation: {randperm_tensor}")

print("\n=== Specific Data Types ===")

# Integer tensors
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
long_tensor = torch.tensor([1, 2, 3], dtype=torch.long)
print(f"Int32 tensor: {int_tensor}, dtype: {int_tensor.dtype}")
print(f"Long tensor: {long_tensor}, dtype: {long_tensor.dtype}")

# Float tensors
float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
double_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.double)
print(f"Float32 tensor: {float_tensor}, dtype: {float_tensor.dtype}")
print(f"Double tensor: {double_tensor}, dtype: {double_tensor.dtype}")

# Boolean tensors
bool_tensor = torch.tensor([True, False, True], dtype=torch.bool)
print(f"Boolean tensor: {bool_tensor}, dtype: {bool_tensor.dtype}")

print("\n=== Complex Numbers ===")

# Complex tensors
complex_tensor = torch.tensor([1+2j, 3+4j], dtype=torch.complex64)
print(f"Complex tensor: {complex_tensor}, dtype: {complex_tensor.dtype}")

# From real and imaginary parts
real_part = torch.tensor([1.0, 3.0])
imag_part = torch.tensor([2.0, 4.0])
complex_from_parts = torch.complex(real_part, imag_part)
print(f"Complex from parts: {complex_from_parts}")

print("\n=== Special Creation Methods ===")

# From file (simulated)
data_for_file = torch.randn(3, 4)
torch.save(data_for_file, 'temp_tensor.pt')
loaded_tensor = torch.load('temp_tensor.pt')
print(f"Loaded tensor shape: {loaded_tensor.shape}")

# Meshgrid for coordinate tensors
x_coords = torch.arange(3)
y_coords = torch.arange(4)
grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')
print(f"Grid X shape: {grid_x.shape}")
print(f"Grid Y shape: {grid_y.shape}")

# Tensor from function
def custom_function(i, j):
    return i * 10 + j

tensor_from_func = torch.tensor([[custom_function(i, j) for j in range(4)] for i in range(3)])
print(f"Tensor from function:\n{tensor_from_func}")

print("\n=== Memory Layout and Device ===")

# Contiguous tensor
contiguous_tensor = torch.randn(2, 3, 4)
print(f"Contiguous: {contiguous_tensor.is_contiguous()}")

# Pinned memory (for faster GPU transfer)
pinned_tensor = torch.randn(2, 3).pin_memory()
print(f"Pinned memory: {pinned_tensor.is_pinned()}")

# GPU tensor (if available)
if torch.cuda.is_available():
    cuda_tensor = torch.randn(2, 3, device='cuda')
    print(f"CUDA tensor device: {cuda_tensor.device}")

# Cleanup
import os
if os.path.exists('temp_tensor.pt'):
    os.remove('temp_tensor.pt')

print("\n=== Tensor Creation Complete ===") 