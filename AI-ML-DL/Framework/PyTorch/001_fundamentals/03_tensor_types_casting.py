#!/usr/bin/env python3
"""PyTorch Tensor Types and Casting Operations"""

import torch
import numpy as np

print("=== PyTorch Data Types ===")

# All PyTorch data types
data_types = {
    'torch.bool': torch.bool,
    'torch.uint8': torch.uint8,
    'torch.int8': torch.int8,
    'torch.int16': torch.int16,
    'torch.int32': torch.int32,
    'torch.int64': torch.int64,
    'torch.float16': torch.float16,
    'torch.float32': torch.float32,
    'torch.float64': torch.float64,
    'torch.complex64': torch.complex64,
    'torch.complex128': torch.complex128,
}

for name, dtype in data_types.items():
    sample = torch.tensor([1, 2, 3], dtype=dtype)
    print(f"{name}: {sample.dtype}, size: {sample.element_size()} bytes")

print("\n=== Creating Tensors with Specific Types ===")

# Integer types
int8_tensor = torch.tensor([1, 2, 3], dtype=torch.int8)
int32_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
int64_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)

print(f"int8: {int8_tensor}, range: {torch.iinfo(torch.int8).min} to {torch.iinfo(torch.int8).max}")
print(f"int32: {int32_tensor}, range: {torch.iinfo(torch.int32).min} to {torch.iinfo(torch.int32).max}")
print(f"int64: {int64_tensor}, range: {torch.iinfo(torch.int64).min} to {torch.iinfo(torch.int64).max}")

# Float types
float16_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
float32_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
float64_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

print(f"float16: {float16_tensor}")
print(f"float32: {float32_tensor}")
print(f"float64: {float64_tensor}")

# Boolean type
bool_tensor = torch.tensor([True, False, True], dtype=torch.bool)
print(f"bool: {bool_tensor}")

print("\n=== Type Casting Methods ===")

# Base tensor for casting
base_tensor = torch.tensor([1.7, 2.3, 3.9])
print(f"Original: {base_tensor}, dtype: {base_tensor.dtype}")

# Method 1: .to() method
int_tensor = base_tensor.to(torch.int32)
print(f"to(int32): {int_tensor}")

long_tensor = base_tensor.to(torch.long)
print(f"to(long): {long_tensor}")

# Method 2: type() method
float_tensor = base_tensor.float()
double_tensor = base_tensor.double()
int_tensor2 = base_tensor.int()
long_tensor2 = base_tensor.long()
bool_tensor2 = base_tensor.bool()

print(f"float(): {float_tensor}")
print(f"double(): {double_tensor}")
print(f"int(): {int_tensor2}")
print(f"long(): {long_tensor2}")
print(f"bool(): {bool_tensor2}")

# Method 3: type_as() method
reference_tensor = torch.tensor([1, 2, 3], dtype=torch.int8)
casted_tensor = base_tensor.type_as(reference_tensor)
print(f"type_as(int8): {casted_tensor}, dtype: {casted_tensor.dtype}")

print("\n=== Automatic Type Promotion ===")

# Mixed operations - PyTorch promotes types automatically
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
float_tensor = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)

result = int_tensor + float_tensor
print(f"int32 + float32 = {result}, dtype: {result.dtype}")

# Type promotion hierarchy
uint8_tensor = torch.tensor([1, 2, 3], dtype=torch.uint8)
int64_tensor = torch.tensor([10, 20, 30], dtype=torch.int64)

result2 = uint8_tensor * int64_tensor
print(f"uint8 * int64 = {result2}, dtype: {result2.dtype}")

print("\n=== Complex Number Casting ===")

# Real to complex
real_tensor = torch.tensor([1.0, 2.0, 3.0])
complex_tensor = real_tensor.to(torch.complex64)
print(f"Real to complex64: {complex_tensor}")

# Complex to real (takes real part)
real_part = complex_tensor.real
imag_part = complex_tensor.imag
print(f"Real part: {real_part}")
print(f"Imaginary part: {imag_part}")

# Absolute value of complex
abs_complex = torch.abs(torch.tensor([1+2j, 3+4j]))
print(f"Absolute of complex: {abs_complex}")

print("\n=== Type Checking and Information ===")

sample_tensor = torch.randn(3, 4)

# Check data type
print(f"dtype: {sample_tensor.dtype}")
print(f"is_floating_point: {sample_tensor.is_floating_point()}")
print(f"is_complex: {sample_tensor.is_complex()}")
print(f"is_signed: {sample_tensor.dtype.is_signed}")

# Type information
print(f"finfo for float32: {torch.finfo(torch.float32)}")
print(f"iinfo for int32: {torch.iinfo(torch.int32)}")

print("\n=== Conditional Type Casting ===")

# Cast based on conditions
mixed_tensor = torch.tensor([1.2, 2.8, 3.1, 4.9])

# Round then cast to int
rounded_int = mixed_tensor.round().int()
print(f"Rounded to int: {rounded_int}")

# Floor and ceiling casts
floor_int = mixed_tensor.floor().int()
ceil_int = mixed_tensor.ceil().int()
print(f"Floor to int: {floor_int}")
print(f"Ceil to int: {ceil_int}")

# Truncate (towards zero)
trunc_int = mixed_tensor.trunc().int()
print(f"Truncate to int: {trunc_int}")

print("\n=== Memory-Efficient Casting ===")

large_tensor = torch.randn(1000, 1000)
print(f"Original memory: {large_tensor.element_size() * large_tensor.numel() / 1e6:.2f} MB")

# In-place casting (saves memory)
large_tensor_half = large_tensor.half()  # Convert to float16
print(f"Half precision memory: {large_tensor_half.element_size() * large_tensor_half.numel() / 1e6:.2f} MB")

# Check if types are compatible for operations
tensor_a = torch.tensor([1, 2, 3], dtype=torch.int32)
tensor_b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

print(f"Can broadcast: {torch.broadcast_tensors(tensor_a.float(), tensor_b)[0].shape}")

print("\n=== Special Casting Cases ===")

# NaN and infinity handling
special_values = torch.tensor([float('inf'), float('-inf'), float('nan')])
print(f"Special values: {special_values}")

# Cast to int (NaN becomes 0, inf becomes max int)
special_int = special_values.to(torch.int32)
print(f"Special to int32: {special_int}")

# Boolean casting
bool_from_numbers = torch.tensor([0, 1, 2, -1]).bool()
print(f"Numbers to bool: {bool_from_numbers}")

# String representation
tensor_for_str = torch.tensor([1.23456789, 2.87654321])
print(f"Default precision: {tensor_for_str}")

# Change print precision
torch.set_printoptions(precision=2)
print(f"Precision 2: {tensor_for_str}")
torch.set_printoptions(precision=4)  # Reset

print("\n=== NumPy Interoperability ===")

# PyTorch to NumPy
torch_tensor = torch.tensor([1.0, 2.0, 3.0])
numpy_array = torch_tensor.numpy()
print(f"PyTorch to NumPy: {numpy_array}, type: {type(numpy_array)}")

# NumPy to PyTorch
numpy_array2 = np.array([4.0, 5.0, 6.0])
torch_tensor2 = torch.from_numpy(numpy_array2)
print(f"NumPy to PyTorch: {torch_tensor2}, type: {type(torch_tensor2)}")

# Shared memory (changes reflect in both)
torch_tensor.add_(1)  # In-place operation
print(f"After in-place op - NumPy: {numpy_array}")  # Also changed

print("\n=== Type Casting Complete ===") 