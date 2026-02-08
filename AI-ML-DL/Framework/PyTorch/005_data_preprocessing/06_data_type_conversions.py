#!/usr/bin/env python3
"""PyTorch Data Type Conversions - Converting between data types"""

import torch
import numpy as np

print("=== Data Type Conversions Overview ===")

print("PyTorch data types:")
print("1. Integer types: uint8, int8, int16, int32, int64")
print("2. Floating point: float16, float32, float64")
print("3. Boolean: bool")
print("4. Complex: complex64, complex128")
print("5. Special considerations for GPU/CPU transfers")

print("\n=== Basic Type Conversions ===")

# Create tensors with different types
int_tensor = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
float_tensor = torch.tensor([1.1, 2.2, 3.3, 4.4], dtype=torch.float32)
bool_tensor = torch.tensor([True, False, True, False], dtype=torch.bool)

print(f"Int tensor: {int_tensor}, dtype: {int_tensor.dtype}")
print(f"Float tensor: {float_tensor}, dtype: {float_tensor.dtype}")
print(f"Bool tensor: {bool_tensor}, dtype: {bool_tensor.dtype}")

# Type conversion methods
int_to_float = int_tensor.float()
float_to_int = float_tensor.int()
int_to_bool = int_tensor.bool()
bool_to_float = bool_tensor.float()

print(f"\nInt to float: {int_to_float}, dtype: {int_to_float.dtype}")
print(f"Float to int: {float_to_int}, dtype: {float_to_int.dtype}")
print(f"Int to bool: {int_to_bool}, dtype: {int_to_bool.dtype}")
print(f"Bool to float: {bool_to_float}, dtype: {bool_to_float.dtype}")

# Using .to() method
converted_to_double = int_tensor.to(torch.float64)
converted_to_long = float_tensor.to(torch.int64)

print(f"To double: {converted_to_double}, dtype: {converted_to_double.dtype}")
print(f"To long: {converted_to_long}, dtype: {converted_to_long.dtype}")

print("\n=== Precision Conversions ===")

# Different floating point precisions
data = torch.randn(3, 3)
print(f"Original (float32): {data.dtype}, memory: {data.element_size() * data.numel()} bytes")

# Convert to different precisions
data_fp16 = data.half()  # float16
data_fp64 = data.double()  # float64

print(f"Half precision: {data_fp16.dtype}, memory: {data_fp16.element_size() * data_fp16.numel()} bytes")
print(f"Double precision: {data_fp64.dtype}, memory: {data_fp64.element_size() * data_fp64.numel()} bytes")

# Precision loss demonstration
precise_value = torch.tensor(3.141592653589793)
fp16_value = precise_value.half()
back_to_fp32 = fp16_value.float()

print(f"Original: {precise_value}")
print(f"FP16: {fp16_value}")
print(f"Back to FP32: {back_to_fp32}")
print(f"Precision loss: {abs(precise_value - back_to_fp32).item()}")

print("\n=== Integer Type Conversions ===")

# Different integer types
large_int = torch.tensor([32767, -32768, 65535, 0])
print(f"Large integers: {large_int}")

# Convert to different integer types
int8_tensor = large_int.to(torch.int8)
uint8_tensor = large_int.to(torch.uint8)
int16_tensor = large_int.to(torch.int16)

print(f"As int8: {int8_tensor}")  # Overflow/underflow
print(f"As uint8: {uint8_tensor}")  # Overflow/underflow
print(f"As int16: {int16_tensor}")

# Safe conversion with clamping
def safe_int_convert(tensor, target_dtype):
    """Safely convert to integer type with clamping"""
    if target_dtype == torch.int8:
        min_val, max_val = -128, 127
    elif target_dtype == torch.uint8:
        min_val, max_val = 0, 255
    elif target_dtype == torch.int16:
        min_val, max_val = -32768, 32767
    elif target_dtype == torch.int32:
        min_val, max_val = -2147483648, 2147483647
    else:
        return tensor.to(target_dtype)
    
    clamped = torch.clamp(tensor, min_val, max_val)
    return clamped.to(target_dtype)

safe_int8 = safe_int_convert(large_int, torch.int8)
safe_uint8 = safe_int_convert(large_int, torch.uint8)

print(f"Safe int8: {safe_int8}")
print(f"Safe uint8: {safe_uint8}")

print("\n=== NumPy Interoperability ===")

# PyTorch to NumPy
torch_tensor = torch.randn(3, 3)
numpy_array = torch_tensor.numpy()

print(f"PyTorch tensor: {torch_tensor.dtype}")
print(f"NumPy array: {numpy_array.dtype}")

# NumPy to PyTorch
numpy_int_array = np.array([1, 2, 3, 4], dtype=np.int64)
torch_from_numpy = torch.from_numpy(numpy_int_array)

print(f"NumPy int64: {numpy_int_array.dtype}")
print(f"PyTorch from NumPy: {torch_from_numpy.dtype}")

# Different NumPy types
numpy_types = [
    (np.float32, torch.float32),
    (np.float64, torch.float64),
    (np.int32, torch.int32),
    (np.uint8, torch.uint8),
    (np.bool_, torch.bool)
]

for np_type, expected_torch in numpy_types:
    np_array = np.array([1, 2, 3], dtype=np_type)
    torch_tensor = torch.from_numpy(np_array)
    print(f"NumPy {np_type} -> PyTorch {torch_tensor.dtype} (expected: {expected_torch})")

print("\n=== Complex Number Conversions ===")

# Complex tensors
real_part = torch.randn(2, 2)
imag_part = torch.randn(2, 2)
complex_tensor = torch.complex(real_part, imag_part)

print(f"Complex tensor: {complex_tensor}")
print(f"Complex dtype: {complex_tensor.dtype}")

# Extract real and imaginary parts
extracted_real = complex_tensor.real
extracted_imag = complex_tensor.imag

print(f"Extracted real: {extracted_real}")
print(f"Extracted imag: {extracted_imag}")

# Complex to real conversions
magnitude = torch.abs(complex_tensor)
phase = torch.angle(complex_tensor)

print(f"Magnitude: {magnitude}")
print(f"Phase: {phase}")

print("\n=== Device and Type Conversions ===")

# CPU tensor
cpu_tensor = torch.randn(3, 3)
print(f"CPU tensor device: {cpu_tensor.device}")

# GPU conversion (if available)
if torch.cuda.is_available():
    gpu_tensor = cpu_tensor.cuda()
    print(f"GPU tensor device: {gpu_tensor.device}")
    
    # Back to CPU
    cpu_again = gpu_tensor.cpu()
    print(f"Back to CPU: {cpu_again.device}")
    
    # Combined device and type conversion
    gpu_half = cpu_tensor.cuda().half()
    print(f"GPU half precision: {gpu_half.device}, {gpu_half.dtype}")
else:
    print("CUDA not available, skipping GPU conversions")

# Using .to() for device and type
target_device = 'cuda' if torch.cuda.is_available() else 'cpu'
converted_tensor = cpu_tensor.to(device=target_device, dtype=torch.float16)
print(f"Converted tensor: {converted_tensor.device}, {converted_tensor.dtype}")

print("\n=== Batch Type Conversions ===")

def convert_batch_types(batch_dict, type_mapping):
    """Convert types for a batch of tensors"""
    converted_batch = {}
    
    for key, tensor in batch_dict.items():
        if key in type_mapping:
            target_type = type_mapping[key]
            converted_batch[key] = tensor.to(target_type)
        else:
            converted_batch[key] = tensor
    
    return converted_batch

# Sample batch
batch_data = {
    'images': torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8),
    'labels': torch.randint(0, 10, (4,), dtype=torch.int64),
    'weights': torch.randn(4, dtype=torch.float64)
}

# Type mapping for conversion
type_map = {
    'images': torch.float32,
    'labels': torch.int64,  # Keep as is
    'weights': torch.float32
}

converted_batch = convert_batch_types(batch_data, type_map)

print("Original batch types:")
for key, tensor in batch_data.items():
    print(f"  {key}: {tensor.dtype}")

print("Converted batch types:")
for key, tensor in converted_batch.items():
    print(f"  {key}: {tensor.dtype}")

print("\n=== Memory-Efficient Conversions ===")

def efficient_type_conversion(tensor, target_dtype, inplace=False):
    """Memory-efficient type conversion"""
    if tensor.dtype == target_dtype:
        return tensor
    
    if inplace and tensor.dtype.is_floating_point and target_dtype.is_floating_point:
        # In-place conversion for floating point types
        tensor.data = tensor.data.to(target_dtype)
        return tensor
    else:
        # Regular conversion
        return tensor.to(target_dtype)

# Test efficient conversion
large_tensor = torch.randn(1000, 1000)
print(f"Original memory: {large_tensor.element_size() * large_tensor.numel() / 1e6:.2f} MB")

# Convert to half precision
half_tensor = efficient_type_conversion(large_tensor, torch.float16)
print(f"Half precision memory: {half_tensor.element_size() * half_tensor.numel() / 1e6:.2f} MB")

print("\n=== Type Checking and Validation ===")

def validate_tensor_types(tensors, expected_types):
    """Validate tensor types match expectations"""
    validation_results = {}
    
    for name, tensor in tensors.items():
        expected = expected_types.get(name)
        if expected is None:
            validation_results[name] = "No expectation set"
        elif tensor.dtype == expected:
            validation_results[name] = "✓ Correct type"
        else:
            validation_results[name] = f"✗ Expected {expected}, got {tensor.dtype}"
    
    return validation_results

# Test validation
test_tensors = {
    'input': torch.randn(10, dtype=torch.float32),
    'target': torch.randint(0, 5, (10,), dtype=torch.int64),
    'mask': torch.ones(10, dtype=torch.bool)
}

expected_types = {
    'input': torch.float32,
    'target': torch.int64,
    'mask': torch.bool
}

validation = validate_tensor_types(test_tensors, expected_types)
print("Type validation results:")
for name, result in validation.items():
    print(f"  {name}: {result}")

print("\n=== Automatic Type Promotion ===")

# Demonstrate automatic type promotion
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
float_tensor = torch.tensor([1.1, 2.2, 3.3], dtype=torch.float32)

# Operations cause type promotion
result = int_tensor + float_tensor
print(f"int32 + float32 = {result.dtype}")

# Different promotions
promotions = [
    (torch.uint8, torch.int8, torch.int8),
    (torch.int8, torch.int16, torch.int16),
    (torch.int32, torch.float32, torch.float32),
    (torch.float32, torch.float64, torch.float64),
]

for type1, type2, expected in promotions:
    t1 = torch.tensor([1], dtype=type1)
    t2 = torch.tensor([2], dtype=type2)
    result = t1 + t2
    print(f"{type1} + {type2} = {result.dtype} (expected: {expected})")

print("\n=== Custom Type Conversion Functions ===")

def smart_convert(tensor, target_type, handle_overflow='clamp'):
    """Smart type conversion with overflow handling"""
    if tensor.dtype == target_type:
        return tensor
    
    # Handle integer overflow
    if tensor.dtype.is_floating_point and target_type in [torch.int8, torch.uint8, torch.int16, torch.int32]:
        if handle_overflow == 'clamp':
            if target_type == torch.int8:
                tensor = torch.clamp(tensor, -128, 127)
            elif target_type == torch.uint8:
                tensor = torch.clamp(tensor, 0, 255)
            elif target_type == torch.int16:
                tensor = torch.clamp(tensor, -32768, 32767)
            elif target_type == torch.int32:
                tensor = torch.clamp(tensor, -2147483648, 2147483647)
        elif handle_overflow == 'round':
            tensor = torch.round(tensor)
    
    return tensor.to(target_type)

def normalize_and_convert(tensor, target_type):
    """Normalize then convert (useful for images)"""
    if tensor.dtype == torch.uint8 and target_type.is_floating_point:
        return tensor.float() / 255.0
    elif tensor.dtype.is_floating_point and target_type == torch.uint8:
        return (tensor * 255).clamp(0, 255).to(torch.uint8)
    else:
        return tensor.to(target_type)

# Test custom conversions
test_data = torch.tensor([256.7, -129.3, 127.8, 0.5])
converted = smart_convert(test_data, torch.int8, handle_overflow='clamp')
print(f"Smart convert to int8: {converted}")

# Test image normalization
image_uint8 = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
normalized = normalize_and_convert(image_uint8, torch.float32)
back_to_uint8 = normalize_and_convert(normalized, torch.uint8)

print(f"Image uint8 range: [{image_uint8.min()}, {image_uint8.max()}]")
print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
print(f"Back to uint8 range: [{back_to_uint8.min()}, {back_to_uint8.max()}]")

print("\n=== Type Conversion Best Practices ===")

print("Data Type Guidelines:")
print("1. Use float32 for most neural network computations")
print("2. Use float16 for memory savings (check for numerical stability)")
print("3. Use int64 for indices and labels")
print("4. Use uint8 for image data storage")
print("5. Be aware of automatic type promotion in operations")
print("6. Handle overflow/underflow explicitly when converting")
print("7. Use appropriate precision for your use case")

print("\nPerformance Considerations:")
print("- float16 saves memory but may be slower on some hardware")
print("- int8 quantization for inference speedup")
print("- Consider mixed precision training")
print("- GPU tensor types should match for operations")

print("\nCommon Pitfalls:")
print("- Silent overflow in integer conversions")
print("- Precision loss in float16")
print("- Device mismatches after conversion")
print("- Automatic promotion changing expected types")

print("\n=== Data Type Conversions Complete ===") 