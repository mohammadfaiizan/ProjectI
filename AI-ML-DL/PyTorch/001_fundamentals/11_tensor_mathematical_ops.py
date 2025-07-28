#!/usr/bin/env python3
"""PyTorch Mathematical Operations - All math functions"""

import torch
import math

print("=== Basic Arithmetic Operations ===")

# Create sample tensors
a = torch.tensor([1.0, 2.0, 3.0, 4.0])
b = torch.tensor([2.0, 3.0, 4.0, 5.0])
c = torch.tensor([[1, 2], [3, 4]])
d = torch.tensor([[5, 6], [7, 8]])

print(f"Tensor a: {a}")
print(f"Tensor b: {b}")

# Element-wise operations
addition = a + b
subtraction = a - b
multiplication = a * b
division = a / b
power = a ** 2
modulo = a % 3

print(f"Addition: {addition}")
print(f"Subtraction: {subtraction}")
print(f"Multiplication: {multiplication}")
print(f"Division: {division}")
print(f"Power: {power}")
print(f"Modulo: {modulo}")

# Floor division
floor_div = a // b
print(f"Floor division: {floor_div}")

print("\n=== Function-based Operations ===")

# Using function syntax
add_func = torch.add(a, b)
sub_func = torch.sub(a, b)
mul_func = torch.mul(a, b)
div_func = torch.div(a, b)

print(f"torch.add: {add_func}")
print(f"torch.sub: {sub_func}")
print(f"torch.mul: {mul_func}")
print(f"torch.div: {div_func}")

# With scalar
scalar_add = torch.add(a, 10)
scalar_mul = torch.mul(a, 0.5)
print(f"Add scalar: {scalar_add}")
print(f"Multiply scalar: {scalar_mul}")

print("\n=== Mathematical Functions ===")

# Trigonometric functions
angles = torch.tensor([0, math.pi/4, math.pi/2, math.pi])
sin_vals = torch.sin(angles)
cos_vals = torch.cos(angles)
tan_vals = torch.tan(angles)

print(f"Angles: {angles}")
print(f"Sin: {sin_vals}")
print(f"Cos: {cos_vals}")
print(f"Tan: {tan_vals}")

# Inverse trigonometric
sin_inverse = torch.asin(torch.tensor([0, 0.5, 1.0]))
cos_inverse = torch.acos(torch.tensor([1, 0.5, 0]))
tan_inverse = torch.atan(torch.tensor([0, 1, float('inf')]))

print(f"Arcsin: {sin_inverse}")
print(f"Arccos: {cos_inverse}")
print(f"Arctan: {tan_inverse}")

# Hyperbolic functions
values = torch.tensor([0, 1, 2])
sinh_vals = torch.sinh(values)
cosh_vals = torch.cosh(values)
tanh_vals = torch.tanh(values)

print(f"Sinh: {sinh_vals}")
print(f"Cosh: {cosh_vals}")
print(f"Tanh: {tanh_vals}")

print("\n=== Exponential and Logarithmic ===")

# Exponential functions
exp_vals = torch.exp(torch.tensor([0, 1, 2]))
exp2_vals = torch.exp2(torch.tensor([0, 1, 2]))
expm1_vals = torch.expm1(torch.tensor([0, 0.1, 0.5]))

print(f"Exp: {exp_vals}")
print(f"Exp2: {exp2_vals}")
print(f"Expm1: {expm1_vals}")

# Logarithmic functions
log_vals = torch.log(torch.tensor([1, math.e, 10]))
log10_vals = torch.log10(torch.tensor([1, 10, 100]))
log2_vals = torch.log2(torch.tensor([1, 2, 8]))
log1p_vals = torch.log1p(torch.tensor([0, 0.1, 1]))

print(f"Log: {log_vals}")
print(f"Log10: {log10_vals}")
print(f"Log2: {log2_vals}")
print(f"Log1p: {log1p_vals}")

print("\n=== Power and Root Functions ===")

# Power functions
base = torch.tensor([2.0, 3.0, 4.0])
exponent = torch.tensor([2.0, 3.0, 0.5])

pow_vals = torch.pow(base, exponent)
sqrt_vals = torch.sqrt(torch.tensor([4, 9, 16]))
rsqrt_vals = torch.rsqrt(torch.tensor([4, 9, 16]))  # 1/sqrt(x)

print(f"Power: {pow_vals}")
print(f"Square root: {sqrt_vals}")
print(f"Reciprocal sqrt: {rsqrt_vals}")

# Square and cube
square_vals = torch.square(base)
# cube_vals = torch.pow(base, 3)  # No direct cube function

print(f"Square: {square_vals}")

print("\n=== Rounding and Ceiling ===")

# Rounding functions
float_vals = torch.tensor([-2.7, -1.3, 0.8, 1.5, 2.9])

floor_vals = torch.floor(float_vals)
ceil_vals = torch.ceil(float_vals)
round_vals = torch.round(float_vals)
trunc_vals = torch.trunc(float_vals)

print(f"Original: {float_vals}")
print(f"Floor: {floor_vals}")
print(f"Ceil: {ceil_vals}")
print(f"Round: {round_vals}")
print(f"Trunc: {trunc_vals}")

# Fractional part
frac_vals = torch.frac(float_vals)
print(f"Fractional: {frac_vals}")

print("\n=== Absolute and Sign ===")

# Absolute and sign functions
signed_vals = torch.tensor([-3, -1, 0, 2, 5])

abs_vals = torch.abs(signed_vals)
sign_vals = torch.sign(signed_vals)
sgn_vals = torch.sgn(signed_vals)  # Similar to sign but with complex support

print(f"Original: {signed_vals}")
print(f"Absolute: {abs_vals}")
print(f"Sign: {sign_vals}")
print(f"Sgn: {sgn_vals}")

# Clamp (clip) values
clamped = torch.clamp(signed_vals, min=-2, max=3)
print(f"Clamped [-2, 3]: {clamped}")

print("\n=== Linear Algebra Operations ===")

# Matrix multiplication
matrix_a = torch.randn(3, 4)
matrix_b = torch.randn(4, 5)

matmul_result = torch.matmul(matrix_a, matrix_b)
mm_result = torch.mm(matrix_a, matrix_b)

print(f"Matrix A shape: {matrix_a.shape}")
print(f"Matrix B shape: {matrix_b.shape}")
print(f"Matmul result shape: {matmul_result.shape}")
print(f"MM result equal: {torch.equal(matmul_result, mm_result)}")

# Dot product
vec1 = torch.tensor([1, 2, 3])
vec2 = torch.tensor([4, 5, 6])
dot_result = torch.dot(vec1, vec2)
print(f"Dot product: {dot_result}")

# Cross product
cross_result = torch.cross(vec1, vec2)
print(f"Cross product: {cross_result}")

print("\n=== Statistical Functions ===")

# Statistical operations (will be detailed in reduction operations file)
data = torch.randn(4, 4)

mean_val = torch.mean(data)
std_val = torch.std(data)
var_val = torch.var(data)

print(f"Mean: {mean_val}")
print(f"Std: {std_val}")
print(f"Variance: {var_val}")

print("\n=== Complex Number Operations ===")

# Complex tensor operations
complex_tensor = torch.tensor([1+2j, 3+4j, 5-6j])

real_part = torch.real(complex_tensor)
imag_part = torch.imag(complex_tensor)
abs_complex = torch.abs(complex_tensor)
angle_complex = torch.angle(complex_tensor)

print(f"Complex: {complex_tensor}")
print(f"Real part: {real_part}")
print(f"Imaginary part: {imag_part}")
print(f"Absolute: {abs_complex}")
print(f"Angle: {angle_complex}")

# Complex conjugate
conj_complex = torch.conj(complex_tensor)
print(f"Conjugate: {conj_complex}")

print("\n=== Element-wise Functions ===")

# Various element-wise operations
positive_vals = torch.tensor([0.1, 0.5, 1.0, 2.0])

reciprocal = torch.reciprocal(positive_vals)
neg_vals = torch.neg(positive_vals)
sigmoid_vals = torch.sigmoid(positive_vals)

print(f"Original: {positive_vals}")
print(f"Reciprocal: {reciprocal}")
print(f"Negative: {neg_vals}")
print(f"Sigmoid: {sigmoid_vals}")

print("\n=== In-place Operations ===")

# In-place mathematical operations
inplace_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(f"Before in-place: {inplace_tensor}")

inplace_tensor.add_(1)  # Add 1 in-place
print(f"After add_(1): {inplace_tensor}")

inplace_tensor.mul_(2)  # Multiply by 2 in-place
print(f"After mul_(2): {inplace_tensor}")

inplace_tensor.div_(2)  # Divide by 2 in-place
print(f"After div_(2): {inplace_tensor}")

inplace_tensor.sqrt_()  # Square root in-place
print(f"After sqrt_(): {inplace_tensor}")

print("\n=== Broadcasting in Math Operations ===")

# Broadcasting examples
scalar = 5
vector = torch.tensor([1, 2, 3])
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Scalar with vector
scalar_vector = scalar + vector
print(f"Scalar + vector: {scalar_vector}")

# Vector with matrix (broadcasting)
vector_matrix = vector + matrix
print(f"Vector + matrix:\n{vector_matrix}")

# Different shaped tensors
tensor_2x1 = torch.tensor([[1], [2]])
tensor_1x3 = torch.tensor([[1, 2, 3]])
broadcast_result = tensor_2x1 + tensor_1x3
print(f"Broadcast (2x1) + (1x3):\n{broadcast_result}")

print("\n=== Error Handling ===")

# Division by zero
try:
    div_by_zero = torch.tensor([1.0, 2.0]) / torch.tensor([0.0, 1.0])
    print(f"Division by zero result: {div_by_zero}")
except:
    print("Division by zero handled")

# Invalid operations
try:
    invalid_log = torch.log(torch.tensor([-1.0, 0.0, 1.0]))
    print(f"Log of negative/zero: {invalid_log}")
except:
    print("Invalid log handled")

# Check for NaN and infinity
result_with_nan = torch.tensor([1.0, float('nan'), float('inf')])
has_nan = torch.isnan(result_with_nan)
has_inf = torch.isinf(result_with_nan)
is_finite = torch.isfinite(result_with_nan)

print(f"Tensor with special values: {result_with_nan}")
print(f"Has NaN: {has_nan}")
print(f"Has Inf: {has_inf}")
print(f"Is finite: {is_finite}")

print("\n=== Mathematical Operations Complete ===") 