#!/usr/bin/env python3
"""PyTorch Comparison and Logical Operations"""

import torch

print("=== Element-wise Comparison Operations ===")

# Create sample tensors for comparison
a = torch.tensor([1, 2, 3, 4, 5])
b = torch.tensor([2, 2, 3, 3, 6])
c = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
d = torch.tensor([[1.5, 1.5], [3.0, 5.0]])

print(f"Tensor a: {a}")
print(f"Tensor b: {b}")

# Equality comparisons
equal_result = torch.eq(a, b)
equal_operator = (a == b)
not_equal = torch.ne(a, b)
not_equal_operator = (a != b)

print(f"a == b: {equal_result}")
print(f"Using operator: {equal_operator}")
print(f"a != b: {not_equal}")
print(f"Results equal: {torch.equal(equal_result, equal_operator)}")

print("\n=== Inequality Comparisons ===")

# Greater than / less than
greater_than = torch.gt(a, b)
greater_than_op = (a > b)
greater_equal = torch.ge(a, b)
greater_equal_op = (a >= b)

less_than = torch.lt(a, b)
less_than_op = (a < b)
less_equal = torch.le(a, b)
less_equal_op = (a <= b)

print(f"a > b: {greater_than}")
print(f"a >= b: {greater_equal}")
print(f"a < b: {less_than}")
print(f"a <= b: {less_equal}")

# Verify operators match functions
print(f"gt matches >: {torch.equal(greater_than, greater_than_op)}")
print(f"ge matches >=: {torch.equal(greater_equal, greater_equal_op)}")

print("\n=== Floating Point Comparisons ===")

# Floating point comparison issues
float_a = torch.tensor([0.1 + 0.2, 0.3])
float_b = torch.tensor([0.3, 0.3])

print(f"Float a: {float_a}")
print(f"Float b: {float_b}")
print(f"Direct equality: {float_a == float_b}")

# Using allclose for floating point comparison
close_comparison = torch.allclose(float_a, float_b, rtol=1e-5, atol=1e-8)
print(f"Using allclose: {close_comparison}")

# isclose for element-wise close comparison
element_close = torch.isclose(float_a, float_b, rtol=1e-5, atol=1e-8)
print(f"Element-wise close: {element_close}")

print("\n=== Logical Operations ===")

# Boolean tensors for logical operations
bool_a = torch.tensor([True, False, True, False])
bool_b = torch.tensor([True, True, False, False])

print(f"Boolean a: {bool_a}")
print(f"Boolean b: {bool_b}")

# Logical AND
logical_and = torch.logical_and(bool_a, bool_b)
and_operator = bool_a & bool_b

print(f"Logical AND: {logical_and}")
print(f"& operator: {and_operator}")
print(f"Results equal: {torch.equal(logical_and, and_operator)}")

# Logical OR
logical_or = torch.logical_or(bool_a, bool_b)
or_operator = bool_a | bool_b

print(f"Logical OR: {logical_or}")
print(f"| operator: {or_operator}")

# Logical XOR
logical_xor = torch.logical_xor(bool_a, bool_b)
xor_operator = bool_a ^ bool_b

print(f"Logical XOR: {logical_xor}")
print(f"^ operator: {xor_operator}")

# Logical NOT
logical_not = torch.logical_not(bool_a)
not_operator = ~bool_a

print(f"Logical NOT: {logical_not}")
print(f"~ operator: {not_operator}")

print("\n=== Chained Comparisons ===")

# Multiple condition combinations
values = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Values between 3 and 7 (inclusive)
between_3_7 = (values >= 3) & (values <= 7)
print(f"Values: {values}")
print(f"Between 3 and 7: {between_3_7}")
print(f"Selected values: {values[between_3_7]}")

# Values less than 3 OR greater than 8
outside_range = (values < 3) | (values > 8)
print(f"Outside 3-8 range: {outside_range}")
print(f"Selected values: {values[outside_range]}")

# Multiple conditions with logical operations
even_and_gt_5 = ((values % 2) == 0) & (values > 5)
print(f"Even and > 5: {even_and_gt_5}")
print(f"Selected values: {values[even_and_gt_5]}")

print("\n=== Special Value Comparisons ===")

# NaN and infinity handling
special_values = torch.tensor([1.0, float('nan'), float('inf'), float('-inf'), 0.0])

print(f"Special values: {special_values}")

# Check for NaN
is_nan = torch.isnan(special_values)
print(f"Is NaN: {is_nan}")

# Check for infinity
is_inf = torch.isinf(special_values)
is_posinf = torch.isposinf(special_values)
is_neginf = torch.isneginf(special_values)

print(f"Is infinite: {is_inf}")
print(f"Is positive infinite: {is_posinf}")
print(f"Is negative infinite: {is_neginf}")

# Check for finite values
is_finite = torch.isfinite(special_values)
print(f"Is finite: {is_finite}")

# NaN comparisons always return False
nan_comparison = special_values == float('nan')
print(f"Comparison with NaN: {nan_comparison}")

print("\n=== Tensor-wise Comparisons ===")

# Whole tensor comparisons
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[1, 2], [3, 4]])
tensor3 = torch.tensor([[1, 2], [3, 5]])

# torch.equal() - exact equality
exact_equal = torch.equal(tensor1, tensor2)
not_exact_equal = torch.equal(tensor1, tensor3)

print(f"Tensor1:\n{tensor1}")
print(f"Tensor2:\n{tensor2}")
print(f"Tensor3:\n{tensor3}")
print(f"tensor1 equals tensor2: {exact_equal}")
print(f"tensor1 equals tensor3: {not_exact_equal}")

# allclose() - approximate equality
float_tensor1 = torch.tensor([[1.0, 2.0001], [3.0, 4.0]])
float_tensor2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

approx_equal = torch.allclose(float_tensor1, float_tensor2, rtol=1e-3)
print(f"Approximate equality (rtol=1e-3): {approx_equal}")

print("\n=== Sorting and Ranking ===")

# Sort and get indices
unsorted = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6])
sorted_vals, sorted_indices = torch.sort(unsorted)
sorted_desc, sorted_desc_indices = torch.sort(unsorted, descending=True)

print(f"Unsorted: {unsorted}")
print(f"Sorted: {sorted_vals}")
print(f"Sort indices: {sorted_indices}")
print(f"Sorted descending: {sorted_desc}")

# Argsort - get sorting indices only
argsort_indices = torch.argsort(unsorted)
print(f"Argsort indices: {argsort_indices}")

# Top-k largest/smallest values
topk_values, topk_indices = torch.topk(unsorted, k=3)
topk_smallest_values, topk_smallest_indices = torch.topk(unsorted, k=3, largest=False)

print(f"Top 3 largest: {topk_values}")
print(f"Top 3 smallest: {topk_smallest_values}")

print("\n=== Conditional Selection ===")

# torch.where() - conditional selection
condition = torch.tensor([True, False, True, False, True])
x_vals = torch.tensor([1, 2, 3, 4, 5])
y_vals = torch.tensor([10, 20, 30, 40, 50])

where_result = torch.where(condition, x_vals, y_vals)
print(f"Condition: {condition}")
print(f"X values: {x_vals}")
print(f"Y values: {y_vals}")
print(f"Where result: {where_result}")

# Conditional selection with scalars
where_scalar = torch.where(x_vals > 3, x_vals, 0)
print(f"Where x > 3, keep x, else 0: {where_scalar}")

# Multiple conditions
matrix = torch.randn(3, 4)
positive_mask = matrix > 0
negative_mask = matrix < 0

selected = torch.where(positive_mask, matrix, torch.where(negative_mask, -matrix, torch.zeros_like(matrix)))
print(f"Conditional selection with multiple conditions shape: {selected.shape}")

print("\n=== Masking and Filtering ===")

# Boolean indexing
data = torch.tensor([1, -2, 3, -4, 5, -6, 7, -8])
positive_mask = data > 0

positive_values = data[positive_mask]
print(f"Original data: {data}")
print(f"Positive mask: {positive_mask}")
print(f"Positive values: {positive_values}")

# Masked select
masked_selected = torch.masked_select(data, positive_mask)
print(f"Masked select result: {masked_selected}")

# Complex masking
matrix_data = torch.randn(4, 4)
complex_mask = (torch.abs(matrix_data) > 0.5) & (matrix_data > 0)
complex_selected = matrix_data[complex_mask]
print(f"Complex mask selection count: {complex_selected.numel()}")

print("\n=== Set Operations ===")

# Unique values
tensor_with_duplicates = torch.tensor([1, 2, 2, 3, 3, 3, 4, 4])
unique_values = torch.unique(tensor_with_duplicates)
unique_counts = torch.unique(tensor_with_duplicates, return_counts=True)
unique_inverse = torch.unique(tensor_with_duplicates, return_inverse=True)

print(f"With duplicates: {tensor_with_duplicates}")
print(f"Unique values: {unique_values}")
print(f"Unique with counts: {unique_counts}")
print(f"Unique with inverse: {unique_inverse}")

print("\n=== Tensor Comparison Functions ===")

# Maximum and minimum element-wise
tensor_x = torch.tensor([1, 5, 3, 9, 2])
tensor_y = torch.tensor([2, 3, 4, 8, 6])

element_max = torch.maximum(tensor_x, tensor_y)
element_min = torch.minimum(tensor_x, tensor_y)

print(f"Tensor X: {tensor_x}")
print(f"Tensor Y: {tensor_y}")
print(f"Element-wise max: {element_max}")
print(f"Element-wise min: {element_min}")

# Clamping values
clamp_result = torch.clamp(tensor_x, min=2, max=7)
print(f"Clamped [2, 7]: {clamp_result}")

# Clamp with tensors
min_vals = torch.tensor([1, 2, 3, 4, 5])
max_vals = torch.tensor([3, 4, 5, 6, 7])
clamp_tensor_result = torch.clamp(tensor_x, min=min_vals, max=max_vals)
print(f"Clamped with tensors: {clamp_tensor_result}")

print("\n=== Broadcasting in Comparisons ===")

# Broadcasting with comparisons
matrix_2d = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
vector_1d = torch.tensor([2, 5, 8])

broadcast_comparison = matrix_2d > vector_1d
print(f"Matrix > Vector (broadcast):\n{broadcast_comparison}")

# Broadcasting with logical operations
bool_matrix = torch.tensor([[True, False, True], [False, True, False]])
bool_vector = torch.tensor([True, False, True])

broadcast_and = bool_matrix & bool_vector
print(f"Boolean matrix & vector:\n{broadcast_and}")

print("\n=== Performance Considerations ===")

import time

# Performance comparison: function vs operator
large_tensor_a = torch.randn(10000)
large_tensor_b = torch.randn(10000)

# Using function
start_time = time.time()
func_result = torch.gt(large_tensor_a, large_tensor_b)
func_time = time.time() - start_time

# Using operator
start_time = time.time()
op_result = large_tensor_a > large_tensor_b
op_time = time.time() - start_time

print(f"Function time: {func_time:.6f} seconds")
print(f"Operator time: {op_time:.6f} seconds")
print(f"Results equal: {torch.equal(func_result, op_result)}")

print("\n=== Comparison and Logical Operations Complete ===") 