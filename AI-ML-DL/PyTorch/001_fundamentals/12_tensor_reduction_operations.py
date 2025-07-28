#!/usr/bin/env python3
"""PyTorch Tensor Reduction Operations - Sum, mean, max, min, etc."""

import torch

print("=== Basic Reduction Operations ===")

# Create sample tensors
tensor_2d = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
tensor_3d = torch.arange(24).float().reshape(2, 3, 4)

print(f"2D tensor:\n{tensor_2d}")
print(f"3D tensor shape: {tensor_3d.shape}")

# Sum operations
sum_all = torch.sum(tensor_2d)
sum_dim0 = torch.sum(tensor_2d, dim=0)
sum_dim1 = torch.sum(tensor_2d, dim=1)

print(f"Sum all: {sum_all}")
print(f"Sum dim 0 (columns): {sum_dim0}")
print(f"Sum dim 1 (rows): {sum_dim1}")

# Keep dimensions
sum_keepdim = torch.sum(tensor_2d, dim=1, keepdim=True)
print(f"Sum keepdim: {sum_keepdim.shape}")

print("\n=== Mean Operations ===")

# Mean operations
mean_all = torch.mean(tensor_2d)
mean_dim0 = torch.mean(tensor_2d, dim=0)
mean_dim1 = torch.mean(tensor_2d, dim=1)

print(f"Mean all: {mean_all}")
print(f"Mean dim 0: {mean_dim0}")
print(f"Mean dim 1: {mean_dim1}")

# Mean with keepdim
mean_keepdim = torch.mean(tensor_2d, dim=0, keepdim=True)
print(f"Mean keepdim shape: {mean_keepdim.shape}")

print("\n=== Min and Max Operations ===")

# Min and max
min_all = torch.min(tensor_2d)
max_all = torch.max(tensor_2d)

print(f"Min all: {min_all}")
print(f"Max all: {max_all}")

# Min and max with dimension (returns values and indices)
min_dim0_vals, min_dim0_indices = torch.min(tensor_2d, dim=0)
max_dim1_vals, max_dim1_indices = torch.max(tensor_2d, dim=1)

print(f"Min dim 0 values: {min_dim0_vals}")
print(f"Min dim 0 indices: {min_dim0_indices}")
print(f"Max dim 1 values: {max_dim1_vals}")
print(f"Max dim 1 indices: {max_dim1_indices}")

# Argmin and argmax
argmin_all = torch.argmin(tensor_2d)
argmax_all = torch.argmax(tensor_2d)
argmin_dim0 = torch.argmin(tensor_2d, dim=0)
argmax_dim1 = torch.argmax(tensor_2d, dim=1)

print(f"Argmin all: {argmin_all}")
print(f"Argmax all: {argmax_all}")
print(f"Argmin dim 0: {argmin_dim0}")
print(f"Argmax dim 1: {argmax_dim1}")

print("\n=== Product Operations ===")

# Product operations
prod_all = torch.prod(tensor_2d)
prod_dim0 = torch.prod(tensor_2d, dim=0)
prod_dim1 = torch.prod(tensor_2d, dim=1)

print(f"Product all: {prod_all}")
print(f"Product dim 0: {prod_dim0}")
print(f"Product dim 1: {prod_dim1}")

# Cumulative operations
cumsum_dim0 = torch.cumsum(tensor_2d, dim=0)
cumsum_dim1 = torch.cumsum(tensor_2d, dim=1)
cumprod_dim1 = torch.cumprod(tensor_2d, dim=1)

print(f"Cumsum dim 0:\n{cumsum_dim0}")
print(f"Cumsum dim 1:\n{cumsum_dim1}")
print(f"Cumprod dim 1:\n{cumprod_dim1}")

print("\n=== Statistical Reductions ===")

# Variance and standard deviation
var_all = torch.var(tensor_2d)
std_all = torch.std(tensor_2d)
var_dim0 = torch.var(tensor_2d, dim=0)
std_dim1 = torch.std(tensor_2d, dim=1)

print(f"Variance all: {var_all}")
print(f"Std all: {std_all}")
print(f"Variance dim 0: {var_dim0}")
print(f"Std dim 1: {std_dim1}")

# Unbiased variance and std
var_unbiased = torch.var(tensor_2d, unbiased=True)
std_unbiased = torch.std(tensor_2d, unbiased=True)

print(f"Unbiased variance: {var_unbiased}")
print(f"Unbiased std: {std_unbiased}")

# Median
median_all = torch.median(tensor_2d)
median_dim0_vals, median_dim0_indices = torch.median(tensor_2d, dim=0)

print(f"Median all: {median_all}")
print(f"Median dim 0 values: {median_dim0_vals}")
print(f"Median dim 0 indices: {median_dim0_indices}")

print("\n=== Quantile Operations ===")

# Quantiles
q_tensor = torch.randn(100)
quantile_50 = torch.quantile(q_tensor, 0.5)  # Median
quantile_25_75 = torch.quantile(q_tensor, torch.tensor([0.25, 0.75]))

print(f"50th percentile (median): {quantile_50}")
print(f"25th and 75th percentiles: {quantile_25_75}")

# Quantile with dimension
tensor_for_quantile = torch.randn(3, 4)
quantile_dim0 = torch.quantile(tensor_for_quantile, 0.5, dim=0)
print(f"Quantile dim 0: {quantile_dim0}")

print("\n=== Sorting and Ranking ===")

# Sort operations
unsorted_tensor = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
sorted_vals, sorted_indices = torch.sort(unsorted_tensor)
sorted_desc_vals, sorted_desc_indices = torch.sort(unsorted_tensor, descending=True)

print(f"Unsorted: {unsorted_tensor}")
print(f"Sorted values: {sorted_vals}")
print(f"Sorted indices: {sorted_indices}")
print(f"Sorted descending: {sorted_desc_vals}")

# Argsort
argsort_indices = torch.argsort(unsorted_tensor)
argsort_desc_indices = torch.argsort(unsorted_tensor, descending=True)

print(f"Argsort indices: {argsort_indices}")
print(f"Argsort descending: {argsort_desc_indices}")

# Top-k values
topk_vals, topk_indices = torch.topk(unsorted_tensor, k=3)
topk_largest = torch.topk(unsorted_tensor, k=2, largest=True)
topk_smallest = torch.topk(unsorted_tensor, k=2, largest=False)

print(f"Top 3 values: {topk_vals}")
print(f"Top 3 indices: {topk_indices}")
print(f"Top 2 largest: {topk_largest}")
print(f"Top 2 smallest: {topk_smallest}")

print("\n=== Unique and Mode ===")

# Unique values
tensor_with_duplicates = torch.tensor([1, 2, 2, 3, 3, 3, 4])
unique_vals = torch.unique(tensor_with_duplicates)
unique_vals_counts = torch.unique(tensor_with_duplicates, return_counts=True)
unique_inverse = torch.unique(tensor_with_duplicates, return_inverse=True)

print(f"Original: {tensor_with_duplicates}")
print(f"Unique: {unique_vals}")
print(f"Unique with counts: {unique_vals_counts}")
print(f"Unique with inverse: {unique_inverse}")

# Mode (most frequent value)
mode_val, mode_indices = torch.mode(tensor_with_duplicates)
print(f"Mode value: {mode_val}")
print(f"Mode indices: {mode_indices}")

print("\n=== Boolean Reductions ===")

# Boolean tensor operations
bool_tensor = torch.tensor([[True, False, True], [False, False, True], [True, True, False]])

any_all = torch.any(bool_tensor)
all_all = torch.all(bool_tensor)
any_dim0 = torch.any(bool_tensor, dim=0)
all_dim1 = torch.all(bool_tensor, dim=1)

print(f"Boolean tensor:\n{bool_tensor}")
print(f"Any all: {any_all}")
print(f"All all: {all_all}")
print(f"Any dim 0: {any_dim0}")
print(f"All dim 1: {all_dim1}")

print("\n=== Counting Operations ===")

# Count non-zero elements
sparse_tensor = torch.tensor([[0, 1, 0], [2, 0, 3], [0, 0, 4]])
nonzero_count = torch.count_nonzero(sparse_tensor)
nonzero_dim0 = torch.count_nonzero(sparse_tensor, dim=0)
nonzero_dim1 = torch.count_nonzero(sparse_tensor, dim=1)

print(f"Sparse tensor:\n{sparse_tensor}")
print(f"Nonzero count all: {nonzero_count}")
print(f"Nonzero count dim 0: {nonzero_dim0}")
print(f"Nonzero count dim 1: {nonzero_dim1}")

print("\n=== Norm Operations ===")

# Vector and matrix norms
vector = torch.tensor([3.0, 4.0])
matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Vector norms
l1_norm = torch.norm(vector, p=1)
l2_norm = torch.norm(vector, p=2)
inf_norm = torch.norm(vector, p=float('inf'))

print(f"Vector: {vector}")
print(f"L1 norm: {l1_norm}")
print(f"L2 norm: {l2_norm}")
print(f"Inf norm: {inf_norm}")

# Matrix norms
frobenius_norm = torch.norm(matrix, p='fro')
nuclear_norm = torch.norm(matrix, p='nuc')

print(f"Matrix:\n{matrix}")
print(f"Frobenius norm: {frobenius_norm}")
print(f"Nuclear norm: {nuclear_norm}")

# Norms along dimensions
norm_dim0 = torch.norm(matrix, dim=0)
norm_dim1 = torch.norm(matrix, dim=1)

print(f"Norm dim 0: {norm_dim0}")
print(f"Norm dim 1: {norm_dim1}")

print("\n=== Differential Operations ===")

# Differences
sequence = torch.tensor([1, 4, 7, 11, 16])
diff_1st = torch.diff(sequence)
diff_2nd = torch.diff(sequence, n=2)

print(f"Sequence: {sequence}")
print(f"First difference: {diff_1st}")
print(f"Second difference: {diff_2nd}")

# Gradient (similar to diff but for multi-dimensional)
tensor_2d_grad = torch.tensor([[1.0, 2.0, 4.0], [5.0, 7.0, 11.0]])
grad_result = torch.gradient(tensor_2d_grad)

print(f"Gradient input:\n{tensor_2d_grad}")
print(f"Gradient result: {len(grad_result)} gradients")

print("\n=== Multi-dimensional Reductions ===")

# Multiple dimension reductions
tensor_4d = torch.randn(2, 3, 4, 5)

# Reduce multiple dimensions
sum_multi_dims = torch.sum(tensor_4d, dim=(1, 3))
mean_multi_dims = torch.mean(tensor_4d, dim=(0, 2))

print(f"4D tensor shape: {tensor_4d.shape}")
print(f"Sum dims (1,3) shape: {sum_multi_dims.shape}")
print(f"Mean dims (0,2) shape: {mean_multi_dims.shape}")

# Flatten and reduce
flattened_sum = torch.sum(tensor_4d.flatten())
print(f"Flattened sum: {flattened_sum}")

print("\n=== Reduction with NaN Handling ===")

# Handle NaN values in reductions
tensor_with_nan = torch.tensor([1.0, 2.0, float('nan'), 4.0, 5.0])

# Regular operations propagate NaN
regular_mean = torch.mean(tensor_with_nan)
regular_sum = torch.sum(tensor_with_nan)

print(f"Tensor with NaN: {tensor_with_nan}")
print(f"Regular mean: {regular_mean}")
print(f"Regular sum: {regular_sum}")

# NaN-aware operations (if available)
finite_values = tensor_with_nan[torch.isfinite(tensor_with_nan)]
nan_safe_mean = torch.mean(finite_values)

print(f"Finite values: {finite_values}")
print(f"NaN-safe mean: {nan_safe_mean}")

print("\n=== Custom Reduction Functions ===")

# Implementing custom reductions
def custom_reduction_geometric_mean(tensor, dim=None):
    """Geometric mean reduction"""
    log_tensor = torch.log(torch.abs(tensor) + 1e-8)  # Add small epsilon
    log_mean = torch.mean(log_tensor, dim=dim)
    return torch.exp(log_mean)

positive_tensor = torch.tensor([[1.0, 2.0, 4.0], [2.0, 4.0, 8.0]])
geom_mean = custom_reduction_geometric_mean(positive_tensor)
geom_mean_dim0 = custom_reduction_geometric_mean(positive_tensor, dim=0)

print(f"Positive tensor:\n{positive_tensor}")
print(f"Geometric mean: {geom_mean}")
print(f"Geometric mean dim 0: {geom_mean_dim0}")

print("\n=== Reduction Operations Complete ===") 