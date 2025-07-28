#!/usr/bin/env python3
"""PyTorch Advanced Tensor Indexing"""

import torch

print("=== Basic Indexing ===")

# Create sample tensor
tensor = torch.arange(24).reshape(4, 6)
print(f"Original tensor:\n{tensor}")

# Single element indexing
element = tensor[2, 3]
print(f"Element [2,3]: {element}")

# Row indexing
row = tensor[1]
print(f"Row 1: {row}")

# Column indexing
column = tensor[:, 2]
print(f"Column 2: {column}")

# Multiple rows
rows = tensor[1:3]
print(f"Rows 1-2:\n{rows}")

# Multiple columns
columns = tensor[:, 1:4]
print(f"Columns 1-3:\n{columns}")

print("\n=== Advanced Slicing ===")

# Step slicing
every_other_row = tensor[::2]
print(f"Every other row:\n{every_other_row}")

every_other_col = tensor[:, ::2]
print(f"Every other column:\n{every_other_col}")

# Negative indexing
last_row = tensor[-1]
last_column = tensor[:, -1]
print(f"Last row: {last_row}")
print(f"Last column: {last_column}")

# Reverse indexing
reversed_rows = tensor[::-1]
reversed_cols = tensor[:, ::-1]
print(f"Reversed rows:\n{reversed_rows}")
print(f"Reversed columns:\n{reversed_cols}")

print("\n=== Ellipsis Indexing ===")

# 3D tensor for ellipsis demo
tensor_3d = torch.arange(60).reshape(3, 4, 5)
print(f"3D tensor shape: {tensor_3d.shape}")

# Ellipsis (...) represents multiple ':'
slice_with_ellipsis = tensor_3d[..., 2]  # Same as tensor_3d[:, :, 2]
print(f"Using ellipsis [..., 2]: {slice_with_ellipsis.shape}")

slice_middle = tensor_3d[1, ...]  # Same as tensor_3d[1, :, :]
print(f"Middle slice [1, ...]: {slice_middle.shape}")

print("\n=== Boolean Indexing ===")

# Create tensor and condition
data = torch.randn(4, 4)
print(f"Data tensor:\n{data}")

# Boolean mask
mask = data > 0
print(f"Positive mask:\n{mask}")

# Extract positive values
positive_values = data[mask]
print(f"Positive values: {positive_values}")

# More complex conditions
complex_mask = (data > -0.5) & (data < 0.5)
moderate_values = data[complex_mask]
print(f"Values between -0.5 and 0.5: {moderate_values}")

# Boolean indexing with multiple dimensions
tensor_2d = torch.randn(3, 4)
row_mask = torch.tensor([True, False, True])
filtered_rows = tensor_2d[row_mask]
print(f"Filtered rows shape: {filtered_rows.shape}")

print("\n=== Fancy Indexing ===")

# Index with tensor of indices
indices_tensor = torch.tensor([0, 2, 1])
selected_rows = tensor[indices_tensor]
print(f"Selected rows with indices [0,2,1]:\n{selected_rows}")

# 2D fancy indexing
row_indices = torch.tensor([0, 1, 2])
col_indices = torch.tensor([1, 3, 5])
selected_elements = tensor[row_indices, col_indices]
print(f"Selected elements: {selected_elements}")

# Advanced fancy indexing
row_idx = torch.tensor([[0, 1], [2, 3]])
col_idx = torch.tensor([[0, 1], [2, 3]])
fancy_selection = tensor[row_idx, col_idx]
print(f"Fancy selection:\n{fancy_selection}")

print("\n=== Index_select Function ===")

# Using index_select
indices = torch.tensor([0, 2, 3])
selected_rows_func = torch.index_select(tensor, 0, indices)
print(f"index_select rows:\n{selected_rows_func}")

selected_cols_func = torch.index_select(tensor, 1, indices)
print(f"index_select columns:\n{selected_cols_func}")

print("\n=== Masked Select ===")

# masked_select returns 1D tensor
large_values = torch.masked_select(data, data > 0.5)
print(f"Values > 0.5: {large_values}")

# nonzero - get indices of non-zero elements
sparse_tensor = torch.tensor([[0, 1, 0], [2, 0, 3], [0, 4, 0]])
nonzero_indices = torch.nonzero(sparse_tensor)
print(f"Sparse tensor:\n{sparse_tensor}")
print(f"Non-zero indices:\n{nonzero_indices}")

# where function
condition = data > 0
positive_replaced = torch.where(condition, data, torch.zeros_like(data))
print(f"Positive values kept, others zero:\n{positive_replaced}")

print("\n=== Advanced Indexing Patterns ===")

# Gather operation
source = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
index = torch.tensor([[0, 1, 2], [2, 0, 1]])
gathered = torch.gather(source, 0, index)
print(f"Source:\n{source}")
print(f"Gather index:\n{index}")
print(f"Gathered:\n{gathered}")

# Scatter operation
target = torch.zeros(3, 3)
src = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
index_scatter = torch.tensor([[0, 1, 2], [2, 0, 1]])
scattered = target.scatter(0, index_scatter, src)
print(f"Scattered:\n{scattered}")

print("\n=== Take and Put ===")

# take - flatten and index
flat_indices = torch.tensor([1, 5, 9])
taken_values = torch.take(tensor, flat_indices)
print(f"Taken values: {taken_values}")

# Advanced take along axis
values_2d = torch.randn(3, 4)
indices_2d = torch.tensor([[1, 2], [0, 3], [2, 1]])
taken_along_axis = torch.gather(values_2d, 1, indices_2d)
print(f"Taken along axis:\n{taken_along_axis}")

print("\n=== Conditional Indexing ===")

# Using conditions for indexing
numbers = torch.tensor([1, -2, 3, -4, 5, -6])
positive_indices = (numbers > 0).nonzero().squeeze()
positive_numbers = numbers[positive_indices]
print(f"Original: {numbers}")
print(f"Positive indices: {positive_indices}")
print(f"Positive numbers: {positive_numbers}")

# topk indexing
values = torch.tensor([1.2, 3.4, 0.8, 2.1, 4.5])
top_values, top_indices = torch.topk(values, 3)
print(f"Top 3 values: {top_values}")
print(f"Top 3 indices: {top_indices}")

print("\n=== Multi-dimensional Advanced Indexing ===")

# 3D tensor advanced indexing
tensor_3d = torch.arange(60).reshape(3, 4, 5)

# Select specific elements from each matrix
batch_indices = torch.tensor([0, 1, 2])
row_indices = torch.tensor([1, 2, 0])
col_indices = torch.tensor([2, 3, 4])

selected_3d = tensor_3d[batch_indices, row_indices, col_indices]
print(f"3D selected elements: {selected_3d}")

# Broadcasting in indexing
expanded_indices = torch.tensor([[0, 1], [1, 2]])
row_broadcast = torch.tensor([0, 2])
selected_broadcast = tensor_3d[expanded_indices, row_broadcast.unsqueeze(1)]
print(f"Broadcast indexing shape: {selected_broadcast.shape}")

print("\n=== In-place Indexing Operations ===")

# Modify tensor using indexing
mutable_tensor = torch.zeros(4, 4)
print(f"Initial tensor:\n{mutable_tensor}")

# Set diagonal elements
diagonal_indices = torch.arange(4)
mutable_tensor[diagonal_indices, diagonal_indices] = 1
print(f"After setting diagonal:\n{mutable_tensor}")

# Set specific region
mutable_tensor[1:3, 1:3] = 5
print(f"After setting region:\n{mutable_tensor}")

# Boolean indexing assignment
mask = mutable_tensor == 0
mutable_tensor[mask] = -1
print(f"After boolean assignment:\n{mutable_tensor}")

print("\n=== Index Errors and Edge Cases ===")

# Out of bounds indexing
try:
    invalid_access = tensor[10, 0]
except IndexError as e:
    print(f"Index error: {e}")

# Empty index
empty_indices = torch.tensor([], dtype=torch.long)
empty_selection = tensor[empty_indices]
print(f"Empty selection shape: {empty_selection.shape}")

# Negative indices beyond bounds
try:
    invalid_negative = tensor[-10, 0]
except IndexError as e:
    print(f"Negative index error: {e}")

print("\n=== Advanced Indexing Complete ===") 