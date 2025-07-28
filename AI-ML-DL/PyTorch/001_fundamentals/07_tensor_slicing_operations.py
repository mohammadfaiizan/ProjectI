#!/usr/bin/env python3
"""PyTorch Tensor Slicing Operations - Advanced slicing, stride operations, views"""

import torch

print("=== Basic Slicing Operations ===")

# Create sample tensor
tensor_3d = torch.arange(60).reshape(3, 4, 5)
print(f"Original tensor shape: {tensor_3d.shape}")
print(f"Original tensor:\n{tensor_3d}")

# Basic indexing and slicing
single_element = tensor_3d[1, 2, 3]
print(f"Single element [1,2,3]: {single_element}")

# Slice along first dimension
slice_dim0 = tensor_3d[1:3]
print(f"Slice [1:3] shape: {slice_dim0.shape}")

# Slice along multiple dimensions
multi_slice = tensor_3d[0:2, 1:3, 2:4]
print(f"Multi-slice [0:2, 1:3, 2:4] shape: {multi_slice.shape}")

print("\n=== Advanced Slicing Patterns ===")

# Step slicing
every_other = tensor_3d[::2, ::2, ::2]
print(f"Every other element shape: {every_other.shape}")

# Negative indices
last_slice = tensor_3d[-1, -2:, -3:]
print(f"Last slice shape: {last_slice.shape}")

# Reverse slicing
reversed_tensor = tensor_3d[::-1, ::-1, ::-1]
print(f"Fully reversed shape: {reversed_tensor.shape}")

# Mixed slicing patterns
mixed_slice = tensor_3d[::2, 1::2, ::-1]
print(f"Mixed slice shape: {mixed_slice.shape}")

print("\n=== Ellipsis and None Indexing ===")

# Using ellipsis (...)
ellipsis_slice = tensor_3d[..., 2]  # Same as tensor_3d[:, :, 2]
print(f"Ellipsis slice [..., 2] shape: {ellipsis_slice.shape}")

middle_ellipsis = tensor_3d[1, ..., :3]  # Same as tensor_3d[1, :, :3]
print(f"Middle ellipsis [1, ..., :3] shape: {middle_ellipsis.shape}")

# Using None for new dimensions
new_dim_slice = tensor_3d[None, 1, :, 2]
print(f"New dimension slice shape: {new_dim_slice.shape}")

multiple_new_dims = tensor_3d[1, :, None, 2, None]
print(f"Multiple new dims shape: {multiple_new_dims.shape}")

print("\n=== Stride Operations ===")

# Understanding strides
regular_tensor = torch.arange(24).reshape(4, 6)
print(f"Regular tensor strides: {regular_tensor.stride()}")
print(f"Regular tensor:\n{regular_tensor}")

# Transposed tensor has different strides
transposed = regular_tensor.t()
print(f"Transposed strides: {transposed.stride()}")

# Custom stride with as_strided
custom_strided = torch.as_strided(regular_tensor, size=(2, 3), stride=(12, 2))
print(f"Custom strided shape: {custom_strided.shape}")
print(f"Custom strided:\n{custom_strided}")

# Sliding window with strides
sliding_window = torch.as_strided(
    torch.arange(10), 
    size=(6, 3), 
    stride=(1, 1)
)
print(f"Sliding window:\n{sliding_window}")

print("\n=== View Operations ===")

# Basic views
base_tensor = torch.arange(24)
reshaped_view = base_tensor.view(4, 6)
print(f"Reshaped view shape: {reshaped_view.shape}")

# View with -1 (automatic dimension)
auto_view = base_tensor.view(3, -1)
print(f"Auto view shape: {auto_view.shape}")

# Flatten view
flattened_view = reshaped_view.view(-1)
print(f"Flattened view shape: {flattened_view.shape}")

# Views share memory
reshaped_view[0, 0] = 999
print(f"Original after view modification: {base_tensor[0]}")

print("\n=== Contiguous vs Non-contiguous ===")

# Contiguous tensor
contiguous_tensor = torch.randn(4, 5)
print(f"Contiguous: {contiguous_tensor.is_contiguous()}")
print(f"Contiguous strides: {contiguous_tensor.stride()}")

# Non-contiguous after transpose
non_contiguous = contiguous_tensor.t()
print(f"Non-contiguous: {non_contiguous.is_contiguous()}")
print(f"Non-contiguous strides: {non_contiguous.stride()}")

# View requires contiguous tensor
try:
    invalid_view = non_contiguous.view(-1)
except RuntimeError as e:
    print(f"View error: {str(e)[:50]}...")

# Make contiguous and then view
made_contiguous = non_contiguous.contiguous()
valid_view = made_contiguous.view(-1)
print(f"Valid view after contiguous: {valid_view.shape}")

print("\n=== Advanced View Operations ===")

# Unfold - sliding window view
sequence = torch.arange(10)
unfolded = sequence.unfold(0, 3, 1)  # window_size=3, step=1
print(f"Original sequence: {sequence}")
print(f"Unfolded:\n{unfolded}")

# 2D unfold
matrix_2d = torch.arange(20).reshape(4, 5)
unfolded_2d = matrix_2d.unfold(1, 3, 1)  # unfold along dim 1
print(f"2D unfold shape: {unfolded_2d.shape}")

# Multiple unfolds
double_unfold = matrix_2d.unfold(0, 2, 1).unfold(2, 3, 1)
print(f"Double unfold shape: {double_unfold.shape}")

print("\n=== Narrow Operations ===")

# Narrow - select slice along dimension
tensor_for_narrow = torch.arange(30).reshape(5, 6)
narrowed = torch.narrow(tensor_for_narrow, 0, 1, 3)  # dim=0, start=1, length=3
print(f"Original shape: {tensor_for_narrow.shape}")
print(f"Narrowed shape: {narrowed.shape}")

# Narrow along different dimensions
narrow_dim1 = torch.narrow(tensor_for_narrow, 1, 2, 3)
print(f"Narrow dim 1 shape: {narrow_dim1.shape}")

print("\n=== Select and Index Select ===")

# Select - choose slice along dimension
selected = torch.select(tensor_for_narrow, 0, 2)  # select index 2 along dim 0
print(f"Selected shape: {selected.shape}")

# Index select with tensor indices
indices = torch.tensor([0, 2, 4])
index_selected = torch.index_select(tensor_for_narrow, 0, indices)
print(f"Index selected shape: {index_selected.shape}")

# Index select along different dimension
col_indices = torch.tensor([1, 3, 5])
col_selected = torch.index_select(tensor_for_narrow, 1, col_indices)
print(f"Column selected shape: {col_selected.shape}")

print("\n=== Slice Assignment ===")

# In-place slice modification
mutable_tensor = torch.zeros(4, 6)
print(f"Initial tensor:\n{mutable_tensor}")

# Assign to slice
mutable_tensor[1:3, 2:5] = 1
print(f"After slice assignment:\n{mutable_tensor}")

# Assign with broadcasting
mutable_tensor[:, 0] = torch.tensor([1, 2, 3, 4])
print(f"After column assignment:\n{mutable_tensor}")

# Assign with fancy indexing
row_indices = torch.tensor([0, 2])
col_indices = torch.tensor([1, 3, 5])
mutable_tensor[row_indices[:, None], col_indices] = 5
print(f"After fancy indexing assignment:\n{mutable_tensor}")

print("\n=== Slice with Boolean Masks ===")

# Boolean slicing
data_tensor = torch.randn(4, 5)
positive_mask = data_tensor > 0

# Boolean indexing returns 1D tensor
positive_values = data_tensor[positive_mask]
print(f"Positive values shape: {positive_values.shape}")

# Boolean assignment
data_tensor[positive_mask] = torch.abs(data_tensor[positive_mask])
print("Applied absolute value to positive elements")

# Complex boolean conditions
complex_mask = (torch.abs(data_tensor) > 0.5) & (data_tensor < 0)
data_tensor[complex_mask] = 0
print("Set complex condition elements to zero")

print("\n=== Advanced Slicing Tricks ===")

# Diagonal slicing
square_matrix = torch.arange(16).reshape(4, 4)
diagonal = square_matrix.diag()
print(f"Diagonal: {diagonal}")

# Anti-diagonal
anti_diagonal = square_matrix.flip(1).diag()
print(f"Anti-diagonal: {anti_diagonal}")

# Upper triangular slicing
upper_tri_indices = torch.triu_indices(4, 4)
upper_tri_values = square_matrix[upper_tri_indices[0], upper_tri_indices[1]]
print(f"Upper triangular values: {upper_tri_values}")

# Lower triangular
lower_tri_indices = torch.tril_indices(4, 4)
lower_tri_values = square_matrix[lower_tri_indices[0], lower_tri_indices[1]]
print(f"Lower triangular values: {lower_tri_values}")

print("\n=== Memory-Efficient Slicing ===")

# Slicing doesn't copy data (creates view)
large_tensor = torch.randn(1000, 1000)
slice_view = large_tensor[100:200, 200:300]

print(f"Original memory ptr: {large_tensor.data_ptr()}")
print(f"Slice memory ptr: {slice_view.data_ptr()}")
print(f"Different pointers (but same storage): {large_tensor.data_ptr() != slice_view.data_ptr()}")
print(f"Same storage: {large_tensor.storage().data_ptr() == slice_view.storage().data_ptr()}")

# Clone creates new memory
slice_clone = slice_view.clone()
print(f"Clone memory ptr: {slice_clone.data_ptr()}")
print(f"Clone has different storage: {slice_view.storage().data_ptr() != slice_clone.storage().data_ptr()}")

print("\n=== Performance Considerations ===")

import time

# Performance comparison: slicing vs copying
large_data = torch.randn(2000, 2000)

# Slicing (fast - creates view)
start_time = time.time()
slice_result = large_data[500:1500, 500:1500]
slice_time = time.time() - start_time

# Copying (slower - creates new tensor)
start_time = time.time()
copy_result = large_data[500:1500, 500:1500].clone()
copy_time = time.time() - start_time

print(f"Slicing time: {slice_time:.6f} seconds")
print(f"Copying time: {copy_time:.6f} seconds")
print(f"Slicing speedup: {copy_time / slice_time:.2f}x")

print("\n=== Common Slicing Patterns ===")

# Batch processing patterns
batch_data = torch.randn(32, 3, 224, 224)  # NCHW format

# Get first 16 samples
first_half = batch_data[:16]
print(f"First half shape: {first_half.shape}")

# Get center crop
center_crop = batch_data[:, :, 50:174, 50:174]
print(f"Center crop shape: {center_crop.shape}")

# Channel selection
red_channel = batch_data[:, 0]  # Get red channel
print(f"Red channel shape: {red_channel.shape}")

# Spatial downsampling
downsampled = batch_data[:, :, ::2, ::2]
print(f"Downsampled shape: {downsampled.shape}")

print("\n=== Error Handling in Slicing ===")

# Out of bounds slicing (clamps to valid range)
test_tensor = torch.arange(10)
out_of_bounds = test_tensor[5:20]  # End index beyond tensor size
print(f"Out of bounds slice: {out_of_bounds}")

# Empty slices
empty_slice = test_tensor[5:5]
print(f"Empty slice shape: {empty_slice.shape}")

# Invalid step size
try:
    invalid_step = test_tensor[::0]  # Step cannot be zero
except ValueError as e:
    print(f"Invalid step error: {e}")

print("\n=== Slicing Operations Complete ===") 