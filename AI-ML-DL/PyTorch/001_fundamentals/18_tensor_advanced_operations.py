#!/usr/bin/env python3
"""PyTorch Advanced Tensor Operations - Masked operations, conditional operations, advanced techniques"""

import torch

print("=== Masked Operations ===")

# Create sample data
data = torch.randn(4, 5)
mask = torch.randint(0, 2, (4, 5), dtype=torch.bool)

print(f"Data shape: {data.shape}")
print(f"Mask shape: {mask.shape}")
print(f"Data:\n{data}")
print(f"Mask:\n{mask}")

# Masked selection
masked_values = torch.masked_select(data, mask)
print(f"Masked values: {masked_values}")
print(f"Masked values shape: {masked_values.shape}")

# Masked assignment
data_copy = data.clone()
data_copy[mask] = 0
print(f"After masked assignment (set to 0):\n{data_copy}")

# Masked fill
filled_data = data.masked_fill(mask, -999)
print(f"Masked fill with -999:\n{filled_data}")

# Masked scatter
source = torch.randn(masked_values.shape)
scattered_data = data.clone()
scattered_data[mask] = source
print(f"Masked scatter shape check: source {source.shape}, masked elements {mask.sum()}")

print("\n=== Advanced Masking Patterns ===")

# 3D masking
tensor_3d = torch.randn(2, 3, 4)
mask_3d = torch.randint(0, 2, (2, 3, 4), dtype=torch.bool)

# Masked operations on 3D tensors
masked_3d = torch.masked_select(tensor_3d, mask_3d)
print(f"3D masked selection shape: {masked_3d.shape}")

# Dimension-specific masking
row_mask = torch.tensor([True, False, True, False, True])
col_mask = torch.tensor([False, True, True, True])

# Apply row mask
row_masked = data[row_mask]
print(f"Row masked shape: {row_masked.shape}")

# Apply column mask
col_masked = data[:, col_mask]
print(f"Column masked shape: {col_masked.shape}")

# Combined masking
combined_masked = data[row_mask][:, col_mask]
print(f"Combined masked shape: {combined_masked.shape}")

print("\n=== Conditional Operations ===")

# torch.where - conditional selection
condition = data > 0
result_where = torch.where(condition, data, torch.zeros_like(data))
print(f"Where (positive values kept):\n{result_where}")

# Multiple conditions
complex_condition = (data > 0) & (data < 1)
result_complex = torch.where(complex_condition, data, -1)
print(f"Complex condition result shape: {result_complex.shape}")

# Nested where operations
nested_result = torch.where(
    data > 1, 
    1,  # If > 1, set to 1
    torch.where(data < -1, -1, data)  # If < -1, set to -1, else keep original
)
print(f"Nested where (clamp to [-1, 1]):\n{nested_result}")

# Conditional assignment with different tensors
tensor_a = torch.randn(3, 3)
tensor_b = torch.randn(3, 3)
condition_2d = torch.rand(3, 3) > 0.5

conditional_mix = torch.where(condition_2d, tensor_a, tensor_b)
print(f"Conditional mix shape: {conditional_mix.shape}")

print("\n=== Advanced Indexing Operations ===")

# Advanced indexing with tensors
indices_1d = torch.tensor([0, 2, 1, 3])
selected_rows = data[indices_1d]
print(f"Advanced indexing rows shape: {selected_rows.shape}")

# 2D advanced indexing
row_indices = torch.tensor([0, 1, 2, 3])
col_indices = torch.tensor([0, 1, 2, 3])
diagonal_elements = data[row_indices, col_indices]
print(f"Diagonal elements: {diagonal_elements}")

# Fancy indexing with broadcasting
row_idx_2d = torch.tensor([[0, 1], [2, 3]])
col_idx_2d = torch.tensor([[0, 1], [2, 3]])
fancy_indexed = data[row_idx_2d, col_idx_2d]
print(f"Fancy indexed shape: {fancy_indexed.shape}")

# Boolean indexing with conditions
bool_idx = data > data.mean()
above_mean = data[bool_idx]
print(f"Above mean values count: {above_mean.numel()}")

print("\n=== Gather and Scatter Operations ===")

# Gather operation
source_tensor = torch.arange(20).float().reshape(4, 5)
index_tensor = torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 0]])

gathered = torch.gather(source_tensor, 1, index_tensor)
print(f"Source tensor:\n{source_tensor}")
print(f"Index tensor:\n{index_tensor}")
print(f"Gathered result:\n{gathered}")

# Scatter operation
target_tensor = torch.zeros(4, 5)
values_to_scatter = torch.ones(4, 3)

scattered = target_tensor.scatter(1, index_tensor, values_to_scatter)
print(f"Scattered result:\n{scattered}")

# Scatter add (accumulate values)
target_scatter_add = torch.zeros(4, 5)
scattered_add = target_scatter_add.scatter_add(1, index_tensor, values_to_scatter)
print(f"Scatter add result:\n{scattered_add}")

print("\n=== Advanced Tensor Manipulation ===")

# Flip operations
tensor_2d = torch.arange(12).reshape(3, 4)
print(f"Original tensor:\n{tensor_2d}")

flipped_rows = torch.flip(tensor_2d, dims=[0])
flipped_cols = torch.flip(tensor_2d, dims=[1])
flipped_both = torch.flip(tensor_2d, dims=[0, 1])

print(f"Flipped rows:\n{flipped_rows}")
print(f"Flipped cols:\n{flipped_cols}")
print(f"Flipped both:\n{flipped_both}")

# Roll operations (circular shift)
rolled_right = torch.roll(tensor_2d, shifts=1, dims=1)
rolled_down = torch.roll(tensor_2d, shifts=1, dims=0)
rolled_diagonal = torch.roll(tensor_2d, shifts=(1, 1), dims=(0, 1))

print(f"Rolled right:\n{rolled_right}")
print(f"Rolled down:\n{rolled_down}")
print(f"Rolled diagonal:\n{rolled_diagonal}")

print("\n=== Tensor Transformation Operations ===")

# Rot90 - 90 degree rotations
rotated_90 = torch.rot90(tensor_2d, k=1, dims=[0, 1])
rotated_180 = torch.rot90(tensor_2d, k=2, dims=[0, 1])
rotated_270 = torch.rot90(tensor_2d, k=3, dims=[0, 1])

print(f"Rotated 90°:\n{rotated_90}")
print(f"Rotated 180°:\n{rotated_180}")
print(f"Rotated 270°:\n{rotated_270}")

# Cross correlation (for signal processing)
signal_1d = torch.randn(100)
kernel_1d = torch.randn(5)

# 1D cross correlation using conv1d
cross_corr = torch.nn.functional.conv1d(
    signal_1d.unsqueeze(0).unsqueeze(0), 
    kernel_1d.flip(0).unsqueeze(0).unsqueeze(0)
).squeeze()

print(f"Cross correlation shape: {cross_corr.shape}")

print("\n=== Sliding Window Operations ===")

# Unfold for sliding windows
sequence = torch.arange(20).float()
window_size = 5
stride = 2

# Create sliding windows
windows = sequence.unfold(0, window_size, stride)
print(f"Original sequence: {sequence}")
print(f"Sliding windows shape: {windows.shape}")
print(f"Sliding windows:\n{windows}")

# 2D sliding windows
image_2d = torch.arange(36).float().reshape(6, 6)
patches = image_2d.unfold(0, 3, 1).unfold(1, 3, 1)
print(f"2D patches shape: {patches.shape}")

# Fold operation (inverse of unfold)
folded_back = torch.nn.functional.fold(
    windows.transpose(0, 1).unsqueeze(0),
    output_size=(20,),
    kernel_size=(window_size,),
    stride=(stride,)
).squeeze()

print(f"Folded back shape: {folded_back.shape}")

print("\n=== Tensor Interpolation ===")

# Linear interpolation between tensors
tensor_start = torch.zeros(3, 3)
tensor_end = torch.ones(3, 3)

# Interpolate at different points
interp_25 = torch.lerp(tensor_start, tensor_end, 0.25)
interp_50 = torch.lerp(tensor_start, tensor_end, 0.5)
interp_75 = torch.lerp(tensor_start, tensor_end, 0.75)

print(f"Interpolation at 25%:\n{interp_25}")
print(f"Interpolation at 50%:\n{interp_50}")
print(f"Interpolation at 75%:\n{interp_75}")

# Tensor interpolation with weights
weights = torch.tensor([0.1, 0.3, 0.6])
tensors_to_interp = torch.stack([tensor_start, interp_50, tensor_end])
weighted_interp = torch.sum(tensors_to_interp * weights.view(-1, 1, 1), dim=0)
print(f"Weighted interpolation:\n{weighted_interp}")

print("\n=== Advanced Reduction Operations ===")

# Custom reduction with scan operations
data_1d = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)

# Cumulative operations
cumsum = torch.cumsum(data_1d, dim=0)
cumprod = torch.cumprod(data_1d, dim=0)
cummax_vals, cummax_indices = torch.cummax(data_1d, dim=0)
cummin_vals, cummin_indices = torch.cummin(data_1d, dim=0)

print(f"Original: {data_1d}")
print(f"Cumsum: {cumsum}")
print(f"Cumprod: {cumprod}")
print(f"Cummax values: {cummax_vals}")
print(f"Cummin values: {cummin_vals}")

# Multi-dimensional cumulative operations
data_2d = torch.randn(3, 4)
cumsum_2d_dim0 = torch.cumsum(data_2d, dim=0)
cumsum_2d_dim1 = torch.cumsum(data_2d, dim=1)

print(f"2D cumsum along dim 0 shape: {cumsum_2d_dim0.shape}")
print(f"2D cumsum along dim 1 shape: {cumsum_2d_dim1.shape}")

print("\n=== Tensor Constraints and Projections ===")

# Clamp operations
unconstrained = torch.randn(5, 5)
clamped = torch.clamp(unconstrained, min=-1, max=1)
clamped_min = torch.clamp(unconstrained, min=0)  # ReLU-like
clamped_max = torch.clamp(unconstrained, max=0)

print(f"Clamped [-1, 1] range applied")
print(f"Clamped min (>= 0) applied")
print(f"Clamped max (<= 0) applied")

# Normalize operations
l2_normalized = torch.nn.functional.normalize(unconstrained, p=2, dim=1)
print(f"L2 normalized (unit norm per row)")

# Projection onto simplex (probability vectors)
def project_simplex(x, dim=-1):
    """Project onto probability simplex"""
    x_sorted, _ = torch.sort(x, dim=dim, descending=True)
    x_cumsum = torch.cumsum(x_sorted, dim=dim)
    k = torch.arange(1, x.size(dim) + 1, device=x.device, dtype=x.dtype)
    
    if dim == -1:
        k = k.view(-1)
        rho = torch.sum((x_sorted - (x_cumsum - 1) / k) > 0, dim=dim) - 1
        theta = (torch.gather(x_cumsum, dim, rho.unsqueeze(dim)) - 1) / (rho + 1).float()
    else:
        # Handle other dimensions
        theta = (x_cumsum[..., -1:] - 1) / x.size(dim)
    
    return torch.clamp(x - theta, min=0)

# Example simplex projection
prob_vector = torch.randn(5)
projected = project_simplex(prob_vector)
print(f"Original sum: {prob_vector.sum():.4f}")
print(f"Projected sum: {projected.sum():.4f}")

print("\n=== Memory-Efficient Advanced Operations ===")

# In-place advanced operations
inplace_tensor = torch.randn(1000, 1000)
original_ptr = inplace_tensor.data_ptr()

# In-place masking
mask_large = inplace_tensor > 0
inplace_tensor.masked_fill_(mask_large, 1.0)
print(f"In-place masked fill same memory: {inplace_tensor.data_ptr() == original_ptr}")

# In-place scatter
indices = torch.randint(0, 1000, (100,))
values = torch.randn(100)
inplace_tensor.scatter_(0, indices.unsqueeze(1).expand(-1, 1000), values.unsqueeze(1).expand(-1, 1000))
print(f"In-place scatter same memory: {inplace_tensor.data_ptr() == original_ptr}")

print("\n=== Advanced Broadcasting with Operations ===")

# Complex broadcasting scenarios
tensor_a = torch.randn(1, 5, 1, 3)
tensor_b = torch.randn(4, 1, 2, 1)

# Masked operations with broadcasting
mask_broadcast = torch.randint(0, 2, (4, 5, 2, 3), dtype=torch.bool)
broadcast_masked = torch.where(mask_broadcast, tensor_a, tensor_b)
print(f"Broadcast masked shape: {broadcast_masked.shape}")

# Advanced indexing with broadcasting
idx_a = torch.randint(0, 2, (3, 1))
idx_b = torch.randint(0, 3, (1, 4))
broadcast_indexed = tensor_a[0, idx_a, 0, idx_b]
print(f"Broadcast indexed shape: {broadcast_indexed.shape}")

print("\n=== Tensor Network Operations ===")

# Einstein summation for complex tensor operations
A = torch.randn(3, 4, 5)
B = torch.randn(5, 6, 7)
C = torch.randn(7, 8)

# Complex tensor contraction
result_einsum = torch.einsum('ijk,klm,mn->ijln', A, B, C)
print(f"Einstein sum result shape: {result_einsum.shape}")

# Batch operations with einsum
batch_A = torch.randn(10, 3, 4)
batch_B = torch.randn(10, 4, 5)
batch_result = torch.einsum('bij,bjk->bik', batch_A, batch_B)
print(f"Batch einsum shape: {batch_result.shape}")

# Trace over multiple dimensions
multi_trace = torch.einsum('iijj->', torch.randn(3, 3, 4, 4))
print(f"Multi-dimensional trace: {multi_trace}")

print("\n=== Advanced Operations Complete ===")

print("Advanced Operations Summary:")
print("1. Masked operations for selective processing")
print("2. Conditional operations with torch.where")
print("3. Advanced indexing and fancy indexing")
print("4. Gather/scatter for flexible data rearrangement")
print("5. Tensor transformations (flip, roll, rotate)")
print("6. Sliding window operations with unfold")
print("7. Tensor interpolation and weighted combinations")
print("8. Cumulative operations and scans")
print("9. Constraint projections and normalizations")
print("10. Memory-efficient in-place operations")
print("11. Complex broadcasting scenarios")
print("12. Tensor network operations with einsum") 