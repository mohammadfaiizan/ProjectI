#!/usr/bin/env python3
"""PyTorch Tensor Shapes and Dimensions"""

import torch

print("=== Tensor Shape Basics ===")

# Create tensors with different shapes
tensor_1d = torch.randn(5)
tensor_2d = torch.randn(3, 4)
tensor_3d = torch.randn(2, 3, 4)
tensor_4d = torch.randn(2, 3, 4, 5)

print(f"1D tensor shape: {tensor_1d.shape}")
print(f"2D tensor shape: {tensor_2d.shape}")
print(f"3D tensor shape: {tensor_3d.shape}")
print(f"4D tensor shape: {tensor_4d.shape}")

# Shape vs size
print(f"Shape: {tensor_3d.shape}")
print(f"Size: {tensor_3d.size()}")
print(f"Size with dim: {tensor_3d.size(0)}, {tensor_3d.size(1)}, {tensor_3d.size(2)}")

print("\n=== Dimension Properties ===")

# Number of dimensions
print(f"1D ndim: {tensor_1d.ndim}")
print(f"2D ndim: {tensor_2d.ndim}")
print(f"3D ndim: {tensor_3d.ndim}")
print(f"4D ndim: {tensor_4d.ndim}")

# Number of elements
print(f"1D numel: {tensor_1d.numel()}")
print(f"2D numel: {tensor_2d.numel()}")
print(f"3D numel: {tensor_3d.numel()}")

# Check if tensor is empty
empty_tensor = torch.empty(0, 3)
print(f"Empty tensor shape: {empty_tensor.shape}")
print(f"Is empty: {empty_tensor.numel() == 0}")

print("\n=== Reshape Operations ===")

# Basic reshaping
original = torch.arange(24)
print(f"Original: {original.shape}")

# Reshape to 2D
reshaped_2d = original.reshape(4, 6)
print(f"Reshape to 4x6: {reshaped_2d.shape}")

# Reshape to 3D
reshaped_3d = original.reshape(2, 3, 4)
print(f"Reshape to 2x3x4: {reshaped_3d.shape}")

# Using -1 for automatic dimension calculation
auto_reshape = original.reshape(4, -1)
print(f"Reshape with -1: {auto_reshape.shape}")

# Multiple -1 would cause error
try:
    invalid_reshape = original.reshape(-1, -1)
except RuntimeError as e:
    print(f"Multiple -1 error: {str(e)[:50]}...")

print("\n=== View Operations ===")

# View vs reshape - view shares memory
base_tensor = torch.arange(12)
viewed_tensor = base_tensor.view(3, 4)
print(f"Original: {base_tensor.shape}")
print(f"Viewed: {viewed_tensor.shape}")

# Modifying view affects original
viewed_tensor[0, 0] = 999
print(f"After modifying view, original[0]: {base_tensor[0]}")

# View with -1
auto_view = base_tensor.view(2, -1)
print(f"Auto view shape: {auto_view.shape}")

# Contiguous requirement for view
non_contiguous = torch.randn(3, 4).transpose(0, 1)
print(f"Is contiguous: {non_contiguous.is_contiguous()}")

# This would fail - need contiguous tensor for view
try:
    invalid_view = non_contiguous.view(-1)
except RuntimeError as e:
    print(f"Non-contiguous view error: {str(e)[:30]}...")

# Make contiguous first
contiguous_view = non_contiguous.contiguous().view(-1)
print(f"Contiguous view shape: {contiguous_view.shape}")

print("\n=== Squeeze and Unsqueeze ===")

# Squeeze - remove dimensions of size 1
tensor_with_ones = torch.randn(1, 3, 1, 4, 1)
print(f"Original with 1s: {tensor_with_ones.shape}")

squeezed = tensor_with_ones.squeeze()
print(f"Squeezed all: {squeezed.shape}")

# Squeeze specific dimension
squeezed_dim = tensor_with_ones.squeeze(0)
print(f"Squeezed dim 0: {squeezed_dim.shape}")

squeezed_dim2 = tensor_with_ones.squeeze(2)
print(f"Squeezed dim 2: {squeezed_dim2.shape}")

# Unsqueeze - add dimension of size 1
tensor_for_unsqueeze = torch.randn(3, 4)
print(f"Original: {tensor_for_unsqueeze.shape}")

unsqueezed_0 = tensor_for_unsqueeze.unsqueeze(0)
unsqueezed_1 = tensor_for_unsqueeze.unsqueeze(1)
unsqueezed_neg1 = tensor_for_unsqueeze.unsqueeze(-1)

print(f"Unsqueezed dim 0: {unsqueezed_0.shape}")
print(f"Unsqueezed dim 1: {unsqueezed_1.shape}")
print(f"Unsqueezed dim -1: {unsqueezed_neg1.shape}")

print("\n=== Transpose and Permute ===")

# Transpose - swap two dimensions
matrix = torch.randn(3, 4)
print(f"Matrix shape: {matrix.shape}")

transposed = matrix.transpose(0, 1)
print(f"Transposed: {transposed.shape}")

# .t() shortcut for 2D matrices
transposed_t = matrix.t()
print(f"Using .t(): {transposed_t.shape}")

# Transpose higher dimensional tensors
tensor_3d = torch.randn(2, 3, 4)
transposed_3d = tensor_3d.transpose(0, 2)
print(f"3D original: {tensor_3d.shape}")
print(f"3D transposed (0,2): {transposed_3d.shape}")

# Permute - rearrange all dimensions
tensor_4d = torch.randn(2, 3, 4, 5)
print(f"4D original: {tensor_4d.shape}")

permuted = tensor_4d.permute(3, 1, 0, 2)
print(f"Permuted (3,1,0,2): {permuted.shape}")

# Common permutation patterns
batch_first = torch.randn(10, 3, 32, 32)  # NCHW format
channels_last = batch_first.permute(0, 2, 3, 1)  # NHWC format
print(f"NCHW: {batch_first.shape}")
print(f"NHWC: {channels_last.shape}")

print("\n=== Flatten Operations ===")

# Flatten tensor
tensor_to_flatten = torch.randn(2, 3, 4)
print(f"Original: {tensor_to_flatten.shape}")

# Flatten all dimensions
flattened_all = tensor_to_flatten.flatten()
print(f"Flattened all: {flattened_all.shape}")

# Flatten from specific dimension
flattened_from_1 = tensor_to_flatten.flatten(start_dim=1)
print(f"Flattened from dim 1: {flattened_from_1.shape}")

# Flatten between dimensions
tensor_4d = torch.randn(2, 3, 4, 5)
flattened_middle = tensor_4d.flatten(start_dim=1, end_dim=2)
print(f"4D original: {tensor_4d.shape}")
print(f"Flattened middle: {flattened_middle.shape}")

print("\n=== Expand and Repeat ===")

# Expand - broadcast to larger size (no memory copy)
small_tensor = torch.randn(1, 3)
print(f"Small tensor: {small_tensor.shape}")

expanded = small_tensor.expand(4, 3)
print(f"Expanded: {expanded.shape}")

# Expand with -1 (keep original size)
expanded_auto = small_tensor.expand(-1, 3)
print(f"Expanded auto: {expanded_auto.shape}")

# Repeat - actually copy data
repeated = small_tensor.repeat(4, 2)
print(f"Repeated: {repeated.shape}")

# Difference between expand and repeat
print("Expand shares memory, repeat creates new memory")
print(f"Expand uses same data: {expanded.data_ptr() == small_tensor.data_ptr()}")

print("\n=== Advanced Shape Operations ===")

# Chunk - split tensor into chunks
tensor_to_chunk = torch.randn(12, 4)
chunks = tensor_to_chunk.chunk(3, dim=0)
print(f"Original: {tensor_to_chunk.shape}")
print(f"Chunks: {[chunk.shape for chunk in chunks]}")

# Split - split with specific sizes
split_sizes = [3, 5, 4]
splits = tensor_to_chunk.split(split_sizes, dim=0)
print(f"Split sizes: {[split.shape for split in splits]}")

# Unfold - sliding window
tensor_for_unfold = torch.arange(10)
unfolded = tensor_for_unfold.unfold(0, 3, 1)  # window size 3, step 1
print(f"Unfolded: {unfolded}")

print("\n=== Shape Compatibility Checks ===")

# Check if shapes are compatible for operations
tensor_a = torch.randn(3, 4)
tensor_b = torch.randn(4, 5)
tensor_c = torch.randn(2, 4)

print(f"A shape: {tensor_a.shape}")
print(f"B shape: {tensor_b.shape}")
print(f"C shape: {tensor_c.shape}")

# Matrix multiplication compatibility
try:
    result_ab = torch.mm(tensor_a, tensor_b)
    print(f"A @ B result: {result_ab.shape}")
except RuntimeError as e:
    print(f"A @ B error: {e}")

try:
    result_ac = torch.mm(tensor_a, tensor_c)
    print(f"A @ C result: {result_ac.shape}")
except RuntimeError as e:
    print(f"A @ C error: {str(e)[:30]}...")

# Broadcasting compatibility
tensor_broadcast_a = torch.randn(3, 1, 4)
tensor_broadcast_b = torch.randn(1, 5, 4)

result_broadcast = tensor_broadcast_a + tensor_broadcast_b
print(f"Broadcast result: {result_broadcast.shape}")

print("\n=== Memory Layout and Shape ===")

# Contiguous vs non-contiguous
contiguous_tensor = torch.randn(3, 4)
non_contiguous = contiguous_tensor.transpose(0, 1)

print(f"Contiguous: {contiguous_tensor.is_contiguous()}")
print(f"Non-contiguous: {non_contiguous.is_contiguous()}")
print(f"Stride contiguous: {contiguous_tensor.stride()}")
print(f"Stride non-contiguous: {non_contiguous.stride()}")

# Make contiguous
made_contiguous = non_contiguous.contiguous()
print(f"Made contiguous: {made_contiguous.is_contiguous()}")

print("\n=== Shape Manipulation Complete ===") 