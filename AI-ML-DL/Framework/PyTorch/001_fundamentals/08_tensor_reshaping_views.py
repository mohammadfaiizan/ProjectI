#!/usr/bin/env python3
"""PyTorch Tensor Reshaping and View Operations"""

import torch

print("=== Basic Reshaping Operations ===")

# Create base tensor
base_tensor = torch.arange(24)
print(f"Base tensor: {base_tensor}")
print(f"Original shape: {base_tensor.shape}")

# Basic reshape
reshaped_2d = base_tensor.reshape(4, 6)
reshaped_3d = base_tensor.reshape(2, 3, 4)
reshaped_4d = base_tensor.reshape(2, 2, 2, 3)

print(f"2D reshape: {reshaped_2d.shape}")
print(f"3D reshape: {reshaped_3d.shape}")
print(f"4D reshape: {reshaped_4d.shape}")

# Using -1 for automatic dimension calculation
auto_2d = base_tensor.reshape(4, -1)
auto_3d = base_tensor.reshape(-1, 3, 2)

print(f"Auto 2D (-1): {auto_2d.shape}")
print(f"Auto 3D (-1): {auto_3d.shape}")

print("\n=== View vs Reshape ===")

# View - must be contiguous, shares memory
contiguous_tensor = torch.arange(12)
view_result = contiguous_tensor.view(3, 4)
print(f"View shape: {view_result.shape}")
print(f"Memory shared: {contiguous_tensor.data_ptr() == view_result.data_ptr()}")

# Modifying view affects original
view_result[0, 0] = 999
print(f"Original after view modification: {contiguous_tensor}")

# Reshape - works on non-contiguous tensors, may copy
non_contiguous = torch.randn(3, 4).t()  # Transpose makes non-contiguous
print(f"Non-contiguous tensor contiguous: {non_contiguous.is_contiguous()}")

# View would fail on non-contiguous
try:
    failed_view = non_contiguous.view(-1)
except RuntimeError as e:
    print(f"View error: {str(e)[:50]}...")

# Reshape works on non-contiguous
reshape_works = non_contiguous.reshape(-1)
print(f"Reshape on non-contiguous shape: {reshape_works.shape}")

print("\n=== Transpose Operations ===")

# 2D transpose
matrix = torch.randn(3, 4)
transposed = matrix.transpose(0, 1)
transposed_t = matrix.t()  # Shortcut for 2D

print(f"Original matrix: {matrix.shape}")
print(f"Transposed: {transposed.shape}")
print(f"Using .t(): {transposed_t.shape}")
print(f"Results equal: {torch.equal(transposed, transposed_t)}")

# Higher dimensional transpose
tensor_3d = torch.randn(2, 3, 4)
transposed_3d = tensor_3d.transpose(0, 2)
print(f"3D original: {tensor_3d.shape}")
print(f"3D transposed (0,2): {transposed_3d.shape}")

# Multiple transposes
double_transpose = tensor_3d.transpose(0, 1).transpose(1, 2)
print(f"Double transpose: {double_transpose.shape}")

print("\n=== Permute Operations ===")

# Permute - rearrange all dimensions
tensor_4d = torch.randn(2, 3, 4, 5)
print(f"4D original: {tensor_4d.shape}")

# Permute dimensions
permuted = tensor_4d.permute(3, 1, 0, 2)
print(f"Permuted (3,1,0,2): {permuted.shape}")

# Common permutation patterns
batch_first = torch.randn(10, 3, 32, 32)  # NCHW
channels_last = batch_first.permute(0, 2, 3, 1)  # NHWC
sequence_first = torch.randn(20, 32, 512)  # Seq, Batch, Features
batch_first_seq = sequence_first.permute(1, 0, 2)  # Batch, Seq, Features

print(f"NCHW to NHWC: {batch_first.shape} -> {channels_last.shape}")
print(f"Seq-first to batch-first: {sequence_first.shape} -> {batch_first_seq.shape}")

print("\n=== Squeeze and Unsqueeze ===")

# Squeeze - remove dimensions of size 1
tensor_with_ones = torch.randn(1, 3, 1, 4, 1)
print(f"Original with 1s: {tensor_with_ones.shape}")

# Squeeze all size-1 dimensions
squeezed_all = tensor_with_ones.squeeze()
print(f"Squeezed all: {squeezed_all.shape}")

# Squeeze specific dimensions
squeezed_0 = tensor_with_ones.squeeze(0)
squeezed_2 = tensor_with_ones.squeeze(2)
squeezed_4 = tensor_with_ones.squeeze(4)

print(f"Squeezed dim 0: {squeezed_0.shape}")
print(f"Squeezed dim 2: {squeezed_2.shape}")
print(f"Squeezed dim 4: {squeezed_4.shape}")

# Unsqueeze - add dimension of size 1
regular_tensor = torch.randn(3, 4)
print(f"Regular tensor: {regular_tensor.shape}")

# Add dimensions at different positions
unsqueezed_0 = regular_tensor.unsqueeze(0)
unsqueezed_1 = regular_tensor.unsqueeze(1)
unsqueezed_2 = regular_tensor.unsqueeze(2)
unsqueezed_neg1 = regular_tensor.unsqueeze(-1)

print(f"Unsqueezed 0: {unsqueezed_0.shape}")
print(f"Unsqueezed 1: {unsqueezed_1.shape}")
print(f"Unsqueezed 2: {unsqueezed_2.shape}")
print(f"Unsqueezed -1: {unsqueezed_neg1.shape}")

print("\n=== Flatten Operations ===")

# Flatten tensor
tensor_to_flatten = torch.randn(2, 3, 4, 5)
print(f"To flatten: {tensor_to_flatten.shape}")

# Flatten all dimensions
flattened_all = tensor_to_flatten.flatten()
print(f"Flattened all: {flattened_all.shape}")

# Flatten from specific dimension
flattened_from_1 = tensor_to_flatten.flatten(start_dim=1)
flattened_from_2 = tensor_to_flatten.flatten(start_dim=2)

print(f"Flattened from dim 1: {flattened_from_1.shape}")
print(f"Flattened from dim 2: {flattened_from_2.shape}")

# Flatten between dimensions
flattened_middle = tensor_to_flatten.flatten(start_dim=1, end_dim=2)
print(f"Flattened dims 1-2: {flattened_middle.shape}")

print("\n=== Expand and Repeat ===")

# Expand - broadcast to larger size (no memory copy)
small_tensor = torch.tensor([[1], [2], [3]])
print(f"Small tensor: {small_tensor.shape}")

# Expand along existing dimensions
expanded = small_tensor.expand(3, 4)
print(f"Expanded: {expanded.shape}")
print(f"Expanded tensor:\n{expanded}")

# Expand with -1 (keep original dimension)
expanded_partial = small_tensor.expand(-1, 5)
print(f"Expanded partial: {expanded_partial.shape}")

# expand_as - expand to match another tensor's shape
target_tensor = torch.randn(3, 6)
expanded_as = small_tensor.expand_as(target_tensor)
print(f"Expanded as target: {expanded_as.shape}")

# Repeat - actually copy data
repeated = small_tensor.repeat(2, 3)
print(f"Repeated: {repeated.shape}")
print(f"Repeated tensor:\n{repeated}")

# Difference in memory usage
print(f"Expand shares memory: {small_tensor.data_ptr() == expanded.data_ptr()}")
print(f"Repeat creates new memory: {small_tensor.data_ptr() != repeated.data_ptr()}")

print("\n=== Advanced Reshaping Patterns ===")

# Batch dimension manipulation
batch_tensor = torch.randn(32, 3, 224, 224)

# Add time dimension for video processing
video_tensor = batch_tensor.unsqueeze(1)  # Add time dim
print(f"Video tensor (N,T,C,H,W): {video_tensor.shape}")

# Reshape for linear layer input
feature_tensor = batch_tensor.flatten(start_dim=1)
print(f"Features for linear layer: {feature_tensor.shape}")

# Reshape for attention mechanisms
seq_length, batch_size, embed_dim = 50, 32, 512
attention_input = torch.randn(seq_length, batch_size, embed_dim)

# Reshape for multi-head attention
num_heads = 8
head_dim = embed_dim // num_heads
multi_head = attention_input.view(seq_length, batch_size, num_heads, head_dim)
print(f"Multi-head attention: {multi_head.shape}")

# Transpose for parallel head processing
heads_parallel = multi_head.permute(1, 2, 0, 3)  # B, H, S, D
print(f"Heads parallel: {heads_parallel.shape}")

print("\n=== Memory Layout Optimization ===")

# Contiguous memory layout
tensor_for_memory = torch.randn(4, 5, 6)
print(f"Original contiguous: {tensor_for_memory.is_contiguous()}")

# Non-contiguous after transpose
non_contiguous_mem = tensor_for_memory.transpose(0, 2)
print(f"After transpose contiguous: {non_contiguous_mem.is_contiguous()}")

# Make contiguous for performance
made_contiguous = non_contiguous_mem.contiguous()
print(f"Made contiguous: {made_contiguous.is_contiguous()}")

# Performance difference
import time

large_tensor = torch.randn(1000, 1000)
non_contig = large_tensor.t()

# Operation on contiguous tensor
start_time = time.time()
contig_result = large_tensor.sum()
contig_time = time.time() - start_time

# Operation on non-contiguous tensor
start_time = time.time()
non_contig_result = non_contig.sum()
non_contig_time = time.time() - start_time

print(f"Contiguous operation time: {contig_time:.6f}s")
print(f"Non-contiguous operation time: {non_contig_time:.6f}s")

print("\n=== Shape Validation and Error Handling ===")

# Invalid reshape
invalid_tensor = torch.randn(3, 4)
try:
    invalid_reshape = invalid_tensor.reshape(3, 5)  # Wrong total elements
except RuntimeError as e:
    print(f"Invalid reshape error: {str(e)[:50]}...")

# Multiple -1 in reshape
try:
    multi_neg_one = invalid_tensor.reshape(-1, -1)
except RuntimeError as e:
    print(f"Multiple -1 error: {str(e)[:50]}...")

# Invalid permutation
try:
    invalid_permute = invalid_tensor.permute(0, 2, 1)  # 3 dims for 2D tensor
except RuntimeError as e:
    print(f"Invalid permute error: {str(e)[:50]}...")

print("\n=== Common CNN Reshaping Patterns ===")

# CNN feature map reshaping
feature_maps = torch.randn(32, 256, 14, 14)  # Batch, Channels, Height, Width
print(f"CNN features: {feature_maps.shape}")

# Global average pooling simulation
global_avg_pool = feature_maps.mean(dim=[2, 3])  # Average over H,W
print(f"Global avg pool: {global_avg_pool.shape}")

# Reshape for classification head
classification_input = global_avg_pool.view(32, -1)
print(f"Classification input: {classification_input.shape}")

# Spatial flatten for transformer
spatial_flattened = feature_maps.flatten(start_dim=2)  # Keep batch and channel
print(f"Spatial flattened: {spatial_flattened.shape}")

# Sequence format for transformer
sequence_format = spatial_flattened.transpose(1, 2)  # B, S, C
print(f"Sequence format: {sequence_format.shape}")

print("\n=== RNN Reshaping Patterns ===")

# RNN input reshaping
batch_size, seq_len, input_size = 16, 20, 100
rnn_input = torch.randn(batch_size, seq_len, input_size)
print(f"RNN input (batch first): {rnn_input.shape}")

# Convert to sequence first for some RNN implementations
seq_first = rnn_input.transpose(0, 1)
print(f"RNN input (seq first): {seq_first.shape}")

# Bidirectional RNN output processing
hidden_size = 256
bidirectional_output = torch.randn(seq_len, batch_size, 2 * hidden_size)

# Split bidirectional output
forward_output = bidirectional_output[:, :, :hidden_size]
backward_output = bidirectional_output[:, :, hidden_size:]
print(f"Forward output: {forward_output.shape}")
print(f"Backward output: {backward_output.shape}")

print("\n=== Dynamic Reshaping ===")

# Reshape based on tensor properties
dynamic_tensor = torch.randn(6, 8)
batch_size = dynamic_tensor.size(0)
feature_size = dynamic_tensor.size(1)

# Create dynamic reshape
dynamic_reshape = dynamic_tensor.view(batch_size, 2, feature_size // 2)
print(f"Dynamic reshape: {dynamic_reshape.shape}")

# Conditional reshaping
if dynamic_tensor.size(1) % 4 == 0:
    quad_reshape = dynamic_tensor.view(batch_size, 4, -1)
    print(f"Quad reshape: {quad_reshape.shape}")

print("\n=== Best Practices ===")

print("Reshaping Best Practices:")
print("1. Use view() for contiguous tensors (faster)")
print("2. Use reshape() for potentially non-contiguous tensors")
print("3. Prefer expand() over repeat() when possible (memory efficient)")
print("4. Make tensors contiguous before intensive operations")
print("5. Use descriptive variable names for reshaped tensors")
print("6. Validate shapes after reshaping in debugging")
print("7. Consider memory layout for performance-critical code")
print("8. Use permute() for dimension reordering, transpose() for 2D swaps")

print("\n=== Reshaping and Views Complete ===") 