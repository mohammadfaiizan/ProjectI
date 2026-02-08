#!/usr/bin/env python3
"""PyTorch Tensor Concatenation and Splitting Operations"""

import torch

print("=== Concatenation Operations ===")

# Create sample tensors for concatenation
tensor_a = torch.tensor([[1, 2], [3, 4]])
tensor_b = torch.tensor([[5, 6], [7, 8]])
tensor_c = torch.tensor([[9, 10], [11, 12]])

print(f"Tensor A:\n{tensor_a}")
print(f"Tensor B:\n{tensor_b}")
print(f"Tensor C:\n{tensor_c}")

# Concatenate along dimension 0 (rows)
cat_dim0 = torch.cat([tensor_a, tensor_b, tensor_c], dim=0)
print(f"Concatenated along dim 0 (rows):\n{cat_dim0}")
print(f"Shape: {cat_dim0.shape}")

# Concatenate along dimension 1 (columns)
cat_dim1 = torch.cat([tensor_a, tensor_b, tensor_c], dim=1)
print(f"Concatenated along dim 1 (columns):\n{cat_dim1}")
print(f"Shape: {cat_dim1.shape}")

print("\n=== Stack Operations ===")

# Stack creates new dimension
stacked_dim0 = torch.stack([tensor_a, tensor_b, tensor_c], dim=0)
stacked_dim1 = torch.stack([tensor_a, tensor_b, tensor_c], dim=1)
stacked_dim2 = torch.stack([tensor_a, tensor_b, tensor_c], dim=2)

print(f"Stacked dim 0 shape: {stacked_dim0.shape}")
print(f"Stacked dim 1 shape: {stacked_dim1.shape}")
print(f"Stacked dim 2 shape: {stacked_dim2.shape}")

print(f"Stacked along dim 0:\n{stacked_dim0}")

# Stack vs cat comparison
print(f"Cat result shape: {cat_dim0.shape}")
print(f"Stack result shape: {stacked_dim0.shape}")

print("\n=== Multi-dimensional Concatenation ===")

# 3D tensor concatenation
tensor_3d_a = torch.randn(2, 3, 4)
tensor_3d_b = torch.randn(2, 3, 4)
tensor_3d_c = torch.randn(2, 3, 4)

# Concatenate along different dimensions
cat_3d_dim0 = torch.cat([tensor_3d_a, tensor_3d_b], dim=0)
cat_3d_dim1 = torch.cat([tensor_3d_a, tensor_3d_b], dim=1)
cat_3d_dim2 = torch.cat([tensor_3d_a, tensor_3d_b], dim=2)

print(f"3D original shape: {tensor_3d_a.shape}")
print(f"Cat dim 0 shape: {cat_3d_dim0.shape}")
print(f"Cat dim 1 shape: {cat_3d_dim1.shape}")
print(f"Cat dim 2 shape: {cat_3d_dim2.shape}")

# Stack 3D tensors
stack_3d_dim0 = torch.stack([tensor_3d_a, tensor_3d_b, tensor_3d_c], dim=0)
stack_3d_dim1 = torch.stack([tensor_3d_a, tensor_3d_b, tensor_3d_c], dim=1)

print(f"Stack 3D dim 0 shape: {stack_3d_dim0.shape}")
print(f"Stack 3D dim 1 shape: {stack_3d_dim1.shape}")

print("\n=== Chunk Operations ===")

# Create tensor to split
large_tensor = torch.arange(24).reshape(4, 6)
print(f"Large tensor to chunk:\n{large_tensor}")
print(f"Shape: {large_tensor.shape}")

# Chunk along dimension 0
chunks_dim0 = torch.chunk(large_tensor, chunks=2, dim=0)
print(f"Chunks along dim 0: {len(chunks_dim0)} chunks")
for i, chunk in enumerate(chunks_dim0):
    print(f"Chunk {i} shape: {chunk.shape}")

# Chunk along dimension 1
chunks_dim1 = torch.chunk(large_tensor, chunks=3, dim=1)
print(f"Chunks along dim 1: {len(chunks_dim1)} chunks")
for i, chunk in enumerate(chunks_dim1):
    print(f"Chunk {i} shape: {chunk.shape}")

# Uneven chunking
uneven_chunks = torch.chunk(large_tensor, chunks=3, dim=0)
print(f"Uneven chunks: {len(uneven_chunks)} chunks")
for i, chunk in enumerate(uneven_chunks):
    print(f"Uneven chunk {i} shape: {chunk.shape}")

print("\n=== Split Operations ===")

# Split with specific sizes
split_sizes = [2, 3, 1]
splits_dim1 = torch.split(large_tensor, split_sizes, dim=1)
print(f"Split with sizes {split_sizes}:")
for i, split in enumerate(splits_dim1):
    print(f"Split {i} shape: {split.shape}")

# Split with equal sizes
equal_splits = torch.split(large_tensor, 2, dim=0)
print(f"Equal splits of size 2:")
for i, split in enumerate(equal_splits):
    print(f"Equal split {i} shape: {split.shape}")

# Split tensor that doesn't divide evenly
uneven_tensor = torch.arange(25).reshape(5, 5)
uneven_splits = torch.split(uneven_tensor, 2, dim=0)
print(f"Uneven tensor splits:")
for i, split in enumerate(uneven_splits):
    print(f"Uneven split {i} shape: {split.shape}")

print("\n=== Concatenation with Different Shapes ===")

# Tensors with compatible shapes for concatenation
tensor_2x3 = torch.randn(2, 3)
tensor_1x3 = torch.randn(1, 3)
tensor_3x3 = torch.randn(3, 3)

# Concatenate tensors with different first dimension
mixed_cat = torch.cat([tensor_2x3, tensor_1x3, tensor_3x3], dim=0)
print(f"Mixed concatenation shape: {mixed_cat.shape}")

# Tensors for column concatenation
tensor_2x2 = torch.randn(2, 2)
tensor_2x4 = torch.randn(2, 4)
col_mixed_cat = torch.cat([tensor_2x2, tensor_2x4], dim=1)
print(f"Column mixed cat shape: {col_mixed_cat.shape}")

print("\n=== Advanced Stacking Patterns ===")

# List of tensors with different operations
tensors_list = [torch.randn(3, 4) for _ in range(5)]

# Stack all tensors
batch_tensor = torch.stack(tensors_list, dim=0)
print(f"Batch tensor shape: {batch_tensor.shape}")

# Create sequence tensor
sequence_tensor = torch.stack(tensors_list, dim=1)
print(f"Sequence tensor shape: {sequence_tensor.shape}")

# Nested stacking
nested_list = [[torch.randn(2, 2) for _ in range(3)] for _ in range(2)]
stacked_inner = [torch.stack(inner_list, dim=0) for inner_list in nested_list]
final_stacked = torch.stack(stacked_inner, dim=0)
print(f"Nested stacking final shape: {final_stacked.shape}")

print("\n=== Batch Operations ===")

# Batch processing with concatenation
batch_size = 4
samples = [torch.randn(3, 224, 224) for _ in range(batch_size)]

# Create batch
batch = torch.stack(samples, dim=0)
print(f"Image batch shape: {batch.shape}")

# Concatenate different batch sizes
batch1 = torch.randn(16, 3, 32, 32)
batch2 = torch.randn(8, 3, 32, 32)
combined_batch = torch.cat([batch1, batch2], dim=0)
print(f"Combined batch shape: {combined_batch.shape}")

print("\n=== Sequential Data Handling ===")

# Variable length sequences
seq1 = torch.randn(10, 128)  # 10 time steps
seq2 = torch.randn(8, 128)   # 8 time steps
seq3 = torch.randn(12, 128)  # 12 time steps

# Pad sequences to same length
max_len = max(seq1.size(0), seq2.size(0), seq3.size(0))
padded_seq1 = torch.cat([seq1, torch.zeros(max_len - seq1.size(0), 128)])
padded_seq2 = torch.cat([seq2, torch.zeros(max_len - seq2.size(0), 128)])
padded_seq3 = torch.cat([seq3, torch.zeros(max_len - seq3.size(0), 128)])

# Stack padded sequences
sequence_batch = torch.stack([padded_seq1, padded_seq2, padded_seq3], dim=0)
print(f"Sequence batch shape: {sequence_batch.shape}")

print("\n=== Feature Concatenation ===")

# Concatenating features from different sources
visual_features = torch.randn(32, 2048)  # Visual features
text_features = torch.randn(32, 768)     # Text features
audio_features = torch.randn(32, 512)    # Audio features

# Multimodal feature concatenation
multimodal_features = torch.cat([visual_features, text_features, audio_features], dim=1)
print(f"Multimodal features shape: {multimodal_features.shape}")

# Channel-wise concatenation for CNNs
conv_feat1 = torch.randn(32, 64, 56, 56)
conv_feat2 = torch.randn(32, 128, 56, 56)
conv_feat3 = torch.randn(32, 256, 56, 56)

concatenated_features = torch.cat([conv_feat1, conv_feat2, conv_feat3], dim=1)
print(f"Concatenated conv features shape: {concatenated_features.shape}")

print("\n=== Split for Parallel Processing ===")

# Split large tensor for parallel processing
large_data = torch.randn(1000, 512)
num_workers = 4
splits = torch.chunk(large_data, num_workers, dim=0)

print(f"Data split for {num_workers} workers:")
for i, split in enumerate(splits):
    print(f"Worker {i} data shape: {split.shape}")

# Split along feature dimension
feature_splits = torch.split(large_data, [128, 128, 256], dim=1)
print(f"Feature splits: {[split.shape for split in feature_splits]}")

print("\n=== Tensor Reconstruction ===")

# Split and reconstruct
original_tensor = torch.randn(6, 8)
print(f"Original tensor shape: {original_tensor.shape}")

# Split into parts
parts = torch.split(original_tensor, 2, dim=0)
print(f"Split into {len(parts)} parts")

# Reconstruct
reconstructed = torch.cat(parts, dim=0)
print(f"Reconstructed shape: {reconstructed.shape}")
print(f"Reconstruction successful: {torch.equal(original_tensor, reconstructed)}")

print("\n=== Memory Considerations ===")

# Memory usage comparison
tensors_to_concat = [torch.randn(100, 100) for _ in range(10)]

# Calculate original memory
original_memory = sum(t.numel() * t.element_size() for t in tensors_to_concat)

# Concatenate
concatenated = torch.cat(tensors_to_concat, dim=0)
concatenated_memory = concatenated.numel() * concatenated.element_size()

print(f"Original tensors memory: {original_memory / 1024:.2f} KB")
print(f"Concatenated memory: {concatenated_memory / 1024:.2f} KB")
print(f"Memory efficiency: {concatenated_memory / original_memory:.2f}")

print("\n=== Error Handling ===")

# Incompatible shapes for concatenation
tensor_incompatible = torch.randn(2, 4)
try:
    invalid_cat = torch.cat([tensor_a, tensor_incompatible], dim=0)
except RuntimeError as e:
    print(f"Incompatible cat error: {str(e)[:50]}...")

# Incompatible shapes for stacking
try:
    invalid_stack = torch.stack([tensor_a, tensor_incompatible], dim=0)
except RuntimeError as e:
    print(f"Incompatible stack error: {str(e)[:50]}...")

# Invalid dimension for operations
try:
    invalid_dim_cat = torch.cat([tensor_a, tensor_b], dim=3)
except IndexError as e:
    print(f"Invalid dimension error: {e}")

print("\n=== Performance Optimization ===")

import time

# Performance comparison: multiple cats vs single cat
many_tensors = [torch.randn(10, 10) for _ in range(100)]

# Multiple concatenations
start_time = time.time()
result_multi = many_tensors[0]
for tensor in many_tensors[1:]:
    result_multi = torch.cat([result_multi, tensor], dim=0)
multi_time = time.time() - start_time

# Single concatenation
start_time = time.time()
result_single = torch.cat(many_tensors, dim=0)
single_time = time.time() - start_time

print(f"Multiple cats time: {multi_time:.6f} seconds")
print(f"Single cat time: {single_time:.6f} seconds")
print(f"Single cat speedup: {multi_time / single_time:.2f}x")
print(f"Results equal: {torch.equal(result_multi, result_single)}")

print("\n=== Common Use Cases ===")

# Data loading and batching
def create_batch_from_samples(samples):
    """Create batch from list of samples"""
    return torch.stack(samples, dim=0)

# Feature fusion
def fuse_features(*feature_tensors):
    """Fuse multiple feature tensors"""
    return torch.cat(feature_tensors, dim=-1)

# Temporal sequence handling
def pad_and_stack_sequences(sequences):
    """Pad sequences to same length and stack"""
    max_len = max(seq.size(0) for seq in sequences)
    padded = []
    for seq in sequences:
        if seq.size(0) < max_len:
            padding = torch.zeros(max_len - seq.size(0), seq.size(1))
            padded_seq = torch.cat([seq, padding], dim=0)
        else:
            padded_seq = seq
        padded.append(padded_seq)
    return torch.stack(padded, dim=0)

# Example usage
sample_sequences = [torch.randn(i+5, 64) for i in range(5)]
batched_sequences = pad_and_stack_sequences(sample_sequences)
print(f"Batched sequences shape: {batched_sequences.shape}")

print("\n=== Concatenation and Splitting Complete ===") 