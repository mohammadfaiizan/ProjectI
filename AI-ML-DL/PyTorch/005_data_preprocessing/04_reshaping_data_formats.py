#!/usr/bin/env python3
"""PyTorch Reshaping Data Formats - Reshaping for different model inputs"""

import torch
import torch.nn.functional as F

print("=== Data Reshaping Overview ===")

print("Common reshaping operations:")
print("1. Basic shape manipulations (view, reshape)")
print("2. Dimension manipulations (squeeze, unsqueeze)")
print("3. Tensor concatenation and stacking")
print("4. Permutations and transpositions")
print("5. Format conversions (channel-first/last)")
print("6. Batch operations")

print("\n=== Basic Reshaping Operations ===")

# Create sample data
data = torch.randn(2, 3, 4, 5)
print(f"Original shape: {data.shape}")

# View vs reshape
viewed = data.view(2, -1)  # Flatten last 3 dimensions
reshaped = data.reshape(2, 60)  # Same result but different method

print(f"Viewed shape: {viewed.shape}")
print(f"Reshaped shape: {reshaped.shape}")
print(f"View and reshape are equal: {torch.equal(viewed, reshaped)}")

# View requires contiguous memory, reshape doesn't
transposed = data.transpose(1, 2)
print(f"Transposed shape: {transposed.shape}")
print(f"Is contiguous: {transposed.is_contiguous()}")

# This would fail: transposed.view(2, -1)
# But this works:
reshaped_transposed = transposed.reshape(2, -1)
print(f"Reshaped transposed shape: {reshaped_transposed.shape}")

# Make contiguous then view
contiguous_transposed = transposed.contiguous()
viewed_transposed = contiguous_transposed.view(2, -1)
print(f"Contiguous then viewed shape: {viewed_transposed.shape}")

print("\n=== Dimension Manipulations ===")

# Squeeze and unsqueeze
tensor_with_ones = torch.randn(1, 3, 1, 4, 1)
print(f"Tensor with size-1 dims: {tensor_with_ones.shape}")

# Remove all size-1 dimensions
squeezed_all = tensor_with_ones.squeeze()
print(f"Squeezed all: {squeezed_all.shape}")

# Remove specific size-1 dimensions
squeezed_dim0 = tensor_with_ones.squeeze(0)
squeezed_dim2 = tensor_with_ones.squeeze(2)
print(f"Squeezed dim 0: {squeezed_dim0.shape}")
print(f"Squeezed dim 2: {squeezed_dim2.shape}")

# Add dimensions
original = torch.randn(3, 4)
print(f"Original 2D: {original.shape}")

# Add dimension at different positions
unsqueezed_0 = original.unsqueeze(0)
unsqueezed_1 = original.unsqueeze(1)
unsqueezed_2 = original.unsqueeze(2)
unsqueezed_neg1 = original.unsqueeze(-1)

print(f"Unsqueezed at 0: {unsqueezed_0.shape}")
print(f"Unsqueezed at 1: {unsqueezed_1.shape}")
print(f"Unsqueezed at 2: {unsqueezed_2.shape}")
print(f"Unsqueezed at -1: {unsqueezed_neg1.shape}")

print("\n=== Tensor Concatenation and Stacking ===")

# Concatenation along existing dimensions
tensor1 = torch.randn(2, 3, 4)
tensor2 = torch.randn(2, 3, 4)
tensor3 = torch.randn(2, 3, 4)

# Concatenate along different dimensions
cat_dim0 = torch.cat([tensor1, tensor2], dim=0)  # (4, 3, 4)
cat_dim1 = torch.cat([tensor1, tensor2], dim=1)  # (2, 6, 4)
cat_dim2 = torch.cat([tensor1, tensor2], dim=2)  # (2, 3, 8)

print(f"Original tensor shape: {tensor1.shape}")
print(f"Cat along dim 0: {cat_dim0.shape}")
print(f"Cat along dim 1: {cat_dim1.shape}")
print(f"Cat along dim 2: {cat_dim2.shape}")

# Stacking creates new dimension
stack_dim0 = torch.stack([tensor1, tensor2, tensor3], dim=0)  # (3, 2, 3, 4)
stack_dim1 = torch.stack([tensor1, tensor2, tensor3], dim=1)  # (2, 3, 3, 4)
stack_dim2 = torch.stack([tensor1, tensor2, tensor3], dim=2)  # (2, 3, 3, 4)

print(f"Stack along dim 0: {stack_dim0.shape}")
print(f"Stack along dim 1: {stack_dim1.shape}")
print(f"Stack along dim 2: {stack_dim2.shape}")

print("\n=== Permutations and Transpositions ===")

# Transpose swaps two dimensions
matrix = torch.randn(3, 5)
transposed_matrix = matrix.transpose(0, 1)  # or matrix.T
print(f"Matrix shape: {matrix.shape}")
print(f"Transposed shape: {transposed_matrix.shape}")

# Permute reorders all dimensions
tensor_4d = torch.randn(2, 3, 4, 5)  # (batch, channel, height, width)
print(f"Original 4D tensor: {tensor_4d.shape}")

# Common permutations
nhwc_to_nchw = tensor_4d.permute(0, 3, 1, 2)  # (batch, width, channel, height)
channel_last = tensor_4d.permute(0, 2, 3, 1)  # (batch, height, width, channel)
reversed_dims = tensor_4d.permute(3, 2, 1, 0)  # Reverse all dimensions

print(f"NHWC to NCHW: {nhwc_to_nchw.shape}")
print(f"Channel last: {channel_last.shape}")
print(f"Reversed dims: {reversed_dims.shape}")

print("\n=== Format Conversions ===")

def convert_channel_format(tensor, from_format, to_format):
    """Convert between different channel formats"""
    if from_format == to_format:
        return tensor
    
    if from_format == "NCHW" and to_format == "NHWC":
        return tensor.permute(0, 2, 3, 1)
    elif from_format == "NHWC" and to_format == "NCHW":
        return tensor.permute(0, 3, 1, 2)
    elif from_format == "CHW" and to_format == "HWC":
        return tensor.permute(1, 2, 0)
    elif from_format == "HWC" and to_format == "CHW":
        return tensor.permute(2, 0, 1)
    else:
        raise ValueError(f"Unsupported conversion: {from_format} -> {to_format}")

# Test format conversions
nchw_tensor = torch.randn(4, 3, 32, 32)  # Batch, Channel, Height, Width
nhwc_tensor = convert_channel_format(nchw_tensor, "NCHW", "NHWC")
back_to_nchw = convert_channel_format(nhwc_tensor, "NHWC", "NCHW")

print(f"NCHW format: {nchw_tensor.shape}")
print(f"NHWC format: {nhwc_tensor.shape}")
print(f"Back to NCHW: {back_to_nchw.shape}")
print(f"Conversion is lossless: {torch.equal(nchw_tensor, back_to_nchw)}")

print("\n=== Sequence Data Reshaping ===")

# Common sequence data operations
batch_size, seq_len, hidden_size = 4, 10, 256

# Sequence tensor
sequences = torch.randn(batch_size, seq_len, hidden_size)
print(f"Sequence tensor: {sequences.shape}")

# Flatten for linear layer
flattened_seq = sequences.view(batch_size, -1)
print(f"Flattened sequence: {flattened_seq.shape}")

# Reshape for CNN (add channel dimension)
cnn_input = sequences.unsqueeze(1)  # Add channel dimension
print(f"CNN input format: {cnn_input.shape}")

# Transpose for attention (seq_len first)
attention_input = sequences.transpose(0, 1)  # (seq_len, batch, hidden)
print(f"Attention input format: {attention_input.shape}")

# Pack sequences (for variable length)
lengths = torch.tensor([10, 8, 6, 9])  # Variable lengths for each sample
packed_seq = torch.nn.utils.rnn.pack_padded_sequence(sequences, lengths, batch_first=True, enforce_sorted=False)
print(f"Packed sequence data shape: {packed_seq.data.shape}")
print(f"Packed sequence batch sizes: {packed_seq.batch_sizes[:5]}")

print("\n=== Time Series Reshaping ===")

# Time series data
time_series = torch.randn(100, 5)  # 100 time steps, 5 features
print(f"Time series shape: {time_series.shape}")

# Create sliding windows
window_size = 10
stride = 1

def create_windows(data, window_size, stride=1):
    """Create sliding windows from time series"""
    windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data[i:i + window_size])
    return torch.stack(windows)

windowed_data = create_windows(time_series, window_size, stride)
print(f"Windowed data shape: {windowed_data.shape}")  # (num_windows, window_size, features)

# Reshape for different model types
# For RNN: (batch, seq, features) - already correct
rnn_format = windowed_data
print(f"RNN format: {rnn_format.shape}")

# For CNN: (batch, channels, length)
cnn_format = windowed_data.transpose(1, 2)
print(f"CNN format: {cnn_format.shape}")

# For transformer: (seq, batch, features)
transformer_format = windowed_data.transpose(0, 1)
print(f"Transformer format: {transformer_format.shape}")

print("\n=== Graph Data Reshaping ===")

# Simulating graph data
num_nodes = 20
num_features = 16
num_edges = 50

# Node features
node_features = torch.randn(num_nodes, num_features)
print(f"Node features: {node_features.shape}")

# Edge indices (COO format)
edge_indices = torch.randint(0, num_nodes, (2, num_edges))
print(f"Edge indices: {edge_indices.shape}")

# Convert to adjacency matrix
def edges_to_adjacency(edge_indices, num_nodes):
    """Convert edge indices to adjacency matrix"""
    adj_matrix = torch.zeros(num_nodes, num_nodes)
    adj_matrix[edge_indices[0], edge_indices[1]] = 1
    return adj_matrix

adjacency_matrix = edges_to_adjacency(edge_indices, num_nodes)
print(f"Adjacency matrix: {adjacency_matrix.shape}")

# Batch graph data
batch_size = 8
batched_node_features = node_features.unsqueeze(0).expand(batch_size, -1, -1)
print(f"Batched node features: {batched_node_features.shape}")

print("\n=== Advanced Reshaping Techniques ===")

# Fancy indexing and advanced slicing
data_3d = torch.randn(4, 5, 6)
print(f"3D data shape: {data_3d.shape}")

# Select specific indices
indices = torch.tensor([0, 2, 3])
selected = data_3d[indices]
print(f"Selected by indices: {selected.shape}")

# Boolean masking
mask = torch.randint(0, 2, (4,), dtype=torch.bool)
masked = data_3d[mask]
print(f"Boolean masked: {masked.shape}")

# Gather operation
gather_indices = torch.tensor([[0, 1], [2, 3], [1, 4], [0, 2]])
gathered = torch.gather(data_3d, 1, gather_indices.unsqueeze(-1).expand(-1, -1, 6))
print(f"Gathered data: {gathered.shape}")

# Index select
selected_features = torch.index_select(data_3d, dim=2, index=torch.tensor([0, 2, 4]))
print(f"Selected features: {selected_features.shape}")

print("\n=== Memory-Efficient Reshaping ===")

def efficient_reshape_large_tensor(tensor, new_shape):
    """Efficiently reshape large tensors"""
    # Check if reshape is possible
    if tensor.numel() != torch.prod(torch.tensor(new_shape)):
        raise ValueError("Total elements must remain the same")
    
    # Use view if tensor is contiguous
    if tensor.is_contiguous():
        return tensor.view(new_shape)
    else:
        # Make contiguous first (costs memory but safer)
        return tensor.contiguous().view(new_shape)

# Test with large tensor simulation
large_tensor = torch.randn(1000, 200)
print(f"Large tensor shape: {large_tensor.shape}")

# Efficient reshape
reshaped_large = efficient_reshape_large_tensor(large_tensor, (200, 1000))
print(f"Efficiently reshaped: {reshaped_large.shape}")

print("\n=== Batch Processing Utilities ===")

def pad_batch(tensors, pad_value=0):
    """Pad tensors to same size for batching"""
    max_shape = []
    for dim in range(len(tensors[0].shape)):
        max_size = max(tensor.shape[dim] for tensor in tensors)
        max_shape.append(max_size)
    
    padded_tensors = []
    for tensor in tensors:
        pad_widths = []
        for dim in range(len(tensor.shape)):
            pad_width = max_shape[dim] - tensor.shape[dim]
            pad_widths.extend([0, pad_width])
        
        # Reverse pad_widths for F.pad (last dim first)
        pad_widths = pad_widths[::-1]
        padded = F.pad(tensor, pad_widths, value=pad_value)
        padded_tensors.append(padded)
    
    return torch.stack(padded_tensors)

# Test padding
variable_tensors = [
    torch.randn(3, 10),
    torch.randn(3, 15),
    torch.randn(3, 8),
    torch.randn(3, 12)
]

padded_batch = pad_batch(variable_tensors)
print(f"Variable tensor shapes: {[t.shape for t in variable_tensors]}")
print(f"Padded batch shape: {padded_batch.shape}")

print("\n=== Reshaping Best Practices ===")

print("Reshaping Guidelines:")
print("1. Use view() for contiguous tensors (faster)")
print("2. Use reshape() for non-contiguous tensors (safer)")
print("3. Check tensor.is_contiguous() when performance matters")
print("4. Use squeeze/unsqueeze for adding/removing singleton dimensions")
print("5. Prefer permute() over transpose() for multi-dimensional reordering")
print("6. Use torch.cat() vs torch.stack() appropriately")
print("7. Be mindful of memory layout (row-major vs column-major)")

print("\nCommon Pitfalls:")
print("- view() fails on non-contiguous tensors")
print("- Forgetting batch dimension in reshaping")
print("- Incorrect dimension ordering for different frameworks")
print("- Memory inefficiency from unnecessary copying")
print("- Shape mismatches in concatenation operations")

print("\nOptimization Tips:")
print("- Minimize reshape operations in training loops")
print("- Use in-place operations when possible")
print("- Batch operations to reduce overhead")
print("- Consider memory layout for cache efficiency")

print("\n=== Reshaping Data Formats Complete ===") 