#!/usr/bin/env python3
"""PyTorch Batch Processing Syntax - Efficient batch processing"""

import torch
import torch.nn.functional as F
import time
import math

print("=== Batch Processing Overview ===")

print("Batch processing benefits:")
print("1. Better GPU utilization through parallelism")
print("2. More stable gradient estimates")
print("3. Efficient memory usage")
print("4. Vectorized operations")
print("5. Faster training and inference")

print("\n=== Basic Batch Operations ===")

# Single sample vs batch processing
single_sample = torch.randn(3, 32, 32)
batch_samples = torch.randn(16, 3, 32, 32)

print(f"Single sample shape: {single_sample.shape}")
print(f"Batch samples shape: {batch_samples.shape}")

# Batch-wise operations
batch_mean = batch_samples.mean(dim=0)  # Mean across batch
batch_std = batch_samples.std(dim=0)    # Std across batch
print(f"Batch mean shape: {batch_mean.shape}")
print(f"Batch std shape: {batch_std.shape}")

# Per-sample operations
per_sample_mean = batch_samples.mean(dim=[1, 2, 3])  # Mean per sample
per_sample_norm = batch_samples.norm(dim=[1, 2, 3])  # Norm per sample
print(f"Per-sample mean shape: {per_sample_mean.shape}")
print(f"Per-sample norm shape: {per_sample_norm.shape}")

print("\n=== Efficient Batch Matrix Operations ===")

# Batch matrix multiplication
batch_size = 32
matrix_a = torch.randn(batch_size, 10, 20)
matrix_b = torch.randn(batch_size, 20, 15)

# Batch matrix multiplication
batch_result = torch.bmm(matrix_a, matrix_b)
print(f"Batch matrices A: {matrix_a.shape}, B: {matrix_b.shape}")
print(f"Batch multiplication result: {batch_result.shape}")

# Broadcasting matrix multiplication
vector = torch.randn(20, 1)
broadcast_result = matrix_a @ vector  # Broadcasting
print(f"Broadcasting result: {broadcast_result.shape}")

# Einstein summation for complex operations
einsum_result = torch.einsum('bik,bkj->bij', matrix_a, matrix_b)
print(f"Einsum result shape: {einsum_result.shape}")
print(f"Einsum equals bmm: {torch.allclose(batch_result, einsum_result)}")

print("\n=== Batch Normalization Operations ===")

def batch_normalize_manual(x, eps=1e-5):
    """Manual batch normalization"""
    # Calculate batch statistics
    batch_mean = x.mean(dim=0, keepdim=True)
    batch_var = x.var(dim=0, keepdim=True, unbiased=False)
    
    # Normalize
    normalized = (x - batch_mean) / torch.sqrt(batch_var + eps)
    return normalized, batch_mean, batch_var

# Test batch normalization
batch_data = torch.randn(64, 128)
normalized, mean, var = batch_normalize_manual(batch_data)

print(f"Original batch mean: {batch_data.mean(dim=0)[:5]}")
print(f"Normalized batch mean: {normalized.mean(dim=0)[:5]}")
print(f"Original batch std: {batch_data.std(dim=0, unbiased=False)[:5]}")
print(f"Normalized batch std: {normalized.std(dim=0, unbiased=False)[:5]}")

print("\n=== Batch Convolution Operations ===")

# Batch convolution
batch_images = torch.randn(8, 3, 64, 64)
conv_kernel = torch.randn(16, 3, 5, 5)
conv_bias = torch.randn(16)

# Manual convolution
conv_result = F.conv2d(batch_images, conv_kernel, conv_bias, padding=2)
print(f"Batch convolution input: {batch_images.shape}")
print(f"Convolution output: {conv_result.shape}")

# Grouped convolution
grouped_conv = F.conv2d(batch_images, conv_kernel, conv_bias, padding=2, groups=1)
print(f"Grouped convolution output: {grouped_conv.shape}")

print("\n=== Batch Attention Mechanisms ===")

def scaled_dot_product_attention_batch(query, key, value, mask=None):
    """Batch scaled dot-product attention"""
    # query, key, value: (batch_size, seq_len, d_model)
    d_k = query.size(-1)
    
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scores.masked_fill_(mask == 0, -1e9)
    
    # Apply softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights

# Test batch attention
batch_size, seq_len, d_model = 4, 10, 64
query = torch.randn(batch_size, seq_len, d_model)
key = torch.randn(batch_size, seq_len, d_model)
value = torch.randn(batch_size, seq_len, d_model)

attention_output, attention_weights = scaled_dot_product_attention_batch(query, key, value)
print(f"Attention input shapes: Q{query.shape}, K{key.shape}, V{value.shape}")
print(f"Attention output: {attention_output.shape}")
print(f"Attention weights: {attention_weights.shape}")

print("\n=== Batch Sampling and Indexing ===")

# Batch indexing
data = torch.randn(100, 50)
batch_indices = torch.randint(0, 100, (32,))
sampled_batch = data[batch_indices]
print(f"Original data: {data.shape}")
print(f"Sampled batch: {sampled_batch.shape}")

# Advanced indexing with multiple dimensions
data_3d = torch.randn(20, 30, 40)
row_indices = torch.randint(0, 20, (8,))
col_indices = torch.randint(0, 30, (8,))
selected_elements = data_3d[row_indices, col_indices]
print(f"3D data: {data_3d.shape}")
print(f"Selected elements: {selected_elements.shape}")

# Batch gather operation
source = torch.randn(5, 10, 15)
indices = torch.randint(0, 10, (5, 3, 15))
gathered = torch.gather(source, 1, indices)
print(f"Gather source: {source.shape}")
print(f"Gathered result: {gathered.shape}")

print("\n=== Memory-Efficient Batch Processing ===")

def process_in_chunks(data, chunk_size, process_func):
    """Process large data in chunks to save memory"""
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        chunk_result = process_func(chunk)
        results.append(chunk_result)
    return torch.cat(results, dim=0)

# Example processing function
def expensive_operation(x):
    """Simulate an expensive operation"""
    return F.conv2d(x, torch.randn(64, x.size(1), 3, 3), padding=1)

# Large dataset simulation
large_data = torch.randn(1000, 3, 32, 32)

# Process in chunks
chunked_result = process_in_chunks(large_data, chunk_size=100, process_func=expensive_operation)
print(f"Large data: {large_data.shape}")
print(f"Chunked processing result: {chunked_result.shape}")

print("\n=== Batch Padding and Masking ===")

def pad_batch_sequences(sequences, pad_value=0):
    """Pad variable-length sequences for batch processing"""
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = []
    masks = []
    
    for seq in sequences:
        # Pad sequence
        pad_length = max_length - len(seq)
        if pad_length > 0:
            padding = torch.full((pad_length,) + seq.shape[1:], pad_value, dtype=seq.dtype)
            padded_seq = torch.cat([seq, padding], dim=0)
        else:
            padded_seq = seq
        
        # Create mask
        mask = torch.zeros(max_length, dtype=torch.bool)
        mask[:len(seq)] = True
        
        padded_sequences.append(padded_seq)
        masks.append(mask)
    
    return torch.stack(padded_sequences), torch.stack(masks)

# Test sequence padding
sequences = [
    torch.randn(5, 10),
    torch.randn(8, 10),
    torch.randn(3, 10),
    torch.randn(7, 10)
]

padded_seqs, masks = pad_batch_sequences(sequences)
print(f"Original sequence lengths: {[len(seq) for seq in sequences]}")
print(f"Padded sequences shape: {padded_seqs.shape}")
print(f"Masks shape: {masks.shape}")
print(f"First mask: {masks[0]}")

print("\n=== Batch Loss Computation ===")

def batch_cross_entropy(predictions, targets, reduction='mean'):
    """Batch cross-entropy loss computation"""
    # predictions: (batch_size, num_classes)
    # targets: (batch_size,)
    
    # Compute log probabilities
    log_probs = F.log_softmax(predictions, dim=1)
    
    # Gather log probabilities for true classes
    batch_size = predictions.size(0)
    true_log_probs = log_probs[torch.arange(batch_size), targets]
    
    # Compute loss
    loss = -true_log_probs
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

# Test batch loss
batch_predictions = torch.randn(32, 10)  # 32 samples, 10 classes
batch_targets = torch.randint(0, 10, (32,))

batch_loss = batch_cross_entropy(batch_predictions, batch_targets)
builtin_loss = F.cross_entropy(batch_predictions, batch_targets)

print(f"Batch predictions shape: {batch_predictions.shape}")
print(f"Batch targets shape: {batch_targets.shape}")
print(f"Custom batch loss: {batch_loss:.6f}")
print(f"Built-in loss: {builtin_loss:.6f}")
print(f"Losses are close: {torch.allclose(batch_loss, builtin_loss)}")

print("\n=== Batch Gradient Computation ===")

def compute_batch_gradients(model, data_batch, target_batch, loss_fn):
    """Compute gradients for a batch"""
    model.zero_grad()
    
    # Forward pass
    outputs = model(data_batch)
    loss = loss_fn(outputs, target_batch)
    
    # Backward pass
    loss.backward()
    
    # Extract gradients
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.clone()
    
    return loss.item(), gradients

# Simple model for testing
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
batch_input = torch.randn(16, 10)
batch_target = torch.randint(0, 5, (16,))

loss_value, grads = compute_batch_gradients(model, batch_input, batch_target, F.cross_entropy)
print(f"Batch loss: {loss_value:.6f}")
print(f"Gradient keys: {list(grads.keys())}")
print(f"Linear weight gradient shape: {grads['linear.weight'].shape}")

print("\n=== Batch Data Loading Simulation ===")

class BatchDataLoader:
    """Simple batch data loader"""
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = torch.arange(len(dataset))
    
    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(len(self.dataset))
            self.indices = self.indices[perm]
        
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch_data = self.dataset[batch_indices]
            yield batch_data
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

# Test batch data loader
dataset = torch.randn(100, 20)  # 100 samples, 20 features
data_loader = BatchDataLoader(dataset, batch_size=16, shuffle=True)

print(f"Dataset size: {len(dataset)}")
print(f"Number of batches: {len(data_loader)}")

batch_sizes = []
for i, batch in enumerate(data_loader):
    batch_sizes.append(len(batch))
    if i >= 3:  # Show first few batches
        break

print(f"First few batch sizes: {batch_sizes}")

print("\n=== Performance Comparison ===")

def time_operation(func, *args, **kwargs):
    """Time an operation"""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    result = func(*args, **kwargs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    return result, end_time - start_time

# Compare single vs batch processing
def single_processing(data):
    results = []
    for sample in data:
        result = F.relu(sample @ torch.randn(sample.size(-1), 10))
        results.append(result)
    return torch.stack(results)

def batch_processing(data):
    weight = torch.randn(data.size(-1), 10)
    return F.relu(data @ weight)

# Test data
test_data = torch.randn(100, 50)

# Time single processing
_, single_time = time_operation(single_processing, test_data)

# Time batch processing
_, batch_time = time_operation(batch_processing, test_data)

print(f"Single processing time: {single_time:.6f} seconds")
print(f"Batch processing time: {batch_time:.6f} seconds")
print(f"Speedup: {single_time / batch_time:.2f}x")

print("\n=== Batch Processing Best Practices ===")

print("Efficient Batching Guidelines:")
print("1. Use appropriate batch sizes (powers of 2 often work well)")
print("2. Maximize GPU memory utilization without overflow")
print("3. Use vectorized operations instead of loops")
print("4. Consider gradient accumulation for large effective batch sizes")
print("5. Handle variable-length sequences with padding and masking")
print("6. Use memory-efficient chunking for large datasets")
print("7. Leverage tensor broadcasting for efficient computations")

print("\nMemory Management:")
print("- Monitor GPU memory usage during training")
print("- Use gradient checkpointing for memory-intensive models")
print("- Clear intermediate results when not needed")
print("- Consider mixed precision training for memory savings")

print("\nOptimization Tips:")
print("- Batch operations across all dimensions when possible")
print("- Use in-place operations carefully (can break autograd)")
print("- Profile your code to identify bottlenecks")
print("- Consider data loading bottlenecks vs computation bottlenecks")

print("\n=== Batch Processing Complete ===") 