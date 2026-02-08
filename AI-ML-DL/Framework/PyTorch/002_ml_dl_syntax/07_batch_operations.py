#!/usr/bin/env python3
"""PyTorch Batch Operations - Batch processing and batch dimensions"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== Basic Batch Concepts ===")

# Standard batch format: (batch_size, features)
batch_size = 32
features = 784
batch_data = torch.randn(batch_size, features)

print(f"Batch data shape: {batch_data.shape}")
print(f"Batch size: {batch_data.size(0)}")
print(f"Feature size: {batch_data.size(1)}")

# Individual samples
sample_0 = batch_data[0]  # First sample
sample_batch = batch_data[:5]  # First 5 samples
print(f"Single sample shape: {sample_0.shape}")
print(f"Sub-batch shape: {sample_batch.shape}")

print("\n=== Batch Operations on Tensors ===")

# Element-wise operations preserve batch dimension
batch_doubled = batch_data * 2
batch_squared = torch.square(batch_data)
batch_sum = batch_data + torch.randn_like(batch_data)

print(f"Batch operations preserve shape: {batch_doubled.shape}")

# Reduction operations across batch
batch_mean = torch.mean(batch_data, dim=0)  # Mean across batch
batch_std = torch.std(batch_data, dim=0)    # Std across batch
batch_sum_total = torch.sum(batch_data, dim=0)  # Sum across batch

print(f"Batch mean shape (per feature): {batch_mean.shape}")
print(f"Batch std shape (per feature): {batch_std.shape}")

# Reduction within each sample
sample_means = torch.mean(batch_data, dim=1)  # Mean per sample
sample_norms = torch.norm(batch_data, dim=1)  # Norm per sample

print(f"Per-sample means shape: {sample_means.shape}")
print(f"Per-sample norms shape: {sample_norms.shape}")

print("\n=== Batch Processing with Neural Networks ===")

# Simple linear layer
linear_layer = nn.Linear(784, 256)
batch_output = linear_layer(batch_data)

print(f"Linear layer input: {batch_data.shape}")
print(f"Linear layer output: {batch_output.shape}")

# Convolutional layers with batches
conv_batch = torch.randn(32, 3, 224, 224)  # (N, C, H, W)
conv_layer = nn.Conv2d(3, 64, kernel_size=3, padding=1)
conv_output = conv_layer(conv_batch)

print(f"Conv input: {conv_batch.shape}")
print(f"Conv output: {conv_output.shape}")

# RNN with batches (batch_first=True)
rnn_batch = torch.randn(32, 50, 128)  # (batch, seq_len, features)
lstm_layer = nn.LSTM(128, 256, batch_first=True)
lstm_output, (hidden, cell) = lstm_layer(rnn_batch)

print(f"LSTM input: {rnn_batch.shape}")
print(f"LSTM output: {lstm_output.shape}")
print(f"LSTM hidden: {hidden.shape}")

print("\n=== Batch Normalization ===")

# Batch normalization across batch dimension
bn_layer = nn.BatchNorm1d(256)
bn_input = torch.randn(32, 256)

# Training mode - uses batch statistics
bn_layer.train()
bn_output_train = bn_layer(bn_input)

print(f"BatchNorm training mode:")
print(f"  Input mean: {bn_input.mean(dim=0)[:5]}")  # First 5 features
print(f"  Output mean: {bn_output_train.mean(dim=0)[:5]}")
print(f"  Running mean: {bn_layer.running_mean[:5]}")

# Evaluation mode - uses running statistics
bn_layer.eval()
bn_output_eval = bn_layer(bn_input)

print(f"BatchNorm eval mode uses running statistics")

print("\n=== Batch Dimension Manipulation ===")

# Unsqueeze to add batch dimension
single_sample = torch.randn(784)
batched_sample = single_sample.unsqueeze(0)  # Add batch dimension

print(f"Single sample: {single_sample.shape}")
print(f"Batched sample: {batched_sample.shape}")

# Squeeze to remove batch dimension
unbatched = batched_sample.squeeze(0)
print(f"Unbatched: {unbatched.shape}")

# Stack multiple samples into batch
samples = [torch.randn(784) for _ in range(5)]
stacked_batch = torch.stack(samples, dim=0)
print(f"Stacked batch: {stacked_batch.shape}")

# Concatenate batches
batch1 = torch.randn(16, 784)
batch2 = torch.randn(16, 784)
combined_batch = torch.cat([batch1, batch2], dim=0)
print(f"Combined batch: {combined_batch.shape}")

print("\n=== Batch Processing for Different Data Types ===")

# Image batches (NCHW format)
image_batch = torch.randn(32, 3, 224, 224)
print(f"Image batch: {image_batch.shape} (N, C, H, W)")

# Process individual images
for i in range(min(3, image_batch.size(0))):
    single_image = image_batch[i]  # Shape: (3, 224, 224)
    print(f"Image {i} shape: {single_image.shape}")

# Sequence batches
sequence_batch = torch.randn(32, 100, 512)  # (batch, seq_len, features)
print(f"Sequence batch: {sequence_batch.shape}")

# Variable length sequences (with padding)
def pad_sequences(sequences, max_length=None, padding_value=0):
    """Pad sequences to same length"""
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    padded = []
    for seq in sequences:
        if len(seq) < max_length:
            padding = torch.full((max_length - len(seq), seq.size(-1)), padding_value)
            padded_seq = torch.cat([seq, padding], dim=0)
        else:
            padded_seq = seq[:max_length]
        padded.append(padded_seq)
    
    return torch.stack(padded)

# Test variable length sequences
var_sequences = [
    torch.randn(10, 128),   # Length 10
    torch.randn(15, 128),   # Length 15
    torch.randn(8, 128),    # Length 8
]

padded_batch = pad_sequences(var_sequences)
print(f"Padded sequences: {padded_batch.shape}")

print("\n=== Mini-batch Training ===")

# Simulate dataset and mini-batch processing
def create_mini_batches(data, targets, batch_size):
    """Create mini-batches from data"""
    dataset_size = data.size(0)
    indices = torch.randperm(dataset_size)  # Shuffle
    
    batches = []
    for i in range(0, dataset_size, batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_data = data[batch_indices]
        batch_targets = targets[batch_indices]
        batches.append((batch_data, batch_targets))
    
    return batches

# Create dummy dataset
dataset_size = 1000
full_data = torch.randn(dataset_size, 784)
full_targets = torch.randint(0, 10, (dataset_size,))

# Create mini-batches
mini_batches = create_mini_batches(full_data, full_targets, batch_size=32)
print(f"Created {len(mini_batches)} mini-batches")
print(f"First batch data shape: {mini_batches[0][0].shape}")
print(f"First batch targets shape: {mini_batches[0][1].shape}")

# Last batch might be smaller
print(f"Last batch data shape: {mini_batches[-1][0].shape}")

print("\n=== Batch-wise Loss Computation ===")

# Model and loss function
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

criterion = nn.CrossEntropyLoss()

# Batch forward pass
batch_data = torch.randn(32, 784)
batch_targets = torch.randint(0, 10, (32,))

batch_logits = model(batch_data)
batch_loss = criterion(batch_logits, batch_targets)

print(f"Batch logits: {batch_logits.shape}")
print(f"Batch loss: {batch_loss.item():.4f}")

# Individual sample losses
individual_losses = F.cross_entropy(batch_logits, batch_targets, reduction='none')
print(f"Individual losses shape: {individual_losses.shape}")
print(f"Mean of individual losses: {individual_losses.mean().item():.4f}")

print("\n=== Dynamic Batching ===")

class DynamicBatcher:
    """Handle variable batch sizes dynamically"""
    
    def __init__(self, model, max_batch_size=32):
        self.model = model
        self.max_batch_size = max_batch_size
    
    def forward(self, data):
        """Process data in dynamic batches"""
        if data.size(0) <= self.max_batch_size:
            return self.model(data)
        
        # Split into smaller batches
        outputs = []
        for i in range(0, data.size(0), self.max_batch_size):
            batch = data[i:i+self.max_batch_size]
            batch_output = self.model(batch)
            outputs.append(batch_output)
        
        return torch.cat(outputs, dim=0)

# Test dynamic batching
dynamic_batcher = DynamicBatcher(model, max_batch_size=16)

# Small batch (fits in one go)
small_data = torch.randn(10, 784)
small_output = dynamic_batcher.forward(small_data)
print(f"Small batch output: {small_output.shape}")

# Large batch (split into smaller batches)
large_data = torch.randn(50, 784)
large_output = dynamic_batcher.forward(large_data)
print(f"Large batch output: {large_output.shape}")

print("\n=== Batch Augmentation ===")

def batch_augmentation(batch, noise_std=0.1, dropout_prob=0.1):
    """Apply augmentation to entire batch"""
    augmented = batch.clone()
    
    # Add noise
    noise = torch.randn_like(batch) * noise_std
    augmented += noise
    
    # Random dropout
    mask = torch.rand_like(batch) > dropout_prob
    augmented *= mask.float()
    
    return augmented

# Test batch augmentation
original_batch = torch.randn(32, 784)
augmented_batch = batch_augmentation(original_batch)

print(f"Original batch mean: {original_batch.mean():.4f}")
print(f"Augmented batch mean: {augmented_batch.mean():.4f}")

print("\n=== Batch Statistics and Monitoring ===")

def analyze_batch(batch, name="Batch"):
    """Analyze batch statistics"""
    print(f"\n{name} Analysis:")
    print(f"  Shape: {batch.shape}")
    print(f"  Mean: {batch.mean():.4f}")
    print(f"  Std: {batch.std():.4f}")
    print(f"  Min: {batch.min():.4f}")
    print(f"  Max: {batch.max():.4f}")
    
    # Check for NaN or Inf
    has_nan = torch.isnan(batch).any()
    has_inf = torch.isinf(batch).any()
    print(f"  Has NaN: {has_nan}")
    print(f"  Has Inf: {has_inf}")

# Analyze different batches
analyze_batch(batch_data, "Input Batch")
analyze_batch(batch_logits, "Output Batch")

print("\n=== Memory-Efficient Batch Processing ===")

def memory_efficient_batch_process(large_data, model, batch_size=32):
    """Process large data in memory-efficient batches"""
    model.eval()
    results = []
    
    with torch.no_grad():
        for i in range(0, large_data.size(0), batch_size):
            batch = large_data[i:i+batch_size]
            batch_result = model(batch)
            
            # Move to CPU to save GPU memory
            if batch_result.is_cuda:
                batch_result = batch_result.cpu()
            
            results.append(batch_result)
            
            # Clear cache periodically
            if i % (batch_size * 10) == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return torch.cat(results, dim=0)

# Test memory-efficient processing
large_test_data = torch.randn(1000, 784)
efficient_results = memory_efficient_batch_process(large_test_data, model)
print(f"Efficient processing result: {efficient_results.shape}")

print("\n=== Batch Gradient Accumulation ===")

def train_with_gradient_accumulation(model, data_loader, optimizer, 
                                   accumulation_steps=4):
    """Train with gradient accumulation for effective larger batches"""
    model.train()
    optimizer.zero_grad()
    
    total_loss = 0
    for i, (data, target) in enumerate(data_loader):
        # Forward pass
        output = model(data)
        loss = criterion(output, target) / accumulation_steps
        
        # Backward pass
        loss.backward()
        total_loss += loss.item()
        
        # Update weights every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    return total_loss

# Simulate gradient accumulation
class FakeDataLoader:
    def __init__(self, num_batches=8):
        self.num_batches = num_batches
        self.current = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current < self.num_batches:
            self.current += 1
            data = torch.randn(8, 784)  # Small batches
            target = torch.randint(0, 10, (8,))
            return data, target
        else:
            self.current = 0
            raise StopIteration

fake_loader = FakeDataLoader()
optimizer = torch.optim.Adam(model.parameters())

accumulated_loss = train_with_gradient_accumulation(
    model, fake_loader, optimizer, accumulation_steps=2
)
print(f"Training with gradient accumulation completed, loss: {accumulated_loss:.4f}")

print("\n=== Batch Operations Best Practices ===")

print("Batch Processing Guidelines:")
print("1. Always use batch dimension as first dimension (N, ...)")
print("2. Keep batch size consistent within training loop")
print("3. Handle variable batch sizes (last batch may be smaller)")
print("4. Use torch.stack() to create batches from samples")
print("5. Use torch.cat() to concatenate batches")
print("6. Monitor batch statistics for debugging")
print("7. Use gradient accumulation for memory-constrained training")

print("\nMemory Optimization:")
print("- Process large datasets in smaller batches")
print("- Move results to CPU when GPU memory is limited")
print("- Clear CUDA cache periodically")
print("- Use torch.no_grad() during inference")
print("- Consider mixed precision training")

print("\nCommon Patterns:")
print("- Images: (N, C, H, W)")
print("- Sequences: (N, L, D) with batch_first=True")
print("- Tabular: (N, features)")
print("- Variable lengths: Use padding + attention masks")

print("\n=== Batch Operations Complete ===") 