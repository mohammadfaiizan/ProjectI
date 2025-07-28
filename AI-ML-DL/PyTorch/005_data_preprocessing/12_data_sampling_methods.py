#!/usr/bin/env python3
"""PyTorch Data Sampling Methods - Sampling techniques with tensors"""

import torch
import random
import math

print("=== Data Sampling Overview ===")

print("Sampling techniques:")
print("1. Random sampling (uniform, weighted)")
print("2. Stratified sampling")
print("3. Systematic sampling")
print("4. Bootstrap sampling")
print("5. Importance sampling")
print("6. Rejection sampling")
print("7. Reservoir sampling")

print("\n=== Basic Random Sampling ===")

def uniform_sampling(data, n_samples, seed=None):
    """Uniform random sampling without replacement"""
    if seed is not None:
        torch.manual_seed(seed)
    
    if n_samples > len(data):
        raise ValueError(f"Cannot sample {n_samples} from {len(data)} items")
    
    indices = torch.randperm(len(data))[:n_samples]
    return data[indices], indices

def uniform_sampling_with_replacement(data, n_samples, seed=None):
    """Uniform random sampling with replacement"""
    if seed is not None:
        torch.manual_seed(seed)
    
    indices = torch.randint(0, len(data), (n_samples,))
    return data[indices], indices

def weighted_sampling(data, weights, n_samples, replacement=True, seed=None):
    """Weighted random sampling"""
    if seed is not None:
        torch.manual_seed(seed)
    
    # Normalize weights
    weights = weights / weights.sum()
    
    if replacement:
        # Multinomial sampling
        indices = torch.multinomial(weights, n_samples, replacement=True)
    else:
        # Sampling without replacement (approximate)
        indices = []
        remaining_indices = torch.arange(len(data))
        remaining_weights = weights.clone()
        
        for _ in range(min(n_samples, len(data))):
            # Sample one index
            idx = torch.multinomial(remaining_weights, 1)
            actual_idx = remaining_indices[idx]
            indices.append(actual_idx)
            
            # Remove sampled index
            mask = torch.ones(len(remaining_indices), dtype=torch.bool)
            mask[idx] = False
            remaining_indices = remaining_indices[mask]
            remaining_weights = remaining_weights[mask]
            
            if len(remaining_weights) > 0:
                remaining_weights = remaining_weights / remaining_weights.sum()
        
        indices = torch.stack(indices).squeeze()
    
    return data[indices], indices

# Test basic sampling
sample_data = torch.randn(1000, 10)

# Uniform sampling
uniform_sample, uniform_idx = uniform_sampling(sample_data, 100, seed=42)
print(f"Original data: {sample_data.shape}")
print(f"Uniform sample: {uniform_sample.shape}")

# Weighted sampling
weights = torch.softmax(torch.randn(1000), dim=0)  # Random weights
weighted_sample, weighted_idx = weighted_sampling(sample_data, weights, 100, seed=42)
print(f"Weighted sample: {weighted_sample.shape}")

print("\n=== Stratified Sampling ===")

def stratified_sampling(data, labels, n_samples_per_class=None, proportional=True, seed=None):
    """Stratified sampling preserving class distribution"""
    if seed is not None:
        torch.manual_seed(seed)
    
    unique_classes = torch.unique(labels)
    sampled_indices = []
    
    if proportional and n_samples_per_class is None:
        # Calculate proportional sample sizes
        total_samples = len(data)
        class_counts = [(labels == cls).sum().item() for cls in unique_classes]
        total_count = sum(class_counts)
        
        # Default to 10% of data if not specified
        target_total = total_samples // 10
        n_samples_per_class = [max(1, int(count * target_total / total_count)) for count in class_counts]
    elif n_samples_per_class is None:
        # Equal samples per class
        n_samples_per_class = [50] * len(unique_classes)
    elif isinstance(n_samples_per_class, int):
        # Same number for all classes
        n_samples_per_class = [n_samples_per_class] * len(unique_classes)
    
    for i, cls in enumerate(unique_classes):
        # Get indices for this class
        class_indices = torch.where(labels == cls)[0]
        n_class_samples = min(n_samples_per_class[i], len(class_indices))
        
        # Sample from this class
        if n_class_samples > 0:
            perm = torch.randperm(len(class_indices))[:n_class_samples]
            selected_indices = class_indices[perm]
            sampled_indices.extend(selected_indices.tolist())
    
    sampled_indices = torch.tensor(sampled_indices)
    return data[sampled_indices], labels[sampled_indices], sampled_indices

# Test stratified sampling
# Create imbalanced dataset
imbalanced_labels = torch.cat([
    torch.zeros(700),      # Class 0: 70%
    torch.ones(200),       # Class 1: 20%
    torch.full((100,), 2)  # Class 2: 10%
])
imbalanced_data = torch.randn(1000, 5)

strat_data, strat_labels, strat_idx = stratified_sampling(
    imbalanced_data, imbalanced_labels, proportional=True, seed=42
)

print("Original class distribution:")
for cls in torch.unique(imbalanced_labels):
    count = (imbalanced_labels == cls).sum()
    print(f"  Class {cls.int()}: {count} ({count/len(imbalanced_labels)*100:.1f}%)")

print("Stratified sample distribution:")
for cls in torch.unique(strat_labels):
    count = (strat_labels == cls).sum()
    print(f"  Class {cls.int()}: {count} ({count/len(strat_labels)*100:.1f}%)")

print("\n=== Systematic Sampling ===")

def systematic_sampling(data, k=None, n_samples=None, start_random=True, seed=None):
    """Systematic sampling with fixed interval"""
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    
    n = len(data)
    
    if k is None and n_samples is not None:
        k = n // n_samples
    elif k is None:
        k = 10  # Default interval
    
    if start_random:
        start_idx = random.randint(0, k - 1)
    else:
        start_idx = 0
    
    # Generate systematic indices
    indices = torch.arange(start_idx, n, k)
    return data[indices], indices

# Test systematic sampling
sys_sample, sys_idx = systematic_sampling(sample_data, n_samples=50, seed=42)
print(f"Systematic sample: {sys_sample.shape}")
print(f"Sample indices: {sys_idx[:10]}...")  # Show first 10 indices

print("\n=== Bootstrap Sampling ===")

def bootstrap_sample(data, n_bootstrap=1000, sample_size=None, seed=None):
    """Generate bootstrap samples"""
    if seed is not None:
        torch.manual_seed(seed)
    
    n = len(data)
    if sample_size is None:
        sample_size = n
    
    bootstrap_samples = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = torch.randint(0, n, (sample_size,))
        bootstrap_samples.append(data[indices])
    
    return bootstrap_samples

def bootstrap_confidence_interval(data, statistic_fn, confidence=0.95, n_bootstrap=1000, seed=None):
    """Compute bootstrap confidence interval for a statistic"""
    bootstrap_samples = bootstrap_sample(data, n_bootstrap, seed=seed)
    
    # Compute statistic for each bootstrap sample
    bootstrap_stats = []
    for sample in bootstrap_samples:
        stat = statistic_fn(sample)
        bootstrap_stats.append(stat.item() if torch.is_tensor(stat) else stat)
    
    bootstrap_stats = torch.tensor(bootstrap_stats)
    
    # Compute confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = torch.quantile(bootstrap_stats, lower_percentile / 100)
    ci_upper = torch.quantile(bootstrap_stats, upper_percentile / 100)
    
    return ci_lower.item(), ci_upper.item(), bootstrap_stats

# Test bootstrap sampling
boot_data = torch.randn(200)

# Bootstrap confidence interval for mean
ci_lower, ci_upper, boot_means = bootstrap_confidence_interval(
    boot_data, lambda x: x.mean(), confidence=0.95, n_bootstrap=1000, seed=42
)

actual_mean = boot_data.mean()
print(f"Actual mean: {actual_mean:.6f}")
print(f"Bootstrap 95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
print(f"Bootstrap mean std: {boot_means.std():.6f}")

print("\n=== Importance Sampling ===")

def importance_sampling(data, proposal_weights, target_weights, n_samples, seed=None):
    """Importance sampling for reweighting data"""
    if seed is not None:
        torch.manual_seed(seed)
    
    # Normalize proposal weights
    proposal_weights = proposal_weights / proposal_weights.sum()
    
    # Sample according to proposal distribution
    indices = torch.multinomial(proposal_weights, n_samples, replacement=True)
    sampled_data = data[indices]
    
    # Compute importance weights
    sampled_proposal_weights = proposal_weights[indices]
    sampled_target_weights = target_weights[indices]
    importance_weights = sampled_target_weights / sampled_proposal_weights
    
    # Normalize importance weights
    importance_weights = importance_weights / importance_weights.sum()
    
    return sampled_data, importance_weights, indices

# Test importance sampling
proposal_dist = torch.softmax(torch.randn(1000), dim=0)
target_dist = torch.softmax(torch.randn(1000) * 0.5, dim=0)  # Different distribution

imp_data, imp_weights, imp_idx = importance_sampling(
    sample_data, proposal_dist, target_dist, n_samples=200, seed=42
)

print(f"Importance sampling result: {imp_data.shape}")
print(f"Importance weights range: [{imp_weights.min():.6f}, {imp_weights.max():.6f}]")
print(f"Effective sample size: {(imp_weights.sum() ** 2 / (imp_weights ** 2).sum()).item():.1f}")

print("\n=== Rejection Sampling ===")

def rejection_sampling(data, acceptance_fn, max_iterations=10000, seed=None):
    """Rejection sampling based on acceptance function"""
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    
    accepted_samples = []
    n_iterations = 0
    
    while len(accepted_samples) < len(data) and n_iterations < max_iterations:
        # Random sample from data
        idx = random.randint(0, len(data) - 1)
        sample = data[idx]
        
        # Accept or reject based on acceptance function
        if acceptance_fn(sample):
            accepted_samples.append(sample)
        
        n_iterations += 1
    
    if accepted_samples:
        return torch.stack(accepted_samples)
    else:
        return torch.empty(0, *data.shape[1:])

# Test rejection sampling
def acceptance_function(sample):
    """Accept samples with positive mean"""
    return sample.mean() > 0

rejected_data = torch.randn(500, 10)
accepted_samples = rejection_sampling(rejected_data, acceptance_function, seed=42)

print(f"Original data: {rejected_data.shape}")
print(f"Accepted samples: {accepted_samples.shape}")
print(f"Acceptance rate: {len(accepted_samples)/len(rejected_data)*100:.1f}%")

if len(accepted_samples) > 0:
    print(f"Mean of accepted samples: {accepted_samples.mean():.6f}")

print("\n=== Reservoir Sampling ===")

def reservoir_sampling(data_stream, k, seed=None):
    """Reservoir sampling for streaming data"""
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    
    reservoir = []
    
    for i, item in enumerate(data_stream):
        if i < k:
            # Fill reservoir
            reservoir.append(item)
        else:
            # Randomly replace items in reservoir
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    
    return torch.stack(reservoir) if reservoir else torch.empty(0)

# Test reservoir sampling
# Simulate streaming data
stream_size = 10000
k_reservoir = 100

def data_stream_generator(size):
    """Generate streaming data"""
    for i in range(size):
        yield torch.randn(5)

stream = data_stream_generator(stream_size)
reservoir_result = reservoir_sampling(stream, k_reservoir, seed=42)

print(f"Stream size: {stream_size}")
print(f"Reservoir size: {reservoir_result.shape}")
print(f"Reservoir mean: {reservoir_result.mean():.6f}")

print("\n=== Advanced Sampling Techniques ===")

def balanced_batch_sampling(data, labels, batch_size, n_classes_per_batch=None, seed=None):
    """Sample balanced batches with specified number of classes"""
    if seed is not None:
        torch.manual_seed(seed)
    
    unique_classes = torch.unique(labels)
    
    if n_classes_per_batch is None:
        n_classes_per_batch = len(unique_classes)
    
    # Group data by class
    class_data = {}
    for cls in unique_classes:
        class_indices = torch.where(labels == cls)[0]
        class_data[cls.item()] = class_indices
    
    batches = []
    samples_per_class = batch_size // n_classes_per_batch
    
    # Generate batches
    max_batches = min(len(class_data[cls.item()]) // samples_per_class 
                     for cls in unique_classes[:n_classes_per_batch])
    
    for batch_idx in range(max_batches):
        batch_indices = []
        
        for cls in unique_classes[:n_classes_per_batch]:
            cls_indices = class_data[cls.item()]
            start_idx = batch_idx * samples_per_class
            end_idx = start_idx + samples_per_class
            
            selected_indices = cls_indices[start_idx:end_idx]
            batch_indices.extend(selected_indices.tolist())
        
        # Shuffle batch
        batch_indices = torch.tensor(batch_indices)
        batch_indices = batch_indices[torch.randperm(len(batch_indices))]
        batches.append(batch_indices)
    
    return batches

def negative_sampling(positive_pairs, all_items, n_negatives_per_positive=1, seed=None):
    """Generate negative samples for contrastive learning"""
    if seed is not None:
        torch.manual_seed(seed)
    
    negative_pairs = []
    
    for pos_pair in positive_pairs:
        anchor, positive = pos_pair
        
        for _ in range(n_negatives_per_positive):
            # Sample random negative (avoid positive item)
            candidates = all_items[all_items != positive]
            if len(candidates) > 0:
                negative_idx = torch.randint(0, len(candidates), (1,))
                negative = candidates[negative_idx]
                negative_pairs.append((anchor, negative.item()))
    
    return negative_pairs

# Test balanced batch sampling
balanced_batches = balanced_batch_sampling(
    imbalanced_data, imbalanced_labels, batch_size=60, n_classes_per_batch=3, seed=42
)

print(f"Generated {len(balanced_batches)} balanced batches")
if balanced_batches:
    sample_batch = balanced_batches[0]
    sample_batch_labels = imbalanced_labels[sample_batch]
    print(f"Sample batch size: {len(sample_batch)}")
    print("Sample batch class distribution:")
    for cls in torch.unique(sample_batch_labels):
        count = (sample_batch_labels == cls).sum()
        print(f"  Class {cls.int()}: {count}")

# Test negative sampling
positive_pairs = [(0, 5), (1, 3), (2, 7), (4, 6)]
all_items = torch.arange(10)
negative_pairs = negative_sampling(positive_pairs, all_items, n_negatives_per_positive=2, seed=42)

print(f"\nPositive pairs: {positive_pairs}")
print(f"Negative pairs: {negative_pairs[:10]}...")  # Show first 10

print("\n=== Sampling Best Practices ===")

print("Sampling Guidelines:")
print("1. Choose sampling method appropriate for your data and task")
print("2. Always set random seeds for reproducibility")
print("3. Validate that samples are representative of population")
print("4. Consider computational efficiency for large datasets")
print("5. Use stratified sampling for imbalanced datasets")
print("6. Monitor sampling bias and variance")
print("7. Consider online/streaming sampling for large-scale data")

print("\nSampling Method Selection:")
print("- Random: When data is i.i.d. and representative")
print("- Stratified: For imbalanced classes or groups")
print("- Systematic: For ordered data with periodic patterns")
print("- Bootstrap: For confidence intervals and uncertainty estimation")
print("- Importance: For distribution shift or rare event sampling")
print("- Rejection: When you have complex acceptance criteria")
print("- Reservoir: For streaming data with unknown size")

print("\nPerformance Considerations:")
print("- Batch sampling for efficient GPU utilization")
print("- Memory-efficient sampling for large datasets")
print("- Parallel sampling for multi-threaded data loading")
print("- Cache-friendly sampling patterns")

print("\n=== Data Sampling Complete ===") 