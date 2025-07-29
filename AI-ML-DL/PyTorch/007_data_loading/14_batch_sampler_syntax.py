"""
PyTorch Batch Sampler Syntax - Custom Batch Sampling Strategies
Comprehensive guide to batch samplers and custom sampling in PyTorch
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import numpy as np
import random
from typing import Iterator, List, Optional, Union, Dict, Any
from collections import defaultdict, Counter
import math

print("=== BATCH SAMPLER SYNTAX ===")

# 1. BASIC SAMPLER UNDERSTANDING
print("\n1. BASIC SAMPLER UNDERSTANDING")

# Create dummy dataset for demonstrations
class DummyDataset(Dataset):
    def __init__(self, size=1000, num_classes=10):
        self.size = size
        self.data = torch.randn(size, 10)
        # Create imbalanced labels for testing
        self.labels = []
        for i in range(num_classes):
            # Different class frequencies
            freq = max(1, int(size * (0.5 ** i) / sum(0.5 ** j for j in range(num_classes))))
            self.labels.extend([i] * freq)
        
        # Pad or trim to exact size
        if len(self.labels) < size:
            self.labels.extend([0] * (size - len(self.labels)))
        self.labels = self.labels[:size]
        random.shuffle(self.labels)
        self.labels = torch.tensor(self.labels)
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dummy_dataset = DummyDataset(1000, 5)
print(f"Dataset size: {len(dummy_dataset)}")
print(f"Label distribution: {torch.bincount(dummy_dataset.labels)}")

# Basic DataLoader sampling
basic_loader = DataLoader(dummy_dataset, batch_size=32, shuffle=True)
print("Basic DataLoader created with shuffle=True")

# Sequential sampling
sequential_loader = DataLoader(dummy_dataset, batch_size=32, shuffle=False)
print("Sequential DataLoader created")

# Subset sampling
subset_indices = list(range(100, 200))
subset_sampler = SubsetRandomSampler(subset_indices)
subset_loader = DataLoader(dummy_dataset, batch_size=16, sampler=subset_sampler)
print(f"Subset sampler for indices {len(subset_indices)}")

# 2. CUSTOM SAMPLER IMPLEMENTATION
print("\n2. CUSTOM SAMPLER IMPLEMENTATION")

class CustomRandomSampler(Sampler):
    """Custom random sampler with seed control"""
    
    def __init__(self, data_source, seed=None):
        self.data_source = data_source
        self.seed = seed
        
    def __iter__(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        indices = torch.randperm(len(self.data_source)).tolist()
        return iter(indices)
        
    def __len__(self):
        return len(self.data_source)

class SequentialSampler(Sampler):
    """Custom sequential sampler"""
    
    def __init__(self, data_source):
        self.data_source = data_source
        
    def __iter__(self):
        return iter(range(len(self.data_source)))
        
    def __len__(self):
        return len(self.data_source)

class ReverseSampler(Sampler):
    """Sample in reverse order"""
    
    def __init__(self, data_source):
        self.data_source = data_source
        
    def __iter__(self):
        return iter(range(len(self.data_source) - 1, -1, -1))
        
    def __len__(self):
        return len(self.data_source)

# Test custom samplers
custom_random_sampler = CustomRandomSampler(dummy_dataset, seed=42)
reverse_sampler = ReverseSampler(dummy_dataset)

custom_loader = DataLoader(dummy_dataset, batch_size=16, sampler=custom_random_sampler)
reverse_loader = DataLoader(dummy_dataset, batch_size=16, sampler=reverse_sampler)

print("Custom samplers created and tested")

# 3. BALANCED SAMPLING
print("\n3. BALANCED SAMPLING")

class BalancedSampler(Sampler):
    """Balanced sampler for imbalanced datasets"""
    
    def __init__(self, dataset, num_samples=None):
        self.dataset = dataset
        self.num_samples = num_samples or len(dataset)
        
        # Get labels and compute class weights
        if hasattr(dataset, 'labels'):
            self.labels = dataset.labels
        else:
            # Extract labels if not directly available
            self.labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])
            
        self.class_counts = torch.bincount(self.labels)
        self.num_classes = len(self.class_counts)
        
        # Compute sampling weights (inverse frequency)
        self.weights = torch.zeros(len(dataset))
        for i, label in enumerate(self.labels):
            self.weights[i] = 1.0 / self.class_counts[label].float()
            
    def __iter__(self):
        # Sample with replacement using weights
        indices = torch.multinomial(self.weights, self.num_samples, replacement=True)
        return iter(indices.tolist())
        
    def __len__(self):
        return self.num_samples

class StratifiedSampler(Sampler):
    """Stratified sampler maintaining class distribution"""
    
    def __init__(self, dataset, samples_per_class=None):
        if hasattr(dataset, 'labels'):
            self.labels = dataset.labels
        else:
            self.labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])
            
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_indices[label.item()].append(idx)
            
        self.num_classes = len(self.class_indices)
        self.samples_per_class = samples_per_class or min(len(indices) for indices in self.class_indices.values())
        
    def __iter__(self):
        indices = []
        for class_idx, class_indices in self.class_indices.items():
            # Sample from each class
            sampled = np.random.choice(class_indices, 
                                     size=min(self.samples_per_class, len(class_indices)), 
                                     replace=False)
            indices.extend(sampled.tolist())
            
        # Shuffle final indices
        np.random.shuffle(indices)
        return iter(indices)
        
    def __len__(self):
        return self.samples_per_class * self.num_classes

# Test balanced and stratified sampling
balanced_sampler = BalancedSampler(dummy_dataset, num_samples=500)
stratified_sampler = StratifiedSampler(dummy_dataset, samples_per_class=50)

balanced_loader = DataLoader(dummy_dataset, batch_size=32, sampler=balanced_sampler)
stratified_loader = DataLoader(dummy_dataset, batch_size=32, sampler=stratified_sampler)

print("Balanced and stratified samplers created")

# Test distribution
def check_distribution(loader, name, max_batches=5):
    """Check label distribution in loader"""
    all_labels = []
    for i, (_, labels) in enumerate(loader):
        if i >= max_batches:
            break
        all_labels.extend(labels.tolist())
    
    if all_labels:
        label_counts = Counter(all_labels)
        print(f"{name} distribution: {dict(label_counts)}")

check_distribution(balanced_loader, "Balanced")
check_distribution(stratified_loader, "Stratified")

# 4. BATCH SAMPLER IMPLEMENTATION
print("\n4. BATCH SAMPLER IMPLEMENTATION")

class CustomBatchSampler(BatchSampler):
    """Custom batch sampler with specific batching logic"""
    
    def __init__(self, sampler, batch_size, drop_last=False):
        super().__init__(sampler, batch_size, drop_last)
        
    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        if len(batch) > 0 and not self.drop_last:
            yield batch

class DynamicBatchSampler(BatchSampler):
    """Batch sampler with dynamic batch sizes"""
    
    def __init__(self, sampler, min_batch_size=16, max_batch_size=64):
        self.sampler = sampler
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        
    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            
            # Dynamic batch size based on some criteria
            current_batch_size = random.randint(self.min_batch_size, self.max_batch_size)
            
            if len(batch) >= current_batch_size:
                yield batch
                batch = []
                
        if batch:
            yield batch
            
    def __len__(self):
        # Approximate length
        return math.ceil(len(self.sampler) / ((self.min_batch_size + self.max_batch_size) / 2))

class BalancedBatchSampler(BatchSampler):
    """Batch sampler ensuring balanced classes in each batch"""
    
    def __init__(self, dataset, batch_size, samples_per_class=None):
        self.dataset = dataset
        self.batch_size = batch_size
        
        if hasattr(dataset, 'labels'):
            self.labels = dataset.labels
        else:
            self.labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])
            
        # Group indices by class
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_indices[label.item()].append(idx)
            
        self.num_classes = len(self.class_indices)
        self.samples_per_class = samples_per_class or (batch_size // self.num_classes)
        
        # Shuffle class indices
        for class_indices in self.class_indices.values():
            random.shuffle(class_indices)
            
    def __iter__(self):
        # Create iterators for each class
        class_iterators = {}
        for class_idx, indices in self.class_indices.items():
            class_iterators[class_idx] = iter(indices * 100)  # Cycle through multiple times
            
        while True:
            batch = []
            
            # Sample from each class
            for class_idx in self.class_indices.keys():
                for _ in range(self.samples_per_class):
                    try:
                        idx = next(class_iterators[class_idx])
                        batch.append(idx)
                    except StopIteration:
                        return
                        
            if len(batch) >= self.batch_size:
                random.shuffle(batch)
                yield batch[:self.batch_size]
            else:
                if batch:
                    yield batch
                return
                
    def __len__(self):
        min_class_size = min(len(indices) for indices in self.class_indices.values())
        return min_class_size // self.samples_per_class

# Test batch samplers
custom_batch_sampler = CustomBatchSampler(
    SequentialSampler(dummy_dataset), batch_size=32, drop_last=True
)

dynamic_batch_sampler = DynamicBatchSampler(
    CustomRandomSampler(dummy_dataset), min_batch_size=16, max_batch_size=48
)

balanced_batch_sampler = BalancedBatchSampler(
    dummy_dataset, batch_size=25, samples_per_class=5
)

# Create DataLoaders with batch samplers
custom_batch_loader = DataLoader(dummy_dataset, batch_sampler=custom_batch_sampler)
dynamic_batch_loader = DataLoader(dummy_dataset, batch_sampler=dynamic_batch_sampler)
balanced_batch_loader = DataLoader(dummy_dataset, batch_sampler=balanced_batch_sampler)

print("Custom batch samplers created")

# Test batch sizes and distributions
def check_batch_properties(loader, name, max_batches=3):
    """Check batch properties"""
    batch_sizes = []
    batch_distributions = []
    
    for i, (_, labels) in enumerate(loader):
        if i >= max_batches:
            break
        batch_sizes.append(len(labels))
        batch_distributions.append(torch.bincount(labels, minlength=5).tolist())
    
    print(f"{name}:")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Distributions: {batch_distributions}")

check_batch_properties(dynamic_batch_loader, "Dynamic Batch")
check_batch_properties(balanced_batch_loader, "Balanced Batch")

# 5. ADVANCED SAMPLING STRATEGIES
print("\n5. ADVANCED SAMPLING STRATEGIES")

class HardNegativeSampler(Sampler):
    """Sample hard negatives based on model predictions"""
    
    def __init__(self, dataset, model=None, difficulty_scores=None, hard_ratio=0.7):
        self.dataset = dataset
        self.model = model
        self.hard_ratio = hard_ratio
        
        if difficulty_scores is not None:
            self.difficulty_scores = difficulty_scores
        else:
            # Generate dummy difficulty scores
            self.difficulty_scores = torch.rand(len(dataset))
            
    def __iter__(self):
        num_hard = int(len(self.dataset) * self.hard_ratio)
        num_easy = len(self.dataset) - num_hard
        
        # Sort by difficulty (higher = harder)
        sorted_indices = torch.argsort(self.difficulty_scores, descending=True)
        
        # Sample more from hard examples
        hard_indices = sorted_indices[:len(sorted_indices)//2]
        easy_indices = sorted_indices[len(sorted_indices)//2:]
        
        # Sample with replacement
        sampled_hard = torch.multinomial(
            torch.ones(len(hard_indices)), num_hard, replacement=True
        )
        sampled_easy = torch.multinomial(
            torch.ones(len(easy_indices)), num_easy, replacement=True
        )
        
        # Combine and shuffle
        all_indices = torch.cat([hard_indices[sampled_hard], easy_indices[sampled_easy]])
        shuffled = all_indices[torch.randperm(len(all_indices))]
        
        return iter(shuffled.tolist())
        
    def __len__(self):
        return len(self.dataset)

class CurriculumSampler(Sampler):
    """Curriculum learning sampler - start with easy examples"""
    
    def __init__(self, dataset, difficulty_scores, curriculum_rate=0.1):
        self.dataset = dataset
        self.difficulty_scores = difficulty_scores
        self.curriculum_rate = curriculum_rate
        self.epoch = 0
        
    def set_epoch(self, epoch):
        """Set current epoch for curriculum progression"""
        self.epoch = epoch
        
    def __iter__(self):
        # Determine difficulty threshold based on epoch
        difficulty_threshold = min(1.0, self.curriculum_rate * (self.epoch + 1))
        max_difficulty = self.difficulty_scores.quantile(difficulty_threshold)
        
        # Filter examples below threshold
        valid_mask = self.difficulty_scores <= max_difficulty
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            valid_indices = torch.arange(len(self.dataset))
            
        # Randomly sample from valid examples
        shuffled = valid_indices[torch.randperm(len(valid_indices))]
        return iter(shuffled.tolist())
        
    def __len__(self):
        return len(self.dataset)

class MemoryBankSampler(Sampler):
    """Sampler with memory bank for previously seen hard examples"""
    
    def __init__(self, dataset, memory_size=1000, update_frequency=100):
        self.dataset = dataset
        self.memory_size = memory_size
        self.update_frequency = update_frequency
        self.memory_bank = []
        self.sample_count = 0
        
    def update_memory(self, hard_indices):
        """Update memory bank with hard examples"""
        self.memory_bank.extend(hard_indices)
        if len(self.memory_bank) > self.memory_size:
            # Keep most recent hard examples
            self.memory_bank = self.memory_bank[-self.memory_size:]
            
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        
        # Mix regular sampling with memory bank
        if self.memory_bank and len(self.memory_bank) > 0:
            memory_ratio = 0.3  # 30% from memory bank
            num_memory = int(len(indices) * memory_ratio)
            
            # Sample from memory bank
            memory_samples = random.choices(self.memory_bank, k=num_memory)
            
            # Sample remaining from regular dataset
            regular_samples = random.choices(indices, k=len(indices) - num_memory)
            
            all_samples = memory_samples + regular_samples
            random.shuffle(all_samples)
            return iter(all_samples)
        else:
            random.shuffle(indices)
            return iter(indices)
            
    def __len__(self):
        return len(self.dataset)

# Test advanced sampling strategies
difficulty_scores = torch.rand(len(dummy_dataset))  # Random difficulty scores

hard_negative_sampler = HardNegativeSampler(dummy_dataset, difficulty_scores=difficulty_scores)
curriculum_sampler = CurriculumSampler(dummy_dataset, difficulty_scores, curriculum_rate=0.2)
memory_bank_sampler = MemoryBankSampler(dummy_dataset, memory_size=200)

print("Advanced samplers created")

# Test curriculum progression
for epoch in range(3):
    curriculum_sampler.set_epoch(epoch)
    curriculum_loader = DataLoader(dummy_dataset, batch_size=32, sampler=curriculum_sampler)
    
    # Count samples (just first batch)
    first_batch = next(iter(curriculum_loader))
    print(f"Epoch {epoch}: Curriculum sampler batch size = {len(first_batch[0])}")

# 6. MULTI-TASK SAMPLING
print("\n6. MULTI-TASK SAMPLING")

class MultiTaskBatchSampler(BatchSampler):
    """Batch sampler for multi-task learning"""
    
    def __init__(self, datasets, batch_size, task_weights=None):
        self.datasets = datasets
        self.batch_size = batch_size
        self.num_tasks = len(datasets)
        
        # Task weights for sampling probability
        if task_weights is None:
            self.task_weights = [1.0] * self.num_tasks
        else:
            self.task_weights = task_weights
            
        # Normalize weights
        total_weight = sum(self.task_weights)
        self.task_weights = [w / total_weight for w in self.task_weights]
        
        # Create samplers for each task
        self.task_samplers = [iter(range(len(dataset))) for dataset in datasets]
        
    def __iter__(self):
        while True:
            batch = []
            task_samples = {i: [] for i in range(self.num_tasks)}
            
            # Determine samples per task for this batch
            for _ in range(self.batch_size):
                # Choose task based on weights
                task_id = np.random.choice(self.num_tasks, p=self.task_weights)
                
                try:
                    # Get next sample from chosen task
                    sample_idx = next(self.task_samplers[task_id])
                    task_samples[task_id].append((task_id, sample_idx))
                except StopIteration:
                    # Reset sampler for this task
                    self.task_samplers[task_id] = iter(range(len(self.datasets[task_id])))
                    sample_idx = next(self.task_samplers[task_id])
                    task_samples[task_id].append((task_id, sample_idx))
                    
            # Flatten and yield
            batch = []
            for task_id, samples in task_samples.items():
                batch.extend(samples)
                
            if batch:
                yield batch
            else:
                break
                
    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets) // self.batch_size

# Create multiple dummy datasets for multi-task
task_datasets = [DummyDataset(200, 3), DummyDataset(300, 5), DummyDataset(150, 2)]
multi_task_sampler = MultiTaskBatchSampler(task_datasets, batch_size=32, task_weights=[0.5, 0.3, 0.2])

print("Multi-task batch sampler created")

# 7. PERFORMANCE OPTIMIZATION
print("\n7. PERFORMANCE OPTIMIZATION")

class CachedSampler(Sampler):
    """Sampler with caching for expensive operations"""
    
    def __init__(self, dataset, cache_size=10000):
        self.dataset = dataset
        self.cache_size = cache_size
        self.cached_indices = None
        self.cache_epoch = -1
        
    def __iter__(self):
        # Cache indices for reuse within epoch
        if self.cached_indices is None:
            self.cached_indices = torch.randperm(len(self.dataset)).tolist()
            
        return iter(self.cached_indices)
        
    def __len__(self):
        return len(self.dataset)
        
    def refresh_cache(self):
        """Manually refresh cached indices"""
        self.cached_indices = None

class ParallelSampler(Sampler):
    """Sampler that prepares indices in parallel"""
    
    def __init__(self, dataset, num_workers=2):
        self.dataset = dataset
        self.num_workers = num_workers
        
    def __iter__(self):
        # In real implementation, this would use multiprocessing
        # For demo, just return shuffled indices
        indices = torch.randperm(len(self.dataset)).tolist()
        return iter(indices)
        
    def __len__(self):
        return len(self.dataset)

# Test performance optimizations
cached_sampler = CachedSampler(dummy_dataset)
parallel_sampler = ParallelSampler(dummy_dataset, num_workers=2)

cached_loader = DataLoader(dummy_dataset, batch_size=32, sampler=cached_sampler)
parallel_loader = DataLoader(dummy_dataset, batch_size=32, sampler=parallel_sampler)

print("Performance-optimized samplers created")

# 8. SAMPLER COMPOSITION
print("\n8. SAMPLER COMPOSITION")

class ComposedSampler(Sampler):
    """Compose multiple sampling strategies"""
    
    def __init__(self, samplers, weights=None):
        self.samplers = samplers
        self.weights = weights or [1.0] * len(samplers)
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
    def __iter__(self):
        # Choose which sampler to use
        sampler_idx = np.random.choice(len(self.samplers), p=self.weights)
        chosen_sampler = self.samplers[sampler_idx]
        
        return iter(chosen_sampler)
        
    def __len__(self):
        return len(self.samplers[0])  # Assume all samplers have same length

# Compose different samplers
composed_sampler = ComposedSampler([
    SequentialSampler(dummy_dataset),
    CustomRandomSampler(dummy_dataset, seed=42),
    BalancedSampler(dummy_dataset)
], weights=[0.2, 0.3, 0.5])

composed_loader = DataLoader(dummy_dataset, batch_size=32, sampler=composed_sampler)

print("Composed sampler created")

print("\n=== BATCH SAMPLER SYNTAX COMPLETE ===")
print("Key concepts covered:")
print("- Basic sampler types and usage")
print("- Custom sampler implementation")
print("- Balanced and stratified sampling")
print("- Custom batch samplers")
print("- Advanced sampling strategies")
print("- Multi-task sampling")
print("- Performance optimization")
print("- Sampler composition")