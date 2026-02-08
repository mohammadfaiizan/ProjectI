#!/usr/bin/env python3
"""PyTorch Data Splitting Utilities - Train/val/test splitting"""

import torch
import math
import random

print("=== Data Splitting Overview ===")

print("Splitting strategies:")
print("1. Random splits")
print("2. Stratified splits (preserving class distribution)")
print("3. Time-based splits (for temporal data)")
print("4. K-fold cross-validation")
print("5. Group-based splits")
print("6. Nested splits (train/val/test)")

print("\n=== Basic Random Splitting ===")

def random_split(dataset_size, ratios=(0.7, 0.2, 0.1), seed=None):
    """Random split indices into train/val/test"""
    if seed is not None:
        torch.manual_seed(seed)
    
    # Validate ratios
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")
    
    # Generate random permutation
    indices = torch.randperm(dataset_size)
    
    # Calculate split sizes
    train_size = int(dataset_size * ratios[0])
    val_size = int(dataset_size * ratios[1])
    test_size = dataset_size - train_size - val_size
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return train_indices, val_indices, test_indices

def simple_train_test_split(data, test_ratio=0.2, seed=None):
    """Simple train/test split for tensors"""
    if seed is not None:
        torch.manual_seed(seed)
    
    dataset_size = len(data)
    test_size = int(dataset_size * test_ratio)
    train_size = dataset_size - test_size
    
    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    return data[train_indices], data[test_indices], train_indices, test_indices

# Test basic splitting
dataset_size = 1000
train_idx, val_idx, test_idx = random_split(dataset_size, ratios=(0.6, 0.2, 0.2), seed=42)

print(f"Dataset size: {dataset_size}")
print(f"Train size: {len(train_idx)} ({len(train_idx)/dataset_size:.1%})")
print(f"Val size: {len(val_idx)} ({len(val_idx)/dataset_size:.1%})")
print(f"Test size: {len(test_idx)} ({len(test_idx)/dataset_size:.1%})")

# Test simple split
sample_data = torch.randn(100, 10)
train_data, test_data, train_indices, test_indices = simple_train_test_split(sample_data, test_ratio=0.3, seed=42)

print(f"\nOriginal data: {sample_data.shape}")
print(f"Train data: {train_data.shape}")
print(f"Test data: {test_data.shape}")

print("\n=== Stratified Splitting ===")

def stratified_split(labels, ratios=(0.7, 0.2, 0.1), seed=None):
    """Stratified split preserving class distribution"""
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    
    # Get unique classes and their counts
    unique_classes = torch.unique(labels)
    class_indices = {}
    
    # Group indices by class
    for cls in unique_classes:
        class_indices[cls.item()] = torch.where(labels == cls)[0]
    
    train_indices, val_indices, test_indices = [], [], []
    
    # Split each class proportionally
    for cls, indices in class_indices.items():
        class_size = len(indices)
        
        # Calculate split sizes for this class
        train_size = int(class_size * ratios[0])
        val_size = int(class_size * ratios[1])
        test_size = class_size - train_size - val_size
        
        # Random permutation within class
        perm_indices = indices[torch.randperm(len(indices))]
        
        # Split this class
        train_indices.extend(perm_indices[:train_size].tolist())
        val_indices.extend(perm_indices[train_size:train_size + val_size].tolist())
        test_indices.extend(perm_indices[train_size + val_size:].tolist())
    
    # Convert back to tensors and shuffle
    train_indices = torch.tensor(train_indices)[torch.randperm(len(train_indices))]
    val_indices = torch.tensor(val_indices)[torch.randperm(len(val_indices))]
    test_indices = torch.tensor(test_indices)[torch.randperm(len(test_indices))]
    
    return train_indices, val_indices, test_indices

# Test stratified splitting
labels = torch.cat([
    torch.zeros(300),  # Class 0: 300 samples
    torch.ones(500),   # Class 1: 500 samples
    torch.full((200,), 2)  # Class 2: 200 samples
])

strat_train, strat_val, strat_test = stratified_split(labels, ratios=(0.6, 0.2, 0.2), seed=42)

# Check class distributions
def check_class_distribution(indices, labels, split_name):
    split_labels = labels[indices]
    unique, counts = torch.unique(split_labels, return_counts=True)
    print(f"{split_name} distribution:")
    for cls, count in zip(unique, counts):
        percentage = count.item() / len(split_labels) * 100
        print(f"  Class {cls.item()}: {count.item()} ({percentage:.1f}%)")

print("Original distribution:")
unique_orig, counts_orig = torch.unique(labels, return_counts=True)
for cls, count in zip(unique_orig, counts_orig):
    percentage = count.item() / len(labels) * 100
    print(f"  Class {cls.item()}: {count.item()} ({percentage:.1f}%)")

check_class_distribution(strat_train, labels, "Train")
check_class_distribution(strat_val, labels, "Val")
check_class_distribution(strat_test, labels, "Test")

print("\n=== Time-based Splitting ===")

def temporal_split(timestamps, ratios=(0.7, 0.2, 0.1)):
    """Split data based on temporal order"""
    # Sort by timestamp
    sorted_indices = torch.argsort(timestamps)
    dataset_size = len(timestamps)
    
    # Calculate split points
    train_size = int(dataset_size * ratios[0])
    val_size = int(dataset_size * ratios[1])
    
    # Split chronologically
    train_indices = sorted_indices[:train_size]
    val_indices = sorted_indices[train_size:train_size + val_size]
    test_indices = sorted_indices[train_size + val_size:]
    
    return train_indices, val_indices, test_indices

def sliding_window_split(data, window_size, step_size=1):
    """Create sliding windows for time series"""
    windows = []
    targets = []
    
    for i in range(0, len(data) - window_size, step_size):
        window = data[i:i + window_size]
        target = data[i + window_size] if i + window_size < len(data) else None
        
        windows.append(window)
        if target is not None:
            targets.append(target)
    
    return torch.stack(windows), torch.stack(targets) if targets else None

# Test temporal splitting
time_series_length = 1000
timestamps = torch.arange(time_series_length).float()
temp_train, temp_val, temp_test = temporal_split(timestamps, ratios=(0.6, 0.2, 0.2))

print(f"Temporal split sizes: {len(temp_train)}, {len(temp_val)}, {len(temp_test)}")
print(f"Train time range: {timestamps[temp_train].min():.0f} - {timestamps[temp_train].max():.0f}")
print(f"Val time range: {timestamps[temp_val].min():.0f} - {timestamps[temp_val].max():.0f}")
print(f"Test time range: {timestamps[temp_test].min():.0f} - {timestamps[temp_test].max():.0f}")

# Test sliding windows
time_series = torch.sin(torch.linspace(0, 4*math.pi, 100))
windows, targets = sliding_window_split(time_series, window_size=10, step_size=1)
print(f"Time series length: {len(time_series)}")
print(f"Number of windows: {len(windows)}")
print(f"Window shape: {windows.shape}")
print(f"Targets shape: {targets.shape}")

print("\n=== K-Fold Cross Validation ===")

def kfold_split(dataset_size, k=5, seed=None):
    """K-fold cross-validation splits"""
    if seed is not None:
        torch.manual_seed(seed)
    
    # Create shuffled indices
    indices = torch.randperm(dataset_size)
    fold_size = dataset_size // k
    
    folds = []
    for i in range(k):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < k - 1 else dataset_size
        
        test_indices = indices[start_idx:end_idx]
        train_indices = torch.cat([
            indices[:start_idx],
            indices[end_idx:]
        ])
        
        folds.append((train_indices, test_indices))
    
    return folds

def stratified_kfold_split(labels, k=5, seed=None):
    """Stratified K-fold cross-validation"""
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    
    unique_classes = torch.unique(labels)
    class_indices = {}
    
    # Group indices by class
    for cls in unique_classes:
        class_indices[cls.item()] = torch.where(labels == cls)[0]
        # Shuffle within each class
        class_indices[cls.item()] = class_indices[cls.item()][torch.randperm(len(class_indices[cls.item()]))]
    
    folds = [[] for _ in range(k)]
    
    # Distribute each class across folds
    for cls, indices in class_indices.items():
        class_size = len(indices)
        fold_size = class_size // k
        
        for i in range(k):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < k - 1 else class_size
            fold_indices = indices[start_idx:end_idx]
            folds[i].extend(fold_indices.tolist())
    
    # Convert to train/test splits
    cv_splits = []
    for i in range(k):
        test_indices = torch.tensor(folds[i])
        train_indices = torch.cat([torch.tensor(folds[j]) for j in range(k) if j != i])
        cv_splits.append((train_indices, test_indices))
    
    return cv_splits

# Test K-fold
kfold_splits = kfold_split(1000, k=5, seed=42)
print(f"K-fold splits created: {len(kfold_splits)}")
for i, (train, test) in enumerate(kfold_splits):
    print(f"Fold {i+1}: Train={len(train)}, Test={len(test)}")

# Test stratified K-fold
strat_kfold_splits = stratified_kfold_split(labels, k=5, seed=42)
print(f"\nStratified K-fold splits: {len(strat_kfold_splits)}")
for i, (train, test) in enumerate(strat_kfold_splits):
    test_labels = labels[test]
    unique, counts = torch.unique(test_labels, return_counts=True)
    distribution = [f"Class {cls.item()}:{count.item()}" for cls, count in zip(unique, counts)]
    print(f"Fold {i+1}: Train={len(train)}, Test={len(test)} ({', '.join(distribution)})")

print("\n=== Group-based Splitting ===")

def group_split(groups, ratios=(0.7, 0.2, 0.1), seed=None):
    """Split data by groups (ensures samples from same group stay together)"""
    if seed is not None:
        torch.manual_seed(seed)
    
    unique_groups = torch.unique(groups)
    num_groups = len(unique_groups)
    
    # Calculate split sizes for groups
    train_groups = int(num_groups * ratios[0])
    val_groups = int(num_groups * ratios[1])
    
    # Shuffle groups
    shuffled_groups = unique_groups[torch.randperm(num_groups)]
    
    # Split groups
    train_group_set = set(shuffled_groups[:train_groups].tolist())
    val_group_set = set(shuffled_groups[train_groups:train_groups + val_groups].tolist())
    test_group_set = set(shuffled_groups[train_groups + val_groups:].tolist())
    
    # Get indices for each split
    train_indices = torch.where(torch.isin(groups, torch.tensor(list(train_group_set))))[0]
    val_indices = torch.where(torch.isin(groups, torch.tensor(list(val_group_set))))[0]
    test_indices = torch.where(torch.isin(groups, torch.tensor(list(test_group_set))))[0]
    
    return train_indices, val_indices, test_indices

# Test group splitting
# Simulate user IDs (groups) - some users have more samples than others
user_ids = torch.cat([
    torch.full((50,), 0),  # User 0: 50 samples
    torch.full((30,), 1),  # User 1: 30 samples
    torch.full((40,), 2),  # User 2: 40 samples
    torch.full((25,), 3),  # User 3: 25 samples
    torch.full((35,), 4),  # User 4: 35 samples
])

group_train, group_val, group_test = group_split(user_ids, ratios=(0.6, 0.2, 0.2), seed=42)

print(f"Group splitting results:")
print(f"Train samples: {len(group_train)} from users: {torch.unique(user_ids[group_train]).tolist()}")
print(f"Val samples: {len(group_val)} from users: {torch.unique(user_ids[group_val]).tolist()}")
print(f"Test samples: {len(group_test)} from users: {torch.unique(user_ids[group_test]).tolist()}")

print("\n=== Advanced Splitting Utilities ===")

class DataSplitter:
    """Comprehensive data splitting utility"""
    
    def __init__(self, seed=None):
        self.seed = seed
        self.splits = {}
    
    def split_dataset(self, data, labels=None, groups=None, 
                     method='random', ratios=(0.7, 0.2, 0.1), **kwargs):
        """Split dataset using specified method"""
        if self.seed is not None:
            torch.manual_seed(self.seed)
        
        dataset_size = len(data)
        
        if method == 'random':
            indices = random_split(dataset_size, ratios, self.seed)
        elif method == 'stratified':
            if labels is None:
                raise ValueError("Labels required for stratified splitting")
            indices = stratified_split(labels, ratios, self.seed)
        elif method == 'temporal':
            timestamps = kwargs.get('timestamps')
            if timestamps is None:
                raise ValueError("Timestamps required for temporal splitting")
            indices = temporal_split(timestamps, ratios)
        elif method == 'group':
            if groups is None:
                raise ValueError("Groups required for group splitting")
            indices = group_split(groups, ratios, self.seed)
        else:
            raise ValueError(f"Unknown splitting method: {method}")
        
        # Store split information
        self.splits = {
            'method': method,
            'ratios': ratios,
            'train_indices': indices[0],
            'val_indices': indices[1],
            'test_indices': indices[2]
        }
        
        return indices
    
    def get_splits(self, data, labels=None):
        """Get actual data splits"""
        if not self.splits:
            raise ValueError("No splits created yet. Call split_dataset first.")
        
        train_data = data[self.splits['train_indices']]
        val_data = data[self.splits['val_indices']]
        test_data = data[self.splits['test_indices']]
        
        splits = {'train': train_data, 'val': val_data, 'test': test_data}
        
        if labels is not None:
            splits['train_labels'] = labels[self.splits['train_indices']]
            splits['val_labels'] = labels[self.splits['val_indices']]
            splits['test_labels'] = labels[self.splits['test_indices']]
        
        return splits
    
    def get_split_info(self):
        """Get information about the splits"""
        if not self.splits:
            return "No splits created yet"
        
        info = f"Splitting method: {self.splits['method']}\n"
        info += f"Ratios: {self.splits['ratios']}\n"
        info += f"Train size: {len(self.splits['train_indices'])}\n"
        info += f"Val size: {len(self.splits['val_indices'])}\n"
        info += f"Test size: {len(self.splits['test_indices'])}\n"
        
        return info

# Test advanced splitter
splitter = DataSplitter(seed=42)

# Sample dataset
dataset = torch.randn(1000, 20)
dataset_labels = torch.randint(0, 3, (1000,))

# Try different splitting methods
for method in ['random', 'stratified']:
    print(f"\n{method.upper()} SPLITTING:")
    
    if method == 'stratified':
        indices = splitter.split_dataset(dataset, labels=dataset_labels, method=method)
    else:
        indices = splitter.split_dataset(dataset, method=method)
    
    splits = splitter.get_splits(dataset, dataset_labels)
    print(splitter.get_split_info())

print("\n=== Data Splitting Best Practices ===")

print("Splitting Guidelines:")
print("1. Always use a fixed seed for reproducibility")
print("2. Choose splitting method appropriate for your data:")
print("   - Random: for i.i.d. data")
print("   - Stratified: for imbalanced classes")
print("   - Temporal: for time series data")
print("   - Group: when samples are grouped (users, subjects)")
print("3. Validate that splits maintain important data properties")
print("4. Consider data leakage when splitting (especially for time series)")
print("5. Use appropriate train/val/test ratios for your dataset size")
print("6. Keep test set untouched until final evaluation")

print("\nCommon Split Ratios:")
print("- Small datasets (< 1K samples): 60/20/20 or 70/15/15")
print("- Medium datasets (1K-10K): 70/15/15 or 80/10/10")
print("- Large datasets (> 10K): 80/10/10 or 90/5/5")
print("- Time series: Use temporal splits, often 70/15/15")

print("\nValidation Strategies:")
print("- Hold-out validation: Single train/val/test split")
print("- K-fold CV: Multiple train/val splits, no test set")
print("- Nested CV: K-fold with inner validation loop")
print("- Time series CV: Forward chaining validation")

print("\n=== Data Splitting Complete ===") 