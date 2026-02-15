#!/usr/bin/env python3
"""PyTorch Custom Datasets - Creating custom datasets"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import json
import pickle
import h5py
from PIL import Image
import cv2
from pathlib import Path
import urllib.request
import zipfile
from typing import Any, Callable, Optional, Tuple, List, Dict
import warnings

print("=== Custom Datasets Overview ===")

print("Custom dataset types covered:")
print("1. File-based datasets (images, text, CSV)")
print("2. In-memory datasets")
print("3. Database-backed datasets")
print("4. Streaming datasets")
print("5. Multi-modal datasets")
print("6. Hierarchical datasets")
print("7. Synthetic data generators")
print("8. Dataset versioning and metadata")

print("\n=== File-based Image Dataset ===")

class CustomImageDataset(Dataset):
    """Custom dataset for loading images from files"""
    
    def __init__(self, root_dir: str, annotations_file: str = None, 
                 transform: Optional[Callable] = None, 
                 target_transform: Optional[Callable] = None,
                 extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp'],
                 create_sample_data: bool = True):
        """
        Initialize image dataset
        
        Args:
            root_dir: Directory with all the images
            annotations_file: CSV file with annotations
            transform: Optional transform to be applied on images
            target_transform: Optional transform to be applied on targets
            extensions: Valid image extensions
            create_sample_data: Create sample data if directory doesn't exist
        """
        self.root_dir = Path(root_dir)
        self.annotations_file = annotations_file
        self.transform = transform
        self.target_transform = target_transform
        self.extensions = extensions
        
        # Create sample data if needed
        if create_sample_data and not self.root_dir.exists():
            self._create_sample_data()
        
        # Load image paths and labels
        self.samples = self._load_samples()
        
        # Create class mapping
        self.classes = sorted(list(set([sample[1] for sample in self.samples])))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
    
    def _create_sample_data(self):
        """Create sample image data for demonstration"""
        print(f"  Creating sample image data in {self.root_dir}")
        
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        # Create class subdirectories
        classes = ['cat', 'dog', 'bird']
        for class_name in classes:
            class_dir = self.root_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            # Create sample images (colored noise)
            for i in range(5):
                # Generate random colored image
                if class_name == 'cat':
                    color = [255, 100, 100]  # Reddish
                elif class_name == 'dog':
                    color = [100, 255, 100]  # Greenish
                else:  # bird
                    color = [100, 100, 255]  # Blueish
                
                # Create 64x64 image with base color + noise
                img_array = np.random.randint(0, 50, (64, 64, 3), dtype=np.uint8)
                for c in range(3):
                    img_array[:, :, c] = np.clip(img_array[:, :, c] + color[c], 0, 255)
                
                # Save as PNG
                img = Image.fromarray(img_array)
                img.save(class_dir / f"{class_name}_{i:03d}.png")
        
        # Create annotations CSV
        if self.annotations_file:
            annotations = []
            for class_name in classes:
                class_dir = self.root_dir / class_name
                for img_file in class_dir.glob("*.png"):
                    annotations.append({
                        'filename': str(img_file.relative_to(self.root_dir)),
                        'class': class_name,
                        'size': 'small' if 'bird' in class_name else 'medium'
                    })
            
            df = pd.DataFrame(annotations)
            df.to_csv(self.root_dir / self.annotations_file, index=False)
            print(f"  Created annotations file with {len(annotations)} entries")
    
    def _load_samples(self):
        """Load all image samples"""
        samples = []
        
        if self.annotations_file and (self.root_dir / self.annotations_file).exists():
            # Load from annotations file
            df = pd.read_csv(self.root_dir / self.annotations_file)
            for _, row in df.iterrows():
                img_path = self.root_dir / row['filename']
                if img_path.exists():
                    samples.append((str(img_path), row['class']))
        else:
            # Scan directory structure
            for ext in self.extensions:
                for img_path in self.root_dir.rglob(f"*{ext}"):
                    # Use parent directory as class name
                    class_name = img_path.parent.name
                    samples.append((str(img_path), class_name))
        
        print(f"  Found {len(samples)} images in {len(set([s[1] for s in samples]))} classes")
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get image and label by index"""
        img_path, class_name = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a default image
            image = Image.new('RGB', (64, 64), color='black')
        
        # Convert to tensor
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # HWC to CHW
        
        # Get class index
        label = self.class_to_idx[class_name]
        
        # Apply transforms
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image_tensor, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes"""
        class_counts = {}
        for _, class_name in self.samples:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return class_counts

# Test image dataset
print("Testing custom image dataset:")
image_dataset = CustomImageDataset("temp_images", annotations_file="annotations.csv")
print(f"  Classes: {image_dataset.classes}")
print(f"  Class distribution: {image_dataset.get_class_distribution()}")

# Test with DataLoader
image_dataloader = DataLoader(image_dataset, batch_size=4, shuffle=True)
batch_images, batch_labels = next(iter(image_dataloader))
print(f"  Batch shape: {batch_images.shape}")
print(f"  Batch labels: {batch_labels.tolist()}")

print("\n=== CSV/Tabular Dataset ===")

class TabularDataset(Dataset):
    """Custom dataset for tabular data from CSV"""
    
    def __init__(self, csv_file: str, target_column: str,
                 feature_columns: List[str] = None,
                 categorical_columns: List[str] = None,
                 normalize: bool = True,
                 create_sample_data: bool = True):
        """
        Initialize tabular dataset
        
        Args:
            csv_file: Path to CSV file
            target_column: Name of target column
            feature_columns: List of feature column names (None for all except target)
            categorical_columns: List of categorical column names
            normalize: Whether to normalize numerical features
            create_sample_data: Create sample data if file doesn't exist
        """
        self.csv_file = csv_file
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.categorical_columns = categorical_columns or []
        self.normalize = normalize
        
        # Create sample data if needed
        if create_sample_data and not os.path.exists(csv_file):
            self._create_sample_data()
        
        # Load and preprocess data
        self.data = self._load_and_preprocess()
        
    def _create_sample_data(self):
        """Create sample tabular data"""
        print(f"  Creating sample tabular data: {self.csv_file}")
        
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic data
        data = {
            'age': np.random.randint(18, 80, n_samples),
            'income': np.random.normal(50000, 20000, n_samples),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'experience': np.random.randint(0, 40, n_samples),
            'city': np.random.choice(['New York', 'San Francisco', 'Chicago', 'Austin'], n_samples),
            'score': np.random.normal(0.5, 0.2, n_samples)
        }
        
        # Make target correlated with features
        target = (
            (data['age'] > 40).astype(int) * 0.3 +
            (data['income'] > 60000).astype(int) * 0.4 +
            (data['experience'] > 10).astype(int) * 0.3 +
            np.random.normal(0, 0.1, n_samples)
        )
        data['target'] = (target > 0.5).astype(int)
        
        df = pd.DataFrame(data)
        df.to_csv(self.csv_file, index=False)
        print(f"  Created {len(df)} samples with {len(df.columns)} columns")
    
    def _load_and_preprocess(self):
        """Load and preprocess the data"""
        # Load CSV
        df = pd.read_csv(self.csv_file)
        
        # Select feature columns
        if self.feature_columns is None:
            self.feature_columns = [col for col in df.columns if col != self.target_column]
        
        # Separate features and targets
        X = df[self.feature_columns].copy()
        y = df[self.target_column].copy()
        
        # Process categorical columns
        self.categorical_mappings = {}
        for col in self.categorical_columns:
            if col in X.columns:
                unique_values = X[col].unique()
                mapping = {val: idx for idx, val in enumerate(unique_values)}
                self.categorical_mappings[col] = mapping
                X[col] = X[col].map(mapping)
        
        # Handle missing values
        X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
        
        # Normalize numerical features
        if self.normalize:
            numerical_columns = X.select_dtypes(include=[np.number]).columns
            self.feature_means = X[numerical_columns].mean()
            self.feature_stds = X[numerical_columns].std()
            X[numerical_columns] = (X[numerical_columns] - self.feature_means) / (self.feature_stds + 1e-8)
        
        # Convert to tensors
        self.features = torch.tensor(X.values, dtype=torch.float32)
        self.targets = torch.tensor(y.values, dtype=torch.long)
        
        print(f"  Processed data: {self.features.shape[0]} samples, {self.features.shape[1]} features")
        
        return X
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]
    
    def get_feature_names(self) -> List[str]:
        """Get feature column names"""
        return self.feature_columns
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        return {
            'num_samples': len(self),
            'num_features': len(self.feature_columns),
            'target_distribution': torch.bincount(self.targets).tolist(),
            'categorical_mappings': self.categorical_mappings
        }

# Test tabular dataset
print("Testing tabular dataset:")
tabular_dataset = TabularDataset(
    "temp_data.csv", 
    target_column="target",
    categorical_columns=['education', 'city']
)

print(f"  Feature names: {tabular_dataset.get_feature_names()}")
print(f"  Statistics: {tabular_dataset.get_statistics()}")

# Clean up sample files
for file in ["temp_data.csv"]:
    if os.path.exists(file):
        os.remove(file)

print("\n=== Text Dataset ===")

class TextDataset(Dataset):
    """Custom dataset for text data"""
    
    def __init__(self, text_files: List[str] = None, 
                 text_data: List[Tuple[str, str]] = None,
                 vocab_size: int = 10000,
                 max_length: int = 128,
                 create_sample_data: bool = True):
        """
        Initialize text dataset
        
        Args:
            text_files: List of text files to load
            text_data: Direct text data as (text, label) pairs
            vocab_size: Maximum vocabulary size
            max_length: Maximum sequence length
            create_sample_data: Create sample data if no data provided
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Load text data
        if text_data:
            self.raw_data = text_data
        elif text_files:
            self.raw_data = self._load_from_files(text_files)
        elif create_sample_data:
            self.raw_data = self._create_sample_text_data()
        else:
            raise ValueError("No text data provided")
        
        # Build vocabulary
        self.vocab = self._build_vocabulary()
        
        # Tokenize and encode data
        self.encoded_data = self._encode_data()
        
    def _create_sample_text_data(self):
        """Create sample text data"""
        print("  Creating sample text data")
        
        # Sample positive and negative movie reviews
        positive_reviews = [
            "This movie is absolutely fantastic and wonderful",
            "Great acting and amazing storyline throughout",
            "Excellent cinematography and brilliant direction",
            "Outstanding performance by all actors involved",
            "Incredible plot with perfect execution overall"
        ]
        
        negative_reviews = [
            "This movie is terrible and completely boring",
            "Poor acting and awful storyline everywhere",
            "Bad cinematography and horrible direction shown",
            "Disappointing performance by most actors here",
            "Weak plot with terrible execution throughout"
        ]
        
        data = []
        for review in positive_reviews:
            data.append((review, "positive"))
        for review in negative_reviews:
            data.append((review, "negative"))
        
        return data
    
    def _load_from_files(self, text_files):
        """Load text data from files"""
        data = []
        for file_path in text_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                # Use filename as label
                label = Path(file_path).stem
                data.append((text, label))
        return data
    
    def _build_vocabulary(self):
        """Build vocabulary from text data"""
        word_counts = {}
        
        for text, _ in self.raw_data:
            words = text.lower().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency and take top vocab_size
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        vocab_words = [word for word, count in sorted_words[:self.vocab_size-4]]
        
        # Create vocabulary mapping
        vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }
        
        for word in vocab_words:
            vocab[word] = len(vocab)
        
        print(f"  Built vocabulary with {len(vocab)} words")
        return vocab
    
    def _encode_data(self):
        """Encode text data using vocabulary"""
        encoded_data = []
        
        # Create label mapping
        unique_labels = list(set([label for _, label in self.raw_data]))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        for text, label in self.raw_data:
            # Tokenize and encode text
            words = text.lower().split()
            token_ids = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
            
            # Truncate or pad to max_length
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
            else:
                token_ids.extend([self.vocab['<PAD>']] * (self.max_length - len(token_ids)))
            
            # Encode label
            label_idx = self.label_to_idx[label]
            
            encoded_data.append((torch.tensor(token_ids, dtype=torch.long), 
                               torch.tensor(label_idx, dtype=torch.long)))
        
        return encoded_data
    
    def __len__(self) -> int:
        return len(self.encoded_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoded_data[idx]
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.vocab)
    
    def get_num_classes(self) -> int:
        """Get number of classes"""
        return len(self.label_to_idx)

# Test text dataset
print("Testing text dataset:")
text_dataset = TextDataset()
print(f"  Vocabulary size: {text_dataset.get_vocab_size()}")
print(f"  Number of classes: {text_dataset.get_num_classes()}")
print(f"  Dataset size: {len(text_dataset)}")

# Test with DataLoader
text_dataloader = DataLoader(text_dataset, batch_size=3, shuffle=True)
batch_texts, batch_labels = next(iter(text_dataloader))
print(f"  Batch text shape: {batch_texts.shape}")
print(f"  Batch labels: {batch_labels.tolist()}")

print("\n=== Custom Dataset Best Practices ===")

print("Design Principles:")
print("1. Keep __getitem__ fast and lightweight")
print("2. Handle data loading errors gracefully")
print("3. Implement proper data validation")
print("4. Use appropriate data types and formats")
print("5. Consider memory usage vs performance trade-offs")

print("\nFile Organization:")
print("1. Use consistent directory structures")
print("2. Separate data files from metadata")
print("3. Implement proper file path handling")
print("4. Consider data compression for large files")
print("5. Use appropriate file formats (HDF5, NPZ, etc.)")

print("\nError Handling:")
print("1. Validate file existence and accessibility")
print("2. Handle corrupted or missing data files")
print("3. Implement fallback strategies")
print("4. Provide meaningful error messages")
print("5. Log errors for debugging")

print("\nPerformance Optimization:")
print("1. Use lazy loading for large datasets")
print("2. Implement intelligent caching strategies")
print("3. Consider data preprocessing pipelines")
print("4. Optimize for common access patterns")
print("5. Profile dataset access times")

print("\nTesting and Validation:")
print("1. Test with different data sizes")
print("2. Validate data integrity and consistency")
print("3. Test error handling scenarios")
print("4. Verify transforms and preprocessing")
print("5. Test with DataLoader integration")

print("\n=== Custom Datasets Complete ===")

# Clean up created sample data
import shutil
for path in ["temp_images", "temp_data.csv"]:
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

# Memory cleanup
torch.cuda.empty_cache() if torch.cuda.is_available() else None