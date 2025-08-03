"""
LPIPS Data Loading Utilities
============================

Data loading utilities for LPIPS training and evaluation using JND (Just Noticeable Difference) datasets.
Supports various dataset formats and provides efficient data loading for 2AFC training.

This module includes:
- JND dataset loader for LPIPS training
- Data preprocessing and augmentation
- 2AFC data formatting
- Evaluation dataset creation
- Traditional metrics baseline datasets

Author: [Your Name]
Date: [Current Date]
"""

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import json
import pandas as pd
import h5py
from typing import Dict, List, Tuple, Optional, Union, Any
import glob
import random
from pathlib import Path
import matplotlib.pyplot as plt


class JNDDataset(Dataset):
    """
    JND (Just Noticeable Difference) Dataset for LPIPS training
    
    Loads triplets of (reference, image1, image2, human_judgment) for 2AFC training.
    Supports multiple JND dataset formats.
    """
    
    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 transform: Optional[transforms.Compose] = None,
                 dataset_format: str = 'auto',
                 load_in_memory: bool = False):
        """
        Initialize JND Dataset
        
        Args:
            data_dir: Path to JND dataset directory
            split: Dataset split ('train', 'val', 'test')
            transform: Image transformations
            dataset_format: Dataset format ('bapps', '2afc', 'csv', 'auto')
            load_in_memory: Whether to load all images in memory
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform or self._get_default_transform()
        self.load_in_memory = load_in_memory
        
        # Detect dataset format if auto
        if dataset_format == 'auto':
            dataset_format = self._detect_dataset_format()
        
        self.dataset_format = dataset_format
        
        # Load dataset based on format
        self.data_triplets = self._load_dataset()
        
        # Load images in memory if requested
        if load_in_memory:
            self._preload_images()
        
        print(f"JND Dataset loaded:")
        print(f"  Format: {dataset_format}")
        print(f"  Split: {split}")
        print(f"  Samples: {len(self.data_triplets)}")
        print(f"  Data directory: {data_dir}")
    
    def _get_default_transform(self) -> transforms.Compose:
        """Get default image transformations"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _detect_dataset_format(self) -> str:
        """Auto-detect dataset format based on directory structure"""
        
        # Check for BAPPS format
        if (self.data_dir / 'train').exists() or (self.data_dir / 'val').exists():
            return 'bapps'
        
        # Check for CSV format
        csv_files = list(self.data_dir.glob('*.csv'))
        if csv_files:
            return 'csv'
        
        # Check for JSON format
        json_files = list(self.data_dir.glob('*.json'))
        if json_files:
            return 'json'
        
        # Default to 2AFC format
        return '2afc'
    
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset based on detected/specified format"""
        
        if self.dataset_format == 'bapps':
            return self._load_bapps_format()
        elif self.dataset_format == 'csv':
            return self._load_csv_format()
        elif self.dataset_format == 'json':
            return self._load_json_format()
        elif self.dataset_format == '2afc':
            return self._load_2afc_format()
        else:
            raise ValueError(f"Unknown dataset format: {self.dataset_format}")
    
    def _load_bapps_format(self) -> List[Dict[str, Any]]:
        """Load BAPPS (Berkeley Adobe Perceptual Patch Similarity) format"""
        triplets = []
        
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            split_dir = self.data_dir  # Fallback to main directory
        
        # Look for subdirectories with perceptual data
        for category_dir in split_dir.iterdir():
            if category_dir.is_dir():
                # Look for triplet files
                ref_images = sorted(category_dir.glob('*ref*.png')) or sorted(category_dir.glob('*ref*.jpg'))
                
                for ref_img in ref_images:
                    base_name = ref_img.stem.replace('_ref', '')
                    
                    # Find corresponding comparison images
                    img1_path = category_dir / f"{base_name}_img1.png"
                    img2_path = category_dir / f"{base_name}_img2.png"
                    
                    if not img1_path.exists():
                        img1_path = category_dir / f"{base_name}_img1.jpg"
                    if not img2_path.exists():
                        img2_path = category_dir / f"{base_name}_img2.jpg"
                    
                    if img1_path.exists() and img2_path.exists():
                        # Look for judgment file
                        judgment_file = category_dir / f"{base_name}_judgment.txt"
                        
                        if judgment_file.exists():
                            with open(judgment_file, 'r') as f:
                                judgment = int(float(f.read().strip()))
                        else:
                            # Random judgment if not available
                            judgment = random.randint(0, 1)
                        
                        triplets.append({
                            'ref_path': str(ref_img),
                            'img1_path': str(img1_path),
                            'img2_path': str(img2_path),
                            'judgment': judgment,
                            'category': category_dir.name
                        })
        
        return triplets
    
    def _load_csv_format(self) -> List[Dict[str, Any]]:
        """Load dataset from CSV file"""
        triplets = []
        
        # Find CSV file for the split
        csv_file = self.data_dir / f"{self.split}.csv"
        if not csv_file.exists():
            # Look for any CSV file
            csv_files = list(self.data_dir.glob('*.csv'))
            if csv_files:
                csv_file = csv_files[0]
            else:
                raise FileNotFoundError(f"No CSV file found in {self.data_dir}")
        
        # Load CSV data
        df = pd.read_csv(csv_file)
        
        required_columns = ['ref_path', 'img1_path', 'img2_path', 'judgment']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        
        for _, row in df.iterrows():
            # Convert relative paths to absolute
            ref_path = self.data_dir / row['ref_path']
            img1_path = self.data_dir / row['img1_path']
            img2_path = self.data_dir / row['img2_path']
            
            if ref_path.exists() and img1_path.exists() and img2_path.exists():
                triplets.append({
                    'ref_path': str(ref_path),
                    'img1_path': str(img1_path),
                    'img2_path': str(img2_path),
                    'judgment': int(row['judgment']),
                    'category': row.get('category', 'unknown')
                })
        
        return triplets
    
    def _load_json_format(self) -> List[Dict[str, Any]]:
        """Load dataset from JSON file"""
        triplets = []
        
        # Find JSON file for the split
        json_file = self.data_dir / f"{self.split}.json"
        if not json_file.exists():
            json_files = list(self.data_dir.glob('*.json'))
            if json_files:
                json_file = json_files[0]
            else:
                raise FileNotFoundError(f"No JSON file found in {self.data_dir}")
        
        # Load JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            triplet_data = data
        elif isinstance(data, dict):
            triplet_data = data.get('triplets', data.get('data', []))
        else:
            raise ValueError("Invalid JSON format")
        
        for item in triplet_data:
            ref_path = self.data_dir / item['ref_path']
            img1_path = self.data_dir / item['img1_path']
            img2_path = self.data_dir / item['img2_path']
            
            if ref_path.exists() and img1_path.exists() and img2_path.exists():
                triplets.append({
                    'ref_path': str(ref_path),
                    'img1_path': str(img1_path),
                    'img2_path': str(img2_path),
                    'judgment': int(item['judgment']),
                    'category': item.get('category', 'unknown')
                })
        
        return triplets
    
    def _load_2afc_format(self) -> List[Dict[str, Any]]:
        """Load generic 2AFC format by scanning directory structure"""
        triplets = []
        
        # Scan for image triplets
        image_extensions = ['*.png', '*.jpg', '*.jpeg']
        all_images = []
        
        for ext in image_extensions:
            all_images.extend(self.data_dir.rglob(ext))
        
        # Group images by base name
        image_groups = {}
        for img_path in all_images:
            base_name = img_path.stem
            
            # Remove common suffixes to group related images
            for suffix in ['_ref', '_img1', '_img2', '_0', '_1', '_reference', '_distorted1', '_distorted2']:
                if base_name.endswith(suffix):
                    base_name = base_name.replace(suffix, '')
                    break
            
            if base_name not in image_groups:
                image_groups[base_name] = []
            image_groups[base_name].append(img_path)
        
        # Create triplets from groups with 3 images
        for base_name, images in image_groups.items():
            if len(images) >= 3:
                # Sort images to have consistent ordering
                images = sorted(images, key=lambda x: x.name)
                
                # Assume first is reference, next two are comparisons
                ref_img = images[0]
                img1 = images[1]
                img2 = images[2]
                
                # Random judgment if not specified
                judgment = random.randint(0, 1)
                
                triplets.append({
                    'ref_path': str(ref_img),
                    'img1_path': str(img1),
                    'img2_path': str(img2),
                    'judgment': judgment,
                    'category': ref_img.parent.name
                })
        
        return triplets
    
    def _preload_images(self):
        """Preload all images into memory"""
        print("Preloading images into memory...")
        self.image_cache = {}
        
        for triplet in self.data_triplets:
            for key in ['ref_path', 'img1_path', 'img2_path']:
                img_path = triplet[key]
                if img_path not in self.image_cache:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        self.image_cache[img_path] = img
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                        self.image_cache[img_path] = None
    
    def __len__(self) -> int:
        return len(self.data_triplets)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a triplet sample
        
        Returns:
            ref_img: Reference image tensor
            img1: Comparison image 1 tensor  
            img2: Comparison image 2 tensor
            judgment: Human judgment (0 for img1 preferred, 1 for img2 preferred)
        """
        triplet = self.data_triplets[idx]
        
        # Load images
        if self.load_in_memory:
            ref_img = self.image_cache[triplet['ref_path']]
            img1 = self.image_cache[triplet['img1_path']]
            img2 = self.image_cache[triplet['img2_path']]
        else:
            ref_img = Image.open(triplet['ref_path']).convert('RGB')
            img1 = Image.open(triplet['img1_path']).convert('RGB')
            img2 = Image.open(triplet['img2_path']).convert('RGB')
        
        # Apply transformations
        if self.transform:
            ref_img = self.transform(ref_img)
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        # Convert judgment to tensor
        judgment = torch.tensor(triplet['judgment'], dtype=torch.long)
        
        return ref_img, img1, img2, judgment
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get information about a sample"""
        return self.data_triplets[idx]
    
    def get_category_distribution(self) -> Dict[str, int]:
        """Get distribution of categories in dataset"""
        categories = {}
        for triplet in self.data_triplets:
            category = triplet.get('category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
        return categories


class LPIPSDataModule:
    """
    Data module for LPIPS training and evaluation
    Handles data loading, splitting, and augmentation
    """
    
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 train_split: float = 0.8,
                 val_split: float = 0.1,
                 augment_training: bool = True):
        """
        Initialize LPIPS data module
        
        Args:
            data_dir: Path to JND dataset
            batch_size: Batch size for data loaders
            num_workers: Number of data loading workers
            pin_memory: Pin memory for faster GPU transfer
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            augment_training: Whether to augment training data
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_split = train_split
        self.val_split = val_split
        self.augment_training = augment_training
        
        # Create transforms
        self.train_transform = self._get_train_transform()
        self.val_transform = self._get_val_transform()
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def _get_train_transform(self) -> transforms.Compose:
        """Get training transforms with augmentation"""
        transforms_list = [
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
        ]
        
        if self.augment_training:
            transforms_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            ])
        
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        return transforms.Compose(transforms_list)
    
    def _get_val_transform(self) -> transforms.Compose:
        """Get validation transforms without augmentation"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training/validation/testing"""
        
        # Try to load pre-split datasets
        train_dir = Path(self.data_dir) / 'train'
        val_dir = Path(self.data_dir) / 'val'
        test_dir = Path(self.data_dir) / 'test'
        
        if train_dir.exists() and val_dir.exists():
            # Use existing splits
            print("Using existing train/val/test splits")
            
            self.train_dataset = JNDDataset(
                data_dir=str(train_dir),
                split='train',
                transform=self.train_transform
            )
            
            self.val_dataset = JNDDataset(
                data_dir=str(val_dir),
                split='val',
                transform=self.val_transform
            )
            
            if test_dir.exists():
                self.test_dataset = JNDDataset(
                    data_dir=str(test_dir),
                    split='test',
                    transform=self.val_transform
                )
        
        else:
            # Create splits from full dataset
            print("Creating train/val/test splits from full dataset")
            
            full_dataset = JNDDataset(
                data_dir=self.data_dir,
                split='all',
                transform=self.val_transform
            )
            
            # Split dataset
            total_size = len(full_dataset)
            train_size = int(self.train_split * total_size)
            val_size = int(self.val_split * total_size)
            test_size = total_size - train_size - val_size
            
            train_indices, val_indices, test_indices = torch.utils.data.random_split(
                range(total_size), [train_size, val_size, test_size]
            )
            
            # Create subset datasets
            self.train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
            self.val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
            self.test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
            
            # Update transforms for training subset
            train_full_dataset = JNDDataset(
                data_dir=self.data_dir,
                split='all',
                transform=self.train_transform
            )
            self.train_dataset = torch.utils.data.Subset(train_full_dataset, train_indices)
    
    def train_dataloader(self) -> DataLoader:
        """Get training data loader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation data loader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test data loader"""
        if self.test_dataset is None:
            return self.val_dataloader()
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the datasets"""
        info = {
            'total_samples': 0,
            'train_samples': 0,
            'val_samples': 0,
            'test_samples': 0,
            'categories': {}
        }
        
        if self.train_dataset:
            info['train_samples'] = len(self.train_dataset)
        if self.val_dataset:
            info['val_samples'] = len(self.val_dataset)
        if self.test_dataset:
            info['test_samples'] = len(self.test_dataset)
        
        info['total_samples'] = info['train_samples'] + info['val_samples'] + info['test_samples']
        
        return info


def create_synthetic_jnd_dataset(output_dir: str, 
                                num_samples: int = 1000,
                                image_size: Tuple[int, int] = (224, 224)) -> None:
    """
    Create a synthetic JND dataset for testing purposes
    
    Args:
        output_dir: Output directory for synthetic dataset
        num_samples: Number of samples to create
        image_size: Size of generated images
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create directories
    (output_path / 'train').mkdir(exist_ok=True)
    (output_path / 'val').mkdir(exist_ok=True)
    (output_path / 'test').mkdir(exist_ok=True)
    
    splits = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    current_idx = 0
    
    for split, ratio in splits.items():
        split_samples = int(num_samples * ratio)
        split_dir = output_path / split
        
        triplets_data = []
        
        for i in range(split_samples):
            # Generate reference image
            ref_img = torch.rand(3, *image_size)
            
            # Generate two comparison images with different noise levels
            noise1 = torch.randn_like(ref_img) * 0.1
            noise2 = torch.randn_like(ref_img) * 0.2
            
            img1 = torch.clamp(ref_img + noise1, 0, 1)
            img2 = torch.clamp(ref_img + noise2, 0, 1)
            
            # Human judgment: prefer less noisy image (img1)
            judgment = 0 if torch.norm(noise1) < torch.norm(noise2) else 1
            
            # Save images
            sample_id = f"sample_{current_idx:06d}"
            
            ref_path = split_dir / f"{sample_id}_ref.png"
            img1_path = split_dir / f"{sample_id}_img1.png"
            img2_path = split_dir / f"{sample_id}_img2.png"
            
            # Convert to PIL and save
            transforms.ToPILImage()(ref_img).save(ref_path)
            transforms.ToPILImage()(img1).save(img1_path)
            transforms.ToPILImage()(img2).save(img2_path)
            
            # Store triplet data
            triplets_data.append({
                'ref_path': f"{sample_id}_ref.png",
                'img1_path': f"{sample_id}_img1.png", 
                'img2_path': f"{sample_id}_img2.png",
                'judgment': judgment,
                'category': 'synthetic'
            })
            
            current_idx += 1
        
        # Save metadata
        import pandas as pd
        df = pd.DataFrame(triplets_data)
        df.to_csv(split_dir / f"{split}.csv", index=False)
    
    print(f"Created synthetic JND dataset with {num_samples} samples in {output_dir}")


def visualize_jnd_dataset(dataset: JNDDataset, num_samples: int = 5) -> None:
    """
    Visualize samples from JND dataset
    
    Args:
        dataset: JND dataset to visualize
        num_samples: Number of samples to show
    """
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(num_samples, len(dataset))):
        ref_img, img1, img2, judgment = dataset[i]
        
        # Convert tensors to numpy for visualization
        def tensor_to_numpy(tensor):
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            denorm = tensor * std + mean
            return torch.clamp(denorm, 0, 1).permute(1, 2, 0).numpy()
        
        # Plot images
        axes[i, 0].imshow(tensor_to_numpy(ref_img))
        axes[i, 0].set_title('Reference')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(tensor_to_numpy(img1))
        axes[i, 1].set_title(f'Image 1 {"(Preferred)" if judgment == 0 else ""}')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(tensor_to_numpy(img2))
        axes[i, 2].set_title(f'Image 2 {"(Preferred)" if judgment == 1 else ""}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    """Demonstration of LPIPS data loading"""
    print("=" * 60)
    print("LPIPS Data Loading Demonstration")
    print("=" * 60)
    
    # Create synthetic dataset for demonstration
    synthetic_dir = "./synthetic_jnd_data"
    print("Creating synthetic JND dataset...")
    create_synthetic_jnd_dataset(synthetic_dir, num_samples=100)
    
    # Test data loading
    print("\nTesting data loading...")
    
    # Create data module
    data_module = LPIPSDataModule(
        data_dir=synthetic_dir,
        batch_size=8,
        num_workers=2
    )
    
    # Setup datasets
    data_module.setup()
    
    # Get dataset info
    info = data_module.get_dataset_info()
    print(f"Dataset info: {info}")
    
    # Test data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test a batch
    for batch_idx, (ref_imgs, img1s, img2s, judgments) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Reference images: {ref_imgs.shape}")
        print(f"  Comparison images 1: {img1s.shape}")
        print(f"  Comparison images 2: {img2s.shape}")
        print(f"  Judgments: {judgments.shape}")
        print(f"  Judgment values: {judgments.tolist()}")
        
        if batch_idx >= 2:  # Show only first few batches
            break
    
    # Visualize some samples
    print("\nVisualizing dataset samples...")
    train_dataset = JNDDataset(
        data_dir=f"{synthetic_dir}/train",
        transform=data_module.val_transform  # Use val transform for visualization
    )
    
    visualize_jnd_dataset(train_dataset, num_samples=3)
    
    print("\nData loading demonstration complete!")


if __name__ == "__main__":
    main()