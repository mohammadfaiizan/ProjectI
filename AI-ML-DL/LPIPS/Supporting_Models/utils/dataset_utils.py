"""
Shared Dataset Utilities for LPIPS Supporting Models
===================================================

Common utilities for ImageNet data loading, preprocessing, and dataset management
for AlexNet, VGG, and SqueezeNet implementations.

This module provides:
- Standardized ImageNet data loading
- Common preprocessing pipelines
- Dataset validation and analysis
- Data augmentation strategies
- Benchmark dataset creation

Author: [Your Name]
Date: [Current Date]
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet, CIFAR10, CIFAR100
import numpy as np
import os
import json
import pickle
from PIL import Image
import requests
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from collections import defaultdict


class StandardImageNetLoader:
    """
    Standardized ImageNet data loader with consistent preprocessing
    across all LPIPS supporting models
    """
    
    def __init__(self, data_dir: str = './data', download_imagenet: bool = False):
        self.data_dir = data_dir
        self.download_imagenet = download_imagenet
        
        # Standard ImageNet statistics
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Check ImageNet availability
        self.imagenet_available = self._check_imagenet_availability()
        
    def _check_imagenet_availability(self) -> bool:
        """Check if ImageNet dataset is available"""
        try:
            # Try to create ImageNet dataset (without loading)
            _ = ImageNet(root=self.data_dir, split='train', download=False)
            print("ImageNet dataset found!")
            return True
        except Exception as e:
            print(f"ImageNet dataset not found: {e}")
            print("Will use CIFAR-10 as fallback for demonstration")
            return False
    
    def get_transforms(self, model_type: str = 'standard', augment: bool = True) -> Dict[str, transforms.Compose]:
        """
        Get appropriate transforms for different models
        
        Args:
            model_type: 'alexnet', 'vgg', 'squeezenet', or 'standard'
            augment: Whether to apply data augmentation for training
            
        Returns:
            Dictionary with 'train' and 'val' transforms
        """
        
        # Base transforms for all models
        base_train_transforms = [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        
        base_val_transforms = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
        
        # Model-specific augmentations
        if model_type == 'alexnet' and augment:
            # AlexNet-style augmentation (from original paper)
            train_augment = [
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ]
        elif model_type == 'vgg' and augment:
            # VGG-style augmentation
            train_augment = [
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            ]
        elif model_type == 'squeezenet' and augment:
            # SqueezeNet-style augmentation (similar to VGG)
            train_augment = [
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            ]
        else:
            # Standard augmentation
            train_augment = [
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ]
        
        # Final transforms (same for all models)
        final_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.imagenet_mean, std=self.imagenet_std)
        ]
        
        # Combine transforms
        if augment:
            train_transform = transforms.Compose(
                base_train_transforms + train_augment + final_transforms
            )
        else:
            train_transform = transforms.Compose(
                base_train_transforms + final_transforms
            )
        
        val_transform = transforms.Compose(
            base_val_transforms + final_transforms
        )
        
        return {
            'train': train_transform,
            'val': val_transform
        }
    
    def create_data_loaders(self, 
                          model_type: str = 'standard',
                          batch_size: int = 256,
                          num_workers: int = 4,
                          pin_memory: bool = True,
                          subset_size: Optional[int] = None) -> Dict[str, DataLoader]:
        """
        Create standardized data loaders
        
        Args:
            model_type: Type of model for appropriate preprocessing
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for faster GPU transfer
            subset_size: If specified, use only a subset of data for quick testing
            
        Returns:
            Dictionary with 'train' and 'val' data loaders
        """
        
        # Get appropriate transforms
        transforms_dict = self.get_transforms(model_type, augment=True)
        
        if self.imagenet_available:
            # Use ImageNet
            print("Loading ImageNet dataset...")
            
            train_dataset = ImageNet(
                root=self.data_dir,
                split='train',
                transform=transforms_dict['train']
            )
            
            val_dataset = ImageNet(
                root=self.data_dir,
                split='val',
                transform=transforms_dict['val']
            )
            
        else:
            # Fallback to CIFAR-10
            print("Loading CIFAR-10 dataset as fallback...")
            
            train_dataset = CIFAR10(
                root=self.data_dir,
                train=True,
                transform=transforms_dict['train'],
                download=True
            )
            
            val_dataset = CIFAR10(
                root=self.data_dir,
                train=False,
                transform=transforms_dict['val'],
                download=True
            )
        
        # Create subsets if requested
        if subset_size is not None:
            print(f"Creating subset with {subset_size} samples per split")
            train_indices = np.random.choice(len(train_dataset), subset_size, replace=False)
            val_indices = np.random.choice(len(val_dataset), min(subset_size//10, len(val_dataset)), replace=False)
            
            train_dataset = Subset(train_dataset, train_indices)
            val_dataset = Subset(val_dataset, val_indices)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Batch size: {batch_size}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        return {
            'train': train_loader,
            'val': val_loader
        }
    
    def create_lpips_evaluation_dataset(self, num_pairs: int = 1000) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Create a dataset specifically for LPIPS evaluation
        
        Args:
            num_pairs: Number of image pairs to create
            
        Returns:
            List of (image1, image2) tensor pairs
        """
        
        # Get validation transform (no augmentation)
        val_transform = self.get_transforms()['val']
        
        if self.imagenet_available:
            dataset = ImageNet(
                root=self.data_dir,
                split='val',
                transform=val_transform
            )
        else:
            dataset = CIFAR10(
                root=self.data_dir,
                train=False,
                transform=val_transform,
                download=True
            )
        
        # Randomly sample pairs
        indices = np.random.choice(len(dataset), num_pairs * 2, replace=False)
        pairs = []
        
        for i in range(0, len(indices), 2):
            img1, _ = dataset[indices[i]]
            img2, _ = dataset[indices[i + 1]]
            pairs.append((img1, img2))
        
        print(f"Created {len(pairs)} image pairs for LPIPS evaluation")
        return pairs
    
    def analyze_dataset_statistics(self) -> Dict:
        """Analyze dataset statistics for verification"""
        
        # Create a small sample loader
        sample_loader = self.create_data_loaders(batch_size=100, subset_size=1000)['train']
        
        # Collect statistics
        mean_values = []
        std_values = []
        min_values = []
        max_values = []
        
        for batch_data, _ in sample_loader:
            # Calculate per-channel statistics
            for c in range(3):  # RGB channels
                channel_data = batch_data[:, c, :, :].flatten()
                mean_values.append(channel_data.mean().item())
                std_values.append(channel_data.std().item())
                min_values.append(channel_data.min().item())
                max_values.append(channel_data.max().item())
        
        # Average across batches
        stats = {
            'mean_per_channel': [
                np.mean([mean_values[i] for i in range(0, len(mean_values), 3)]),
                np.mean([mean_values[i] for i in range(1, len(mean_values), 3)]),
                np.mean([mean_values[i] for i in range(2, len(mean_values), 3)])
            ],
            'std_per_channel': [
                np.mean([std_values[i] for i in range(0, len(std_values), 3)]),
                np.mean([std_values[i] for i in range(1, len(std_values), 3)]),
                np.mean([std_values[i] for i in range(2, len(std_values), 3)])
            ],
            'expected_imagenet_mean': self.imagenet_mean,
            'expected_imagenet_std': self.imagenet_std
        }
        
        return stats


class LPIPSDatasetCreator:
    """
    Create specialized datasets for LPIPS training and evaluation
    """
    
    def __init__(self, base_loader: StandardImageNetLoader):
        self.base_loader = base_loader
    
    def create_distortion_dataset(self, distortion_types: List[str], num_samples: int = 1000) -> Dict:
        """
        Create dataset with various distortions for LPIPS evaluation
        
        Args:
            distortion_types: List of distortion types to apply
            num_samples: Number of samples per distortion type
            
        Returns:
            Dictionary with distorted image pairs and metadata
        """
        
        # Get clean images
        val_transform = self.base_loader.get_transforms()['val']
        
        if self.base_loader.imagenet_available:
            clean_dataset = ImageNet(
                root=self.base_loader.data_dir,
                split='val',
                transform=val_transform
            )
        else:
            clean_dataset = CIFAR10(
                root=self.base_loader.data_dir,
                train=False,
                transform=val_transform,
                download=True
            )
        
        distortion_dataset = {}
        
        for distortion_type in distortion_types:
            print(f"Creating {distortion_type} distortion dataset...")
            
            distorted_pairs = []
            indices = np.random.choice(len(clean_dataset), num_samples, replace=False)
            
            for idx in indices:
                clean_img, label = clean_dataset[idx]
                distorted_img = self._apply_distortion(clean_img, distortion_type)
                
                distorted_pairs.append({
                    'clean': clean_img,
                    'distorted': distorted_img,
                    'label': label,
                    'distortion_type': distortion_type
                })
            
            distortion_dataset[distortion_type] = distorted_pairs
        
        return distortion_dataset
    
    def _apply_distortion(self, image: torch.Tensor, distortion_type: str) -> torch.Tensor:
        """Apply specific distortion to an image"""
        
        # Convert to PIL for easier manipulation
        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(
            mean=self.base_loader.imagenet_mean,
            std=self.base_loader.imagenet_std
        )
        
        # Denormalize first
        denorm_img = image * torch.tensor(self.base_loader.imagenet_std).view(3, 1, 1)
        denorm_img += torch.tensor(self.base_loader.imagenet_mean).view(3, 1, 1)
        denorm_img = torch.clamp(denorm_img, 0, 1)
        
        pil_img = to_pil(denorm_img)
        
        if distortion_type == 'jpeg_compression':
            # JPEG compression
            quality = np.random.randint(10, 50)  # Low quality
            pil_img.save('temp.jpg', 'JPEG', quality=quality)
            pil_img = Image.open('temp.jpg')
            os.remove('temp.jpg')
            
        elif distortion_type == 'gaussian_noise':
            # Gaussian noise
            img_array = np.array(pil_img)
            noise = np.random.normal(0, 25, img_array.shape)
            noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(noisy_img)
            
        elif distortion_type == 'gaussian_blur':
            # Gaussian blur
            from PIL import ImageFilter
            radius = np.random.uniform(1.0, 3.0)
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
            
        elif distortion_type == 'color_shift':
            # Color shift
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Color(pil_img)
            factor = np.random.uniform(0.5, 1.5)
            pil_img = enhancer.enhance(factor)
            
        elif distortion_type == 'brightness_change':
            # Brightness change
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(pil_img)
            factor = np.random.uniform(0.5, 1.5)
            pil_img = enhancer.enhance(factor)
            
        # Convert back to tensor and normalize
        tensor_img = to_tensor(pil_img)
        normalized_img = normalize(tensor_img)
        
        return normalized_img
    
    def create_human_preference_dataset(self, num_triplets: int = 500) -> List[Dict]:
        """
        Create dataset for human preference simulation
        (For research purposes - actual human annotations would be needed for real training)
        
        Args:
            num_triplets: Number of (reference, option1, option2) triplets
            
        Returns:
            List of triplet dictionaries with simulated preferences
        """
        
        print("Creating simulated human preference dataset...")
        print("Note: This uses algorithmic preferences, not real human annotations")
        
        # Create distorted images
        distortions = ['jpeg_compression', 'gaussian_noise', 'gaussian_blur']
        distortion_data = self.create_distortion_dataset(distortions, num_triplets)
        
        triplets = []
        
        for i in range(num_triplets):
            # Get reference image
            distortion_type = np.random.choice(distortions)
            sample_idx = np.random.randint(len(distortion_data[distortion_type]))
            
            reference = distortion_data[distortion_type][sample_idx]['clean']
            
            # Create two distorted versions with different severities
            option1 = distortion_data[distortion_type][sample_idx]['distorted']
            
            # Create second option with different distortion
            other_distortion = np.random.choice([d for d in distortions if d != distortion_type])
            option2 = self._apply_distortion(reference, other_distortion)
            
            # Simulate preference (closer to reference is preferred)
            # In real scenarios, this would come from human annotators
            ref_option1_dist = torch.nn.functional.mse_loss(reference, option1).item()
            ref_option2_dist = torch.nn.functional.mse_loss(reference, option2).item()
            
            preference = 0 if ref_option1_dist < ref_option2_dist else 1
            
            triplets.append({
                'reference': reference,
                'option1': option1,
                'option2': option2,
                'preference': preference,  # 0 for option1, 1 for option2
                'option1_distortion': distortion_type,
                'option2_distortion': other_distortion
            })
        
        return triplets


class DatasetValidator:
    """
    Validate and analyze datasets for LPIPS training
    """
    
    def __init__(self):
        pass
    
    def validate_data_loader(self, data_loader: DataLoader) -> Dict:
        """Validate a data loader"""
        
        validation_results = {
            'total_batches': len(data_loader),
            'batch_size': data_loader.batch_size,
            'dataset_size': len(data_loader.dataset),
            'shape_consistency': True,
            'value_range_valid': True,
            'sample_shapes': [],
            'sample_value_ranges': []
        }
        
        # Check first few batches
        for i, (data, target) in enumerate(data_loader):
            if i >= 5:  # Check only first 5 batches
                break
                
            # Check shapes
            batch_shape = data.shape
            validation_results['sample_shapes'].append(batch_shape)
            
            # Check value ranges (should be normalized)
            min_val = data.min().item()
            max_val = data.max().item()
            validation_results['sample_value_ranges'].append((min_val, max_val))
            
            # Validate expected ranges for ImageNet normalization
            if min_val < -3.0 or max_val > 3.0:
                validation_results['value_range_valid'] = False
        
        # Check shape consistency
        if len(set(validation_results['sample_shapes'])) > 1:
            validation_results['shape_consistency'] = False
        
        return validation_results
    
    def analyze_class_distribution(self, data_loader: DataLoader) -> Dict:
        """Analyze class distribution in dataset"""
        
        class_counts = defaultdict(int)
        total_samples = 0
        
        for _, targets in data_loader:
            for target in targets:
                class_counts[target.item()] += 1
                total_samples += 1
        
        # Calculate statistics
        num_classes = len(class_counts)
        samples_per_class = list(class_counts.values())
        
        return {
            'num_classes': num_classes,
            'total_samples': total_samples,
            'samples_per_class': samples_per_class,
            'mean_samples_per_class': np.mean(samples_per_class),
            'std_samples_per_class': np.std(samples_per_class),
            'min_samples_per_class': np.min(samples_per_class),
            'max_samples_per_class': np.max(samples_per_class),
            'class_balance_ratio': np.min(samples_per_class) / np.max(samples_per_class)
        }
    
    def create_validation_report(self, data_loaders: Dict[str, DataLoader], output_file: str = 'dataset_validation_report.json'):
        """Create comprehensive validation report"""
        
        report = {
            'validation_timestamp': str(torch.datetime.now()),
            'data_loaders': {}
        }
        
        for loader_name, loader in data_loaders.items():
            print(f"Validating {loader_name} data loader...")
            
            loader_validation = self.validate_data_loader(loader)
            class_analysis = self.analyze_class_distribution(loader)
            
            report['data_loaders'][loader_name] = {
                'validation': loader_validation,
                'class_analysis': class_analysis
            }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Validation report saved to {output_file}")
        return report


def main():
    """Demonstration of dataset utilities"""
    
    print("=== Dataset Utilities Demonstration ===\n")
    
    # Create standard ImageNet loader
    loader = StandardImageNetLoader(data_dir='./data')
    
    # Test dataset statistics
    print("1. Analyzing dataset statistics...")
    stats = loader.analyze_dataset_statistics()
    print(f"Computed mean: {stats['mean_per_channel']}")
    print(f"Expected mean: {stats['expected_imagenet_mean']}")
    print(f"Computed std: {stats['std_per_channel']}")
    print(f"Expected std: {stats['expected_imagenet_std']}")
    
    # Create data loaders for different models
    print("\n2. Creating data loaders for different models...")
    
    models = ['alexnet', 'vgg', 'squeezenet']
    data_loaders = {}
    
    for model in models:
        print(f"Creating data loaders for {model}...")
        loaders = loader.create_data_loaders(
            model_type=model,
            batch_size=64,
            subset_size=1000  # Small subset for demo
        )
        data_loaders[model] = loaders
    
    # Validate data loaders
    print("\n3. Validating data loaders...")
    validator = DatasetValidator()
    
    for model in models:
        validation_report = validator.create_validation_report(
            {f'{model}_train': data_loaders[model]['train'],
             f'{model}_val': data_loaders[model]['val']},
            output_file=f'{model}_validation_report.json'
        )
    
    # Create LPIPS evaluation dataset
    print("\n4. Creating LPIPS evaluation dataset...")
    lpips_creator = LPIPSDatasetCreator(loader)
    
    # Create distortion dataset
    distortion_dataset = lpips_creator.create_distortion_dataset(
        distortion_types=['jpeg_compression', 'gaussian_noise', 'gaussian_blur'],
        num_samples=100
    )
    
    print(f"Created distortion dataset with {len(distortion_dataset)} distortion types")
    for distortion_type, samples in distortion_dataset.items():
        print(f"  {distortion_type}: {len(samples)} samples")
    
    # Create human preference dataset (simulated)
    preference_dataset = lpips_creator.create_human_preference_dataset(num_triplets=50)
    print(f"Created preference dataset with {len(preference_dataset)} triplets")
    
    # Create LPIPS evaluation pairs
    eval_pairs = loader.create_lpips_evaluation_dataset(num_pairs=100)
    print(f"Created {len(eval_pairs)} image pairs for LPIPS evaluation")
    
    print("\n=== Dataset utilities demonstration completed ===")


if __name__ == "__main__":
    main()