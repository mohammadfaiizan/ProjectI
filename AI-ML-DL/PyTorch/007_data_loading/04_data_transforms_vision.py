#!/usr/bin/env python3
"""PyTorch Data Transforms Vision - Vision transforms and syntax"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import v2 as transforms_v2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
import random
import math
from typing import Tuple, List, Optional, Union

print("=== Vision Transforms Overview ===")

print("Vision transforms covered:")
print("1. Basic geometric transforms")
print("2. Color and intensity transforms") 
print("3. Tensor conversion and normalization")
print("4. Composition and pipelines")
print("5. Data augmentation transforms")
print("6. Custom transform implementation")
print("7. Transform pipelines for different tasks")
print("8. Performance optimization")

print("\n=== Basic Geometric Transforms ===")

# Create sample image for demonstration
def create_sample_image(size=(224, 224), color='RGB'):
    """Create a sample image for testing transforms"""
    img = Image.new(color, size, color=(100, 150, 200))
    
    # Add some patterns
    draw = ImageDraw.Draw(img)
    
    # Draw rectangles
    draw.rectangle([20, 20, 80, 80], fill=(255, 100, 100))
    draw.rectangle([120, 120, 180, 180], fill=(100, 255, 100))
    
    # Draw circles
    draw.ellipse([50, 150, 100, 200], fill=(100, 100, 255))
    draw.ellipse([150, 50, 200, 100], fill=(255, 255, 100))
    
    return img

sample_image = create_sample_image()
print(f"Sample image size: {sample_image.size}")

# Basic geometric transforms
print("\nTesting basic geometric transforms:")

geometric_transforms = {
    'Resize': transforms.Resize((128, 128)),
    'CenterCrop': transforms.CenterCrop(150),
    'RandomCrop': transforms.RandomCrop(100),
    'RandomResizedCrop': transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    'Pad': transforms.Pad(20, fill=0, padding_mode='constant'),
    'RandomRotation': transforms.RandomRotation(30),
    'RandomHorizontalFlip': transforms.RandomHorizontalFlip(p=1.0),
    'RandomVerticalFlip': transforms.RandomVerticalFlip(p=1.0),
    'RandomAffine': transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    'RandomPerspective': transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
}

for name, transform in geometric_transforms.items():
    try:
        transformed = transform(sample_image)
        print(f"  {name:20}: {sample_image.size} -> {transformed.size}")
    except Exception as e:
        print(f"  {name:20}: Error - {e}")

print("\n=== Color and Intensity Transforms ===")

# Color and intensity transforms
color_transforms = {
    'ColorJitter': transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
    'Grayscale': transforms.Grayscale(num_output_channels=3),
    'RandomGrayscale': transforms.RandomGrayscale(p=1.0),
    'GaussianBlur': transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    'RandomInvert': transforms.RandomInvert(p=1.0),
    'RandomPosterize': transforms.RandomPosterize(bits=2, p=1.0),
    'RandomSolarize': transforms.RandomSolarize(threshold=128, p=1.0),
    'RandomEqualize': transforms.RandomEqualize(p=1.0),
    'RandomAutocontrast': transforms.RandomAutocontrast(p=1.0),
}

print("Testing color and intensity transforms:")
for name, transform in color_transforms.items():
    try:
        transformed = transform(sample_image)
        print(f"  {name:20}: Applied successfully")
    except Exception as e:
        print(f"  {name:20}: Error - {e}")

print("\n=== Tensor Conversion and Normalization ===")

# Tensor conversion and normalization
print("Testing tensor conversion and normalization:")

# Convert to tensor
to_tensor = transforms.ToTensor()
tensor_image = to_tensor(sample_image)
print(f"  ToTensor: PIL Image -> Tensor shape {tensor_image.shape}")
print(f"  Tensor range: [{tensor_image.min():.3f}, {tensor_image.max():.3f}]")

# Normalization
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalized_tensor = normalize(tensor_image)
print(f"  Normalize: mean={normalized_tensor.mean(dim=[1,2])}")
print(f"             std={normalized_tensor.std(dim=[1,2])}")

# Convert back to PIL
to_pil = transforms.ToPILImage()
pil_image = to_pil(tensor_image)
print(f"  ToPILImage: Tensor -> PIL Image size {pil_image.size}")

print("\n=== Transform Composition ===")

# Compose multiple transforms
print("Testing transform composition:")

# Training pipeline
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation pipeline
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Training transform pipeline:")
train_result = train_transform(sample_image)
print(f"  Result shape: {train_result.shape}")
print(f"  Result range: [{train_result.min():.3f}, {train_result.max():.3f}]")

print("Validation transform pipeline:")
val_result = val_transform(sample_image)
print(f"  Result shape: {val_result.shape}")
print(f"  Result range: [{val_result.min():.3f}, {val_result.max():.3f}]")

print("\n=== Advanced Data Augmentation ===")

class AdvancedAugmentation:
    """Advanced augmentation techniques"""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def cutout(self, img, n_holes=1, length=16):
        """Apply cutout augmentation"""
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)
        
        h, w = img.shape[1], img.shape[2]
        mask = torch.ones((h, w), dtype=torch.float32)
        
        for _ in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)
            
            mask[y1:y2, x1:x2] = 0.0
        
        img = img * mask.unsqueeze(0)
        return img
    
    def mixup(self, img1, img2, alpha=0.2):
        """Apply mixup augmentation"""
        if isinstance(img1, Image.Image):
            img1 = transforms.ToTensor()(img1)
        if isinstance(img2, Image.Image):
            img2 = transforms.ToTensor()(img2)
        
        lam = np.random.beta(alpha, alpha)
        mixed_img = lam * img1 + (1 - lam) * img2
        return mixed_img, lam
    
    def random_erasing(self, img, area_ratio_range=(0.02, 0.4), aspect_ratio_range=(0.3, 3.3)):
        """Apply random erasing"""
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)
        
        img = img.clone()
        _, h, w = img.shape
        
        target_area = np.random.uniform(*area_ratio_range) * area
        aspect_ratio = np.random.uniform(*aspect_ratio_range)
        
        h_erased = int(round(np.sqrt(target_area * aspect_ratio)))
        w_erased = int(round(np.sqrt(target_area / aspect_ratio)))
        
        if h_erased < h and w_erased < w:
            x1 = np.random.randint(0, w - w_erased)
            y1 = np.random.randint(0, h - h_erased)
            
            img[:, y1:y1+h_erased, x1:x1+w_erased] = torch.rand_like(
                img[:, y1:y1+h_erased, x1:x1+w_erased]
            )
        
        return img

# Test advanced augmentation
print("Testing advanced augmentation techniques:")
advanced_aug = AdvancedAugmentation()

# Cutout
cutout_result = advanced_aug.cutout(sample_image, n_holes=3, length=20)
print(f"  Cutout: Applied to image, result shape {cutout_result.shape}")

# Mixup (need two images)
sample_image2 = create_sample_image(color='RGB')
mixed_result, lam = advanced_aug.mixup(sample_image, sample_image2, alpha=0.5)
print(f"  Mixup: Mixed with lambda={lam:.3f}, result shape {mixed_result.shape}")

# Random erasing
erased_result = advanced_aug.random_erasing(sample_image)
print(f"  Random Erasing: Applied, result shape {erased_result.shape}")

print("\n=== Custom Transform Implementation ===")

class CustomTransforms:
    """Collection of custom transform implementations"""
    
    class RandomNoise:
        """Add random noise to image"""
        
        def __init__(self, noise_factor=0.1):
            self.noise_factor = noise_factor
        
        def __call__(self, img):
            if isinstance(img, Image.Image):
                img = transforms.ToTensor()(img)
            
            noise = torch.randn_like(img) * self.noise_factor
            noisy_img = torch.clamp(img + noise, 0, 1)
            return noisy_img
        
        def __repr__(self):
            return f"RandomNoise(noise_factor={self.noise_factor})"
    
    class ElasticTransform:
        """Apply elastic deformation"""
        
        def __init__(self, alpha=34, sigma=4, alpha_affine=0.05, random_state=None):
            self.alpha = alpha
            self.sigma = sigma
            self.alpha_affine = alpha_affine
            self.random_state = random_state
        
        def __call__(self, img):
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            
            img_array = np.array(img)
            shape = img_array.shape
            
            # Generate random displacement fields
            dx = cv2.GaussianBlur((np.random.rand(*shape[:2]) * 2 - 1), (0, 0), self.sigma) * self.alpha
            dy = cv2.GaussianBlur((np.random.rand(*shape[:2]) * 2 - 1), (0, 0), self.sigma) * self.alpha
            
            # Create coordinate grids
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            x_new = np.clip(x + dx, 0, shape[1] - 1).astype(np.float32)
            y_new = np.clip(y + dy, 0, shape[0] - 1).astype(np.float32)
            
            # Apply deformation
            if len(shape) == 3:  # Color image
                deformed = np.zeros_like(img_array)
                for c in range(shape[2]):
                    deformed[:, :, c] = cv2.remap(
                        img_array[:, :, c], x_new, y_new, 
                        interpolation=cv2.INTER_LINEAR, 
                        borderMode=cv2.BORDER_REFLECT
                    )
            else:  # Grayscale
                deformed = cv2.remap(
                    img_array, x_new, y_new, 
                    interpolation=cv2.INTER_LINEAR, 
                    borderMode=cv2.BORDER_REFLECT
                )
            
            return Image.fromarray(deformed.astype(np.uint8))
        
        def __repr__(self):
            return f"ElasticTransform(alpha={self.alpha}, sigma={self.sigma})"
    
    class GridDistortion:
        """Apply grid-based distortion"""
        
        def __init__(self, num_steps=5, distort_limit=0.3):
            self.num_steps = num_steps
            self.distort_limit = distort_limit
        
        def __call__(self, img):
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            
            img_array = np.array(img)
            height, width = img_array.shape[:2]
            
            # Create grid
            x_step = width // self.num_steps
            y_step = height // self.num_steps
            
            # Generate distorted grid points
            xx = np.zeros((self.num_steps + 1, self.num_steps + 1), dtype=np.float32)
            yy = np.zeros((self.num_steps + 1, self.num_steps + 1), dtype=np.float32)
            
            for i in range(self.num_steps + 1):
                for j in range(self.num_steps + 1):
                    xx[i, j] = j * x_step + np.random.uniform(
                        -self.distort_limit * x_step, 
                        self.distort_limit * x_step
                    )
                    yy[i, j] = i * y_step + np.random.uniform(
                        -self.distort_limit * y_step, 
                        self.distort_limit * y_step
                    )
            
            # Interpolate to full resolution
            xx = cv2.resize(xx, (width, height))
            yy = cv2.resize(yy, (width, height))
            
            # Apply distortion
            if len(img_array.shape) == 3:
                distorted = np.zeros_like(img_array)
                for c in range(img_array.shape[2]):
                    distorted[:, :, c] = cv2.remap(
                        img_array[:, :, c], xx, yy, 
                        interpolation=cv2.INTER_LINEAR
                    )
            else:
                distorted = cv2.remap(
                    img_array, xx, yy, 
                    interpolation=cv2.INTER_LINEAR
                )
            
            return Image.fromarray(distorted.astype(np.uint8))
    
    class ChannelShuffle:
        """Randomly shuffle color channels"""
        
        def __call__(self, img):
            if isinstance(img, Image.Image):
                img = transforms.ToTensor()(img)
            
            if img.shape[0] == 3:  # RGB image
                perm = torch.randperm(3)
                img = img[perm]
            
            return img

# Test custom transforms
print("Testing custom transforms:")

custom_transforms = {
    'RandomNoise': CustomTransforms.RandomNoise(noise_factor=0.1),
    'ElasticTransform': CustomTransforms.ElasticTransform(alpha=20, sigma=3),
    'GridDistortion': CustomTransforms.GridDistortion(num_steps=4, distort_limit=0.2),
    'ChannelShuffle': CustomTransforms.ChannelShuffle(),
}

for name, transform in custom_transforms.items():
    try:
        result = transform(sample_image)
        if isinstance(result, torch.Tensor):
            print(f"  {name:15}: Applied, result shape {result.shape}")
        else:
            print(f"  {name:15}: Applied, result size {result.size}")
    except Exception as e:
        print(f"  {name:15}: Error - {e}")

print("\n=== Task-Specific Transform Pipelines ===")

class TaskSpecificTransforms:
    """Transform pipelines for different computer vision tasks"""
    
    @staticmethod
    def classification_transforms(input_size=224, augment=True):
        """Transforms for image classification"""
        if augment:
            return transforms.Compose([
                transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(int(input_size * 1.14)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    @staticmethod
    def segmentation_transforms(augment=True):
        """Transforms for semantic segmentation (must be applied to both image and mask)"""
        if augment:
            return transforms.Compose([
                transforms.RandomResizedCrop(512, scale=(0.5, 2.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.ToTensor(),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
            ])
    
    @staticmethod
    def detection_transforms(augment=True):
        """Transforms for object detection"""
        if augment:
            return transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    @staticmethod
    def medical_transforms(augment=True):
        """Transforms for medical imaging"""
        if augment:
            return transforms.Compose([
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ElasticTransform(alpha=34, sigma=4) if hasattr(transforms, 'ElasticTransform') else transforms.RandomRotation(0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # Assuming grayscale
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

# Test task-specific transforms
print("Testing task-specific transform pipelines:")

task_transforms = {
    'Classification (train)': TaskSpecificTransforms.classification_transforms(224, augment=True),
    'Classification (val)': TaskSpecificTransforms.classification_transforms(224, augment=False),
    'Segmentation (train)': TaskSpecificTransforms.segmentation_transforms(augment=True),
    'Detection (train)': TaskSpecificTransforms.detection_transforms(augment=True),
}

for name, pipeline in task_transforms.items():
    try:
        result = pipeline(sample_image)
        print(f"  {name:20}: Result shape {result.shape}")
    except Exception as e:
        print(f"  {name:20}: Error - {e}")

print("\n=== Functional Transforms API ===")

# Using functional transforms for more control
print("Testing functional transforms API:")

# Create a transform that applies different operations based on some condition
def conditional_transform(img, apply_rotation=True, apply_flip=True):
    """Apply transforms conditionally using functional API"""
    
    # Convert to tensor if needed
    if isinstance(img, Image.Image):
        img = TF.to_tensor(img)
    
    # Apply rotation conditionally
    if apply_rotation and random.random() > 0.5:
        angle = random.uniform(-30, 30)
        img = TF.rotate(img, angle)
        print(f"    Applied rotation: {angle:.1f} degrees")
    
    # Apply flip conditionally
    if apply_flip and random.random() > 0.5:
        img = TF.hflip(img)
        print(f"    Applied horizontal flip")
    
    # Apply color jitter
    brightness = random.uniform(0.8, 1.2)
    contrast = random.uniform(0.8, 1.2)
    img = TF.adjust_brightness(img, brightness)
    img = TF.adjust_contrast(img, contrast)
    print(f"    Applied brightness: {brightness:.2f}, contrast: {contrast:.2f}")
    
    return img

print("Conditional transform with functional API:")
functional_result = conditional_transform(sample_image, apply_rotation=True, apply_flip=True)
print(f"  Final result shape: {functional_result.shape}")

print("\n=== Transform Composition Strategies ===")

class AdaptiveTransforms:
    """Adaptive transform strategies"""
    
    def __init__(self, transforms_list, probabilities=None):
        self.transforms_list = transforms_list
        self.probabilities = probabilities or [1.0] * len(transforms_list)
    
    def __call__(self, img):
        """Apply transforms with given probabilities"""
        for transform, prob in zip(self.transforms_list, self.probabilities):
            if random.random() < prob:
                img = transform(img)
        return img

class RandomChoice:
    """Randomly choose one transform from a list"""
    
    def __init__(self, transforms_list):
        self.transforms_list = transforms_list
    
    def __call__(self, img):
        transform = random.choice(self.transforms_list)
        return transform(img)

# Test composition strategies
print("Testing transform composition strategies:")

# Adaptive transforms
adaptive_transform = AdaptiveTransforms([
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomHorizontalFlip()
], probabilities=[0.5, 0.7, 0.5])

print("Adaptive transforms:")
adaptive_result = adaptive_transform(sample_image)
print(f"  Result: {'Applied' if adaptive_result else 'Failed'}")

# Random choice
choice_transform = RandomChoice([
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.5)
])

print("Random choice transform:")
choice_result = choice_transform(sample_image)
print(f"  Result: {'Applied' if choice_result else 'Failed'}")

print("\n=== Performance Optimization ===")

import time

def benchmark_transforms(img, transforms_dict, num_iterations=100):
    """Benchmark transform performance"""
    results = {}
    
    for name, transform in transforms_dict.items():
        start_time = time.time()
        
        for _ in range(num_iterations):
            try:
                _ = transform(img)
            except:
                pass
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations * 1000  # ms
        results[name] = avg_time
    
    return results

# Benchmark common transforms
benchmark_transforms_dict = {
    'ToTensor': transforms.ToTensor(),
    'Resize': transforms.Resize((224, 224)),
    'RandomCrop': transforms.RandomCrop(200),
    'ColorJitter': transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    'Normalize': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'FullPipeline': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

print("Benchmarking transforms (100 iterations):")
benchmark_results = benchmark_transforms(sample_image, benchmark_transforms_dict, num_iterations=50)

for name, avg_time in sorted(benchmark_results.items(), key=lambda x: x[1]):
    print(f"  {name:15}: {avg_time:.2f} ms per transform")

print("\n=== Vision Transforms Best Practices ===")

print("Transform Selection Guidelines:")
print("1. Use appropriate transforms for your task")
print("2. Balance augmentation strength with data preservation")
print("3. Consider computational cost vs benefit")
print("4. Test transforms on sample data first")
print("5. Use validation transforms for consistent evaluation")

print("\nPerformance Tips:")
print("1. Compose transforms efficiently")
print("2. Use tensor operations when possible")
print("3. Consider caching preprocessed data")
print("4. Optimize transform order")
print("5. Use appropriate image backends")

print("\nCommon Pitfalls:")
print("1. Applying normalization before other transforms")
print("2. Inconsistent train/validation transforms")
print("3. Over-aggressive augmentation")
print("4. Not handling edge cases in custom transforms")
print("5. Memory leaks in custom transform implementations")

print("\nDebugging Transforms:")
print("1. Visualize transformed images")
print("2. Check tensor shapes and ranges")
print("3. Test with different input types")
print("4. Monitor training metrics")
print("5. Compare with baseline performance")

print("\n=== Vision Transforms Complete ===")

# Memory cleanup
del sample_image
torch.cuda.empty_cache() if torch.cuda.is_available() else None