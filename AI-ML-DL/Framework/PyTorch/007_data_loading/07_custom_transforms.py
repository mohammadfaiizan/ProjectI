#!/usr/bin/env python3
"""PyTorch Custom Transforms - Creating custom transform functions"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps
import cv2
import random
import math
from typing import Any, Callable, List, Tuple, Optional, Union, Dict
import warnings

print("=== Custom Transforms Overview ===")

print("Custom transform topics covered:")
print("1. Transform base classes and interfaces")
print("2. Simple function-based transforms")
print("3. Class-based transforms with parameters")
print("4. Stateful transforms with memory")
print("5. Probabilistic and conditional transforms")
print("6. Geometric and spatial transforms")
print("7. Color and intensity transforms")
print("8. Composition and chaining transforms")

print("\n=== Transform Base Classes and Interfaces ===")

# Understanding the transform interface
print("Basic transform interface requirements:")
print("1. Callable object (function or class with __call__)")
print("2. Consistent input/output types")
print("3. Deterministic or controllable randomness")
print("4. Optional parameters and configuration")
print("5. Proper error handling")

class TransformBase:
    """Base class for custom transforms"""
    
    def __init__(self, **kwargs):
        self.params = kwargs
    
    def __call__(self, sample):
        """Apply transform to sample"""
        raise NotImplementedError("Subclasses must implement __call__")
    
    def __repr__(self):
        """String representation of transform"""
        param_str = ', '.join([f'{k}={v}' for k, v in self.params.items()])
        return f"{self.__class__.__name__}({param_str})"

print("\n=== Simple Function-Based Transforms ===")

def simple_normalize(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """Simple normalization transform"""
    return (tensor - mean) / std

def add_gaussian_noise(tensor: torch.Tensor, noise_std: float = 0.1) -> torch.Tensor:
    """Add Gaussian noise to tensor"""
    noise = torch.randn_like(tensor) * noise_std
    return tensor + noise

def random_channel_dropout(tensor: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    """Randomly set channels to zero"""
    if random.random() < p and tensor.dim() >= 3:
        # Randomly select channel to dropout
        channel_idx = random.randint(0, tensor.shape[0] - 1)
        tensor = tensor.clone()
        tensor[channel_idx] = 0
    return tensor

def tensor_clamp(tensor: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
    """Clamp tensor values to range"""
    return torch.clamp(tensor, min_val, max_val)

# Test simple function transforms
print("Testing simple function-based transforms:")

sample_tensor = torch.randn(3, 32, 32)
print(f"Original tensor: shape={sample_tensor.shape}, mean={sample_tensor.mean():.3f}, std={sample_tensor.std():.3f}")

transforms_to_test = [
    ("Normalize", lambda x: simple_normalize(x, mean=0.0, std=1.0)),
    ("Add Noise", lambda x: add_gaussian_noise(x, noise_std=0.1)),
    ("Channel Dropout", lambda x: random_channel_dropout(x, p=0.5)),
    ("Clamp", lambda x: tensor_clamp(x, min_val=-2.0, max_val=2.0))
]

for name, transform in transforms_to_test:
    result = transform(sample_tensor.clone())
    print(f"  {name:15}: mean={result.mean():.3f}, std={result.std():.3f}, range=[{result.min():.3f}, {result.max():.3f}]")

print("\n=== Class-Based Transforms with Parameters ===")

class CustomNormalize(TransformBase):
    """Custom normalization transform with learnable parameters"""
    
    def __init__(self, mean: Union[float, List[float]] = 0.0, 
                 std: Union[float, List[float]] = 1.0, 
                 adaptive: bool = False):
        super().__init__(mean=mean, std=std, adaptive=adaptive)
        self.mean = mean
        self.std = std
        self.adaptive = adaptive
        
        # For adaptive normalization
        self.running_mean = None
        self.running_std = None
        self.momentum = 0.1
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.adaptive:
            return self._adaptive_normalize(tensor)
        else:
            return self._fixed_normalize(tensor)
    
    def _fixed_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Fixed parameter normalization"""
        mean = torch.tensor(self.mean) if isinstance(self.mean, (list, tuple)) else self.mean
        std = torch.tensor(self.std) if isinstance(self.std, (list, tuple)) else self.std
        
        # Reshape for broadcasting if needed
        if tensor.dim() == 3 and isinstance(mean, torch.Tensor):
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)
        
        return (tensor - mean) / std
    
    def _adaptive_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Adaptive normalization with running statistics"""
        # Calculate current statistics
        if tensor.dim() == 3:  # (C, H, W)
            current_mean = tensor.mean(dim=[1, 2])
            current_std = tensor.std(dim=[1, 2])
        else:
            current_mean = tensor.mean()
            current_std = tensor.std()
        
        # Update running statistics
        if self.running_mean is None:
            self.running_mean = current_mean.clone()
            self.running_std = current_std.clone()
        else:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * current_mean
            self.running_std = (1 - self.momentum) * self.running_std + self.momentum * current_std
        
        # Normalize using running statistics
        if tensor.dim() == 3:
            mean = self.running_mean.view(-1, 1, 1)
            std = self.running_std.view(-1, 1, 1)
        else:
            mean = self.running_mean
            std = self.running_std
        
        return (tensor - mean) / (std + 1e-8)

class RandomColorBalance(TransformBase):
    """Random color balance adjustment"""
    
    def __init__(self, temperature_range: Tuple[float, float] = (0.8, 1.2),
                 tint_range: Tuple[float, float] = (0.9, 1.1)):
        super().__init__(temperature_range=temperature_range, tint_range=tint_range)
        self.temperature_range = temperature_range
        self.tint_range = tint_range
    
    def __call__(self, image: Union[Image.Image, torch.Tensor]) -> Union[Image.Image, torch.Tensor]:
        # Convert tensor to PIL if needed
        was_tensor = isinstance(image, torch.Tensor)
        if was_tensor:
            image = TF.to_pil_image(image)
        
        # Random temperature and tint adjustments
        temperature = random.uniform(*self.temperature_range)
        tint = random.uniform(*self.tint_range)
        
        # Apply color balance (simplified version)
        enhancer_color = ImageEnhance.Color(image)
        image = enhancer_color.enhance(temperature)
        
        # Convert back to tensor if needed
        if was_tensor:
            image = TF.to_tensor(image)
        
        return image

class AdaptiveContrast(TransformBase):
    """Adaptive contrast enhancement based on local statistics"""
    
    def __init__(self, window_size: int = 64, clip_limit: float = 2.0):
        super().__init__(window_size=window_size, clip_limit=clip_limit)
        self.window_size = window_size
        self.clip_limit = clip_limit
    
    def __call__(self, image: Union[Image.Image, torch.Tensor]) -> Union[Image.Image, torch.Tensor]:
        was_tensor = isinstance(image, torch.Tensor)
        
        if was_tensor:
            # Convert tensor to numpy for processing
            if image.dim() == 3:
                img_np = image.permute(1, 2, 0).numpy()
            else:
                img_np = image.numpy()
            
            # Convert to uint8 for CLAHE
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = np.array(image)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if len(img_np.shape) == 3:
            # Color image - apply to each channel
            enhanced = np.zeros_like(img_np)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, 
                                   tileGridSize=(self.window_size//8, self.window_size//8))
            
            for i in range(3):
                enhanced[:, :, i] = clahe.apply(img_np[:, :, i])
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, 
                                   tileGridSize=(self.window_size//8, self.window_size//8))
            enhanced = clahe.apply(img_np)
        
        # Convert back to original format
        if was_tensor:
            enhanced = enhanced.astype(np.float32) / 255.0
            if len(enhanced.shape) == 3:
                enhanced = torch.from_numpy(enhanced).permute(2, 0, 1)
            else:
                enhanced = torch.from_numpy(enhanced)
            return enhanced
        else:
            return Image.fromarray(enhanced)

# Test class-based transforms
print("Testing class-based transforms:")

# Test adaptive normalization
adaptive_norm = CustomNormalize(adaptive=True)
test_tensors = [torch.randn(3, 16, 16) * 2 + 1 for _ in range(3)]

print("Adaptive normalization:")
for i, tensor in enumerate(test_tensors):
    normalized = adaptive_norm(tensor)
    print(f"  Iteration {i+1}: input_mean={tensor.mean():.3f}, output_mean={normalized.mean():.3f}")

# Test color balance
color_balance = RandomColorBalance(temperature_range=(0.5, 1.5))
test_image = Image.new('RGB', (64, 64), color=(128, 100, 150))

balanced = color_balance(test_image)
print(f"\nColor balance: applied to {test_image.size} image")

# Test adaptive contrast
try:
    adaptive_contrast = AdaptiveContrast(window_size=32, clip_limit=2.0)
    contrast_enhanced = adaptive_contrast(test_image)
    print(f"Adaptive contrast: applied to {test_image.size} image")
except Exception as e:
    print(f"Adaptive contrast: Error - {e}")

print("\n=== Stateful Transforms with Memory ===")

class RunningStatistics(TransformBase):
    """Transform that maintains running statistics"""
    
    def __init__(self, momentum: float = 0.1):
        super().__init__(momentum=momentum)
        self.momentum = momentum
        self.count = 0
        self.running_mean = None
        self.running_var = None
    
    def __call__(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Update running statistics
        batch_mean = tensor.mean()
        batch_var = tensor.var()
        
        if self.running_mean is None:
            self.running_mean = batch_mean.clone()
            self.running_var = batch_var.clone()
        else:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        
        self.count += 1
        
        return {
            'normalized': (tensor - self.running_mean) / torch.sqrt(self.running_var + 1e-8),
            'running_mean': self.running_mean,
            'running_var': self.running_var,
            'count': self.count
        }

class TemporalAugmentation(TransformBase):
    """Augmentation that depends on previous samples"""
    
    def __init__(self, memory_size: int = 5, mix_probability: float = 0.3):
        super().__init__(memory_size=memory_size, mix_probability=mix_probability)
        self.memory_size = memory_size
        self.mix_probability = mix_probability
        self.memory = []
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        result = tensor.clone()
        
        # Mix with previous samples if available
        if len(self.memory) > 0 and random.random() < self.mix_probability:
            # Select random sample from memory
            memory_sample = random.choice(self.memory)
            mix_ratio = random.uniform(0.1, 0.5)
            result = (1 - mix_ratio) * result + mix_ratio * memory_sample
        
        # Add current sample to memory
        self.memory.append(tensor.clone())
        
        # Maintain memory size
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        
        return result

# Test stateful transforms
print("Testing stateful transforms:")

# Test running statistics
running_stats = RunningStatistics(momentum=0.2)
print("Running statistics:")

for i in range(5):
    test_tensor = torch.randn(100) * (i + 1) + i  # Different distributions
    result = running_stats(test_tensor)
    print(f"  Step {i+1}: running_mean={result['running_mean']:.3f}, "
          f"running_var={result['running_var']:.3f}")

# Test temporal augmentation
temporal_aug = TemporalAugmentation(memory_size=3, mix_probability=0.5)
print("\nTemporal augmentation:")

for i in range(5):
    test_tensor = torch.randn(3, 8, 8)
    augmented = temporal_aug(test_tensor)
    print(f"  Step {i+1}: memory_size={len(temporal_aug.memory)}, "
          f"difference_norm={torch.norm(augmented - test_tensor):.3f}")

print("\n=== Probabilistic and Conditional Transforms ===")

class ProbabilisticTransform(TransformBase):
    """Transform that applies different operations with given probabilities"""
    
    def __init__(self, transform_dict: Dict[str, Tuple[Callable, float]]):
        super().__init__(transform_dict=transform_dict)
        self.transform_dict = transform_dict
        
        # Normalize probabilities
        total_prob = sum(prob for _, prob in transform_dict.values())
        if total_prob > 1.0:
            warnings.warn(f"Total probability {total_prob} > 1.0, normalizing")
            for name, (transform, prob) in transform_dict.items():
                self.transform_dict[name] = (transform, prob / total_prob)
    
    def __call__(self, data: Any) -> Any:
        for name, (transform, probability) in self.transform_dict.items():
            if random.random() < probability:
                data = transform(data)
        return data

class ConditionalTransform(TransformBase):
    """Transform that applies operations based on data properties"""
    
    def __init__(self, brightness_threshold: float = 0.5):
        super().__init__(brightness_threshold=brightness_threshold)
        self.brightness_threshold = brightness_threshold
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Calculate brightness
        brightness = tensor.mean()
        
        if brightness < self.brightness_threshold:
            # Dark image - apply brightness enhancement
            return torch.clamp(tensor * 1.2 + 0.1, 0, 1)
        else:
            # Bright image - apply contrast enhancement
            return torch.clamp((tensor - 0.5) * 1.1 + 0.5, 0, 1)

class AdaptiveTransform(TransformBase):
    """Transform that adapts based on data statistics"""
    
    def __init__(self):
        super().__init__()
        self.global_stats = {'mean': 0.0, 'std': 1.0, 'count': 0}
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Update global statistics
        current_mean = tensor.mean().item()
        current_std = tensor.std().item()
        
        if self.global_stats['count'] == 0:
            self.global_stats['mean'] = current_mean
            self.global_stats['std'] = current_std
        else:
            # Exponential moving average
            alpha = 0.1
            self.global_stats['mean'] = (1 - alpha) * self.global_stats['mean'] + alpha * current_mean
            self.global_stats['std'] = (1 - alpha) * self.global_stats['std'] + alpha * current_std
        
        self.global_stats['count'] += 1
        
        # Adaptive normalization
        deviation_factor = abs(current_mean - self.global_stats['mean']) / (self.global_stats['std'] + 1e-8)
        
        if deviation_factor > 2.0:
            # Outlier - apply strong normalization
            return (tensor - current_mean) / (current_std + 1e-8)
        else:
            # Normal - apply mild normalization
            return (tensor - self.global_stats['mean']) / (self.global_stats['std'] + 1e-8)

# Test probabilistic and conditional transforms
print("Testing probabilistic and conditional transforms:")

# Probabilistic transform
prob_transforms = {
    'noise': (lambda x: x + torch.randn_like(x) * 0.1, 0.3),
    'blur': (lambda x: F.avg_pool2d(x.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0) if x.dim() >= 2 else x, 0.2),
    'brightness': (lambda x: torch.clamp(x * 1.2, 0, 1), 0.4)
}

prob_transform = ProbabilisticTransform(prob_transforms)

print("Probabilistic transform test:")
for i in range(5):
    test_tensor = torch.rand(3, 16, 16)
    result = prob_transform(test_tensor)
    diff = torch.norm(result - test_tensor)
    print(f"  Trial {i+1}: difference_norm={diff:.3f}")

# Conditional transform
conditional_transform = ConditionalTransform(brightness_threshold=0.5)

print("\nConditional transform test:")
test_cases = [
    ("Dark image", torch.rand(3, 8, 8) * 0.3),
    ("Bright image", torch.rand(3, 8, 8) * 0.7 + 0.3)
]

for name, test_tensor in test_cases:
    result = conditional_transform(test_tensor)
    print(f"  {name}: input_brightness={test_tensor.mean():.3f}, "
          f"output_brightness={result.mean():.3f}")

print("\n=== Geometric and Spatial Transforms ===")

class CustomRotation(TransformBase):
    """Custom rotation with different interpolation methods"""
    
    def __init__(self, angles: List[float] = [0, 90, 180, 270], 
                 interpolation: str = 'bilinear'):
        super().__init__(angles=angles, interpolation=interpolation)
        self.angles = angles
        self.interpolation = interpolation
    
    def __call__(self, image: Union[Image.Image, torch.Tensor]) -> Union[Image.Image, torch.Tensor]:
        angle = random.choice(self.angles)
        
        if isinstance(image, torch.Tensor):
            # Use functional API for tensor
            return TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
        else:
            # Use PIL for image
            return image.rotate(angle, expand=True)

class ElasticDeformation(TransformBase):
    """Elastic deformation transform"""
    
    def __init__(self, alpha: float = 34, sigma: float = 4):
        super().__init__(alpha=alpha, sigma=sigma)
        self.alpha = alpha
        self.sigma = sigma
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() != 3:
            return image
        
        _, height, width = image.shape
        
        # Generate displacement fields
        dx = torch.randn(height, width) * self.alpha
        dy = torch.randn(height, width) * self.alpha
        
        # Apply Gaussian smoothing
        dx = F.conv2d(dx.unsqueeze(0).unsqueeze(0), 
                     self._gaussian_kernel(self.sigma).unsqueeze(0).unsqueeze(0),
                     padding=self.sigma//2).squeeze()
        dy = F.conv2d(dy.unsqueeze(0).unsqueeze(0),
                     self._gaussian_kernel(self.sigma).unsqueeze(0).unsqueeze(0),
                     padding=self.sigma//2).squeeze()
        
        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        
        # Apply displacement
        new_x = torch.clamp(x_coords.float() + dx, 0, width - 1)
        new_y = torch.clamp(y_coords.float() + dy, 0, height - 1)
        
        # Sample using grid_sample
        grid = torch.stack([new_x, new_y], dim=-1).unsqueeze(0)
        grid[..., 0] = grid[..., 0] / (width - 1) * 2 - 1  # Normalize to [-1, 1]
        grid[..., 1] = grid[..., 1] / (height - 1) * 2 - 1
        
        deformed = F.grid_sample(image.unsqueeze(0), grid, mode='bilinear', padding_mode='border')
        return deformed.squeeze(0)
    
    def _gaussian_kernel(self, sigma: float, kernel_size: int = None) -> torch.Tensor:
        """Create Gaussian kernel"""
        if kernel_size is None:
            kernel_size = int(2 * sigma + 1)
        
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
        kernel_1d /= kernel_1d.sum()
        
        return kernel_1d.outer(kernel_1d)

class PolarTransform(TransformBase):
    """Convert between Cartesian and polar coordinates"""
    
    def __init__(self, mode: str = 'to_polar'):
        super().__init__(mode=mode)
        self.mode = mode  # 'to_polar' or 'from_polar'
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() != 3:
            return image
        
        C, H, W = image.shape
        center_x, center_y = W // 2, H // 2
        
        if self.mode == 'to_polar':
            return self._cartesian_to_polar(image, center_x, center_y)
        else:
            return self._polar_to_cartesian(image, center_x, center_y)
    
    def _cartesian_to_polar(self, image: torch.Tensor, cx: int, cy: int) -> torch.Tensor:
        """Convert Cartesian to polar coordinates"""
        C, H, W = image.shape
        max_radius = min(cx, cy, W - cx, H - cy)
        
        # Create polar coordinate grid
        angles = torch.linspace(0, 2 * math.pi, W)
        radii = torch.linspace(0, max_radius, H)
        
        polar_image = torch.zeros_like(image)
        
        for i, radius in enumerate(radii):
            for j, angle in enumerate(angles):
                x = int(cx + radius * torch.cos(angle))
                y = int(cy + radius * torch.sin(angle))
                
                if 0 <= x < W and 0 <= y < H:
                    polar_image[:, i, j] = image[:, y, x]
        
        return polar_image
    
    def _polar_to_cartesian(self, polar_image: torch.Tensor, cx: int, cy: int) -> torch.Tensor:
        """Convert polar to Cartesian coordinates"""
        C, H, W = polar_image.shape
        max_radius = H
        
        cartesian_image = torch.zeros_like(polar_image)
        
        for y in range(H):
            for x in range(W):
                dx, dy = x - cx, y - cy
                radius = torch.sqrt(torch.tensor(dx**2 + dy**2, dtype=torch.float32))
                angle = torch.atan2(torch.tensor(dy, dtype=torch.float32), 
                                  torch.tensor(dx, dtype=torch.float32))
                
                if angle < 0:
                    angle += 2 * math.pi
                
                # Map to polar coordinates
                r_idx = int(radius * (H - 1) / max_radius)
                a_idx = int(angle * (W - 1) / (2 * math.pi))
                
                if 0 <= r_idx < H and 0 <= a_idx < W:
                    cartesian_image[:, y, x] = polar_image[:, r_idx, a_idx]
        
        return cartesian_image

# Test geometric transforms
print("Testing geometric and spatial transforms:")

test_tensor = torch.rand(3, 32, 32)

# Test custom rotation
custom_rotation = CustomRotation(angles=[45, 90, 135], interpolation='bilinear')
rotated = custom_rotation(test_tensor)
print(f"Custom rotation: input {test_tensor.shape} -> output {rotated.shape}")

# Test elastic deformation
try:
    elastic_transform = ElasticDeformation(alpha=10, sigma=2)
    deformed = elastic_transform(test_tensor)
    print(f"Elastic deformation: applied, output shape {deformed.shape}")
except Exception as e:
    print(f"Elastic deformation: Error - {e}")

# Test polar transform
polar_transform = PolarTransform(mode='to_polar')
polar_result = polar_transform(test_tensor)
print(f"Polar transform: input {test_tensor.shape} -> output {polar_result.shape}")

print("\n=== Transform Composition and Chaining ===")

class CompositeTransform(TransformBase):
    """Compose multiple transforms with optional branching"""
    
    def __init__(self, transforms: List[Callable], mode: str = 'sequential'):
        super().__init__(transforms=transforms, mode=mode)
        self.transforms = transforms
        self.mode = mode  # 'sequential', 'parallel', 'random_choice'
    
    def __call__(self, data: Any) -> Any:
        if self.mode == 'sequential':
            return self._sequential_apply(data)
        elif self.mode == 'parallel':
            return self._parallel_apply(data)
        elif self.mode == 'random_choice':
            return self._random_choice_apply(data)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _sequential_apply(self, data: Any) -> Any:
        """Apply transforms sequentially"""
        for transform in self.transforms:
            data = transform(data)
        return data
    
    def _parallel_apply(self, data: Any) -> List[Any]:
        """Apply transforms in parallel, return list of results"""
        results = []
        for transform in self.transforms:
            results.append(transform(data.clone() if hasattr(data, 'clone') else data))
        return results
    
    def _random_choice_apply(self, data: Any) -> Any:
        """Randomly choose one transform to apply"""
        transform = random.choice(self.transforms)
        return transform(data)

class ConditionalComposer(TransformBase):
    """Conditional transform composition based on data properties"""
    
    def __init__(self, condition_fn: Callable, true_transforms: List[Callable], 
                 false_transforms: List[Callable]):
        super().__init__()
        self.condition_fn = condition_fn
        self.true_transforms = true_transforms
        self.false_transforms = false_transforms
    
    def __call__(self, data: Any) -> Any:
        if self.condition_fn(data):
            transforms = self.true_transforms
        else:
            transforms = self.false_transforms
        
        for transform in transforms:
            data = transform(data)
        
        return data

class TransformChain(TransformBase):
    """Advanced transform chaining with error recovery"""
    
    def __init__(self, transforms: List[Tuple[Callable, Dict]], 
                 error_strategy: str = 'skip'):
        super().__init__(transforms=transforms, error_strategy=error_strategy)
        self.transforms = transforms
        self.error_strategy = error_strategy  # 'skip', 'stop', 'default'
    
    def __call__(self, data: Any) -> Dict[str, Any]:
        results = {'data': data, 'applied_transforms': [], 'errors': []}
        
        for i, (transform, config) in enumerate(self.transforms):
            try:
                # Apply transform with configuration
                if config:
                    # If transform is a class, instantiate with config
                    if isinstance(transform, type):
                        transform_instance = transform(**config)
                        data = transform_instance(data)
                    else:
                        # Function with partial application
                        data = transform(data, **config)
                else:
                    data = transform(data)
                
                results['applied_transforms'].append(f"Transform_{i}")
                
            except Exception as e:
                error_info = {'transform_index': i, 'error': str(e)}
                results['errors'].append(error_info)
                
                if self.error_strategy == 'stop':
                    break
                elif self.error_strategy == 'skip':
                    continue
                elif self.error_strategy == 'default':
                    # Apply default transform (identity)
                    pass
        
        results['data'] = data
        return results

# Test transform composition
print("Testing transform composition and chaining:")

# Test composite transform
simple_transforms = [
    lambda x: x + 0.1,
    lambda x: x * 1.1,
    lambda x: torch.clamp(x, 0, 1)
]

composite_seq = CompositeTransform(simple_transforms, mode='sequential')
composite_choice = CompositeTransform(simple_transforms, mode='random_choice')

test_data = torch.rand(3, 8, 8)
seq_result = composite_seq(test_data)
choice_result = composite_choice(test_data)

print(f"Sequential composition: input_mean={test_data.mean():.3f}, output_mean={seq_result.mean():.3f}")
print(f"Random choice composition: output_mean={choice_result.mean():.3f}")

# Test conditional composer
def is_bright(tensor):
    return tensor.mean() > 0.5

conditional_composer = ConditionalComposer(
    condition_fn=is_bright,
    true_transforms=[lambda x: x * 0.8],  # Darken bright images
    false_transforms=[lambda x: x * 1.2]  # Brighten dark images
)

bright_data = torch.rand(3, 8, 8) * 0.8 + 0.2
dark_data = torch.rand(3, 8, 8) * 0.3

bright_result = conditional_composer(bright_data)
dark_result = conditional_composer(dark_data)

print(f"Conditional composer - Bright: {bright_data.mean():.3f} -> {bright_result.mean():.3f}")
print(f"Conditional composer - Dark: {dark_data.mean():.3f} -> {dark_result.mean():.3f}")

# Test transform chain
transform_configs = [
    (lambda x: x + torch.randn_like(x) * 0.05, {}),
    (lambda x: torch.clamp(x, 0, 1), {}),
    (lambda x: x / x.max(), {})  # This might cause error if max is 0
]

transform_chain = TransformChain(transform_configs, error_strategy='skip')
chain_result = transform_chain(test_data)

print(f"Transform chain: applied {len(chain_result['applied_transforms'])} transforms, "
      f"{len(chain_result['errors'])} errors")

print("\n=== Custom Transform Best Practices ===")

print("Design Principles:")
print("1. Keep transforms pure and stateless when possible")
print("2. Handle different input types gracefully") 
print("3. Provide meaningful error messages")
print("4. Document expected input/output formats")
print("5. Make transforms composable and chainable")

print("\nImplementation Tips:")
print("1. Use type hints for better code clarity")
print("2. Implement __repr__ for debugging")
print("3. Handle edge cases (empty tensors, invalid ranges)")
print("4. Consider memory efficiency for large data")
print("5. Make random behavior controllable via seeds")

print("\nTesting and Validation:")
print("1. Test with different tensor shapes and types")
print("2. Verify mathematical correctness of operations")
print("3. Check for memory leaks with large datasets")
print("4. Validate that transforms are invertible when needed")
print("5. Benchmark performance for real-world usage")

print("\nCommon Pitfalls:")
print("1. Modifying input data in-place unexpectedly")
print("2. Not handling batch dimensions correctly")
print("3. Ignoring device placement (CPU vs GPU)")
print("4. Poor error handling causing training crashes")
print("5. Not considering numerical stability")

print("\n=== Custom Transforms Complete ===")

# Memory cleanup
torch.cuda.empty_cache() if torch.cuda.is_available() else None