#!/usr/bin/env python3
"""PyTorch Data Augmentation Pipeline - Complete augmentation pipelines"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import random
import math
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2

print("=== Data Augmentation Pipeline Overview ===")

print("Augmentation pipeline topics:")
print("1. Vision augmentation pipelines")
print("2. Audio augmentation strategies")
print("3. Text augmentation pipelines")
print("4. Multi-modal augmentation")
print("5. Task-specific augmentation")
print("6. Adaptive augmentation")
print("7. Pipeline composition and management")
print("8. Performance and efficiency")

print("\n=== Vision Augmentation Pipelines ===")

class VisionAugmentationPipeline:
    """Comprehensive vision augmentation pipeline"""
    
    def __init__(self, policy: str = 'light', input_size: int = 224):
        self.policy = policy
        self.input_size = input_size
        self.pipelines = self._create_pipelines()
    
    def _create_pipelines(self) -> Dict[str, transforms.Compose]:
        """Create different augmentation policies"""
        
        # Light augmentation (for stable training)
        light_transforms = transforms.Compose([
            transforms.RandomResizedCrop(self.input_size, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Medium augmentation (balanced approach)
        medium_transforms = transforms.Compose([
            transforms.RandomResizedCrop(self.input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Heavy augmentation (aggressive for data-scarce scenarios)
        heavy_transforms = transforms.Compose([
            transforms.RandomResizedCrop(self.input_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Validation transforms (no augmentation)
        val_transforms = transforms.Compose([
            transforms.Resize(int(self.input_size * 1.14)),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return {
            'light': light_transforms,
            'medium': medium_transforms,
            'heavy': heavy_transforms,
            'validation': val_transforms
        }
    
    def get_transform(self, mode: str = None) -> transforms.Compose:
        """Get transform pipeline for specified mode"""
        mode = mode or self.policy
        return self.pipelines.get(mode, self.pipelines['medium'])
    
    def __call__(self, image: Image.Image, mode: str = None) -> torch.Tensor:
        """Apply augmentation pipeline"""
        transform = self.get_transform(mode)
        return transform(image)

# Create sample image for testing
def create_test_image(size=(224, 224)):
    """Create a test image with patterns"""
    img = Image.new('RGB', size, color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    
    # Add some geometric shapes
    draw.rectangle([50, 50, 100, 100], fill=(255, 0, 0))
    draw.ellipse([150, 50, 200, 100], fill=(0, 255, 0))
    draw.polygon([(100, 150), (125, 120), (150, 150)], fill=(0, 0, 255))
    
    return img

# Test vision augmentation pipelines
print("Testing vision augmentation pipelines:")

test_image = create_test_image()
vision_pipeline = VisionAugmentationPipeline(input_size=224)

policies = ['light', 'medium', 'heavy', 'validation']
for policy in policies:
    try:
        augmented = vision_pipeline(test_image, mode=policy)
        print(f"  {policy:10} policy: Output shape {augmented.shape}, "
              f"range [{augmented.min():.3f}, {augmented.max():.3f}]")
    except Exception as e:
        print(f"  {policy:10} policy: Error - {e}")

print("\n=== Advanced Vision Augmentation with Albumentations ===")

class AlbumentationsAugmentation:
    """Advanced augmentation using Albumentations library"""
    
    def __init__(self, policy: str = 'medium'):
        self.policy = policy
        self.transforms = self._create_albumentations_pipelines()
    
    def _create_albumentations_pipelines(self) -> Dict[str, A.Compose]:
        """Create Albumentations augmentation pipelines"""
        
        # Light augmentation
        light_aug = A.Compose([
            A.RandomResizedCrop(224, 224, scale=(0.9, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Medium augmentation
        medium_aug = A.Compose([
            A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.MotionBlur(blur_limit=3, p=0.3),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Heavy augmentation
        heavy_aug = A.Compose([
            A.RandomResizedCrop(224, 224, scale=(0.6, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.7),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.7),
            A.OneOf([
                A.GaussianNoise(var_limit=(10, 50), p=0.3),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=0.3),
            ], p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.3),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.3),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            ], p=0.5),
            A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, fill_value=0, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Validation transforms
        val_aug = A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        return {
            'light': light_aug,
            'medium': medium_aug,
            'heavy': heavy_aug,
            'validation': val_aug
        }
    
    def __call__(self, image: Union[Image.Image, np.ndarray], mode: str = None) -> torch.Tensor:
        """Apply Albumentations augmentation"""
        mode = mode or self.policy
        
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        transform = self.transforms[mode]
        augmented = transform(image=image)
        return augmented['image']

# Test Albumentations augmentation (if available)
print("Testing Albumentations augmentation:")

try:
    albu_pipeline = AlbumentationsAugmentation()
    test_image_np = np.array(test_image)
    
    for policy in ['light', 'medium', 'heavy']:
        try:
            augmented = albu_pipeline(test_image_np, mode=policy)
            print(f"  {policy:6} policy: Output shape {augmented.shape}, "
                  f"range [{augmented.min():.3f}, {augmented.max():.3f}]")
        except Exception as e:
            print(f"  {policy:6} policy: Error - {e}")

except ImportError:
    print("  Albumentations not available - skipping advanced augmentation")

print("\n=== Audio Augmentation Pipeline ===")

class AudioAugmentationPipeline:
    """Audio augmentation techniques"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def time_shift(self, audio: torch.Tensor, shift_limit: float = 0.2) -> torch.Tensor:
        """Time shift augmentation"""
        shift_amt = int(random.random() * shift_limit * len(audio))
        return torch.roll(audio, shift_amt)
    
    def speed_change(self, audio: torch.Tensor, speed_factor: float = None) -> torch.Tensor:
        """Change audio speed (time stretching)"""
        if speed_factor is None:
            speed_factor = random.uniform(0.8, 1.2)
        
        # Simple resampling simulation
        indices = torch.arange(0, len(audio), speed_factor)
        indices = indices[indices < len(audio)].long()
        return audio[indices]
    
    def add_noise(self, audio: torch.Tensor, noise_factor: float = 0.01) -> torch.Tensor:
        """Add random noise"""
        noise = torch.randn_like(audio) * noise_factor
        return audio + noise
    
    def pitch_shift(self, audio: torch.Tensor, n_steps: int = None) -> torch.Tensor:
        """Pitch shifting (simplified)"""
        if n_steps is None:
            n_steps = random.randint(-2, 2)
        
        # Simple pitch shift simulation using resampling
        factor = 2 ** (n_steps / 12.0)
        return self.speed_change(audio, factor)
    
    def time_masking(self, audio: torch.Tensor, mask_time: int = None) -> torch.Tensor:
        """Time masking (SpecAugment style)"""
        if mask_time is None:
            mask_time = random.randint(0, min(50, len(audio) // 10))
        
        start = random.randint(0, max(0, len(audio) - mask_time))
        audio_masked = audio.clone()
        audio_masked[start:start + mask_time] = 0
        return audio_masked
    
    def reverb(self, audio: torch.Tensor, room_size: float = 0.3) -> torch.Tensor:
        """Simple reverb effect"""
        # Simple echo-based reverb simulation
        delay_samples = int(0.05 * self.sample_rate)  # 50ms delay
        echo = torch.zeros_like(audio)
        
        if delay_samples < len(audio):
            echo[delay_samples:] = audio[:-delay_samples] * room_size
            return audio + echo
        return audio
    
    def create_pipeline(self, augmentation_list: List[str], 
                       probabilities: List[float] = None) -> Callable:
        """Create augmentation pipeline"""
        
        augmentation_map = {
            'time_shift': self.time_shift,
            'speed_change': self.speed_change,
            'add_noise': self.add_noise,
            'pitch_shift': self.pitch_shift,
            'time_masking': self.time_masking,
            'reverb': self.reverb
        }
        
        if probabilities is None:
            probabilities = [0.5] * len(augmentation_list)
        
        def pipeline(audio: torch.Tensor) -> torch.Tensor:
            for aug_name, prob in zip(augmentation_list, probabilities):
                if random.random() < prob and aug_name in augmentation_map:
                    audio = augmentation_map[aug_name](audio)
            return audio
        
        return pipeline

# Test audio augmentation
print("Testing audio augmentation pipeline:")

# Create sample audio signal
sample_rate = 16000
duration = 2.0  # seconds
t = torch.linspace(0, duration, int(sample_rate * duration))
sample_audio = torch.sin(2 * math.pi * 440 * t) + 0.5 * torch.sin(2 * math.pi * 880 * t)  # A4 + A5

audio_pipeline = AudioAugmentationPipeline(sample_rate=sample_rate)

# Test individual augmentations
augmentations = ['time_shift', 'speed_change', 'add_noise', 'pitch_shift', 'time_masking', 'reverb']

print(f"Original audio: length={len(sample_audio)}, RMS={sample_audio.pow(2).mean().sqrt():.4f}")

for aug_name in augmentations:
    try:
        aug_method = getattr(audio_pipeline, aug_name)
        augmented = aug_method(sample_audio.clone())
        rms = augmented.pow(2).mean().sqrt()
        print(f"  {aug_name:12}: length={len(augmented)}, RMS={rms:.4f}")
    except Exception as e:
        print(f"  {aug_name:12}: Error - {e}")

# Test pipeline
pipeline = audio_pipeline.create_pipeline(
    ['add_noise', 'time_shift', 'speed_change'], 
    probabilities=[0.7, 0.5, 0.3]
)
pipeline_result = pipeline(sample_audio.clone())
print(f"  Pipeline result: length={len(pipeline_result)}, "
      f"RMS={pipeline_result.pow(2).mean().sqrt():.4f}")

print("\n=== Text Augmentation Pipeline ===")

class TextAugmentationPipeline:
    """Comprehensive text augmentation pipeline"""
    
    def __init__(self):
        self.word_replacements = {
            'good': ['great', 'excellent', 'wonderful', 'amazing'],
            'bad': ['terrible', 'awful', 'horrible', 'dreadful'],
            'big': ['large', 'huge', 'enormous', 'massive'],
            'small': ['tiny', 'little', 'miniature', 'petite'],
            'fast': ['quick', 'rapid', 'swift', 'speedy'],
            'slow': ['sluggish', 'gradual', 'leisurely', 'unhurried']
        }
    
    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """Replace words with synonyms"""
        words = text.split()
        new_words = words.copy()
        
        for _ in range(n):
            if not words:
                break
            
            random_word_idx = random.randint(0, len(words) - 1)
            word = words[random_word_idx].lower()
            
            if word in self.word_replacements:
                new_words[random_word_idx] = random.choice(self.word_replacements[word])
        
        return ' '.join(new_words)
    
    def random_insertion(self, text: str, n: int = 1) -> str:
        """Randomly insert words"""
        words = text.split()
        
        for _ in range(n):
            if not words:
                break
            
            # Insert a random word from the sentence
            random_word = random.choice(words)
            random_idx = random.randint(0, len(words))
            words.insert(random_idx, random_word)
        
        return ' '.join(words)
    
    def random_swap(self, text: str, n: int = 1) -> str:
        """Randomly swap words"""
        words = text.split()
        
        for _ in range(n):
            if len(words) < 2:
                break
            
            idx1 = random.randint(0, len(words) - 1)
            idx2 = random.randint(0, len(words) - 1)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete words"""
        words = text.split()
        
        if len(words) == 1:
            return text
        
        new_words = [word for word in words if random.random() > p]
        return ' '.join(new_words) if new_words else text
    
    def sentence_shuffle(self, text: str) -> str:
        """Shuffle sentences within text"""
        sentences = text.split('. ')
        random.shuffle(sentences)
        return '. '.join(sentences)
    
    def create_pipeline(self, augmentation_config: Dict[str, Dict]) -> Callable:
        """Create text augmentation pipeline"""
        
        def pipeline(text: str) -> str:
            for aug_name, config in augmentation_config.items():
                prob = config.get('probability', 1.0)
                params = config.get('params', {})
                
                if random.random() < prob:
                    if aug_name == 'synonym_replacement':
                        text = self.synonym_replacement(text, **params)
                    elif aug_name == 'random_insertion':
                        text = self.random_insertion(text, **params)
                    elif aug_name == 'random_swap':
                        text = self.random_swap(text, **params)
                    elif aug_name == 'random_deletion':
                        text = self.random_deletion(text, **params)
                    elif aug_name == 'sentence_shuffle':
                        text = self.sentence_shuffle(text)
            
            return text
        
        return pipeline

# Test text augmentation pipeline
print("Testing text augmentation pipeline:")

text_pipeline = TextAugmentationPipeline()
original_text = "The quick brown fox jumps over the lazy dog. This is a good example sentence."

# Test individual augmentations
augmentations = [
    ('synonym_replacement', {'n': 2}),
    ('random_insertion', {'n': 1}),
    ('random_swap', {'n': 2}),
    ('random_deletion', {'p': 0.2}),
    ('sentence_shuffle', {})
]

print(f"Original: '{original_text}'")

for aug_name, params in augmentations:
    try:
        method = getattr(text_pipeline, aug_name)
        augmented = method(original_text, **params)
        print(f"  {aug_name:18}: '{augmented}'")
    except Exception as e:
        print(f"  {aug_name:18}: Error - {e}")

# Test pipeline
augmentation_config = {
    'synonym_replacement': {'probability': 0.7, 'params': {'n': 1}},
    'random_swap': {'probability': 0.5, 'params': {'n': 1}},
    'random_deletion': {'probability': 0.3, 'params': {'p': 0.1}}
}

pipeline = text_pipeline.create_pipeline(augmentation_config)
pipeline_result = pipeline(original_text)
print(f"  Pipeline result: '{pipeline_result}'")

print("\n=== Multi-Modal Augmentation ===")

class MultiModalAugmentation:
    """Augmentation for multi-modal data"""
    
    def __init__(self):
        self.vision_aug = VisionAugmentationPipeline(policy='medium')
        self.text_aug = TextAugmentationPipeline()
    
    def synchronized_augmentation(self, image: Image.Image, text: str, 
                                sync_probability: float = 0.5) -> Tuple[torch.Tensor, str]:
        """Apply synchronized augmentation to image-text pairs"""
        
        # Some augmentations should be synchronized
        if random.random() < sync_probability:
            # Apply consistent intensity augmentations
            if random.random() < 0.5:
                # Heavy augmentation for both
                augmented_image = self.vision_aug(image, mode='heavy')
                augmented_text = self.text_aug.synonym_replacement(text, n=2)
            else:
                # Light augmentation for both
                augmented_image = self.vision_aug(image, mode='light')
                augmented_text = self.text_aug.random_swap(text, n=1)
        else:
            # Independent augmentation
            augmented_image = self.vision_aug(image, mode='medium')
            
            # Apply text augmentation
            text_config = {
                'synonym_replacement': {'probability': 0.5, 'params': {'n': 1}},
                'random_insertion': {'probability': 0.3, 'params': {'n': 1}}
            }
            text_pipeline = self.text_aug.create_pipeline(text_config)
            augmented_text = text_pipeline(text)
        
        return augmented_image, augmented_text
    
    def cross_modal_augmentation(self, image: Image.Image, text: str) -> Tuple[torch.Tensor, str]:
        """Apply cross-modal augmentation techniques"""
        
        # Image-guided text augmentation
        # (In practice, this could use image features to guide text changes)
        if random.random() < 0.3:
            # If image is heavily augmented, apply more text augmentation
            augmented_image = self.vision_aug(image, mode='heavy')
            augmented_text = self.text_aug.random_insertion(text, n=2)
        else:
            augmented_image = self.vision_aug(image, mode='medium')
            augmented_text = self.text_aug.synonym_replacement(text, n=1)
        
        return augmented_image, augmented_text

# Test multi-modal augmentation
print("Testing multi-modal augmentation:")

multimodal_aug = MultiModalAugmentation()
sample_text = "A beautiful landscape with mountains and trees."

try:
    # Synchronized augmentation
    sync_image, sync_text = multimodal_aug.synchronized_augmentation(test_image, sample_text)
    print(f"  Synchronized - Image shape: {sync_image.shape}")
    print(f"  Synchronized - Text: '{sync_text}'")
    
    # Cross-modal augmentation
    cross_image, cross_text = multimodal_aug.cross_modal_augmentation(test_image, sample_text)
    print(f"  Cross-modal - Image shape: {cross_image.shape}")
    print(f"  Cross-modal - Text: '{cross_text}'")

except Exception as e:
    print(f"  Multi-modal augmentation error: {e}")

print("\n=== Adaptive Augmentation ===")

class AdaptiveAugmentation:
    """Adaptive augmentation based on model performance"""
    
    def __init__(self, initial_policy: str = 'medium'):
        self.current_policy = initial_policy
        self.performance_history = []
        self.policy_strengths = {'light': 0.3, 'medium': 0.6, 'heavy': 0.9}
        
        self.vision_pipeline = VisionAugmentationPipeline()
    
    def update_policy(self, current_performance: float, target_performance: float = 0.8):
        """Update augmentation policy based on performance"""
        self.performance_history.append(current_performance)
        
        # Simple adaptive strategy
        if len(self.performance_history) >= 3:
            recent_trend = np.mean(self.performance_history[-3:]) - np.mean(self.performance_history[-6:-3]) if len(self.performance_history) >= 6 else 0
            
            if current_performance > target_performance and recent_trend > 0:
                # Model is doing well, can handle more augmentation
                if self.current_policy == 'light':
                    self.current_policy = 'medium'
                elif self.current_policy == 'medium':
                    self.current_policy = 'heavy'
            elif current_performance < target_performance * 0.8:
                # Model struggling, reduce augmentation
                if self.current_policy == 'heavy':
                    self.current_policy = 'medium'
                elif self.current_policy == 'medium':
                    self.current_policy = 'light'
        
        print(f"    Updated policy to: {self.current_policy}")
    
    def get_current_transform(self) -> transforms.Compose:
        """Get current augmentation transform"""
        return self.vision_pipeline.get_transform(self.current_policy)
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Apply current augmentation policy"""
        return self.vision_pipeline(image, mode=self.current_policy)

# Test adaptive augmentation
print("Testing adaptive augmentation:")

adaptive_aug = AdaptiveAugmentation(initial_policy='light')

# Simulate training progress
performance_scenarios = [
    (0.5, "Initial low performance"),
    (0.6, "Improving performance"),
    (0.75, "Good performance"),
    (0.85, "High performance"),
    (0.9, "Excellent performance"),
    (0.7, "Performance drop"),
    (0.8, "Recovery")
]

for performance, description in performance_scenarios:
    print(f"  {description}: Performance = {performance:.2f}")
    adaptive_aug.update_policy(performance, target_performance=0.8)
    
    # Test current transform
    try:
        result = adaptive_aug(test_image)
        print(f"    Current policy '{adaptive_aug.current_policy}' - Output shape: {result.shape}")
    except Exception as e:
        print(f"    Error with current policy: {e}")

print("\n=== Pipeline Management and Composition ===")

class AugmentationManager:
    """Manage and compose different augmentation pipelines"""
    
    def __init__(self):
        self.pipelines = {}
        self.active_pipelines = []
    
    def register_pipeline(self, name: str, pipeline: Callable, weight: float = 1.0):
        """Register an augmentation pipeline"""
        self.pipelines[name] = {'pipeline': pipeline, 'weight': weight}
        print(f"  Registered pipeline: {name} (weight: {weight})")
    
    def set_active_pipelines(self, pipeline_names: List[str]):
        """Set which pipelines are currently active"""
        self.active_pipelines = pipeline_names
        print(f"  Active pipelines: {pipeline_names}")
    
    def random_pipeline_selection(self, data: Any) -> Any:
        """Randomly select and apply a pipeline"""
        if not self.active_pipelines:
            return data
        
        # Weight-based selection
        weights = [self.pipelines[name]['weight'] for name in self.active_pipelines]
        selected_pipeline = random.choices(self.active_pipelines, weights=weights)[0]
        
        pipeline = self.pipelines[selected_pipeline]['pipeline']
        return pipeline(data)
    
    def ensemble_augmentation(self, data: Any, num_variants: int = 3) -> List[Any]:
        """Create multiple augmented variants"""
        variants = []
        
        for _ in range(num_variants):
            augmented = self.random_pipeline_selection(data)
            variants.append(augmented)
        
        return variants
    
    def progressive_augmentation(self, data: Any, num_steps: int = 3) -> List[Any]:
        """Apply augmentation progressively"""
        results = [data]
        current_data = data
        
        for step in range(num_steps):
            if self.active_pipelines:
                # Select pipeline based on step intensity
                pipeline_idx = min(step, len(self.active_pipelines) - 1)
                pipeline_name = self.active_pipelines[pipeline_idx]
                pipeline = self.pipelines[pipeline_name]['pipeline']
                
                current_data = pipeline(current_data)
                results.append(current_data)
        
        return results

# Test pipeline management
print("Testing pipeline management:")

manager = AugmentationManager()

# Register different pipelines
light_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

medium_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor()
])

manager.register_pipeline('light', light_transform, weight=1.0)
manager.register_pipeline('medium', medium_transform, weight=2.0)

manager.set_active_pipelines(['light', 'medium'])

# Test random selection
try:
    random_result = manager.random_pipeline_selection(test_image)
    print(f"  Random selection result shape: {random_result.shape}")
except Exception as e:
    print(f"  Random selection error: {e}")

# Test ensemble augmentation
try:
    ensemble_results = manager.ensemble_augmentation(test_image, num_variants=3)
    print(f"  Ensemble augmentation created {len(ensemble_results)} variants")
    for i, variant in enumerate(ensemble_results):
        print(f"    Variant {i}: shape {variant.shape}")
except Exception as e:
    print(f"  Ensemble augmentation error: {e}")

print("\n=== Augmentation Best Practices ===")

print("Policy Selection Guidelines:")
print("1. Start with light augmentation and gradually increase")
print("2. Monitor validation performance with different policies")
print("3. Consider dataset size when choosing augmentation strength")
print("4. Use task-specific augmentation strategies")
print("5. Validate augmentation impact on model performance")

print("\nImplementation Best Practices:")
print("1. Apply augmentation only to training data")
print("2. Use consistent validation transforms")
print("3. Cache augmented data when computationally expensive")
print("4. Consider multi-processing for augmentation pipelines")
print("5. Profile augmentation performance impact")

print("\nCommon Pitfalls:")
print("1. Over-aggressive augmentation destroying important features")
print("2. Inconsistent augmentation between train/validation")
print("3. Not considering domain-specific constraints")
print("4. Memory issues with large augmentation pipelines")
print("5. Poor augmentation-to-dataset-size ratio")

print("\nPerformance Optimization:")
print("1. Use efficient augmentation libraries (Albumentations)")
print("2. Batch augmentation operations when possible")
print("3. Consider GPU-accelerated augmentation")
print("4. Use appropriate data types and formats")
print("5. Profile and optimize bottlenecks")

print("\n=== Data Augmentation Pipeline Complete ===")

# Memory cleanup
del test_image
torch.cuda.empty_cache() if torch.cuda.is_available() else None