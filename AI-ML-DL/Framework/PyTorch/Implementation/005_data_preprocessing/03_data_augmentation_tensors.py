#!/usr/bin/env python3
"""PyTorch Data Augmentation with Tensors - Tensor-based data augmentation"""

import torch
import torch.nn.functional as F
import math
import random

print("=== Data Augmentation Overview ===")

print("Augmentation categories:")
print("1. Geometric transformations (rotation, scaling, flipping)")
print("2. Color/intensity transformations")
print("3. Noise injection")
print("4. Cutout/masking techniques")
print("5. Mixup and advanced techniques")
print("6. Text augmentation")
print("7. Audio augmentation")

print("\n=== Image Geometric Augmentations ===")

def random_flip(tensor, p=0.5, dim=-1):
    """Random horizontal/vertical flip"""
    if random.random() < p:
        return torch.flip(tensor, dims=[dim])
    return tensor

def random_rotation(tensor, max_angle=30):
    """Random rotation using affine transformation"""
    angle = random.uniform(-max_angle, max_angle)
    angle_rad = math.radians(angle)
    
    # Create rotation matrix
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    rotation_matrix = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0]
    ], dtype=tensor.dtype, device=tensor.device)
    
    # Apply affine transformation
    grid = F.affine_grid(rotation_matrix.unsqueeze(0), tensor.unsqueeze(0).shape, align_corners=False)
    rotated = F.grid_sample(tensor.unsqueeze(0), grid, align_corners=False, mode='bilinear', padding_mode='zeros')
    
    return rotated.squeeze(0)

def random_scale(tensor, scale_range=(0.8, 1.2)):
    """Random scaling transformation"""
    scale = random.uniform(*scale_range)
    
    # Create scaling matrix
    scale_matrix = torch.tensor([
        [scale, 0, 0],
        [0, scale, 0]
    ], dtype=tensor.dtype, device=tensor.device)
    
    grid = F.affine_grid(scale_matrix.unsqueeze(0), tensor.unsqueeze(0).shape, align_corners=False)
    scaled = F.grid_sample(tensor.unsqueeze(0), grid, align_corners=False, mode='bilinear', padding_mode='zeros')
    
    return scaled.squeeze(0)

def random_translation(tensor, max_translate=0.1):
    """Random translation transformation"""
    h, w = tensor.shape[-2:]
    tx = random.uniform(-max_translate, max_translate) * w
    ty = random.uniform(-max_translate, max_translate) * h
    
    # Create translation matrix
    translation_matrix = torch.tensor([
        [1, 0, tx],
        [0, 1, ty]
    ], dtype=tensor.dtype, device=tensor.device)
    
    grid = F.affine_grid(translation_matrix.unsqueeze(0), tensor.unsqueeze(0).shape, align_corners=False)
    translated = F.grid_sample(tensor.unsqueeze(0), grid, align_corners=False, mode='bilinear', padding_mode='zeros')
    
    return translated.squeeze(0)

# Test geometric augmentations
sample_image = torch.randn(3, 64, 64)

flipped = random_flip(sample_image, p=1.0, dim=-1)  # Always flip for demo
rotated = random_rotation(sample_image, max_angle=45)
scaled = random_scale(sample_image, scale_range=(0.7, 1.3))
translated = random_translation(sample_image, max_translate=0.2)

print(f"Original image shape: {sample_image.shape}")
print(f"Flipped shape: {flipped.shape}")
print(f"Rotated shape: {rotated.shape}")
print(f"Scaled shape: {scaled.shape}")
print(f"Translated shape: {translated.shape}")

print("\n=== Color/Intensity Augmentations ===")

def random_brightness(tensor, brightness_range=(-0.2, 0.2)):
    """Random brightness adjustment"""
    brightness_factor = random.uniform(*brightness_range)
    return torch.clamp(tensor + brightness_factor, 0, 1)

def random_contrast(tensor, contrast_range=(0.8, 1.2)):
    """Random contrast adjustment"""
    contrast_factor = random.uniform(*contrast_range)
    mean = tensor.mean(dim=[-2, -1], keepdim=True)
    return torch.clamp((tensor - mean) * contrast_factor + mean, 0, 1)

def random_saturation(tensor, saturation_range=(0.8, 1.2)):
    """Random saturation adjustment (for RGB images)"""
    if tensor.shape[0] != 3:
        return tensor
    
    saturation_factor = random.uniform(*saturation_range)
    
    # Convert to grayscale weights
    gray_weights = torch.tensor([0.299, 0.587, 0.114], device=tensor.device, dtype=tensor.dtype)
    gray = (tensor * gray_weights.view(3, 1, 1)).sum(dim=0, keepdim=True)
    gray = gray.expand_as(tensor)
    
    return torch.clamp(gray + saturation_factor * (tensor - gray), 0, 1)

def random_hue(tensor, hue_range=(-0.1, 0.1)):
    """Random hue adjustment (simplified RGB version)"""
    if tensor.shape[0] != 3:
        return tensor
    
    hue_factor = random.uniform(*hue_range)
    
    # Simple hue shift by rotating RGB channels
    if abs(hue_factor) < 1e-6:
        return tensor
    
    # Apply a simple hue transformation
    cos_h = math.cos(hue_factor * 2 * math.pi)
    sin_h = math.sin(hue_factor * 2 * math.pi)
    
    # Simplified hue transformation matrix
    transform_matrix = torch.tensor([
        [cos_h + (1 - cos_h) / 3, (1 - cos_h) / 3 - sin_h / math.sqrt(3), (1 - cos_h) / 3 + sin_h / math.sqrt(3)],
        [(1 - cos_h) / 3 + sin_h / math.sqrt(3), cos_h + (1 - cos_h) / 3, (1 - cos_h) / 3 - sin_h / math.sqrt(3)],
        [(1 - cos_h) / 3 - sin_h / math.sqrt(3), (1 - cos_h) / 3 + sin_h / math.sqrt(3), cos_h + (1 - cos_h) / 3]
    ], device=tensor.device, dtype=tensor.dtype)
    
    # Apply transformation
    flattened = tensor.view(3, -1)
    transformed = torch.mm(transform_matrix, flattened)
    return torch.clamp(transformed.view_as(tensor), 0, 1)

# Test color augmentations
rgb_image = torch.rand(3, 32, 32)  # Random RGB image in [0, 1]

bright_image = random_brightness(rgb_image, brightness_range=(0.2, 0.2))  # Fixed for demo
contrast_image = random_contrast(rgb_image, contrast_range=(1.5, 1.5))  # Fixed for demo
saturated_image = random_saturation(rgb_image, saturation_range=(1.5, 1.5))  # Fixed for demo
hue_shifted = random_hue(rgb_image, hue_range=(0.1, 0.1))  # Fixed for demo

print(f"Original RGB range: [{rgb_image.min():.3f}, {rgb_image.max():.3f}]")
print(f"Brightness adjusted: [{bright_image.min():.3f}, {bright_image.max():.3f}]")
print(f"Contrast adjusted: [{contrast_image.min():.3f}, {contrast_image.max():.3f}]")

print("\n=== Noise Augmentations ===")

def add_gaussian_noise(tensor, noise_std=0.1):
    """Add Gaussian noise"""
    noise = torch.randn_like(tensor) * noise_std
    return torch.clamp(tensor + noise, 0, 1)

def add_salt_pepper_noise(tensor, noise_prob=0.05):
    """Add salt and pepper noise"""
    noise_mask = torch.rand_like(tensor) < noise_prob
    salt_mask = torch.rand_like(tensor) < 0.5
    
    noisy_tensor = tensor.clone()
    noisy_tensor[noise_mask & salt_mask] = 1.0  # Salt (white)
    noisy_tensor[noise_mask & ~salt_mask] = 0.0  # Pepper (black)
    
    return noisy_tensor

def add_speckle_noise(tensor, noise_std=0.1):
    """Add speckle (multiplicative) noise"""
    noise = torch.randn_like(tensor) * noise_std + 1
    return torch.clamp(tensor * noise, 0, 1)

# Test noise augmentations
noisy_gaussian = add_gaussian_noise(rgb_image, noise_std=0.1)
noisy_salt_pepper = add_salt_pepper_noise(rgb_image, noise_prob=0.02)
noisy_speckle = add_speckle_noise(rgb_image, noise_std=0.1)

print(f"Original vs Gaussian noise MSE: {F.mse_loss(rgb_image, noisy_gaussian):.6f}")
print(f"Original vs Salt&Pepper noise MSE: {F.mse_loss(rgb_image, noisy_salt_pepper):.6f}")
print(f"Original vs Speckle noise MSE: {F.mse_loss(rgb_image, noisy_speckle):.6f}")

print("\n=== Cutout/Masking Augmentations ===")

def random_cutout(tensor, cutout_size=16, n_holes=1):
    """Random cutout/erasing"""
    h, w = tensor.shape[-2:]
    mask = torch.ones_like(tensor)
    
    for _ in range(n_holes):
        y = random.randint(0, h - cutout_size)
        x = random.randint(0, w - cutout_size)
        mask[..., y:y+cutout_size, x:x+cutout_size] = 0
    
    return tensor * mask

def random_erasing(tensor, p=0.5, scale_range=(0.02, 0.33), ratio_range=(0.3, 3.3)):
    """Random erasing augmentation"""
    if random.random() > p:
        return tensor
    
    area = tensor.shape[-2] * tensor.shape[-1]
    
    for _ in range(100):  # Try up to 100 times
        target_area = random.uniform(*scale_range) * area
        aspect_ratio = random.uniform(*ratio_range)
        
        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))
        
        if h < tensor.shape[-2] and w < tensor.shape[-1]:
            y = random.randint(0, tensor.shape[-2] - h)
            x = random.randint(0, tensor.shape[-1] - w)
            
            tensor_copy = tensor.clone()
            tensor_copy[..., y:y+h, x:x+w] = torch.rand_like(tensor_copy[..., y:y+h, x:x+w])
            return tensor_copy
    
    return tensor

def mixup(tensor1, tensor2, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = random.betavariate(alpha, alpha)
    else:
        lam = 1
    
    mixed_tensor = lam * tensor1 + (1 - lam) * tensor2
    return mixed_tensor, lam

def cutmix(tensor1, tensor2, alpha=1.0):
    """CutMix augmentation"""
    lam = random.betavariate(alpha, alpha) if alpha > 0 else 1
    
    h, w = tensor1.shape[-2:]
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    
    # Random center
    cx = random.randint(0, w)
    cy = random.randint(0, h)
    
    # Bound the box
    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, w)
    bby2 = min(cy + cut_h // 2, h)
    
    mixed_tensor = tensor1.clone()
    mixed_tensor[..., bby1:bby2, bbx1:bbx2] = tensor2[..., bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to match actual area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    
    return mixed_tensor, lam

# Test cutout and masking
cutout_image = random_cutout(rgb_image, cutout_size=8, n_holes=2)
erased_image = random_erasing(rgb_image, p=1.0)  # Always apply for demo

# Test mixup and cutmix
rgb_image2 = torch.rand(3, 32, 32)
mixed_image, mix_lambda = mixup(rgb_image, rgb_image2, alpha=0.4)
cutmix_image, cutmix_lambda = cutmix(rgb_image, rgb_image2, alpha=1.0)

print(f"Cutout applied: {(cutout_image == 0).sum().item()} pixels zeroed")
print(f"Random erasing applied successfully")
print(f"Mixup lambda: {mix_lambda:.3f}")
print(f"CutMix lambda: {cutmix_lambda:.3f}")

print("\n=== Text Augmentation ===")

def token_dropout(tokens, dropout_prob=0.1, mask_token_id=0):
    """Random token dropout for text"""
    mask = torch.rand(tokens.shape) < dropout_prob
    augmented_tokens = tokens.clone()
    augmented_tokens[mask] = mask_token_id
    return augmented_tokens

def token_replacement(tokens, vocab_size, replacement_prob=0.1):
    """Random token replacement"""
    mask = torch.rand(tokens.shape) < replacement_prob
    random_tokens = torch.randint(1, vocab_size, tokens.shape)
    augmented_tokens = torch.where(mask, random_tokens, tokens)
    return augmented_tokens

def token_insertion(tokens, vocab_size, insertion_prob=0.1, max_length=None):
    """Random token insertion"""
    if max_length is None:
        max_length = tokens.shape[-1] * 2
    
    augmented_tokens = []
    for token in tokens:
        augmented_tokens.append(token)
        if random.random() < insertion_prob and len(augmented_tokens) < max_length:
            random_token = torch.randint(1, vocab_size, (1,))
            augmented_tokens.append(random_token)
    
    # Pad or truncate to original length
    if len(augmented_tokens) > tokens.shape[-1]:
        augmented_tokens = augmented_tokens[:tokens.shape[-1]]
    else:
        augmented_tokens.extend([torch.tensor(0)] * (tokens.shape[-1] - len(augmented_tokens)))
    
    return torch.stack(augmented_tokens)

# Test text augmentations
text_tokens = torch.randint(1, 1000, (20,))  # 20 tokens from vocab of 1000

dropout_tokens = token_dropout(text_tokens, dropout_prob=0.2)
replaced_tokens = token_replacement(text_tokens, vocab_size=1000, replacement_prob=0.1)
inserted_tokens = token_insertion(text_tokens, vocab_size=1000, insertion_prob=0.1)

print(f"Original tokens shape: {text_tokens.shape}")
print(f"Tokens changed by dropout: {(text_tokens != dropout_tokens).sum().item()}")
print(f"Tokens changed by replacement: {(text_tokens != replaced_tokens).sum().item()}")
print(f"Shape after insertion: {inserted_tokens.shape}")

print("\n=== Audio Augmentation ===")

def time_stretch(audio, stretch_factor=1.0):
    """Simple time stretching (naive implementation)"""
    if stretch_factor == 1.0:
        return audio
    
    # Simple linear interpolation for time stretching
    original_length = len(audio)
    new_length = int(original_length / stretch_factor)
    
    indices = torch.linspace(0, original_length - 1, new_length)
    stretched_audio = F.interpolate(audio.unsqueeze(0).unsqueeze(0), size=new_length, mode='linear', align_corners=True)
    
    return stretched_audio.squeeze()

def pitch_shift(audio, semitones=0):
    """Simple pitch shifting using resampling"""
    if semitones == 0:
        return audio
    
    # Calculate stretch factor (2^(semitones/12))
    stretch_factor = 2.0 ** (semitones / 12.0)
    
    # Time stretch then resample back to original length
    stretched = time_stretch(audio, stretch_factor)
    
    # Resample back to original length
    original_length = len(audio)
    resampled = F.interpolate(stretched.unsqueeze(0).unsqueeze(0), size=original_length, mode='linear', align_corners=True)
    
    return resampled.squeeze()

def add_background_noise(audio, noise_level=0.1):
    """Add background noise to audio"""
    noise = torch.randn_like(audio) * noise_level
    return audio + noise

# Test audio augmentations
sample_audio = torch.sin(2 * math.pi * 440 * torch.linspace(0, 1, 16000))  # 1-second 440Hz tone

stretched_audio = time_stretch(sample_audio, stretch_factor=1.2)
pitch_shifted = pitch_shift(sample_audio, semitones=2)
noisy_audio = add_background_noise(sample_audio, noise_level=0.05)

print(f"Original audio length: {len(sample_audio)}")
print(f"Stretched audio length: {len(stretched_audio)}")
print(f"Pitch shifted length: {len(pitch_shifted)}")
print(f"Noisy audio SNR: {20 * torch.log10(sample_audio.std() / (noisy_audio - sample_audio).std()):.1f} dB")

print("\n=== Augmentation Pipeline ===")

class AugmentationPipeline:
    """Composable augmentation pipeline"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, tensor):
        for transform in self.transforms:
            tensor = transform(tensor)
        return tensor

# Define augmentation functions with probability
def random_apply(transform, p=0.5):
    """Apply transform with probability p"""
    def wrapper(tensor):
        if random.random() < p:
            return transform(tensor)
        return tensor
    return wrapper

# Create augmentation pipeline
image_pipeline = AugmentationPipeline([
    random_apply(lambda x: random_flip(x, p=1.0), p=0.5),
    random_apply(lambda x: random_rotation(x, max_angle=15), p=0.3),
    random_apply(lambda x: random_brightness(x, brightness_range=(-0.1, 0.1)), p=0.7),
    random_apply(lambda x: random_contrast(x, contrast_range=(0.9, 1.1)), p=0.7),
    random_apply(lambda x: add_gaussian_noise(x, noise_std=0.05), p=0.2),
    random_apply(lambda x: random_erasing(x, p=1.0), p=0.1)
])

# Apply pipeline
augmented_image = image_pipeline(rgb_image)
print(f"Augmentation pipeline applied successfully")
print(f"Output shape: {augmented_image.shape}")

print("\n=== Augmentation Best Practices ===")

print("Augmentation Guidelines:")
print("1. Choose augmentations appropriate for your domain")
print("2. Use probability-based application (not all at once)")
print("3. Preserve label invariance (don't change ground truth)")
print("4. Test augmentations with small datasets first")
print("5. Monitor validation performance with/without augmentation")
print("6. Consider computational cost vs benefit")
print("7. Use different augmentations for training vs testing")

print("\nDomain-specific recommendations:")
print("- Images: geometric + color + noise (avoid changing semantic content)")
print("- Text: dropout + replacement (preserve grammar when possible)")
print("- Audio: time/pitch shifts + noise (preserve phonetic content)")
print("- Time series: jittering + warping + masking")
print("- Medical images: careful geometric only (preserve diagnostic features)")

print("\n=== Data Augmentation Complete ===") 