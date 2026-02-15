#!/usr/bin/env python3
"""PyTorch Tensor Preprocessing - Data preprocessing with tensors"""

import torch
import torch.nn.functional as F
import numpy as np

print("=== Tensor Preprocessing Overview ===")

print("Common preprocessing operations:")
print("1. Type conversions and device management")
print("2. Shape transformations and resizing")
print("3. Value range normalization")
print("4. Data cleaning and filtering")
print("5. Feature extraction and selection")

print("\n=== Basic Tensor Preprocessing ===")

# Create sample data (simulating image data)
raw_data = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
print(f"Raw data shape: {raw_data.shape}, dtype: {raw_data.dtype}")
print(f"Value range: [{raw_data.min()}, {raw_data.max()}]")

# Convert to float and normalize to [0, 1]
normalized_data = raw_data.float() / 255.0
print(f"Normalized shape: {normalized_data.shape}, dtype: {normalized_data.dtype}")
print(f"Normalized range: [{normalized_data.min():.3f}, {normalized_data.max():.3f}]")

# Standardize to zero mean, unit variance
mean = normalized_data.mean()
std = normalized_data.std()
standardized_data = (normalized_data - mean) / std
print(f"Standardized mean: {standardized_data.mean():.6f}")
print(f"Standardized std: {standardized_data.std():.6f}")

print("\n=== Batch Preprocessing ===")

# Batch processing multiple samples
batch_size = 8
batch_data = torch.randint(0, 256, (batch_size, 3, 32, 32), dtype=torch.uint8)

# Efficient batch normalization
batch_normalized = batch_data.float() / 255.0

# Per-sample normalization
per_sample_mean = batch_normalized.view(batch_size, -1).mean(dim=1, keepdim=True)
per_sample_std = batch_normalized.view(batch_size, -1).std(dim=1, keepdim=True)
batch_standardized = (batch_normalized.view(batch_size, -1) - per_sample_mean) / (per_sample_std + 1e-8)
batch_standardized = batch_standardized.view(batch_size, 3, 32, 32)

print(f"Batch shape: {batch_data.shape}")
print(f"Per-sample means: {per_sample_mean.squeeze()}")
print(f"Per-sample stds: {per_sample_std.squeeze()}")

print("\n=== Image Preprocessing Pipeline ===")

def preprocess_image_tensor(tensor, target_size=(224, 224), normalize=True):
    """Complete image preprocessing pipeline"""
    # Convert to float if needed
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    
    # Normalize pixel values
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    
    # Resize if needed
    if tensor.shape[-2:] != target_size:
        tensor = F.interpolate(tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
        tensor = tensor.squeeze(0)
    
    # Standardize using ImageNet stats (common practice)
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
    
    return tensor

# Test preprocessing pipeline
sample_image = torch.randint(0, 256, (3, 128, 128), dtype=torch.uint8)
processed_image = preprocess_image_tensor(sample_image)

print(f"Original image: {sample_image.shape}, {sample_image.dtype}")
print(f"Processed image: {processed_image.shape}, {processed_image.dtype}")
print(f"Processed range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")

print("\n=== Text Preprocessing with Tensors ===")

# Simulating text preprocessing
vocab_size = 10000
sequence_length = 50

# Create sample token sequences
token_sequences = torch.randint(1, vocab_size, (32, sequence_length))

# Padding handling
def pad_sequences(sequences, max_length=None, pad_value=0):
    """Pad sequences to same length"""
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    padded = torch.full((len(sequences), max_length), pad_value, dtype=sequences[0].dtype)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_length)
        padded[i, :length] = seq[:length]
    
    return padded

# Variable length sequences
var_sequences = [torch.randint(1, vocab_size, (torch.randint(10, 50, (1,)).item(),)) for _ in range(5)]
padded_sequences = pad_sequences(var_sequences, max_length=50)

print(f"Variable lengths: {[len(seq) for seq in var_sequences]}")
print(f"Padded shape: {padded_sequences.shape}")

# Create attention masks
attention_masks = (padded_sequences != 0).float()
print(f"Attention mask sample:\n{attention_masks[0][:20]}")

print("\n=== Tabular Data Preprocessing ===")

# Simulating tabular data
num_samples = 1000
num_features = 20

# Mixed data types (continuous and categorical)
continuous_features = torch.randn(num_samples, 15)
categorical_features = torch.randint(0, 5, (num_samples, 5))

# Feature scaling
def scale_features(features, method='standard'):
    """Scale continuous features"""
    if method == 'standard':
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)
        return (features - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_vals = features.min(dim=0, keepdim=True)[0]
        max_vals = features.max(dim=0, keepdim=True)[0]
        return (features - min_vals) / (max_vals - min_vals + 1e-8)
    elif method == 'robust':
        median = features.median(dim=0, keepdim=True)[0]
        mad = torch.median(torch.abs(features - median), dim=0, keepdim=True)[0]
        return (features - median) / (mad + 1e-8)

scaled_continuous = scale_features(continuous_features, 'standard')

# One-hot encoding for categorical
def one_hot_encode(categorical, num_classes):
    """One-hot encode categorical features"""
    return F.one_hot(categorical, num_classes=num_classes).float()

categorical_onehot = one_hot_encode(categorical_features, num_classes=5)
categorical_onehot = categorical_onehot.view(num_samples, -1)  # Flatten

# Combine features
combined_features = torch.cat([scaled_continuous, categorical_onehot], dim=1)

print(f"Continuous features: {continuous_features.shape}")
print(f"Categorical features: {categorical_features.shape}")
print(f"One-hot encoded: {categorical_onehot.shape}")
print(f"Combined features: {combined_features.shape}")

print("\n=== Audio Preprocessing ===")

# Simulating audio signal preprocessing
sample_rate = 16000
duration = 2  # seconds
audio_signal = torch.randn(sample_rate * duration)

# Windowing and framing
def frame_signal(signal, frame_length, hop_length):
    """Frame audio signal into overlapping windows"""
    num_frames = (len(signal) - frame_length) // hop_length + 1
    frames = torch.zeros(num_frames, frame_length)
    
    for i in range(num_frames):
        start = i * hop_length
        frames[i] = signal[start:start + frame_length]
    
    return frames

frame_length = 400  # 25ms at 16kHz
hop_length = 160    # 10ms at 16kHz
audio_frames = frame_signal(audio_signal, frame_length, hop_length)

print(f"Audio signal length: {len(audio_signal)}")
print(f"Number of frames: {audio_frames.shape[0]}")
print(f"Frame length: {audio_frames.shape[1]}")

# Apply window function
window = torch.hann_window(frame_length)
windowed_frames = audio_frames * window

# Simple spectral features (magnitude spectrum)
fft_frames = torch.fft.fft(windowed_frames, dim=1)
magnitude_spectrum = torch.abs(fft_frames[:, :frame_length//2 + 1])

print(f"Magnitude spectrum shape: {magnitude_spectrum.shape}")

print("\n=== Data Quality Checks ===")

def check_data_quality(tensor):
    """Perform data quality checks"""
    checks = {}
    
    # Check for NaN values
    checks['has_nan'] = torch.isnan(tensor).any().item()
    checks['nan_count'] = torch.isnan(tensor).sum().item()
    
    # Check for infinite values
    checks['has_inf'] = torch.isinf(tensor).any().item()
    checks['inf_count'] = torch.isinf(tensor).sum().item()
    
    # Check value ranges
    checks['min_value'] = tensor.min().item() if not torch.isnan(tensor).all() else float('nan')
    checks['max_value'] = tensor.max().item() if not torch.isnan(tensor).all() else float('nan')
    checks['mean_value'] = tensor.mean().item() if not torch.isnan(tensor).all() else float('nan')
    checks['std_value'] = tensor.std().item() if not torch.isnan(tensor).all() else float('nan')
    
    # Check for outliers (using IQR method)
    if not torch.isnan(tensor).all():
        q1, q3 = torch.quantile(tensor.flatten(), torch.tensor([0.25, 0.75]))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = (tensor < lower_bound) | (tensor > upper_bound)
        checks['outlier_count'] = outliers.sum().item()
        checks['outlier_percentage'] = (outliers.sum().float() / tensor.numel() * 100).item()
    
    return checks

# Test with clean data
clean_data = torch.randn(100, 10)
clean_checks = check_data_quality(clean_data)

# Test with problematic data
problematic_data = torch.randn(100, 10)
problematic_data[5, 3] = float('nan')
problematic_data[10, 7] = float('inf')
problematic_data[15:20, 2] = 100  # Outliers

problematic_checks = check_data_quality(problematic_data)

print("\nClean data quality:")
for key, value in clean_checks.items():
    print(f"  {key}: {value}")

print("\nProblematic data quality:")
for key, value in problematic_checks.items():
    print(f"  {key}: {value}")

print("\n=== Advanced Preprocessing Techniques ===")

# Principal Component Analysis (simplified)
def simple_pca(data, n_components):
    """Simple PCA implementation"""
    # Center the data
    mean = data.mean(dim=0, keepdim=True)
    centered_data = data - mean
    
    # Compute covariance matrix
    cov_matrix = torch.mm(centered_data.T, centered_data) / (data.shape[0] - 1)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eig(cov_matrix)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    
    # Sort by eigenvalues
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    top_eigenvectors = eigenvectors[:, sorted_indices[:n_components]]
    
    # Transform data
    transformed_data = torch.mm(centered_data, top_eigenvectors)
    
    return transformed_data, top_eigenvectors, eigenvalues[sorted_indices[:n_components]]

# Test PCA
high_dim_data = torch.randn(100, 20)
pca_data, components, explained_variance = simple_pca(high_dim_data, n_components=5)

print(f"Original data shape: {high_dim_data.shape}")
print(f"PCA transformed shape: {pca_data.shape}")
print(f"Explained variance: {explained_variance}")

print("\n=== Preprocessing Best Practices ===")

print("Data Preprocessing Guidelines:")
print("1. Always check data quality first (NaN, inf, outliers)")
print("2. Handle missing values before normalization")
print("3. Apply transformations consistently to train/val/test")
print("4. Use appropriate normalization for your data type")
print("5. Consider data distribution when choosing scaling methods")
print("6. Preserve preprocessing parameters for inference")
print("7. Use vectorized operations for efficiency")
print("8. Monitor memory usage with large datasets")

print("\nCommon Preprocessing Steps:")
print("- Images: uint8 -> float32, /255.0, resize, normalize")
print("- Text: tokenize, pad/truncate, create attention masks")
print("- Audio: frame, window, extract features (MFCC, spectogram)")
print("- Tabular: handle missing, scale continuous, encode categorical")
print("- Time series: normalize, create windows, handle seasonality")

print("\n=== Tensor Preprocessing Complete ===") 