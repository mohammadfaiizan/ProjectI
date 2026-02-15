#!/usr/bin/env python3
"""PyTorch Data Transformation Operations - Various transformation operations"""

import torch
import torch.nn.functional as F
import math

print("=== Data Transformation Overview ===")

print("Transformation categories:")
print("1. Mathematical transformations (log, power, trigonometric)")
print("2. Functional transformations (activation functions)")
print("3. Spatial transformations (geometric, filtering)")
print("4. Frequency domain transformations (FFT, wavelets)")
print("5. Statistical transformations (standardization, PCA)")
print("6. Custom transformations")

print("\n=== Mathematical Transformations ===")

def log_transform(tensor, base='e', offset=1e-8):
    """Logarithmic transformation"""
    if base == 'e':
        return torch.log(tensor + offset)
    elif base == 10:
        return torch.log10(tensor + offset)
    elif base == 2:
        return torch.log2(tensor + offset)
    else:
        return torch.log(tensor + offset) / math.log(base)

def power_transform(tensor, power=0.5):
    """Power transformation (Box-Cox style)"""
    if power == 0:
        return torch.log(tensor + 1e-8)
    else:
        return torch.sign(tensor) * torch.pow(torch.abs(tensor), power)

def reciprocal_transform(tensor, offset=1e-8):
    """Reciprocal transformation"""
    return 1.0 / (tensor + offset)

def sqrt_transform(tensor):
    """Square root transformation"""
    return torch.sqrt(torch.abs(tensor)) * torch.sign(tensor)

# Test mathematical transformations
positive_data = torch.abs(torch.randn(100)) + 1
skewed_data = torch.exp(torch.randn(100))

log_transformed = log_transform(positive_data, base='e')
power_transformed = power_transform(skewed_data, power=0.3)
sqrt_transformed = sqrt_transform(positive_data - 1)

print(f"Original data range: [{positive_data.min():.3f}, {positive_data.max():.3f}]")
print(f"Log transformed range: [{log_transformed.min():.3f}, {log_transformed.max():.3f}]")
print(f"Power transformed range: [{power_transformed.min():.3f}, {power_transformed.max():.3f}]")
print(f"Sqrt transformed range: [{sqrt_transformed.min():.3f}, {sqrt_transformed.max():.3f}]")

print("\n=== Trigonometric Transformations ===")

def cyclical_encoding(values, period):
    """Encode cyclical values (e.g., time, angles) as sin/cos"""
    normalized = 2 * math.pi * values / period
    sin_encoded = torch.sin(normalized)
    cos_encoded = torch.cos(normalized)
    return torch.stack([sin_encoded, cos_encoded], dim=-1)

def polar_transform(x, y):
    """Convert Cartesian to polar coordinates"""
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    return r, theta

def cartesian_transform(r, theta):
    """Convert polar to Cartesian coordinates"""
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return x, y

# Test trigonometric transformations
hours = torch.arange(24.0)  # 24-hour cycle
cyclical_hours = cyclical_encoding(hours, period=24)

print(f"Hours: {hours[:5]}")
print(f"Cyclical encoding shape: {cyclical_hours.shape}")
print(f"Hour 0 encoding: {cyclical_hours[0]}")
print(f"Hour 6 encoding: {cyclical_hours[6]}")
print(f"Hour 12 encoding: {cyclical_hours[12]}")

# Test coordinate transformations
x_coords = torch.randn(100)
y_coords = torch.randn(100)
r_coords, theta_coords = polar_transform(x_coords, y_coords)
x_back, y_back = cartesian_transform(r_coords, theta_coords)

print(f"Coordinate transformation error: {torch.mean(torch.abs(x_coords - x_back)):.6f}")

print("\n=== Activation Function Transformations ===")

def apply_activation(tensor, activation='relu'):
    """Apply various activation functions"""
    if activation == 'relu':
        return F.relu(tensor)
    elif activation == 'sigmoid':
        return torch.sigmoid(tensor)
    elif activation == 'tanh':
        return torch.tanh(tensor)
    elif activation == 'softmax':
        return F.softmax(tensor, dim=-1)
    elif activation == 'gelu':
        return F.gelu(tensor)
    elif activation == 'swish':
        return tensor * torch.sigmoid(tensor)
    elif activation == 'mish':
        return tensor * torch.tanh(F.softplus(tensor))
    else:
        raise ValueError(f"Unknown activation: {activation}")

def custom_activation(tensor, alpha=0.1, beta=1.0):
    """Custom parameterized activation function"""
    return torch.where(tensor > 0, beta * tensor, alpha * tensor)

# Test activation transformations
activation_data = torch.randn(1000)

activations = ['relu', 'sigmoid', 'tanh', 'gelu', 'swish', 'mish']
for activation in activations:
    transformed = apply_activation(activation_data, activation)
    print(f"{activation.upper()}: range [{transformed.min():.3f}, {transformed.max():.3f}], mean {transformed.mean():.3f}")

custom_transformed = custom_activation(activation_data, alpha=0.2, beta=1.5)
print(f"Custom activation: range [{custom_transformed.min():.3f}, {custom_transformed.max():.3f}]")

print("\n=== Spatial Transformations ===")

def gaussian_filter_1d(tensor, sigma=1.0, kernel_size=None):
    """1D Gaussian filtering"""
    if kernel_size is None:
        kernel_size = int(2 * 3 * sigma + 1)  # 6*sigma + 1
    
    # Create Gaussian kernel
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel = torch.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    
    # Apply convolution
    padding = kernel_size // 2
    padded_tensor = F.pad(tensor.unsqueeze(0).unsqueeze(0), (padding, padding), mode='reflect')
    filtered = F.conv1d(padded_tensor, kernel.unsqueeze(0).unsqueeze(0))
    
    return filtered.squeeze()

def sobel_edge_detection(image):
    """Simple Sobel edge detection for 2D tensors"""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    
    # Add batch and channel dimensions
    sobel_x = sobel_x.unsqueeze(0).unsqueeze(0)
    sobel_y = sobel_y.unsqueeze(0).unsqueeze(0)
    
    # Apply convolution
    image_4d = image.unsqueeze(0).unsqueeze(0)
    edges_x = F.conv2d(image_4d, sobel_x, padding=1)
    edges_y = F.conv2d(image_4d, sobel_y, padding=1)
    
    # Combine gradients
    edges = torch.sqrt(edges_x**2 + edges_y**2)
    
    return edges.squeeze()

def morphological_operations(binary_image, operation='dilation', kernel_size=3):
    """Basic morphological operations"""
    kernel = torch.ones(kernel_size, kernel_size)
    
    if operation == 'dilation':
        # Maximum filter (dilation)
        result = F.max_pool2d(binary_image.unsqueeze(0).unsqueeze(0), 
                             kernel_size, stride=1, padding=kernel_size//2)
    elif operation == 'erosion':
        # Minimum filter (erosion) - implemented as negative max pool of negative image
        result = -F.max_pool2d(-binary_image.unsqueeze(0).unsqueeze(0), 
                              kernel_size, stride=1, padding=kernel_size//2)
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    return result.squeeze()

# Test spatial transformations
signal_1d = torch.sin(torch.linspace(0, 4*math.pi, 100)) + 0.1 * torch.randn(100)
filtered_signal = gaussian_filter_1d(signal_1d, sigma=2.0)

print(f"Original signal std: {signal_1d.std():.6f}")
print(f"Filtered signal std: {filtered_signal.std():.6f}")

# Test edge detection
test_image = torch.zeros(50, 50)
test_image[20:30, 20:30] = 1.0  # Square
edges = sobel_edge_detection(test_image)

print(f"Original image range: [{test_image.min():.1f}, {test_image.max():.1f}]")
print(f"Edge image range: [{edges.min():.3f}, {edges.max():.3f}]")

print("\n=== Frequency Domain Transformations ===")

def fft_transform(signal, dim=-1):
    """Fast Fourier Transform"""
    fft_result = torch.fft.fft(signal, dim=dim)
    magnitude = torch.abs(fft_result)
    phase = torch.angle(fft_result)
    return fft_result, magnitude, phase

def spectral_filtering(signal, low_freq=None, high_freq=None):
    """Frequency domain filtering"""
    fft_signal = torch.fft.fft(signal)
    freqs = torch.fft.fftfreq(len(signal))
    
    # Create filter
    filter_mask = torch.ones_like(freqs)
    
    if low_freq is not None:
        filter_mask[torch.abs(freqs) < low_freq] = 0
    
    if high_freq is not None:
        filter_mask[torch.abs(freqs) > high_freq] = 0
    
    # Apply filter and inverse transform
    filtered_fft = fft_signal * filter_mask
    filtered_signal = torch.fft.ifft(filtered_fft).real
    
    return filtered_signal

def spectrogram(signal, window_size=64, hop_length=32):
    """Compute spectrogram"""
    n_frames = (len(signal) - window_size) // hop_length + 1
    spectrogram_data = torch.zeros(n_frames, window_size // 2 + 1)
    
    # Hanning window
    window = torch.hann_window(window_size)
    
    for i in range(n_frames):
        start = i * hop_length
        frame = signal[start:start + window_size] * window
        
        # Zero-pad if necessary
        if len(frame) < window_size:
            frame = F.pad(frame, (0, window_size - len(frame)))
        
        # FFT
        fft_frame = torch.fft.fft(frame)
        magnitude = torch.abs(fft_frame[:window_size // 2 + 1])
        spectrogram_data[i] = magnitude
    
    return spectrogram_data

# Test frequency transformations
test_signal = torch.sin(2 * math.pi * 5 * torch.linspace(0, 1, 1000)) + \
              0.5 * torch.sin(2 * math.pi * 20 * torch.linspace(0, 1, 1000)) + \
              0.1 * torch.randn(1000)

fft_result, magnitude, phase = fft_transform(test_signal)
print(f"FFT magnitude shape: {magnitude.shape}")
print(f"Dominant frequency bin: {magnitude.argmax().item()}")

# High-pass filtering
filtered_signal = spectral_filtering(test_signal, low_freq=0.01)
print(f"Original signal energy: {test_signal.var():.6f}")
print(f"Filtered signal energy: {filtered_signal.var():.6f}")

# Spectrogram
spec = spectrogram(test_signal, window_size=128)
print(f"Spectrogram shape: {spec.shape}")

print("\n=== Statistical Transformations ===")

def z_score_transform(tensor, dim=None, eps=1e-8):
    """Z-score normalization"""
    mean = tensor.mean(dim=dim, keepdim=True)
    std = tensor.std(dim=dim, keepdim=True)
    return (tensor - mean) / (std + eps)

def quantile_transform(tensor, n_quantiles=1000):
    """Transform to uniform distribution using quantiles"""
    # Sort the data
    sorted_tensor, sort_indices = torch.sort(tensor.flatten())
    
    # Create quantile mapping
    quantiles = torch.linspace(0, 1, n_quantiles)
    quantile_values = torch.quantile(sorted_tensor, quantiles)
    
    # Map each value to its quantile
    transformed = torch.zeros_like(tensor.flatten())
    for i, value in enumerate(tensor.flatten()):
        # Find closest quantile
        distances = torch.abs(quantile_values - value)
        closest_quantile_idx = distances.argmin()
        transformed[i] = quantiles[closest_quantile_idx]
    
    return transformed.reshape(tensor.shape)

def rank_transform(tensor, dim=None):
    """Convert values to their ranks"""
    if dim is None:
        flat_tensor = tensor.flatten()
        sorted_vals, sorted_indices = torch.sort(flat_tensor)
        ranks = torch.zeros_like(flat_tensor)
        ranks[sorted_indices] = torch.arange(len(flat_tensor), dtype=torch.float32)
        return ranks.reshape(tensor.shape)
    else:
        # Rank along specific dimension
        sorted_vals, sorted_indices = torch.sort(tensor, dim=dim)
        ranks = torch.zeros_like(tensor)
        
        # This is a simplified version - proper ranking needs tie handling
        for i in range(tensor.shape[dim]):
            indices = sorted_indices.select(dim, i)
            ranks.scatter_(dim, indices.unsqueeze(dim), 
                          torch.tensor(float(i)).expand_as(indices.unsqueeze(dim)))
        
        return ranks

# Test statistical transformations
stat_data = torch.exp(torch.randn(1000))  # Log-normal distribution

z_transformed = z_score_transform(stat_data)
quantile_transformed = quantile_transform(stat_data)
rank_transformed = rank_transform(stat_data)

print(f"Original data: mean={stat_data.mean():.3f}, std={stat_data.std():.3f}")
print(f"Z-score: mean={z_transformed.mean():.6f}, std={z_transformed.std():.3f}")
print(f"Quantile transform range: [{quantile_transformed.min():.3f}, {quantile_transformed.max():.3f}]")
print(f"Rank transform range: [{rank_transformed.min():.1f}, {rank_transformed.max():.1f}]")

print("\n=== Custom Transformations ===")

class Transform:
    """Base class for custom transformations"""
    
    def __init__(self):
        self.fitted = False
    
    def fit(self, data):
        """Fit transformation parameters"""
        raise NotImplementedError
    
    def transform(self, data):
        """Apply transformation"""
        raise NotImplementedError
    
    def inverse_transform(self, data):
        """Reverse transformation"""
        raise NotImplementedError
    
    def fit_transform(self, data):
        """Fit and transform in one step"""
        self.fit(data)
        return self.transform(data)

class PowerTransform(Transform):
    """Box-Cox style power transformation"""
    
    def __init__(self, method='box-cox'):
        super().__init__()
        self.method = method
        self.lambda_param = None
        self.shift = None
    
    def fit(self, data):
        """Find optimal lambda parameter"""
        # Ensure positive data
        self.shift = -data.min() + 1e-8 if data.min() <= 0 else 0
        shifted_data = data + self.shift
        
        # Simple grid search for optimal lambda
        lambdas = torch.linspace(-2, 2, 41)
        best_lambda = 0
        best_score = float('inf')
        
        for lam in lambdas:
            if lam == 0:
                transformed = torch.log(shifted_data)
            else:
                transformed = (torch.pow(shifted_data, lam) - 1) / lam
            
            # Score based on normality (simplified using skewness)
            mean = transformed.mean()
            std = transformed.std()
            if std > 0:
                skewness = ((transformed - mean) / std).pow(3).mean()
                score = abs(skewness)
                
                if score < best_score:
                    best_score = score
                    best_lambda = lam
        
        self.lambda_param = best_lambda
        self.fitted = True
    
    def transform(self, data):
        if not self.fitted:
            raise ValueError("Transform must be fitted first")
        
        shifted_data = data + self.shift
        
        if self.lambda_param == 0:
            return torch.log(shifted_data)
        else:
            return (torch.pow(shifted_data, self.lambda_param) - 1) / self.lambda_param
    
    def inverse_transform(self, data):
        if not self.fitted:
            raise ValueError("Transform must be fitted first")
        
        if self.lambda_param == 0:
            original = torch.exp(data)
        else:
            original = torch.pow(self.lambda_param * data + 1, 1 / self.lambda_param)
        
        return original - self.shift

class CompositeTransform(Transform):
    """Chain multiple transformations"""
    
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms
    
    def fit(self, data):
        current_data = data
        for transform in self.transforms:
            transform.fit(current_data)
            current_data = transform.transform(current_data)
        self.fitted = True
    
    def transform(self, data):
        current_data = data
        for transform in self.transforms:
            current_data = transform.transform(current_data)
        return current_data
    
    def inverse_transform(self, data):
        current_data = data
        for transform in reversed(self.transforms):
            current_data = transform.inverse_transform(current_data)
        return current_data

# Test custom transformations
skewed_data = torch.exp(torch.randn(1000))

power_transform = PowerTransform()
transformed_data = power_transform.fit_transform(skewed_data)
recovered_data = power_transform.inverse_transform(transformed_data)

print(f"Custom power transform lambda: {power_transform.lambda_param:.3f}")
print(f"Original skewness: {((skewed_data - skewed_data.mean()) / skewed_data.std()).pow(3).mean():.3f}")
print(f"Transformed skewness: {((transformed_data - transformed_data.mean()) / transformed_data.std()).pow(3).mean():.3f}")
print(f"Recovery error: {torch.mean(torch.abs(skewed_data - recovered_data)):.6f}")

print("\n=== Batch Transformations ===")

def batch_transform(batch_data, transform_fn, **kwargs):
    """Apply transformation to batch of data"""
    if batch_data.dim() < 2:
        return transform_fn(batch_data, **kwargs)
    
    batch_size = batch_data.shape[0]
    transformed_batch = []
    
    for i in range(batch_size):
        sample = batch_data[i]
        transformed_sample = transform_fn(sample, **kwargs)
        transformed_batch.append(transformed_sample)
    
    return torch.stack(transformed_batch)

def augmentation_pipeline(data, transformations):
    """Apply sequence of transformations with random parameters"""
    result = data.clone()
    
    for transform_name, params in transformations:
        if transform_name == 'noise':
            noise_std = params.get('std', 0.1)
            result = result + torch.randn_like(result) * noise_std
        
        elif transform_name == 'scale':
            scale_range = params.get('range', (0.8, 1.2))
            scale = torch.rand(1) * (scale_range[1] - scale_range[0]) + scale_range[0]
            result = result * scale
        
        elif transform_name == 'shift':
            shift_range = params.get('range', (-0.1, 0.1))
            shift = torch.rand(1) * (shift_range[1] - shift_range[0]) + shift_range[0]
            result = result + shift
        
        elif transform_name == 'clip':
            min_val = params.get('min', None)
            max_val = params.get('max', None)
            if min_val is not None or max_val is not None:
                result = torch.clamp(result, min_val, max_val)
    
    return result

# Test batch transformations
batch_data = torch.randn(32, 100)

# Apply log transform to each sample
batch_log = batch_transform(torch.abs(batch_data) + 1, log_transform)
print(f"Batch shape: {batch_data.shape}")
print(f"Batch log transform shape: {batch_log.shape}")

# Test augmentation pipeline
augmentation_config = [
    ('noise', {'std': 0.05}),
    ('scale', {'range': (0.9, 1.1)}),
    ('shift', {'range': (-0.05, 0.05)}),
    ('clip', {'min': -3, 'max': 3})
]

sample_data = torch.randn(100)
augmented_data = augmentation_pipeline(sample_data, augmentation_config)

print(f"Original data range: [{sample_data.min():.3f}, {sample_data.max():.3f}]")
print(f"Augmented data range: [{augmented_data.min():.3f}, {augmented_data.max():.3f}]")

print("\n=== Transformation Best Practices ===")

print("Transformation Guidelines:")
print("1. Choose transformations based on data distribution and downstream task")
print("2. Always validate that transformations preserve important information")
print("3. Consider invertibility when transformation needs to be reversed")
print("4. Apply same transformations to train/val/test consistently")
print("5. Monitor for numerical stability and edge cases")
print("6. Document transformation parameters for reproducibility")
print("7. Consider computational efficiency for large-scale data")

print("\nCommon Use Cases:")
print("- Log transform: Reduce skewness, handle exponential growth")
print("- Power transform: Normalize distributions, reduce outlier impact")
print("- Cyclical encoding: Handle periodic features (time, angles)")
print("- Frequency domain: Signal processing, noise reduction")
print("- Rank transform: Non-parametric, robust to outliers")
print("- Activation functions: Introduce non-linearity, bound ranges")

print("\nValidation Checklist:")
print("□ Transformation preserves semantic meaning")
print("□ Distribution is more suitable for the task")
print("□ No numerical instabilities (NaN, inf)")
print("□ Invertible if needed")
print("□ Consistent across data splits")
print("□ Computationally efficient")
print("□ Parameters are documented")

print("\n=== Data Transformation Operations Complete ===") 