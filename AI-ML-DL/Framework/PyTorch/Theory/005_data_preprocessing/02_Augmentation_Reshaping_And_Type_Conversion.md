# Augmentation, Reshaping, and Type Conversion

## Table of Contents

- [Tensor-Level Data Augmentation](#tensor-level-data-augmentation)
- [Reshaping for Different Model Inputs](#reshaping-for-different-model-inputs)
- [Data Type Conversions](#data-type-conversions)

---

## Tensor-Level Data Augmentation

**Data augmentation** increases dataset diversity by applying random transformations. Tensor-level augmentation operates directly on PyTorch tensors without external libraries.

### Geometric Transformations

Geometric augmentations include flipping, rotation, scaling, and translation. Use `F.affine_grid` and `F.grid_sample` for differentiable spatial transforms.

```python
import torch
import torch.nn.functional as F
import math
import random

def random_flip(tensor, p=0.5, dim=-1):
    if random.random() < p:
        return torch.flip(tensor, dims=[dim])
    return tensor

def random_rotation(tensor, max_angle=30):
    angle = random.uniform(-max_angle, max_angle)
    angle_rad = math.radians(angle)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    rotation_matrix = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0]
    ], dtype=tensor.dtype, device=tensor.device)
    grid = F.affine_grid(rotation_matrix.unsqueeze(0), tensor.unsqueeze(0).shape, align_corners=False)
    rotated = F.grid_sample(tensor.unsqueeze(0), grid, align_corners=False, mode='bilinear', padding_mode='zeros')
    return rotated.squeeze(0)

def random_scale(tensor, scale_range=(0.8, 1.2)):
    scale = random.uniform(*scale_range)
    scale_matrix = torch.tensor([
        [scale, 0, 0],
        [0, scale, 0]
    ], dtype=tensor.dtype, device=tensor.device)
    grid = F.affine_grid(scale_matrix.unsqueeze(0), tensor.unsqueeze(0).shape, align_corners=False)
    return F.grid_sample(tensor.unsqueeze(0), grid, align_corners=False, mode='bilinear', padding_mode='zeros').squeeze(0)
```

### Color and Intensity Augmentations

For RGB images in [0, 1], brightness, contrast, saturation, and hue adjustments modify appearance while preserving semantic content.

```python
def random_brightness(tensor, brightness_range=(-0.2, 0.2)):
    brightness_factor = random.uniform(*brightness_range)
    return torch.clamp(tensor + brightness_factor, 0, 1)

def random_contrast(tensor, contrast_range=(0.8, 1.2)):
    contrast_factor = random.uniform(*contrast_range)
    mean = tensor.mean(dim=[-2, -1], keepdim=True)
    return torch.clamp((tensor - mean) * contrast_factor + mean, 0, 1)
```

### Noise Injection

**Gaussian noise** adds random values from a normal distribution. **Salt-and-pepper noise** randomly sets pixels to 0 or 1. **Speckle noise** is multiplicative.

```python
def add_gaussian_noise(tensor, noise_std=0.1):
    noise = torch.randn_like(tensor) * noise_std
    return torch.clamp(tensor + noise, 0, 1)

def add_salt_pepper_noise(tensor, noise_prob=0.05):
    noise_mask = torch.rand_like(tensor) < noise_prob
    salt_mask = torch.rand_like(tensor) < 0.5
    noisy_tensor = tensor.clone()
    noisy_tensor[noise_mask & salt_mask] = 1.0
    noisy_tensor[noise_mask & ~salt_mask] = 0.0
    return noisy_tensor

def add_speckle_noise(tensor, noise_std=0.1):
    noise = torch.randn_like(tensor) * noise_std + 1
    return torch.clamp(tensor * noise, 0, 1)
```

### Cutout and Mixup

**Random cutout** zeros rectangular regions. **Mixup** blends two samples linearly. **CutMix** replaces a region of one image with the corresponding region of another.

```python
def random_cutout(tensor, cutout_size=16, n_holes=1):
    h, w = tensor.shape[-2:]
    mask = torch.ones_like(tensor)
    for _ in range(n_holes):
        y = random.randint(0, h - cutout_size)
        x = random.randint(0, w - cutout_size)
        mask[..., y:y+cutout_size, x:x+cutout_size] = 0
    return tensor * mask

def mixup(tensor1, tensor2, alpha=0.2):
    lam = random.betavariate(alpha, alpha) if alpha > 0 else 1
    mixed_tensor = lam * tensor1 + (1 - lam) * tensor2
    return mixed_tensor, lam

def cutmix(tensor1, tensor2, alpha=1.0):
    lam = random.betavariate(alpha, alpha) if alpha > 0 else 1
    h, w = tensor1.shape[-2:]
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    cx = random.randint(0, w)
    cy = random.randint(0, h)
    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, w)
    bby2 = min(cy + cut_h // 2, h)
    mixed_tensor = tensor1.clone()
    mixed_tensor[..., bby1:bby2, bbx1:bbx2] = tensor2[..., bby1:bby2, bbx1:bbx2]
    return mixed_tensor, 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
```

### Text Augmentation

Token-level augmentation for NLP: dropout (mask tokens), replacement (random tokens), and insertion.

```python
def token_dropout(tokens, dropout_prob=0.1, mask_token_id=0):
    mask = torch.rand(tokens.shape) < dropout_prob
    augmented_tokens = tokens.clone()
    augmented_tokens[mask] = mask_token_id
    return augmented_tokens

def token_replacement(tokens, vocab_size, replacement_prob=0.1):
    mask = torch.rand(tokens.shape) < replacement_prob
    random_tokens = torch.randint(1, vocab_size, tokens.shape)
    return torch.where(mask, random_tokens, tokens)
```

### Augmentation Pipeline

Compose multiple augmentations with per-transform application probability.

```python
def random_apply(transform, p=0.5):
    def wrapper(tensor):
        if random.random() < p:
            return transform(tensor)
        return tensor
    return wrapper

class AugmentationPipeline:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, tensor):
        for transform in self.transforms:
            tensor = transform(tensor)
        return tensor
```

---

## Reshaping for Different Model Inputs

Different model architectures expect different tensor layouts. **Reshaping** reorganizes dimensions without changing underlying data (when possible).

### Basic Reshaping Operations

**view** requires contiguous memory; **reshape** handles non-contiguous tensors by copying when necessary. **squeeze** and **unsqueeze** remove or add size-1 dimensions.

```python
data = torch.randn(2, 3, 4, 5)
viewed = data.view(2, -1)
reshaped = data.reshape(2, 60)

tensor_with_ones = torch.randn(1, 3, 1, 4, 1)
squeezed_all = tensor_with_ones.squeeze()
squeezed_dim0 = tensor_with_ones.squeeze(0)

original = torch.randn(3, 4)
unsqueezed_0 = original.unsqueeze(0)
unsqueezed_neg1 = original.unsqueeze(-1)
```

### Concatenation and Stacking

**torch.cat** concatenates along an existing dimension. **torch.stack** creates a new dimension.

```python
tensor1 = torch.randn(2, 3, 4)
tensor2 = torch.randn(2, 3, 4)
cat_dim0 = torch.cat([tensor1, tensor2], dim=0)
cat_dim1 = torch.cat([tensor1, tensor2], dim=1)
stack_dim0 = torch.stack([tensor1, tensor2], dim=0)
```

### Format Conversions: NCHW vs NHWC

**NCHW** (batch, channels, height, width) is PyTorch default for images. **NHWC** (batch, height, width, channels) is used by TensorFlow and some APIs.

```python
def convert_channel_format(tensor, from_format, to_format):
    if from_format == to_format:
        return tensor
    if from_format == "NCHW" and to_format == "NHWC":
        return tensor.permute(0, 2, 3, 1)
    elif from_format == "NHWC" and to_format == "NCHW":
        return tensor.permute(0, 3, 1, 2)
    elif from_format == "CHW" and to_format == "HWC":
        return tensor.permute(1, 2, 0)
    elif from_format == "HWC" and to_format == "CHW":
        return tensor.permute(2, 0, 1)
    raise ValueError(f"Unsupported conversion: {from_format} -> {to_format}")

nchw_tensor = torch.randn(4, 3, 32, 32)
nhwc_tensor = convert_channel_format(nchw_tensor, "NCHW", "NHWC")
```

### Sequence Data Formats

| Format | Shape | Use Case |
|--------|-------|----------|
| RNN/LSTM | (batch, seq_len, hidden) | Recurrent models |
| Attention | (seq_len, batch, hidden) | Transformer encoders |
| CNN | (batch, 1, seq_len, hidden) | 1D convolutions |
| Flattened | (batch, seq_len * hidden) | Linear layers |

```python
batch_size, seq_len, hidden_size = 4, 10, 256
sequences = torch.randn(batch_size, seq_len, hidden_size)

flattened_seq = sequences.view(batch_size, -1)
cnn_input = sequences.unsqueeze(1)
attention_input = sequences.transpose(0, 1)

lengths = torch.tensor([10, 8, 6, 9])
packed_seq = torch.nn.utils.rnn.pack_padded_sequence(sequences, lengths, batch_first=True, enforce_sorted=False)
```

### Time Series Windowing

Sliding windows convert time series into supervised samples for RNN, CNN, or transformer inputs.

```python
def create_windows(data, window_size, stride=1):
    windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data[i:i + window_size])
    return torch.stack(windows)

time_series = torch.randn(100, 5)
windowed_data = create_windows(time_series, window_size=10, stride=1)
rnn_format = windowed_data
cnn_format = windowed_data.transpose(1, 2)
transformer_format = windowed_data.transpose(0, 1)
```

### Padding Variable-Length Sequences

```python
def pad_batch(tensors, pad_value=0):
    max_shape = [max(t.shape[dim] for t in tensors) for dim in range(len(tensors[0].shape))]
    padded_tensors = []
    for tensor in tensors:
        pad_widths = []
        for dim in range(len(tensor.shape)):
            pad_width = max_shape[dim] - tensor.shape[dim]
            pad_widths.extend([0, pad_width])
        pad_widths = pad_widths[::-1]
        padded = F.pad(tensor, pad_widths, value=pad_value)
        padded_tensors.append(padded)
    return torch.stack(padded_tensors)
```

---

## Data Type Conversions

**Data type conversions** between Python, NumPy, and PyTorch are common. Correct handling prevents overflow, precision loss, and device mismatches.

### Basic PyTorch Type Conversions

```python
int_tensor = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
float_tensor = torch.tensor([1.1, 2.2, 3.3, 4.4], dtype=torch.float32)

int_to_float = int_tensor.float()
float_to_int = float_tensor.int()
converted_to_double = int_tensor.to(torch.float64)
converted_to_long = float_tensor.to(torch.int64)
```

### Precision Conversions

| Type | Bytes | Typical Use |
|------|-------|--------------|
| float32 | 4 | Default for training |
| float16 | 2 | Mixed precision, memory savings |
| float64 | 8 | High-precision computation |

```python
data = torch.randn(3, 3)
data_fp16 = data.half()
data_fp64 = data.double()

precise_value = torch.tensor(3.141592653589793)
fp16_value = precise_value.half()
back_to_fp32 = fp16_value.float()
```

### Integer Overflow Handling

Integer conversion can overflow silently. Use clamping for safe conversion.

```python
def safe_int_convert(tensor, target_dtype):
    if target_dtype == torch.int8:
        min_val, max_val = -128, 127
    elif target_dtype == torch.uint8:
        min_val, max_val = 0, 255
    elif target_dtype == torch.int16:
        min_val, max_val = -32768, 32767
    else:
        return tensor.to(target_dtype)
    clamped = torch.clamp(tensor, min_val, max_val)
    return clamped.to(target_dtype)
```

### NumPy Interoperability

**torch.from_numpy** creates a tensor that shares memory with the NumPy array. **tensor.numpy()** returns a NumPy view when the tensor is on CPU.

```python
import numpy as np

torch_tensor = torch.randn(3, 3)
numpy_array = torch_tensor.numpy()

numpy_int_array = np.array([1, 2, 3, 4], dtype=np.int64)
torch_from_numpy = torch.from_numpy(numpy_int_array)
```

### Device and Type Combined Conversion

```python
cpu_tensor = torch.randn(3, 3)
if torch.cuda.is_available():
    gpu_tensor = cpu_tensor.cuda()
    cpu_again = gpu_tensor.cpu()
    gpu_half = cpu_tensor.cuda().half()

target_device = 'cuda' if torch.cuda.is_available() else 'cpu'
converted_tensor = cpu_tensor.to(device=target_device, dtype=torch.float16)
```

### Image-Specific Conversions

For images, uint8 to float32 typically involves division by 255. The reverse requires clamping before conversion.

```python
def normalize_and_convert(tensor, target_type):
    if tensor.dtype == torch.uint8 and target_type.is_floating_point:
        return tensor.float() / 255.0
    elif tensor.dtype.is_floating_point and target_type == torch.uint8:
        return (tensor * 255).clamp(0, 255).to(torch.uint8)
    return tensor.to(target_type)
```

### Automatic Type Promotion

Operations between different dtypes promote to the more general type (e.g., int32 + float32 yields float32).

```python
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
float_tensor = torch.tensor([1.1, 2.2, 3.3], dtype=torch.float32)
result = int_tensor + float_tensor
```
