#!/usr/bin/env python3
"""PyTorch Convolutional Layers - Conv1d, Conv2d, Conv3d, padding, stride"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== Convolutional Layers Overview ===")

print("Convolutional layers provide:")
print("1. Local feature detection through kernels")
print("2. Translation equivariance")
print("3. Parameter sharing across spatial dimensions")
print("4. Efficient processing of spatial data")
print("5. Hierarchical feature learning")

print("\n=== 1D Convolution (Conv1d) ===")

# Basic 1D convolution
conv1d_basic = nn.Conv1d(
    in_channels=1,     # Input channels
    out_channels=3,    # Output channels (number of filters)
    kernel_size=3      # Kernel size
)

# 1D input: (batch_size, channels, length)
input_1d = torch.randn(2, 1, 10)  # 2 samples, 1 channel, length 10
output_1d = conv1d_basic(input_1d)

print(f"Conv1d input shape: {input_1d.shape}")
print(f"Conv1d output shape: {output_1d.shape}")
print(f"Conv1d parameters: {sum(p.numel() for p in conv1d_basic.parameters())}")

# Output size calculation: (L_in + 2*padding - kernel_size) / stride + 1
print(f"Expected output length: {(10 + 2*0 - 3) // 1 + 1} = {output_1d.shape[2]}")

# 1D convolution with different parameters
conv1d_params = nn.Conv1d(
    in_channels=4,
    out_channels=8,
    kernel_size=5,
    stride=2,
    padding=2,
    bias=True
)

input_1d_multi = torch.randn(3, 4, 20)
output_1d_multi = conv1d_params(input_1d_multi)

print(f"\nConv1d with parameters:")
print(f"  Input: {input_1d_multi.shape}")
print(f"  Kernel size: 5, Stride: 2, Padding: 2")
print(f"  Output: {output_1d_multi.shape}")
print(f"  Weight shape: {conv1d_params.weight.shape}")
print(f"  Bias shape: {conv1d_params.bias.shape}")

print("\n=== 2D Convolution (Conv2d) ===")

# Basic 2D convolution
conv2d_basic = nn.Conv2d(
    in_channels=3,     # RGB input
    out_channels=16,   # 16 feature maps
    kernel_size=3      # 3x3 kernel
)

# 2D input: (batch_size, channels, height, width)
input_2d = torch.randn(4, 3, 32, 32)  # 4 images, 3 channels, 32x32
output_2d = conv2d_basic(input_2d)

print(f"Conv2d input shape: {input_2d.shape}")
print(f"Conv2d output shape: {output_2d.shape}")
print(f"Conv2d weight shape: {conv2d_basic.weight.shape}")

# Different kernel sizes
kernel_sizes = [1, 3, 5, 7]
for ks in kernel_sizes:
    conv = nn.Conv2d(3, 16, kernel_size=ks, padding=ks//2)  # Same padding
    out = conv(input_2d)
    print(f"Kernel {ks}x{ks}: {input_2d.shape} -> {out.shape}")

print("\n=== Padding Types ===")

# No padding
conv2d_no_pad = nn.Conv2d(3, 16, kernel_size=3, padding=0)
out_no_pad = conv2d_no_pad(input_2d)

# Same padding (preserves spatial dimensions)
conv2d_same_pad = nn.Conv2d(3, 16, kernel_size=3, padding=1)
out_same_pad = conv2d_same_pad(input_2d)

# Custom padding
conv2d_custom_pad = nn.Conv2d(3, 16, kernel_size=3, padding=2)
out_custom_pad = conv2d_custom_pad(input_2d)

print(f"No padding (0): {input_2d.shape} -> {out_no_pad.shape}")
print(f"Same padding (1): {input_2d.shape} -> {out_same_pad.shape}")
print(f"Custom padding (2): {input_2d.shape} -> {out_custom_pad.shape}")

# Asymmetric padding
conv2d_asym = nn.Conv2d(3, 16, kernel_size=3, padding=(1, 2))  # (height, width)
out_asym = conv2d_asym(input_2d)
print(f"Asymmetric padding (1,2): {input_2d.shape} -> {out_asym.shape}")

print("\n=== Stride and Dilation ===")

# Different strides
strides = [1, 2, 3]
for stride in strides:
    conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=stride)
    out = conv(input_2d)
    print(f"Stride {stride}: {input_2d.shape} -> {out.shape}")

# Dilation (atrous convolution)
dilations = [1, 2, 3]
for dilation in dilations:
    # Adjust padding for dilation
    padding = dilation
    conv = nn.Conv2d(3, 16, kernel_size=3, padding=padding, dilation=dilation)
    out = conv(input_2d)
    print(f"Dilation {dilation}: {input_2d.shape} -> {out.shape}")

print("\n=== Grouped Convolution ===")

# Standard convolution
conv_standard = nn.Conv2d(8, 16, kernel_size=3, padding=1)
standard_params = sum(p.numel() for p in conv_standard.parameters())

# Grouped convolution
conv_grouped = nn.Conv2d(8, 16, kernel_size=3, padding=1, groups=2)
grouped_params = sum(p.numel() for p in conv_grouped.parameters())

# Depthwise convolution (groups = in_channels)
conv_depthwise = nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8)
depthwise_params = sum(p.numel() for p in conv_depthwise.parameters())

input_grouped = torch.randn(2, 8, 16, 16)

print(f"Standard conv parameters: {standard_params}")
print(f"Grouped conv parameters: {grouped_params}")
print(f"Depthwise conv parameters: {depthwise_params}")

print(f"Standard output: {conv_standard(input_grouped).shape}")
print(f"Grouped output: {conv_grouped(input_grouped).shape}")
print(f"Depthwise output: {conv_depthwise(input_grouped).shape}")

print("\n=== 3D Convolution (Conv3d) ===")

# 3D convolution for video/volumetric data
conv3d_basic = nn.Conv3d(
    in_channels=1,
    out_channels=8,
    kernel_size=3
)

# 3D input: (batch_size, channels, depth, height, width)
input_3d = torch.randn(2, 1, 16, 32, 32)  # 2 volumes, 1 channel, 16x32x32
output_3d = conv3d_basic(input_3d)

print(f"Conv3d input shape: {input_3d.shape}")
print(f"Conv3d output shape: {output_3d.shape}")
print(f"Conv3d weight shape: {conv3d_basic.weight.shape}")

# 3D convolution with different kernel sizes
conv3d_varied = nn.Conv3d(
    in_channels=3,
    out_channels=16,
    kernel_size=(3, 5, 5),  # Different sizes for each dimension
    padding=(1, 2, 2)
)

input_3d_rgb = torch.randn(1, 3, 8, 64, 64)
output_3d_varied = conv3d_varied(input_3d_rgb)
print(f"Varied 3D conv: {input_3d_rgb.shape} -> {output_3d_varied.shape}")

print("\n=== Transposed Convolution ===")

# Transposed convolution (deconvolution)
conv_transpose_2d = nn.ConvTranspose2d(
    in_channels=16,
    out_channels=3,
    kernel_size=3,
    stride=2,
    padding=1,
    output_padding=1
)

input_transpose = torch.randn(2, 16, 16, 16)
output_transpose = conv_transpose_2d(input_transpose)

print(f"ConvTranspose2d: {input_transpose.shape} -> {output_transpose.shape}")

# 1D transposed convolution
conv_transpose_1d = nn.ConvTranspose1d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1)
input_1d_transpose = torch.randn(3, 8, 10)
output_1d_transpose = conv_transpose_1d(input_1d_transpose)
print(f"ConvTranspose1d: {input_1d_transpose.shape} -> {output_1d_transpose.shape}")

# 3D transposed convolution
conv_transpose_3d = nn.ConvTranspose3d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
input_3d_transpose = torch.randn(1, 8, 8, 16, 16)
output_3d_transpose = conv_transpose_3d(input_3d_transpose)
print(f"ConvTranspose3d: {input_3d_transpose.shape} -> {output_3d_transpose.shape}")

print("\n=== Functional Convolutions ===")

# Using functional interface
weight_2d = torch.randn(16, 3, 3, 3)  # (out_channels, in_channels, H, W)
bias_2d = torch.randn(16)

# Functional 2D convolution
output_functional = F.conv2d(input_2d, weight_2d, bias_2d, stride=1, padding=1)
print(f"Functional conv2d: {input_2d.shape} -> {output_functional.shape}")

# 1D functional convolution
weight_1d = torch.randn(8, 4, 5)  # (out_channels, in_channels, kernel_size)
input_1d_func = torch.randn(2, 4, 20)
output_1d_func = F.conv1d(input_1d_func, weight_1d, stride=2, padding=2)
print(f"Functional conv1d: {input_1d_func.shape} -> {output_1d_func.shape}")

print("\n=== Custom Convolution Blocks ===")

class ConvBlock2d(nn.Module):
    """Standard convolution block with normalization and activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, padding=None, activation=True, norm=True):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride, padding, bias=not norm)
        
        self.norm = nn.BatchNorm2d(out_channels) if norm else None
        self.activation = nn.ReLU(inplace=True) if activation else None
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x

# Test custom blocks
conv_block = ConvBlock2d(3, 64, kernel_size=7, stride=2, padding=3)
dsconv = DepthwiseSeparableConv(64, 128)

test_input = torch.randn(1, 3, 224, 224)
block_output = conv_block(test_input)
dsconv_output = dsconv(block_output)

print(f"ConvBlock: {test_input.shape} -> {block_output.shape}")
print(f"DepthwiseSeparable: {block_output.shape} -> {dsconv_output.shape}")

# Parameter comparison
standard_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
standard_params = sum(p.numel() for p in standard_conv.parameters())
dsconv_params = sum(p.numel() for p in dsconv.parameters())

print(f"Standard conv parameters: {standard_params}")
print(f"Depthwise separable parameters: {dsconv_params}")
print(f"Parameter reduction: {standard_params / dsconv_params:.2f}x")

print("\n=== Dilated Convolution Networks ===")

class DilatedConvBlock(nn.Module):
    """Dilated convolution block for capturing multi-scale features"""
    def __init__(self, in_channels, out_channels, dilations=[1, 2, 4, 8]):
        super().__init__()
        
        self.convs = nn.ModuleList()
        for dilation in dilations:
            padding = dilation  # For kernel_size=3
            self.convs.append(
                nn.Conv2d(in_channels, out_channels // len(dilations), 
                         kernel_size=3, padding=padding, dilation=dilation)
            )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        outputs = []
        for conv in self.convs:
            outputs.append(conv(x))
        
        x = torch.cat(outputs, dim=1)
        x = self.bn(x)
        x = self.relu(x)
        return x

dilated_block = DilatedConvBlock(64, 256)
dilated_input = torch.randn(1, 64, 32, 32)
dilated_output = dilated_block(dilated_input)

print(f"Dilated conv block: {dilated_input.shape} -> {dilated_output.shape}")

print("\n=== Convolution Output Size Calculation ===")

def calculate_conv_output_size(input_size, kernel_size, stride=1, padding=0, dilation=1):
    """Calculate convolution output size"""
    if isinstance(input_size, int):
        input_size = [input_size]
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * len(input_size)
    if isinstance(stride, int):
        stride = [stride] * len(input_size)
    if isinstance(padding, int):
        padding = [padding] * len(input_size)
    if isinstance(dilation, int):
        dilation = [dilation] * len(input_size)
    
    output_size = []
    for i in range(len(input_size)):
        effective_kernel = dilation[i] * (kernel_size[i] - 1) + 1
        out = (input_size[i] + 2 * padding[i] - effective_kernel) // stride[i] + 1
        output_size.append(out)
    
    return output_size[0] if len(output_size) == 1 else output_size

def calculate_transpose_conv_output_size(input_size, kernel_size, stride=1, padding=0, output_padding=0, dilation=1):
    """Calculate transposed convolution output size"""
    if isinstance(input_size, int):
        input_size = [input_size]
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * len(input_size)
    if isinstance(stride, int):
        stride = [stride] * len(input_size)
    if isinstance(padding, int):
        padding = [padding] * len(input_size)
    if isinstance(output_padding, int):
        output_padding = [output_padding] * len(input_size)
    if isinstance(dilation, int):
        dilation = [dilation] * len(input_size)
    
    output_size = []
    for i in range(len(input_size)):
        effective_kernel = dilation[i] * (kernel_size[i] - 1) + 1
        out = (input_size[i] - 1) * stride[i] - 2 * padding[i] + effective_kernel + output_padding[i]
        output_size.append(out)
    
    return output_size[0] if len(output_size) == 1 else output_size

# Test size calculations
print("Convolution output size calculations:")
print(f"Input 32, kernel 3, stride 1, padding 1: {calculate_conv_output_size(32, 3, 1, 1)}")
print(f"Input 32, kernel 3, stride 2, padding 1: {calculate_conv_output_size(32, 3, 2, 1)}")
print(f"Input [32,32], kernel 5, stride 2, padding 2: {calculate_conv_output_size([32,32], 5, 2, 2)}")

print("\nTransposed convolution output size calculations:")
print(f"Input 16, kernel 3, stride 2, padding 1, output_padding 1: {calculate_transpose_conv_output_size(16, 3, 2, 1, 1)}")

print("\n=== Convolution Performance Analysis ===")

def benchmark_convolutions():
    """Benchmark different convolution configurations"""
    import time
    
    input_data = torch.randn(8, 64, 32, 32)
    if torch.cuda.is_available():
        input_data = input_data.cuda()
    
    configs = [
        ("Standard 3x3", nn.Conv2d(64, 128, 3, padding=1)),
        ("Depthwise 3x3", nn.Conv2d(64, 64, 3, padding=1, groups=64)),
        ("1x1 Pointwise", nn.Conv2d(64, 128, 1)),
        ("5x5 Large", nn.Conv2d(64, 128, 5, padding=2)),
        ("Dilated 3x3", nn.Conv2d(64, 128, 3, padding=2, dilation=2)),
    ]
    
    results = {}
    iterations = 50
    
    for name, conv in configs:
        if torch.cuda.is_available():
            conv = conv.cuda()
        
        # Warmup
        for _ in range(5):
            _ = conv(input_data)
        
        # Benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(iterations):
            output = conv(input_data)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        elapsed = time.time() - start_time
        results[name] = elapsed / iterations
        
        params = sum(p.numel() for p in conv.parameters())
        print(f"{name}: {elapsed/iterations*1000:.3f}ms, {params} params, output: {output.shape}")
    
    return results

print("Convolution performance benchmark:")
if torch.cuda.is_available():
    benchmark_results = benchmark_convolutions()
else:
    print("CUDA not available, skipping benchmark")

print("\n=== Convolution Best Practices ===")

print("Convolution Layer Guidelines:")
print("1. Use appropriate padding to control output size")
print("2. Consider depthwise separable convs for efficiency")
print("3. Use dilated convolutions for larger receptive fields")
print("4. Group convolutions reduce parameters")
print("5. Stride > 1 for downsampling, transposed conv for upsampling")
print("6. Initialize weights properly (He initialization for ReLU)")
print("7. Consider 1x1 convs for channel mixing")

print("\nArchitecture Design:")
print("- Start with small kernels (3x3) and stack them")
print("- Use pooling or strided convs for downsampling")
print("- Increase channels as spatial dimensions decrease")
print("- Add skip connections for deep networks")
print("- Use batch normalization after convolutions")
print("- Consider attention mechanisms for long-range dependencies")

print("\nPerformance Tips:")
print("- Prefer 3x3 convolutions over larger kernels")
print("- Use depthwise separable convs for mobile deployment")
print("- Group convolutions for parameter efficiency")
print("- Optimize for your target hardware")
print("- Use mixed precision for faster training")
print("- Profile memory usage and computation time")

print("\nCommon Pitfalls:")
print("- Forgetting to adjust padding for kernel size")
print("- Not considering receptive field size")
print("- Inefficient parameter usage")
print("- Gradient vanishing in very deep networks")
print("- Memory overflow with large feature maps")

print("\n=== Convolutional Layers Complete ===") 