#!/usr/bin/env python3
"""PyTorch Pooling Layers Syntax - All pooling operations and syntax"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== Pooling Layers Overview ===")

print("Pooling layers provide:")
print("1. Spatial dimension reduction")
print("2. Translation invariance")
print("3. Computational efficiency")
print("4. Feature summarization")
print("5. Overfitting reduction")

print("\n=== 1D Pooling Operations ===")

# MaxPool1d
maxpool1d = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
input_1d = torch.randn(2, 4, 20)  # (batch, channels, length)
output_maxpool1d = maxpool1d(input_1d)

print(f"MaxPool1d input: {input_1d.shape}")
print(f"MaxPool1d output: {output_maxpool1d.shape}")

# AvgPool1d
avgpool1d = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
output_avgpool1d = avgpool1d(input_1d)
print(f"AvgPool1d output: {output_avgpool1d.shape}")

# AdaptiveMaxPool1d - outputs fixed size regardless of input
adaptive_maxpool1d = nn.AdaptiveMaxPool1d(output_size=5)
output_adaptive_max1d = adaptive_maxpool1d(input_1d)
print(f"AdaptiveMaxPool1d output: {output_adaptive_max1d.shape}")

# AdaptiveAvgPool1d
adaptive_avgpool1d = nn.AdaptiveAvgPool1d(output_size=3)
output_adaptive_avg1d = adaptive_avgpool1d(input_1d)
print(f"AdaptiveAvgPool1d output: {output_adaptive_avg1d.shape}")

print("\n=== 2D Pooling Operations ===")

# Sample 2D input
input_2d = torch.randn(4, 3, 32, 32)  # (batch, channels, height, width)

# MaxPool2d with different configurations
maxpool2d_basic = nn.MaxPool2d(kernel_size=2)
maxpool2d_stride = nn.MaxPool2d(kernel_size=3, stride=2)
maxpool2d_padding = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

output_max2d_basic = maxpool2d_basic(input_2d)
output_max2d_stride = maxpool2d_stride(input_2d)
output_max2d_padding = maxpool2d_padding(input_2d)

print(f"MaxPool2d basic (2x2): {input_2d.shape} -> {output_max2d_basic.shape}")
print(f"MaxPool2d stride (3x3, s=2): {input_2d.shape} -> {output_max2d_stride.shape}")
print(f"MaxPool2d padding (3x3, s=1, p=1): {input_2d.shape} -> {output_max2d_padding.shape}")

# AvgPool2d
avgpool2d = nn.AvgPool2d(kernel_size=2, stride=2)
output_avg2d = avgpool2d(input_2d)
print(f"AvgPool2d: {input_2d.shape} -> {output_avg2d.shape}")

# Asymmetric kernel sizes
asymmetric_pool = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))
output_asymmetric = asymmetric_pool(input_2d)
print(f"Asymmetric pool (2x4): {input_2d.shape} -> {output_asymmetric.shape}")

print("\n=== Adaptive Pooling 2D ===")

# AdaptiveMaxPool2d - useful for classification networks
adaptive_maxpool2d = nn.AdaptiveMaxPool2d(output_size=(7, 7))
output_adaptive_max2d = adaptive_maxpool2d(input_2d)
print(f"AdaptiveMaxPool2d (7x7): {input_2d.shape} -> {output_adaptive_max2d.shape}")

# AdaptiveAvgPool2d - common in modern architectures
adaptive_avgpool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # Global average pooling
output_adaptive_avg2d = adaptive_avgpool2d(input_2d)
print(f"AdaptiveAvgPool2d (1x1): {input_2d.shape} -> {output_adaptive_avg2d.shape}")

# Different output sizes
output_sizes = [(1, 1), (2, 2), (3, 3), (5, 7)]
for size in output_sizes:
    adaptive_pool = nn.AdaptiveAvgPool2d(output_size=size)
    output = adaptive_pool(input_2d)
    print(f"AdaptiveAvgPool2d {size}: {input_2d.shape} -> {output.shape}")

print("\n=== 3D Pooling Operations ===")

# Sample 3D input (video/volumetric data)
input_3d = torch.randn(2, 1, 16, 32, 32)  # (batch, channels, depth, height, width)

# MaxPool3d
maxpool3d = nn.MaxPool3d(kernel_size=2)
output_max3d = maxpool3d(input_3d)
print(f"MaxPool3d: {input_3d.shape} -> {output_max3d.shape}")

# AvgPool3d
avgpool3d = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=2)
output_avg3d = avgpool3d(input_3d)
print(f"AvgPool3d: {input_3d.shape} -> {output_avg3d.shape}")

# AdaptiveMaxPool3d
adaptive_maxpool3d = nn.AdaptiveMaxPool3d(output_size=(8, 16, 16))
output_adaptive_max3d = adaptive_maxpool3d(input_3d)
print(f"AdaptiveMaxPool3d: {input_3d.shape} -> {output_adaptive_max3d.shape}")

# AdaptiveAvgPool3d
adaptive_avgpool3d = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
output_adaptive_avg3d = adaptive_avgpool3d(input_3d)
print(f"AdaptiveAvgPool3d: {input_3d.shape} -> {output_adaptive_avg3d.shape}")

print("\n=== Functional Pooling ===")

# Using functional interface
input_func = torch.randn(2, 8, 16, 16)

# Functional max pooling
output_func_max = F.max_pool2d(input_func, kernel_size=2, stride=2)
print(f"F.max_pool2d: {input_func.shape} -> {output_func_max.shape}")

# Functional average pooling
output_func_avg = F.avg_pool2d(input_func, kernel_size=2, stride=2)
print(f"F.avg_pool2d: {input_func.shape} -> {output_func_avg.shape}")

# Functional adaptive pooling
output_func_adaptive = F.adaptive_avg_pool2d(input_func, output_size=(4, 4))
print(f"F.adaptive_avg_pool2d: {input_func.shape} -> {output_func_adaptive.shape}")

# MaxPool with return indices
output_with_indices, indices = F.max_pool2d(input_func, kernel_size=2, stride=2, return_indices=True)
print(f"Max pool with indices: {output_with_indices.shape}, indices: {indices.shape}")

print("\n=== Unpooling Operations ===")

# MaxUnpool2d - reverses max pooling using indices
maxpool_unpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
maxunpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

input_unpool = torch.randn(1, 1, 4, 4)
print(f"Unpooling input: {input_unpool.shape}")

# Pool with indices
pooled, indices = maxpool_unpool(input_unpool)
print(f"After pooling: {pooled.shape}")

# Unpool
unpooled = maxunpool(pooled, indices)
print(f"After unpooling: {unpooled.shape}")

# Verify sizes match
print(f"Original and unpooled shapes match: {input_unpool.shape == unpooled.shape}")

# Unpooling with output size specification
unpooled_with_size = maxunpool(pooled, indices, output_size=input_unpool.size())
print(f"Unpooled with size spec: {unpooled_with_size.shape}")

print("\n=== Global Pooling ===")

class GlobalPooling(nn.Module):
    """Different global pooling strategies"""
    def __init__(self, pool_type='avg'):
        super().__init__()
        self.pool_type = pool_type
    
    def forward(self, x):
        if self.pool_type == 'avg':
            # Global average pooling
            return F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        elif self.pool_type == 'max':
            # Global max pooling
            return F.adaptive_max_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        elif self.pool_type == 'sum':
            # Global sum pooling
            return x.sum(dim=[2, 3])
        elif self.pool_type == 'std':
            # Global standard deviation pooling
            return x.std(dim=[2, 3])
        elif self.pool_type == 'min':
            # Global min pooling
            return x.view(x.size(0), x.size(1), -1).min(dim=2)[0]
        else:
            raise ValueError(f"Unknown pool type: {self.pool_type}")

# Test global pooling
global_input = torch.randn(3, 64, 8, 8)

for pool_type in ['avg', 'max', 'sum', 'std', 'min']:
    global_pool = GlobalPooling(pool_type)
    global_output = global_pool(global_input)
    print(f"Global {pool_type} pooling: {global_input.shape} -> {global_output.shape}")

print("\n=== Advanced Pooling Techniques ===")

class FractionalMaxPool2d(nn.Module):
    """Fractional max pooling for random downsampling"""
    def __init__(self, kernel_size, output_ratio=0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.output_ratio = output_ratio
    
    def forward(self, x):
        return F.fractional_max_pool2d(
            x, kernel_size=self.kernel_size, 
            output_ratio=self.output_ratio
        )

class LPPool2d(nn.Module):
    """Lp pooling (generalization of max and average pooling)"""
    def __init__(self, norm_type=2, kernel_size=2, stride=None):
        super().__init__()
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
    
    def forward(self, x):
        return F.lp_pool2d(x, norm_type=self.norm_type, 
                          kernel_size=self.kernel_size, 
                          stride=self.stride)

class AdaptiveConcatPool2d(nn.Module):
    """Concatenate adaptive max and average pooling"""
    def __init__(self, output_size=(1, 1)):
        super().__init__()
        self.output_size = output_size
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size)
        self.max_pool = nn.AdaptiveMaxPool2d(output_size)
    
    def forward(self, x):
        avg = self.avg_pool(x)
        max_val = self.max_pool(x)
        return torch.cat([avg, max_val], dim=1)

# Test advanced pooling
adv_input = torch.randn(2, 32, 16, 16)

# Fractional pooling
frac_pool = FractionalMaxPool2d(kernel_size=2, output_ratio=0.7)
frac_output = frac_pool(adv_input)
print(f"Fractional max pool: {adv_input.shape} -> {frac_output.shape}")

# Lp pooling
lp_pool = LPPool2d(norm_type=2, kernel_size=2)
lp_output = lp_pool(adv_input)
print(f"L2 pool: {adv_input.shape} -> {lp_output.shape}")

# Adaptive concat pooling
concat_pool = AdaptiveConcatPool2d(output_size=(1, 1))
concat_output = concat_pool(adv_input)
print(f"Adaptive concat pool: {adv_input.shape} -> {concat_output.shape}")

print("\n=== Pooling with Padding Modes ===")

# Different padding modes for edge handling
input_padding = torch.randn(1, 1, 5, 5)
print(f"Padding test input: {input_padding.shape}")

# Reflection padding
reflect_pad = nn.ReflectionPad2d(1)
input_reflect = reflect_pad(input_padding)
pool_reflect = nn.MaxPool2d(3, stride=1)
output_reflect = pool_reflect(input_reflect)
print(f"Reflection padding: {input_padding.shape} -> {input_reflect.shape} -> {output_reflect.shape}")

# Replication padding
replicate_pad = nn.ReplicationPad2d(1)
input_replicate = replicate_pad(input_padding)
output_replicate = pool_reflect(input_replicate)
print(f"Replication padding: {input_padding.shape} -> {input_replicate.shape} -> {output_replicate.shape}")

print("\n=== Pooling in Different Network Architectures ===")

class CNN_WithPooling(nn.Module):
    """CNN with different pooling strategies"""
    def __init__(self, pooling_strategy='standard'):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        if pooling_strategy == 'standard':
            self.pool1 = nn.MaxPool2d(2)
            self.pool2 = nn.MaxPool2d(2)
            self.pool3 = nn.AdaptiveAvgPool2d((1, 1))
        elif pooling_strategy == 'average':
            self.pool1 = nn.AvgPool2d(2)
            self.pool2 = nn.AvgPool2d(2)
            self.pool3 = nn.AdaptiveAvgPool2d((1, 1))
        elif pooling_strategy == 'mixed':
            self.pool1 = nn.MaxPool2d(2)
            self.pool2 = nn.AvgPool2d(2)
            self.pool3 = AdaptiveConcatPool2d((1, 1))
        
        self.pooling_strategy = pooling_strategy
        self.relu = nn.ReLU()
        
        # Adjust classifier based on pooling strategy
        if pooling_strategy == 'mixed':
            self.classifier = nn.Linear(256, 10)  # Doubled due to concat
        else:
            self.classifier = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

# Test different pooling strategies
pooling_strategies = ['standard', 'average', 'mixed']
test_input = torch.randn(2, 3, 32, 32)

for strategy in pooling_strategies:
    model = CNN_WithPooling(strategy)
    output = model(test_input)
    params = sum(p.numel() for p in model.parameters())
    print(f"{strategy} pooling: {test_input.shape} -> {output.shape}, {params} params")

print("\n=== Pooling Output Size Calculation ===")

def calculate_pool_output_size(input_size, kernel_size, stride=None, padding=0, dilation=1):
    """Calculate pooling output size"""
    if stride is None:
        stride = kernel_size
    
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

# Test size calculations
print("Pooling output size calculations:")
print(f"Input 32, kernel 2, stride 2: {calculate_pool_output_size(32, 2, 2)}")
print(f"Input 32, kernel 3, stride 2, padding 1: {calculate_pool_output_size(32, 3, 2, 1)}")
print(f"Input [32,32], kernel 2: {calculate_pool_output_size([32,32], 2)}")

print("\n=== Pooling Performance Comparison ===")

def benchmark_pooling_operations():
    """Benchmark different pooling operations"""
    import time
    
    input_data = torch.randn(16, 64, 32, 32)
    if torch.cuda.is_available():
        input_data = input_data.cuda()
    
    operations = [
        ("MaxPool2d", nn.MaxPool2d(2)),
        ("AvgPool2d", nn.AvgPool2d(2)),
        ("AdaptiveMaxPool2d", nn.AdaptiveMaxPool2d((16, 16))),
        ("AdaptiveAvgPool2d", nn.AdaptiveAvgPool2d((16, 16))),
        ("Global AvgPool", nn.AdaptiveAvgPool2d((1, 1))),
    ]
    
    iterations = 100
    results = {}
    
    for name, operation in operations:
        if torch.cuda.is_available():
            operation = operation.cuda()
        
        # Warmup
        for _ in range(5):
            _ = operation(input_data)
        
        # Benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(iterations):
            output = operation(input_data)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        elapsed = time.time() - start_time
        results[name] = elapsed / iterations
        
        print(f"{name}: {elapsed/iterations*1000:.3f}ms, output: {output.shape}")
    
    return results

print("Pooling performance benchmark:")
if torch.cuda.is_available():
    benchmark_results = benchmark_pooling_operations()
else:
    print("CUDA not available, skipping benchmark")

print("\n=== Custom Pooling Operations ===")

class AttentionPool2d(nn.Module):
    """Attention-based pooling"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.attention = nn.Conv2d(in_channels, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, out_channels)
    
    def forward(self, x):
        # Compute attention weights
        attention = torch.sigmoid(self.attention(x))
        
        # Apply attention
        weighted = x * attention
        
        # Global pool
        pooled = self.global_pool(weighted)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Final projection
        output = self.fc(pooled)
        
        return output

class SpatialPyramidPool2d(nn.Module):
    """Spatial Pyramid Pooling"""
    def __init__(self, levels=[1, 2, 4]):
        super().__init__()
        self.levels = levels
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        pooled_features = []
        
        for level in self.levels:
            kernel_size = (height // level, width // level)
            stride = kernel_size
            pooled = F.adaptive_avg_pool2d(x, (level, level))
            pooled = pooled.view(batch_size, -1)
            pooled_features.append(pooled)
        
        return torch.cat(pooled_features, dim=1)

# Test custom pooling
custom_input = torch.randn(2, 64, 8, 8)

attention_pool = AttentionPool2d(64, 32)
attention_output = attention_pool(custom_input)
print(f"Attention pooling: {custom_input.shape} -> {attention_output.shape}")

spp = SpatialPyramidPool2d([1, 2, 4])
spp_output = spp(custom_input)
print(f"Spatial pyramid pooling: {custom_input.shape} -> {spp_output.shape}")

print("\n=== Pooling Best Practices ===")

print("Pooling Layer Guidelines:")
print("1. Use max pooling for feature detection tasks")
print("2. Use average pooling for feature averaging")
print("3. Consider adaptive pooling for variable input sizes")
print("4. Global pooling replaces fully connected layers")
print("5. Use appropriate kernel size and stride")
print("6. Consider fractional pooling for regularization")
print("7. Match pooling to your architecture's needs")

print("\nArchitecture Design:")
print("- Max pooling for translation invariance")
print("- Average pooling for smoother downsampling")
print("- Adaptive pooling for flexible architectures")
print("- Global pooling for classification heads")
print("- Consider learned pooling strategies")
print("- Use unpooling for upsampling tasks")

print("\nPerformance Considerations:")
print("- Max pooling is generally faster than average")
print("- Adaptive pooling has slight overhead")
print("- Global pooling reduces parameter count")
print("- Consider memory usage with large kernels")
print("- Use appropriate data types for efficiency")

print("\nCommon Use Cases:")
print("- CNN feature extraction: Max/Average pooling")
print("- Classification networks: Global average pooling")
print("- Segmentation: Careful pooling to preserve details")
print("- Object detection: Multi-scale pooling")
print("- Video analysis: 3D pooling operations")

print("\n=== Pooling Layers Complete ===") 