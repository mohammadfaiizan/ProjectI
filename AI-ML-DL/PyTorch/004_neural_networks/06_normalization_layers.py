#!/usr/bin/env python3
"""PyTorch Normalization Layers - BatchNorm, LayerNorm, GroupNorm, InstanceNorm"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=== Normalization Layers Overview ===")

print("Normalization layers provide:")
print("1. Gradient flow stabilization")
print("2. Faster convergence")
print("3. Reduced internal covariate shift")
print("4. Regularization effects")
print("5. Improved training stability")

print("\n=== Batch Normalization ===")

# BatchNorm1d for fully connected layers
bn1d = nn.BatchNorm1d(64)
input_1d = torch.randn(32, 64)  # (batch_size, features)
output_bn1d = bn1d(input_1d)

print(f"BatchNorm1d input: {input_1d.shape}")
print(f"BatchNorm1d output: {output_bn1d.shape}")
print(f"Running mean shape: {bn1d.running_mean.shape}")
print(f"Running var shape: {bn1d.running_var.shape}")
print(f"Weight shape: {bn1d.weight.shape}")
print(f"Bias shape: {bn1d.bias.shape}")

# Statistics
print(f"Input mean: {input_1d.mean(dim=0)[:5]}")
print(f"Output mean: {output_bn1d.mean(dim=0)[:5]}")
print(f"Input std: {input_1d.std(dim=0)[:5]}")
print(f"Output std: {output_bn1d.std(dim=0)[:5]}")

# BatchNorm2d for convolutional layers
bn2d = nn.BatchNorm2d(32)
input_2d = torch.randn(16, 32, 28, 28)  # (batch, channels, height, width)
output_bn2d = bn2d(input_2d)

print(f"\nBatchNorm2d input: {input_2d.shape}")
print(f"BatchNorm2d output: {output_bn2d.shape}")

# Per-channel statistics
print(f"Input per-channel mean: {input_2d.mean(dim=[0, 2, 3])[:5]}")
print(f"Output per-channel mean: {output_bn2d.mean(dim=[0, 2, 3])[:5]}")

# BatchNorm3d for 3D data
bn3d = nn.BatchNorm3d(16)
input_3d = torch.randn(8, 16, 10, 16, 16)  # (batch, channels, depth, height, width)
output_bn3d = bn3d(input_3d)

print(f"\nBatchNorm3d input: {input_3d.shape}")
print(f"BatchNorm3d output: {output_bn3d.shape}")

print("\n=== BatchNorm Parameters and Modes ===")

# BatchNorm with different parameters
bn_custom = nn.BatchNorm2d(
    num_features=64,
    eps=1e-5,          # Small constant for numerical stability
    momentum=0.1,      # Momentum for running statistics
    affine=True,       # Learnable affine parameters
    track_running_stats=True  # Track running statistics
)

# BatchNorm without affine transformation
bn_no_affine = nn.BatchNorm2d(64, affine=False)
input_test = torch.randn(8, 64, 16, 16)

output_affine = bn_custom(input_test)
output_no_affine = bn_no_affine(input_test)

print(f"BatchNorm with affine - weight: {bn_custom.weight is not None}")
print(f"BatchNorm without affine - weight: {bn_no_affine.weight is not None}")

# Training vs Evaluation mode
bn_mode_test = nn.BatchNorm1d(32)
input_mode = torch.randn(16, 32)

# Training mode
bn_mode_test.train()
output_train = bn_mode_test(input_mode)

# Evaluation mode
bn_mode_test.eval()
output_eval = bn_mode_test(input_mode)

print(f"\nTraining mode output mean: {output_train.mean():.6f}")
print(f"Evaluation mode output mean: {output_eval.mean():.6f}")
print(f"Training mode uses batch statistics: {bn_mode_test.training}")

print("\n=== Layer Normalization ===")

# LayerNorm normalizes across features for each sample
ln = nn.LayerNorm(64)
input_ln = torch.randn(32, 64)
output_ln = ln(input_ln)

print(f"LayerNorm input: {input_ln.shape}")
print(f"LayerNorm output: {output_ln.shape}")

# Per-sample statistics
print(f"Input per-sample mean: {input_ln.mean(dim=1)[:5]}")
print(f"Output per-sample mean: {output_ln.mean(dim=1)[:5]}")
print(f"Input per-sample std: {input_ln.std(dim=1)[:5]}")
print(f"Output per-sample std: {output_ln.std(dim=1)[:5]}")

# Multi-dimensional LayerNorm
ln_2d = nn.LayerNorm([32, 32])  # Normalize over last two dimensions
input_ln_2d = torch.randn(16, 64, 32, 32)
output_ln_2d = ln_2d(input_ln_2d)

print(f"\nLayerNorm 2D input: {input_ln_2d.shape}")
print(f"LayerNorm 2D output: {output_ln_2d.shape}")

# LayerNorm for sequences (common in transformers)
ln_seq = nn.LayerNorm(256)
input_seq = torch.randn(8, 50, 256)  # (batch, sequence, features)
output_seq = ln_seq(input_seq)

print(f"Sequence LayerNorm: {input_seq.shape} -> {output_seq.shape}")

print("\n=== Group Normalization ===")

# GroupNorm divides channels into groups and normalizes within each group
gn = nn.GroupNorm(num_groups=8, num_channels=32)
input_gn = torch.randn(16, 32, 28, 28)
output_gn = gn(input_gn)

print(f"GroupNorm input: {input_gn.shape}")
print(f"GroupNorm output: {output_gn.shape}")
print(f"Groups: 8, Channels per group: {32 // 8}")

# Different group configurations
group_configs = [1, 2, 4, 8, 16, 32]
input_gn_test = torch.randn(4, 32, 16, 16)

for num_groups in group_configs:
    if 32 % num_groups == 0:  # Must be divisible
        gn_test = nn.GroupNorm(num_groups, 32)
        output_gn_test = gn_test(input_gn_test)
        channels_per_group = 32 // num_groups
        print(f"Groups: {num_groups}, Channels per group: {channels_per_group}")

print("\n=== Instance Normalization ===")

# InstanceNorm normalizes each channel independently for each sample
in1d = nn.InstanceNorm1d(64)
in2d = nn.InstanceNorm2d(32)
in3d = nn.InstanceNorm3d(16)

input_in1d = torch.randn(8, 64, 100)
input_in2d = torch.randn(8, 32, 28, 28)
input_in3d = torch.randn(4, 16, 10, 16, 16)

output_in1d = in1d(input_in1d)
output_in2d = in2d(input_in2d)
output_in3d = in3d(input_in3d)

print(f"InstanceNorm1d: {input_in1d.shape} -> {output_in1d.shape}")
print(f"InstanceNorm2d: {input_in2d.shape} -> {output_in2d.shape}")
print(f"InstanceNorm3d: {input_in3d.shape} -> {output_in3d.shape}")

# InstanceNorm statistics (per channel, per sample)
sample_idx, channel_idx = 0, 0
input_sample_channel = input_in2d[sample_idx, channel_idx]
output_sample_channel = output_in2d[sample_idx, channel_idx]

print(f"\nSample {sample_idx}, Channel {channel_idx}:")
print(f"Input mean: {input_sample_channel.mean():.6f}")
print(f"Output mean: {output_sample_channel.mean():.6f}")
print(f"Input std: {input_sample_channel.std():.6f}")
print(f"Output std: {output_sample_channel.std():.6f}")

print("\n=== Normalization Comparison ===")

def compare_normalizations(input_tensor):
    """Compare different normalization techniques"""
    batch_size, channels, height, width = input_tensor.shape
    
    # Initialize normalizations
    batch_norm = nn.BatchNorm2d(channels)
    layer_norm = nn.LayerNorm([channels, height, width])
    group_norm = nn.GroupNorm(8, channels)  # 8 groups
    instance_norm = nn.InstanceNorm2d(channels)
    
    # Apply normalizations
    results = {}
    results['input'] = input_tensor
    results['batch_norm'] = batch_norm(input_tensor)
    results['layer_norm'] = layer_norm(input_tensor)
    results['group_norm'] = group_norm(input_tensor)
    results['instance_norm'] = instance_norm(input_tensor)
    
    # Compute statistics
    for name, tensor in results.items():
        mean_val = tensor.mean().item()
        std_val = tensor.std().item()
        print(f"{name:12}: mean={mean_val:8.6f}, std={std_val:8.6f}")
    
    return results

# Compare normalizations
comparison_input = torch.randn(4, 32, 8, 8) * 2 + 3  # Non-zero mean, larger std
print("Normalization comparison:")
comparison_results = compare_normalizations(comparison_input)

print("\n=== Advanced Normalization Techniques ===")

class RMSNorm(nn.Module):
    """Root Mean Square Normalization"""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # Compute RMS
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return x / rms * self.scale

class AdaptiveInstanceNorm(nn.Module):
    """Adaptive Instance Normalization"""
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False)
    
    def forward(self, x, style_mean, style_std):
        # Normalize
        normalized = self.instance_norm(x)
        # Apply style statistics
        return style_std.unsqueeze(2).unsqueeze(3) * normalized + style_mean.unsqueeze(2).unsqueeze(3)

class SpectralNorm(nn.Module):
    """Spectral Normalization wrapper"""
    def __init__(self, module, power_iterations=1):
        super().__init__()
        self.module = module
        self.power_iterations = power_iterations
        
        # Initialize spectral normalization
        if hasattr(module, 'weight'):
            self._make_params()
    
    def _make_params(self):
        """Create parameters for spectral normalization"""
        weight = self.module.weight
        height = weight.size(0)
        width = weight.view(height, -1).size(1)
        
        u = weight.new_empty(height).normal_(0, 1)
        v = weight.new_empty(width).normal_(0, 1)
        
        self.register_buffer('u', u / torch.norm(u))
        self.register_buffer('v', v / torch.norm(v))
    
    def forward(self, x):
        if self.training:
            self._update_u_v()
        return self.module(x)
    
    def _update_u_v(self):
        """Update u and v vectors"""
        weight = self.module.weight
        weight_mat = weight.view(weight.size(0), -1)
        
        for _ in range(self.power_iterations):
            v = F.normalize(torch.mv(weight_mat.t(), self.u), dim=0)
            u = F.normalize(torch.mv(weight_mat, v), dim=0)
        
        sigma = torch.dot(u, torch.mv(weight_mat, v))
        self.module.weight.data = weight / sigma

# Test advanced normalizations
rms_norm = RMSNorm(64)
rms_input = torch.randn(8, 64)
rms_output = rms_norm(rms_input)
print(f"RMSNorm: {rms_input.shape} -> {rms_output.shape}")

# Adaptive Instance Norm
ada_in = AdaptiveInstanceNorm(32)
ada_input = torch.randn(4, 32, 16, 16)
style_mean = torch.randn(4, 32)
style_std = torch.randn(4, 32)
ada_output = ada_in(ada_input, style_mean, style_std)
print(f"AdaptiveInstanceNorm: {ada_input.shape} -> {ada_output.shape}")

# Spectral Norm
conv_layer = nn.Conv2d(32, 64, 3, padding=1)
spectral_conv = SpectralNorm(conv_layer)
spec_input = torch.randn(4, 32, 16, 16)
spec_output = spectral_conv(spec_input)
print(f"SpectralNorm Conv: {spec_input.shape} -> {spec_output.shape}")

print("\n=== Normalization in Different Architectures ===")

class ConvBlock(nn.Module):
    """Convolution block with different normalization options"""
    def __init__(self, in_channels, out_channels, norm_type='batch'):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=(norm_type == 'none'))
        
        if norm_type == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == 'layer':
            self.norm = lambda x: F.layer_norm(x, x.shape[1:])
        elif norm_type == 'group':
            self.norm = nn.GroupNorm(8, out_channels)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels)
        else:
            self.norm = nn.Identity()
        
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block with layer normalization"""
    def __init__(self, d_model, nhead):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

# Test different normalization in conv blocks
norm_types = ['batch', 'layer', 'group', 'instance', 'none']
conv_input = torch.randn(4, 32, 16, 16)

for norm_type in norm_types:
    conv_block = ConvBlock(32, 64, norm_type)
    conv_output = conv_block(conv_input)
    params = sum(p.numel() for p in conv_block.parameters())
    print(f"{norm_type:8} norm: {conv_input.shape} -> {conv_output.shape}, {params} params")

# Test transformer block
transformer = TransformerBlock(256, 8)
transformer_input = torch.randn(4, 10, 256)  # (batch, sequence, features)
transformer_output = transformer(transformer_input)
print(f"\nTransformer: {transformer_input.shape} -> {transformer_output.shape}")

print("\n=== Normalization Statistics and Training ===")

class NormalizationAnalyzer:
    """Analyze normalization behavior during training"""
    def __init__(self, norm_layer):
        self.norm_layer = norm_layer
        self.stats_history = []
    
    def analyze_step(self, input_tensor):
        """Analyze one training step"""
        with torch.no_grad():
            # Forward pass
            output = self.norm_layer(input_tensor)
            
            # Collect statistics
            stats = {
                'input_mean': input_tensor.mean().item(),
                'input_std': input_tensor.std().item(),
                'output_mean': output.mean().item(),
                'output_std': output.std().item(),
            }
            
            # Add normalization-specific stats
            if hasattr(self.norm_layer, 'running_mean'):
                stats['running_mean'] = self.norm_layer.running_mean.mean().item()
                stats['running_var'] = self.norm_layer.running_var.mean().item()
            
            self.stats_history.append(stats)
            return stats

# Analyze BatchNorm behavior
bn_analyzer = NormalizationAnalyzer(nn.BatchNorm1d(64))

print("BatchNorm training analysis (first 5 steps):")
for step in range(5):
    # Simulate training data
    train_input = torch.randn(32, 64) * (step + 1) + step  # Changing distribution
    stats = bn_analyzer.analyze_step(train_input)
    
    print(f"Step {step}: input_mean={stats['input_mean']:.3f}, "
          f"output_mean={stats['output_mean']:.3f}, "
          f"running_mean={stats['running_mean']:.3f}")

print("\n=== Functional Normalization ===")

# Using functional interface
input_func = torch.randn(8, 32, 16, 16)

# Functional batch norm
output_func_bn = F.batch_norm(
    input_func, 
    running_mean=torch.zeros(32),
    running_var=torch.ones(32),
    weight=torch.ones(32),
    bias=torch.zeros(32),
    training=True
)

# Functional layer norm
output_func_ln = F.layer_norm(input_func, input_func.shape[1:])

# Functional group norm
output_func_gn = F.group_norm(input_func, num_groups=8)

# Functional instance norm
output_func_in = F.instance_norm(input_func)

print(f"Functional batch_norm: {input_func.shape} -> {output_func_bn.shape}")
print(f"Functional layer_norm: {input_func.shape} -> {output_func_ln.shape}")
print(f"Functional group_norm: {input_func.shape} -> {output_func_gn.shape}")
print(f"Functional instance_norm: {input_func.shape} -> {output_func_in.shape}")

print("\n=== Normalization Performance Impact ===")

def benchmark_normalizations():
    """Benchmark different normalization layers"""
    import time
    
    input_data = torch.randn(16, 64, 32, 32)
    if torch.cuda.is_available():
        input_data = input_data.cuda()
    
    normalizations = [
        ("BatchNorm2d", nn.BatchNorm2d(64)),
        ("GroupNorm", nn.GroupNorm(8, 64)),
        ("InstanceNorm2d", nn.InstanceNorm2d(64)),
        ("LayerNorm", nn.LayerNorm([64, 32, 32])),
    ]
    
    iterations = 100
    
    for name, norm in normalizations:
        if torch.cuda.is_available():
            norm = norm.cuda()
        
        # Warmup
        for _ in range(5):
            _ = norm(input_data)
        
        # Benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(iterations):
            output = norm(input_data)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        elapsed = time.time() - start_time
        params = sum(p.numel() for p in norm.parameters())
        
        print(f"{name:15}: {elapsed/iterations*1000:.3f}ms, {params} params")

print("Normalization performance benchmark:")
if torch.cuda.is_available():
    benchmark_normalizations()
else:
    print("CUDA not available, skipping benchmark")

print("\n=== Normalization Best Practices ===")

print("Normalization Guidelines:")
print("1. BatchNorm: Standard choice for most CNN architectures")
print("2. LayerNorm: Preferred for transformers and RNNs")
print("3. GroupNorm: Good alternative when batch size is small")
print("4. InstanceNorm: Useful for style transfer and GANs")
print("5. Place normalization before or after activation (experiment)")
print("6. Use appropriate momentum for BatchNorm")
print("7. Consider normalization's effect on gradient flow")

print("\nArchitecture-Specific Choices:")
print("- CNNs: BatchNorm2d, GroupNorm for small batches")
print("- Transformers: LayerNorm")
print("- GANs: InstanceNorm, SpectralNorm")
print("- Style Transfer: AdaptiveInstanceNorm")
print("- Small batch training: GroupNorm, LayerNorm")
print("- Mobile networks: Consider computational cost")

print("\nTraining Considerations:")
print("- BatchNorm requires sufficient batch size (>2)")
print("- Set correct training/evaluation modes")
print("- Initialize normalization parameters carefully")
print("- Monitor running statistics convergence")
print("- Consider frozen BatchNorm for transfer learning")
print("- Use appropriate epsilon for numerical stability")

print("\nCommon Issues:")
print("- BatchNorm breaks with batch_size=1")
print("- LayerNorm can be expensive for large feature maps")
print("- GroupNorm requires channels divisible by groups")
print("- InstanceNorm removes important statistics")
print("- Normalization placement affects performance")
print("- Different behavior in train vs eval mode")

print("\n=== Normalization Layers Complete ===") 