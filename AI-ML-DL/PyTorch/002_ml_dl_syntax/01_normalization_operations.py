#!/usr/bin/env python3
"""PyTorch Normalization Operations - All normalization techniques"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== Batch Normalization ===")

# 1D Batch Normalization
batch_norm_1d = nn.BatchNorm1d(num_features=128)
input_1d = torch.randn(32, 128)  # (batch, features)
output_1d = batch_norm_1d(input_1d)
print(f"BatchNorm1d input: {input_1d.shape}, output: {output_1d.shape}")

# 2D Batch Normalization (for CNNs)
batch_norm_2d = nn.BatchNorm2d(num_features=64)
input_2d = torch.randn(16, 64, 32, 32)  # (batch, channels, height, width)
output_2d = batch_norm_2d(input_2d)
print(f"BatchNorm2d input: {input_2d.shape}, output: {output_2d.shape}")

# 3D Batch Normalization
batch_norm_3d = nn.BatchNorm3d(num_features=32)
input_3d = torch.randn(8, 32, 16, 16, 16)  # (batch, channels, depth, height, width)
output_3d = batch_norm_3d(input_3d)
print(f"BatchNorm3d input: {input_3d.shape}, output: {output_3d.shape}")

# Batch normalization parameters
print(f"BN running mean shape: {batch_norm_2d.running_mean.shape}")
print(f"BN running var shape: {batch_norm_2d.running_var.shape}")
print(f"BN weight shape: {batch_norm_2d.weight.shape}")
print(f"BN bias shape: {batch_norm_2d.bias.shape}")

print("\n=== Layer Normalization ===")

# Layer Normalization
layer_norm = nn.LayerNorm(normalized_shape=128)
input_ln = torch.randn(32, 10, 128)  # (batch, seq_len, features)
output_ln = layer_norm(input_ln)
print(f"LayerNorm input: {input_ln.shape}, output: {output_ln.shape}")

# Layer norm with multiple dimensions
layer_norm_2d = nn.LayerNorm([64, 32])
input_ln_2d = torch.randn(16, 64, 32)
output_ln_2d = layer_norm_2d(input_ln_2d)
print(f"LayerNorm2D input: {input_ln_2d.shape}, output: {output_ln_2d.shape}")

# Layer norm parameters
print(f"LayerNorm weight shape: {layer_norm.weight.shape}")
print(f"LayerNorm bias shape: {layer_norm.bias.shape}")

print("\n=== Instance Normalization ===")

# Instance Normalization
instance_norm_1d = nn.InstanceNorm1d(num_features=64)
instance_norm_2d = nn.InstanceNorm2d(num_features=32)
instance_norm_3d = nn.InstanceNorm3d(num_features=16)

input_in_1d = torch.randn(8, 64, 100)
input_in_2d = torch.randn(4, 32, 64, 64)
input_in_3d = torch.randn(2, 16, 32, 32, 32)

output_in_1d = instance_norm_1d(input_in_1d)
output_in_2d = instance_norm_2d(input_in_2d)
output_in_3d = instance_norm_3d(input_in_3d)

print(f"InstanceNorm1D: {input_in_1d.shape} -> {output_in_1d.shape}")
print(f"InstanceNorm2D: {input_in_2d.shape} -> {output_in_2d.shape}")
print(f"InstanceNorm3D: {input_in_3d.shape} -> {output_in_3d.shape}")

print("\n=== Group Normalization ===")

# Group Normalization
group_norm = nn.GroupNorm(num_groups=8, num_channels=32)
input_gn = torch.randn(4, 32, 64, 64)
output_gn = group_norm(input_gn)
print(f"GroupNorm input: {input_gn.shape}, output: {output_gn.shape}")

# Different group configurations
group_norm_4 = nn.GroupNorm(num_groups=4, num_channels=32)
group_norm_16 = nn.GroupNorm(num_groups=16, num_channels=32)
output_gn_4 = group_norm_4(input_gn)
output_gn_16 = group_norm_16(input_gn)
print(f"GroupNorm (4 groups): {output_gn_4.shape}")
print(f"GroupNorm (16 groups): {output_gn_16.shape}")

print("\n=== Local Response Normalization ===")

# Local Response Normalization
lrn = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
input_lrn = torch.randn(8, 32, 64, 64)
output_lrn = lrn(input_lrn)
print(f"LRN input: {input_lrn.shape}, output: {output_lrn.shape}")

print("\n=== Functional Normalization ===")

# Functional batch normalization
input_func = torch.randn(16, 64, 32, 32)
running_mean = torch.zeros(64)
running_var = torch.ones(64)
weight = torch.ones(64)
bias = torch.zeros(64)

output_func = F.batch_norm(input_func, running_mean, running_var, weight, bias, training=True)
print(f"Functional BatchNorm: {input_func.shape} -> {output_func.shape}")

# Functional layer normalization
input_func_ln = torch.randn(32, 128)
output_func_ln = F.layer_norm(input_func_ln, normalized_shape=[128])
print(f"Functional LayerNorm: {input_func_ln.shape} -> {output_func_ln.shape}")

# Functional instance normalization
output_func_in = F.instance_norm(input_func)
print(f"Functional InstanceNorm: {input_func.shape} -> {output_func_in.shape}")

# Functional group normalization
output_func_gn = F.group_norm(input_func, num_groups=8)
print(f"Functional GroupNorm: {input_func.shape} -> {output_func_gn.shape}")

print("\n=== Custom Normalization ===")

# Manual batch normalization implementation
def manual_batch_norm(x, eps=1e-5):
    mean = x.mean(dim=0, keepdim=True)
    var = x.var(dim=0, keepdim=True, unbiased=False)
    normalized = (x - mean) / torch.sqrt(var + eps)
    return normalized

# Manual layer normalization implementation
def manual_layer_norm(x, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    normalized = (x - mean) / torch.sqrt(var + eps)
    return normalized

# Test manual implementations
input_manual = torch.randn(32, 128)
manual_bn_result = manual_batch_norm(input_manual)
manual_ln_result = manual_layer_norm(input_manual)

print(f"Manual BatchNorm result shape: {manual_bn_result.shape}")
print(f"Manual LayerNorm result shape: {manual_ln_result.shape}")

print("\n=== Normalization with Affine Parameters ===")

# Normalization with learnable parameters
batch_norm_affine = nn.BatchNorm1d(128, affine=True)
batch_norm_no_affine = nn.BatchNorm1d(128, affine=False)

input_affine = torch.randn(32, 128)
output_affine = batch_norm_affine(input_affine)
output_no_affine = batch_norm_no_affine(input_affine)

print(f"With affine parameters: {output_affine.shape}")
print(f"Without affine parameters: {output_no_affine.shape}")
print(f"Affine weight exists: {batch_norm_affine.weight is not None}")
print(f"No affine weight: {batch_norm_no_affine.weight is None}")

print("\n=== Training vs Evaluation Mode ===")

# Batch normalization behavior in train vs eval mode
bn_train_eval = nn.BatchNorm2d(32)
input_mode = torch.randn(16, 32, 64, 64)

# Training mode
bn_train_eval.train()
output_train = bn_train_eval(input_mode)
running_mean_train = bn_train_eval.running_mean.clone()

# Evaluation mode
bn_train_eval.eval()
output_eval = bn_train_eval(input_mode)
running_mean_eval = bn_train_eval.running_mean.clone()

print(f"Training mode output: {output_train.shape}")
print(f"Evaluation mode output: {output_eval.shape}")
print(f"Running mean changed: {not torch.equal(running_mean_train, running_mean_eval)}")

print("\n=== Momentum and Track Running Stats ===")

# Batch normalization with different momentum
bn_momentum_01 = nn.BatchNorm2d(32, momentum=0.1)
bn_momentum_09 = nn.BatchNorm2d(32, momentum=0.9)

# Track running stats control
bn_track_stats = nn.BatchNorm2d(32, track_running_stats=True)
bn_no_track = nn.BatchNorm2d(32, track_running_stats=False)

input_momentum = torch.randn(8, 32, 32, 32)

output_m01 = bn_momentum_01(input_momentum)
output_m09 = bn_momentum_09(input_momentum)
output_track = bn_track_stats(input_momentum)
output_no_track = bn_no_track(input_momentum)

print(f"Momentum 0.1 output: {output_m01.shape}")
print(f"Momentum 0.9 output: {output_m09.shape}")
print(f"Track stats has running_mean: {hasattr(bn_track_stats, 'running_mean')}")
print(f"No track has running_mean: {hasattr(bn_no_track, 'running_mean')}")

print("\n=== Normalization in Different Contexts ===")

# Normalization in convolution layers
class ConvWithNorm(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='batch'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        if norm_type == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm_type == 'group':
            self.norm = nn.GroupNorm(8, out_channels)
        else:
            self.norm = nn.Identity()
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

# Test different normalization in conv layers
conv_bn = ConvWithNorm(3, 64, 'batch')
conv_in = ConvWithNorm(3, 64, 'instance')
conv_gn = ConvWithNorm(3, 64, 'group')

input_conv = torch.randn(8, 3, 224, 224)
output_conv_bn = conv_bn(input_conv)
output_conv_in = conv_in(input_conv)
output_conv_gn = conv_gn(input_conv)

print(f"Conv + BatchNorm: {input_conv.shape} -> {output_conv_bn.shape}")
print(f"Conv + InstanceNorm: {input_conv.shape} -> {output_conv_in.shape}")
print(f"Conv + GroupNorm: {input_conv.shape} -> {output_conv_gn.shape}")

print("\n=== Normalization for Different Tasks ===")

# RNN Layer Normalization
class RNNWithLayerNorm(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        output, (h, c) = self.rnn(x)
        normalized_output = self.layer_norm(output)
        return normalized_output

# Test RNN with layer normalization
rnn_ln = RNNWithLayerNorm(128, 256)
input_rnn = torch.randn(32, 50, 128)  # (batch, seq_len, features)
output_rnn = rnn_ln(input_rnn)
print(f"RNN + LayerNorm: {input_rnn.shape} -> {output_rnn.shape}")

# Transformer-style normalization
class TransformerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        # Pre-normalization style
        normalized = self.norm1(x)
        output = self.linear(normalized)
        # Post-normalization
        final = self.norm2(x + output)  # Residual connection
        return final

transformer_norm = TransformerNorm(512)
input_transformer = torch.randn(32, 100, 512)
output_transformer = transformer_norm(input_transformer)
print(f"Transformer norm: {input_transformer.shape} -> {output_transformer.shape}")

print("\n=== Advanced Normalization Techniques ===")

# Weight Standardization
def weight_standardization(weight, eps=1e-5):
    mean = weight.mean(dim=[1, 2, 3], keepdim=True)
    var = weight.var(dim=[1, 2, 3], keepdim=True, unbiased=False)
    weight_standardized = (weight - mean) / torch.sqrt(var + eps)
    return weight_standardized

# Example weight standardization
conv_weight = torch.randn(64, 32, 3, 3)
standardized_weight = weight_standardization(conv_weight)
print(f"Weight standardization: {conv_weight.shape} -> {standardized_weight.shape}")

# Spectral Normalization
spectral_norm_conv = nn.utils.spectral_norm(nn.Conv2d(32, 64, 3))
input_spectral = torch.randn(8, 32, 64, 64)
output_spectral = spectral_norm_conv(input_spectral)
print(f"Spectral norm conv: {input_spectral.shape} -> {output_spectral.shape}")

print("\n=== Normalization Comparison ===")

# Compare different normalizations
input_compare = torch.randn(16, 32, 64, 64)

bn_compare = nn.BatchNorm2d(32)
ln_compare = nn.LayerNorm([32, 64, 64])
in_compare = nn.InstanceNorm2d(32)
gn_compare = nn.GroupNorm(8, 32)

output_bn_comp = bn_compare(input_compare)
output_ln_comp = ln_compare(input_compare)
output_in_comp = in_compare(input_compare)
output_gn_comp = gn_compare(input_compare)

print(f"Input stats - Mean: {input_compare.mean():.4f}, Std: {input_compare.std():.4f}")
print(f"BatchNorm stats - Mean: {output_bn_comp.mean():.4f}, Std: {output_bn_comp.std():.4f}")
print(f"LayerNorm stats - Mean: {output_ln_comp.mean():.4f}, Std: {output_ln_comp.std():.4f}")
print(f"InstanceNorm stats - Mean: {output_in_comp.mean():.4f}, Std: {output_in_comp.std():.4f}")
print(f"GroupNorm stats - Mean: {output_gn_comp.mean():.4f}, Std: {output_gn_comp.std():.4f}")

print("\n=== Normalization Best Practices ===")

print("Normalization Guidelines:")
print("1. BatchNorm: Use for CNN training with sufficient batch size")
print("2. LayerNorm: Use for RNNs, Transformers, and small batches")
print("3. InstanceNorm: Use for style transfer and GANs")
print("4. GroupNorm: Use when batch size is small or varies")
print("5. Apply normalization before or after activation (experiment)")
print("6. Use appropriate momentum for BatchNorm based on training")
print("7. Consider normalization placement in residual connections")

print("\n=== Normalization Operations Complete ===") 