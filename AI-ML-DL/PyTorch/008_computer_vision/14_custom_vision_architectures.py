import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple

# Basic Building Blocks
class ConvBlock(nn.Module):
    """Basic convolutional block with batch norm and activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 activation='relu', use_bn=True, dropout=0.0):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class ResidualBlock(nn.Module):
    """Residual block with bottleneck design"""
    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super().__init__()
        hidden_channels = out_channels // expansion
        
        self.conv1 = ConvBlock(in_channels, hidden_channels, kernel_size=1, padding=0)
        self.conv2 = ConvBlock(hidden_channels, hidden_channels, kernel_size=3, stride=stride)
        self.conv3 = ConvBlock(hidden_channels, out_channels, kernel_size=1, padding=0, activation='none')
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
            self.skip_bn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = nn.Identity()
            self.skip_bn = nn.Identity()
    
    def forward(self, x):
        residual = self.skip_bn(self.skip(x))
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        out += residual
        return F.relu(out)

class InvertedResidualBlock(nn.Module):
    """Inverted residual block (MobileNet style)"""
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super().__init__()
        hidden_channels = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.append(ConvBlock(in_channels, hidden_channels, kernel_size=1, padding=0))
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride, 1, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True),
            # Pointwise compression
            nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            out += x
        return out

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution (EfficientNet style)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 expand_ratio=1, se_ratio=0.25, drop_rate=0.0):
        super().__init__()
        hidden_channels = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        
        # Expansion phase
        if expand_ratio != 1:
            layers.append(ConvBlock(in_channels, hidden_channels, kernel_size=1, padding=0, activation='swish'))
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride, 
                     kernel_size//2, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU()
        ])
        
        # Squeeze-and-Excitation
        if se_ratio > 0:
            layers.append(SEBlock(hidden_channels, int(1/se_ratio)))
        
        # Output phase
        layers.extend([
            nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout2d(drop_rate) if drop_rate > 0 else nn.Identity()
    
    def forward(self, x):
        out = self.conv(x)
        out = self.dropout(out)
        
        if self.use_residual:
            out += x
        return out

# Attention Mechanisms for Custom Architectures
class MultiHeadSelfAttention2D(nn.Module):
    """2D Multi-head self-attention for feature maps"""
    def __init__(self, channels, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(0.1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        qkv = self.qkv(x_flat).reshape(B, H*W, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x_attn = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)
        
        return x_attn.transpose(1, 2).reshape(B, C, H, W)

class FPNBlock(nn.Module):
    """Feature Pyramid Network block"""
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])
    
    def forward(self, features):
        """features: list of feature maps from low to high resolution"""
        laterals = [lateral_conv(features[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        
        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] += F.interpolate(laterals[i + 1], scale_factor=2, mode='nearest')
        
        # Final convolutions
        fpn_outputs = [fpn_conv(lateral) for fpn_conv, lateral in zip(self.fpn_convs, laterals)]
        
        return fpn_outputs

# Custom Architecture Implementations
class CustomResNet(nn.Module):
    """Custom ResNet with configurable depth and width"""
    def __init__(self, layers=[2, 2, 2, 2], channels=[64, 128, 256, 512], 
                 num_classes=1000, use_se=False):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 7, 2, 3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        
        self.stage1 = self._make_stage(channels[0], channels[0], layers[0], stride=1, use_se=use_se)
        self.stage2 = self._make_stage(channels[0], channels[1], layers[1], stride=2, use_se=use_se)
        self.stage3 = self._make_stage(channels[1], channels[2], layers[2], stride=2, use_se=use_se)
        self.stage4 = self._make_stage(channels[2], channels[3], layers[3], stride=2, use_se=use_se)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels[3], num_classes)
    
    def _make_stage(self, in_channels, out_channels, num_blocks, stride, use_se):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
            if use_se:
                layers.append(SEBlock(out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

class EfficientNetCustom(nn.Module):
    """Custom EfficientNet-style architecture"""
    def __init__(self, width_mult=1.0, depth_mult=1.0, num_classes=1000):
        super().__init__()
        
        # Scale channels and depths
        def scale_width(channels):
            return int(channels * width_mult)
        
        def scale_depth(depth):
            return int(math.ceil(depth * depth_mult))
        
        # Stem
        self.stem = ConvBlock(3, scale_width(32), kernel_size=3, stride=2, activation='swish')
        
        # MBConv blocks configuration: (expand_ratio, channels, num_blocks, stride)
        configs = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 40, 2, 2),
            (6, 80, 3, 2),
            (6, 112, 3, 1),
            (6, 192, 4, 2),
            (6, 320, 1, 1),
        ]
        
        self.blocks = nn.ModuleList()
        in_channels = scale_width(32)
        
        for expand_ratio, channels, num_blocks, stride in configs:
            out_channels = scale_width(channels)
            for i in range(scale_depth(num_blocks)):
                self.blocks.append(MBConvBlock(
                    in_channels, out_channels,
                    stride=stride if i == 0 else 1,
                    expand_ratio=expand_ratio,
                    se_ratio=0.25
                ))
                in_channels = out_channels
        
        # Head
        self.head_conv = ConvBlock(in_channels, scale_width(1280), kernel_size=1, padding=0, activation='swish')
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(scale_width(1280), num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head_conv(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

class VisionMixer(nn.Module):
    """Vision architecture mixing CNNs and attention"""
    def __init__(self, num_classes=1000, embed_dim=768):
        super().__init__()
        
        # CNN stem
        self.cnn_stem = nn.Sequential(
            ConvBlock(3, 64, 7, 2),
            nn.MaxPool2d(3, 2, 1),
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 256, 2),
        )
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadSelfAttention2D(256, num_heads=8) for _ in range(4)
        ])
        
        # Feature mixing
        self.feature_mix = nn.Sequential(
            nn.Conv2d(256, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        
        # Final layers
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        # CNN feature extraction
        x = self.cnn_stem(x)
        
        # Apply attention layers
        for attn_layer in self.attention_layers:
            x = x + attn_layer(x)  # Residual connection
        
        # Feature mixing and classification
        x = self.feature_mix(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

class PyramidVisionNet(nn.Module):
    """Custom network with Feature Pyramid Network"""
    def __init__(self, num_classes=1000):
        super().__init__()
        
        # Backbone stages
        self.stage1 = nn.Sequential(
            ConvBlock(3, 64, 7, 2),
            nn.MaxPool2d(3, 2, 1),
            ResidualBlock(64, 64)
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 128)
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, 2),
            ResidualBlock(256, 256)
        )
        self.stage4 = nn.Sequential(
            ResidualBlock(256, 512, 2),
            ResidualBlock(512, 512)
        )
        
        # FPN
        self.fpn = FPNBlock([64, 128, 256, 512], 256)
        
        # Classifier
        self.global_pools = nn.ModuleList([nn.AdaptiveAvgPool2d(1) for _ in range(4)])
        self.classifier = nn.Linear(256 * 4, num_classes)
    
    def forward(self, x):
        # Extract multi-scale features
        features = []
        x = self.stage1(x)
        features.append(x)
        x = self.stage2(x)
        features.append(x)
        x = self.stage3(x)
        features.append(x)
        x = self.stage4(x)
        features.append(x)
        
        # FPN processing
        fpn_features = self.fpn(features)
        
        # Global pooling and concatenation
        pooled_features = []
        for i, (feat, pool) in enumerate(zip(fpn_features, self.global_pools)):
            pooled = pool(feat).flatten(1)
            pooled_features.append(pooled)
        
        combined = torch.cat(pooled_features, dim=1)
        return self.classifier(combined)

# Architecture Factory
def create_custom_architecture(arch_name, **kwargs):
    """Factory function to create custom architectures"""
    architectures = {
        'custom_resnet': CustomResNet,
        'efficientnet_custom': EfficientNetCustom,
        'vision_mixer': VisionMixer,
        'pyramid_vision_net': PyramidVisionNet
    }
    
    if arch_name in architectures:
        return architectures[arch_name](**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")

# Training utilities
def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def model_summary(model, input_size=(3, 224, 224)):
    """Print model summary"""
    total_params, trainable_params = count_parameters(model)
    
    # Test forward pass
    x = torch.randn(1, *input_size)
    with torch.no_grad():
        output = model(x)
    
    print(f"Model Summary:")
    print(f"Input shape: {input_size}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (MB): {total_params * 4 / (1024**2):.2f}")

if __name__ == "__main__":
    # Test custom architectures
    print("Testing custom vision architectures...")
    
    # Custom ResNet
    custom_resnet = CustomResNet(layers=[2, 2, 2, 2], num_classes=10)
    model_summary(custom_resnet)
    print()
    
    # Custom EfficientNet
    efficient_custom = EfficientNetCustom(width_mult=0.5, depth_mult=0.5, num_classes=10)
    model_summary(efficient_custom)
    print()
    
    # Vision Mixer
    vision_mixer = VisionMixer(num_classes=10, embed_dim=384)
    model_summary(vision_mixer)
    print()
    
    # Pyramid Vision Net
    pyramid_net = PyramidVisionNet(num_classes=10)
    model_summary(pyramid_net)
    print()
    
    # Test individual blocks
    print("Testing individual blocks...")
    
    # MBConv block
    mbconv = MBConvBlock(64, 128, expand_ratio=6)
    x = torch.randn(1, 64, 32, 32)
    out = mbconv(x)
    print(f"MBConv block output shape: {out.shape}")
    
    # SE block
    se_block = SEBlock(128)
    x = torch.randn(1, 128, 16, 16)
    out = se_block(x)
    print(f"SE block output shape: {out.shape}")
    
    # Multi-head attention
    mha = MultiHeadSelfAttention2D(256, num_heads=8)
    x = torch.randn(1, 256, 8, 8)
    out = mha(x)
    print(f"Multi-head attention output shape: {out.shape}")
    
    # FPN
    fpn = FPNBlock([64, 128, 256, 512], 256)
    features = [torch.randn(1, ch, 32//(2**i), 32//(2**i)) for i, ch in enumerate([64, 128, 256, 512])]
    fpn_out = fpn(features)
    print(f"FPN output shapes: {[f.shape for f in fpn_out]}")
    
    # Architecture factory
    model = create_custom_architecture('custom_resnet', layers=[1, 1, 1, 1], num_classes=100)
    print(f"Factory created model output shape: {model(torch.randn(1, 3, 224, 224)).shape}")
    
    print("\nCustom vision architectures testing completed!")