import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import math

# Spatial Attention Mechanisms
class SpatialAttention(nn.Module):
    """Spatial attention module focusing on 'where' is important"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                             padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Aggregate channel information
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Average pooling
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Max pooling
        
        # Concatenate and convolve
        combined = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.conv(combined)
        attention_map = self.sigmoid(attention_map)
        
        return x * attention_map

class ChannelAttention(nn.Module):
    """Channel attention module focusing on 'what' is important"""
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        
        # Global average pooling
        avg_out = self.avg_pool(x).view(batch_size, channels)
        avg_out = self.fc(avg_out)
        
        # Global max pooling
        max_out = self.max_pool(x).view(batch_size, channels)
        max_out = self.fc(max_out)
        
        # Combine and apply attention
        attention = self.sigmoid(avg_out + max_out).view(batch_size, channels, 1, 1)
        return x * attention

class CBAM(nn.Module):
    """Convolutional Block Attention Module - combines channel and spatial attention"""
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        # Apply channel attention first
        x = self.channel_attention(x)
        # Then apply spatial attention
        x = self.spatial_attention(x)
        return x

# Self-Attention for Vision
class SelfAttention2D(nn.Module):
    """2D Self-attention for feature maps"""
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.inter_channels = in_channels // reduction_ratio
        
        self.query_conv = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.value_conv = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.output_conv = nn.Conv2d(self.inter_channels, in_channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate query, key, value
        query = self.query_conv(x).view(batch_size, self.inter_channels, -1)
        key = self.key_conv(x).view(batch_size, self.inter_channels, -1)
        value = self.value_conv(x).view(batch_size, self.inter_channels, -1)
        
        # Compute attention scores
        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = self.softmax(attention)
        
        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, self.inter_channels, height, width)
        out = self.output_conv(out)
        
        # Residual connection
        return self.gamma * out + x

class NonLocalAttention(nn.Module):
    """Non-local attention for capturing long-range dependencies"""
    def __init__(self, in_channels, reduction_ratio=2):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // reduction_ratio
        
        self.theta = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.g = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.output = nn.Conv2d(self.inter_channels, in_channels, 1)
        
        # Optional downsampling for efficiency
        self.max_pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate theta (query)
        theta = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta = theta.permute(0, 2, 1)  # [B, H*W, C']
        
        # Generate phi (key) with downsampling
        phi = self.phi(x)
        phi = self.max_pool(phi).view(batch_size, self.inter_channels, -1)  # [B, C', H*W/4]
        
        # Generate g (value) with downsampling
        g = self.g(x)
        g = self.max_pool(g).view(batch_size, self.inter_channels, -1)  # [B, C', H*W/4]
        g = g.permute(0, 2, 1)  # [B, H*W/4, C']
        
        # Compute attention
        attention = torch.bmm(theta, phi)  # [B, H*W, H*W/4]
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention
        y = torch.bmm(attention, g)  # [B, H*W, C']
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, height, width)
        
        # Output projection
        y = self.output(y)
        
        return x + y

# Cross-Attention for Multi-Modal Vision
class CrossAttention(nn.Module):
    """Cross-attention between two feature maps"""
    def __init__(self, query_dim, key_dim, value_dim, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.query_proj = nn.Linear(query_dim, embed_dim)
        self.key_proj = nn.Linear(key_dim, embed_dim)
        self.value_proj = nn.Linear(value_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, query_dim)
        
    def forward(self, query, key, value):
        # query: [B, N, query_dim]
        # key: [B, M, key_dim]
        # value: [B, M, value_dim]
        
        Q = self.query_proj(query)  # [B, N, embed_dim]
        K = self.key_proj(key)      # [B, M, embed_dim]
        V = self.value_proj(value)  # [B, M, embed_dim]
        
        # Compute attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.embed_dim)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attended = torch.bmm(attention_weights, V)
        output = self.output_proj(attended)
        
        return output, attention_weights

# Multi-Scale Attention
class MultiScaleAttention(nn.Module):
    """Multi-scale spatial attention"""
    def __init__(self, in_channels, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.attentions = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, in_channels, 1),
                nn.Sigmoid()
            ) for scale in scales
        ])
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        attention_maps = []
        
        for attention_module in self.attentions:
            attention_map = attention_module(x)
            attention_map = F.interpolate(attention_map, size=(height, width), 
                                        mode='bilinear', align_corners=False)
            attention_maps.append(attention_map)
        
        # Combine multi-scale attention maps
        combined_attention = torch.stack(attention_maps, dim=0).mean(dim=0)
        return x * combined_attention

# Attention-Enhanced CNN Block
class AttentionBlock(nn.Module):
    """CNN block enhanced with attention mechanisms"""
    def __init__(self, in_channels, out_channels, use_cbam=True, use_self_attention=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.cbam = CBAM(out_channels) if use_cbam else nn.Identity()
        self.self_attention = SelfAttention2D(out_channels) if use_self_attention else nn.Identity()
        
        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply attention mechanisms
        out = self.cbam(out)
        out = self.self_attention(out)
        
        return F.relu(out + residual)

# Complete Attention-Enhanced Network
class AttentionNet(nn.Module):
    """CNN with various attention mechanisms"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            AttentionBlock(3, 64),
            nn.MaxPool2d(2),
            AttentionBlock(64, 128),
            nn.MaxPool2d(2),
            AttentionBlock(128, 256),
            nn.MaxPool2d(2),
            AttentionBlock(256, 512),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Training function
def train_attention_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model
    model = AttentionNet(num_classes=10).to(device)
    
    # Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.FakeData(size=1000, image_size=(3, 224, 224), 
                                    num_classes=10, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(2):
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 20 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

if __name__ == "__main__":
    # Test individual attention mechanisms
    x = torch.randn(4, 64, 32, 32)
    
    # Spatial attention
    spatial_attn = SpatialAttention()
    spatial_out = spatial_attn(x)
    print(f"Spatial attention output shape: {spatial_out.shape}")
    
    # Channel attention
    channel_attn = ChannelAttention(64)
    channel_out = channel_attn(x)
    print(f"Channel attention output shape: {channel_out.shape}")
    
    # CBAM
    cbam = CBAM(64)
    cbam_out = cbam(x)
    print(f"CBAM output shape: {cbam_out.shape}")
    
    # Self-attention
    self_attn = SelfAttention2D(64)
    self_attn_out = self_attn(x)
    print(f"Self-attention output shape: {self_attn_out.shape}")
    
    # Non-local attention
    nonlocal_attn = NonLocalAttention(64)
    nonlocal_out = nonlocal_attn(x)
    print(f"Non-local attention output shape: {nonlocal_out.shape}")
    
    # Cross-attention
    query = torch.randn(4, 100, 256)
    key = torch.randn(4, 50, 256)
    value = torch.randn(4, 50, 256)
    cross_attn = CrossAttention(256, 256, 256, 256)
    cross_out, weights = cross_attn(query, key, value)
    print(f"Cross-attention output shape: {cross_out.shape}")
    print(f"Cross-attention weights shape: {weights.shape}")
    
    # Complete model
    model = AttentionNet(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"AttentionNet output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("\nStarting training...")
    train_attention_model()