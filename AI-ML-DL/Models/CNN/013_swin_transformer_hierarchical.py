"""
ERA 5: ATTENTION AND TRANSFORMER VISION - Swin Transformer Hierarchical
======================================================================

Year: 2021
Paper: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (Liu et al., 2021)
Innovation: Hierarchical transformer with shifted window attention for efficient computation
Previous Limitation: ViT's quadratic complexity, lack of hierarchical features, fixed resolution
Performance Gain: Linear complexity, multi-scale features, state-of-the-art on various vision tasks
Impact: Made transformers practical for dense prediction tasks, established hierarchical vision transformers

This file implements Swin Transformer that addressed ViT limitations through hierarchical architecture
and shifted window attention mechanism for efficient and effective vision processing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
from collections import defaultdict
import math
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# HISTORICAL CONTEXT & MOTIVATION
# ============================================================================

YEAR = "2021"
INNOVATION = "Hierarchical transformer with shifted window attention for efficient computation"
PREVIOUS_LIMITATION = "ViT quadratic complexity, lack of hierarchical features, fixed resolution"
IMPACT = "Made transformers practical for dense tasks, established hierarchical vision transformers"

print(f"=== Swin Transformer Hierarchical ({YEAR}) ===")
print(f"Innovation: {INNOVATION}")
print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
print(f"Impact: {IMPACT}")
print("="*60)

# ============================================================================
# STANDARDIZED DATASET LOADING (CIFAR-10)
# ============================================================================

def load_cifar10_dataset():
    """Load CIFAR-10 dataset with Swin Transformer preprocessing"""
    print("Loading CIFAR-10 dataset for Swin Transformer hierarchical study...")
    
    # Swin Transformer preprocessing with multi-scale awareness
    transform_train = transforms.Compose([
        transforms.Resize(224),  # Swin works with larger images
        transforms.RandomCrop(224, padding=28),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # CIFAR-10 class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print(f"Classes: {len(classes)}")
    print(f"Image size: 224x224 (hierarchical processing)")
    
    return train_loader, test_loader, classes

# ============================================================================
# PATCH MERGING
# ============================================================================

class PatchMerging(nn.Module):
    """
    Patch Merging Operation - Key to Hierarchical Structure
    
    Reduces spatial resolution while increasing channel dimension:
    1. Reorganize 2x2 neighboring patches
    2. Concatenate them along channel dimension
    3. Apply linear layer to reduce dimension
    4. Creates hierarchical feature pyramid
    """
    
    def __init__(self, input_resolution, dim):
        super(PatchMerging, self).__init__()
        
        self.input_resolution = input_resolution
        self.dim = dim
        
        # Linear layer to reduce dimension after merging
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
        
        print(f"    Patch Merging: {input_resolution}→{input_resolution//2}, {dim}→{2*dim}")
    
    def forward(self, x):
        """
        Forward pass: Merge 2x2 patches
        
        Args:
            x: Input features (B, H*W, C)
            
        Returns:
            Merged features (B, H*W/4, 2*C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        
        assert L == H * W, "Input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        
        # Reshape to spatial format
        x = x.view(B, H, W, C)
        
        # Extract 2x2 patches and concatenate
        x0 = x[:, 0::2, 0::2, :]  # Top-left
        x1 = x[:, 1::2, 0::2, :]  # Bottom-left
        x2 = x[:, 0::2, 1::2, :]  # Top-right
        x3 = x[:, 1::2, 1::2, :]  # Bottom-right
        
        # Concatenate along channel dimension
        x = torch.cat([x0, x1, x2, x3], -1)  # (B, H/2, W/2, 4*C)
        x = x.view(B, -1, 4 * C)  # (B, H*W/4, 4*C)
        
        # Apply normalization and reduction
        x = self.norm(x)
        x = self.reduction(x)
        
        return x

# ============================================================================
# WINDOW ATTENTION
# ============================================================================

def window_partition(x, window_size):
    """
    Partition feature map into non-overlapping windows
    
    Args:
        x: Input tensor (B, H, W, C)
        window_size: Window size (M)
        
    Returns:
        Windows (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Reverse window partition operation
    
    Args:
        windows: Windows (num_windows*B, window_size, window_size, C)
        window_size: Window size (M)
        H: Height of image
        W: Width of image
        
    Returns:
        Restored tensor (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    """
    Window-based Multi-Head Self Attention with Relative Position Bias
    
    Key Innovation: Limit attention computation to local windows
    - Reduces complexity from O(H*W)² to O(M²) per window
    - Enables processing of high-resolution images
    - Maintains local modeling capability
    """
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1):
        super(WindowAttention, self).__init__()
        
        self.dim = dim
        self.window_size = window_size  # (Mh, Mw)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        
        # Get pair-wise relative position indices
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # (2, Mh, Mw)
        coords_flatten = torch.flatten(coords, 1)  # (2, Mh*Mw)
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, Mh*Mw, Mh*Mw)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (Mh*Mw, Mh*Mw, 2)
        relative_coords[:, :, 0] += self.window_size[0] - 1  # Shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # (Mh*Mw, Mh*Mw)
        
        self.register_buffer("relative_position_index", relative_position_index)
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Initialize relative position bias
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        print(f"      Window Attention: {dim}D, {num_heads} heads, window {window_size}")
    
    def forward(self, x, mask=None):
        """
        Forward pass: Window-based attention
        
        Args:
            x: Input features (num_windows*B, N, C)
            mask: Attention mask for shifted windows
            
        Returns:
            Attended features (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        
        # QKV computation
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B_, num_heads, N, C//num_heads)
        
        # Scaled dot-product attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, N, N)
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # Apply attention mask for shifted windows
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)
        
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

# ============================================================================
# SWIN TRANSFORMER BLOCK
# ============================================================================

class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block with Window/Shifted Window Attention
    
    Key Innovation: Alternating regular and shifted window attention
    - Regular windows: Efficient local attention
    - Shifted windows: Cross-window connections
    - Maintains linear complexity while enabling global modeling
    """
    
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4.0, qkv_bias=True, drop=0.1, attn_drop=0.1, drop_path=0.1):
        super(SwinTransformerBlock, self).__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            # If window size is larger than input resolution, no partitioning
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        
        assert 0 <= self.shift_size < self.window_size, "shift_size must be smaller than window_size"
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        
        # Window attention
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        
        # Drop path (stochastic depth)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # MLP
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        # Create attention mask for shifted windows
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # (1, H, W, 1)
            h_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            
            mask_windows = window_partition(img_mask, self.window_size)  # (nW, window_size, window_size, 1)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        
        self.register_buffer("attn_mask", attn_mask)
        
        is_shifted = "Shifted" if shift_size > 0 else "Regular"
        print(f"    Swin Block: {dim}D, {is_shifted} Window {window_size}x{window_size}")
    
    def forward(self, x):
        """Forward pass through Swin Transformer block"""
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Cyclic shift for shifted window attention
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # (nW*B, window_size, window_size, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # (nW*B, window_size*window_size, C)
        
        # Window/Shifted window attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # (nW*B, window_size*window_size, C)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # (B, H', W', C)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.view(B, H * W, C)
        
        # FFN with skip connection
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

# ============================================================================
# PATCH EMBEDDING FOR SWIN
# ============================================================================

class SwinPatchEmbedding(nn.Module):
    """
    Patch Embedding for Swin Transformer
    Similar to ViT but prepares for hierarchical processing
    """
    
    def __init__(self, img_size=224, patch_size=4, in_channels=3, embed_dim=96):
        super(SwinPatchEmbedding, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size // patch_size, img_size // patch_size]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Patch projection
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
        print(f"  Swin Patch Embedding:")
        print(f"    Image size: {img_size}x{img_size}")
        print(f"    Patch size: {patch_size}x{patch_size}")
        print(f"    Patches resolution: {self.patches_resolution}")
        print(f"    Number of patches: {self.num_patches}")
        print(f"    Embedding dimension: {embed_dim}")
    
    def forward(self, x):
        """Forward pass: Image → Patch embeddings"""
        B, C, H, W = x.shape
        
        # Patch projection
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.norm(x)
        
        return x

# ============================================================================
# SWIN TRANSFORMER STAGE
# ============================================================================

class SwinTransformerStage(nn.Module):
    """
    Swin Transformer Stage consisting of multiple Swin blocks
    Implements the core hierarchical processing with alternating attention patterns
    """
    
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4.0, qkv_bias=True, drop=0.1, attn_drop=0.1, drop_path=0.1,
                 downsample=None):
        super(SwinTransformerStage, self).__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        
        # Build blocks with alternating regular/shifted window attention
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path
            )
            for i in range(depth)
        ])
        
        # Patch merging layer for downsampling
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim)
        else:
            self.downsample = None
        
        print(f"  Swin Stage: {depth} blocks, {dim}D, {input_resolution} resolution")
    
    def forward(self, x):
        """Forward pass through Swin stage"""
        for block in self.blocks:
            x = block(x)
        
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x

# ============================================================================
# SWIN TRANSFORMER ARCHITECTURE
# ============================================================================

class SwinTransformer_Hierarchical(nn.Module):
    """
    Swin Transformer - Hierarchical Vision Transformer with Shifted Windows
    
    Revolutionary Innovation:
    - Hierarchical architecture with patch merging
    - Shifted window attention for linear complexity
    - Multi-scale feature representation
    - Efficient processing of high-resolution images
    """
    
    def __init__(self, img_size=224, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4.0, qkv_bias=True, drop_rate=0.1,
                 attn_drop_rate=0.1, drop_path_rate=0.1):
        super(SwinTransformer_Hierarchical, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        
        print(f"Building Swin Transformer with hierarchical architecture...")
        
        # Patch embedding
        self.patch_embed = SwinPatchEmbedding(
            img_size=img_size, patch_size=patch_size, 
            in_channels=in_channels, embed_dim=embed_dim
        )
        
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        
        # Absolute position embedding
        self.absolute_pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build hierarchical stages
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            stage = SwinTransformerStage(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None
            )
            self.layers.append(stage)
        
        # Final normalization and classification head
        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (self.num_layers - 1)))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(int(embed_dim * 2 ** (self.num_layers - 1)), num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        # Calculate statistics
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"Swin Transformer Architecture Summary:")
        print(f"  Input resolution: {img_size}x{img_size}")
        print(f"  Patch size: {patch_size}x{patch_size}")
        print(f"  Embedding dimension: {embed_dim}")
        print(f"  Depths: {depths}")
        print(f"  Number of heads: {num_heads}")
        print(f"  Window size: {window_size}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Key innovation: Hierarchical transformer with shifted windows")
    
    def _initialize_weights(self):
        """Initialize Swin Transformer weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize absolute position embedding
        nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)
    
    def forward(self, x):
        """Forward pass through Swin Transformer"""
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add absolute position embedding
        x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        
        # Hierarchical stages
        for layer in self.layers:
            x = layer(x)
        
        # Final processing
        x = self.norm(x)  # (B, L, C)
        x = self.avgpool(x.transpose(1, 2))  # (B, C, 1)
        x = torch.flatten(x, 1)
        x = self.head(x)
        
        return x
    
    def get_hierarchical_analysis(self):
        """Analyze hierarchical feature processing"""
        stages_info = []
        current_resolution = self.patches_resolution
        current_dim = self.embed_dim
        
        for i, (depth, num_heads) in enumerate(zip([2, 2, 6, 2], [3, 6, 12, 24])):
            stages_info.append({
                'stage': i + 1,
                'resolution': current_resolution.copy(),
                'dimension': current_dim,
                'depth': depth,
                'num_heads': num_heads,
                'complexity': f"O({current_resolution[0] * current_resolution[1]})"
            })
            
            # Update for next stage
            if i < 3:  # Not last stage
                current_resolution = [r // 2 for r in current_resolution]
                current_dim *= 2
        
        return {
            'stages': stages_info,
            'total_stages': len(stages_info),
            'feature_pyramid': 'Multi-scale hierarchical features',
            'complexity': 'Linear O(H*W) per stage'
        }

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_swin_transformer(model, train_loader, test_loader, epochs=80, learning_rate=1e-3):
    """Train Swin Transformer with appropriate optimization"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Swin Transformer training configuration
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.05,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduling
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    criterion = nn.CrossEntropyLoss()
    
    # Training tracking
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    model_name = model.__class__.__name__
    print(f"Training {model_name} on device: {device}")
    
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()
            
            if batch_idx % 400 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.4f}, LR: {current_lr:.6f}')
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_train_acc = 100. * correct_train / total_train
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_train_acc)
        
        # Test evaluation
        test_acc = evaluate_model(model, test_loader, device)
        test_accuracies.append(test_acc)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), f'AI-ML-DL/Models/CNN/swin_transformer_best.pth')
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {epoch_loss:.4f}, '
              f'Train Acc: {epoch_train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        # Early stopping for demonstration
        if test_acc > 88.0:
            print(f"Good performance reached at epoch {epoch+1}")
            break
    
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    return train_losses, train_accuracies, test_accuracies

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

# ============================================================================
# VISUALIZATION AND ANALYSIS
# ============================================================================

def visualize_swin_innovations():
    """Visualize Swin Transformer's hierarchical innovations"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Hierarchical architecture
    ax = axes[0, 0]
    ax.set_title('Hierarchical Feature Pyramid', fontsize=14)
    
    # Draw feature pyramid
    stage_sizes = [56, 28, 14, 7]  # Feature map sizes
    stage_dims = [96, 192, 384, 768]  # Channel dimensions
    colors = ['#3498DB', '#E67E22', '#9B59B6', '#E74C3C']
    
    for i, (size, dim, color) in enumerate(zip(stage_sizes, stage_dims, colors)):
        # Draw feature map representation
        rect_size = 0.8 * (size / 56)  # Scale relative to first stage
        x_pos = i * 2
        y_pos = 2 - rect_size / 2
        
        rect = plt.Rectangle((x_pos, y_pos), rect_size, rect_size, 
                           facecolor=color, alpha=0.7, edgecolor='black')
        ax.add_patch(rect)
        
        ax.text(x_pos + rect_size/2, y_pos - 0.3, f'Stage {i+1}', 
               ha='center', va='top', fontweight='bold')
        ax.text(x_pos + rect_size/2, y_pos - 0.5, f'{size}×{size}', 
               ha='center', va='top', fontsize=10)
        ax.text(x_pos + rect_size/2, y_pos - 0.7, f'{dim}D', 
               ha='center', va='top', fontsize=10)
        
        # Draw patch merging arrows
        if i < len(stage_sizes) - 1:
            ax.annotate('', xy=((i+1)*2 - 0.1, 2), xytext=(x_pos + rect_size + 0.1, 2),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
            ax.text(x_pos + rect_size + 0.5, 2.3, 'Patch\nMerging', 
                   ha='center', va='bottom', fontsize=8, style='italic')
    
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Window attention pattern
    ax = axes[0, 1]
    ax.set_title('Shifted Window Attention', fontsize=14)
    
    # Draw regular windows
    for i in range(2):
        for j in range(2):
            rect = plt.Rectangle((j*3, i*3), 2.8, 2.8, 
                               facecolor='lightblue', alpha=0.7, edgecolor='blue', linewidth=2)
            ax.add_patch(rect)
            ax.text(j*3 + 1.4, i*3 + 1.4, f'W{i*2+j+1}', 
                   ha='center', va='center', fontweight='bold')
    
    ax.text(3, -0.5, 'Regular Windows (Layer l)', ha='center', va='top', fontweight='bold')
    
    # Draw shifted windows
    shift = 1.4
    for i in range(2):
        for j in range(2):
            rect = plt.Rectangle((j*3 + shift, i*3 + shift + 7), 2.8, 2.8, 
                               facecolor='lightcoral', alpha=0.7, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(j*3 + shift + 1.4, i*3 + shift + 1.4 + 7, f'SW{i*2+j+1}', 
                   ha='center', va='center', fontweight='bold')
    
    ax.text(3 + shift, 6.5, 'Shifted Windows (Layer l+1)', ha='center', va='top', fontweight='bold')
    
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-1, 13)
    ax.axis('off')
    
    # Complexity comparison
    ax = axes[1, 0]
    models = ['ViT\n(Global)', 'Swin\n(Local)', 'CNN\n(Local)']
    complexities = [196*196, 49, 9]  # Attention complexities (simplified)
    
    bars = ax.bar(models, complexities, color=['#E74C3C', '#27AE60', '#3498DB'])
    ax.set_title('Attention Complexity Comparison', fontsize=14)
    ax.set_ylabel('Computational Complexity')
    ax.set_yscale('log')
    
    for bar, comp in zip(bars, complexities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2, 
                f'{comp}', ha='center', va='bottom')
    
    # Performance vs efficiency
    ax = axes[1, 1]
    models = ['ViT-Base', 'Swin-Tiny', 'Swin-Small', 'Swin-Base']
    params = [86, 29, 50, 88]  # Millions of parameters
    efficiency = [6, 9, 8, 7]  # Efficiency scores
    
    scatter = ax.scatter(params, efficiency, s=100, 
                        c=['#E74C3C', '#27AE60', '#F39C12', '#9B59B6'], alpha=0.7)
    
    for i, model in enumerate(models):
        ax.annotate(model, (params[i], efficiency[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax.set_title('Parameter Efficiency Trade-off', fontsize=14)
    ax.set_xlabel('Parameters (Millions)')
    ax.set_ylabel('Efficiency Score')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/CNN/013_swin_innovations.png', dpi=300, bbox_inches='tight')
    print("Swin Transformer innovations visualization saved: 013_swin_innovations.png")

def visualize_window_shifting_mechanism():
    """Visualize the window shifting mechanism"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Regular windows
    ax = axes[0]
    ax.set_title('Layer l: Regular Windows', fontsize=14, fontweight='bold')
    
    # Draw 8x8 grid with 4 windows
    for i in range(8):
        for j in range(8):
            color = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'][
                (i//4)*2 + (j//4)
            ]
            rect = plt.Rectangle((j, 7-i), 0.9, 0.9, 
                               facecolor=color, edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
    
    # Draw window boundaries
    for i in [0, 4, 8]:
        ax.axvline(x=i, color='blue', linewidth=3)
        ax.axhline(y=i, color='blue', linewidth=3)
    
    ax.text(2, -0.5, 'Window 1', ha='center', va='top', fontweight='bold', color='blue')
    ax.text(6, -0.5, 'Window 2', ha='center', va='top', fontweight='bold', color='green')
    ax.text(2, 8.5, 'Window 3', ha='center', va='bottom', fontweight='bold', color='orange')
    ax.text(6, 8.5, 'Window 4', ha='center', va='bottom', fontweight='bold', color='red')
    
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-1, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Shifted windows
    ax = axes[1]
    ax.set_title('Layer l+1: Shifted Windows', fontsize=14, fontweight='bold')
    
    # Draw shifted grid
    shift = 2
    window_colors = ['lightpink', 'lightsteelblue', 'lightseagreen', 'lightsalmon', 
                    'lightgray', 'lightgoldenrodyellow', 'lightcyan', 'lavender', 'mistyrose']
    
    # Create shifted window assignment
    for i in range(8):
        for j in range(8):
            # Determine shifted window
            shifted_i = (i + shift) % 8
            shifted_j = (j + shift) % 8
            window_id = (shifted_i // 4) * 3 + (shifted_j // 4)
            if shifted_i >= 4 and shifted_j >= 4:
                window_id = 0
            elif shifted_i >= 4:
                window_id = 1
            elif shifted_j >= 4:
                window_id = 2
            else:
                window_id = 3
            
            color = window_colors[window_id % len(window_colors)]
            rect = plt.Rectangle((j, 7-i), 0.9, 0.9, 
                               facecolor=color, edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
    
    # Draw shifted window boundaries (conceptual)
    ax.axvline(x=2, color='red', linewidth=2, linestyle='--')
    ax.axvline(x=6, color='red', linewidth=2, linestyle='--')
    ax.axhline(y=2, color='red', linewidth=2, linestyle='--')
    ax.axhline(y=6, color='red', linewidth=2, linestyle='--')
    
    ax.text(4, -0.5, 'Shifted by 2 pixels', ha='center', va='top', 
           fontweight='bold', color='red', style='italic')
    
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-1, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Cross-window connections
    ax = axes[2]
    ax.set_title('Cross-Window Information Flow', fontsize=14, fontweight='bold')
    
    # Draw conceptual information flow
    window_centers = [(1.5, 1.5), (5.5, 1.5), (1.5, 5.5), (5.5, 5.5)]
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, ((x, y), color) in enumerate(zip(window_centers, colors)):
        circle = plt.Circle((x, y), 0.8, facecolor=color, alpha=0.7, edgecolor='black')
        ax.add_patch(circle)
        ax.text(x, y, f'W{i+1}', ha='center', va='center', fontweight='bold', color='white')
    
    # Draw information flow arrows
    connections = [(0, 1), (0, 2), (1, 3), (2, 3)]
    for i, j in connections:
        x1, y1 = window_centers[i]
        x2, y2 = window_centers[j]
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='<->', lw=3, color='purple', alpha=0.7))
    
    ax.text(3.5, 0.5, 'Information exchange\nthrough shifted windows', 
           ha='center', va='center', fontsize=12, style='italic',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
    
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/CNN/013_window_shifting.png', dpi=300, bbox_inches='tight')
    print("Window shifting mechanism saved: 013_window_shifting.png")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"=== Swin Transformer Hierarchical Implementation ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load CIFAR-10 dataset
    train_loader, test_loader, classes = load_cifar10_dataset()
    
    # Initialize models
    swin_tiny = SwinTransformer_Hierarchical(
        img_size=224, patch_size=4, embed_dim=96, 
        depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]
    )
    
    swin_small = SwinTransformer_Hierarchical(
        img_size=224, patch_size=4, embed_dim=96,
        depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24]  # Deeper
    )
    
    # Compare model complexities
    swin_tiny_params = sum(p.numel() for p in swin_tiny.parameters())
    swin_small_params = sum(p.numel() for p in swin_small.parameters())
    
    print(f"\nModel Complexity Comparison:")
    print(f"  Swin-Tiny: {swin_tiny_params:,} parameters")
    print(f"  Swin-Small: {swin_small_params:,} parameters")
    print(f"  Parameter ratio: {swin_small_params/swin_tiny_params:.2f}x")
    
    # Analyze hierarchical processing
    hierarchical_analysis = swin_tiny.get_hierarchical_analysis()
    
    print(f"\nHierarchical Architecture Analysis:")
    for stage_info in hierarchical_analysis['stages']:
        print(f"  Stage {stage_info['stage']}: {stage_info['resolution'][0]}×{stage_info['resolution'][1]} "
              f"resolution, {stage_info['dimension']}D, {stage_info['depth']} blocks")
    
    # Generate visualizations
    print("\nGenerating Swin Transformer analysis...")
    visualize_swin_innovations()
    visualize_window_shifting_mechanism()
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("SWIN TRANSFORMER HIERARCHICAL SUMMARY")
    print("="*70)
    
    print(f"\nHistorical Context ({YEAR}):")
    print(f"  Innovation: {INNOVATION}")
    print(f"  Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"  Impact: {IMPACT}")
    
    print(f"\nSWIN TRANSFORMER REVOLUTIONARY INNOVATIONS:")
    print("="*50)
    print("1. HIERARCHICAL ARCHITECTURE:")
    print("   • Multi-stage feature pyramid like CNNs")
    print("   • Patch merging for spatial downsampling")
    print("   • Progressive channel dimension increase")
    print("   • Multi-scale feature representation")
    
    print("\n2. SHIFTED WINDOW ATTENTION:")
    print("   • Regular windows: Efficient local attention")
    print("   • Shifted windows: Cross-window connections")
    print("   • Linear complexity O(M²) instead of O(H²W²)")
    print("   • Maintains modeling power with efficiency")
    
    print("\n3. RELATIVE POSITION BIAS:")
    print("   • Learnable relative position embeddings")
    print("   • Better spatial awareness than absolute positions")
    print("   • Adaptive to different window configurations")
    print("   • Improves attention quality")
    
    print("\n4. COMPUTATIONAL EFFICIENCY:")
    print("   • Linear complexity with image size")
    print("   • Practical for high-resolution images")
    print("   • Efficient memory usage")
    print("   • Scalable to dense prediction tasks")
    
    print(f"\nTECHNICAL ACHIEVEMENTS:")
    print("="*40)
    print("• Solved ViT quadratic complexity problem")
    print("• Maintained hierarchical feature learning")
    print("• State-of-the-art on multiple vision benchmarks")
    print("• Enabled transformer use in dense prediction")
    print("• Linear complexity with image resolution")
    
    print(f"\nHIERARCHICAL PROCESSING STAGES:")
    for stage_info in hierarchical_analysis['stages']:
        print(f"  Stage {stage_info['stage']}: {stage_info['resolution'][0]}×{stage_info['resolution'][1]} "
              f"→ {stage_info['dimension']}D, {stage_info['depth']} blocks, {stage_info['num_heads']} heads")
    
    print(f"\nWINDOW ATTENTION MECHANISM:")
    print("• Layer l: Regular windows (4×4 or 7×7)")
    print("• Layer l+1: Shifted windows (offset by window_size/2)")
    print("• Alternating pattern throughout network")
    print("• Enables global modeling with local computation")
    print("• Relative position bias for spatial awareness")
    
    print(f"\nCOMPLEXITY COMPARISON:")
    print("• ViT: O(H²W²) - Quadratic with image size")
    print("• Swin: O(HW) - Linear with image size")
    print("• CNN: O(HW) - Linear but local receptive field")
    print("• Swin achieves best of both worlds")
    
    print(f"\nHISTORICAL IMPACT:")
    print("="*40)
    print("• Made transformers practical for computer vision")
    print("• Established hierarchical vision transformers")
    print("• Enabled dense prediction tasks (detection, segmentation)")
    print("• Inspired numerous efficient transformer variants")
    print("• Bridged gap between CNNs and transformers")
    print("• Demonstrated importance of inductive bias balance")
    
    print(f"\nSWIN VS VIT COMPARISON:")
    print("="*40)
    print("• ViT: Global attention, quadratic complexity")
    print("• Swin: Local + shifted attention, linear complexity")
    print("• ViT: Fixed resolution, single scale")
    print("• Swin: Hierarchical, multi-scale features")
    print("• ViT: Better for large datasets")
    print("• Swin: Better for diverse vision tasks")
    
    # Update TODO status
    print("\n" + "="*70)
    print("ERA 5: ATTENTION AND TRANSFORMER VISION COMPLETED")
    print("="*70)
    print("• Vision Transformer (ViT): Pure transformer for vision")
    print("• Swin Transformer: Hierarchical transformer with shifted windows")
    print("• Established transformer paradigm in computer vision")
    print("• Achieved superior scaling and efficiency properties")
    print("• Enabled multimodal and dense prediction applications")
    
    return {
        'model': 'Swin Transformer Hierarchical',
        'year': YEAR,
        'innovation': INNOVATION,
        'hierarchical_analysis': hierarchical_analysis,
        'parameter_comparison': {
            'swin_tiny': swin_tiny_params,
            'swin_small': swin_small_params
        }
    }

if __name__ == "__main__":
    results = main()