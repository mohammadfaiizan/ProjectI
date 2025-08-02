"""
ERA 5: ATTENTION AND TRANSFORMER VISION - Vision Transformer (ViT)
=================================================================

Year: 2020
Paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020)
Innovation: Pure transformer architecture for computer vision without convolutions
Previous Limitation: CNNs dominate vision, limited long-range dependencies, inductive biases
Performance Gain: Matches/exceeds CNNs on large datasets, superior scaling properties
Impact: Paradigm shift from convolutions to attention in computer vision

This file implements Vision Transformer that revolutionized computer vision by applying
pure transformer architecture to images, treating image patches as sequence tokens.
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

YEAR = "2020"
INNOVATION = "Pure transformer architecture for computer vision without convolutions"
PREVIOUS_LIMITATION = "CNNs dominate vision, limited long-range dependencies, translation equivariance constraints"
IMPACT = "Paradigm shift from convolutions to attention, superior scaling properties"

print(f"=== Vision Transformer (ViT) ({YEAR}) ===")
print(f"Innovation: {INNOVATION}")
print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
print(f"Impact: {IMPACT}")
print("="*60)

# ============================================================================
# STANDARDIZED DATASET LOADING (CIFAR-10)
# ============================================================================

def load_cifar10_dataset():
    """Load CIFAR-10 dataset with ViT-style preprocessing"""
    print("Loading CIFAR-10 dataset for Vision Transformer study...")
    
    # ViT-style preprocessing with appropriate augmentation
    transform_train = transforms.Compose([
        transforms.Resize(224),  # ViT expects larger images
        transforms.RandomCrop(224, padding=28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
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
    
    # Create data loaders with smaller batch size for ViT
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # CIFAR-10 class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print(f"Classes: {len(classes)}")
    print(f"Image size: 224x224 (upscaled from 32x32)")
    
    return train_loader, test_loader, classes

# ============================================================================
# PATCH EMBEDDING
# ============================================================================

class PatchEmbedding(nn.Module):
    """
    Patch Embedding - Core Innovation of ViT
    
    Convert image into sequence of flattened patches:
    1. Divide image into fixed-size patches
    2. Linearly embed each patch
    3. Add positional embeddings
    4. Prepend learnable class token
    
    Image (H, W, C) → Patches (N, P²·C) → Embeddings (N, D)
    """
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        
        # Linear projection of patches
        self.patch_proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # Class token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim)
        )
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        print(f"  Patch Embedding:")
        print(f"    Image size: {img_size}x{img_size}")
        print(f"    Patch size: {patch_size}x{patch_size}")
        print(f"    Number of patches: {self.num_patches}")
        print(f"    Embedding dimension: {embed_dim}")
        print(f"    Patch dimension: {self.patch_dim}")
    
    def forward(self, x):
        """
        Forward pass: Image → Patch embeddings
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Patch embeddings with class token (B, N+1, D)
        """
        B = x.shape[0]
        
        # Extract patches and project to embedding dimension
        # (B, C, H, W) → (B, embed_dim, H/P, W/P) → (B, embed_dim, N) → (B, N, embed_dim)
        x = self.patch_proj(x).flatten(2).transpose(1, 2)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Apply dropout
        x = self.dropout(x)
        
        return x

# ============================================================================
# MULTI-HEAD SELF ATTENTION
# ============================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self Attention - Heart of Transformer
    
    Attention mechanism for processing sequence of patches:
    1. Compute Query, Key, Value for each patch
    2. Calculate attention weights between all patch pairs
    3. Aggregate information based on attention weights
    4. Enable long-range dependencies
    """
    
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        
        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Scale factor for attention
        self.scale = self.head_dim ** -0.5
        
        print(f"    Multi-Head Attention: {num_heads} heads, {self.head_dim} dim per head")
    
    def forward(self, x):
        """
        Forward pass: Self-attention over patch sequence
        
        Args:
            x: Input sequence (B, N, D)
            
        Returns:
            Attended features (B, N, D)
        """
        B, N, D = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)
        
        # Scaled dot-product attention
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        x = (attn_weights @ v).transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x, attn_weights

# ============================================================================
# TRANSFORMER ENCODER BLOCK
# ============================================================================

class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block
    
    Standard transformer architecture:
    1. Multi-head self attention with residual connection
    2. Layer normalization
    3. MLP with residual connection
    4. Layer normalization
    """
    
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        
        # Multi-head self attention
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        print(f"    Transformer Block: {embed_dim}D, {num_heads} heads, {mlp_hidden_dim} MLP")
    
    def forward(self, x):
        """Forward pass through transformer block"""
        # Self attention with residual connection
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + attn_out
        
        # MLP with residual connection
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        
        return x, attn_weights

# ============================================================================
# VISION TRANSFORMER ARCHITECTURE
# ============================================================================

class VisionTransformer_ViT(nn.Module):
    """
    Vision Transformer (ViT) Architecture
    
    Revolutionary Innovation:
    - Pure transformer for computer vision (no convolutions)
    - Image patches as sequence tokens
    - Global self-attention for long-range dependencies
    - Superior scaling properties with large datasets
    """
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=10,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super(VisionTransformer_ViT, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        
        print(f"Building Vision Transformer (ViT)...")
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
        
        # Calculate statistics
        total_params = sum(p.numel() for p in self.parameters())
        num_patches = self.patch_embed.num_patches
        
        print(f"Vision Transformer Architecture Summary:")
        print(f"  Image size: {img_size}x{img_size}")
        print(f"  Patch size: {patch_size}x{patch_size}")
        print(f"  Number of patches: {num_patches}")
        print(f"  Embedding dimension: {embed_dim}")
        print(f"  Transformer depth: {depth}")
        print(f"  Number of heads: {num_heads}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Key innovation: Pure transformer for vision")
    
    def _initialize_weights(self):
        """Initialize ViT weights"""
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
    
    def forward(self, x):
        """Forward pass through Vision Transformer"""
        # Patch embedding
        x = self.patch_embed(x)  # (B, N+1, D)
        
        # Store attention weights for analysis
        attention_weights = []
        
        # Transformer encoder blocks
        for block in self.transformer_blocks:
            x, attn_weights = block(x)
            attention_weights.append(attn_weights)
        
        # Layer normalization
        x = self.norm(x)
        
        # Classification (use class token)
        cls_token = x[:, 0]  # First token is class token
        logits = self.head(cls_token)
        
        return logits
    
    def get_attention_maps(self, x):
        """Get attention maps for visualization"""
        # Patch embedding
        x = self.patch_embed(x)
        
        attention_maps = []
        
        # Forward through transformer blocks and collect attention
        for block in self.transformer_blocks:
            x, attn_weights = block(x)
            attention_maps.append(attn_weights.detach())
        
        return attention_maps
    
    def get_patch_analysis(self):
        """Analyze patch-based processing"""
        num_patches = self.patch_embed.num_patches
        patch_area = self.patch_size ** 2
        total_pixels = self.img_size ** 2
        
        return {
            'num_patches': num_patches,
            'patch_size': self.patch_size,
            'patch_area': patch_area,
            'total_pixels': total_pixels,
            'patches_per_side': self.img_size // self.patch_size,
            'receptive_field': 'Global (via attention)',
            'inductive_bias': 'Minimal (position embeddings only)'
        }

# ============================================================================
# COMPARISON: HYBRID CNN-TRANSFORMER
# ============================================================================

class HybridCNNTransformer(nn.Module):
    """
    Hybrid CNN-Transformer for comparison
    Uses CNN backbone for feature extraction before transformer
    """
    
    def __init__(self, num_classes=10, embed_dim=768, depth=6):
        super(HybridCNNTransformer, self).__init__()
        
        print("Building Hybrid CNN-Transformer for comparison...")
        
        # CNN backbone (similar to early CNN stages)
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Feature map to sequence projection
        self.feature_proj = nn.Linear(256, embed_dim)
        
        # Positional embedding for feature map patches
        # Feature map size after CNN: 14x14 (for 224x224 input)
        self.pos_embed = nn.Parameter(torch.randn(1, 196, embed_dim))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, 12, 4.0, 0.1)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Hybrid model parameters: {total_params:,}")
    
    def forward(self, x):
        # CNN feature extraction
        x = self.cnn_backbone(x)  # (B, 256, 14, 14)
        
        # Flatten spatial dimensions
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Project to embedding dimension
        x = self.feature_proj(x)  # (B, H*W, embed_dim)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Transformer processing
        for block in self.transformer_blocks:
            x, _ = block(x)
        
        # Global average pooling and classification
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.head(x)
        
        return x

# ============================================================================
# ATTENTION VISUALIZATION
# ============================================================================

def visualize_attention_patterns(model, data_loader, device, save_path):
    """Visualize attention patterns from ViT"""
    model.eval()
    
    # Get a batch of images
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    images = images[:4].to(device)  # Use first 4 images
    
    # Get attention maps
    with torch.no_grad():
        attention_maps = model.get_attention_maps(images)
    
    # Visualize attention from different layers
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    for img_idx in range(4):
        for layer_idx in [0, 3, 6, 11]:  # Sample layers
            if layer_idx < len(attention_maps):
                # Get attention from class token to patches
                attn = attention_maps[layer_idx][img_idx, 0, 0, 1:].reshape(14, 14)  # Skip class token
                
                ax = axes[img_idx, layer_idx // 3]
                
                # Show attention map
                im = ax.imshow(attn.cpu().numpy(), cmap='hot', interpolation='nearest')
                ax.set_title(f'Image {img_idx+1}, Layer {layer_idx+1}')
                ax.axis('off')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Attention visualization saved: {save_path}")

def analyze_patch_importance(model, data_loader, device):
    """Analyze which patches are most important"""
    model.eval()
    
    patch_importance = []
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            if batch_idx >= 10:  # Analyze first 10 batches
                break
                
            data = data.to(device)
            attention_maps = model.get_attention_maps(data)
            
            # Average attention from class token to patches across all layers
            avg_attention = torch.stack([attn[:, :, 0, 1:].mean(dim=1) for attn in attention_maps])
            avg_attention = avg_attention.mean(dim=0).mean(dim=0)  # Average across layers and batch
            
            patch_importance.append(avg_attention.cpu())
    
    # Combine all patch importance scores
    all_importance = torch.cat(patch_importance, dim=0).mean(dim=0)
    
    return all_importance.reshape(14, 14)  # Reshape to spatial grid

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_vision_transformer(model, train_loader, test_loader, epochs=100, learning_rate=3e-4):
    """Train Vision Transformer with proper optimization"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # ViT training configuration
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.05,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduling with warmup
    warmup_epochs = 10
    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * warmup_epochs
    
    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    
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
            
            # Gradient clipping for ViT stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()
            
            if batch_idx % 300 == 0:
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
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), f'AI-ML-DL/Models/CNN/vit_best.pth')
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {epoch_loss:.4f}, '
              f'Train Acc: {epoch_train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        # Early stopping for demonstration
        if test_acc > 85.0:
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

def visualize_vit_innovations():
    """Visualize Vision Transformer innovations"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Patch processing concept
    ax = axes[0, 0]
    ax.set_title('Patch-Based Processing', fontsize=14)
    
    # Draw image grid
    for i in range(4):
        for j in range(4):
            rect = plt.Rectangle((j, 3-i), 0.9, 0.9, 
                               facecolor='lightblue', edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
            ax.text(j+0.45, 3-i+0.45, f'P{i*4+j+1}', ha='center', va='center', fontweight='bold')
    
    ax.set_xlim(-0.1, 4.1)
    ax.set_ylim(-0.1, 4.1)
    ax.set_xlabel('Patches treated as sequence tokens')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    # Attention vs Convolution
    ax = axes[0, 1]
    architectures = ['CNN\n(Local)', 'ViT\n(Global)']
    receptive_fields = [3, 196]  # Local vs global attention
    
    bars = ax.bar(architectures, receptive_fields, color=['#E74C3C', '#27AE60'])
    ax.set_title('Receptive Field Comparison', fontsize=14)
    ax.set_ylabel('Effective Receptive Field')
    ax.set_yscale('log')
    
    for bar, field in zip(bars, receptive_fields):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                f'{field}', ha='center', va='bottom')
    
    # Scaling properties
    ax = axes[1, 0]
    dataset_sizes = ['1M', '10M', '100M', '1B']
    cnn_performance = [70, 75, 78, 79]  # Hypothetical performance
    vit_performance = [65, 80, 88, 95]  # Superior scaling
    
    x = np.arange(len(dataset_sizes))
    ax.plot(x, cnn_performance, 'o-', label='CNN', color='#E74C3C', linewidth=2)
    ax.plot(x, vit_performance, 's-', label='ViT', color='#27AE60', linewidth=2)
    
    ax.set_title('Scaling Properties', fontsize=14)
    ax.set_xlabel('Dataset Size')
    ax.set_ylabel('Performance (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_sizes)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Architecture comparison
    ax = axes[1, 1]
    models = ['ResNet-50', 'EfficientNet-B4', 'ViT-Base', 'ViT-Large']
    parameters = [25.6, 19.0, 86.0, 307.0]  # Millions
    colors = ['#95A5A6', '#3498DB', '#27AE60', '#F39C12']
    
    bars = ax.bar(models, parameters, color=colors)
    ax.set_title('Parameter Comparison', fontsize=14)
    ax.set_ylabel('Parameters (Millions)')
    ax.tick_params(axis='x', rotation=45)
    
    for bar, params in zip(bars, parameters):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{params}M', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/CNN/012_vit_innovations.png', dpi=300, bbox_inches='tight')
    print("ViT innovations visualization saved: 012_vit_innovations.png")

def visualize_transformer_vs_cnn():
    """Compare transformer and CNN processing"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # CNN processing
    ax = axes[0]
    ax.set_title('CNN: Hierarchical Local Processing', fontsize=14, fontweight='bold')
    
    # Draw CNN layers
    layer_sizes = [8, 6, 4, 2]
    layer_names = ['Conv1', 'Conv2', 'Conv3', 'Conv4']
    colors = ['#E74C3C', '#E67E22', '#F39C12', '#27AE60']
    
    for i, (size, name, color) in enumerate(zip(layer_sizes, layer_names, colors)):
        y_start = (8 - size) / 2
        for j in range(size):
            for k in range(size):
                rect = plt.Rectangle((i*2 + k*0.8/size, y_start + j*0.8/size), 
                                   0.7/size, 0.7/size, facecolor=color, alpha=0.7, edgecolor='black')
                ax.add_patch(rect)
        
        ax.text(i*2 + 0.4, -0.5, name, ha='center', va='top', fontweight='bold')
        
        # Draw arrows
        if i < len(layer_sizes) - 1:
            ax.annotate('', xy=((i+1)*2 - 0.1, 4), xytext=(i*2 + 0.9, 4),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.text(4, -1.2, 'Increasing abstraction\nDecreasing resolution', 
           ha='center', va='top', style='italic')
    
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-1.5, 8.5)
    ax.axis('off')
    
    # ViT processing
    ax = axes[1]
    ax.set_title('ViT: Global Attention Processing', fontsize=14, fontweight='bold')
    
    # Draw patch tokens
    patch_positions = [(i%4, i//4) for i in range(16)]
    
    for i, (x, y) in enumerate(patch_positions):
        circle = plt.Circle((x*1.5 + 1, y*1.5 + 1), 0.3, 
                          facecolor='lightblue', edgecolor='black', alpha=0.7)
        ax.add_patch(circle)
        ax.text(x*1.5 + 1, y*1.5 + 1, f'P{i+1}', ha='center', va='center', fontweight='bold')
    
    # Draw attention connections (sample)
    attention_pairs = [(0, 15), (5, 10), (3, 12), (8, 1)]
    for i, j in attention_pairs:
        x1, y1 = patch_positions[i][0]*1.5 + 1, patch_positions[i][1]*1.5 + 1
        x2, y2 = patch_positions[j][0]*1.5 + 1, patch_positions[j][1]*1.5 + 1
        ax.plot([x1, x2], [y1, y2], 'r--', alpha=0.6, linewidth=2)
    
    ax.text(3, -0.5, 'Global attention connections\nAcross all patches', 
           ha='center', va='top', style='italic')
    
    ax.set_xlim(0, 6)
    ax.set_ylim(-1, 6)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/CNN/012_transformer_vs_cnn.png', dpi=300, bbox_inches='tight')
    print("Transformer vs CNN comparison saved: 012_transformer_vs_cnn.png")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"=== Vision Transformer (ViT) Implementation ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load CIFAR-10 dataset
    train_loader, test_loader, classes = load_cifar10_dataset()
    
    # Initialize models
    vit_base = VisionTransformer_ViT(
        img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12
    )
    
    vit_small = VisionTransformer_ViT(
        img_size=224, patch_size=16, embed_dim=384, depth=6, num_heads=6
    )
    
    hybrid_model = HybridCNNTransformer()
    
    # Compare model complexities
    vit_base_params = sum(p.numel() for p in vit_base.parameters())
    vit_small_params = sum(p.numel() for p in vit_small.parameters())
    hybrid_params = sum(p.numel() for p in hybrid_model.parameters())
    
    print(f"\nModel Complexity Comparison:")
    print(f"  ViT-Base: {vit_base_params:,} parameters")
    print(f"  ViT-Small: {vit_small_params:,} parameters")
    print(f"  Hybrid CNN-Transformer: {hybrid_params:,} parameters")
    
    # Analyze patch processing
    patch_analysis = vit_base.get_patch_analysis()
    
    print(f"\nPatch Processing Analysis:")
    for key, value in patch_analysis.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Generate visualizations
    print("\nGenerating Vision Transformer analysis...")
    visualize_vit_innovations()
    visualize_transformer_vs_cnn()
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("VISION TRANSFORMER (VIT) SUMMARY")
    print("="*70)
    
    print(f"\nHistorical Context ({YEAR}):")
    print(f"  Innovation: {INNOVATION}")
    print(f"  Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"  Impact: {IMPACT}")
    
    print(f"\nVIT REVOLUTIONARY INNOVATIONS:")
    print("="*50)
    print("1. PATCH-BASED PROCESSING:")
    print("   • Treats image patches as sequence tokens")
    print("   • Linear embedding of flattened patches")
    print("   • Minimal inductive bias (only position embeddings)")
    print("   • Pure sequence processing without convolutions")
    
    print("\n2. GLOBAL SELF-ATTENTION:")
    print("   • Every patch attends to every other patch")
    print("   • Global receptive field from layer 1")
    print("   • Long-range dependency modeling")
    print("   • Attention-based feature aggregation")
    
    print("\n3. TRANSFORMER ARCHITECTURE:")
    print("   • Multi-head self-attention mechanisms")
    print("   • Position embeddings for spatial awareness")
    print("   • Learnable class token for classification")
    print("   • Layer normalization and residual connections")
    
    print("\n4. SUPERIOR SCALING PROPERTIES:")
    print("   • Performance improves with larger datasets")
    print("   • Outperforms CNNs on large-scale datasets")
    print("   • Better parameter efficiency at scale")
    print("   • Transfer learning capabilities")
    
    print(f"\nTECHNICAL ACHIEVEMENTS:")
    print("="*40)
    print("• First pure transformer for computer vision")
    print("• State-of-the-art performance on ImageNet")
    print("• Superior scaling with dataset size")
    print("• Global attention from first layer")
    print("• Minimal architectural inductive bias")
    
    print(f"\nPATCH PROCESSING INNOVATIONS:")
    print(f"  Image Size: {patch_analysis['total_pixels']} pixels")
    print(f"  Patch Size: {patch_analysis['patch_area']} pixels per patch") 
    print(f"  Number of Patches: {patch_analysis['num_patches']}")
    print(f"  Receptive Field: {patch_analysis['receptive_field']}")
    print(f"  Inductive Bias: {patch_analysis['inductive_bias']}")
    
    print(f"\nARCHITECTURAL COMPARISON:")
    print("• CNN: Local convolutions → hierarchical features")
    print("• ViT: Global attention → direct global processing")
    print("• CNN: Strong inductive bias (locality, translation equivariance)")
    print("• ViT: Minimal bias (learns spatial relationships)")
    print("• CNN: Better with small datasets")
    print("• ViT: Superior scaling with large datasets")
    
    print(f"\nHISTORICAL IMPACT:")
    print("="*40)
    print("• Paradigm shift from convolutions to attention")
    print("• Established transformers in computer vision")
    print("• Inspired numerous vision transformer variants")
    print("• Demonstrated importance of scale in vision")
    print("• Unified NLP and vision architectures")
    print("• Enabled multimodal models (CLIP, DALL-E)")
    
    print(f"\nVIT ARCHITECTURAL INSIGHTS:")
    print("="*40)
    print("• Patch embedding replaces convolution")
    print("• Position embedding provides spatial information")
    print("• Class token enables classification")
    print("• Global attention models long-range dependencies")
    print("• Transformer blocks process patch sequences")
    print("• No spatial downsampling (unlike CNNs)")
    
    return {
        'model': 'Vision Transformer (ViT)',
        'year': YEAR,
        'innovation': INNOVATION,
        'patch_analysis': patch_analysis,
        'parameter_comparison': {
            'vit_base': vit_base_params,
            'vit_small': vit_small_params,
            'hybrid': hybrid_params
        }
    }

if __name__ == "__main__":
    results = main()