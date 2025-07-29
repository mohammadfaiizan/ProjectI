import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import math

# Vision Transformer Components
class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Patch projection using conv2d
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x):
        # x: [batch_size, channels, height, width]
        x = self.proj(x)  # [batch_size, embed_dim, H//patch_size, W//patch_size]
        x = x.flatten(2)  # [batch_size, embed_dim, n_patches]
        x = x.transpose(1, 2)  # [batch_size, n_patches, embed_dim]
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # [batch_size, seq_len, 3 * embed_dim]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        out = self.proj(out)
        
        return out, attn_weights

class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual connection
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + attn_out
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x, attn_weights

class VisionTransformer(nn.Module):
    """Complete Vision Transformer model"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [batch_size, n_patches, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [batch_size, n_patches + 1, embed_dim]
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer blocks
        attention_weights = []
        for block in self.blocks:
            x, attn_weights = block(x)
            attention_weights.append(attn_weights)
        
        # Classification
        x = self.norm(x)
        cls_token_final = x[:, 0]  # Use class token for classification
        logits = self.head(cls_token_final)
        
        return logits, attention_weights

# Model variants
def vit_tiny(num_classes=1000):
    return VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, 
        mlp_ratio=4, num_classes=num_classes
    )

def vit_small(num_classes=1000):
    return VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, 
        mlp_ratio=4, num_classes=num_classes
    )

def vit_base(num_classes=1000):
    return VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, 
        mlp_ratio=4, num_classes=num_classes
    )

def vit_large(num_classes=1000):
    return VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, 
        mlp_ratio=4, num_classes=num_classes
    )

# Training example
def train_vit():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model
    model = vit_base(num_classes=10).to(device)
    
    # Data transforms for ViT
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Dummy dataset
    train_dataset = datasets.FakeData(size=1000, image_size=(3, 224, 224), 
                                    num_classes=10, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    
    model.train()
    for epoch in range(2):
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits, attention_weights = model(data)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

if __name__ == "__main__":
    # Test patch embedding
    patch_embed = PatchEmbedding(img_size=224, patch_size=16, embed_dim=768)
    x = torch.randn(4, 3, 224, 224)
    patches = patch_embed(x)
    print(f"Patch embedding output shape: {patches.shape}")
    
    # Test attention
    attn = MultiHeadAttention(embed_dim=768, num_heads=12)
    attn_out, weights = attn(patches)
    print(f"Attention output shape: {attn_out.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    # Test full model
    model = vit_base(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    logits, all_attention = model(x)
    print(f"Model output shape: {logits.shape}")
    print(f"Number of attention layers: {len(all_attention)}")
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Training example
    print("\nStarting training example...")
    train_vit()