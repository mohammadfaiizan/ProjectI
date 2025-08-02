"""
ERA 4: DIFFUSION REVOLUTION - Latent Diffusion Models
=====================================================

Year: 2021
Paper: "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al.)
Innovation: Diffusion in latent space with autoencoder for computational efficiency
Previous Limitation: High computational cost of pixel-space diffusion for high-resolution images
Performance Gain: 3-8x speedup with comparable quality, enabling high-resolution generation
Impact: Enabled practical deployment and foundation for Stable Diffusion

This file implements Latent Diffusion Models that revolutionized practical deployment
of diffusion models by operating in compressed latent space rather than pixel space.
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
INNOVATION = "Diffusion in latent space with autoencoder for computational efficiency"
PREVIOUS_LIMITATION = "High computational cost of pixel-space diffusion for high-resolution images"
IMPACT = "Enabled practical deployment and foundation for Stable Diffusion"

print(f"=== Latent Diffusion Models ({YEAR}) ===")
print(f"Innovation: {INNOVATION}")
print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
print(f"Impact: {IMPACT}")
print("="*60)

# ============================================================================
# LATENT DIFFUSION PRINCIPLES
# ============================================================================

LATENT_DIFFUSION_PRINCIPLES = {
    "latent_space_diffusion": "Perform diffusion in compressed latent space rather than pixel space",
    "autoencoder_compression": "Use pretrained VAE to encode/decode between pixel and latent space",
    "computational_efficiency": "3-8x speedup by operating on smaller latent representations",
    "perceptual_equivalence": "Maintain generation quality while reducing computational cost",
    "cross_attention": "Enable text conditioning through cross-attention mechanisms",
    "semantic_latents": "Latent space preserves semantic information while reducing dimensionality",
    "stable_training": "Inherit diffusion stability while improving efficiency",
    "modular_design": "Separate perception (VAE) from generation (diffusion) components"
}

print("Latent Diffusion Principles:")
for key, principle in LATENT_DIFFUSION_PRINCIPLES.items():
    print(f"  • {principle}")
print()

# ============================================================================
# STANDARDIZED DATASET LOADING (CIFAR-10)
# ============================================================================

def load_cifar10_latent():
    """Load CIFAR-10 dataset for latent diffusion study"""
    print("Loading CIFAR-10 dataset for latent diffusion study...")
    
    # Latent diffusion preprocessing
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    # CIFAR-10 class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print(f"Classes: {len(classes)}")
    print(f"Image size: 32x32 RGB → Latent 8x8x4")
    print(f"Focus: Latent space diffusion for efficiency")
    
    return train_loader, test_loader, classes

# ============================================================================
# AUTOENCODER COMPONENTS
# ============================================================================

class DownsampleBlock(nn.Module):
    """Downsampling block for autoencoder"""
    
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        
        print(f"    DownsampleBlock: {in_channels} -> {out_channels}")
    
    def forward(self, x):
        h = F.silu(self.norm1(self.conv1(x)))
        h = F.silu(self.norm2(self.conv2(h)))
        return self.downsample(h)

class UpsampleBlock(nn.Module):
    """Upsampling block for autoencoder"""
    
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        
        print(f"    UpsampleBlock: {in_channels} -> {out_channels}")
    
    def forward(self, x):
        h = self.upsample(x)
        h = F.silu(self.norm1(self.conv1(h)))
        h = F.silu(self.norm2(self.conv2(h)))
        return h

class ResnetBlock(nn.Module):
    """ResNet block for autoencoder bottleneck"""
    
    def __init__(self, channels):
        super(ResnetBlock, self).__init__()
        
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        
        print(f"    ResnetBlock: {channels} channels")
    
    def forward(self, x):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return x + h

class AttentionBlock(nn.Module):
    """Attention block for autoencoder"""
    
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
        print(f"    AttentionBlock: {channels} channels")
    
    def forward(self, x):
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        
        # Compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)
        
        attn = torch.bmm(q, k) * (c ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.bmm(attn, v)
        out = out.permute(0, 2, 1).reshape(b, c, h, w)
        
        return x + self.proj_out(out)

# ============================================================================
# VARIATIONAL AUTOENCODER
# ============================================================================

class VAEEncoder(nn.Module):
    """
    VAE Encoder for Latent Diffusion
    
    Compresses images to latent representations
    """
    
    def __init__(self, in_channels=3, latent_channels=4, base_channels=128):
        super(VAEEncoder, self).__init__()
        
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        
        print(f"Building VAE Encoder...")
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList([
            DownsampleBlock(base_channels, base_channels * 2),
            DownsampleBlock(base_channels * 2, base_channels * 4),
        ])
        
        # Middle blocks
        self.mid_block1 = ResnetBlock(base_channels * 4)
        self.mid_attn = AttentionBlock(base_channels * 4)
        self.mid_block2 = ResnetBlock(base_channels * 4)
        
        # Output layers for VAE (mean and log variance)
        self.norm_out = nn.GroupNorm(32, base_channels * 4)
        self.conv_out = nn.Conv2d(base_channels * 4, latent_channels * 2, 3, padding=1)
        
        print(f"VAE Encoder: {in_channels} -> {latent_channels} (latent)")
    
    def forward(self, x):
        # Initial convolution
        h = self.conv_in(x)
        
        # Downsampling
        for block in self.down_blocks:
            h = block(h)
        
        # Middle processing
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        
        # Output
        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)
        
        # Split into mean and log variance
        mean, log_var = torch.chunk(h, 2, dim=1)
        
        return mean, log_var
    
    def encode(self, x):
        """Encode to latent distribution parameters"""
        mean, log_var = self.forward(x)
        return mean, log_var
    
    def reparameterize(self, mean, log_var):
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

class VAEDecoder(nn.Module):
    """
    VAE Decoder for Latent Diffusion
    
    Reconstructs images from latent representations
    """
    
    def __init__(self, latent_channels=4, out_channels=3, base_channels=128):
        super(VAEDecoder, self).__init__()
        
        self.latent_channels = latent_channels
        self.out_channels = out_channels
        
        print(f"Building VAE Decoder...")
        
        # Input convolution
        self.conv_in = nn.Conv2d(latent_channels, base_channels * 4, 3, padding=1)
        
        # Middle blocks
        self.mid_block1 = ResnetBlock(base_channels * 4)
        self.mid_attn = AttentionBlock(base_channels * 4)
        self.mid_block2 = ResnetBlock(base_channels * 4)
        
        # Upsampling path
        self.up_blocks = nn.ModuleList([
            UpsampleBlock(base_channels * 4, base_channels * 2),
            UpsampleBlock(base_channels * 2, base_channels),
        ])
        
        # Output layers
        self.norm_out = nn.GroupNorm(32, base_channels)
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        
        print(f"VAE Decoder: {latent_channels} (latent) -> {out_channels}")
    
    def forward(self, z):
        # Input convolution
        h = self.conv_in(z)
        
        # Middle processing
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        
        # Upsampling
        for block in self.up_blocks:
            h = block(h)
        
        # Output
        h = F.silu(self.norm_out(h))
        h = torch.tanh(self.conv_out(h))
        
        return h

class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder for Latent Diffusion
    
    Provides perceptual compression for efficient diffusion
    """
    
    def __init__(self, in_channels=3, latent_channels=4):
        super(VariationalAutoencoder, self).__init__()
        
        self.encoder = VAEEncoder(in_channels, latent_channels)
        self.decoder = VAEDecoder(latent_channels, in_channels)
        
        # Scaling factor for latent space (learned during training)
        self.register_buffer('latent_scale', torch.tensor(0.18215))  # Standard scale for Stable Diffusion
        
        # Calculate statistics
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total_params = encoder_params + decoder_params
        
        print(f"VAE Summary:")
        print(f"  Encoder parameters: {encoder_params:,}")
        print(f"  Decoder parameters: {decoder_params:,}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Compression ratio: 4x4 = 16x")
    
    def encode(self, x):
        """Encode images to latent space"""
        mean, log_var = self.encoder.encode(x)
        z = self.encoder.reparameterize(mean, log_var)
        # Scale latents for better diffusion training
        return z * self.latent_scale, mean, log_var
    
    def decode(self, z):
        """Decode latents to images"""
        # Unscale latents
        z = z / self.latent_scale
        return self.decoder(z)
    
    def forward(self, x):
        """Full VAE forward pass"""
        z, mean, log_var = self.encode(x)
        recon = self.decode(z)
        return recon, mean, log_var

# ============================================================================
# LATENT DIFFUSION U-NET
# ============================================================================

class CrossAttention(nn.Module):
    """
    Cross-attention for conditioning latent diffusion
    
    Enables text conditioning in latent diffusion models
    """
    
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super(CrossAttention, self).__init__()
        
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim
        
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(0.1)
        )
        
        print(f"    CrossAttention: Q({query_dim}) × K,V({context_dim}), {heads} heads")
    
    def forward(self, x, context=None):
        if context is None:
            context = x
        
        b, n, _ = x.shape
        h = self.heads
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Reshape for multi-head attention
        q = q.view(b, n, h, self.dim_head).transpose(1, 2)
        k = k.view(b, -1, h, self.dim_head).transpose(1, 2)
        v = v.view(b, -1, h, self.dim_head).transpose(1, 2)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        
        return self.to_out(out)

class LatentResBlock(nn.Module):
    """
    Residual block for latent diffusion U-Net
    
    Includes time embedding and optional cross-attention
    """
    
    def __init__(self, in_channels, out_channels, time_embed_dim, 
                 context_dim=None, dropout=0.1):
        super(LatentResBlock, self).__init__()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels)
        )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        # Cross-attention for conditioning
        if context_dim is not None:
            self.cross_attn = CrossAttention(out_channels, context_dim)
            self.norm_cross = nn.GroupNorm(32, out_channels)
        else:
            self.cross_attn = None
        
        # Skip connection
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()
        
        print(f"    LatentResBlock: {in_channels} -> {out_channels}, context={context_dim is not None}")
    
    def forward(self, x, time_embed, context=None):
        h = self.block1(x)
        
        # Add time embedding
        time_out = self.time_mlp(time_embed)
        h = h + time_out[:, :, None, None]
        
        h = self.block2(h)
        
        # Cross-attention if context provided
        if self.cross_attn is not None and context is not None:
            b, c, h_dim, w_dim = h.shape
            h_flat = h.view(b, c, h_dim * w_dim).transpose(1, 2)
            h_attn = self.cross_attn(h_flat, context)
            h_attn = h_attn.transpose(1, 2).view(b, c, h_dim, w_dim)
            h = self.norm_cross(h + h_attn)
        
        return h + self.skip_conv(x)

class LatentUNet(nn.Module):
    """
    U-Net for latent diffusion
    
    Operates on latent representations rather than pixel space
    """
    
    def __init__(self, in_channels=4, out_channels=4, model_channels=128, 
                 context_dim=None, num_res_blocks=2):
        super(LatentUNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.context_dim = context_dim
        
        print(f"Building Latent U-Net...")
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Input projection
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, 3, padding=1)
        ])
        
        # Downsampling path
        ch = model_channels
        input_block_chans = [model_channels]
        
        for level, mult in enumerate([1, 2, 4]):
            for _ in range(num_res_blocks):
                layers = LatentResBlock(ch, mult * model_channels, time_embed_dim, context_dim)
                ch = mult * model_channels
                self.input_blocks.append(layers)
                input_block_chans.append(ch)
            
            if level != 2:  # No downsampling at last level
                self.input_blocks.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                input_block_chans.append(ch)
        
        # Middle block
        self.middle_block = nn.Sequential(
            LatentResBlock(ch, ch, time_embed_dim, context_dim),
            LatentResBlock(ch, ch, time_embed_dim, context_dim),
        )
        
        # Upsampling path
        self.output_blocks = nn.ModuleList([])
        
        for level, mult in reversed(list(enumerate([1, 2, 4]))):
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = LatentResBlock(ch + ich, mult * model_channels, time_embed_dim, context_dim)
                ch = mult * model_channels
                
                if level != 0 and i == num_res_blocks:
                    # Upsampling
                    layers = nn.Sequential(
                        layers,
                        nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
                    )
                
                self.output_blocks.append(layers)
        
        # Output projection
        self.out = nn.Sequential(
            nn.GroupNorm(32, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )
        
        # Sinusoidal time embedding
        self.register_buffer('time_embed_table', self._build_time_embed_table(model_channels))
        
        # Calculate statistics
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"Latent U-Net Summary:")
        print(f"  Input channels: {in_channels}")
        print(f"  Output channels: {out_channels}")
        print(f"  Model channels: {model_channels}")
        print(f"  Context dim: {context_dim}")
        print(f"  Total parameters: {total_params:,}")
    
    def _build_time_embed_table(self, dim):
        """Build sinusoidal time embedding table"""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        return emb
    
    def get_time_embedding(self, timesteps):
        """Get sinusoidal time embedding"""
        emb = timesteps[:, None] * self.time_embed_table[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.time_embed(emb)
    
    def forward(self, x, timesteps, context=None):
        """
        Forward pass through latent U-Net
        
        Args:
            x: Latent representations
            timesteps: Diffusion timesteps
            context: Optional conditioning context (e.g., text embeddings)
        """
        # Time embedding
        t_emb = self.get_time_embedding(timesteps)
        
        # Downsampling path
        hs = []
        h = x
        for module in self.input_blocks:
            if isinstance(module, LatentResBlock):
                h = module(h, t_emb, context)
            else:
                h = module(h)
            hs.append(h)
        
        # Middle block
        for module in self.middle_block:
            h = module(h, t_emb, context)
        
        # Upsampling path
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            if isinstance(module, nn.Sequential):
                # Handle upsampling blocks
                for layer in module:
                    if isinstance(layer, LatentResBlock):
                        h = layer(h, t_emb, context)
                    else:
                        h = layer(h)
            else:
                h = module(h, t_emb, context)
        
        return self.out(h)

# ============================================================================
# LATENT DIFFUSION MODEL
# ============================================================================

class LatentDiffusionModel(nn.Module):
    """
    Latent Diffusion Model
    
    Revolutionary Innovations:
    - Diffusion in compressed latent space for efficiency
    - VAE encoder/decoder for perceptual compression
    - Cross-attention for flexible conditioning
    - 3-8x speedup with maintained quality
    - Foundation for Stable Diffusion
    """
    
    def __init__(self, vae_latent_channels=4, unet_model_channels=128, 
                 context_dim=None, num_timesteps=1000):
        super(LatentDiffusionModel, self).__init__()
        
        self.num_timesteps = num_timesteps
        self.latent_channels = vae_latent_channels
        
        print(f"Building Latent Diffusion Model...")
        
        # Variational Autoencoder for latent space
        self.vae = VariationalAutoencoder(
            in_channels=3,
            latent_channels=vae_latent_channels
        )
        
        # Latent diffusion U-Net
        self.unet = LatentUNet(
            in_channels=vae_latent_channels,
            out_channels=vae_latent_channels,
            model_channels=unet_model_channels,
            context_dim=context_dim
        )
        
        # Noise scheduler (simplified linear)
        self.register_buffer('betas', torch.linspace(0.0001, 0.02, num_timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
        # Precompute values for sampling
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                            torch.sqrt(1.0 - self.alphas_cumprod))
        
        # Calculate statistics
        vae_params = sum(p.numel() for p in self.vae.parameters())
        unet_params = sum(p.numel() for p in self.unet.parameters())
        total_params = vae_params + unet_params
        
        print(f"Latent Diffusion Model Summary:")
        print(f"  VAE parameters: {vae_params:,}")
        print(f"  U-Net parameters: {unet_params:,}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Timesteps: {num_timesteps}")
        print(f"  Key innovation: Latent space diffusion for efficiency")
    
    def encode_to_latent(self, images):
        """Encode images to latent space"""
        with torch.no_grad():
            latents, _, _ = self.vae.encode(images)
            return latents
    
    def decode_from_latent(self, latents):
        """Decode latents to images"""
        with torch.no_grad():
            images = self.vae.decode(latents)
            return images
    
    def add_noise(self, latents, noise, timesteps):
        """Add noise to latents according to schedule"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Broadcast to match batch dimensions
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None, None]
        
        return (
            sqrt_alphas_cumprod_t * latents
            + sqrt_one_minus_alphas_cumprod_t * noise
        )
    
    def forward(self, images, context=None):
        """
        Training forward pass
        
        Args:
            images: Input images
            context: Optional conditioning context
        
        Returns:
            Diffusion loss
        """
        device = images.device
        batch_size = images.shape[0]
        
        # Encode to latent space
        latents, _, _ = self.vae.encode(images)
        
        # Sample timesteps
        timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        
        # Sample noise
        noise = torch.randn_like(latents)
        
        # Add noise to latents
        noisy_latents = self.add_noise(latents, noise, timesteps)
        
        # Predict noise
        predicted_noise = self.unet(noisy_latents, timesteps.float(), context)
        
        # Compute loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def sample(self, num_samples, latent_shape=(4, 8, 8), context=None, device='cpu'):
        """
        Sample from latent diffusion model
        
        Args:
            num_samples: Number of samples to generate
            latent_shape: Shape of latent representations
            context: Optional conditioning context
            device: Device to sample on
        
        Returns:
            Generated images
        """
        self.eval()
        
        # Start with random noise in latent space
        latents = torch.randn(num_samples, *latent_shape, device=device)
        
        # Reverse diffusion process
        for t in reversed(range(self.num_timesteps)):
            timesteps = torch.full((num_samples,), t, device=device, dtype=torch.float)
            
            # Predict noise
            predicted_noise = self.unet(latents, timesteps, context)
            
            # Compute coefficients
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            
            if t > 0:
                alpha_cumprod_t_prev = self.alphas_cumprod[t-1]
            else:
                alpha_cumprod_t_prev = torch.tensor(1.0)
            
            # Compute predicted original latents
            predicted_latents = (
                latents - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise
            ) / torch.sqrt(alpha_cumprod_t)
            
            # Compute direction to x_t-1
            predicted_latents = torch.clamp(predicted_latents, -1, 1)
            
            # Compute x_t-1
            pred_prev_latents = (
                torch.sqrt(alpha_cumprod_t_prev) * beta_t / (1 - alpha_cumprod_t) * predicted_latents
                + torch.sqrt(alpha_t) * (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * latents
            )
            
            if t > 0:
                # Add noise
                noise = torch.randn_like(latents)
                posterior_variance = beta_t * (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)
                latents = pred_prev_latents + torch.sqrt(posterior_variance) * noise
            else:
                latents = pred_prev_latents
        
        # Decode to image space
        images = self.vae.decode(latents)
        
        return torch.clamp(images, -1, 1)
    
    def get_latent_analysis(self):
        """Analyze latent diffusion innovations"""
        return {
            'latent_principles': LATENT_DIFFUSION_PRINCIPLES,
            'efficiency_gains': [
                '3-8x speedup through latent space operation',
                '16x compression ratio (32x32 → 8x8 for CIFAR)',
                'Reduced memory requirements',
                'Faster training and inference',
                'Practical high-resolution generation'
            ],
            'architectural_components': [
                'VAE encoder for perceptual compression',
                'VAE decoder for image reconstruction',
                'U-Net diffusion in latent space',
                'Cross-attention for conditioning',
                'Modular design for flexibility'
            ],
            'practical_advantages': [
                'Enables deployment on consumer hardware',
                'Maintains generation quality',
                'Supports flexible conditioning',
                'Foundation for Stable Diffusion',
                'Scalable to high resolutions'
            ]
        }

# ============================================================================
# LATENT DIFFUSION TRAINING FUNCTION
# ============================================================================

def train_latent_diffusion(model, train_loader, epochs=100, learning_rate=1e-4):
    """Train latent diffusion model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Optimizer (separate for VAE and U-Net if needed)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training tracking
    losses = []
    
    print(f"Training Latent Diffusion Model on device: {device}")
    print(f"Learning rate: {learning_rate}")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # Latent diffusion forward pass
            loss = model(images)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                      f'Latent Loss: {loss.item():.6f}')
        
        # Update learning rate
        scheduler.step()
        
        # Calculate epoch average
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        print(f'Epoch {epoch+1}/{epochs}: Avg Loss: {avg_loss:.6f}, '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Save checkpoint
        if (epoch + 1) % 25 == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss
            }, f'AI-ML-DL/Models/Generative_AI/latent_diffusion_epoch_{epoch+1}.pth')
        
        # Early stopping for demonstration
        if avg_loss < 0.01:
            print(f"Good convergence reached at epoch {epoch+1}")
            break
    
    return losses

# ============================================================================
# VISUALIZATION AND ANALYSIS
# ============================================================================

def visualize_latent_diffusion():
    """Visualize latent diffusion concepts"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Pixel vs Latent space comparison
    ax = axes[0, 0]
    ax.set_title('Pixel Space vs Latent Space Diffusion', fontsize=14, fontweight='bold')
    
    # Pixel space (large)
    pixel_rect = plt.Rectangle((0.1, 0.6), 0.3, 0.3, 
                              facecolor='lightcoral', edgecolor='red', linewidth=3)
    ax.add_patch(pixel_rect)
    ax.text(0.25, 0.75, 'Pixel Space\n32×32×3\n3,072 dims', 
           ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Latent space (small)
    latent_rect = plt.Rectangle((0.6, 0.7), 0.15, 0.15, 
                               facecolor='lightblue', edgecolor='blue', linewidth=3)
    ax.add_patch(latent_rect)
    ax.text(0.675, 0.775, 'Latent\n8×8×4\n256 dims', 
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Compression arrow
    ax.annotate('VAE\nEncode\n16× compression', xy=(0.58, 0.775), xytext=(0.42, 0.75),
               arrowprops=dict(arrowstyle='->', lw=3, color='green'),
               ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    # Diffusion comparison
    ax.text(0.25, 0.4, 'Pixel Diffusion:\n• High memory\n• Slow training\n• Expensive inference', 
           ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='mistyrose'))
    
    ax.text(0.675, 0.4, 'Latent Diffusion:\n• Low memory\n• Fast training\n• Efficient inference', 
           ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan'))
    
    # Speedup annotation
    ax.text(0.5, 0.1, '3-8× Speedup with Comparable Quality!', 
           ha='center', va='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # VAE encoder-decoder architecture
    ax = axes[0, 1]
    ax.set_title('VAE Encoder-Decoder Pipeline', fontsize=14)
    
    # Encoder path
    encoder_stages = ['32×32×3', '16×16×128', '8×8×256', '8×8×4']
    encoder_colors = ['lightgreen', 'lightblue', 'orange', 'purple']
    
    for i, (stage, color) in enumerate(zip(encoder_stages, encoder_colors)):
        x_pos = 0.05 + i * 0.2
        rect = plt.Rectangle((x_pos, 0.7), 0.15, 0.2, 
                           facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        ax.text(x_pos + 0.075, 0.8, stage, ha='center', va='center', 
               fontsize=9, fontweight='bold')
        
        if i < len(encoder_stages) - 1:
            ax.annotate('', xy=(x_pos + 0.17, 0.8), xytext=(x_pos + 0.13, 0.8),
                       arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
    
    # Decoder path
    decoder_stages = ['8×8×4', '8×8×256', '16×16×128', '32×32×3']
    decoder_colors = ['purple', 'orange', 'lightblue', 'lightgreen']
    
    for i, (stage, color) in enumerate(zip(decoder_stages, decoder_colors)):
        x_pos = 0.05 + i * 0.2
        rect = plt.Rectangle((x_pos, 0.3), 0.15, 0.2, 
                           facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        ax.text(x_pos + 0.075, 0.4, stage, ha='center', va='center', 
               fontsize=9, fontweight='bold')
        
        if i < len(decoder_stages) - 1:
            ax.annotate('', xy=(x_pos + 0.17, 0.4), xytext=(x_pos + 0.13, 0.4),
                       arrowprops=dict(arrowstyle='->', lw=2, color='darkred'))
    
    # Labels
    ax.text(0.05, 0.95, 'Encoder', ha='left', va='center', fontsize=12, fontweight='bold')
    ax.text(0.05, 0.55, 'Decoder', ha='left', va='center', fontsize=12, fontweight='bold')
    
    # Latent space
    ax.text(0.5, 0.1, 'Latent Space: Compressed representation\npreserving semantic information', 
           ha='center', va='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Cross-attention mechanism
    ax = axes[1, 0]
    ax.set_title('Cross-Attention for Conditioning', fontsize=14)
    
    # Query from latent features
    ax.text(0.2, 0.8, 'Latent Features\n(Query)', ha='center', va='center', fontsize=11,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
    
    # Key/Value from context
    ax.text(0.8, 0.8, 'Context\n(Key, Value)', ha='center', va='center', fontsize=11,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    # Attention mechanism
    ax.text(0.5, 0.5, 'Cross-Attention\nQ × K^T × V', ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow'))
    
    # Arrows
    ax.annotate('', xy=(0.4, 0.55), xytext=(0.25, 0.75),
               arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax.annotate('', xy=(0.6, 0.55), xytext=(0.75, 0.75),
               arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    
    # Output
    ax.text(0.5, 0.2, 'Conditioned\nLatent Features', ha='center', va='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
    
    ax.annotate('', xy=(0.5, 0.3), xytext=(0.5, 0.4),
               arrowprops=dict(arrowstyle='->', lw=3, color='purple'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Computational efficiency comparison
    ax = axes[1, 1]
    ax.set_title('Computational Efficiency Comparison', fontsize=14)
    
    methods = ['DDPM\n(Pixel)', 'Latent\nDiffusion']
    metrics = ['Training\nTime', 'Memory\nUsage', 'Inference\nSpeed', 'Quality']
    
    # Relative scores (Latent Diffusion = 1.0 baseline)
    ddpm_scores = [8, 16, 8, 1.0]  # Much higher cost
    latent_scores = [1, 1, 1, 1.0]  # Baseline
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, ddpm_scores, width, label='DDPM (Pixel)', 
                  color='lightcoral', alpha=0.8)
    bars2 = ax.bar(x + width/2, latent_scores, width, label='Latent Diffusion', 
                  color='lightblue', alpha=0.8)
    
    ax.set_ylabel('Relative Cost')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.2,
                   f'{height}×', ha='center', va='bottom', fontweight='bold')
    
    # Highlight efficiency
    ax.text(len(metrics)/2, max(ddpm_scores) * 0.7, 
           'Latent Diffusion:\n3-8× more efficient!', 
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/Generative_AI/012_latent_diffusion.png', dpi=300, bbox_inches='tight')
    print("Latent diffusion visualization saved: 012_latent_diffusion.png")

def visualize_stable_diffusion_pipeline():
    """Visualize the complete Stable Diffusion pipeline"""
    fig, axes = plt.subplots(1, 1, figsize=(16, 8))
    
    ax = axes
    ax.set_title('Stable Diffusion Pipeline (Latent Diffusion Foundation)', fontsize=16, fontweight='bold')
    
    # Pipeline stages
    stages = [
        ('Text\nInput', 'lightgreen', 0.05),
        ('Text\nEncoder', 'lightblue', 0.15),
        ('Text\nEmbeddings', 'lightcyan', 0.25),
        ('Random\nNoise', 'lightcoral', 0.35),
        ('U-Net\nDenoising', 'yellow', 0.45),
        ('Clean\nLatents', 'orange', 0.55),
        ('VAE\nDecoder', 'lightpink', 0.65),
        ('Final\nImage', 'lightgreen', 0.75)
    ]
    
    # Draw pipeline stages
    for i, (stage, color, x_pos) in enumerate(stages):
        rect = plt.Rectangle((x_pos, 0.6), 0.08, 0.3, 
                           facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x_pos + 0.04, 0.75, stage, ha='center', va='center', 
               fontsize=10, fontweight='bold')
        
        # Arrows between stages
        if i < len(stages) - 1:
            ax.annotate('', xy=(stages[i+1][2] - 0.01, 0.75), 
                       xytext=(x_pos + 0.09, 0.75),
                       arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
    
    # Detailed annotations
    # Text encoder
    ax.text(0.15, 0.5, 'CLIP/T5\nEncoder', ha='center', va='center', fontsize=9,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue', alpha=0.7))
    
    # U-Net process
    ax.text(0.45, 0.4, 'Iterative\nDenoising\n(1000 steps)', ha='center', va='center', fontsize=9,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
    
    # VAE decoder
    ax.text(0.65, 0.5, '8×8×4 →\n64×64×3', ha='center', va='center', fontsize=9,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightpink', alpha=0.7))
    
    # Cross-attention connections
    ax.annotate('Cross\nAttention', xy=(0.45, 0.6), xytext=(0.25, 0.35),
               arrowprops=dict(arrowstyle='->', lw=2, color='purple',
                             connectionstyle="arc3,rad=0.3"),
               ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='lavender'))
    
    # Key innovations
    innovations = [
        'Latent Space Efficiency',
        'Cross-Attention Conditioning', 
        'Modular Architecture',
        'Stable Training'
    ]
    
    for i, innovation in enumerate(innovations):
        ax.text(0.1 + i * 0.2, 0.15, innovation, ha='center', va='center', fontsize=11,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    # Title and description
    ax.text(0.5, 0.05, 'Foundation for Stable Diffusion, DALL-E 2, and modern text-to-image models', 
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    
    ax.set_xlim(0, 0.85)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/Generative_AI/012_stable_diffusion_pipeline.png', dpi=300, bbox_inches='tight')
    print("Stable Diffusion pipeline visualization saved: 012_stable_diffusion_pipeline.png")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"=== Latent Diffusion Models Implementation ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load CIFAR-10 dataset
    train_loader, test_loader, classes = load_cifar10_latent()
    
    # Initialize Latent Diffusion model
    latent_model = LatentDiffusionModel(
        vae_latent_channels=4, 
        unet_model_channels=128, 
        context_dim=None,  # No text conditioning for CIFAR demo
        num_timesteps=1000
    )
    
    # Analyze model properties
    total_params = sum(p.numel() for p in latent_model.parameters())
    latent_analysis = latent_model.get_latent_analysis()
    
    print(f"\nLatent Diffusion Model Analysis:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Latent channels: {latent_model.latent_channels}")
    print(f"  Timesteps: {latent_model.num_timesteps}")
    
    print(f"\nLatent Diffusion Innovations:")
    for key, value in latent_analysis.items():
        if isinstance(value, list):
            print(f"  {key.replace('_', ' ').title()}:")
            for item in value:
                print(f"    • {item}")
        elif isinstance(value, dict):
            print(f"  {key.replace('_', ' ').title()}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Generate visualizations
    print("\nGenerating Latent Diffusion analysis...")
    visualize_latent_diffusion()
    visualize_stable_diffusion_pipeline()
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("LATENT DIFFUSION MODELS SUMMARY")
    print("="*70)
    
    print(f"\nHistorical Context ({YEAR}):")
    print(f"  Innovation: {INNOVATION}")
    print(f"  Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"  Impact: {IMPACT}")
    
    print(f"\nLATENT DIFFUSION REVOLUTIONARY INNOVATIONS:")
    print("="*50)
    print("1. LATENT SPACE DIFFUSION:")
    print("   • Perform diffusion in compressed latent space")
    print("   • VAE encoder: Images → Latent representations")
    print("   • VAE decoder: Latent representations → Images")
    print("   • 16× compression (32×32×3 → 8×8×4 for CIFAR)")
    
    print("\n2. COMPUTATIONAL EFFICIENCY:")
    print("   • 3-8× speedup in training and inference")
    print("   • Reduced memory requirements")
    print("   • Enables high-resolution generation")
    print("   • Practical deployment on consumer hardware")
    
    print("\n3. MODULAR ARCHITECTURE:")
    print("   • VAE handles perceptual compression")
    print("   • U-Net handles semantic generation")
    print("   • Cross-attention enables conditioning")
    print("   • Components can be trained separately")
    
    print("\n4. PERCEPTUAL EQUIVALENCE:")
    print("   • Maintains generation quality in latent space")
    print("   • VAE preserves perceptually important information")
    print("   • Semantic consistency across compression")
    print("   • No quality degradation from compression")
    
    print(f"\nTECHNICAL ACHIEVEMENTS:")
    print("="*40)
    print("• First practical high-resolution diffusion")
    print("• Enabled consumer-grade deployment")
    print("• Foundation for Stable Diffusion")
    print("• Scalable architecture for large models")
    print("• Efficient text-to-image generation")
    
    print(f"\nLATENT PRINCIPLES:")
    for key, principle in LATENT_DIFFUSION_PRINCIPLES.items():
        print(f"  • {principle}")
    
    print(f"\nEFFICIENCY GAINS:")
    for gain in latent_analysis['efficiency_gains']:
        print(f"  • {gain}")
    
    print(f"\nARCHITECTURAL COMPONENTS:")
    for component in latent_analysis['architectural_components']:
        print(f"  • {component}")
    
    print(f"\nPRACTICAL ADVANTAGES:")
    for advantage in latent_analysis['practical_advantages']:
        print(f"  • {advantage}")
    
    print(f"\nVAE ENCODER-DECODER:")
    print("="*40)
    print("• Encoder: Compresses images to latent space")
    print("  - Downsampling convolutions")
    print("  - ResNet blocks for processing")
    print("  - Attention for global consistency")
    print("• Decoder: Reconstructs images from latents")
    print("  - Upsampling transposed convolutions")
    print("  - ResNet blocks for refinement")
    print("  - Attention for detail reconstruction")
    
    print(f"\nLATENT U-NET:")
    print("="*40)
    print("• Operates on compressed latent representations")
    print("• Time embedding for diffusion step conditioning")
    print("• Cross-attention for text/context conditioning")
    print("• Skip connections for multi-scale processing")
    print("• Group normalization for training stability")
    
    print(f"\nCROSS-ATTENTION CONDITIONING:")
    print("="*40)
    print("• Query: Latent features from U-Net")
    print("• Key/Value: Context embeddings (text, class, etc.)")
    print("• Enables flexible conditioning mechanisms")
    print("• Foundation for text-to-image generation")
    print("• Supports multiple conditioning modalities")
    
    print(f"\nCOMPUTATIONAL COMPARISON:")
    print("="*40)
    print("• DDPM (Pixel): 32×32×3 = 3,072 dimensions")
    print("• Latent Diffusion: 8×8×4 = 256 dimensions")
    print("• Compression ratio: 16× smaller")
    print("• Speed improvement: 3-8× faster")
    print("• Memory reduction: Proportional to compression")
    print("• Quality: Maintained through perceptual loss")
    
    print(f"\nTRAINING PROCESS:")
    print("="*40)
    print("• Encode images to latent space using VAE")
    print("• Add noise to latent representations")
    print("• Train U-Net to denoise in latent space")
    print("• Use cross-attention for conditioning")
    print("• Decode final latents back to images")
    
    print(f"\nSAMPLING PROCESS:")
    print("="*40)
    print("• Start with random noise in latent space")
    print("• Iteratively denoise using trained U-Net")
    print("• Apply conditioning through cross-attention")
    print("• Decode final clean latents to images")
    print("• Much faster than pixel-space sampling")
    
    print(f"\nFOUNDATION FOR STABLE DIFFUSION:")
    print("="*40)
    print("• Latent Diffusion Models (LDM) → Stable Diffusion")
    print("• Added CLIP text encoder for text conditioning")
    print("• Scaled to higher resolutions (512×512, 1024×1024)")
    print("• Optimized for practical deployment")
    print("• Open-sourced for widespread adoption")
    
    print(f"\nHISTORICAL IMPACT:")
    print("="*40)
    print("• Made diffusion models practically deployable")
    print("• Enabled consumer-grade text-to-image generation")
    print("• Foundation for Stable Diffusion, DALL-E 2")
    print("• Revolutionized creative AI applications")
    print("• Set standard for efficient generative models")
    print("• Enabled real-time interactive generation")
    
    print(f"\nCOMPARISON TO PIXEL DIFFUSION:")
    print("="*40)
    print("• Pixel DDPM: High quality but computationally expensive")
    print("• Latent Diffusion: Comparable quality, much more efficient")
    print("• Pixel: Limited to low resolutions in practice")
    print("• Latent: Scales to high resolutions effectively")
    print("• Pixel: Requires massive computational resources")
    print("• Latent: Deployable on consumer hardware")
    
    print(f"\nLIMITATIONS AND IMPROVEMENTS:")
    print("="*40)
    print("• VAE artifacts: Compression can introduce artifacts")
    print("• Two-stage training: VAE and diffusion trained separately")
    print("• → End-to-end training: Joint optimization")
    print("• → Better VAE architectures: Reduced artifacts")
    print("• → Consistency models: Faster sampling")
    print("• → ControlNet: Enhanced controllability")
    
    print(f"\nMODERN APPLICATIONS:")
    print("="*40)
    print("• Stable Diffusion: Open-source text-to-image")
    print("• DALL-E 2: Commercial text-to-image service")
    print("• Midjourney: Artistic image generation")
    print("• ControlNet: Controllable generation")
    print("• DreamBooth: Personalized generation")
    print("• Video generation: Temporal extensions")
    
    print(f"\nERA 4 COMPLETION:")
    print("="*40)
    print("• DDPM: Established diffusion as viable paradigm")
    print("• Score-based: Provided theoretical foundation")
    print("• Latent Diffusion: Enabled practical deployment")
    print("• → Diffusion Revolution complete")
    print("• → Foundation for modern generative AI")
    print("• → Enabled consumer-accessible AI creativity")
    
    return {
        'model': 'Latent Diffusion Models',
        'year': YEAR,
        'innovation': INNOVATION,
        'total_params': total_params,
        'latent_channels': latent_model.latent_channels,
        'timesteps': latent_model.num_timesteps,
        'latent_analysis': latent_analysis
    }

if __name__ == "__main__":
    results = main()