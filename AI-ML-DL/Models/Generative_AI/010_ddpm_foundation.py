"""
ERA 4: DIFFUSION REVOLUTION - DDPM Foundation
=============================================

Year: 2020
Paper: "Denoising Diffusion Probabilistic Models" (Ho, Jain & Abbeel)
Innovation: Denoising diffusion process for high-quality generation via gradual noise removal
Previous Limitation: Mode collapse, training instability, and limited sample diversity in GANs
Performance Gain: Stable training, high sample quality, and excellent mode coverage
Impact: Established diffusion as the dominant generative modeling paradigm

This file implements DDPM that revolutionized generative modeling through the diffusion process,
establishing the foundation for modern text-to-image models like Stable Diffusion and DALL-E.
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
INNOVATION = "Denoising diffusion process for high-quality generation via gradual noise removal"
PREVIOUS_LIMITATION = "Mode collapse, training instability, and limited sample diversity in GANs"
IMPACT = "Established diffusion as the dominant generative modeling paradigm"

print(f"=== DDPM Foundation ({YEAR}) ===")
print(f"Innovation: {INNOVATION}")
print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
print(f"Impact: {IMPACT}")
print("="*60)

# ============================================================================
# DDPM PRINCIPLES
# ============================================================================

DDPM_PRINCIPLES = {
    "forward_process": "Gradually add Gaussian noise: q(x_t|x_{t-1}) = N(√(1-β_t)x_{t-1}, β_t I)",
    "reverse_process": "Learn to denoise: p_θ(x_{t-1}|x_t) = N(μ_θ(x_t,t), Σ_θ(x_t,t))",
    "variational_bound": "Optimize ELBO to train the denoising network",
    "noise_schedule": "Predetermined β_t schedule controls noise addition rate",
    "reparameterization": "x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε, where ε ~ N(0,I)",
    "loss_simplification": "L_simple = E[||ε - ε_θ(x_t,t)||²] for noise prediction",
    "sampling_process": "Reverse diffusion from pure noise to clean image",
    "stable_training": "No adversarial training, stable gradient flow"
}

print("DDPM Principles:")
for key, principle in DDPM_PRINCIPLES.items():
    print(f"  • {principle}")
print()

# ============================================================================
# STANDARDIZED DATASET LOADING (CIFAR-10)
# ============================================================================

def load_cifar10_ddpm():
    """Load CIFAR-10 dataset for DDPM foundation study"""
    print("Loading CIFAR-10 dataset for DDPM foundation study...")
    
    # DDPM preprocessing - normalize to [-1, 1]
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
    
    # DDPM typically uses larger batch sizes
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # CIFAR-10 class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print(f"Classes: {len(classes)}")
    print(f"Image size: 32x32 RGB")
    print(f"Focus: Denoising diffusion probabilistic models")
    
    return train_loader, test_loader, classes

# ============================================================================
# NOISE SCHEDULE
# ============================================================================

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    """
    Linear noise schedule for DDPM
    
    Args:
        timesteps: Number of diffusion steps
        start: Starting β value
        end: Ending β value
    
    Returns:
        β_t schedule for noise addition
    """
    return torch.linspace(start, end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine noise schedule (improved schedule from later work)
    
    Args:
        timesteps: Number of diffusion steps
        s: Small offset to prevent β_t = 0
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class NoiseScheduler:
    """
    DDPM Noise Scheduler
    
    Manages the noise schedule and provides utilities for
    forward and reverse diffusion processes
    """
    
    def __init__(self, num_timesteps=1000, schedule='linear'):
        self.num_timesteps = num_timesteps
        
        # Choose noise schedule
        if schedule == 'linear':
            self.betas = linear_beta_schedule(num_timesteps)
        elif schedule == 'cosine':
            self.betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        # Compute derived quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_0) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        print(f"  DDPM Noise Scheduler: {num_timesteps} steps, {schedule} schedule")
        print(f"    β range: [{self.betas.min():.6f}, {self.betas.max():.6f}]")
    
    def add_noise(self, x_start, noise, timesteps):
        """
        Add noise to clean images according to noise schedule
        
        q(x_t | x_0) = N(√ᾱ_t x_0, (1-ᾱ_t)I)
        x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
        """
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Broadcast to match batch dimensions
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None, None]
        
        return (
            sqrt_alphas_cumprod_t * x_start
            + sqrt_one_minus_alphas_cumprod_t * noise
        )
    
    def sample_timesteps(self, batch_size, device):
        """Sample random timesteps for training"""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()

# ============================================================================
# ATTENTION MECHANISM
# ============================================================================

class SelfAttention(nn.Module):
    """
    Self-Attention layer for DDPM U-Net
    
    Helps the model attend to different spatial locations
    for better global consistency
    """
    
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
        print(f"    SelfAttention: {channels} channels")
    
    def forward(self, x):
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        
        # Compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)  # (b, hw, c)
        k = k.reshape(b, c, h * w)  # (b, c, hw)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)  # (b, hw, c)
        
        # Attention weights
        attn = torch.bmm(q, k) * (c ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.bmm(attn, v)
        out = out.permute(0, 2, 1).reshape(b, c, h, w)
        
        return x + self.proj_out(out)

# ============================================================================
# TIME EMBEDDING
# ============================================================================

class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for diffusion timesteps
    
    Encodes the current timestep information for the denoising network
    """
    
    def __init__(self, dim):
        super(TimeEmbedding, self).__init__()
        self.dim = dim
        
        print(f"    TimeEmbedding: {dim} dimensions")
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# ============================================================================
# RESIDUAL BLOCK
# ============================================================================

class ResidualBlock(nn.Module):
    """
    Residual block with time embedding for DDPM U-Net
    
    Incorporates time information into the denoising process
    """
    
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
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
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
        
        print(f"    ResidualBlock: {in_channels} -> {out_channels}")
    
    def forward(self, x, time_emb):
        h = self.block1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.block2(h)
        
        return h + self.residual_conv(x)

# ============================================================================
# DDPM U-NET
# ============================================================================

class DDPM_UNet(nn.Module):
    """
    U-Net architecture for DDPM denoising
    
    Takes noisy image x_t and timestep t, predicts noise ε
    """
    
    def __init__(self, in_channels=3, model_channels=128, out_channels=3, 
                 num_res_blocks=2, attention_resolutions=[16], dropout=0.1):
        super(DDPM_UNet, self).__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        
        print(f"Building DDPM U-Net...")
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            TimeEmbedding(model_channels),
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
        ds = 1
        
        for level, mult in enumerate([1, 2, 4, 8]):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)
                ]
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    layers.append(SelfAttention(ch))
                
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_chans.append(ch)
            
            if level != 3:  # No downsampling at the last level
                self.input_blocks.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                input_block_chans.append(ch)
                ds *= 2
        
        # Middle block
        self.middle_block = nn.Sequential(
            ResidualBlock(ch, ch, time_embed_dim, dropout),
            SelfAttention(ch),
            ResidualBlock(ch, ch, time_embed_dim, dropout),
        )
        
        # Upsampling path
        self.output_blocks = nn.ModuleList([])
        for level, mult in reversed(list(enumerate([1, 2, 4, 8]))):
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResidualBlock(ch + ich, mult * model_channels, time_embed_dim, dropout)
                ]
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    layers.append(SelfAttention(ch))
                
                if level != 0 and i == num_res_blocks:
                    layers.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
                    ds //= 2
                
                self.output_blocks.append(nn.Sequential(*layers))
        
        # Output projection
        self.out = nn.Sequential(
            nn.GroupNorm(32, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )
        
        # Calculate statistics
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"DDPM U-Net Summary:")
        print(f"  Input channels: {in_channels}")
        print(f"  Model channels: {model_channels}")
        print(f"  Output channels: {out_channels}")
        print(f"  Res blocks per level: {num_res_blocks}")
        print(f"  Attention at resolutions: {attention_resolutions}")
        print(f"  Total parameters: {total_params:,}")
    
    def forward(self, x, timesteps):
        """
        Forward pass: predict noise given noisy image and timestep
        
        Args:
            x: Noisy image at timestep t
            timesteps: Current timestep
        
        Returns:
            Predicted noise ε_θ(x_t, t)
        """
        # Time embedding
        t_emb = self.time_embed(timesteps)
        
        # Downsampling path
        hs = []
        h = x
        for module in self.input_blocks:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, ResidualBlock):
                        h = layer(h, t_emb)
                    else:
                        h = layer(h)
            else:
                h = module(h)
            hs.append(h)
        
        # Middle block
        for layer in self.middle_block:
            if isinstance(layer, ResidualBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)
        
        # Upsampling path
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, t_emb)
                else:
                    h = layer(h)
        
        # Output
        return self.out(h)

# ============================================================================
# DDPM MODEL
# ============================================================================

class DDPM_Foundation(nn.Module):
    """
    Denoising Diffusion Probabilistic Models
    
    Revolutionary Innovations:
    - Diffusion process for stable training without adversarial dynamics
    - Gradual denoising from pure noise to clean images
    - Variational lower bound optimization
    - Excellent mode coverage and sample diversity
    - Foundation for modern text-to-image models
    """
    
    def __init__(self, num_timesteps=1000, schedule='linear', model_channels=128):
        super(DDPM_Foundation, self).__init__()
        
        self.num_timesteps = num_timesteps
        
        print(f"Building DDPM Foundation...")
        
        # Noise scheduler
        self.noise_scheduler = NoiseScheduler(num_timesteps, schedule)
        
        # Denoising U-Net
        self.unet = DDPM_UNet(
            in_channels=3,
            model_channels=model_channels,
            out_channels=3,
            num_res_blocks=2,
            attention_resolutions=[16],
            dropout=0.1
        )
        
        # Move noise scheduler tensors to correct device when model is moved
        self.register_buffer('_dummy', torch.tensor(0.0))
        
        # Calculate statistics
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"DDPM Foundation Summary:")
        print(f"  Timesteps: {num_timesteps}")
        print(f"  Schedule: {schedule}")
        print(f"  Model channels: {model_channels}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Key innovation: Diffusion-based stable generation")
    
    def _move_to_device(self, device):
        """Move noise scheduler tensors to device"""
        self.noise_scheduler.betas = self.noise_scheduler.betas.to(device)
        self.noise_scheduler.alphas = self.noise_scheduler.alphas.to(device)
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device)
        self.noise_scheduler.alphas_cumprod_prev = self.noise_scheduler.alphas_cumprod_prev.to(device)
        self.noise_scheduler.sqrt_alphas_cumprod = self.noise_scheduler.sqrt_alphas_cumprod.to(device)
        self.noise_scheduler.sqrt_one_minus_alphas_cumprod = self.noise_scheduler.sqrt_one_minus_alphas_cumprod.to(device)
        self.noise_scheduler.posterior_variance = self.noise_scheduler.posterior_variance.to(device)
    
    def forward(self, x_0):
        """
        Training forward pass: add noise and predict it
        
        Args:
            x_0: Clean images
        
        Returns:
            loss: Simplified DDPM loss ||ε - ε_θ(x_t, t)||²
        """
        device = x_0.device
        batch_size = x_0.shape[0]
        
        # Ensure noise scheduler is on correct device
        if self.noise_scheduler.betas.device != device:
            self._move_to_device(device)
        
        # Sample timesteps
        t = self.noise_scheduler.sample_timesteps(batch_size, device)
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Add noise to images
        x_t = self.noise_scheduler.add_noise(x_0, noise, t)
        
        # Predict noise
        predicted_noise = self.unet(x_t, t)
        
        # Simplified loss (equivalent to full ELBO)
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def sample(self, num_samples, image_size=(3, 32, 32), device='cpu'):
        """
        Sample from DDPM by reversing the diffusion process
        
        Args:
            num_samples: Number of samples to generate
            image_size: Size of images to generate
            device: Device to generate on
        
        Returns:
            Generated samples
        """
        self.eval()
        
        # Ensure noise scheduler is on correct device
        if self.noise_scheduler.betas.device != device:
            self._move_to_device(device)
        
        # Start with pure noise
        x = torch.randn(num_samples, *image_size, device=device)
        
        # Reverse diffusion process
        for t in reversed(range(self.num_timesteps)):
            # Prepare timestep
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.unet(x, t_batch)
            
            # Compute coefficients
            alpha_t = self.noise_scheduler.alphas[t]
            alpha_cumprod_t = self.noise_scheduler.alphas_cumprod[t]
            beta_t = self.noise_scheduler.betas[t]
            
            if t > 0:
                alpha_cumprod_t_prev = self.noise_scheduler.alphas_cumprod[t-1]
            else:
                alpha_cumprod_t_prev = torch.tensor(1.0, device=device)
            
            # Compute predicted x_0
            predicted_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            
            # Compute direction to x_t-1
            predicted_x0 = torch.clamp(predicted_x0, -1, 1)
            
            # Compute x_t-1
            pred_prev_mean = (
                torch.sqrt(alpha_cumprod_t_prev) * beta_t / (1 - alpha_cumprod_t) * predicted_x0
                + torch.sqrt(alpha_t) * (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * x
            )
            
            if t > 0:
                # Add noise
                posterior_variance_t = self.noise_scheduler.posterior_variance[t]
                noise = torch.randn_like(x)
                x = pred_prev_mean + torch.sqrt(posterior_variance_t) * noise
            else:
                x = pred_prev_mean
        
        return torch.clamp(x, -1, 1)
    
    def get_ddpm_analysis(self):
        """Analyze DDPM innovations"""
        return {
            'ddmp_principles': DDPM_PRINCIPLES,
            'diffusion_advantages': [
                'Stable training without adversarial dynamics',
                'Excellent mode coverage and sample diversity',
                'High-quality generation comparable to GANs',
                'Principled probabilistic framework',
                'Gradual refinement process'
            ],
            'mathematical_framework': [
                'Forward process: q(x_t|x_{t-1}) = N(√(1-β_t)x_{t-1}, β_t I)',
                'Reverse process: p_θ(x_{t-1}|x_t) = N(μ_θ(x_t,t), Σ_θ(x_t,t))',
                'Reparameterization: x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε',
                'Simplified loss: L = E[||ε - ε_θ(x_t,t)||²]'
            ],
            'architectural_components': [
                'U-Net with time embedding for denoising',
                'Self-attention for global consistency',
                'Residual blocks with time conditioning',
                'Noise scheduler for diffusion process'
            ]
        }

# ============================================================================
# DDPM TRAINING FUNCTION
# ============================================================================

def train_ddpm(model, train_loader, epochs=100, learning_rate=2e-4):
    """Train DDPM with simplified loss function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training tracking
    losses = []
    
    print(f"Training DDPM on device: {device}")
    print(f"Learning rate: {learning_rate}")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # DDPM forward pass and loss
            loss = model(images)
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.6f}')
        
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
            }, f'AI-ML-DL/Models/Generative_AI/ddpm_epoch_{epoch+1}.pth')
        
        # Early stopping for demonstration
        if avg_loss < 0.01:
            print(f"Good convergence reached at epoch {epoch+1}")
            break
    
    return losses

# ============================================================================
# VISUALIZATION AND ANALYSIS
# ============================================================================

def visualize_ddpm_process():
    """Visualize DDPM diffusion process"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Forward diffusion process
    ax = axes[0, 0]
    ax.set_title('Forward Diffusion Process', fontsize=14, fontweight='bold')
    
    # Simulate forward process
    timesteps = [0, 250, 500, 750, 1000]
    noise_levels = [0, 0.3, 0.6, 0.8, 1.0]
    
    for i, (t, noise) in enumerate(zip(timesteps, noise_levels)):
        # Create noisy image representation
        x_pos = i * 0.18 + 0.1
        
        # Draw image box
        rect = plt.Rectangle((x_pos, 0.6), 0.15, 0.3, 
                           facecolor='lightblue', 
                           alpha=1-noise, 
                           edgecolor='black')
        ax.add_patch(rect)
        
        # Add noise overlay
        if noise > 0:
            noise_rect = plt.Rectangle((x_pos, 0.6), 0.15, 0.3, 
                                     facecolor='red', 
                                     alpha=noise*0.7, 
                                     edgecolor='black')
            ax.add_patch(noise_rect)
        
        # Labels
        ax.text(x_pos + 0.075, 0.75, f't={t}', ha='center', va='center', 
               fontsize=10, fontweight='bold')
        ax.text(x_pos + 0.075, 0.5, f'Noise: {noise:.1f}', ha='center', va='center', 
               fontsize=9)
        
        # Arrow to next step
        if i < len(timesteps) - 1:
            ax.annotate('', xy=(x_pos + 0.16, 0.75), xytext=(x_pos + 0.14, 0.75),
                       arrowprops=dict(arrowstyle='->', lw=2, color='darkred'))
    
    ax.text(0.5, 0.3, 'q(x_t | x_{t-1}) = N(√(1-β_t) x_{t-1}, β_t I)', 
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    ax.text(0.5, 0.1, 'Gradually add Gaussian noise until pure noise', 
           ha='center', va='center', fontsize=11,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Reverse diffusion process
    ax = axes[0, 1]
    ax.set_title('Reverse Diffusion Process (Sampling)', fontsize=14)
    
    # Reverse process
    for i, (t, noise) in enumerate(zip(reversed(timesteps), reversed(noise_levels))):
        x_pos = i * 0.18 + 0.1
        
        # Draw image box (getting cleaner)
        rect = plt.Rectangle((x_pos, 0.6), 0.15, 0.3, 
                           facecolor='lightgreen', 
                           alpha=1-noise, 
                           edgecolor='black')
        ax.add_patch(rect)
        
        # Add remaining noise
        if noise > 0:
            noise_rect = plt.Rectangle((x_pos, 0.6), 0.15, 0.3, 
                                     facecolor='red', 
                                     alpha=noise*0.7, 
                                     edgecolor='black')
            ax.add_patch(noise_rect)
        
        # Labels
        ax.text(x_pos + 0.075, 0.75, f't={t}', ha='center', va='center', 
               fontsize=10, fontweight='bold')
        ax.text(x_pos + 0.075, 0.5, f'Clean: {1-noise:.1f}', ha='center', va='center', 
               fontsize=9)
        
        # Arrow to next step
        if i < len(timesteps) - 1:
            ax.annotate('', xy=(x_pos + 0.16, 0.75), xytext=(x_pos + 0.14, 0.75),
                       arrowprops=dict(arrowstyle='->', lw=2, color='darkgreen'))
    
    ax.text(0.5, 0.3, 'p_θ(x_{t-1} | x_t) = N(μ_θ(x_t,t), Σ_θ(x_t,t))', 
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    ax.text(0.5, 0.1, 'Learn to denoise: predict and remove noise', 
           ha='center', va='center', fontsize=11,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Noise schedule visualization
    ax = axes[1, 0]
    ax.set_title('Noise Schedules', fontsize=14)
    
    timesteps = np.arange(1000)
    
    # Linear schedule
    linear_betas = np.linspace(0.0001, 0.02, 1000)
    linear_alphas_cumprod = np.cumprod(1 - linear_betas)
    
    # Cosine schedule (approximation)
    cosine_alphas_cumprod = np.cos(((timesteps / 1000) + 0.008) / 1.008 * np.pi * 0.5) ** 2
    cosine_alphas_cumprod = cosine_alphas_cumprod / cosine_alphas_cumprod[0]
    
    ax.plot(timesteps, linear_alphas_cumprod, 'b-', label='Linear Schedule', linewidth=2)
    ax.plot(timesteps, cosine_alphas_cumprod, 'r-', label='Cosine Schedule', linewidth=2)
    
    ax.set_xlabel('Timestep t')
    ax.set_ylabel('α̃_t (Signal Retention)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Highlight key regions
    ax.axvspan(0, 200, alpha=0.2, color='green', label='Low Noise')
    ax.axvspan(800, 1000, alpha=0.2, color='red', label='High Noise')
    
    # DDPM vs GAN comparison
    ax = axes[1, 1]
    ax.set_title('DDPM vs GAN Training Stability', fontsize=14)
    
    epochs = np.arange(1, 101)
    
    # GAN training (unstable)
    gan_loss = 2.0 + 0.8 * np.sin(epochs/5) + 0.3 * np.random.randn(100)
    gan_loss = np.maximum(0.1, gan_loss)
    
    # DDPM training (stable)
    ddpm_loss = 0.5 * np.exp(-epochs/50) + 0.1 + 0.02 * np.sin(epochs/10)
    
    ax.plot(epochs, gan_loss, 'r--', label='GAN Loss (Unstable)', linewidth=2, alpha=0.8)
    ax.plot(epochs, ddpm_loss, 'b-', label='DDPM Loss (Stable)', linewidth=2)
    
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Highlight stability
    ax.text(50, max(gan_loss) * 0.8, 'DDPM: Stable\nTraining!', 
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    ax.text(50, max(gan_loss) * 0.5, 'GAN: Adversarial\nInstability', 
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/Generative_AI/010_ddpm_process.png', dpi=300, bbox_inches='tight')
    print("DDPM process visualization saved: 010_ddpm_process.png")

def visualize_ddpm_architecture():
    """Visualize DDPM U-Net architecture"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # U-Net architecture
    ax = axes[0]
    ax.set_title('DDPM U-Net Architecture with Time Embedding', fontsize=16, fontweight='bold')
    
    # Encoder path
    encoder_channels = [128, 256, 512, 512]
    encoder_resolutions = [32, 16, 8, 4]
    
    for i, (ch, res) in enumerate(zip(encoder_channels, encoder_resolutions)):
        x_pos = i * 0.15 + 0.1
        
        # Encoder block
        rect = plt.Rectangle((x_pos, 0.6), 0.12, 0.3, 
                           facecolor='lightblue', edgecolor='blue', linewidth=2)
        ax.add_patch(rect)
        
        ax.text(x_pos + 0.06, 0.75, f'{res}²\n{ch}ch', 
               ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Downsampling arrow
        if i < len(encoder_channels) - 1:
            ax.annotate('', xy=(x_pos + 0.13, 0.6), xytext=(x_pos + 0.11, 0.75),
                       arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # Middle block
    middle_x = 0.4
    rect = plt.Rectangle((middle_x, 0.4), 0.2, 0.2, 
                       facecolor='yellow', edgecolor='orange', linewidth=3)
    ax.add_patch(rect)
    ax.text(middle_x + 0.1, 0.5, 'Middle\nResNet+Attn\n512ch', 
           ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Decoder path
    decoder_channels = [512, 256, 128, 3]
    decoder_resolutions = [8, 16, 32, 32]
    
    for i, (ch, res) in enumerate(zip(decoder_channels, decoder_resolutions)):
        x_pos = 0.65 + i * 0.15
        
        # Decoder block
        rect = plt.Rectangle((x_pos, 0.6), 0.12, 0.3, 
                           facecolor='lightgreen', edgecolor='green', linewidth=2)
        ax.add_patch(rect)
        
        ax.text(x_pos + 0.06, 0.75, f'{res}²\n{ch}ch', 
               ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Upsampling arrow
        if i < len(decoder_channels) - 1:
            ax.annotate('', xy=(x_pos + 0.13, 0.75), xytext=(x_pos + 0.11, 0.6),
                       arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    
    # Skip connections
    skip_positions = [(0.16, 0.9), (0.31, 0.9), (0.46, 0.9)]
    skip_targets = [(0.89, 0.9), (0.74, 0.9), (0.59, 0.9)]
    
    for (start_x, start_y), (end_x, end_y) in zip(skip_positions, skip_targets):
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', lw=2, color='purple', 
                                 connectionstyle="arc3,rad=0.3"))
    
    ax.text(0.5, 0.95, 'Skip Connections', ha='center', va='center', 
           fontsize=12, fontweight='bold', color='purple')
    
    # Time embedding
    ax.text(0.5, 0.2, 'Time Embedding: t → Sinusoidal → MLP → Inject into ResBlocks', 
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan'))
    
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Training vs sampling process
    ax = axes[1]
    ax.set_title('DDPM Training vs Sampling Process', fontsize=16, fontweight='bold')
    
    # Training process
    ax.text(0.25, 0.9, 'Training Process', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='blue')
    
    train_steps = [
        'Clean Image x₀',
        'Sample t ~ Uniform',
        'Sample ε ~ N(0,I)',
        'Create x_t = √ᾱ_t x₀ + √(1-ᾱ_t) ε',
        'Predict ε̂ = U-Net(x_t, t)',
        'Loss = ||ε - ε̂||²'
    ]
    
    for i, step in enumerate(train_steps):
        color = 'lightblue' if i % 2 == 0 else 'lightcyan'
        ax.text(0.25, 0.75 - i*0.1, step, ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.2", facecolor=color))
    
    # Sampling process
    ax.text(0.75, 0.9, 'Sampling Process', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='green')
    
    sample_steps = [
        'Start with x_T ~ N(0,I)',
        'For t = T-1 to 0:',
        '  Predict ε̂ = U-Net(x_t, t)',
        '  Compute x₀ = (x_t - √(1-ᾱ_t)ε̂)/√ᾱ_t',
        '  Compute x_{t-1} from x_t, x₀',
        'Return x₀ (clean image)'
    ]
    
    for i, step in enumerate(sample_steps):
        color = 'lightgreen' if i % 2 == 0 else 'lightgray'
        ax.text(0.75, 0.75 - i*0.1, step, ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.2", facecolor=color))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/Generative_AI/010_ddpm_architecture.png', dpi=300, bbox_inches='tight')
    print("DDPM architecture visualization saved: 010_ddpm_architecture.png")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"=== DDPM Foundation Implementation ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load CIFAR-10 dataset
    train_loader, test_loader, classes = load_cifar10_ddpm()
    
    # Initialize DDPM model
    ddpm_model = DDPM_Foundation(num_timesteps=1000, schedule='linear', model_channels=128)
    
    # Analyze model properties
    total_params = sum(p.numel() for p in ddpm_model.parameters())
    ddpm_analysis = ddpm_model.get_ddpm_analysis()
    
    print(f"\nDDPM Analysis:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Timesteps: {ddpm_model.num_timesteps}")
    
    print(f"\nDDPM Innovations:")
    for key, value in ddpm_analysis.items():
        if isinstance(value, list):
            print(f"  {key.replace('_', ' ').title()}:")
            for item in value:
                print(f"    • {item}")
        elif isinstance(value, dict):
            print(f"  {key.replace('_', ' ').title()}:")
            for k, v in value.items():
                print(f"    • {v}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Generate visualizations
    print("\nGenerating DDPM analysis...")
    visualize_ddpm_process()
    visualize_ddpm_architecture()
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("DDPM FOUNDATION SUMMARY")
    print("="*70)
    
    print(f"\nHistorical Context ({YEAR}):")
    print(f"  Innovation: {INNOVATION}")
    print(f"  Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"  Impact: {IMPACT}")
    
    print(f"\nDDPM REVOLUTIONARY INNOVATIONS:")
    print("="*50)
    print("1. DIFFUSION PROCESS:")
    print("   • Forward: Gradually add Gaussian noise over T steps")
    print("   • Reverse: Learn to denoise step by step")
    print("   • q(x_t|x_{t-1}) = N(√(1-β_t)x_{t-1}, β_t I)")
    print("   • p_θ(x_{t-1}|x_t) = N(μ_θ(x_t,t), Σ_θ(x_t,t))")
    
    print("\n2. STABLE TRAINING:")
    print("   • No adversarial dynamics or minimax games")
    print("   • Simple MSE loss: L = E[||ε - ε_θ(x_t,t)||²]")
    print("   • Stable gradients throughout training")
    print("   • No mode collapse or training instability")
    
    print("\n3. REPARAMETERIZATION TRICK:")
    print("   • Direct sampling: x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε")
    print("   • Skip intermediate steps during training")
    print("   • Efficient computation of any timestep")
    print("   • Enables practical training on large datasets")
    
    print("\n4. VARIATIONAL FRAMEWORK:")
    print("   • Optimize evidence lower bound (ELBO)")
    print("   • Principled probabilistic formulation")
    print("   • Connection to score-based models")
    print("   • Theoretical guarantees for convergence")
    
    print(f"\nTECHNICAL ACHIEVEMENTS:")
    print("="*40)
    print("• Stable training without adversarial dynamics")
    print("• High-quality generation comparable to GANs")
    print("• Excellent mode coverage and sample diversity")
    print("• Scalable to high-resolution images")
    print("• Foundation for modern text-to-image models")
    
    print(f"\nDDPM PRINCIPLES:")
    for key, principle in DDPM_PRINCIPLES.items():
        print(f"  • {principle}")
    
    print(f"\nDIFFUSION ADVANTAGES:")
    for advantage in ddpm_analysis['diffusion_advantages']:
        print(f"  • {advantage}")
    
    print(f"\nMATHEMATICAL FRAMEWORK:")
    for framework in ddpm_analysis['mathematical_framework']:
        print(f"  • {framework}")
    
    print(f"\nARCHITECTURAL COMPONENTS:")
    for component in ddpm_analysis['architectural_components']:
        print(f"  • {component}")
    
    print(f"\nU-NET ARCHITECTURE:")
    print("="*40)
    print("• Encoder-decoder with skip connections")
    print("• Time embedding via sinusoidal encoding")
    print("• ResNet blocks with time conditioning")
    print("• Self-attention at key resolutions")
    print("• Group normalization for stability")
    
    print(f"\nNOISE SCHEDULES:")
    print("="*40)
    print("• Linear: β_t increases linearly from 0.0001 to 0.02")
    print("• Cosine: Slower noise addition, better for high-res")
    print("• Controls signal-to-noise ratio over time")
    print("• Critical for generation quality")
    
    print(f"\nTRAINING PROCESS:")
    print("="*40)
    print("• Sample clean image x_0 from dataset")
    print("• Sample timestep t ~ Uniform(0, T)")
    print("• Sample noise ε ~ N(0, I)")
    print("• Create noisy image x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε")
    print("• Predict noise ε̂ = ε_θ(x_t, t)")
    print("• Minimize L = ||ε - ε̂||²")
    
    print(f"\nSAMPLING PROCESS:")
    print("="*40)
    print("• Start with pure noise x_T ~ N(0, I)")
    print("• For t = T-1 down to 0:")
    print("  - Predict noise ε̂ = ε_θ(x_t, t)")
    print("  - Compute predicted x_0")
    print("  - Compute x_{t-1} using diffusion equations")
    print("• Return final denoised image x_0")
    
    print(f"\nDDPM vs GAN COMPARISON:")
    print("="*40)
    print("• GAN: Adversarial training, unstable, mode collapse")
    print("• DDPM: Stable training, excellent mode coverage")
    print("• GAN: Fast sampling (one step)")
    print("• DDPM: Slow sampling (T steps), higher quality")
    print("• GAN: Sharp images but training difficulties")
    print("• DDPM: High quality with stable training")
    
    print(f"\nHISTORICAL IMPACT:")
    print("="*40)
    print("• Established diffusion as dominant paradigm")
    print("• Solved major GAN training problems")
    print("• Enabled large-scale generative modeling")
    print("• Foundation for Stable Diffusion, DALL-E")
    print("• Revolutionized text-to-image generation")
    print("• Set new standards for sample quality and diversity")
    
    print(f"\nLIMITATIONS AND IMPROVEMENTS:")
    print("="*40)
    print("• Slow sampling: T steps vs GAN's 1 step")
    print("• High computational cost during inference")
    print("• → DDIM: Deterministic sampling, fewer steps")
    print("• → Latent Diffusion: Operate in latent space")
    print("• → Classifier guidance: Conditional generation")
    print("• → Score-based: Continuous-time formulation")
    
    print(f"\nMODERN RELEVANCE:")
    print("="*40)
    print("• Stable Diffusion: Latent space diffusion")
    print("• DALL-E 2: Text-to-image with diffusion")
    print("• Midjourney: Commercial diffusion applications")
    print("• Video generation: Temporal diffusion models")
    print("• 3D generation: Diffusion in 3D space")
    print("• Audio generation: Diffusion for speech/music")
    
    return {
        'model': 'DDPM Foundation',
        'year': YEAR,
        'innovation': INNOVATION,
        'total_params': total_params,
        'timesteps': ddpm_model.num_timesteps,
        'ddpm_analysis': ddpm_analysis
    }

if __name__ == "__main__":
    results = main()