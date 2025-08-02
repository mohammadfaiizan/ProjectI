"""
ERA 4: DIFFUSION REVOLUTION - Score-Based Generation
====================================================

Year: 2020-2021
Papers: "Generative Modeling by Estimating Gradients of the Data Distribution" (Song & Ermon, 2019)
        "Score-Based Generative Modeling through Stochastic Differential Equations" (Song et al., 2021)
Innovation: Score matching and stochastic differential equations for continuous-time generation
Previous Limitation: Discrete timesteps and limited theoretical understanding of diffusion
Performance Gain: Flexible sampling, better quality control, and unified theoretical framework
Impact: Provided theoretical foundation for diffusion and enabled advanced sampling techniques

This file implements Score-Based Generative Models that provide the theoretical foundation
for diffusion models through score matching and stochastic differential equations.
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

YEAR = "2020-2021"
INNOVATION = "Score matching and stochastic differential equations for continuous-time generation"
PREVIOUS_LIMITATION = "Discrete timesteps and limited theoretical understanding of diffusion"
IMPACT = "Provided theoretical foundation for diffusion and enabled advanced sampling techniques"

print(f"=== Score-Based Generation ({YEAR}) ===")
print(f"Innovation: {INNOVATION}")
print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
print(f"Impact: {IMPACT}")
print("="*60)

# ============================================================================
# SCORE-BASED PRINCIPLES
# ============================================================================

SCORE_PRINCIPLES = {
    "score_function": "∇_x log p(x) - gradient of log-density points toward high-density regions",
    "score_matching": "Train neural network to estimate score function without knowing normalizing constant",
    "denoising_score_matching": "Add noise and estimate score of perturbed distribution",
    "continuous_time": "Stochastic differential equations (SDEs) for continuous diffusion",
    "probability_flow": "Deterministic ODE counterpart to stochastic SDE",
    "flexible_sampling": "Control generation through SDE/ODE solvers",
    "variance_exploding": "VE SDE: dx = √(dβ/dt) dw with increasing noise",
    "variance_preserving": "VP SDE: dx = -½β(t)x dt + √β(t) dw with preserved variance"
}

print("Score-Based Principles:")
for key, principle in SCORE_PRINCIPLES.items():
    print(f"  • {principle}")
print()

# ============================================================================
# STANDARDIZED DATASET LOADING (CIFAR-10)
# ============================================================================

def load_cifar10_score():
    """Load CIFAR-10 dataset for score-based generation study"""
    print("Loading CIFAR-10 dataset for score-based generation study...")
    
    # Score-based preprocessing
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
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # CIFAR-10 class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print(f"Classes: {len(classes)}")
    print(f"Image size: 32x32 RGB")
    print(f"Focus: Score matching and SDE-based generation")
    
    return train_loader, test_loader, classes

# ============================================================================
# NOISE PERTURBATION
# ============================================================================

class NoiseConditioning:
    """
    Noise conditioning for score-based models
    
    Manages different noise levels and perturbation schedules
    for training score networks
    """
    
    def __init__(self, sigma_min=0.01, sigma_max=50.0, num_scales=1000):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_scales = num_scales
        
        # Geometric progression of noise scales
        self.sigmas = torch.exp(torch.linspace(
            math.log(sigma_min), math.log(sigma_max), num_scales
        ))
        
        print(f"  Noise Conditioning: {num_scales} scales")
        print(f"    σ range: [{sigma_min:.3f}, {sigma_max:.1f}]")
    
    def sample_noise_levels(self, batch_size, device):
        """Sample random noise levels for training"""
        indices = torch.randint(0, self.num_scales, (batch_size,), device=device)
        return self.sigmas[indices].to(device)
    
    def perturb_data(self, x, sigma):
        """Add Gaussian noise to data"""
        noise = torch.randn_like(x)
        sigma = sigma.view(-1, 1, 1, 1)  # Broadcast for image dimensions
        perturbed_x = x + sigma * noise
        return perturbed_x, noise
    
    def marginal_prob_std(self, t):
        """Standard deviation of perturbation kernel p_t(x|x_0)"""
        return torch.sqrt((self.sigma_max ** (2 * t) - self.sigma_min ** (2 * t)) * 
                         2 * math.log(self.sigma_max / self.sigma_min) + self.sigma_min ** (2 * t))
    
    def diffusion_coeff(self, t):
        """Diffusion coefficient for SDE"""
        return self.sigma_max ** t * math.sqrt(2 * math.log(self.sigma_max / self.sigma_min))

# ============================================================================
# TIME EMBEDDING (CONTINUOUS)
# ============================================================================

class ContinuousTimeEmbedding(nn.Module):
    """
    Continuous time embedding for score networks
    
    Handles continuous time values rather than discrete timesteps
    """
    
    def __init__(self, embed_dim):
        super(ContinuousTimeEmbedding, self).__init__()
        self.embed_dim = embed_dim
        
        print(f"    ContinuousTimeEmbedding: {embed_dim} dimensions")
    
    def forward(self, t):
        """Embed continuous time values"""
        if len(t.shape) == 0:
            t = t.unsqueeze(0)
        
        half_dim = self.embed_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        
        return embeddings

# ============================================================================
# SCORE NETWORK BLOCK
# ============================================================================

class ScoreBlock(nn.Module):
    """
    Building block for score networks
    
    Incorporates continuous time information for score estimation
    """
    
    def __init__(self, in_channels, out_channels, time_embed_dim, dropout=0.1):
        super(ScoreBlock, self).__init__()
        
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, out_channels),
            nn.SiLU()
        )
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()
        
        print(f"    ScoreBlock: {in_channels} -> {out_channels}")
    
    def forward(self, x, time_embed):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        
        # Add time conditioning
        time_out = self.time_mlp(time_embed)
        h = h + time_out[:, :, None, None]
        
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip_conv(x)

# ============================================================================
# ATTENTION FOR SCORE NETWORKS
# ============================================================================

class ScoreAttention(nn.Module):
    """
    Multi-head self-attention for score networks
    
    Enables global context awareness in score estimation
    """
    
    def __init__(self, channels, num_heads=8):
        super(ScoreAttention, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
        print(f"    ScoreAttention: {channels} channels, {num_heads} heads")
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        
        # Compute Q, K, V
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, self.head_dim, H * W).transpose(-2, -1)
        k = k.view(B, self.num_heads, self.head_dim, H * W)
        v = v.view(B, self.num_heads, self.head_dim, H * W).transpose(-2, -1)
        
        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v)
        out = out.transpose(-2, -1).contiguous().view(B, C, H, W)
        
        return x + self.proj_out(out)

# ============================================================================
# SCORE NETWORK
# ============================================================================

class ScoreNetwork(nn.Module):
    """
    Neural network for estimating score functions
    
    Estimates ∇_x log p_t(x) for noise-perturbed data
    """
    
    def __init__(self, channels=3, model_channels=128, num_blocks=4, time_embed_dim=128):
        super(ScoreNetwork, self).__init__()
        
        self.channels = channels
        self.model_channels = model_channels
        
        print(f"Building Score Network...")
        
        # Time embedding
        self.time_embed = nn.Sequential(
            ContinuousTimeEmbedding(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Input projection
        self.input_conv = nn.Conv2d(channels, model_channels, 3, padding=1)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        
        ch = model_channels
        for i in range(num_blocks):
            # Score blocks
            self.down_blocks.append(ScoreBlock(ch, ch * 2, time_embed_dim))
            ch *= 2
            
            # Attention at middle resolutions
            if i == num_blocks // 2:
                self.down_attns.append(ScoreAttention(ch))
            else:
                self.down_attns.append(nn.Identity())
            
            # Downsampling (except last block)
            if i < num_blocks - 1:
                self.down_blocks.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                self.down_attns.append(nn.Identity())
        
        # Middle block
        self.middle_block = nn.Sequential(
            ScoreBlock(ch, ch, time_embed_dim),
            ScoreAttention(ch),
            ScoreBlock(ch, ch, time_embed_dim)
        )
        
        # Upsampling path
        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        
        for i in range(num_blocks):
            # Upsampling (except first block)
            if i > 0:
                self.up_blocks.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
                self.up_attns.append(nn.Identity())
            
            # Score blocks
            self.up_blocks.append(ScoreBlock(ch + ch, ch // 2, time_embed_dim))
            ch //= 2
            
            # Attention
            if i == num_blocks // 2:
                self.up_attns.append(ScoreAttention(ch))
            else:
                self.up_attns.append(nn.Identity())
        
        # Output projection
        self.output_conv = nn.Sequential(
            nn.GroupNorm(32, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, channels, 3, padding=1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Calculate statistics
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"Score Network Summary:")
        print(f"  Input channels: {channels}")
        print(f"  Model channels: {model_channels}")
        print(f"  Number of blocks: {num_blocks}")
        print(f"  Time embed dim: {time_embed_dim}")
        print(f"  Total parameters: {total_params:,}")
    
    def _initialize_weights(self):
        """Initialize weights for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x, sigma):
        """
        Estimate score function ∇_x log p_sigma(x)
        
        Args:
            x: Perturbed data
            sigma: Noise level
        
        Returns:
            Estimated score (same shape as x)
        """
        # Convert sigma to time for embedding
        # Normalize sigma to [0, 1] range for time embedding
        sigma_normalized = torch.log(sigma) / math.log(50.0)  # Assuming max sigma = 50
        sigma_normalized = torch.clamp(sigma_normalized, 0, 1)
        
        # Time embedding
        time_embed = self.time_embed(sigma_normalized)
        
        # Input
        h = self.input_conv(x)
        
        # Downsampling with skip connections
        skip_connections = [h]
        
        for i, (block, attn) in enumerate(zip(self.down_blocks, self.down_attns)):
            if isinstance(block, ScoreBlock):
                h = block(h, time_embed)
            else:
                h = block(h)  # Downsampling conv
            
            h = attn(h)
            skip_connections.append(h)
        
        # Middle
        for layer in self.middle_block:
            if isinstance(layer, ScoreBlock):
                h = layer(h, time_embed)
            else:
                h = layer(h)
        
        # Upsampling with skip connections
        for i, (block, attn) in enumerate(zip(self.up_blocks, self.up_attns)):
            if isinstance(block, ScoreBlock):
                skip = skip_connections.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, time_embed)
            else:
                h = block(h)  # Upsampling conv
            
            h = attn(h)
        
        # Output
        score = self.output_conv(h)
        
        # Scale output by noise level (important for score matching)
        sigma_expanded = sigma.view(-1, 1, 1, 1)
        return score / sigma_expanded

# ============================================================================
# SDE DEFINITIONS
# ============================================================================

class VarianceExplodingSDE:
    """
    Variance Exploding SDE
    
    dx = √(dβ/dt) dw where β(t) = σ_min^2 * (σ_max/σ_min)^(2t) - σ_min^2
    """
    
    def __init__(self, sigma_min=0.01, sigma_max=50.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
        print(f"  VE SDE: σ_min={sigma_min}, σ_max={sigma_max}")
    
    def marginal_prob(self, x, t):
        """Marginal probability parameters"""
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std
    
    def prior_sampling(self, shape, device):
        """Sample from prior distribution"""
        return torch.randn(*shape, device=device) * self.sigma_max
    
    def discretize(self, x, t, dt):
        """Euler-Maruyama discretization"""
        sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma_t * math.sqrt(2 * math.log(self.sigma_max / self.sigma_min))
        noise = torch.randn_like(x)
        return x + drift * dt + diffusion * math.sqrt(dt) * noise

class VariancePreservingSDE:
    """
    Variance Preserving SDE
    
    dx = -½β(t)x dt + √β(t) dw where β(t) increases linearly
    """
    
    def __init__(self, beta_min=0.1, beta_max=20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
        
        print(f"  VP SDE: β_min={beta_min}, β_max={beta_max}")
    
    def marginal_prob(self, x, t):
        """Marginal probability parameters"""
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1 - torch.exp(2 * log_mean_coeff))
        return mean, std
    
    def prior_sampling(self, shape, device):
        """Sample from prior distribution"""
        return torch.randn(*shape, device=device)
    
    def discretize(self, x, t, dt):
        """Euler-Maruyama discretization"""
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)[:, None, None, None]
        noise = torch.randn_like(x)
        return x + drift * dt + diffusion * math.sqrt(dt) * noise

# ============================================================================
# SCORE-BASED MODEL
# ============================================================================

class ScoreBasedGeneration(nn.Module):
    """
    Score-Based Generative Model
    
    Revolutionary Innovations:
    - Score matching for training without normalizing constants
    - Continuous-time stochastic differential equations
    - Flexible sampling via SDE/ODE solvers
    - Theoretical foundation for diffusion models
    - Connection between discrete and continuous formulations
    """
    
    def __init__(self, sde_type='VE', model_channels=128):
        super(ScoreBasedGeneration, self).__init__()
        
        self.sde_type = sde_type
        
        print(f"Building Score-Based Generation...")
        
        # SDE choice
        if sde_type == 'VE':
            self.sde = VarianceExplodingSDE()
        elif sde_type == 'VP':
            self.sde = VariancePreservingSDE()
        else:
            raise ValueError(f"Unknown SDE type: {sde_type}")
        
        # Noise conditioning
        self.noise_conditioning = NoiseConditioning()
        
        # Score network
        self.score_network = ScoreNetwork(
            channels=3,
            model_channels=model_channels,
            num_blocks=4,
            time_embed_dim=model_channels * 2
        )
        
        # Calculate statistics
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"Score-Based Model Summary:")
        print(f"  SDE type: {sde_type}")
        print(f"  Model channels: {model_channels}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Key innovation: Score matching + continuous-time SDEs")
    
    def forward(self, x):
        """
        Training forward pass using denoising score matching
        
        Args:
            x: Clean data
        
        Returns:
            Score matching loss
        """
        device = x.device
        batch_size = x.shape[0]
        
        # Sample noise levels
        sigmas = self.noise_conditioning.sample_noise_levels(batch_size, device)
        
        # Perturb data
        perturbed_x, noise = self.noise_conditioning.perturb_data(x, sigmas)
        
        # Estimate score
        score = self.score_network(perturbed_x, sigmas)
        
        # True score is -noise / sigma^2 (gradient of log p_sigma(x))
        sigmas_expanded = sigmas.view(-1, 1, 1, 1)
        target_score = -noise / (sigmas_expanded ** 2)
        
        # Score matching loss
        loss = F.mse_loss(score, target_score)
        
        return loss
    
    @torch.no_grad()
    def sample_sde(self, num_samples, image_size=(3, 32, 32), num_steps=1000, device='cpu'):
        """
        Sample using SDE (stochastic sampling)
        
        Args:
            num_samples: Number of samples
            image_size: Image dimensions
            num_steps: Number of discretization steps
            device: Device to sample on
        
        Returns:
            Generated samples
        """
        self.eval()
        
        # Start from prior
        x = self.sde.prior_sampling((num_samples, *image_size), device)
        
        # Time discretization
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.ones(num_samples, device=device) * (1 - i * dt)
            
            # Get current noise level
            if self.sde_type == 'VE':
                sigma = self.noise_conditioning.sigma_min * (
                    self.noise_conditioning.sigma_max / self.noise_conditioning.sigma_min
                ) ** t
            else:  # VP
                _, std = self.sde.marginal_prob(x, t)
                sigma = std
            
            # Estimate score
            score = self.score_network(x, sigma)
            
            # SDE step (reverse-time)
            if self.sde_type == 'VE':
                # Reverse VE SDE
                drift = score
                diffusion = self.noise_conditioning.diffusion_coeff(t[0])
                noise = torch.randn_like(x)
                x = x + drift * dt + diffusion * math.sqrt(dt) * noise
            else:  # VP
                # Reverse VP SDE
                beta_t = self.sde.beta_min + t * (self.sde.beta_max - self.sde.beta_min)
                drift = -0.5 * beta_t[:, None, None, None] * x + beta_t[:, None, None, None] * score
                diffusion = torch.sqrt(beta_t)[:, None, None, None]
                noise = torch.randn_like(x)
                x = x + drift * dt + diffusion * math.sqrt(dt) * noise
        
        return torch.clamp(x, -1, 1)
    
    @torch.no_grad()
    def sample_ode(self, num_samples, image_size=(3, 32, 32), num_steps=1000, device='cpu'):
        """
        Sample using probability flow ODE (deterministic sampling)
        
        Args:
            num_samples: Number of samples
            image_size: Image dimensions
            num_steps: Number of integration steps
            device: Device to sample on
        
        Returns:
            Generated samples
        """
        self.eval()
        
        # Start from prior
        x = self.sde.prior_sampling((num_samples, *image_size), device)
        
        # Time discretization
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.ones(num_samples, device=device) * (1 - i * dt)
            
            # Get current noise level
            if self.sde_type == 'VE':
                sigma = self.noise_conditioning.sigma_min * (
                    self.noise_conditioning.sigma_max / self.noise_conditioning.sigma_min
                ) ** t
            else:  # VP
                _, std = self.sde.marginal_prob(x, t)
                sigma = std
            
            # Estimate score
            score = self.score_network(x, sigma)
            
            # Probability flow ODE step
            if self.sde_type == 'VE':
                # VE probability flow ODE
                drift = -0.5 * self.noise_conditioning.diffusion_coeff(t[0]) ** 2 * score
            else:  # VP
                # VP probability flow ODE
                beta_t = self.sde.beta_min + t * (self.sde.beta_max - self.sde.beta_min)
                drift = -0.5 * beta_t[:, None, None, None] * x + 0.5 * beta_t[:, None, None, None] * score
            
            x = x + drift * dt
        
        return torch.clamp(x, -1, 1)
    
    def get_score_analysis(self):
        """Analyze score-based model innovations"""
        return {
            'score_principles': SCORE_PRINCIPLES,
            'theoretical_advantages': [
                'Principled score matching without normalizing constants',
                'Continuous-time formulation via SDEs',
                'Flexible sampling through SDE/ODE solvers',
                'Connection to diffusion models',
                'Deterministic and stochastic sampling options'
            ],
            'mathematical_framework': [
                'Score function: ∇_x log p(x)',
                'Denoising score matching: E[||∇_x log p_σ(x|x_0) - s_θ(x,σ)||²]',
                'Forward SDE: dx = f(x,t)dt + g(t)dw',
                'Reverse SDE: dx = [f(x,t) - g(t)²∇_x log p_t(x)]dt + g(t)dw̃',
                'Probability flow ODE: dx = [f(x,t) - ½g(t)²∇_x log p_t(x)]dt'
            ],
            'sde_types': {
                'VE': 'Variance Exploding - noise grows without bound',
                'VP': 'Variance Preserving - maintains unit variance'
            }
        }

# ============================================================================
# SCORE-BASED TRAINING FUNCTION
# ============================================================================

def train_score_based(model, train_loader, epochs=100, learning_rate=1e-4):
    """Train score-based model with score matching"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training tracking
    losses = []
    
    print(f"Training Score-Based Model on device: {device}")
    print(f"Learning rate: {learning_rate}")
    print(f"SDE type: {model.sde_type}")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # Score matching loss
            loss = model(images)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                      f'Score Loss: {loss.item():.6f}')
        
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
            }, f'AI-ML-DL/Models/Generative_AI/score_based_epoch_{epoch+1}.pth')
        
        # Early stopping for demonstration
        if avg_loss < 0.05:
            print(f"Good convergence reached at epoch {epoch+1}")
            break
    
    return losses

# ============================================================================
# VISUALIZATION AND ANALYSIS
# ============================================================================

def visualize_score_concepts():
    """Visualize score-based model concepts"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Score function visualization
    ax = axes[0, 0]
    ax.set_title('Score Function Concept', fontsize=14, fontweight='bold')
    
    # Create 2D density visualization
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # Gaussian mixture for density
    Z1 = np.exp(-((X-1)**2 + (Y-1)**2)/0.5)
    Z2 = np.exp(-((X+1)**2 + (Y+1)**2)/0.5)
    Z = Z1 + Z2
    Z = Z / Z.max()
    
    # Plot density
    contour = ax.contour(X, Y, Z, levels=10, colors='blue', alpha=0.7)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Score function arrows (gradient of log density)
    dx = np.gradient(np.log(Z + 1e-8), axis=1)
    dy = np.gradient(np.log(Z + 1e-8), axis=0)
    
    # Subsample for clarity
    skip = 8
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
              dx[::skip, ::skip], dy[::skip, ::skip],
              color='red', alpha=0.8, scale=10)
    
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title('Score Function ∇log p(x) Points to High Density')
    ax.grid(True, alpha=0.3)
    
    # Score matching visualization
    ax = axes[0, 1]
    ax.set_title('Denoising Score Matching', fontsize=14)
    
    # Show noising process
    steps = ['Clean\nData', 'Add\nNoise', 'Perturbed\nData', 'Estimate\nScore', 'Loss\nComputation']
    y_pos = [0.8, 0.8, 0.8, 0.5, 0.2]
    x_pos = [0.1, 0.3, 0.5, 0.7, 0.9]
    colors = ['lightgreen', 'yellow', 'lightcoral', 'lightblue', 'orange']
    
    for i, (step, x, y, color) in enumerate(zip(steps, x_pos, y_pos, colors)):
        ax.text(x, y, step, ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
        
        if i < len(steps) - 1:
            ax.annotate('', xy=(x_pos[i+1]-0.05, y_pos[i+1]), xytext=(x+0.05, y),
                       arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
    
    # Mathematical formulation
    ax.text(0.5, 0.05, 'Loss = E[||∇log p_σ(x̃|x₀) - s_θ(x̃,σ)||²]', 
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # SDE vs ODE sampling
    ax = axes[1, 0]
    ax.set_title('SDE vs ODE Sampling', fontsize=14)
    
    # Simulate sampling trajectories
    time_steps = np.linspace(0, 1, 100)
    
    # SDE trajectory (stochastic)
    np.random.seed(42)
    sde_traj = np.cumsum(np.random.randn(100) * 0.1) + 2 * np.sin(2 * np.pi * time_steps)
    
    # ODE trajectory (deterministic)
    ode_traj = 2 * np.sin(2 * np.pi * time_steps) + 0.5 * np.cos(4 * np.pi * time_steps)
    
    ax.plot(time_steps, sde_traj, 'r-', linewidth=2, label='SDE (Stochastic)', alpha=0.8)
    ax.plot(time_steps, ode_traj, 'b-', linewidth=2, label='ODE (Deterministic)')
    
    ax.set_xlabel('Time t')
    ax.set_ylabel('Sample Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    ax.text(0.2, max(sde_traj) * 0.8, 'SDE:\nStochastic\nDiverse samples', 
           ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
    
    ax.text(0.8, max(ode_traj) * 0.8, 'ODE:\nDeterministic\nFast sampling', 
           ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
    
    # Continuous vs discrete time
    ax = axes[1, 1]
    ax.set_title('Continuous vs Discrete Time Formulation', fontsize=14)
    
    # Discrete timesteps (DDPM style)
    discrete_times = [0, 0.25, 0.5, 0.75, 1.0]
    discrete_noise = [0, 0.3, 0.6, 0.8, 1.0]
    
    ax.plot(discrete_times, discrete_noise, 'ro-', linewidth=3, markersize=8, 
           label='Discrete (DDPM)', alpha=0.8)
    
    # Continuous time (Score-based)
    continuous_times = np.linspace(0, 1, 100)
    continuous_noise = continuous_times ** 1.5  # Smooth curve
    
    ax.plot(continuous_times, continuous_noise, 'b-', linewidth=3, 
           label='Continuous (Score-based)')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Noise Level')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Highlight advantages
    ax.text(0.5, 0.3, 'Score-based Models:\n• Continuous time formulation\n• Flexible sampling\n• Theoretical foundation', 
           ha='center', va='center', fontsize=11,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/Generative_AI/011_score_concepts.png', dpi=300, bbox_inches='tight')
    print("Score-based concepts visualization saved: 011_score_concepts.png")

def visualize_sde_types():
    """Visualize different SDE types"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Variance Exploding SDE
    ax = axes[0, 0]
    ax.set_title('Variance Exploding (VE) SDE', fontsize=14, fontweight='bold')
    
    time = np.linspace(0, 1, 100)
    sigma_min, sigma_max = 0.01, 50.0
    
    # VE noise schedule
    ve_sigma = sigma_min * (sigma_max / sigma_min) ** time
    ve_variance = ve_sigma ** 2
    
    ax.semilogy(time, ve_sigma, 'r-', linewidth=3, label='Noise Level σ(t)')
    ax.semilogy(time, ve_variance, 'b--', linewidth=2, label='Variance σ²(t)')
    
    ax.set_xlabel('Time t')
    ax.set_ylabel('Log Scale')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add equation
    ax.text(0.5, 0.7, 'dx = √(dβ/dt) dw\nVariance explodes', 
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'),
           transform=ax.transAxes)
    
    # Variance Preserving SDE
    ax = axes[0, 1]
    ax.set_title('Variance Preserving (VP) SDE', fontsize=14, fontweight='bold')
    
    beta_min, beta_max = 0.1, 20.0
    
    # VP noise schedule
    log_mean_coeff = -0.25 * time ** 2 * (beta_max - beta_min) - 0.5 * time * beta_min
    vp_mean_coeff = np.exp(log_mean_coeff)
    vp_variance = 1 - np.exp(2 * log_mean_coeff)
    
    ax.plot(time, vp_mean_coeff, 'g-', linewidth=3, label='Mean Coefficient')
    ax.plot(time, vp_variance, 'orange', linewidth=2, label='Noise Variance')
    ax.plot(time, vp_mean_coeff ** 2 + vp_variance, 'purple', linewidth=2, label='Total Variance')
    
    ax.set_xlabel('Time t')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add equation
    ax.text(0.5, 0.7, 'dx = -½β(t)x dt + √β(t) dw\nVariance preserved at 1', 
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'),
           transform=ax.transAxes)
    
    # Sampling trajectories comparison
    ax = axes[1, 0]
    ax.set_title('Reverse SDE Trajectories', fontsize=14)
    
    # Simulate reverse sampling
    np.random.seed(42)
    num_steps = 50
    
    # VE reverse trajectory
    ve_x = [10.0]  # Start from high noise
    for i in range(num_steps):
        t = 1 - i / num_steps
        sigma_t = sigma_min * (sigma_max / sigma_min) ** t
        # Simplified reverse step
        score_estimate = -ve_x[-1] / (sigma_t ** 2) * 0.1  # Mock score
        drift = score_estimate
        diffusion = sigma_t * math.sqrt(2 * math.log(sigma_max / sigma_min))
        noise = np.random.randn() * 0.1
        next_x = ve_x[-1] + drift * (1/num_steps) + diffusion * math.sqrt(1/num_steps) * noise
        ve_x.append(next_x)
    
    # VP reverse trajectory
    vp_x = [2.0]  # Start from normal noise
    for i in range(num_steps):
        t = 1 - i / num_steps
        beta_t = beta_min + t * (beta_max - beta_min)
        # Simplified reverse step
        score_estimate = -vp_x[-1] * 0.1  # Mock score
        drift = -0.5 * beta_t * vp_x[-1] + beta_t * score_estimate
        diffusion = math.sqrt(beta_t)
        noise = np.random.randn() * 0.1
        next_x = vp_x[-1] + drift * (1/num_steps) + diffusion * math.sqrt(1/num_steps) * noise
        vp_x.append(next_x)
    
    reverse_time = np.linspace(1, 0, num_steps + 1)
    ax.plot(reverse_time, ve_x, 'r-', linewidth=2, label='VE SDE', marker='o', markersize=3)
    ax.plot(reverse_time, vp_x, 'b-', linewidth=2, label='VP SDE', marker='s', markersize=3)
    
    ax.set_xlabel('Reverse Time (1 → 0)')
    ax.set_ylabel('Sample Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Probability flow ODE
    ax = axes[1, 1]
    ax.set_title('Probability Flow ODE vs SDE', fontsize=14)
    
    methods = ['SDE\n(Stochastic)', 'ODE\n(Deterministic)']
    properties = ['Speed', 'Diversity', 'Consistency', 'Memory']
    
    # Scores for each method
    sde_scores = [6, 9, 7, 8]
    ode_scores = [9, 6, 9, 9]
    
    x = np.arange(len(properties))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, sde_scores, width, label='SDE', color='lightcoral', alpha=0.8)
    bars2 = ax.bar(x + width/2, ode_scores, width, label='ODE', color='lightblue', alpha=0.8)
    
    ax.set_ylabel('Score (1-10)')
    ax.set_xticks(x)
    ax.set_xticklabels(properties)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                   f'{height}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/Generative_AI/011_sde_types.png', dpi=300, bbox_inches='tight')
    print("SDE types visualization saved: 011_sde_types.png")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"=== Score-Based Generation Implementation ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load CIFAR-10 dataset
    train_loader, test_loader, classes = load_cifar10_score()
    
    # Initialize Score-Based model
    score_model = ScoreBasedGeneration(sde_type='VE', model_channels=128)
    
    # Analyze model properties
    total_params = sum(p.numel() for p in score_model.parameters())
    score_analysis = score_model.get_score_analysis()
    
    print(f"\nScore-Based Model Analysis:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  SDE type: {score_model.sde_type}")
    
    print(f"\nScore-Based Innovations:")
    for key, value in score_analysis.items():
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
    print("\nGenerating Score-Based analysis...")
    visualize_score_concepts()
    visualize_sde_types()
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("SCORE-BASED GENERATION SUMMARY")
    print("="*70)
    
    print(f"\nHistorical Context ({YEAR}):")
    print(f"  Innovation: {INNOVATION}")
    print(f"  Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"  Impact: {IMPACT}")
    
    print(f"\nSCORE-BASED REVOLUTIONARY INNOVATIONS:")
    print("="*50)
    print("1. SCORE FUNCTION ESTIMATION:")
    print("   • Score function: ∇_x log p(x) points toward high-density regions")
    print("   • No need for normalizing constants in training")
    print("   • Denoising score matching for tractable training")
    print("   • Connection to energy-based models")
    
    print("\n2. CONTINUOUS-TIME FORMULATION:")
    print("   • Stochastic Differential Equations (SDEs) for diffusion")
    print("   • Forward SDE: dx = f(x,t)dt + g(t)dw")
    print("   • Reverse SDE: dx = [f(x,t) - g(t)²∇log p_t(x)]dt + g(t)dw̃")
    print("   • Unified framework for discrete and continuous models")
    
    print("\n3. FLEXIBLE SAMPLING:")
    print("   • SDE sampling: Stochastic, diverse samples")
    print("   • ODE sampling: Deterministic, fast sampling")
    print("   • Probability flow ODE: dx = [f(x,t) - ½g(t)²∇log p_t(x)]dt")
    print("   • Controllable quality vs speed trade-off")
    
    print("\n4. THEORETICAL FOUNDATION:")
    print("   • Rigorous mathematical framework")
    print("   • Connection between diffusion and score matching")
    print("   • Proof of equivalence to DDPM")
    print("   • Principled sampling procedures")
    
    print(f"\nTECHNICAL ACHIEVEMENTS:")
    print("="*40)
    print("• Unified theory for diffusion models")
    print("• Flexible continuous-time formulation")
    print("• Multiple sampling strategies (SDE/ODE)")
    print("• Strong theoretical guarantees")
    print("• Foundation for advanced diffusion techniques")
    
    print(f"\nSCORE PRINCIPLES:")
    for key, principle in SCORE_PRINCIPLES.items():
        print(f"  • {principle}")
    
    print(f"\nTHEORETICAL ADVANTAGES:")
    for advantage in score_analysis['theoretical_advantages']:
        print(f"  • {advantage}")
    
    print(f"\nMATHEMATICAL FRAMEWORK:")
    for framework in score_analysis['mathematical_framework']:
        print(f"  • {framework}")
    
    print(f"\nSDE TYPES:")
    for sde_type, description in score_analysis['sde_types'].items():
        print(f"  • {sde_type}: {description}")
    
    print(f"\nVARIANCE EXPLODING (VE) SDE:")
    print("="*40)
    print("• Forward: dx = √(dβ/dt) dw")
    print("• Variance grows without bound")
    print("• σ(t) = σ_min * (σ_max/σ_min)^t")
    print("• Good for high-resolution generation")
    print("• Used in original score-based models")
    
    print(f"\nVARIANCE PRESERVING (VP) SDE:")
    print("="*40)
    print("• Forward: dx = -½β(t)x dt + √β(t) dw")
    print("• Maintains unit variance throughout")
    print("• Equivalent to DDPM formulation")
    print("• Better numerical stability")
    print("• Standard in modern implementations")
    
    print(f"\nSAMPLING STRATEGIES:")
    print("="*40)
    print("• SDE Sampling:")
    print("  - Stochastic sampling with noise")
    print("  - Higher diversity, slower generation")
    print("  - Multiple independent samples")
    print("• ODE Sampling:")
    print("  - Deterministic probability flow")
    print("  - Faster generation, lower diversity")
    print("  - Exact likelihood computation")
    print("  - Invertible generation process")
    
    print(f"\nTRAINING PROCESS:")
    print("="*40)
    print("• Sample clean data x₀")
    print("• Sample noise level σ")
    print("• Perturb data: x̃ = x₀ + σε")
    print("• Estimate score: s_θ(x̃, σ)")
    print("• True score: ∇log p_σ(x̃|x₀) = -ε/σ²")
    print("• Loss: E[||s_θ(x̃,σ) - (-ε/σ²)||²]")
    
    print(f"\nCONNECTION TO DDPM:")
    print("="*40)
    print("• DDPM is special case of score-based models")
    print("• Discrete-time VP SDE corresponds to DDPM")
    print("• Score network ≡ Noise prediction network")
    print("• ε_θ(x_t,t) = -σ_t * s_θ(x_t,t)")
    print("• Unified theoretical understanding")
    
    print(f"\nHISTORICAL IMPACT:")
    print("="*40)
    print("• Provided theoretical foundation for diffusion")
    print("• Unified discrete and continuous formulations")
    print("• Enabled advanced sampling techniques")
    print("• Inspired numerous theoretical advances")
    print("• Set stage for controllable generation")
    print("• Foundation for classifier-free guidance")
    
    print(f"\nADVANTAGES OVER DDPM:")
    print("="*40)
    print("• Continuous-time formulation")
    print("• Flexible sampling strategies")
    print("• Better theoretical understanding")
    print("• Exact likelihood computation (ODE)")
    print("• Principled interpolation")
    print("• Advanced solver integration")
    
    print(f"\nLIMITATIONS AND IMPROVEMENTS:")
    print("="*40)
    print("• Computational cost of SDE/ODE solvers")
    print("• Still slower than single-step generators")
    print("• → Classifier guidance: Conditional generation")
    print("• → DDIM: Fast deterministic sampling")
    print("• → Progressive distillation: Fewer steps")
    print("• → Latent diffusion: Efficient high-res generation")
    
    print(f"\nMODERN RELEVANCE:")
    print("="*40)
    print("• Theoretical backbone of diffusion models")
    print("• Foundation for Stable Diffusion")
    print("• Enables advanced sampling techniques")
    print("• Basis for controllable generation")
    print("• Used in audio and video generation")
    print("• Inspires new generative model designs")
    
    return {
        'model': 'Score-Based Generation',
        'year': YEAR,
        'innovation': INNOVATION,
        'total_params': total_params,
        'sde_type': score_model.sde_type,
        'score_analysis': score_analysis
    }

if __name__ == "__main__":
    results = main()