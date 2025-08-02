"""
ERA 3: ADVANCED GANS & ARCHITECTURAL INNOVATIONS - StyleGAN Control
==================================================================

Year: 2018
Paper: "A Style-Based Generator Architecture for Generative Adversarial Networks" (Karras et al.)
Innovation: Style-based generation with disentangled control over image synthesis
Previous Limitation: Lack of fine-grained control over generated image attributes
Performance Gain: Unprecedented control over generation with disentangled latent space
Impact: Revolutionized controllable generation and established style-based paradigm

This file implements StyleGAN that transformed generative modeling through style-based
synthesis, enabling unprecedented control over generated content.
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

YEAR = "2018"
INNOVATION = "Style-based generation with disentangled control over image synthesis"
PREVIOUS_LIMITATION = "Lack of fine-grained control over generated image attributes"
IMPACT = "Revolutionized controllable generation and established style-based paradigm"

print(f"=== StyleGAN Control ({YEAR}) ===")
print(f"Innovation: {INNOVATION}")
print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
print(f"Impact: {IMPACT}")
print("="*60)

# ============================================================================
# STYLEGAN PRINCIPLES
# ============================================================================

STYLEGAN_PRINCIPLES = {
    "style_based_synthesis": "Generate images by controlling style at different resolutions",
    "mapping_network": "Transform input latent z to intermediate latent w for better disentanglement",
    "adaptive_instance_norm": "Apply styles to features via Adaptive Instance Normalization (AdaIN)",
    "stochastic_variation": "Add noise for fine-grained stochastic details",
    "progressive_generation": "Build upon Progressive GAN architecture",
    "style_mixing": "Combine styles from different latent codes for controlled generation",
    "truncation_trick": "Control generation quality vs diversity trade-off",
    "perceptual_path_length": "Measure and optimize for smooth latent space interpolation"
}

print("StyleGAN Principles:")
for key, principle in STYLEGAN_PRINCIPLES.items():
    print(f"  • {principle}")
print()

# ============================================================================
# STANDARDIZED DATASET LOADING (CIFAR-10)
# ============================================================================

def load_cifar10_stylegan():
    """Load CIFAR-10 dataset for StyleGAN control study"""
    print("Loading CIFAR-10 dataset for StyleGAN control study...")
    print("Note: StyleGAN typically uses CelebA-HQ or FFHQ for face generation")
    
    # StyleGAN preprocessing
    transform_train = transforms.Compose([
        transforms.Resize(64),  # StyleGAN works with power-of-2 resolutions
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(64),
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
    
    # StyleGAN typically uses larger batch sizes
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2, drop_last=True)
    
    # CIFAR-10 class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print(f"Classes: {len(classes)}")
    print(f"Resolution: 64x64 (StyleGAN scales to 1024x1024)")
    print(f"Focus: Style-based controllable generation")
    
    return train_loader, test_loader, classes

# ============================================================================
# MAPPING NETWORK
# ============================================================================

class MappingNetwork(nn.Module):
    """
    StyleGAN Mapping Network
    
    Innovation: Transform input latent z to intermediate latent w
    for better disentanglement and control
    
    Key benefits:
    - Reduces correlation between features
    - Enables better disentanglement
    - Improves interpolation quality
    """
    
    def __init__(self, latent_dim=512, hidden_dim=512, num_layers=8):
        super(MappingNetwork, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Build fully connected mapping network
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(latent_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        
        self.mapping = nn.Sequential(*layers)
        
        # Initialize with careful scaling for stable training
        self._initialize_weights()
        
        print(f"  Mapping Network: {latent_dim}D -> {hidden_dim}D, {num_layers} layers")
        print(f"    Purpose: z -> w transformation for disentanglement")
    
    def _initialize_weights(self):
        """Initialize weights for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(module.bias)
    
    def forward(self, z):
        """Transform input latent z to intermediate latent w"""
        # Normalize input latent code
        z_normalized = F.normalize(z, dim=1)
        
        # Map to intermediate latent space
        w = self.mapping(z_normalized)
        
        return w

# ============================================================================
# ADAPTIVE INSTANCE NORMALIZATION (ADAIN)
# ============================================================================

class AdaptiveInstanceNorm(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN)
    
    Innovation: Apply style information to feature maps
    by modulating the statistics (mean and std) of features
    
    AdaIN(x, y) = σ(y) * normalize(x) + μ(y)
    where normalize(x) = (x - μ(x)) / σ(x)
    """
    
    def __init__(self, feature_channels, style_dim):
        super(AdaptiveInstanceNorm, self).__init__()
        
        self.feature_channels = feature_channels
        self.style_dim = style_dim
        
        # Style transformation layers
        self.style_scale = nn.Linear(style_dim, feature_channels)
        self.style_bias = nn.Linear(style_dim, feature_channels)
        
        # Initialize style modulation
        nn.init.ones_(self.style_scale.weight)
        nn.init.zeros_(self.style_scale.bias)
        nn.init.zeros_(self.style_bias.weight)
        nn.init.zeros_(self.style_bias.bias)
        
        print(f"    AdaIN: {feature_channels} features modulated by {style_dim}D style")
    
    def forward(self, features, style):
        """Apply style to features via adaptive instance normalization"""
        batch_size, channels, height, width = features.shape
        
        # Compute feature statistics
        features_mean = features.view(batch_size, channels, -1).mean(dim=2, keepdim=True)
        features_std = features.view(batch_size, channels, -1).std(dim=2, keepdim=True)
        
        # Normalize features
        features_normalized = (features.view(batch_size, channels, -1) - features_mean) / (features_std + 1e-8)
        
        # Get style modulation parameters
        style_scale = self.style_scale(style).unsqueeze(2)  # (B, C, 1)
        style_bias = self.style_bias(style).unsqueeze(2)    # (B, C, 1)
        
        # Apply style modulation
        styled_features = style_scale * features_normalized + style_bias
        
        # Reshape back to spatial dimensions
        styled_features = styled_features.view(batch_size, channels, height, width)
        
        return styled_features

# ============================================================================
# STYLE BLOCK
# ============================================================================

class StyleBlock(nn.Module):
    """
    StyleGAN Style Block
    
    Combines convolution, style modulation (AdaIN), and noise injection
    for controllable feature generation at each resolution
    """
    
    def __init__(self, in_channels, out_channels, style_dim, kernel_size=3, upsample=False):
        super(StyleBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = upsample
        
        # Upsampling layer
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Convolution layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        # Style modulation (AdaIN)
        self.adain1 = AdaptiveInstanceNorm(out_channels, style_dim)
        self.adain2 = AdaptiveInstanceNorm(out_channels, style_dim)
        
        # Noise injection parameters
        self.noise_scale1 = nn.Parameter(torch.zeros(1))
        self.noise_scale2 = nn.Parameter(torch.zeros(1))
        
        print(f"    StyleBlock: {in_channels}->{out_channels}, upsample={upsample}")
    
    def add_noise(self, features, noise_scale):
        """Add stochastic variation via noise injection"""
        batch_size, channels, height, width = features.shape
        
        # Generate noise
        noise = torch.randn(batch_size, 1, height, width, device=features.device)
        
        # Scale and add noise
        return features + noise_scale * noise
    
    def forward(self, x, style1, style2):
        """Forward pass with style modulation and noise injection"""
        # Optional upsampling
        if self.upsample:
            x = self.upsample_layer(x)
        
        # First convolution + style + noise
        x = self.conv1(x)
        x = self.add_noise(x, self.noise_scale1)
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.adain1(x, style1)
        
        # Second convolution + style + noise
        x = self.conv2(x)
        x = self.add_noise(x, self.noise_scale2)
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.adain2(x, style2)
        
        return x

# ============================================================================
# STYLEGAN GENERATOR
# ============================================================================

class StyleGANGenerator(nn.Module):
    """
    StyleGAN Generator
    
    Revolutionary architecture combining:
    - Mapping network for disentanglement
    - Style-based synthesis via AdaIN
    - Progressive generation structure
    - Stochastic variation via noise injection
    """
    
    def __init__(self, latent_dim=512, style_dim=512, max_resolution=64, base_channels=512):
        super(StyleGANGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.max_resolution = max_resolution
        self.base_channels = base_channels
        
        print(f"Building StyleGAN Generator...")
        
        # Mapping network: z -> w
        self.mapping_network = MappingNetwork(latent_dim, style_dim)
        
        # Constant input (learned)
        self.constant_input = nn.Parameter(torch.randn(1, base_channels, 4, 4))
        
        # Style blocks for progressive generation
        self.style_blocks = nn.ModuleList()
        self.to_rgb_layers = nn.ModuleList()
        
        # Calculate number of resolutions: 4x4 -> max_resolution
        num_resolutions = int(math.log2(max_resolution // 4)) + 1
        
        current_channels = base_channels
        for i in range(num_resolutions):
            # Determine output channels (halve each resolution)
            if i == 0:
                out_channels = current_channels
            else:
                out_channels = max(current_channels // 2, 32)
            
            # Create style block
            style_block = StyleBlock(
                current_channels, out_channels, style_dim, 
                upsample=(i > 0)
            )
            self.style_blocks.append(style_block)
            
            # To RGB layer for each resolution
            to_rgb = nn.Conv2d(out_channels, 3, 1)
            self.to_rgb_layers.append(to_rgb)
            
            current_channels = out_channels
        
        # Initialize weights
        self._initialize_weights()
        
        # Calculate statistics
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"StyleGAN Generator Summary:")
        print(f"  Latent dimension: {latent_dim}")
        print(f"  Style dimension: {style_dim}")
        print(f"  Max resolution: {max_resolution}x{max_resolution}")
        print(f"  Number of resolutions: {num_resolutions}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Key innovation: Style-based controllable synthesis")
    
    def _initialize_weights(self):
        """Initialize weights for stable StyleGAN training"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(module.bias)
    
    def get_style_codes(self, latent_codes, num_layers):
        """
        Get style codes for each layer
        
        Supports style mixing by using different latent codes
        for different layers
        """
        if isinstance(latent_codes, torch.Tensor):
            # Single latent code - replicate for all layers
            styles = [self.mapping_network(latent_codes) for _ in range(num_layers)]
        else:
            # Multiple latent codes for style mixing
            styles = []
            for i in range(num_layers):
                # Use different codes for different layers
                code_idx = min(i // 2, len(latent_codes) - 1)
                style = self.mapping_network(latent_codes[code_idx])
                styles.append(style)
        
        return styles
    
    def forward(self, latent_codes, truncation_psi=1.0, truncation_cutoff=None):
        """
        StyleGAN forward pass with controllable generation
        
        Args:
            latent_codes: Latent code(s) for generation
            truncation_psi: Truncation factor for quality/diversity trade-off
            truncation_cutoff: Layer cutoff for truncation
        """
        if isinstance(latent_codes, list):
            batch_size = latent_codes[0].shape[0]
        else:
            batch_size = latent_codes.shape[0]
        
        # Start with constant input
        x = self.constant_input.expand(batch_size, -1, -1, -1)
        
        # Get style codes for all layers
        num_style_layers = len(self.style_blocks) * 2  # Two styles per block
        styles = self.get_style_codes(latent_codes, num_style_layers)
        
        # Apply truncation trick if specified
        if truncation_psi < 1.0:
            styles = self.apply_truncation(styles, truncation_psi, truncation_cutoff)
        
        # Progressive generation through style blocks
        for i, style_block in enumerate(self.style_blocks):
            # Get styles for this block
            style1 = styles[i * 2]
            style2 = styles[i * 2 + 1]
            
            # Apply style block
            x = style_block(x, style1, style2)
        
        # Convert to RGB
        rgb = torch.tanh(self.to_rgb_layers[-1](x))
        
        return rgb
    
    def apply_truncation(self, styles, truncation_psi, cutoff=None):
        """
        Apply truncation trick for quality/diversity trade-off
        
        Moves style codes closer to average for better quality
        """
        if cutoff is None:
            cutoff = len(styles)
        
        # Compute average style (would be computed from training data)
        avg_style = torch.zeros_like(styles[0])
        
        truncated_styles = []
        for i, style in enumerate(styles):
            if i < cutoff:
                # Apply truncation: w = w_avg + ψ(w - w_avg)
                truncated_style = avg_style + truncation_psi * (style - avg_style)
                truncated_styles.append(truncated_style)
            else:
                truncated_styles.append(style)
        
        return truncated_styles
    
    def style_mixing(self, latent1, latent2, mixing_layers=None):
        """
        Demonstrate style mixing between two latent codes
        
        Uses latent1 for some layers and latent2 for others
        """
        if mixing_layers is None:
            # Default: mix at middle layers
            mixing_layers = list(range(4, 8))
        
        # Generate styles from both latents
        styles1 = self.get_style_codes(latent1, len(self.style_blocks) * 2)
        styles2 = self.get_style_codes(latent2, len(self.style_blocks) * 2)
        
        # Mix styles
        mixed_styles = []
        for i in range(len(styles1)):
            if i in mixing_layers:
                mixed_styles.append(styles2[i])
            else:
                mixed_styles.append(styles1[i])
        
        # Generate with mixed styles
        return self.forward_with_styles(mixed_styles)
    
    def forward_with_styles(self, styles):
        """Forward pass with pre-computed styles"""
        batch_size = styles[0].shape[0]
        
        # Start with constant input
        x = self.constant_input.expand(batch_size, -1, -1, -1)
        
        # Progressive generation
        for i, style_block in enumerate(self.style_blocks):
            style1 = styles[i * 2]
            style2 = styles[i * 2 + 1]
            x = style_block(x, style1, style2)
        
        # Convert to RGB
        rgb = torch.tanh(self.to_rgb_layers[-1](x))
        return rgb

# ============================================================================
# STYLEGAN DISCRIMINATOR
# ============================================================================

class StyleGANDiscriminator(nn.Module):
    """
    StyleGAN Discriminator
    
    Progressive discriminator with improved training stability
    """
    
    def __init__(self, max_resolution=64, base_channels=512):
        super(StyleGANDiscriminator, self).__init__()
        
        self.max_resolution = max_resolution
        self.base_channels = base_channels
        
        # From RGB layers
        self.from_rgb_layers = nn.ModuleList()
        
        # Discriminator blocks
        self.blocks = nn.ModuleList()
        
        # Calculate number of resolutions
        num_resolutions = int(math.log2(max_resolution // 4)) + 1
        
        current_channels = 32
        for i in range(num_resolutions):
            # From RGB layer
            from_rgb = nn.Conv2d(3, current_channels, 1)
            self.from_rgb_layers.append(from_rgb)
            
            # Discriminator block
            if i == num_resolutions - 1:
                # Final block
                out_channels = base_channels
            else:
                out_channels = min(current_channels * 2, base_channels)
            
            block = self._make_disc_block(current_channels, out_channels)
            self.blocks.append(block)
            
            current_channels = out_channels
        
        # Final classification layers
        self.final_conv = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.final_linear = nn.Linear(base_channels * 4 * 4, 1)
        
        print(f"  StyleGAN Discriminator: {max_resolution}x{max_resolution} -> real/fake")
    
    def _make_disc_block(self, in_channels, out_channels):
        """Create discriminator block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2)
        )
    
    def forward(self, x):
        """Discriminator forward pass"""
        # Start from highest resolution
        x = F.leaky_relu(self.from_rgb_layers[-1](x), 0.2, inplace=True)
        
        # Progressive downsampling
        for block in reversed(self.blocks):
            x = block(x)
        
        # Final classification
        x = F.leaky_relu(self.final_conv(x), 0.2, inplace=True)
        x = x.view(x.size(0), -1)
        x = self.final_linear(x)
        
        return x

# ============================================================================
# STYLEGAN ARCHITECTURE
# ============================================================================

class StyleGAN_Control(nn.Module):
    """
    StyleGAN - Style-Based Controllable Generation
    
    Revolutionary Innovations:
    - Mapping network for disentangled latent space
    - Style-based synthesis via Adaptive Instance Normalization
    - Controllable generation through style mixing
    - Stochastic variation through noise injection
    - Progressive generation with style control
    - Truncation trick for quality/diversity trade-off
    """
    
    def __init__(self, latent_dim=512, style_dim=512, max_resolution=64):
        super(StyleGAN_Control, self).__init__()
        
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.max_resolution = max_resolution
        
        print(f"Building StyleGAN Control...")
        
        # Generator and Discriminator
        self.generator = StyleGANGenerator(latent_dim, style_dim, max_resolution)
        self.discriminator = StyleGANDiscriminator(max_resolution)
        
        # Calculate statistics
        gen_params = sum(p.numel() for p in self.generator.parameters())
        disc_params = sum(p.numel() for p in self.discriminator.parameters())
        total_params = gen_params + disc_params
        
        print(f"StyleGAN Architecture Summary:")
        print(f"  Latent dimension: {latent_dim}")
        print(f"  Style dimension: {style_dim}")
        print(f"  Max resolution: {max_resolution}x{max_resolution}")
        print(f"  Generator parameters: {gen_params:,}")
        print(f"  Discriminator parameters: {disc_params:,}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Key innovation: Style-based controllable generation")
    
    def generate_latent(self, batch_size, device):
        """Generate random latent codes"""
        return torch.randn(batch_size, self.latent_dim, device=device)
    
    def generate_samples(self, num_samples, device, truncation_psi=1.0):
        """Generate samples with StyleGAN"""
        self.generator.eval()
        
        with torch.no_grad():
            latent_codes = self.generate_latent(num_samples, device)
            samples = self.generator(latent_codes, truncation_psi=truncation_psi)
        
        return samples
    
    def demonstrate_style_mixing(self, device, num_samples=4):
        """Demonstrate style mixing capabilities"""
        self.generator.eval()
        
        with torch.no_grad():
            # Generate two sets of latent codes
            latent1 = self.generate_latent(num_samples, device)
            latent2 = self.generate_latent(num_samples, device)
            
            # Generate with style mixing
            mixed_samples = self.generator.style_mixing(latent1, latent2)
        
        return mixed_samples
    
    def get_stylegan_analysis(self):
        """Analyze StyleGAN innovations"""
        return {
            'stylegan_principles': STYLEGAN_PRINCIPLES,
            'architectural_innovations': [
                'Mapping network for disentangled latent space',
                'Style-based synthesis via AdaIN modulation',
                'Stochastic variation through noise injection',
                'Progressive generation with style control',
                'Truncation trick for quality control'
            ],
            'control_mechanisms': [
                'Style mixing between different latent codes',
                'Layer-specific style control',
                'Truncation for quality/diversity trade-off',
                'Noise injection for stochastic details',
                'Disentangled latent space manipulation'
            ],
            'generation_advantages': [
                'Unprecedented control over generated content',
                'High-quality realistic image synthesis',
                'Smooth latent space interpolation',
                'Disentangled attribute control',
                'Stable training with progressive growth'
            ]
        }

# ============================================================================
# STYLEGAN TRAINING FUNCTION
# ============================================================================

def train_stylegan(model, train_loader, epochs=100, learning_rate=0.002, 
                   r1_regularization=10.0):
    """
    Train StyleGAN with R1 regularization
    
    R1 regularization helps stabilize training by penalizing
    the discriminator gradient norm
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.generator.to(device)
    model.discriminator.to(device)
    
    # Optimizers (Adam with specific β values for StyleGAN)
    optimizer_G = optim.Adam(model.generator.parameters(), lr=learning_rate, betas=(0.0, 0.99))
    optimizer_D = optim.Adam(model.discriminator.parameters(), lr=learning_rate, betas=(0.0, 0.99))
    
    # Training tracking
    generator_losses = []
    discriminator_losses = []
    
    print(f"Training StyleGAN on device: {device}")
    print(f"R1 regularization: {r1_regularization}")
    
    for epoch in range(epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        for batch_idx, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # ================================================================
            # Train Discriminator with R1 regularization
            # ================================================================
            optimizer_D.zero_grad()
            
            # Real images
            real_images.requires_grad_(True)
            real_output = model.discriminator(real_images)
            real_loss = F.softplus(-real_output).mean()
            
            # R1 regularization
            grad_real = torch.autograd.grad(
                outputs=real_output.sum(), inputs=real_images,
                create_graph=True, retain_graph=True
            )[0]
            r1_penalty = (grad_real.view(grad_real.size(0), -1).norm(dim=1) ** 2).mean()
            
            # Fake images
            latent_codes = model.generate_latent(batch_size, device)
            fake_images = model.generator(latent_codes)
            fake_output = model.discriminator(fake_images.detach())
            fake_loss = F.softplus(fake_output).mean()
            
            # Total discriminator loss
            d_loss = real_loss + fake_loss + r1_regularization * r1_penalty
            d_loss.backward()
            optimizer_D.step()
            
            # ================================================================
            # Train Generator
            # ================================================================
            optimizer_G.zero_grad()
            
            # Generate fake images
            fake_output = model.discriminator(fake_images)
            g_loss = F.softplus(-fake_output).mean()
            
            g_loss.backward()
            optimizer_G.step()
            
            # Track statistics
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            
            if batch_idx % 200 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                      f'G_Loss: {g_loss.item():.4f}, D_Loss: {d_loss.item():.4f}, '
                      f'R1: {r1_penalty.item():.4f}')
        
        # Calculate epoch averages
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader)
        
        generator_losses.append(avg_g_loss)
        discriminator_losses.append(avg_d_loss)
        
        print(f'Epoch {epoch+1}/{epochs}: G_Loss: {avg_g_loss:.4f}, D_Loss: {avg_d_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 25 == 0:
            torch.save({
                'generator': model.generator.state_dict(),
                'discriminator': model.discriminator.state_dict(),
                'epoch': epoch
            }, f'AI-ML-DL/Models/Generative_AI/stylegan_epoch_{epoch+1}.pth')
        
        # Early stopping for demonstration
        if avg_g_loss < 1.0 and avg_d_loss < 2.0:
            print(f"Good convergence reached at epoch {epoch+1}")
            break
    
    return generator_losses, discriminator_losses

# ============================================================================
# VISUALIZATION AND ANALYSIS
# ============================================================================

def visualize_stylegan_innovations():
    """Visualize StyleGAN innovations"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Mapping network concept
    ax = axes[0, 0]
    ax.set_title('Mapping Network: z → w Transformation', fontsize=14, fontweight='bold')
    
    # Traditional GAN
    ax.text(0.2, 0.8, 'Traditional GAN', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    ax.text(0.2, 0.6, 'z (input)', ha='center', va='center', fontsize=11,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcoral'))
    ax.text(0.2, 0.4, 'Generator', ha='center', va='center', fontsize=11,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgray'))
    ax.text(0.2, 0.2, 'Image', ha='center', va='center', fontsize=11,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen'))
    
    # Draw arrows
    ax.annotate('', xy=(0.2, 0.5), xytext=(0.2, 0.7),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax.annotate('', xy=(0.2, 0.3), xytext=(0.2, 0.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    # StyleGAN
    ax.text(0.8, 0.8, 'StyleGAN', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    ax.text(0.8, 0.7, 'z (input)', ha='center', va='center', fontsize=11,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcoral'))
    ax.text(0.8, 0.55, 'Mapping Net', ha='center', va='center', fontsize=11,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow'))
    ax.text(0.8, 0.4, 'w (style)', ha='center', va='center', fontsize=11,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue'))
    ax.text(0.8, 0.25, 'Generator', ha='center', va='center', fontsize=11,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgray'))
    ax.text(0.8, 0.1, 'Image', ha='center', va='center', fontsize=11,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen'))
    
    # StyleGAN arrows
    ax.annotate('', xy=(0.8, 0.62), xytext=(0.8, 0.68),
               arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax.annotate('', xy=(0.8, 0.47), xytext=(0.8, 0.53),
               arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax.annotate('', xy=(0.8, 0.32), xytext=(0.8, 0.38),
               arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax.annotate('', xy=(0.8, 0.17), xytext=(0.8, 0.23),
               arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # Advantages
    ax.text(0.5, 0.05, 'StyleGAN Advantages:\n• Better disentanglement\n• Smoother interpolation\n• Style control', 
           ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Style-based synthesis
    ax = axes[0, 1]
    ax.set_title('Style-Based Synthesis via AdaIN', fontsize=14)
    
    # Show AdaIN process
    ax.text(0.2, 0.8, 'Features\n(x)', ha='center', va='center', fontsize=11,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
    
    ax.text(0.8, 0.8, 'Style\n(w)', ha='center', va='center', fontsize=11,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow'))
    
    ax.text(0.2, 0.5, 'Normalize:\n(x - μ) / σ', ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan'))
    
    ax.text(0.8, 0.5, 'Style Transform:\nScale & Bias', ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    ax.text(0.5, 0.2, 'AdaIN Output:\nγ(w) * normalize(x) + β(w)', ha='center', va='center', fontsize=11,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    # Draw arrows
    ax.annotate('', xy=(0.2, 0.6), xytext=(0.2, 0.7),
               arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax.annotate('', xy=(0.8, 0.6), xytext=(0.8, 0.7),
               arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
    ax.annotate('', xy=(0.4, 0.25), xytext=(0.3, 0.45),
               arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.annotate('', xy=(0.6, 0.25), xytext=(0.7, 0.45),
               arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Style mixing demonstration
    ax = axes[1, 0]
    ax.set_title('Style Mixing for Controllable Generation', fontsize=14)
    
    # Show style mixing concept
    styles = ['Style A\n(Coarse)', 'Style B\n(Medium)', 'Style C\n(Fine)']
    layers = ['4x4-8x8', '16x16-32x32', '64x64+']
    colors = ['red', 'blue', 'green']
    
    for i, (style, layer, color) in enumerate(zip(styles, layers, colors)):
        # Style source
        ax.text(0.1, 0.8 - i*0.25, style, ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.3))
        
        # Layer range
        ax.text(0.5, 0.8 - i*0.25, layer, ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.5))
        
        # Arrow
        ax.annotate('', xy=(0.4, 0.8 - i*0.25), xytext=(0.2, 0.8 - i*0.25),
                   arrowprops=dict(arrowstyle='->', lw=2, color=color))
    
    # Final result
    ax.text(0.8, 0.5, 'Mixed\nResult', ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    
    # Arrows to result
    for i in range(3):
        ax.annotate('', xy=(0.7, 0.5), xytext=(0.6, 0.8 - i*0.25),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Truncation trick
    ax = axes[1, 1]
    ax.set_title('Truncation Trick: Quality vs Diversity', fontsize=14)
    
    # Show truncation effect
    psi_values = [0.5, 0.7, 1.0, 1.5]
    quality_scores = [9, 8, 6, 4]
    diversity_scores = [3, 5, 8, 9]
    
    x = np.arange(len(psi_values))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, quality_scores, width, label='Quality', color='lightblue')
    bars2 = ax.bar(x + width/2, diversity_scores, width, label='Diversity', color='lightcoral')
    
    ax.set_xlabel('Truncation ψ')
    ax.set_ylabel('Score (1-10)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'ψ={psi}' for psi in psi_values])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                   f'{height}', ha='center', va='bottom')
    
    # Highlight optimal range
    ax.axvspan(0.5, 1.5, alpha=0.2, color='green', label='Optimal Range')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/Generative_AI/009_stylegan_innovations.png', dpi=300, bbox_inches='tight')
    print("StyleGAN innovations visualization saved: 009_stylegan_innovations.png")

def visualize_stylegan_control():
    """Visualize StyleGAN control mechanisms"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Disentanglement comparison
    ax = axes[0, 0]
    ax.set_title('Latent Space Disentanglement', fontsize=14, fontweight='bold')
    
    # Traditional GAN latent space (entangled)
    ax.text(0.25, 0.9, 'Traditional GAN (z space)', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    
    # Entangled representation
    theta = np.linspace(0, 2*np.pi, 100)
    x_ent = 0.25 + 0.15 * np.cos(theta) + 0.05 * np.cos(3*theta)
    y_ent = 0.5 + 0.15 * np.sin(theta) + 0.05 * np.sin(5*theta)
    ax.plot(x_ent, y_ent, 'r-', linewidth=3, alpha=0.7, label='Entangled')
    
    # StyleGAN latent space (disentangled)
    ax.text(0.75, 0.9, 'StyleGAN (w space)', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    
    # Disentangled representation
    x_dis = np.linspace(0.6, 0.9, 50)
    y_dis1 = np.full_like(x_dis, 0.6)
    y_dis2 = np.full_like(x_dis, 0.4)
    ax.plot(x_dis, y_dis1, 'b-', linewidth=3, label='Attribute 1')
    ax.plot(x_dis, y_dis2, 'g-', linewidth=3, label='Attribute 2')
    
    # Add labels
    ax.text(0.25, 0.3, 'Attributes\nMixed', ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcoral'))
    ax.text(0.75, 0.3, 'Attributes\nSeparated', ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left')
    ax.axis('off')
    
    # Layer-wise style control
    ax = axes[0, 1]
    ax.set_title('Layer-wise Style Control', fontsize=14)
    
    # Show different layers control different features
    layers = ['4x4\n(Pose)', '8x8\n(Face Shape)', '16x16\n(Eyes/Nose)', '32x32\n(Hair)', '64x64\n(Details)']
    resolutions = [4, 8, 16, 32, 64]
    
    for i, (layer, res) in enumerate(zip(layers, resolutions)):
        color = plt.cm.viridis(i / len(layers))
        
        # Layer block
        rect = plt.Rectangle((i*0.18 + 0.1, 0.6), 0.15, 0.3, 
                           facecolor=color, alpha=0.7, edgecolor='black')
        ax.add_patch(rect)
        
        # Layer label
        ax.text(i*0.18 + 0.175, 0.75, layer, ha='center', va='center', 
               fontsize=9, fontweight='bold', color='white')
        
        # Resolution progression
        ax.text(i*0.18 + 0.175, 0.4, f'{res}²', ha='center', va='center', 
               fontsize=11, fontweight='bold')
    
    # Control arrows
    ax.text(0.5, 0.2, 'Different layers control different semantic attributes', 
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Style mixing results simulation
    ax = axes[1, 0]
    ax.set_title('Style Mixing Results', fontsize=14)
    
    # Create a heatmap showing style mixing effects
    mixing_data = np.array([
        [1.0, 0.8, 0.6, 0.3, 0.1],  # Source A dominant
        [0.8, 0.9, 0.7, 0.4, 0.2],  # Mixed early
        [0.6, 0.7, 0.8, 0.6, 0.3],  # Mixed middle
        [0.3, 0.4, 0.6, 0.8, 0.7],  # Mixed late
        [0.1, 0.2, 0.3, 0.7, 1.0]   # Source B dominant
    ])
    
    im = ax.imshow(mixing_data, cmap='RdYlBu', aspect='equal')
    
    # Labels
    ax.set_xticks(range(5))
    ax.set_xticklabels(['4x4', '8x8', '16x16', '32x32', '64x64'])
    ax.set_yticks(range(5))
    ax.set_yticklabels(['All A', 'Early A', 'Mixed', 'Late A', 'All B'])
    
    ax.set_xlabel('Resolution Layer')
    ax.set_ylabel('Style Source')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Source A Influence')
    
    # Control capabilities comparison
    ax = axes[1, 1]
    ax.set_title('Generation Control Capabilities', fontsize=14)
    
    methods = ['Traditional\nGAN', 'VAE', 'StyleGAN']
    capabilities = ['Fine Control', 'Disentanglement', 'Quality', 'Diversity', 'Stability']
    
    # Scores for each method
    scores = np.array([
        [2, 3, 6, 7, 4],  # Traditional GAN
        [4, 6, 5, 6, 8],  # VAE
        [9, 9, 9, 7, 8]   # StyleGAN
    ])
    
    # Create radar chart style comparison
    x = np.arange(len(capabilities))
    width = 0.25
    
    for i, (method, score) in enumerate(zip(methods, scores)):
        bars = ax.bar(x + i*width, score, width, label=method, alpha=0.7)
    
    ax.set_xlabel('Capabilities')
    ax.set_ylabel('Score (1-10)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(capabilities, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/Generative_AI/009_stylegan_control.png', dpi=300, bbox_inches='tight')
    print("StyleGAN control visualization saved: 009_stylegan_control.png")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"=== StyleGAN Control Implementation ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load CIFAR-10 dataset
    train_loader, test_loader, classes = load_cifar10_stylegan()
    
    # Initialize StyleGAN model
    stylegan_model = StyleGAN_Control()
    
    # Analyze model properties
    gen_params = sum(p.numel() for p in stylegan_model.generator.parameters())
    disc_params = sum(p.numel() for p in stylegan_model.discriminator.parameters())
    total_params = gen_params + disc_params
    stylegan_analysis = stylegan_model.get_stylegan_analysis()
    
    print(f"\nStyleGAN Analysis:")
    print(f"  Generator parameters: {gen_params:,}")
    print(f"  Discriminator parameters: {disc_params:,}")
    print(f"  Total parameters: {total_params:,}")
    
    print(f"\nStyleGAN Innovations:")
    for key, value in stylegan_analysis.items():
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
    print("\nGenerating StyleGAN analysis...")
    visualize_stylegan_innovations()
    visualize_stylegan_control()
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("STYLEGAN CONTROL SUMMARY")
    print("="*70)
    
    print(f"\nHistorical Context ({YEAR}):")
    print(f"  Innovation: {INNOVATION}")
    print(f"  Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"  Impact: {IMPACT}")
    
    print(f"\nSTYLEGAN INNOVATIONS:")
    print("="*50)
    print("1. MAPPING NETWORK:")
    print("   • Transform input latent z to intermediate latent w")
    print("   • 8-layer fully connected network")
    print("   • Reduces correlation between latent dimensions")
    print("   • Enables better disentanglement and control")
    
    print("\n2. STYLE-BASED SYNTHESIS:")
    print("   • Apply styles via Adaptive Instance Normalization (AdaIN)")
    print("   • AdaIN(x, w) = γ(w) * normalize(x) + β(w)")
    print("   • Layer-specific style control")
    print("   • Separates content from style information")
    
    print("\n3. STOCHASTIC VARIATION:")
    print("   • Add noise at each resolution for fine details")
    print("   • Enables stochastic features (hair texture, skin pores)")
    print("   • Learnable noise scaling parameters")
    print("   • Separates deterministic structure from random details")
    
    print("\n4. STYLE MIXING:")
    print("   • Use different latent codes for different layers")
    print("   • Coarse layers: pose, face shape")
    print("   • Fine layers: hair, skin texture, details")
    print("   • Enables fine-grained attribute control")
    
    print("\n5. TRUNCATION TRICK:")
    print("   • w = w_avg + ψ(w - w_avg) for quality control")
    print("   • ψ < 1: Higher quality, lower diversity")
    print("   • ψ > 1: Lower quality, higher diversity")
    print("   • Controllable quality/diversity trade-off")
    
    print(f"\nTECHNICAL ACHIEVEMENTS:")
    print("="*40)
    print("• Unprecedented control over generated attributes")
    print("• Disentangled latent space representation")
    print("• High-quality photorealistic image synthesis")
    print("• Stable training with progressive growth")
    print("• Smooth latent space interpolation")
    
    print(f"\nSTYLEGAN PRINCIPLES:")
    for key, principle in STYLEGAN_PRINCIPLES.items():
        print(f"  • {principle}")
    
    print(f"\nARCHITECTURAL INNOVATIONS:")
    for innovation in stylegan_analysis['architectural_innovations']:
        print(f"  • {innovation}")
    
    print(f"\nCONTROL MECHANISMS:")
    for mechanism in stylegan_analysis['control_mechanisms']:
        print(f"  • {mechanism}")
    
    print(f"\nGENERATION ADVANTAGES:")
    for advantage in stylegan_analysis['generation_advantages']:
        print(f"  • {advantage}")
    
    print(f"\nMATHEMATICAL FORMULATION:")
    print("="*40)
    print("• Mapping: w = MLP(z) where z ~ N(0,I)")
    print("• AdaIN: AdaIN(x,w) = γ(A(w)) * (x-μ)/σ + β(A(w))")
    print("• Style Control: Different w for different layers")
    print("• Noise Injection: x = x + N(0,I) * learned_scale")
    print("• Truncation: w = w_avg + ψ(w - w_avg)")
    
    print(f"\nLAYER-WISE STYLE CONTROL:")
    print("="*40)
    print("• 4x4-8x8: Pose, general face structure")
    print("• 16x16-32x32: Facial features, eyes, nose, mouth")
    print("• 64x64-512x512: Hair style, skin texture, fine details")
    print("• 1024x1024: Micro-details, skin pores, hair strands")
    
    print(f"\nDISENTANGLEMENT BENEFITS:")
    print("="*40)
    print("• Traditional z space: Entangled attributes")
    print("• StyleGAN w space: Disentangled attributes")
    print("• Smooth interpolation: Linear changes in w space")
    print("• Semantic editing: Modify specific attributes independently")
    print("• Style transfer: Copy styles between different images")
    
    print(f"\nSTYLE MIXING APPLICATIONS:")
    print("="*40)
    print("• Attribute transfer: Take pose from A, features from B")
    print("• Creative exploration: Mix unexpected combinations")
    print("• Data augmentation: Generate controlled variations")
    print("• Style analysis: Understand layer-wise contributions")
    print("• Interactive editing: Real-time style manipulation")
    
    print(f"\nHISTORICAL IMPACT:")
    print("="*40)
    print("• Revolutionized controllable image generation")
    print("• Established style-based generation paradigm")
    print("• Enabled practical image editing applications")
    print("• Set new standards for generation quality")
    print("• Inspired numerous follow-up works and applications")
    print("• Made GANs accessible for creative and commercial use")
    
    print(f"\nCOMPARISON TO PREVIOUS WORK:")
    print("="*40)
    print("• Progressive GAN: High resolution but limited control")
    print("• StyleGAN: High resolution + unprecedented control")
    print("• Traditional GANs: Entangled latent space")
    print("• StyleGAN: Disentangled style-based control")
    print("• Previous: Global image manipulation")
    print("• StyleGAN: Layer-wise semantic control")
    
    print(f"\nLIMITATIONS AND FUTURE DIRECTIONS:")
    print("="*40)
    print("• Training complexity: Requires careful tuning")
    print("• Computational cost: Large model and training time")
    print("• Domain specificity: Works best on aligned domains")
    print("• → StyleGAN2: Improved architecture and quality")
    print("• → StyleGAN3: Translation and rotation equivariance")
    print("• → Various editing methods: Latent space manipulation")
    
    print(f"\nMODERN RELEVANCE:")
    print("="*40)
    print("• Style-based generation: Used across generative modeling")
    print("• Controllable synthesis: Standard expectation")
    print("• Disentangled representations: Active research area")
    print("• Creative applications: Widely adopted in art and design")
    print("• Commercial deployment: Fashion, gaming, entertainment")
    
    print(f"\nERA 3 COMPLETION:")
    print("="*40)
    print("• Progressive GAN: Enabled high-resolution generation")
    print("• CycleGAN: Enabled unpaired domain translation")
    print("• StyleGAN: Enabled unprecedented controllable generation")
    print("• → Advanced GANs established practical deployment")
    print("• → Set stage for modern generative AI applications")
    print("• → Foundation for creative and commercial tools")
    
    return {
        'model': 'StyleGAN Control',
        'year': YEAR,
        'innovation': INNOVATION,
        'generator_params': gen_params,
        'discriminator_params': disc_params,
        'total_params': total_params,
        'stylegan_analysis': stylegan_analysis
    }

if __name__ == "__main__":
    results = main()