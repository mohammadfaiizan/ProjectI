"""
ERA 2: GAN REVOLUTION & STABILIZATION - DCGAN Architectural Revolution
======================================================================

Year: 2015
Paper: "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" (Radford et al.)
Innovation: Convolutional architectures with architectural guidelines for stable GAN training
Previous Limitation: Training instability and architectural chaos in early GANs
Performance Gain: Stable training with high-quality image generation at higher resolutions
Impact: Established architectural best practices that enabled practical GAN deployment

This file implements DCGAN that revolutionized GAN training through careful architectural design,
establishing the convolutional GAN paradigm that remains influential today.
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

YEAR = "2015"
INNOVATION = "Convolutional architectures with architectural guidelines for stable GAN training"
PREVIOUS_LIMITATION = "Training instability and architectural chaos in early GANs"
IMPACT = "Established architectural best practices that enabled practical GAN deployment"

print(f"=== DCGAN Architectural Revolution ({YEAR}) ===")
print(f"Innovation: {INNOVATION}")
print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
print(f"Impact: {IMPACT}")
print("="*60)

# ============================================================================
# DCGAN ARCHITECTURAL GUIDELINES
# ============================================================================

DCGAN_GUIDELINES = {
    "replace_pooling": "Replace pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator)",
    "batch_normalization": "Use batch normalization in both generator and discriminator",
    "remove_fc_layers": "Remove fully connected hidden layers for deeper architectures",
    "activation_generator": "Use ReLU activation in generator for all layers except output (tanh)",
    "activation_discriminator": "Use LeakyReLU activation in discriminator for all layers",
    "no_max_pooling": "Eliminate max pooling layers completely",
    "transposed_conv": "Use transposed convolutions for upsampling in generator",
    "strided_conv": "Use strided convolutions for downsampling in discriminator"
}

print("DCGAN Architectural Guidelines:")
for key, guideline in DCGAN_GUIDELINES.items():
    print(f"  • {guideline}")
print()

# ============================================================================
# STANDARDIZED DATASET LOADING (CIFAR-10)
# ============================================================================

def load_cifar10_dataset():
    """Load CIFAR-10 dataset with DCGAN preprocessing"""
    print("Loading CIFAR-10 dataset for DCGAN architectural study...")
    
    # DCGAN preprocessing - normalize to [-1, 1]
    transform_train = transforms.Compose([
        transforms.Resize(64),  # DCGAN typically works with 64x64
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
    
    # Create data loaders with DCGAN batch size
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2, drop_last=True)
    
    # CIFAR-10 class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print(f"Classes: {len(classes)}")
    print(f"Processed image size: 64x64 RGB (DCGAN standard)")
    print(f"Batch size: 128 (DCGAN recommendation)")
    print(f"Pixel range: [-1, 1] (DCGAN standard)")
    
    return train_loader, test_loader, classes

# ============================================================================
# DCGAN GENERATOR
# ============================================================================

class DCGAN_Generator(nn.Module):
    """
    DCGAN Generator - Revolutionary Convolutional Architecture
    
    Key Innovations:
    - Transposed convolutions for upsampling (no pooling)
    - Batch normalization for training stability
    - ReLU activations (Tanh for output)
    - No fully connected layers in main path
    - Progressive upsampling from noise to image
    """
    
    def __init__(self, noise_dim=100, num_channels=3, feature_maps=64):
        super(DCGAN_Generator, self).__init__()
        
        self.noise_dim = noise_dim
        self.num_channels = num_channels
        self.feature_maps = feature_maps  # ngf in original paper
        
        # Initial projection from noise to feature maps
        # 100 -> 4x4x(8*feature_maps)
        self.initial_conv = nn.ConvTranspose2d(
            noise_dim, feature_maps * 8, 4, 1, 0, bias=False
        )
        self.initial_bn = nn.BatchNorm2d(feature_maps * 8)
        
        # Progressive upsampling layers following DCGAN architecture
        # 4x4x(8*fm) -> 8x8x(4*fm)
        self.conv1 = nn.ConvTranspose2d(
            feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(feature_maps * 4)
        
        # 8x8x(4*fm) -> 16x16x(2*fm)
        self.conv2 = nn.ConvTranspose2d(
            feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(feature_maps * 2)
        
        # 16x16x(2*fm) -> 32x32x(fm)
        self.conv3 = nn.ConvTranspose2d(
            feature_maps * 2, feature_maps, 4, 2, 1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(feature_maps)
        
        # 32x32x(fm) -> 64x64x3
        self.conv4 = nn.ConvTranspose2d(
            feature_maps, num_channels, 4, 2, 1, bias=False
        )
        
        # Initialize weights according to DCGAN paper
        self._initialize_weights()
        
        print(f"  DCGAN Generator Architecture:")
        print(f"    Input: {noise_dim}D noise vector")
        print(f"    4x4x{feature_maps*8} -> 8x8x{feature_maps*4} -> 16x16x{feature_maps*2} -> 32x32x{feature_maps} -> 64x64x{num_channels}")
        print(f"    Key features: Transposed convs, BatchNorm, ReLU/Tanh, No FC layers")
    
    def _initialize_weights(self):
        """Initialize weights according to DCGAN paper"""
        for module in self.modules():
            if isinstance(module, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(module.weight, 0.0, 0.02)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight, 1.0, 0.02)
                nn.init.zeros_(module.bias)
    
    def forward(self, noise):
        """
        DCGAN Generator Forward Pass
        Noise -> 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 image
        """
        # Reshape noise for convolution: (batch, noise_dim, 1, 1)
        x = noise.view(noise.size(0), noise.size(1), 1, 1)
        
        # Initial convolution: noise -> 4x4 feature maps
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = F.relu(x, inplace=True)
        
        # Progressive upsampling with DCGAN architecture
        # 4x4 -> 8x8
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        
        # 8x8 -> 16x16
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        
        # 16x16 -> 32x32
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)
        
        # 32x32 -> 64x64 (final layer)
        x = self.conv4(x)
        x = torch.tanh(x)  # Output in [-1, 1] range
        
        return x

# ============================================================================
# DCGAN DISCRIMINATOR
# ============================================================================

class DCGAN_Discriminator(nn.Module):
    """
    DCGAN Discriminator - Revolutionary Convolutional Architecture
    
    Key Innovations:
    - Strided convolutions for downsampling (no pooling)
    - Batch normalization (except first layer)
    - LeakyReLU activations throughout
    - No fully connected layers except final classification
    - Progressive downsampling from image to classification
    """
    
    def __init__(self, num_channels=3, feature_maps=64):
        super(DCGAN_Discriminator, self).__init__()
        
        self.num_channels = num_channels
        self.feature_maps = feature_maps  # ndf in original paper
        
        # Progressive downsampling layers following DCGAN architecture
        # 64x64x3 -> 32x32x(fm)
        self.conv1 = nn.Conv2d(
            num_channels, feature_maps, 4, 2, 1, bias=False
        )
        # No BatchNorm on first layer (DCGAN guideline)
        
        # 32x32x(fm) -> 16x16x(2*fm)
        self.conv2 = nn.Conv2d(
            feature_maps, feature_maps * 2, 4, 2, 1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(feature_maps * 2)
        
        # 16x16x(2*fm) -> 8x8x(4*fm)
        self.conv3 = nn.Conv2d(
            feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(feature_maps * 4)
        
        # 8x8x(4*fm) -> 4x4x(8*fm)
        self.conv4 = nn.Conv2d(
            feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(feature_maps * 8)
        
        # 4x4x(8*fm) -> 1x1x1 (classification)
        self.conv5 = nn.Conv2d(
            feature_maps * 8, 1, 4, 1, 0, bias=False
        )
        
        # Initialize weights according to DCGAN paper
        self._initialize_weights()
        
        print(f"  DCGAN Discriminator Architecture:")
        print(f"    64x64x{num_channels} -> 32x32x{feature_maps} -> 16x16x{feature_maps*2} -> 8x8x{feature_maps*4} -> 4x4x{feature_maps*8} -> 1x1x1")
        print(f"    Key features: Strided convs, BatchNorm (except first), LeakyReLU, No pooling")
    
    def _initialize_weights(self):
        """Initialize weights according to DCGAN paper"""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(module.weight, 0.0, 0.02)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight, 1.0, 0.02)
                nn.init.zeros_(module.bias)
    
    def forward(self, img):
        """
        DCGAN Discriminator Forward Pass
        64x64 image -> 32x32 -> 16x16 -> 8x8 -> 4x4 -> classification
        """
        # Progressive downsampling with DCGAN architecture
        # 64x64 -> 32x32 (no BatchNorm on first layer)
        x = self.conv1(img)
        x = F.leaky_relu(x, 0.2, inplace=True)
        
        # 32x32 -> 16x16
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        
        # 16x16 -> 8x8
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        
        # 8x8 -> 4x4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        
        # 4x4 -> 1x1 (classification)
        x = self.conv5(x)
        x = torch.sigmoid(x)  # Probability of being real
        
        return x.view(-1, 1).squeeze(1)  # Flatten to (batch_size,)

# ============================================================================
# DCGAN ARCHITECTURE
# ============================================================================

class DCGAN_ArchitecturalRevolution(nn.Module):
    """
    DCGAN - Deep Convolutional Generative Adversarial Networks
    
    Architectural Revolution:
    - Convolutional architectures throughout (no FC layers)
    - Batch normalization for training stability
    - Specific activation functions (ReLU/LeakyReLU)
    - Strided convolutions replace pooling
    - Architectural guidelines for stable training
    - Higher resolution generation (64x64)
    """
    
    def __init__(self, noise_dim=100, num_channels=3, feature_maps_g=64, feature_maps_d=64):
        super(DCGAN_ArchitecturalRevolution, self).__init__()
        
        self.noise_dim = noise_dim
        self.num_channels = num_channels
        
        print(f"Building DCGAN Architectural Revolution...")
        
        # Generator and Discriminator with DCGAN architecture
        self.generator = DCGAN_Generator(noise_dim, num_channels, feature_maps_g)
        self.discriminator = DCGAN_Discriminator(num_channels, feature_maps_d)
        
        # Calculate statistics
        gen_params = sum(p.numel() for p in self.generator.parameters())
        disc_params = sum(p.numel() for p in self.discriminator.parameters())
        total_params = gen_params + disc_params
        
        print(f"DCGAN Architecture Summary:")
        print(f"  Noise dimension: {noise_dim}")
        print(f"  Output resolution: 64x64x{num_channels}")
        print(f"  Generator parameters: {gen_params:,}")
        print(f"  Discriminator parameters: {disc_params:,}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Key innovation: Convolutional architecture + training guidelines")
    
    def generate_noise(self, batch_size, device):
        """Generate random noise for DCGAN generator"""
        return torch.randn(batch_size, self.noise_dim, device=device)
    
    def generate_samples(self, num_samples, device):
        """Generate samples using DCGAN generator"""
        self.generator.eval()
        
        with torch.no_grad():
            noise = self.generate_noise(num_samples, device)
            samples = self.generator(noise)
        
        return samples
    
    def get_architectural_analysis(self):
        """Analyze DCGAN architectural innovations"""
        return {
            'architectural_guidelines': DCGAN_GUIDELINES,
            'generator_features': [
                'Transposed convolutions for upsampling',
                'Batch normalization (except output layer)',
                'ReLU activations (Tanh for output)',
                'No fully connected layers',
                'Progressive upsampling: 4x4 -> 64x64'
            ],
            'discriminator_features': [
                'Strided convolutions for downsampling',
                'Batch normalization (except first layer)',
                'LeakyReLU activations throughout',
                'No max pooling layers',
                'Progressive downsampling: 64x64 -> 4x4'
            ],
            'training_improvements': [
                'More stable training than vanilla GAN',
                'Higher resolution generation capability',
                'Better gradient flow through architecture',
                'Reduced mode collapse tendency',
                'Faster convergence'
            ]
        }

# ============================================================================
# DCGAN TRAINING FUNCTION
# ============================================================================

def train_dcgan(model, train_loader, epochs=100, learning_rate=0.0002, beta1=0.5):
    """Train DCGAN with architectural best practices"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.generator.to(device)
    model.discriminator.to(device)
    
    # DCGAN optimizer settings (from paper)
    optimizer_G = optim.Adam(model.generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(model.discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Training tracking
    generator_losses = []
    discriminator_losses = []
    real_scores = []
    fake_scores = []
    
    print(f"Training DCGAN on device: {device}")
    print(f"Learning rate: {learning_rate}, Beta1: {beta1}")
    
    # Fixed noise for consistent generation tracking
    fixed_noise = model.generate_noise(64, device)
    
    for epoch in range(epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_real_score = 0.0
        epoch_fake_score = 0.0
        
        for batch_idx, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Labels for real and fake
            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)
            
            # ================================================================
            # Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            # ================================================================
            optimizer_D.zero_grad()
            
            # Real images
            real_output = model.discriminator(real_images)
            d_loss_real = criterion(real_output, real_labels)
            d_loss_real.backward()
            
            # Fake images
            noise = model.generate_noise(batch_size, device)
            fake_images = model.generator(noise)
            fake_output = model.discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            d_loss_fake.backward()
            
            # Update discriminator
            optimizer_D.step()
            d_loss = d_loss_real + d_loss_fake
            
            # ================================================================
            # Train Generator: maximize log(D(G(z)))
            # ================================================================
            optimizer_G.zero_grad()
            
            # Generate fake images and get discriminator output
            fake_output = model.discriminator(fake_images)
            g_loss = criterion(fake_output, real_labels)  # Want D to think these are real
            g_loss.backward()
            optimizer_G.step()
            
            # Track statistics
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_real_score += real_output.mean().item()
            epoch_fake_score += fake_output.mean().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                      f'G_Loss: {g_loss.item():.4f}, D_Loss: {d_loss.item():.4f}, '
                      f'D(x): {real_output.mean().item():.4f}, D(G(z)): {fake_output.mean().item():.4f}')
        
        # Calculate epoch averages
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader)
        avg_real_score = epoch_real_score / len(train_loader)
        avg_fake_score = epoch_fake_score / len(train_loader)
        
        generator_losses.append(avg_g_loss)
        discriminator_losses.append(avg_d_loss)
        real_scores.append(avg_real_score)
        fake_scores.append(avg_fake_score)
        
        print(f'Epoch {epoch+1}/{epochs}: G_Loss: {avg_g_loss:.4f}, D_Loss: {avg_d_loss:.4f}, '
              f'D(x): {avg_real_score:.4f}, D(G(z)): {avg_fake_score:.4f}')
        
        # Save model checkpoint
        if (epoch + 1) % 25 == 0:
            torch.save({
                'generator': model.generator.state_dict(),
                'discriminator': model.discriminator.state_dict(),
                'epoch': epoch,
                'g_loss': avg_g_loss,
                'd_loss': avg_d_loss
            }, f'AI-ML-DL/Models/Generative_AI/dcgan_epoch_{epoch+1}.pth')
        
        # Early stopping for demonstration
        if avg_g_loss < 1.0 and avg_d_loss < 1.5 and abs(avg_real_score - avg_fake_score) < 0.3:
            print(f"Good training balance reached at epoch {epoch+1}")
            break
    
    return generator_losses, discriminator_losses, real_scores, fake_scores

# ============================================================================
# VISUALIZATION AND ANALYSIS
# ============================================================================

def visualize_dcgan_innovations():
    """Visualize DCGAN's architectural innovations"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # DCGAN architecture diagram
    ax = axes[0, 0]
    ax.set_title('DCGAN Architecture vs Vanilla GAN', fontsize=14, fontweight='bold')
    
    # Generator comparison
    ax.text(0.25, 0.9, 'Generator Evolution', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    
    # Vanilla GAN
    ax.text(0.1, 0.7, 'Vanilla GAN:\nFC Layers\nReLU/Sigmoid\nNo BatchNorm', 
           ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
    
    # DCGAN
    ax.text(0.4, 0.7, 'DCGAN:\nTransposed Conv\nReLU/Tanh\nBatchNorm', 
           ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    # Discriminator comparison
    ax.text(0.75, 0.9, 'Discriminator Evolution', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    
    # Vanilla GAN
    ax.text(0.6, 0.7, 'Vanilla GAN:\nFC Layers\nSigmoid\nNo BatchNorm', 
           ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
    
    # DCGAN
    ax.text(0.9, 0.7, 'DCGAN:\nStrided Conv\nLeakyReLU\nBatchNorm', 
           ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    # Improvements
    improvements = [
        'Higher Resolution (64x64)',
        'Stable Training',
        'Better Gradients',
        'Reduced Mode Collapse'
    ]
    
    for i, improvement in enumerate(improvements):
        ax.text(0.5, 0.4 - i*0.08, f'✓ {improvement}', ha='center', va='center', 
               fontsize=10, color='darkgreen', fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # DCGAN guidelines visualization
    ax = axes[0, 1]
    ax.set_title('DCGAN Architectural Guidelines', fontsize=14)
    
    guidelines = [
        'Replace pooling with strided convolutions',
        'Use batch normalization',
        'Remove fully connected layers',
        'Use ReLU in generator (Tanh output)',
        'Use LeakyReLU in discriminator'
    ]
    
    for i, guideline in enumerate(guidelines):
        color = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink'][i]
        ax.text(0.5, 0.9 - i*0.15, guideline, ha='center', va='center', 
               fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Training stability comparison
    ax = axes[1, 0]
    ax.set_title('Training Stability: Vanilla GAN vs DCGAN', fontsize=14)
    
    epochs = np.arange(1, 51)
    
    # Vanilla GAN (unstable)
    vanilla_g = 3.0 + 0.5 * np.sin(epochs/2) + 0.3 * np.random.randn(50)
    vanilla_d = 2.5 + 0.8 * np.cos(epochs/3) + 0.4 * np.random.randn(50)
    
    # DCGAN (stable)
    dcgan_g = 2.0 * np.exp(-epochs/30) + 1.0 + 0.1 * np.sin(epochs/5)
    dcgan_d = 1.8 * np.exp(-epochs/25) + 0.8 + 0.1 * np.cos(epochs/4)
    
    ax.plot(epochs, vanilla_g, 'r--', label='Vanilla G Loss', alpha=0.7, linewidth=2)
    ax.plot(epochs, vanilla_d, 'b--', label='Vanilla D Loss', alpha=0.7, linewidth=2)
    ax.plot(epochs, dcgan_g, 'r-', label='DCGAN G Loss', linewidth=2)
    ax.plot(epochs, dcgan_d, 'b-', label='DCGAN D Loss', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('DCGAN: Much More Stable Training')
    
    # Resolution and quality comparison
    ax = axes[1, 1]
    ax.set_title('Generation Quality Improvements', fontsize=14)
    
    models = ['Vanilla GAN\n(28x28)', 'DCGAN\n(64x64)', 'Future\n(256x256+)']
    quality_scores = [4, 7, 9]
    resolution = [28*28, 64*64, 256*256]
    stability = [3, 8, 9]
    
    x = np.arange(len(models))
    width = 0.25
    
    bars1 = ax.bar(x - width, quality_scores, width, label='Quality', color='skyblue')
    bars2 = ax.bar(x, [r/1000 for r in resolution], width, label='Resolution (k pixels)', color='lightgreen')
    bars3 = ax.bar(x + width, stability, width, label='Training Stability', color='lightcoral')
    
    ax.set_ylabel('Score / Value')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                   f'{height}', ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
               f'{height:.1f}k', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/Generative_AI/004_dcgan_innovations.png', dpi=300, bbox_inches='tight')
    print("DCGAN innovations visualization saved: 004_dcgan_innovations.png")

def visualize_dcgan_architecture():
    """Visualize detailed DCGAN architecture"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Generator architecture
    ax = axes[0]
    ax.set_title('DCGAN Generator Architecture', fontsize=16, fontweight='bold')
    
    # Generator layers
    layers = [
        ('Input', '100D\nNoise', 'lightyellow', (0.05, 0.5)),
        ('Conv1', '4x4x512\n(Transposed)', 'lightblue', (0.2, 0.5)),
        ('Conv2', '8x8x256\n(Transposed)', 'lightgreen', (0.35, 0.5)),
        ('Conv3', '16x16x128\n(Transposed)', 'lightcoral', (0.5, 0.5)),
        ('Conv4', '32x32x64\n(Transposed)', 'lightpink', (0.65, 0.5)),
        ('Output', '64x64x3\nImage', 'lightgray', (0.8, 0.5))
    ]
    
    for name, size, color, (x, y) in layers:
        # Draw layer box
        rect = plt.Rectangle((x-0.06, y-0.15), 0.12, 0.3, 
                           facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Add text
        ax.text(x, y+0.05, name, ha='center', va='center', 
               fontsize=11, fontweight='bold')
        ax.text(x, y-0.05, size, ha='center', va='center', 
               fontsize=9)
    
    # Draw arrows
    arrow_positions = [(0.11, 0.5), (0.26, 0.5), (0.41, 0.5), (0.56, 0.5), (0.71, 0.5)]
    for x, y in arrow_positions:
        ax.annotate('', xy=(x+0.07, y), xytext=(x, y),
                   arrowprops=dict(arrowstyle='->', lw=3, color='darkblue'))
    
    # Add activation info
    activations = ['ReLU+BN', 'ReLU+BN', 'ReLU+BN', 'ReLU+BN', 'Tanh']
    act_positions = [0.2, 0.35, 0.5, 0.65, 0.8]
    
    for act, x in zip(activations, act_positions):
        ax.text(x, 0.8, act, ha='center', va='center', 
               fontsize=9, color='darkred', fontweight='bold')
    
    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Discriminator architecture
    ax = axes[1]
    ax.set_title('DCGAN Discriminator Architecture', fontsize=16, fontweight='bold')
    
    # Discriminator layers
    d_layers = [
        ('Input', '64x64x3\nImage', 'lightgray', (0.05, 0.5)),
        ('Conv1', '32x32x64\n(Strided)', 'lightpink', (0.2, 0.5)),
        ('Conv2', '16x16x128\n(Strided)', 'lightcoral', (0.35, 0.5)),
        ('Conv3', '8x8x256\n(Strided)', 'lightgreen', (0.5, 0.5)),
        ('Conv4', '4x4x512\n(Strided)', 'lightblue', (0.65, 0.5)),
        ('Output', '1x1x1\nReal/Fake', 'lightyellow', (0.8, 0.5))
    ]
    
    for name, size, color, (x, y) in d_layers:
        # Draw layer box
        rect = plt.Rectangle((x-0.06, y-0.15), 0.12, 0.3, 
                           facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Add text
        ax.text(x, y+0.05, name, ha='center', va='center', 
               fontsize=11, fontweight='bold')
        ax.text(x, y-0.05, size, ha='center', va='center', 
               fontsize=9)
    
    # Draw arrows
    for x, y in arrow_positions:
        ax.annotate('', xy=(x+0.07, y), xytext=(x, y),
                   arrowprops=dict(arrowstyle='->', lw=3, color='darkred'))
    
    # Add activation info
    d_activations = ['LeakyReLU', 'LeakyReLU+BN', 'LeakyReLU+BN', 'LeakyReLU+BN', 'Sigmoid']
    
    for act, x in zip(d_activations, act_positions):
        ax.text(x, 0.8, act, ha='center', va='center', 
               fontsize=9, color='darkblue', fontweight='bold')
    
    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/Generative_AI/004_dcgan_architecture.png', dpi=300, bbox_inches='tight')
    print("DCGAN architecture visualization saved: 004_dcgan_architecture.png")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"=== DCGAN Architectural Revolution Implementation ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load CIFAR-10 dataset
    train_loader, test_loader, classes = load_cifar10_dataset()
    
    # Initialize DCGAN model
    dcgan_model = DCGAN_ArchitecturalRevolution()
    
    # Analyze model properties
    gen_params = sum(p.numel() for p in dcgan_model.generator.parameters())
    disc_params = sum(p.numel() for p in dcgan_model.discriminator.parameters())
    total_params = gen_params + disc_params
    architectural_analysis = dcgan_model.get_architectural_analysis()
    
    print(f"\nDCGAN Analysis:")
    print(f"  Generator parameters: {gen_params:,}")
    print(f"  Discriminator parameters: {disc_params:,}")
    print(f"  Total parameters: {total_params:,}")
    
    print(f"\nArchitectural Features:")
    for key, value in architectural_analysis.items():
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
    print("\nGenerating DCGAN analysis...")
    visualize_dcgan_innovations()
    visualize_dcgan_architecture()
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("DCGAN ARCHITECTURAL REVOLUTION SUMMARY")
    print("="*70)
    
    print(f"\nHistorical Context ({YEAR}):")
    print(f"  Innovation: {INNOVATION}")
    print(f"  Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"  Impact: {IMPACT}")
    
    print(f"\nDCGAN ARCHITECTURAL INNOVATIONS:")
    print("="*50)
    print("1. CONVOLUTIONAL ARCHITECTURE:")
    print("   • Replaced fully connected layers with convolutions")
    print("   • Transposed convolutions for generator upsampling")
    print("   • Strided convolutions for discriminator downsampling")
    print("   • Eliminated max pooling layers completely")
    
    print("\n2. BATCH NORMALIZATION:")
    print("   • Applied to both generator and discriminator")
    print("   • Exception: No BatchNorm on discriminator first layer")
    print("   • Exception: No BatchNorm on generator output layer")
    print("   • Significantly stabilized training dynamics")
    
    print("\n3. ACTIVATION FUNCTIONS:")
    print("   • Generator: ReLU for hidden layers, Tanh for output")
    print("   • Discriminator: LeakyReLU for all layers")
    print("   • Careful choice based on empirical analysis")
    print("   • Improved gradient flow and training stability")
    
    print("\n4. ARCHITECTURAL GUIDELINES:")
    print("   • Systematic design principles for stable GAN training")
    print("   • Higher resolution generation (64x64 vs 28x28)")
    print("   • Better feature learning through convolutional structure")
    print("   • Reproducible training procedures")
    
    print(f"\nTECHNICAL ACHIEVEMENTS:")
    print("="*40)
    print("• First stable convolutional GAN architecture")
    print("• Higher resolution image generation (64x64)")
    print("• Systematic architectural design principles")
    print("• Reproducible training procedures")
    print("• Foundation for all modern GAN architectures")
    
    print(f"\nARCHITECTURAL GUIDELINES:")
    for key, guideline in DCGAN_GUIDELINES.items():
        print(f"  • {guideline}")
    
    print(f"\nDCGAN vs VANILLA GAN:")
    print("="*40)
    print("• Vanilla: Fully connected, unstable, low resolution")
    print("• DCGAN: Convolutional, stable, higher resolution")
    print("• Vanilla: No batch normalization, poor gradients")
    print("• DCGAN: Batch normalization, improved gradients")
    print("• Vanilla: ReLU/Sigmoid without guidelines")
    print("• DCGAN: Systematic activation function choices")
    
    print(f"\nTRAINING IMPROVEMENTS:")
    for improvement in architectural_analysis['training_improvements']:
        print(f"  • {improvement}")
    
    print(f"\nGENERATOR INNOVATIONS:")
    for feature in architectural_analysis['generator_features']:
        print(f"  • {feature}")
    
    print(f"\nDISCRIMINATOR INNOVATIONS:")
    for feature in architectural_analysis['discriminator_features']:
        print(f"  • {feature}")
    
    print(f"\nHISTORICAL IMPACT:")
    print("="*40)
    print("• Established convolutional GAN paradigm")
    print("• Enabled practical GAN applications")
    print("• Set architectural standards for future GANs")
    print("• Bridged gap between research and deployment")
    print("• Inspired Progressive GAN, StyleGAN, BigGAN")
    print("• Made GANs accessible to broader research community")
    
    print(f"\nARCHITECTURAL LEGACY:")
    print("="*40)
    print("• Transposed convolutions → Standard for generators")
    print("• Strided convolutions → Standard for discriminators")
    print("• Batch normalization → Adopted across deep learning")
    print("• Systematic design → Influenced all neural architectures")
    print("• Training guidelines → Foundation for stable GAN training")
    
    print(f"\nLIMITATIONS ADDRESSED BY LATER WORK:")
    print("="*40)
    print("• Limited resolution (64x64) → Progressive GAN (1024x1024)")
    print("• Training instability → WGAN, Spectral Normalization")
    print("• Mode collapse → Feature matching, Minibatch discrimination")
    print("• Evaluation metrics → Inception Score, FID")
    print("• Architecture rigidity → Self-attention, Progressive growing")
    
    print(f"\nMODERN RELEVANCE:")
    print("="*40)
    print("• DCGAN principles still used in StyleGAN, BigGAN")
    print("• Convolutional architectures standard in generative AI")
    print("• Batch normalization ubiquitous in deep learning")
    print("• Training guidelines influence modern GAN research")
    print("• Architectural thinking applied to Transformers, Diffusion")
    
    return {
        'model': 'DCGAN Architectural Revolution',
        'year': YEAR,
        'innovation': INNOVATION,
        'generator_params': gen_params,
        'discriminator_params': disc_params,
        'total_params': total_params,
        'architectural_analysis': architectural_analysis
    }

if __name__ == "__main__":
    results = main()