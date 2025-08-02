"""
ERA 2: GAN REVOLUTION & STABILIZATION - Wasserstein GAN Theoretical Foundation
=============================================================================

Year: 2017
Paper: "Wasserstein GAN" (Arjovsky, Chintala & Bottou)
Innovation: Wasserstein distance for principled GAN training with theoretical guarantees
Previous Limitation: Vanishing gradients, training instability, and lack of meaningful loss
Performance Gain: Stable training, meaningful loss correlation with quality, no mode collapse
Impact: Provided theoretical foundation for GAN training and inspired distance-based approaches

This file implements WGAN that revolutionized GAN training through principled mathematical
foundations using the Wasserstein distance, solving fundamental issues with the original formulation.
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

YEAR = "2017"
INNOVATION = "Wasserstein distance for principled GAN training with theoretical guarantees"
PREVIOUS_LIMITATION = "Vanishing gradients, training instability, and lack of meaningful loss"
IMPACT = "Provided theoretical foundation for GAN training and inspired distance-based approaches"

print(f"=== Wasserstein GAN Theoretical Foundation ({YEAR}) ===")
print(f"Innovation: {INNOVATION}")
print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
print(f"Impact: {IMPACT}")
print("="*60)

# ============================================================================
# WASSERSTEIN DISTANCE THEORY
# ============================================================================

WGAN_THEORY = {
    "wasserstein_distance": "W(P_r, P_g) = inf_{γ∈Π(P_r,P_g)} E_{(x,y)~γ}[||x-y||]",
    "kantorovich_rubinstein": "W(P_r, P_g) = sup_{||f||_L≤1} E_{x~P_r}[f(x)] - E_{x~P_g}[f(x)]",
    "lipschitz_constraint": "Critic function f must be 1-Lipschitz: |f(x) - f(y)| ≤ ||x - y||",
    "critic_vs_discriminator": "Critic outputs real values, not probabilities",
    "weight_clipping": "Enforce Lipschitz constraint by clipping weights to [-c, c]",
    "meaningful_loss": "WGAN loss correlates with sample quality",
    "gradient_properties": "Always provides useful gradients, even at optimum"
}

print("WGAN Theoretical Foundations:")
for key, theory in WGAN_THEORY.items():
    print(f"  • {theory}")
print()

# ============================================================================
# STANDARDIZED DATASET LOADING (CIFAR-10)
# ============================================================================

def load_cifar10_dataset():
    """Load CIFAR-10 dataset for WGAN theoretical study"""
    print("Loading CIFAR-10 dataset for WGAN theoretical foundation study...")
    
    # WGAN preprocessing
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
    
    # Create data loaders with WGAN batch size
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, drop_last=True)
    
    # CIFAR-10 class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print(f"Classes: {len(classes)}")
    print(f"Image size: 32x32 RGB")
    print(f"Batch size: 64 (WGAN optimization)")
    print(f"Focus: Theoretical foundations and stable training")
    
    return train_loader, test_loader, classes

# ============================================================================
# WGAN GENERATOR
# ============================================================================

class WGAN_Generator(nn.Module):
    """
    WGAN Generator - Same architecture as DCGAN but adapted for Wasserstein loss
    
    Key Features:
    - DCGAN-style architecture
    - Optimized for Wasserstein distance minimization
    - No sigmoid output (uses tanh)
    - Stable training with WGAN loss
    """
    
    def __init__(self, noise_dim=100, num_channels=3, feature_maps=64):
        super(WGAN_Generator, self).__init__()
        
        self.noise_dim = noise_dim
        self.num_channels = num_channels
        self.feature_maps = feature_maps
        
        # Generator architecture optimized for WGAN
        self.model = nn.Sequential(
            # Project noise to feature maps
            nn.Linear(noise_dim, feature_maps * 8 * 4 * 4),
            nn.BatchNorm1d(feature_maps * 8 * 4 * 4),
            nn.ReLU(inplace=True),
            
            # Reshape to spatial dimensions
            nn.Unflatten(1, (feature_maps * 8, 4, 4)),
            
            # Progressive upsampling
            # 4x4 -> 8x8
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(inplace=True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(feature_maps * 2, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        # Initialize weights for WGAN
        self._initialize_weights()
        
        print(f"  WGAN Generator: {noise_dim}D -> 32x32x{num_channels}")
        print(f"    Architecture: DCGAN-style optimized for Wasserstein loss")
    
    def _initialize_weights(self):
        """Initialize weights for stable WGAN training"""
        for module in self.modules():
            if isinstance(module, (nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(module.weight, 0.0, 0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.normal_(module.weight, 1.0, 0.02)
                nn.init.zeros_(module.bias)
    
    def forward(self, noise):
        """Generate images using WGAN-optimized architecture"""
        return self.model(noise)

# ============================================================================
# WGAN CRITIC (NOT DISCRIMINATOR)
# ============================================================================

class WGAN_Critic(nn.Module):
    """
    WGAN Critic - Approximates Wasserstein distance (NOT a discriminator)
    
    Key Differences from Standard Discriminator:
    - Outputs real values (not probabilities)
    - No sigmoid activation
    - Enforces Lipschitz constraint via weight clipping
    - Maximizes W(P_r, P_g) = E[f(x)] - E[f(G(z))]
    """
    
    def __init__(self, num_channels=3, feature_maps=64):
        super(WGAN_Critic, self).__init__()
        
        self.num_channels = num_channels
        self.feature_maps = feature_maps
        
        # Critic architecture (no batch normalization for stability)
        self.model = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(num_channels, feature_maps, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4x4 -> 1
            nn.Conv2d(feature_maps * 4, 1, 4, 1, 0),
            # No sigmoid! Output real values
        )
        
        # Initialize weights for WGAN
        self._initialize_weights()
        
        print(f"  WGAN Critic: 32x32x{num_channels} -> real value")
        print(f"    Key feature: Outputs real values, not probabilities")
        print(f"    Constraint: 1-Lipschitz via weight clipping")
    
    def _initialize_weights(self):
        """Initialize weights for stable WGAN training"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0.0, 0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, img):
        """
        WGAN Critic Forward Pass
        
        Returns real values (not probabilities)
        Higher values indicate more "real-like"
        """
        output = self.model(img)
        return output.view(-1)  # Flatten to (batch_size,)
    
    def clip_weights(self, clip_value=0.01):
        """
        Enforce Lipschitz constraint via weight clipping
        
        This is the original WGAN method (later improved by gradient penalty)
        Clips all weights to [-c, c] to maintain 1-Lipschitz property
        """
        for param in self.parameters():
            param.data.clamp_(-clip_value, clip_value)

# ============================================================================
# WGAN ARCHITECTURE
# ============================================================================

class WassersteinGAN_Theory(nn.Module):
    """
    Wasserstein GAN - Theoretical Foundation
    
    Revolutionary Innovations:
    - Wasserstein distance instead of Jensen-Shannon divergence
    - Kantorovich-Rubinstein dual formulation
    - 1-Lipschitz critic constraint
    - Meaningful loss that correlates with quality
    - Stable training with theoretical guarantees
    - No mode collapse issues
    """
    
    def __init__(self, noise_dim=100, num_channels=3, feature_maps=64):
        super(WassersteinGAN_Theory, self).__init__()
        
        self.noise_dim = noise_dim
        self.num_channels = num_channels
        
        print(f"Building Wasserstein GAN Theoretical Foundation...")
        
        # Generator and Critic (not discriminator!)
        self.generator = WGAN_Generator(noise_dim, num_channels, feature_maps)
        self.critic = WGAN_Critic(num_channels, feature_maps)
        
        # Calculate statistics
        gen_params = sum(p.numel() for p in self.generator.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        total_params = gen_params + critic_params
        
        print(f"WGAN Architecture Summary:")
        print(f"  Noise dimension: {noise_dim}")
        print(f"  Image size: 32x32x{num_channels}")
        print(f"  Generator parameters: {gen_params:,}")
        print(f"  Critic parameters: {critic_params:,}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Key innovation: Wasserstein distance with theoretical guarantees")
    
    def generate_noise(self, batch_size, device):
        """Generate random noise for WGAN generator"""
        return torch.randn(batch_size, self.noise_dim, device=device)
    
    def generate_samples(self, num_samples, device):
        """Generate samples using WGAN generator"""
        self.generator.eval()
        
        with torch.no_grad():
            noise = self.generate_noise(num_samples, device)
            samples = self.generator(noise)
        
        return samples
    
    def get_theoretical_analysis(self):
        """Analyze WGAN theoretical foundations"""
        return {
            'wasserstein_theory': WGAN_THEORY,
            'advantages_over_standard_gan': [
                'Meaningful loss function that correlates with quality',
                'No vanishing gradient problem',
                'Stable training without careful balancing',
                'No mode collapse issues',
                'Theoretical convergence guarantees'
            ],
            'lipschitz_constraint': [
                '1-Lipschitz constraint on critic function',
                'Enforced by weight clipping in original WGAN',
                'Ensures critic approximates Wasserstein distance',
                'Critical for theoretical guarantees'
            ],
            'training_dynamics': [
                'Critic trained to optimality before generator update',
                'Multiple critic updates per generator update',
                'RMSprop optimizer for stability',
                'Lower learning rates for stable convergence'
            ]
        }

# ============================================================================
# WGAN LOSS FUNCTIONS
# ============================================================================

def wasserstein_distance_loss(real_output, fake_output):
    """
    Wasserstein Distance Loss
    
    WGAN Objective: W(P_r, P_g) = E[f(x)] - E[f(G(z))]
    
    Critic Loss: -(E[f(x)] - E[f(G(z))]) = -E[f(x)] + E[f(G(z))]
    Generator Loss: -E[f(G(z))]
    
    Note: We minimize negative Wasserstein distance
    """
    # Critic wants to maximize: E[f(x)] - E[f(G(z))]
    # So we minimize: -E[f(x)] + E[f(G(z))]
    critic_loss = -torch.mean(real_output) + torch.mean(fake_output)
    
    # Generator wants to minimize: -E[f(G(z))]
    generator_loss = -torch.mean(fake_output)
    
    return critic_loss, generator_loss

def compute_wasserstein_estimate(real_output, fake_output):
    """
    Compute Wasserstein distance estimate
    
    W(P_r, P_g) ≈ E[f(x)] - E[f(G(z))]
    """
    return torch.mean(real_output) - torch.mean(fake_output)

# ============================================================================
# WGAN TRAINING FUNCTION
# ============================================================================

def train_wgan(model, train_loader, epochs=100, learning_rate=5e-5, 
               clip_value=0.01, critic_iters=5):
    """
    Train WGAN with theoretical foundations
    
    Key Training Principles:
    1. Train critic to optimality before generator update
    2. Clip weights to enforce Lipschitz constraint
    3. Use RMSprop optimizer for stability
    4. Multiple critic updates per generator update
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.generator.to(device)
    model.critic.to(device)
    
    # WGAN optimizers (RMSprop recommended in paper)
    optimizer_G = optim.RMSprop(model.generator.parameters(), lr=learning_rate)
    optimizer_C = optim.RMSprop(model.critic.parameters(), lr=learning_rate)
    
    # Training tracking
    wasserstein_distances = []
    generator_losses = []
    critic_losses = []
    
    print(f"Training WGAN on device: {device}")
    print(f"Learning rate: {learning_rate}")
    print(f"Clip value: {clip_value}")
    print(f"Critic iterations per generator update: {critic_iters}")
    
    for epoch in range(epochs):
        epoch_wd = 0.0
        epoch_g_loss = 0.0
        epoch_c_loss = 0.0
        
        for batch_idx, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # ================================================================
            # Train Critic: maximize W(P_r, P_g) = E[f(x)] - E[f(G(z))]
            # ================================================================
            for _ in range(critic_iters):
                optimizer_C.zero_grad()
                
                # Real images
                real_output = model.critic(real_images)
                
                # Fake images
                noise = model.generate_noise(batch_size, device)
                fake_images = model.generator(noise).detach()  # Detach to not train G
                fake_output = model.critic(fake_images)
                
                # WGAN critic loss
                critic_loss, _ = wasserstein_distance_loss(real_output, fake_output)
                critic_loss.backward()
                optimizer_C.step()
                
                # Clip weights to enforce Lipschitz constraint
                model.critic.clip_weights(clip_value)
            
            # ================================================================
            # Train Generator: minimize -E[f(G(z))]
            # ================================================================
            optimizer_G.zero_grad()
            
            # Generate fake images (without detaching)
            noise = model.generate_noise(batch_size, device)
            fake_images = model.generator(noise)
            fake_output = model.critic(fake_images)
            
            # WGAN generator loss
            _, generator_loss = wasserstein_distance_loss(None, fake_output)
            generator_loss.backward()
            optimizer_G.step()
            
            # Compute Wasserstein distance estimate
            with torch.no_grad():
                real_output = model.critic(real_images)
                fake_output = model.critic(fake_images.detach())
                wd_estimate = compute_wasserstein_estimate(real_output, fake_output)
            
            # Track statistics
            epoch_wd += wd_estimate.item()
            epoch_g_loss += generator_loss.item()
            epoch_c_loss += critic_loss.item()
            
            if batch_idx % 200 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                      f'WD: {wd_estimate.item():.4f}, G_Loss: {generator_loss.item():.4f}, '
                      f'C_Loss: {critic_loss.item():.4f}')
        
        # Calculate epoch averages
        avg_wd = epoch_wd / len(train_loader)
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_c_loss = epoch_c_loss / len(train_loader)
        
        wasserstein_distances.append(avg_wd)
        generator_losses.append(avg_g_loss)
        critic_losses.append(avg_c_loss)
        
        print(f'Epoch {epoch+1}/{epochs}: WD: {avg_wd:.4f}, G_Loss: {avg_g_loss:.4f}, '
              f'C_Loss: {avg_c_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 25 == 0:
            torch.save({
                'generator': model.generator.state_dict(),
                'critic': model.critic.state_dict(),
                'epoch': epoch,
                'wasserstein_distance': avg_wd
            }, f'AI-ML-DL/Models/Generative_AI/wgan_epoch_{epoch+1}.pth')
        
        # Early stopping based on convergence
        if abs(avg_wd) < 0.1:  # Wasserstein distance close to 0
            print(f"Good convergence reached at epoch {epoch+1}")
            break
    
    return wasserstein_distances, generator_losses, critic_losses

# ============================================================================
# VISUALIZATION AND ANALYSIS
# ============================================================================

def visualize_wgan_theory():
    """Visualize WGAN theoretical foundations"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Wasserstein vs JS divergence
    ax = axes[0, 0]
    ax.set_title('Wasserstein vs Jensen-Shannon Divergence', fontsize=14, fontweight='bold')
    
    # Simulate different overlap scenarios
    x = np.linspace(-3, 7, 1000)
    
    # Overlapping distributions
    p1 = 0.5 * (np.exp(-(x-1)**2/0.5) / np.sqrt(2*np.pi*0.5))
    p2 = 0.5 * (np.exp(-(x-2)**2/0.5) / np.sqrt(2*np.pi*0.5))
    
    # Non-overlapping distributions
    p3 = 0.5 * (np.exp(-(x+1)**2/0.5) / np.sqrt(2*np.pi*0.5))
    p4 = 0.5 * (np.exp(-(x-5)**2/0.5) / np.sqrt(2*np.pi*0.5))
    
    ax.plot(x, p1, 'b-', label='P_r (overlapping)', linewidth=2)
    ax.plot(x, p2, 'r-', label='P_g (overlapping)', linewidth=2)
    ax.plot(x, p3, 'b--', label='P_r (separated)', linewidth=2, alpha=0.7)
    ax.plot(x, p4, 'r--', label='P_g (separated)', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text annotations
    ax.text(1.5, 0.15, 'JS: Finite\nWass: Small', ha='center', va='center', 
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    ax.text(2, 0.05, 'JS: log(2)\nWass: Distance', ha='center', va='center',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
    
    # Kantorovich-Rubinstein dual
    ax = axes[0, 1]
    ax.set_title('Kantorovich-Rubinstein Dual Formulation', fontsize=14)
    
    # Show the dual formulation concept
    ax.text(0.5, 0.8, 'Primal Problem:', ha='center', va='center', 
           fontsize=14, fontweight='bold')
    ax.text(0.5, 0.7, r'$W(P_r, P_g) = \inf_{\gamma \in \Pi(P_r, P_g)} \mathbb{E}_{(x,y) \sim \gamma}[||x-y||]$',
           ha='center', va='center', fontsize=12)
    
    ax.text(0.5, 0.5, '↕ Kantorovich-Rubinstein', ha='center', va='center',
           fontsize=12, fontweight='bold', color='darkblue')
    
    ax.text(0.5, 0.3, 'Dual Problem:', ha='center', va='center', 
           fontsize=14, fontweight='bold')
    ax.text(0.5, 0.2, r'$W(P_r, P_g) = \sup_{||f||_L \leq 1} \mathbb{E}_{x \sim P_r}[f(x)] - \mathbb{E}_{x \sim P_g}[f(x)]$',
           ha='center', va='center', fontsize=12)
    
    ax.text(0.5, 0.05, '1-Lipschitz Constraint: |f(x) - f(y)| ≤ ||x - y||',
           ha='center', va='center', fontsize=11, color='darkred',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Training dynamics comparison
    ax = axes[1, 0]
    ax.set_title('Training Dynamics: Standard GAN vs WGAN', fontsize=14)
    
    epochs = np.arange(1, 51)
    
    # Standard GAN (oscillatory)
    standard_loss = 2.0 + 0.8 * np.sin(epochs/3) + 0.5 * np.random.randn(50) * 0.3
    standard_loss[standard_loss < 0] = 0.1  # Ensure positive
    
    # WGAN (smooth convergence)
    wgan_loss = 3.0 * np.exp(-epochs/15) + 0.2 * np.sin(epochs/10) + 0.1
    
    ax.plot(epochs, standard_loss, 'r--', label='Standard GAN Loss', 
           linewidth=2, alpha=0.8)
    ax.plot(epochs, wgan_loss, 'b-', label='WGAN Wasserstein Distance', 
           linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss / Distance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Highlight WGAN advantages
    ax.text(35, 2.5, 'WGAN:\n✓ Smooth convergence\n✓ Meaningful loss\n✓ No oscillations', 
           ha='left', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    # Weight clipping visualization
    ax = axes[1, 1]
    ax.set_title('Weight Clipping for Lipschitz Constraint', fontsize=14)
    
    # Show weight distribution before and after clipping
    weights_before = np.random.normal(0, 0.5, 1000)
    weights_after = np.clip(weights_before, -0.01, 0.01)
    
    ax.hist(weights_before, bins=50, alpha=0.7, label='Before Clipping', 
           color='red', density=True)
    ax.hist(weights_after, bins=50, alpha=0.7, label='After Clipping', 
           color='blue', density=True)
    
    ax.axvline(-0.01, color='black', linestyle='--', linewidth=2, label='Clip Boundaries')
    ax.axvline(0.01, color='black', linestyle='--', linewidth=2)
    
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.text(0, 30, 'Enforces\n1-Lipschitz\nConstraint', ha='center', va='center',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/Generative_AI/006_wgan_theory.png', dpi=300, bbox_inches='tight')
    print("WGAN theory visualization saved: 006_wgan_theory.png")

def visualize_wgan_advantages():
    """Visualize WGAN advantages over standard GANs"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Loss correlation with quality
    ax = axes[0, 0]
    ax.set_title('Loss Correlation with Sample Quality', fontsize=14, fontweight='bold')
    
    # Simulate quality vs loss relationship
    epochs = np.arange(1, 21)
    
    # Standard GAN (no clear correlation)
    standard_quality = 5 + 2 * np.sin(epochs/3) + np.random.randn(20) * 0.5
    standard_loss = 1.5 + 0.8 * np.cos(epochs/2) + np.random.randn(20) * 0.3
    
    # WGAN (clear correlation)
    wgan_quality = 2 + 6 * (1 - np.exp(-epochs/8))
    wgan_loss = 4 * np.exp(-epochs/8) + 0.2
    
    ax2 = ax.twinx()
    
    line1 = ax.plot(epochs, standard_quality, 'r--', label='Standard GAN Quality', linewidth=2)
    line2 = ax2.plot(epochs, standard_loss, 'r:', label='Standard GAN Loss', linewidth=2)
    line3 = ax.plot(epochs, wgan_quality, 'b-', label='WGAN Quality', linewidth=2)
    line4 = ax2.plot(epochs, wgan_loss, 'b:', label='WGAN Loss', linewidth=2)
    
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Sample Quality', color='blue')
    ax2.set_ylabel('Loss Value', color='red')
    
    # Combine legends
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right')
    
    ax.grid(True, alpha=0.3)
    
    # Gradient quality comparison
    ax = axes[0, 1]
    ax.set_title('Gradient Quality Throughout Training', fontsize=14)
    
    # Distance from optimum
    distance = np.linspace(0, 2, 100)
    
    # Standard GAN gradients (vanish near optimum)
    standard_grad = np.maximum(0.1, 1 - distance**2) * np.exp(-distance*3)
    
    # WGAN gradients (remain useful)
    wgan_grad = np.maximum(0.2, 1 - distance)
    
    ax.plot(distance, standard_grad, 'r--', label='Standard GAN', linewidth=2)
    ax.plot(distance, wgan_grad, 'b-', label='WGAN', linewidth=2)
    
    ax.set_xlabel('Distance from Optimum')
    ax.set_ylabel('Gradient Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Highlight vanishing gradient problem
    ax.fill_between(distance[80:], 0, standard_grad[80:], alpha=0.3, color='red', 
                   label='Vanishing Gradients')
    ax.text(1.5, 0.7, 'WGAN provides\nuseful gradients\neverywhere!', 
           ha='center', va='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    # Mode collapse comparison
    ax = axes[1, 0]
    ax.set_title('Mode Collapse Resistance', fontsize=14)
    
    # Simulate mode coverage over training
    epochs = np.arange(1, 31)
    
    # Standard GAN (mode collapse)
    standard_modes = 8 * np.exp(-epochs/10) + 1 + 0.5 * np.sin(epochs/5)
    standard_modes = np.maximum(1, standard_modes)
    
    # WGAN (maintains diversity)
    wgan_modes = 8 - 2 * np.exp(-epochs/15) + 0.2 * np.sin(epochs/8)
    wgan_modes = np.maximum(6, wgan_modes)
    
    ax.plot(epochs, standard_modes, 'r--', label='Standard GAN', linewidth=2, marker='o')
    ax.plot(epochs, wgan_modes, 'b-', label='WGAN', linewidth=2, marker='s')
    
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Number of Modes Covered')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Highlight mode collapse
    ax.fill_between(epochs[15:], 0, standard_modes[15:], alpha=0.3, color='red')
    ax.text(20, 5, 'Mode\nCollapse!', ha='center', va='center', fontsize=12, 
           fontweight='bold', color='darkred')
    ax.text(20, 7, 'WGAN maintains\ndiversity!', ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    # Theoretical advantages summary
    ax = axes[1, 1]
    ax.set_title('WGAN Theoretical Advantages', fontsize=14)
    
    advantages = ['Meaningful\nLoss', 'No Vanishing\nGradients', 'Mode Coverage\nGuarantee', 'Training\nStability']
    standard_scores = [3, 2, 3, 4]
    wgan_scores = [9, 9, 8, 8]
    
    x = np.arange(len(advantages))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, standard_scores, width, label='Standard GAN', color='lightcoral')
    bars2 = ax.bar(x + width/2, wgan_scores, width, label='WGAN', color='lightgreen')
    
    ax.set_ylabel('Score (1-10)')
    ax.set_xticks(x)
    ax.set_xticklabels(advantages)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add improvement arrows
    for i, (std, wgan) in enumerate(zip(standard_scores, wgan_scores)):
        improvement = ((wgan - std) / std) * 100
        ax.annotate('', xy=(i + width/2, wgan - 0.2), xytext=(i - width/2, std + 0.2),
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkgreen'))
        ax.text(i, max(std, wgan) + 0.5, f'+{improvement:.0f}%', 
               ha='center', va='bottom', fontweight='bold', color='darkgreen')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/Generative_AI/006_wgan_advantages.png', dpi=300, bbox_inches='tight')
    print("WGAN advantages visualization saved: 006_wgan_advantages.png")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"=== Wasserstein GAN Theoretical Foundation Implementation ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load CIFAR-10 dataset
    train_loader, test_loader, classes = load_cifar10_dataset()
    
    # Initialize WGAN model
    wgan_model = WassersteinGAN_Theory()
    
    # Analyze model properties
    gen_params = sum(p.numel() for p in wgan_model.generator.parameters())
    critic_params = sum(p.numel() for p in wgan_model.critic.parameters())
    total_params = gen_params + critic_params
    theoretical_analysis = wgan_model.get_theoretical_analysis()
    
    print(f"\nWGAN Analysis:")
    print(f"  Generator parameters: {gen_params:,}")
    print(f"  Critic parameters: {critic_params:,}")
    print(f"  Total parameters: {total_params:,}")
    
    print(f"\nTheoretical Foundations:")
    for key, value in theoretical_analysis.items():
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
    print("\nGenerating WGAN theoretical analysis...")
    visualize_wgan_theory()
    visualize_wgan_advantages()
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("WASSERSTEIN GAN THEORETICAL FOUNDATION SUMMARY")
    print("="*70)
    
    print(f"\nHistorical Context ({YEAR}):")
    print(f"  Innovation: {INNOVATION}")
    print(f"  Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"  Impact: {IMPACT}")
    
    print(f"\nWGAN THEORETICAL INNOVATIONS:")
    print("="*50)
    print("1. WASSERSTEIN DISTANCE:")
    print("   • Earth Mover's Distance between distributions")
    print("   • Meaningful even when supports don't overlap")
    print("   • Provides smooth metric on probability space")
    print("   • W(P_r, P_g) = inf_γ E[(x,y)~γ][||x-y||]")
    
    print("\n2. KANTOROVICH-RUBINSTEIN DUALITY:")
    print("   • Transforms intractable primal to tractable dual")
    print("   • W(P_r, P_g) = sup_{||f||_L≤1} E[f(x)] - E[f(G(z))]")
    print("   • Critic approximates optimal transport map")
    print("   • 1-Lipschitz constraint essential for validity")
    
    print("\n3. CRITIC (NOT DISCRIMINATOR):")
    print("   • Outputs real values, not probabilities")
    print("   • No sigmoid activation function")
    print("   • Enforces 1-Lipschitz constraint")
    print("   • Approximates Wasserstein distance")
    
    print("\n4. TRAINING PROCEDURE:")
    print("   • Train critic to optimality before generator")
    print("   • Multiple critic updates per generator update")
    print("   • Weight clipping to enforce Lipschitz constraint")
    print("   • RMSprop optimizer for stability")
    
    print(f"\nTECHNICAL ACHIEVEMENTS:")
    print("="*40)
    print("• First theoretically principled GAN loss function")
    print("• Meaningful loss that correlates with sample quality")
    print("• Eliminated vanishing gradient problem")
    print("• Provided convergence guarantees")
    print("• Solved mode collapse through theoretical foundation")
    
    print(f"\nWASSERSTEIN THEORY:")
    for key, theory in WGAN_THEORY.items():
        print(f"  • {theory}")
    
    print(f"\nADVANTAGES OVER STANDARD GAN:")
    for advantage in theoretical_analysis['advantages_over_standard_gan']:
        print(f"  • {advantage}")
    
    print(f"\nLIPSCHITZ CONSTRAINT:")
    for constraint in theoretical_analysis['lipschitz_constraint']:
        print(f"  • {constraint}")
    
    print(f"\nTRAINING DYNAMICS:")
    for dynamic in theoretical_analysis['training_dynamics']:
        print(f"  • {dynamic}")
    
    print(f"\nMATHEMATICAL FOUNDATION:")
    print("="*40)
    print("• Wasserstein-1 Distance: W₁(μ,ν) = inf_π ∫||x-y||dπ(x,y)")
    print("• Kantorovich-Rubinstein: W₁(μ,ν) = sup_{||f||_L≤1} ∫f dμ - ∫f dν")
    print("• WGAN Objective: min_G max_{||D||_L≤1} E[D(x)] - E[D(G(z))]")
    print("• Critic Loss: L_C = -E[f(x)] + E[f(G(z))]")
    print("• Generator Loss: L_G = -E[f(G(z))]")
    print("• Lipschitz Constraint: |f(x) - f(y)| ≤ ||x - y||")
    
    print(f"\nTHEORETICAL GUARANTEES:")
    print("="*40)
    print("• Convergence: WGAN converges to Nash equilibrium")
    print("• Meaningful Loss: Correlates with sample quality")
    print("• Gradient Quality: Useful gradients everywhere")
    print("• Mode Coverage: Theoretical guarantee against collapse")
    print("• Stability: Robust to hyperparameter choices")
    
    print(f"\nWGAN vs STANDARD GAN:")
    print("="*40)
    print("• Standard: Jensen-Shannon divergence, mode collapse prone")
    print("• WGAN: Wasserstein distance, mode coverage guarantee")
    print("• Standard: Vanishing gradients at optimum")
    print("• WGAN: Useful gradients throughout training")
    print("• Standard: Loss doesn't correlate with quality")
    print("• WGAN: Loss directly measures sample quality")
    
    print(f"\nHISTORICAL IMPACT:")
    print("="*40)
    print("• Provided first principled theoretical foundation for GANs")
    print("• Inspired distance-based approaches in generative modeling")
    print("• Led to WGAN-GP and improved Lipschitz constraints")
    print("• Influenced optimal transport in machine learning")
    print("• Set standard for meaningful loss functions")
    print("• Established theoretical rigor in generative AI")
    
    print(f"\nLIMITATIONS AND IMPROVEMENTS:")
    print("="*40)
    print("• Weight clipping is crude way to enforce Lipschitz constraint")
    print("• Can lead to capacity underutilization")
    print("• Gradient penalty (WGAN-GP) provides better constraint")
    print("• → WGAN-GP: Replaces clipping with gradient penalty")
    print("• → Spectral normalization: More sophisticated constraint")
    print("• → Progressive growing: Higher resolution generation")
    
    print(f"\nERA 2 FOUNDATION COMPLETE:")
    print("="*40)
    print("• DCGAN: Architectural guidelines for stable training")
    print("• Improved Training: Techniques for mode collapse prevention")
    print("• WGAN: Theoretical foundation with guarantees")
    print("• → Established GAN training as mature technology")
    print("• → Set stage for high-resolution generation")
    print("• → Provided theoretical understanding for future work")
    
    return {
        'model': 'Wasserstein GAN Theory',
        'year': YEAR,
        'innovation': INNOVATION,
        'generator_params': gen_params,
        'critic_params': critic_params,
        'total_params': total_params,
        'theoretical_analysis': theoretical_analysis
    }

if __name__ == "__main__":
    results = main()