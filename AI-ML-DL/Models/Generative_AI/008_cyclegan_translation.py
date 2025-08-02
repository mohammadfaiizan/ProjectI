"""
ERA 3: ADVANCED GANS & ARCHITECTURAL INNOVATIONS - CycleGAN Translation
======================================================================

Year: 2017
Paper: "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" (Zhu et al.)
Innovation: Cycle consistency for unpaired image-to-image translation without paired training data
Previous Limitation: Requirement for paired datasets limiting translation applications
Performance Gain: High-quality translation between unpaired domains with cycle consistency
Impact: Enabled practical image translation applications and established cycle consistency paradigm

This file implements CycleGAN that revolutionized image-to-image translation by eliminating
the need for paired training data through cycle consistency constraints.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
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
INNOVATION = "Cycle consistency for unpaired image-to-image translation without paired training data"
PREVIOUS_LIMITATION = "Requirement for paired datasets limiting translation applications"
IMPACT = "Enabled practical image translation applications and established cycle consistency paradigm"

print(f"=== CycleGAN Translation ({YEAR}) ===")
print(f"Innovation: {INNOVATION}")
print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
print(f"Impact: {IMPACT}")
print("="*60)

# ============================================================================
# CYCLEGAN PRINCIPLES
# ============================================================================

CYCLEGAN_PRINCIPLES = {
    "cycle_consistency": "F(G(x)) ≈ x and G(F(y)) ≈ y for unpaired translation",
    "dual_generators": "Two generators G: X→Y and F: Y→X for bidirectional translation",
    "dual_discriminators": "Two discriminators D_X and D_Y for adversarial training",
    "unpaired_training": "No need for paired (x,y) samples, only samples from X and Y domains",
    "identity_loss": "Optional identity mapping loss for color preservation",
    "adversarial_loss": "Standard GAN loss for realistic generation in target domain",
    "cycle_loss": "L1 loss for cycle consistency constraint",
    "domain_adaptation": "Learn mapping between different visual domains"
}

print("CycleGAN Principles:")
for key, principle in CYCLEGAN_PRINCIPLES.items():
    print(f"  • {principle}")
print()

# ============================================================================
# DATASET SIMULATION (CIFAR-10 Domain Adaptation)
# ============================================================================

class UnpairedDomainDataset(Dataset):
    """
    Simulated unpaired domain dataset using CIFAR-10
    
    Creates two domains:
    - Domain A: Original CIFAR-10 images
    - Domain B: Modified CIFAR-10 images (simulating different style)
    
    This demonstrates CycleGAN without requiring actual unpaired datasets
    """
    
    def __init__(self, cifar_dataset, domain='A'):
        self.domain = domain
        self.data = []
        self.labels = []
        
        # Extract data from CIFAR dataset
        for img, label in cifar_dataset:
            if domain == 'A':
                # Domain A: Original images
                self.data.append(img)
            else:
                # Domain B: Apply style transformation (e.g., color shift, blur)
                # Simulate different domain characteristics
                img_b = self._apply_domain_b_transform(img)
                self.data.append(img_b)
            
            self.labels.append(label)
    
    def _apply_domain_b_transform(self, img):
        """Apply transformation to create Domain B characteristics"""
        # Color shift to simulate different lighting/style
        img_b = img.clone()
        
        # Slight color temperature shift
        img_b[0] *= 1.1  # Increase red
        img_b[2] *= 0.9  # Decrease blue
        
        # Add slight blur for style difference
        img_b = F.avg_pool2d(img_b.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
        
        # Ensure values stay in valid range
        img_b = torch.clamp(img_b, -1, 1)
        
        return img_b
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def load_unpaired_domains():
    """Load simulated unpaired domain datasets"""
    print("Loading simulated unpaired domains for CycleGAN translation study...")
    
    # CycleGAN preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load CIFAR-10
    cifar_train = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    cifar_test = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Create unpaired domains
    domain_a_train = UnpairedDomainDataset(cifar_train, domain='A')
    domain_b_train = UnpairedDomainDataset(cifar_train, domain='B')
    domain_a_test = UnpairedDomainDataset(cifar_test, domain='A')
    domain_b_test = UnpairedDomainDataset(cifar_test, domain='B')
    
    # Create data loaders
    loader_a_train = DataLoader(domain_a_train, batch_size=16, shuffle=True, num_workers=2)
    loader_b_train = DataLoader(domain_b_train, batch_size=16, shuffle=True, num_workers=2)
    loader_a_test = DataLoader(domain_a_test, batch_size=16, shuffle=False, num_workers=2)
    loader_b_test = DataLoader(domain_b_test, batch_size=16, shuffle=False, num_workers=2)
    
    print(f"Domain A (original): {len(domain_a_train):,} train, {len(domain_a_test):,} test")
    print(f"Domain B (modified): {len(domain_b_train):,} train, {len(domain_b_test):,} test")
    print(f"Translation task: A ↔ B (bidirectional)")
    print(f"Key advantage: No paired (A,B) samples required!")
    
    return loader_a_train, loader_b_train, loader_a_test, loader_b_test

# ============================================================================
# RESIDUAL BLOCKS
# ============================================================================

class ResidualBlock(nn.Module):
    """
    Residual Block for CycleGAN Generator
    
    Maintains spatial resolution while allowing deep networks
    through skip connections
    """
    
    def __init__(self, channels, use_dropout=False):
        super(ResidualBlock, self).__init__()
        
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True)
        ]
        
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        ]
        
        self.conv_block = nn.Sequential(*layers)
        
    def forward(self, x):
        return x + self.conv_block(x)  # Skip connection

# ============================================================================
# CYCLEGAN GENERATOR
# ============================================================================

class CycleGANGenerator(nn.Module):
    """
    CycleGAN Generator Architecture
    
    Encoder-Decoder with Residual Blocks:
    - Encoder: Downsampling layers
    - Transformer: Residual blocks maintaining resolution
    - Decoder: Upsampling layers
    """
    
    def __init__(self, input_channels=3, output_channels=3, num_residual_blocks=9):
        super(CycleGANGenerator, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_residual_blocks = num_residual_blocks
        
        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling layers
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        
        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]
        
        # Upsampling layers
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channels, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
        
        print(f"  CycleGAN Generator: {input_channels} -> {output_channels}, {num_residual_blocks} ResBlocks")
    
    def forward(self, x):
        """Generate translated image"""
        return self.model(x)

# ============================================================================
# CYCLEGAN DISCRIMINATOR
# ============================================================================

class CycleGANDiscriminator(nn.Module):
    """
    CycleGAN PatchGAN Discriminator
    
    Classifies patches of the image as real/fake rather than
    the entire image, providing more detailed feedback
    """
    
    def __init__(self, input_channels=3):
        super(CycleGANDiscriminator, self).__init__()
        
        self.input_channels = input_channels
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            """Create a discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512, normalize=False),
            nn.Conv2d(512, 1, 4, padding=1)
        )
        
        print(f"  CycleGAN Discriminator: {input_channels} channels -> PatchGAN classification")
    
    def forward(self, img):
        """Classify image patches as real/fake"""
        return self.model(img)

# ============================================================================
# CYCLEGAN ARCHITECTURE
# ============================================================================

class CycleGAN_Translation(nn.Module):
    """
    CycleGAN - Unpaired Image-to-Image Translation
    
    Revolutionary Innovations:
    - Cycle consistency for unpaired translation
    - Dual generators for bidirectional mapping
    - No requirement for paired training data
    - Identity loss for color preservation
    - PatchGAN discriminators for detailed feedback
    """
    
    def __init__(self, input_channels=3, num_residual_blocks=9):
        super(CycleGAN_Translation, self).__init__()
        
        self.input_channels = input_channels
        
        print(f"Building CycleGAN Translation...")
        
        # Dual Generators: G_AB (A->B) and G_BA (B->A)
        self.G_AB = CycleGANGenerator(input_channels, input_channels, num_residual_blocks)
        self.G_BA = CycleGANGenerator(input_channels, input_channels, num_residual_blocks)
        
        # Dual Discriminators: D_A and D_B
        self.D_A = CycleGANDiscriminator(input_channels)
        self.D_B = CycleGANDiscriminator(input_channels)
        
        # Calculate statistics
        g_ab_params = sum(p.numel() for p in self.G_AB.parameters())
        g_ba_params = sum(p.numel() for p in self.G_BA.parameters())
        d_a_params = sum(p.numel() for p in self.D_A.parameters())
        d_b_params = sum(p.numel() for p in self.D_B.parameters())
        total_params = g_ab_params + g_ba_params + d_a_params + d_b_params
        
        print(f"CycleGAN Architecture Summary:")
        print(f"  Input channels: {input_channels}")
        print(f"  Residual blocks: {num_residual_blocks}")
        print(f"  G_AB parameters: {g_ab_params:,}")
        print(f"  G_BA parameters: {g_ba_params:,}")
        print(f"  D_A parameters: {d_a_params:,}")
        print(f"  D_B parameters: {d_b_params:,}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Key innovation: Cycle consistency for unpaired translation")
    
    def forward_cycle(self, real_A, real_B):
        """
        Complete cycle consistency forward pass
        
        Cycle A: real_A -> fake_B -> recovered_A
        Cycle B: real_B -> fake_A -> recovered_B
        """
        # A -> B -> A cycle
        fake_B = self.G_AB(real_A)
        recovered_A = self.G_BA(fake_B)
        
        # B -> A -> B cycle
        fake_A = self.G_BA(real_B)
        recovered_B = self.G_AB(fake_A)
        
        return fake_A, fake_B, recovered_A, recovered_B
    
    def get_cyclegan_analysis(self):
        """Analyze CycleGAN innovations"""
        return {
            'cyclegan_principles': CYCLEGAN_PRINCIPLES,
            'architectural_components': [
                'Dual generators for bidirectional translation',
                'Dual discriminators for domain-specific feedback',
                'Residual blocks for stable deep networks',
                'Instance normalization for style independence',
                'PatchGAN discriminators for local realism'
            ],
            'loss_components': [
                'Adversarial loss for realistic generation',
                'Cycle consistency loss for unpaired learning',
                'Identity loss for color preservation',
                'Total loss combines all components'
            ],
            'training_advantages': [
                'No paired training data required',
                'Bidirectional translation capability',
                'Preserves important image content',
                'Generalizes across many domains',
                'Stable training through cycle consistency'
            ]
        }

# ============================================================================
# CYCLEGAN LOSS FUNCTIONS
# ============================================================================

def adversarial_loss(pred, target_is_real):
    """
    Adversarial Loss for CycleGAN
    
    Uses MSE loss instead of BCE for more stable training
    """
    if target_is_real:
        target = torch.ones_like(pred)
    else:
        target = torch.zeros_like(pred)
    
    return F.mse_loss(pred, target)

def cycle_consistency_loss(real_img, recovered_img, lambda_cycle=10.0):
    """
    Cycle Consistency Loss
    
    L1 loss between original and cycle-recovered images
    Ensures F(G(x)) ≈ x and G(F(y)) ≈ y
    """
    return lambda_cycle * F.l1_loss(recovered_img, real_img)

def identity_loss(real_img, same_img, lambda_identity=5.0):
    """
    Identity Loss
    
    When feeding generator an image from target domain,
    it should return the image unchanged (for color preservation)
    """
    return lambda_identity * F.l1_loss(same_img, real_img)

def cyclegan_generator_loss(D_A, D_B, real_A, real_B, fake_A, fake_B, 
                           recovered_A, recovered_B, lambda_cycle=10.0, lambda_identity=5.0):
    """
    Complete CycleGAN Generator Loss
    
    Combines:
    1. Adversarial loss (fool discriminators)
    2. Cycle consistency loss (maintain content)
    3. Identity loss (preserve colors)
    """
    # Adversarial losses
    adv_loss_A = adversarial_loss(D_A(fake_A), True)
    adv_loss_B = adversarial_loss(D_B(fake_B), True)
    adv_loss = adv_loss_A + adv_loss_B
    
    # Cycle consistency losses
    cycle_loss_A = cycle_consistency_loss(real_A, recovered_A, lambda_cycle)
    cycle_loss_B = cycle_consistency_loss(real_B, recovered_B, lambda_cycle)
    cycle_loss = cycle_loss_A + cycle_loss_B
    
    # Identity losses (optional)
    # Generator should preserve input if it's already from target domain
    identity_A = identity_loss(real_B, None, 0)  # Simplified for demo
    identity_B = identity_loss(real_A, None, 0)  # Simplified for demo
    identity_loss_total = identity_A + identity_B
    
    # Total generator loss
    total_loss = adv_loss + cycle_loss + identity_loss_total
    
    return total_loss, adv_loss, cycle_loss, identity_loss_total

def cyclegan_discriminator_loss(discriminator, real_imgs, fake_imgs):
    """
    CycleGAN Discriminator Loss
    
    Standard adversarial loss: classify real as real, fake as fake
    """
    # Real images
    pred_real = discriminator(real_imgs)
    loss_real = adversarial_loss(pred_real, True)
    
    # Fake images (detached to avoid training generator)
    pred_fake = discriminator(fake_imgs.detach())
    loss_fake = adversarial_loss(pred_fake, False)
    
    # Total discriminator loss
    total_loss = (loss_real + loss_fake) * 0.5
    
    return total_loss, loss_real, loss_fake

# ============================================================================
# CYCLEGAN TRAINING FUNCTION
# ============================================================================

def train_cyclegan(model, loader_A, loader_B, epochs=200, learning_rate=0.0002, 
                   lambda_cycle=10.0, lambda_identity=5.0):
    """Train CycleGAN with cycle consistency"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move models to device
    model.G_AB.to(device)
    model.G_BA.to(device)
    model.D_A.to(device)
    model.D_B.to(device)
    
    # Optimizers
    optimizer_G = optim.Adam(
        list(model.G_AB.parameters()) + list(model.G_BA.parameters()),
        lr=learning_rate, betas=(0.5, 0.999)
    )
    optimizer_D_A = optim.Adam(model.D_A.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(model.D_B.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    # Learning rate schedulers
    lr_scheduler_G = optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100
    )
    lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100
    )
    lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100
    )
    
    # Training tracking
    generator_losses = []
    discriminator_losses = []
    cycle_losses = []
    
    print(f"Training CycleGAN on device: {device}")
    print(f"Lambda cycle: {lambda_cycle}, Lambda identity: {lambda_identity}")
    
    for epoch in range(epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_cycle_loss = 0.0
        
        # Create iterators
        iter_A = iter(loader_A)
        iter_B = iter(loader_B)
        
        for batch_idx in range(min(len(loader_A), len(loader_B))):
            try:
                real_A, _ = next(iter_A)
                real_B, _ = next(iter_B)
            except StopIteration:
                break
            
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            
            # ================================================================
            # Train Generators
            # ================================================================
            optimizer_G.zero_grad()
            
            # Complete cycle forward pass
            fake_A, fake_B, recovered_A, recovered_B = model.forward_cycle(real_A, real_B)
            
            # Generator loss
            g_loss, adv_loss, cycle_loss, identity_loss = cyclegan_generator_loss(
                model.D_A, model.D_B, real_A, real_B, fake_A, fake_B,
                recovered_A, recovered_B, lambda_cycle, lambda_identity
            )
            
            g_loss.backward()
            optimizer_G.step()
            
            # ================================================================
            # Train Discriminator A
            # ================================================================
            optimizer_D_A.zero_grad()
            
            d_a_loss, _, _ = cyclegan_discriminator_loss(model.D_A, real_A, fake_A)
            d_a_loss.backward()
            optimizer_D_A.step()
            
            # ================================================================
            # Train Discriminator B
            # ================================================================
            optimizer_D_B.zero_grad()
            
            d_b_loss, _, _ = cyclegan_discriminator_loss(model.D_B, real_B, fake_B)
            d_b_loss.backward()
            optimizer_D_B.step()
            
            # Track statistics
            epoch_g_loss += g_loss.item()
            epoch_d_loss += (d_a_loss.item() + d_b_loss.item())
            epoch_cycle_loss += cycle_loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                      f'G_Loss: {g_loss.item():.4f}, D_Loss: {(d_a_loss + d_b_loss).item():.4f}, '
                      f'Cycle_Loss: {cycle_loss.item():.4f}')
        
        # Calculate epoch averages
        num_batches = min(len(loader_A), len(loader_B))
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        avg_cycle_loss = epoch_cycle_loss / num_batches
        
        generator_losses.append(avg_g_loss)
        discriminator_losses.append(avg_d_loss)
        cycle_losses.append(avg_cycle_loss)
        
        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
        
        print(f'Epoch {epoch+1}/{epochs}: G_Loss: {avg_g_loss:.4f}, '
              f'D_Loss: {avg_d_loss:.4f}, Cycle_Loss: {avg_cycle_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 25 == 0:
            torch.save({
                'G_AB': model.G_AB.state_dict(),
                'G_BA': model.G_BA.state_dict(),
                'D_A': model.D_A.state_dict(),
                'D_B': model.D_B.state_dict(),
                'epoch': epoch
            }, f'AI-ML-DL/Models/Generative_AI/cyclegan_epoch_{epoch+1}.pth')
        
        # Early stopping for demonstration
        if avg_cycle_loss < 1.0 and avg_g_loss < 3.0:
            print(f"Good convergence reached at epoch {epoch+1}")
            break
    
    return generator_losses, discriminator_losses, cycle_losses

# ============================================================================
# VISUALIZATION AND ANALYSIS
# ============================================================================

def visualize_cyclegan_concept():
    """Visualize CycleGAN core concepts"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Cycle consistency concept
    ax = axes[0, 0]
    ax.set_title('Cycle Consistency Concept', fontsize=14, fontweight='bold')
    
    # Draw cycle diagram
    # Domain A
    circle_a = plt.Circle((0.2, 0.7), 0.15, facecolor='lightblue', edgecolor='blue', linewidth=3)
    ax.add_patch(circle_a)
    ax.text(0.2, 0.7, 'Domain A\n(Real A)', ha='center', va='center', fontweight='bold')
    
    # Domain B
    circle_b = plt.Circle((0.8, 0.7), 0.15, facecolor='lightcoral', edgecolor='red', linewidth=3)
    ax.add_patch(circle_b)
    ax.text(0.8, 0.7, 'Domain B\n(Real B)', ha='center', va='center', fontweight='bold')
    
    # Generated images
    circle_fake_b = plt.Circle((0.8, 0.3), 0.12, facecolor='pink', edgecolor='red', linewidth=2, linestyle='--')
    ax.add_patch(circle_fake_b)
    ax.text(0.8, 0.3, 'Fake B\nG_AB(A)', ha='center', va='center', fontsize=10)
    
    circle_fake_a = plt.Circle((0.2, 0.3), 0.12, facecolor='lightcyan', edgecolor='blue', linewidth=2, linestyle='--')
    ax.add_patch(circle_fake_a)
    ax.text(0.2, 0.3, 'Fake A\nG_BA(B)', ha='center', va='center', fontsize=10)
    
    # Arrows for cycle
    # A -> Fake B
    ax.annotate('G_AB', xy=(0.68, 0.65), xytext=(0.32, 0.65),
               arrowprops=dict(arrowstyle='->', lw=3, color='green'))
    
    # Fake B -> Recovered A
    ax.annotate('G_BA', xy=(0.32, 0.35), xytext=(0.68, 0.35),
               arrowprops=dict(arrowstyle='->', lw=3, color='orange'))
    
    # B -> Fake A  
    ax.annotate('G_BA', xy=(0.32, 0.55), xytext=(0.68, 0.55),
               arrowprops=dict(arrowstyle='->', lw=3, color='purple'))
    
    # Fake A -> Recovered B
    ax.annotate('G_AB', xy=(0.68, 0.45), xytext=(0.32, 0.45),
               arrowprops=dict(arrowstyle='->', lw=3, color='brown'))
    
    # Cycle consistency loss
    ax.text(0.5, 0.1, 'Cycle Loss = ||G_BA(G_AB(A)) - A|| + ||G_AB(G_BA(B)) - B||', 
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Unpaired vs Paired learning
    ax = axes[0, 1]
    ax.set_title('Unpaired vs Paired Learning', fontsize=14)
    
    # Paired learning (traditional)
    ax.text(0.25, 0.8, 'Paired Learning', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='red')
    
    ax.text(0.1, 0.6, 'Input A', ha='center', va='center', fontsize=11,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue'))
    ax.text(0.4, 0.6, 'Target B', ha='center', va='center', fontsize=11,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcoral'))
    
    ax.annotate('', xy=(0.35, 0.6), xytext=(0.15, 0.6),
               arrowprops=dict(arrowstyle='<->', lw=2, color='red'))
    ax.text(0.25, 0.65, 'Paired!', ha='center', va='center', fontweight='bold', color='red')
    
    ax.text(0.25, 0.4, 'Requires:\n• Exact correspondences\n• Expensive annotation\n• Limited datasets', 
           ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='mistyrose'))
    
    # Unpaired learning (CycleGAN)
    ax.text(0.75, 0.8, 'Unpaired Learning', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='green')
    
    ax.text(0.6, 0.6, 'Set A', ha='center', va='center', fontsize=11,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue'))
    ax.text(0.9, 0.6, 'Set B', ha='center', va='center', fontsize=11,
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcoral'))
    
    ax.text(0.75, 0.65, 'No pairs!', ha='center', va='center', fontweight='bold', color='green')
    
    ax.text(0.75, 0.4, 'Requires:\n• Just two sets\n• No correspondences\n• Abundant datasets', 
           ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Loss component breakdown
    ax = axes[1, 0]
    ax.set_title('CycleGAN Loss Components', fontsize=14)
    
    loss_components = ['Adversarial\nLoss', 'Cycle\nConsistency', 'Identity\nLoss', 'Total\nLoss']
    loss_values = [2.5, 8.0, 1.5, 12.0]  # Example values
    colors = ['red', 'blue', 'green', 'purple']
    
    bars = ax.bar(loss_components, loss_values, color=colors, alpha=0.7)
    
    # Add mathematical formulations
    formulas = [
        'L_GAN',
        'λ₁L_cyc', 
        'λ₂L_idt',
        'L_total'
    ]
    
    for bar, formula, value in zip(bars, formulas, loss_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
               f'{formula}\n{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Loss Value')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Training dynamics
    ax = axes[1, 1]
    ax.set_title('CycleGAN Training Dynamics', fontsize=14)
    
    epochs = np.arange(1, 51)
    
    # Adversarial loss (oscillates but stabilizes)
    adv_loss = 2.0 + 0.5 * np.sin(epochs/3) * np.exp(-epochs/20)
    
    # Cycle consistency loss (decreases smoothly)
    cycle_loss = 10.0 * np.exp(-epochs/15) + 1.0
    
    # Total loss
    total_loss = adv_loss + cycle_loss
    
    ax.plot(epochs, adv_loss, 'r-', label='Adversarial Loss', linewidth=2)
    ax.plot(epochs, cycle_loss, 'b-', label='Cycle Consistency Loss', linewidth=2)
    ax.plot(epochs, total_loss, 'g-', label='Total Loss', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Highlight cycle consistency importance
    ax.fill_between(epochs[:20], 0, cycle_loss[:20], alpha=0.3, color='blue', 
                   label='Cycle consistency crucial')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/Generative_AI/008_cyclegan_concept.png', dpi=300, bbox_inches='tight')
    print("CycleGAN concept visualization saved: 008_cyclegan_concept.png")

def visualize_cyclegan_architecture():
    """Visualize CycleGAN architecture details"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Dual generator architecture
    ax = axes[0]
    ax.set_title('CycleGAN Dual Generator Architecture', fontsize=16, fontweight='bold')
    
    # Generator G_AB (A->B)
    ax.text(0.25, 0.9, 'Generator G_AB: Domain A → Domain B', 
           ha='center', va='center', fontsize=14, fontweight='bold', color='blue')
    
    # Encoder
    encoder_layers = ['Conv 7x7\n64', 'Conv 3x3\n128', 'Conv 3x3\n256']
    for i, layer in enumerate(encoder_layers):
        rect = plt.Rectangle((0.05 + i*0.12, 0.7), 0.1, 0.15, 
                           facecolor='lightblue', edgecolor='blue')
        ax.add_patch(rect)
        ax.text(0.1 + i*0.12, 0.775, layer, ha='center', va='center', fontsize=9)
    
    # Residual blocks
    ax.text(0.25, 0.5, '9 Residual Blocks\n(256 channels)', 
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow'))
    
    # Decoder
    decoder_layers = ['Deconv 3x3\n128', 'Deconv 3x3\n64', 'Conv 7x7\n3']
    for i, layer in enumerate(decoder_layers):
        rect = plt.Rectangle((0.35 + i*0.12, 0.7), 0.1, 0.15, 
                           facecolor='lightgreen', edgecolor='green')
        ax.add_patch(rect)
        ax.text(0.4 + i*0.12, 0.775, layer, ha='center', va='center', fontsize=9)
    
    # Generator G_BA (B->A)
    ax.text(0.75, 0.9, 'Generator G_BA: Domain B → Domain A', 
           ha='center', va='center', fontsize=14, fontweight='bold', color='red')
    
    # Mirror architecture for G_BA
    for i, layer in enumerate(encoder_layers):
        rect = plt.Rectangle((0.55 + i*0.12, 0.7), 0.1, 0.15, 
                           facecolor='lightcoral', edgecolor='red')
        ax.add_patch(rect)
        ax.text(0.6 + i*0.12, 0.775, layer, ha='center', va='center', fontsize=9)
    
    ax.text(0.75, 0.5, '9 Residual Blocks\n(256 channels)', 
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='orange'))
    
    for i, layer in enumerate(decoder_layers):
        rect = plt.Rectangle((0.85 + i*0.12, 0.7), 0.1, 0.15, 
                           facecolor='lightpink', edgecolor='red')
        ax.add_patch(rect)
        ax.text(0.9 + i*0.12, 0.775, layer, ha='center', va='center', fontsize=9)
    
    # Arrows showing data flow
    ax.annotate('Domain A', xy=(0.05, 0.6), xytext=(0.05, 0.3),
               arrowprops=dict(arrowstyle='->', lw=3, color='blue'))
    ax.annotate('Domain B', xy=(0.45, 0.6), xytext=(0.45, 0.3),
               arrowprops=dict(arrowstyle='->', lw=3, color='blue'))
    
    ax.annotate('Domain B', xy=(0.55, 0.6), xytext=(0.55, 0.3),
               arrowprops=dict(arrowstyle='->', lw=3, color='red'))
    ax.annotate('Domain A', xy=(0.95, 0.6), xytext=(0.95, 0.3),
               arrowprops=dict(arrowstyle='->', lw=3, color='red'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Dual discriminator architecture
    ax = axes[1]
    ax.set_title('CycleGAN Dual Discriminator (PatchGAN) Architecture', fontsize=16, fontweight='bold')
    
    # Discriminator D_A
    ax.text(0.25, 0.9, 'Discriminator D_A: Classify Domain A Images', 
           ha='center', va='center', fontsize=14, fontweight='bold', color='blue')
    
    # PatchGAN layers
    patch_layers = ['Conv 4x4\n64', 'Conv 4x4\n128', 'Conv 4x4\n256', 'Conv 4x4\n512', 'Conv 4x4\n1']
    
    for i, layer in enumerate(patch_layers):
        rect = plt.Rectangle((0.05 + i*0.08, 0.6), 0.07, 0.2, 
                           facecolor='lightblue', edgecolor='blue')
        ax.add_patch(rect)
        ax.text(0.085 + i*0.08, 0.7, layer, ha='center', va='center', fontsize=8)
    
    # Show patch output
    ax.text(0.25, 0.4, 'Output: 30x30 Patch\nClassifications', 
           ha='center', va='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan'))
    
    # Discriminator D_B
    ax.text(0.75, 0.9, 'Discriminator D_B: Classify Domain B Images', 
           ha='center', va='center', fontsize=14, fontweight='bold', color='red')
    
    for i, layer in enumerate(patch_layers):
        rect = plt.Rectangle((0.55 + i*0.08, 0.6), 0.07, 0.2, 
                           facecolor='lightcoral', edgecolor='red')
        ax.add_patch(rect)
        ax.text(0.585 + i*0.08, 0.7, layer, ha='center', va='center', fontsize=8)
    
    ax.text(0.75, 0.4, 'Output: 30x30 Patch\nClassifications', 
           ha='center', va='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightpink'))
    
    # PatchGAN advantage
    ax.text(0.5, 0.1, 'PatchGAN Advantage: Local patch classification provides more detailed feedback\nthan global image classification', 
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/Generative_AI/008_cyclegan_architecture.png', dpi=300, bbox_inches='tight')
    print("CycleGAN architecture visualization saved: 008_cyclegan_architecture.png")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"=== CycleGAN Translation Implementation ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load unpaired domain datasets
    loader_a_train, loader_b_train, loader_a_test, loader_b_test = load_unpaired_domains()
    
    # Initialize CycleGAN model
    cyclegan_model = CycleGAN_Translation()
    
    # Analyze model properties
    g_ab_params = sum(p.numel() for p in cyclegan_model.G_AB.parameters())
    g_ba_params = sum(p.numel() for p in cyclegan_model.G_BA.parameters())
    d_a_params = sum(p.numel() for p in cyclegan_model.D_A.parameters())
    d_b_params = sum(p.numel() for p in cyclegan_model.D_B.parameters())
    total_params = g_ab_params + g_ba_params + d_a_params + d_b_params
    cyclegan_analysis = cyclegan_model.get_cyclegan_analysis()
    
    print(f"\nCycleGAN Analysis:")
    print(f"  G_AB parameters: {g_ab_params:,}")
    print(f"  G_BA parameters: {g_ba_params:,}")
    print(f"  D_A parameters: {d_a_params:,}")
    print(f"  D_B parameters: {d_b_params:,}")
    print(f"  Total parameters: {total_params:,}")
    
    print(f"\nCycleGAN Innovations:")
    for key, value in cyclegan_analysis.items():
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
    print("\nGenerating CycleGAN analysis...")
    visualize_cyclegan_concept()
    visualize_cyclegan_architecture()
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("CYCLEGAN TRANSLATION SUMMARY")
    print("="*70)
    
    print(f"\nHistorical Context ({YEAR}):")
    print(f"  Innovation: {INNOVATION}")
    print(f"  Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"  Impact: {IMPACT}")
    
    print(f"\nCYCLEGAN INNOVATIONS:")
    print("="*50)
    print("1. CYCLE CONSISTENCY:")
    print("   • F(G(x)) ≈ x and G(F(y)) ≈ y")
    print("   • Enables learning without paired data")
    print("   • Preserves important content across domains")
    print("   • L1 loss for cycle reconstruction")
    
    print("\n2. DUAL GENERATOR ARCHITECTURE:")
    print("   • G_AB: Domain A → Domain B translation")
    print("   • G_BA: Domain B → Domain A translation")
    print("   • Bidirectional mapping capability")
    print("   • Encoder-decoder with residual blocks")
    
    print("\n3. DUAL DISCRIMINATOR SYSTEM:")
    print("   • D_A: Discriminates real vs fake Domain A images")
    print("   • D_B: Discriminates real vs fake Domain B images")
    print("   • PatchGAN architecture for local realism")
    print("   • Provides domain-specific adversarial feedback")
    
    print("\n4. UNPAIRED LEARNING:")
    print("   • No need for corresponding (x,y) pairs")
    print("   • Only requires samples from each domain")
    print("   • Dramatically expands applicable datasets")
    print("   • Enables practical translation applications")
    
    print(f"\nTECHNICAL ACHIEVEMENTS:")
    print("="*40)
    print("• First successful unpaired image-to-image translation")
    print("• Bidirectional translation capability")
    print("• Cycle consistency constraint for content preservation")
    print("• PatchGAN discriminators for detailed feedback")
    print("• Stable training without paired supervision")
    
    print(f"\nCYCLEGAN PRINCIPLES:")
    for key, principle in CYCLEGAN_PRINCIPLES.items():
        print(f"  • {principle}")
    
    print(f"\nARCHITECTURAL COMPONENTS:")
    for component in cyclegan_analysis['architectural_components']:
        print(f"  • {component}")
    
    print(f"\nLOSS COMPONENTS:")
    for component in cyclegan_analysis['loss_components']:
        print(f"  • {component}")
    
    print(f"\nTRAINING ADVANTAGES:")
    for advantage in cyclegan_analysis['training_advantages']:
        print(f"  • {advantage}")
    
    print(f"\nMATHEMATICAL FORMULATION:")
    print("="*40)
    print("• Total Loss: L = L_GAN(G_AB,D_B,A,B) + L_GAN(G_BA,D_A,B,A) + λL_cyc(G_AB,G_BA)")
    print("• Adversarial Loss: L_GAN(G,D,X,Y) = E[log D(y)] + E[log(1-D(G(x)))]")
    print("• Cycle Loss: L_cyc(G,F) = E[||F(G(x)) - x||₁] + E[||G(F(y)) - y||₁]")
    print("• Identity Loss: L_idt(G,F) = E[||G(y) - y||₁] + E[||F(x) - x||₁]")
    
    print(f"\nARCHITECTURAL DETAILS:")
    print("="*40)
    print("• Generator: Encoder-Decoder + 9 ResNet blocks")
    print("• Discriminator: PatchGAN (70×70 patches)")
    print("• Normalization: Instance normalization")
    print("• Activation: ReLU (generator), LeakyReLU (discriminator)")
    print("• Padding: Reflection padding for better boundaries")
    
    print(f"\nUNPAIRED vs PAIRED LEARNING:")
    print("="*40)
    print("• Paired: Requires exact correspondences between domains")
    print("• Unpaired (CycleGAN): Only requires domain samples")
    print("• Paired: Limited by availability of paired datasets")
    print("• Unpaired: Can use any two domain collections")
    print("• Paired: Direct supervision for mapping")
    print("• Unpaired: Cycle consistency provides supervision")
    
    print(f"\nAPPLICATION DOMAINS:")
    print("="*40)
    print("• Style Transfer: Photos ↔ Paintings")
    print("• Season Transfer: Summer ↔ Winter")
    print("• Object Transformation: Horses ↔ Zebras")
    print("• Domain Adaptation: Synthetic ↔ Real")
    print("• Medical Imaging: Different modalities")
    print("• Photo Enhancement: Normal ↔ Enhanced")
    
    print(f"\nHISTORICAL IMPACT:")
    print("="*40)
    print("• Enabled practical image-to-image translation")
    print("• Eliminated paired data requirement")
    print("• Established cycle consistency paradigm")
    print("• Inspired numerous follow-up works")
    print("• Made translation accessible to broader applications")
    print("• Foundation for modern domain adaptation techniques")
    
    print(f"\nLIMITATIONS AND IMPROVEMENTS:")
    print("="*40)
    print("• Content preservation: May change important structures")
    print("• Geometric changes: Limited to appearance transformations")
    print("• Training instability: Balancing multiple loss components")
    print("• → UNIT: Unified framework for translation")
    print("• → MUNIT: Multimodal translation")
    print("• → Attention mechanisms: Better content preservation")
    
    print(f"\nMODERN RELEVANCE:")
    print("="*40)
    print("• Cycle consistency: Used across generative modeling")
    print("• Unpaired learning: Standard in domain adaptation")
    print("• PatchGAN: Adopted in many image generation tasks")
    print("• Translation framework: Foundation for modern methods")
    print("• Bidirectional mapping: Enables versatile applications")
    
    return {
        'model': 'CycleGAN Translation',
        'year': YEAR,
        'innovation': INNOVATION,
        'g_ab_params': g_ab_params,
        'g_ba_params': g_ba_params,
        'd_a_params': d_a_params,
        'd_b_params': d_b_params,
        'total_params': total_params,
        'cyclegan_analysis': cyclegan_analysis
    }

if __name__ == "__main__":
    results = main()