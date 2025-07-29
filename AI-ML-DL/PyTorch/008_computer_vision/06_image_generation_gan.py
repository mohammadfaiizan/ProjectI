"""
PyTorch Image Generation with GANs - Generative Adversarial Networks
Comprehensive guide to implementing GANs for image generation in PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import random

print("=== IMAGE GENERATION WITH GANS ===")

# 1. BASIC GAN IMPLEMENTATION
print("\n1. BASIC GAN IMPLEMENTATION")

class SimpleGenerator(nn.Module):
    """Simple generator network for basic GAN"""
    
    def __init__(self, latent_dim: int = 100, img_shape: Tuple[int, int, int] = (1, 28, 28)):
        super(SimpleGenerator, self).__init__()
        
        self.img_shape = img_shape
        self.img_size = int(np.prod(img_shape))
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self.img_size),
            nn.Tanh()
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class SimpleDiscriminator(nn.Module):
    """Simple discriminator network for basic GAN"""
    
    def __init__(self, img_shape: Tuple[int, int, int] = (1, 28, 28)):
        super(SimpleDiscriminator, self).__init__()
        
        self.img_size = int(np.prod(img_shape))
        
        self.model = nn.Sequential(
            nn.Linear(self.img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Test basic GAN components
latent_dim = 100
img_shape = (1, 28, 28)

generator = SimpleGenerator(latent_dim, img_shape)
discriminator = SimpleDiscriminator(img_shape)

# Test forward passes
z = torch.randn(16, latent_dim)
fake_imgs = generator(z)
print(f"Generated images shape: {fake_imgs.shape}")

real_imgs = torch.randn(16, *img_shape)
real_output = discriminator(real_imgs)
fake_output = discriminator(fake_imgs)
print(f"Discriminator real output: {real_output.shape}")
print(f"Discriminator fake output: {fake_output.shape}")

# 2. DCGAN IMPLEMENTATION
print("\n2. DCGAN IMPLEMENTATION")

class DCGANGenerator(nn.Module):
    """Deep Convolutional GAN Generator"""
    
    def __init__(self, latent_dim: int = 100, feature_map_size: int = 64, num_channels: int = 3):
        super(DCGANGenerator, self).__init__()
        
        self.init_size = 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 512 * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            
            # Upsample to 8x8
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Upsample to 16x16
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Upsample to 32x32
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Upsample to 64x64
            nn.ConvTranspose2d(64, num_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.l1(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class DCGANDiscriminator(nn.Module):
    """Deep Convolutional GAN Discriminator"""
    
    def __init__(self, num_channels: int = 3, feature_map_size: int = 64):
        super(DCGANDiscriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, bn=True):
            """Discriminator block with conv, batchnorm, leaky relu"""
            block = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block
        
        self.model = nn.Sequential(
            # Input: 64x64
            *discriminator_block(num_channels, 64, bn=False),
            # 32x32
            *discriminator_block(64, 128),
            # 16x16
            *discriminator_block(128, 256),
            # 8x8
            *discriminator_block(256, 512),
            # 4x4
            nn.Conv2d(512, 1, 4, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        validity = self.model(img)
        return validity.view(-1, 1)

# Test DCGAN
dcgan_generator = DCGANGenerator(latent_dim=100, num_channels=3)
dcgan_discriminator = DCGANDiscriminator(num_channels=3)

z = torch.randn(16, 100)
generated_imgs = dcgan_generator(z)
print(f"DCGAN generated images shape: {generated_imgs.shape}")

disc_output = dcgan_discriminator(generated_imgs)
print(f"DCGAN discriminator output shape: {disc_output.shape}")

# 3. IMPROVED GAN TECHNIQUES
print("\n3. IMPROVED GAN TECHNIQUES")

class SelfAttention(nn.Module):
    """Self-attention module for SAGAN"""
    
    def __init__(self, in_dim: int):
        super(SelfAttention, self).__init__()
        
        self.in_dim = in_dim
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, C, H, W = x.size()
        
        # Generate query, key, value
        proj_query = self.query_conv(x).view(batch_size, -1, W * H).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, W * H)
        proj_value = self.value_conv(x).view(batch_size, -1, W * H)
        
        # Compute attention
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        # Apply attention to value
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        return out

class SpectralNorm:
    """Spectral normalization for stable training"""
    
    def __init__(self, name: str = 'weight'):
        self.name = name
        
    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        
        # Power iteration
        v = F.normalize(torch.mv(weight.view(weight.size(0), -1).t(), u), dim=0)
        u = F.normalize(torch.mv(weight.view(weight.size(0), -1), v), dim=0)
        
        # Compute spectral norm
        sigma = torch.dot(u, torch.mv(weight.view(weight.size(0), -1), v))
        
        # Normalize weight
        weight = weight / sigma.expand_as(weight)
        
        # Update u
        setattr(module, self.name + '_u', u)
        
        return weight
    
    @staticmethod
    def apply(module, name):
        fn = SpectralNorm(name)
        
        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight))
        u = F.normalize(torch.randn(weight.size(0)), dim=0)
        module.register_buffer(name + '_u', u)
        
        module.register_forward_pre_hook(fn)
        return fn
    
    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)

def spectral_norm(module, name='weight'):
    """Apply spectral normalization to a module"""
    SpectralNorm.apply(module, name)
    return module

class ImprovedGenerator(nn.Module):
    """Improved generator with self-attention and spectral norm"""
    
    def __init__(self, latent_dim: int = 100, num_channels: int = 3):
        super(ImprovedGenerator, self).__init__()
        
        self.init_size = 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 512 * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Self-attention at 16x16
            SelfAttention(128),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, num_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.l1(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class ImprovedDiscriminator(nn.Module):
    """Improved discriminator with self-attention and spectral norm"""
    
    def __init__(self, num_channels: int = 3):
        super(ImprovedDiscriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, bn=True, sn=True):
            block = []
            conv = nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)
            if sn:
                conv = spectral_norm(conv)
            block.append(conv)
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block
        
        self.conv_blocks = nn.Sequential(
            # 64x64 -> 32x32
            *discriminator_block(num_channels, 64, bn=False),
            # 32x32 -> 16x16
            *discriminator_block(64, 128),
            # Self-attention at 16x16
            SelfAttention(128),
            # 16x16 -> 8x8
            *discriminator_block(128, 256),
            # 8x8 -> 4x4
            *discriminator_block(256, 512),
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(512, 1, 4, padding=0),
        )
        
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        features = self.conv_blocks(img)
        validity = self.final(features)
        return validity.view(-1, 1)

# Test improved GAN
improved_gen = ImprovedGenerator(latent_dim=100, num_channels=3)
improved_disc = ImprovedDiscriminator(num_channels=3)

z = torch.randn(4, 100)
improved_imgs = improved_gen(z)
improved_output = improved_disc(improved_imgs)

print(f"Improved GAN generated images: {improved_imgs.shape}")
print(f"Improved GAN discriminator output: {improved_output.shape}")

# 4. GAN LOSS FUNCTIONS
print("\n4. GAN LOSS FUNCTIONS")

class GANLoss:
    """Collection of GAN loss functions"""
    
    @staticmethod
    def vanilla_loss(real_output: torch.Tensor, fake_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Original GAN loss (Binary Cross Entropy)"""
        real_labels = torch.ones_like(real_output)
        fake_labels = torch.zeros_like(fake_output)
        
        # Discriminator loss
        d_loss_real = F.binary_cross_entropy(real_output, real_labels)
        d_loss_fake = F.binary_cross_entropy(fake_output, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        
        # Generator loss
        g_loss = F.binary_cross_entropy(fake_output, real_labels)
        
        return d_loss, g_loss
    
    @staticmethod
    def lsgan_loss(real_output: torch.Tensor, fake_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Least Squares GAN loss"""
        # Discriminator loss
        d_loss_real = 0.5 * torch.mean((real_output - 1) ** 2)
        d_loss_fake = 0.5 * torch.mean(fake_output ** 2)
        d_loss = d_loss_real + d_loss_fake
        
        # Generator loss
        g_loss = 0.5 * torch.mean((fake_output - 1) ** 2)
        
        return d_loss, g_loss
    
    @staticmethod
    def wgan_loss(real_output: torch.Tensor, fake_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Wasserstein GAN loss"""
        # Discriminator loss (maximize)
        d_loss = -torch.mean(real_output) + torch.mean(fake_output)
        
        # Generator loss (minimize)
        g_loss = -torch.mean(fake_output)
        
        return d_loss, g_loss
    
    @staticmethod
    def wgan_gp_loss(discriminator: nn.Module, real_imgs: torch.Tensor, 
                     fake_imgs: torch.Tensor, device: str = 'cpu') -> torch.Tensor:
        """Gradient penalty for WGAN-GP"""
        batch_size = real_imgs.size(0)
        
        # Random interpolation between real and fake
        alpha = torch.rand(batch_size, 1, 1, 1).to(device)
        interpolated = alpha * real_imgs + (1 - alpha) * fake_imgs
        interpolated.requires_grad_(True)
        
        # Get discriminator output for interpolated samples
        d_interpolated = discriminator(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated, inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True, retain_graph=True
        )[0]
        
        # Gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
        
        return gradient_penalty

# Test loss functions
real_output = torch.rand(16, 1) * 0.9 + 0.1  # Real images should have high scores
fake_output = torch.rand(16, 1) * 0.1  # Fake images should have low scores

vanilla_d_loss, vanilla_g_loss = GANLoss.vanilla_loss(real_output, fake_output)
lsgan_d_loss, lsgan_g_loss = GANLoss.lsgan_loss(real_output, fake_output)
wgan_d_loss, wgan_g_loss = GANLoss.wgan_loss(real_output, fake_output)

print(f"Vanilla GAN losses - D: {vanilla_d_loss:.4f}, G: {vanilla_g_loss:.4f}")
print(f"LSGAN losses - D: {lsgan_d_loss:.4f}, G: {lsgan_g_loss:.4f}")
print(f"WGAN losses - D: {wgan_d_loss:.4f}, G: {wgan_g_loss:.4f}")

# 5. GAN TRAINING LOOP
print("\n5. GAN TRAINING LOOP")

class GANTrainer:
    """Training class for GANs"""
    
    def __init__(self, generator: nn.Module, discriminator: nn.Module, 
                 latent_dim: int, device: str = 'cpu', loss_type: str = 'vanilla'):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.latent_dim = latent_dim
        self.device = device
        self.loss_type = loss_type
        
        # Optimizers
        self.g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Training history
        self.g_losses = []
        self.d_losses = []
        
    def train_step(self, real_imgs: torch.Tensor) -> Tuple[float, float]:
        """Single training step"""
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(self.device)
        
        # Generate fake images
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_imgs = self.generator(z)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        real_output = self.discriminator(real_imgs)
        fake_output = self.discriminator(fake_imgs.detach())
        
        if self.loss_type == 'vanilla':
            d_loss, _ = GANLoss.vanilla_loss(real_output, fake_output)
        elif self.loss_type == 'lsgan':
            d_loss, _ = GANLoss.lsgan_loss(real_output, fake_output)
        elif self.loss_type == 'wgan':
            d_loss, _ = GANLoss.wgan_loss(real_output, fake_output)
        elif self.loss_type == 'wgan-gp':
            d_loss, _ = GANLoss.wgan_loss(real_output, fake_output)
            gp = GANLoss.wgan_gp_loss(self.discriminator, real_imgs, fake_imgs, self.device)
            d_loss += 10 * gp  # Gradient penalty coefficient
        
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.g_optimizer.zero_grad()
        
        fake_output = self.discriminator(fake_imgs)
        
        if self.loss_type == 'vanilla':
            _, g_loss = GANLoss.vanilla_loss(real_output, fake_output)
        elif self.loss_type == 'lsgan':
            _, g_loss = GANLoss.lsgan_loss(real_output, fake_output)
        elif self.loss_type in ['wgan', 'wgan-gp']:
            _, g_loss = GANLoss.wgan_loss(real_output, fake_output)
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return d_loss.item(), g_loss.item()
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        num_batches = 0
        
        for batch_idx, (real_imgs, _) in enumerate(dataloader):
            d_loss, g_loss = self.train_step(real_imgs)
            
            epoch_d_loss += d_loss
            epoch_g_loss += g_loss
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}: D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}')
        
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        
        self.d_losses.append(avg_d_loss)
        self.g_losses.append(avg_g_loss)
        
        return avg_d_loss, avg_g_loss
    
    def generate_samples(self, num_samples: int = 16) -> torch.Tensor:
        """Generate sample images"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            samples = self.generator(z)
        self.generator.train()
        return samples

# 6. CONDITIONAL GAN
print("\n6. CONDITIONAL GAN")

class ConditionalGenerator(nn.Module):
    """Conditional generator with class embedding"""
    
    def __init__(self, latent_dim: int = 100, num_classes: int = 10, 
                 embed_dim: int = 100, img_shape: Tuple[int, int, int] = (1, 28, 28)):
        super(ConditionalGenerator, self).__init__()
        
        self.img_shape = img_shape
        self.img_size = int(np.prod(img_shape))
        
        # Label embedding
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self.img_size),
            nn.Tanh()
        )
        
    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Embed labels
        label_embed = self.label_emb(labels)
        
        # Concatenate noise and label embedding
        gen_input = torch.cat([z, label_embed], dim=1)
        
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img

class ConditionalDiscriminator(nn.Module):
    """Conditional discriminator with class embedding"""
    
    def __init__(self, num_classes: int = 10, embed_dim: int = 100,
                 img_shape: Tuple[int, int, int] = (1, 28, 28)):
        super(ConditionalDiscriminator, self).__init__()
        
        self.img_size = int(np.prod(img_shape))
        
        # Label embedding
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        
        self.model = nn.Sequential(
            nn.Linear(self.img_size + embed_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Flatten image
        img_flat = img.view(img.size(0), -1)
        
        # Embed labels
        label_embed = self.label_emb(labels)
        
        # Concatenate image and label embedding
        disc_input = torch.cat([img_flat, label_embed], dim=1)
        
        validity = self.model(disc_input)
        return validity

# Test conditional GAN
cond_gen = ConditionalGenerator(latent_dim=100, num_classes=10)
cond_disc = ConditionalDiscriminator(num_classes=10)

z = torch.randn(16, 100)
labels = torch.randint(0, 10, (16,))

cond_fake_imgs = cond_gen(z, labels)
cond_output = cond_disc(cond_fake_imgs, labels)

print(f"Conditional GAN generated images: {cond_fake_imgs.shape}")
print(f"Conditional GAN discriminator output: {cond_output.shape}")

# 7. EVALUATION METRICS
print("\n7. EVALUATION METRICS")

class GANEvaluator:
    """Evaluation metrics for GANs"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
    def inception_score(self, images: torch.Tensor, batch_size: int = 32) -> Tuple[float, float]:
        """Calculate Inception Score (simplified)"""
        # This is a simplified version - in practice, you'd use a pretrained Inception model
        print("Note: This is a simplified IS calculation for demonstration")
        
        # Dummy implementation - replace with actual Inception model
        num_samples = images.size(0)
        num_classes = 10  # Assume 10 classes
        
        # Simulate class predictions
        predictions = torch.softmax(torch.randn(num_samples, num_classes), dim=1)
        
        # Calculate IS
        kl_divs = []
        marginal = predictions.mean(dim=0)
        
        for i in range(num_samples):
            kl_div = F.kl_div(torch.log(predictions[i] + 1e-8), marginal, reduction='sum')
            kl_divs.append(kl_div)
        
        is_score = torch.exp(torch.mean(torch.stack(kl_divs)))
        is_std = torch.std(torch.stack(kl_divs))
        
        return is_score.item(), is_std.item()
    
    def fid_score(self, real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
        """Calculate Fréchet Inception Distance (simplified)"""
        print("Note: This is a simplified FID calculation for demonstration")
        
        # Calculate means and covariances
        mu_real = torch.mean(real_features, dim=0)
        mu_fake = torch.mean(fake_features, dim=0)
        
        sigma_real = torch.cov(real_features.T)
        sigma_fake = torch.cov(fake_features.T)
        
        # Calculate FID
        diff = mu_real - mu_fake
        covmean = torch.sqrt(sigma_real @ sigma_fake)
        
        fid = torch.sum(diff ** 2) + torch.trace(sigma_real + sigma_fake - 2 * covmean)
        
        return fid.item()
    
    def mode_score(self, generated_images: torch.Tensor, real_images: torch.Tensor) -> float:
        """Calculate mode score to detect mode collapse"""
        # Simple mode score based on diversity
        gen_features = generated_images.view(generated_images.size(0), -1)
        real_features = real_images.view(real_images.size(0), -1)
        
        # Calculate pairwise distances
        gen_distances = torch.cdist(gen_features, gen_features)
        real_distances = torch.cdist(real_features, real_features)
        
        # Mode score based on average distance ratios
        gen_avg_dist = torch.mean(gen_distances[gen_distances > 0])
        real_avg_dist = torch.mean(real_distances[real_distances > 0])
        
        mode_score = gen_avg_dist / real_avg_dist
        
        return mode_score.item()

# Test evaluation metrics
evaluator = GANEvaluator()

# Generate dummy data for testing
dummy_generated = torch.randn(100, 3, 64, 64)
dummy_real = torch.randn(100, 3, 64, 64)

is_score, is_std = evaluator.inception_score(dummy_generated)
print(f"Inception Score: {is_score:.4f} ± {is_std:.4f}")

# Dummy feature vectors for FID
dummy_real_features = torch.randn(100, 2048)
dummy_fake_features = torch.randn(100, 2048)
fid = evaluator.fid_score(dummy_real_features, dummy_fake_features)
print(f"FID Score: {fid:.4f}")

mode_score = evaluator.mode_score(dummy_generated, dummy_real)
print(f"Mode Score: {mode_score:.4f}")

# 8. DUMMY DATASET AND TRAINING EXAMPLE
print("\n8. DUMMY DATASET AND TRAINING EXAMPLE")

class DummyImageDataset(Dataset):
    """Dummy image dataset for demonstration"""
    
    def __init__(self, size: int = 1000, img_size: int = 64, num_channels: int = 3):
        self.size = size
        self.img_size = img_size
        self.num_channels = num_channels
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Generate random image
        img = torch.randn(self.num_channels, self.img_size, self.img_size)
        # Normalize to [-1, 1] range
        img = torch.tanh(img)
        label = torch.randint(0, 10, (1,)).item()
        return img, label

# Create dummy dataset and training example
dummy_dataset = DummyImageDataset(size=200, img_size=64, num_channels=3)
dummy_loader = DataLoader(dummy_dataset, batch_size=16, shuffle=True)

# Initialize trainer
trainer = GANTrainer(
    generator=DCGANGenerator(latent_dim=100, num_channels=3),
    discriminator=DCGANDiscriminator(num_channels=3),
    latent_dim=100,
    device='cpu',
    loss_type='vanilla'
)

print("Training GAN for 2 epochs...")
for epoch in range(2):
    d_loss, g_loss = trainer.train_epoch(dummy_loader)
    print(f'Epoch {epoch+1}: D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}')

# Generate samples
samples = trainer.generate_samples(num_samples=8)
print(f"Generated samples shape: {samples.shape}")

print("\n=== IMAGE GENERATION WITH GANS COMPLETE ===")
print("Key concepts covered:")
print("- Basic GAN architecture (Generator & Discriminator)")
print("- DCGAN with convolutional layers")
print("- Improved techniques (Self-attention, Spectral norm)")
print("- Various GAN loss functions (Vanilla, LSGAN, WGAN)")
print("- Complete training pipeline")
print("- Conditional GANs")
print("- Evaluation metrics (IS, FID, Mode Score)")
print("- Training examples and best practices")