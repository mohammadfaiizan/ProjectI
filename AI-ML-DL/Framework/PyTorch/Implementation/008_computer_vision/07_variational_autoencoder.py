"""
PyTorch Variational Autoencoder - VAE for Image Generation
Comprehensive guide to implementing Variational Autoencoders in PyTorch
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
import math

print("=== VARIATIONAL AUTOENCODER ===")

# 1. BASIC VAE IMPLEMENTATION
print("\n1. BASIC VAE IMPLEMENTATION")

class VAEEncoder(nn.Module):
    """Encoder network for VAE"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super(VAEEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Mean and log variance heads
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Encode
        h = self.encoder(x)
        
        # Get mean and log variance
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar

class VAEDecoder(nn.Module):
    """Decoder network for VAE"""
    
    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int, output_shape: Tuple[int, ...]):
        super(VAEDecoder, self).__init__()
        
        self.output_dim = output_dim
        self.output_shape = output_shape
        
        # Reverse hidden dims for decoder
        hidden_dims = hidden_dims[::-1]
        
        # Build decoder layers
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # For image reconstruction
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x_recon = self.decoder(z)
        x_recon = x_recon.view(x_recon.size(0), *self.output_shape)
        return x_recon

class VAE(nn.Module):
    """Basic Variational Autoencoder"""
    
    def __init__(self, input_shape: Tuple[int, ...] = (1, 28, 28), 
                 hidden_dims: List[int] = [512, 256], latent_dim: int = 20):
        super(VAE, self).__init__()
        
        self.input_shape = input_shape
        self.input_dim = int(np.prod(input_shape))
        self.latent_dim = latent_dim
        
        # Initialize encoder and decoder
        self.encoder = VAEEncoder(self.input_dim, hidden_dims, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dims, self.input_dim, input_shape)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decoder(z)
        
        return x_recon, mu, logvar
    
    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """Generate samples from the VAE"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decoder(z)
        return samples

# Test basic VAE
vae = VAE(input_shape=(1, 28, 28), hidden_dims=[512, 256], latent_dim=20)
test_input = torch.randn(16, 1, 28, 28)

x_recon, mu, logvar = vae(test_input)
print(f"Input shape: {test_input.shape}")
print(f"Reconstructed shape: {x_recon.shape}")
print(f"Latent mu shape: {mu.shape}")
print(f"Latent logvar shape: {logvar.shape}")

# Generate samples
samples = vae.sample(8)
print(f"Generated samples shape: {samples.shape}")

# 2. CONVOLUTIONAL VAE
print("\n2. CONVOLUTIONAL VAE")

class ConvVAEEncoder(nn.Module):
    """Convolutional encoder for VAE"""
    
    def __init__(self, input_channels: int = 3, latent_dim: int = 128):
        super(ConvVAEEncoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(input_channels, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Calculate flattened size (256 * 4 * 4 for 64x64 input)
        self.flatten_size = 256 * 4 * 4
        
        # Latent space projections
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convolutional encoding
        h = self.conv_layers(x)
        h = h.view(h.size(0), -1)  # Flatten
        
        # Get mean and log variance
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar

class ConvVAEDecoder(nn.Module):
    """Convolutional decoder for VAE"""
    
    def __init__(self, latent_dim: int = 128, output_channels: int = 3):
        super(ConvVAEDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.init_size = 4
        
        # Project latent to initial feature map
        self.fc = nn.Linear(latent_dim, 256 * self.init_size * self.init_size)
        
        self.conv_layers = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, output_channels, 4, stride=2, padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Project and reshape
        h = self.fc(z)
        h = h.view(h.size(0), 256, self.init_size, self.init_size)
        
        # Deconvolutional decoding
        x_recon = self.conv_layers(h)
        
        return x_recon

class ConvVAE(nn.Module):
    """Convolutional Variational Autoencoder"""
    
    def __init__(self, input_channels: int = 3, latent_dim: int = 128):
        super(ConvVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.encoder = ConvVAEEncoder(input_channels, latent_dim)
        self.decoder = ConvVAEDecoder(latent_dim, input_channels)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    
    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """Generate samples"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decoder(z)
        return samples

# Test Convolutional VAE
conv_vae = ConvVAE(input_channels=3, latent_dim=128)
test_input = torch.randn(8, 3, 64, 64)

conv_x_recon, conv_mu, conv_logvar = conv_vae(test_input)
print(f"Conv VAE input: {test_input.shape}")
print(f"Conv VAE reconstruction: {conv_x_recon.shape}")
print(f"Conv VAE latent mu: {conv_mu.shape}")

conv_samples = conv_vae.sample(4)
print(f"Conv VAE samples: {conv_samples.shape}")

# 3. VAE LOSS FUNCTION
print("\n3. VAE LOSS FUNCTION")

class VAELoss:
    """VAE loss functions"""
    
    @staticmethod
    def vae_loss(x_recon: torch.Tensor, x: torch.Tensor, 
                 mu: torch.Tensor, logvar: torch.Tensor, 
                 beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Standard VAE loss with optional beta weighting
        
        Args:
            x_recon: Reconstructed input
            x: Original input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            beta: Weight for KL divergence term (beta-VAE)
        """
        # Reconstruction loss (Binary Cross Entropy)
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    @staticmethod
    def mse_vae_loss(x_recon: torch.Tensor, x: torch.Tensor,
                     mu: torch.Tensor, logvar: torch.Tensor,
                     beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """VAE loss with MSE reconstruction loss"""
        # Reconstruction loss (Mean Squared Error)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss

# Test VAE loss
x_recon, mu, logvar = conv_vae(test_input)
total_loss, recon_loss, kl_loss = VAELoss.vae_loss(x_recon, test_input, mu, logvar)

print(f"Total VAE loss: {total_loss.item():.4f}")
print(f"Reconstruction loss: {recon_loss.item():.4f}")
print(f"KL divergence loss: {kl_loss.item():.4f}")

# Test beta-VAE
beta_total_loss, beta_recon_loss, beta_kl_loss = VAELoss.vae_loss(
    x_recon, test_input, mu, logvar, beta=4.0
)
print(f"Beta-VAE (β=4.0) total loss: {beta_total_loss.item():.4f}")

# 4. BETA-VAE IMPLEMENTATION
print("\n4. BETA-VAE IMPLEMENTATION")

class BetaVAE(ConvVAE):
    """Beta-VAE for disentangled representation learning"""
    
    def __init__(self, input_channels: int = 3, latent_dim: int = 128, beta: float = 4.0):
        super(BetaVAE, self).__init__(input_channels, latent_dim)
        self.beta = beta
        
    def loss_function(self, x_recon: torch.Tensor, x: torch.Tensor,
                     mu: torch.Tensor, logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Beta-VAE loss function"""
        total_loss, recon_loss, kl_loss = VAELoss.vae_loss(x_recon, x, mu, logvar, self.beta)
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'beta': torch.tensor(self.beta)
        }

# Test Beta-VAE
beta_vae = BetaVAE(input_channels=3, latent_dim=128, beta=4.0)
beta_x_recon, beta_mu, beta_logvar = beta_vae(test_input)
beta_losses = beta_vae.loss_function(beta_x_recon, test_input, beta_mu, beta_logvar)

print("Beta-VAE losses:")
for loss_name, loss_value in beta_losses.items():
    print(f"  {loss_name}: {loss_value.item():.4f}")

# 5. CONDITIONAL VAE
print("\n5. CONDITIONAL VAE")

class ConditionalVAEEncoder(nn.Module):
    """Conditional VAE encoder with class conditioning"""
    
    def __init__(self, input_channels: int = 3, num_classes: int = 10, 
                 latent_dim: int = 128, embed_dim: int = 100):
        super(ConditionalVAEEncoder, self).__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Class embedding
        self.class_emb = nn.Embedding(num_classes, embed_dim)
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels + 1, 32, 4, stride=2, padding=1),  # +1 for embedded class
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.flatten_size = 256 * 4 * 4
        
        # Add class embedding to flattened features
        self.fc_mu = nn.Linear(self.flatten_size + embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size + embed_dim, latent_dim)
        
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        
        # Get class embeddings
        class_embed = self.class_emb(labels)  # [batch_size, embed_dim]
        
        # Expand class embedding to spatial dimensions
        class_embed_spatial = class_embed.view(batch_size, 1, 1, 1).expand(-1, 1, x.size(2), x.size(3))
        
        # Concatenate with input
        x_cond = torch.cat([x, class_embed_spatial], dim=1)
        
        # Convolutional encoding
        h = self.conv_layers(x_cond)
        h = h.view(batch_size, -1)
        
        # Concatenate with class embedding
        h_cond = torch.cat([h, class_embed], dim=1)
        
        # Get mean and log variance
        mu = self.fc_mu(h_cond)
        logvar = self.fc_logvar(h_cond)
        
        return mu, logvar

class ConditionalVAEDecoder(nn.Module):
    """Conditional VAE decoder with class conditioning"""
    
    def __init__(self, latent_dim: int = 128, num_classes: int = 10,
                 output_channels: int = 3, embed_dim: int = 100):
        super(ConditionalVAEDecoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.init_size = 4
        
        # Class embedding
        self.class_emb = nn.Embedding(num_classes, embed_dim)
        
        # Project latent + class to feature map
        self.fc = nn.Linear(latent_dim + embed_dim, 256 * self.init_size * self.init_size)
        
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, output_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Get class embeddings
        class_embed = self.class_emb(labels)
        
        # Concatenate latent code with class embedding
        z_cond = torch.cat([z, class_embed], dim=1)
        
        # Project and reshape
        h = self.fc(z_cond)
        h = h.view(h.size(0), 256, self.init_size, self.init_size)
        
        # Deconvolutional decoding
        x_recon = self.conv_layers(h)
        
        return x_recon

class ConditionalVAE(nn.Module):
    """Conditional Variational Autoencoder"""
    
    def __init__(self, input_channels: int = 3, num_classes: int = 10, 
                 latent_dim: int = 128, embed_dim: int = 100):
        super(ConditionalVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        self.encoder = ConditionalVAEEncoder(input_channels, num_classes, latent_dim, embed_dim)
        self.decoder = ConditionalVAEDecoder(latent_dim, num_classes, input_channels, embed_dim)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x, labels)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, labels)
        return x_recon, mu, logvar
    
    def sample(self, num_samples: int, labels: torch.Tensor, device: str = 'cpu') -> torch.Tensor:
        """Generate conditional samples"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decoder(z, labels)
        return samples

# Test Conditional VAE
cvae = ConditionalVAE(input_channels=3, num_classes=10, latent_dim=128)
test_labels = torch.randint(0, 10, (8,))

cvae_x_recon, cvae_mu, cvae_logvar = cvae(test_input, test_labels)
print(f"Conditional VAE reconstruction: {cvae_x_recon.shape}")

# Generate conditional samples
cond_samples = cvae.sample(4, test_labels[:4])
print(f"Conditional VAE samples: {cond_samples.shape}")

# 6. VAE TRAINING
print("\n6. VAE TRAINING")

class VAETrainer:
    """Training class for VAE models"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu', beta: float = 1.0):
        self.model = model.to(device)
        self.device = device
        self.beta = beta
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Training history
        self.train_losses = []
        self.recon_losses = []
        self.kl_losses = []
        
    def train_step(self, x: torch.Tensor, labels: torch.Tensor = None) -> Dict[str, float]:
        """Single training step"""
        x = x.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass
        if labels is not None:
            # Conditional VAE
            x_recon, mu, logvar = self.model(x, labels)
        else:
            # Standard VAE
            x_recon, mu, logvar = self.model(x)
        
        # Calculate loss
        total_loss, recon_loss, kl_loss = VAELoss.vae_loss(x_recon, x, mu, logvar, self.beta)
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item()
        }
    
    def train_epoch(self, dataloader: DataLoader, conditional: bool = False) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = {'total_loss': 0, 'recon_loss': 0, 'kl_loss': 0}
        num_batches = 0
        
        for batch_idx, (data, labels) in enumerate(dataloader):
            if conditional:
                losses = self.train_step(data, labels)
            else:
                losses = self.train_step(data)
            
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}: Total Loss: {losses["total_loss"]:.4f}, '
                      f'Recon: {losses["recon_loss"]:.4f}, KL: {losses["kl_loss"]:.4f}')
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        self.train_losses.append(epoch_losses['total_loss'])
        self.recon_losses.append(epoch_losses['recon_loss'])
        self.kl_losses.append(epoch_losses['kl_loss'])
        
        return epoch_losses

# 7. LATENT SPACE INTERPOLATION
print("\n7. LATENT SPACE INTERPOLATION")

class LatentInterpolator:
    """Utilities for latent space interpolation and visualization"""
    
    def __init__(self, vae_model: nn.Module, device: str = 'cpu'):
        self.model = vae_model
        self.device = device
        
    def encode(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """Encode input to latent space"""
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            if labels is not None:
                labels = labels.to(self.device)
                mu, logvar = self.model.encoder(x, labels)
            else:
                mu, logvar = self.model.encoder(x)
            # Use mean for interpolation
            return mu
    
    def decode(self, z: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """Decode latent codes to images"""
        self.model.eval()
        with torch.no_grad():
            z = z.to(self.device)
            if labels is not None:
                labels = labels.to(self.device)
                return self.model.decoder(z, labels)
            else:
                return self.model.decoder(z)
    
    def interpolate(self, z1: torch.Tensor, z2: torch.Tensor, 
                   num_steps: int = 10, labels: torch.Tensor = None) -> torch.Tensor:
        """Interpolate between two latent codes"""
        # Create interpolation coefficients
        alphas = torch.linspace(0, 1, num_steps).view(-1, 1).to(self.device)
        
        # Interpolate
        z1_expanded = z1.unsqueeze(0).expand(num_steps, -1)
        z2_expanded = z2.unsqueeze(0).expand(num_steps, -1)
        
        z_interp = alphas * z2_expanded + (1 - alphas) * z1_expanded
        
        # Decode interpolated latent codes
        if labels is not None:
            labels_expanded = labels.expand(num_steps)
            interpolated_images = self.decode(z_interp, labels_expanded)
        else:
            interpolated_images = self.decode(z_interp)
        
        return interpolated_images
    
    def sample_prior(self, num_samples: int, labels: torch.Tensor = None) -> torch.Tensor:
        """Sample from prior distribution"""
        z = torch.randn(num_samples, self.model.latent_dim).to(self.device)
        return self.decode(z, labels)

# Test interpolation
interpolator = LatentInterpolator(conv_vae)

# Create two random images and interpolate
img1 = torch.randn(1, 3, 64, 64)
img2 = torch.randn(1, 3, 64, 64)

z1 = interpolator.encode(img1)
z2 = interpolator.encode(img2)

interpolated = interpolator.interpolate(z1[0], z2[0], num_steps=5)
print(f"Interpolated images shape: {interpolated.shape}")

# Sample from prior
prior_samples = interpolator.sample_prior(4)
print(f"Prior samples shape: {prior_samples.shape}")

# 8. DUMMY DATASET AND TRAINING EXAMPLE
print("\n8. DUMMY DATASET AND TRAINING EXAMPLE")

class DummyImageDataset(Dataset):
    """Dummy image dataset for VAE training"""
    
    def __init__(self, size: int = 1000, img_size: int = 64, num_channels: int = 3):
        self.size = size
        self.img_size = img_size
        self.num_channels = num_channels
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Generate random image in [0, 1] range
        img = torch.rand(self.num_channels, self.img_size, self.img_size)
        label = torch.randint(0, 10, (1,)).item()
        return img, label

# Create dataset and train VAE
dummy_dataset = DummyImageDataset(size=200, img_size=64, num_channels=3)
dummy_loader = DataLoader(dummy_dataset, batch_size=16, shuffle=True)

# Train standard VAE
print("Training standard VAE...")
vae_trainer = VAETrainer(ConvVAE(input_channels=3, latent_dim=64), beta=1.0)

for epoch in range(2):
    losses = vae_trainer.train_epoch(dummy_loader, conditional=False)
    print(f'Epoch {epoch+1}: Total: {losses["total_loss"]:.4f}, '
          f'Recon: {losses["recon_loss"]:.4f}, KL: {losses["kl_loss"]:.4f}')

# Train conditional VAE
print("\nTraining conditional VAE...")
cvae_trainer = VAETrainer(ConditionalVAE(input_channels=3, num_classes=10, latent_dim=64), beta=1.0)

for epoch in range(2):
    losses = cvae_trainer.train_epoch(dummy_loader, conditional=True)
    print(f'Epoch {epoch+1}: Total: {losses["total_loss"]:.4f}, '
          f'Recon: {losses["recon_loss"]:.4f}, KL: {losses["kl_loss"]:.4f}')

print("\n=== VARIATIONAL AUTOENCODER COMPLETE ===")
print("Key concepts covered:")
print("- Basic VAE architecture (Encoder-Decoder with reparameterization)")
print("- Convolutional VAE for image data")
print("- VAE loss function (reconstruction + KL divergence)")
print("- Beta-VAE for disentangled representations")
print("- Conditional VAE for class-conditional generation")
print("- VAE training pipeline")
print("- Latent space interpolation and visualization")
print("- Training examples and best practices")