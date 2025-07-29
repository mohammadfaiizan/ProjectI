"""
PyTorch Image Super-Resolution - Deep Learning Super-Resolution Networks
Comprehensive guide to implementing super-resolution networks in PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, List, Optional, Dict, Union
import math

print("=== IMAGE SUPER-RESOLUTION ===")

# 1. BASIC SRCNN IMPLEMENTATION
print("\n1. BASIC SRCNN IMPLEMENTATION")

class SRCNN(nn.Module):
    """Super-Resolution Convolutional Neural Network"""
    
    def __init__(self, num_channels: int = 3, upscale_factor: int = 2):
        super(SRCNN, self).__init__()
        
        self.upscale_factor = upscale_factor
        
        # Three-layer CNN architecture
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upscale input using bicubic interpolation
        x = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        
        # SRCNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        
        return x

# Test SRCNN
srcnn = SRCNN(num_channels=3, upscale_factor=2)
test_lr = torch.randn(1, 3, 64, 64)  # Low resolution input
test_sr = srcnn(test_lr)

print(f"SRCNN input shape: {test_lr.shape}")
print(f"SRCNN output shape: {test_sr.shape}")

# Calculate parameters
total_params = sum(p.numel() for p in srcnn.parameters())
print(f"SRCNN parameters: {total_params:,}")

# 2. EFFICIENT SUB-PIXEL CONVOLUTION
print("\n2. EFFICIENT SUB-PIXEL CONVOLUTION")

class PixelShuffle(nn.Module):
    """Efficient sub-pixel convolution layer"""
    
    def __init__(self, upscale_factor: int):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_shuffle(x, self.upscale_factor)

class SubPixelCNN(nn.Module):
    """Sub-pixel CNN for efficient super-resolution"""
    
    def __init__(self, num_channels: int = 3, upscale_factor: int = 2, num_features: int = 64):
        super(SubPixelCNN, self).__init__()
        
        self.upscale_factor = upscale_factor
        
        # Feature extraction
        self.conv1 = nn.Conv2d(num_channels, num_features, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        
        # Sub-pixel convolution
        self.conv_up = nn.Conv2d(num_features, num_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = PixelShuffle(upscale_factor)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Sub-pixel convolution for upsampling
        x = self.conv_up(x)
        x = self.pixel_shuffle(x)
        
        return x

# Test Sub-pixel CNN
subpixel_cnn = SubPixelCNN(num_channels=3, upscale_factor=2)
test_subpixel_sr = subpixel_cnn(test_lr)

print(f"Sub-pixel CNN output shape: {test_subpixel_sr.shape}")

subpixel_params = sum(p.numel() for p in subpixel_cnn.parameters())
print(f"Sub-pixel CNN parameters: {subpixel_params:,}")

# 3. RESIDUAL BLOCKS FOR SUPER-RESOLUTION
print("\n3. RESIDUAL BLOCKS FOR SUPER-RESOLUTION")

class ResidualBlock(nn.Module):
    """Residual block for super-resolution networks"""
    
    def __init__(self, num_features: int = 64, use_batch_norm: bool = True):
        super(ResidualBlock, self).__init__()
        
        layers = []
        layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(num_features))
        
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(num_features))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)

class ResNet(nn.Module):
    """ResNet-based super-resolution network"""
    
    def __init__(self, num_channels: int = 3, num_features: int = 64, 
                 num_residual_blocks: int = 16, upscale_factor: int = 2):
        super(ResNet, self).__init__()
        
        self.upscale_factor = upscale_factor
        
        # Initial convolution
        self.conv_first = nn.Conv2d(num_channels, num_features, kernel_size=9, padding=4)
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(num_features) for _ in range(num_residual_blocks)
        ])
        
        # Middle convolution
        self.conv_mid = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.bn_mid = nn.BatchNorm2d(num_features)
        
        # Upsampling layers
        upsampling = []
        if upscale_factor == 2 or upscale_factor == 4 or upscale_factor == 8:
            for _ in range(int(math.log(upscale_factor, 2))):
                upsampling.append(nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1))
                upsampling.append(PixelShuffle(2))
                upsampling.append(nn.ReLU(inplace=True))
        elif upscale_factor == 3:
            upsampling.append(nn.Conv2d(num_features, num_features * 9, kernel_size=3, padding=1))
            upsampling.append(PixelShuffle(3))
            upsampling.append(nn.ReLU(inplace=True))
        
        self.upsampling = nn.Sequential(*upsampling)
        
        # Final convolution
        self.conv_last = nn.Conv2d(num_features, num_channels, kernel_size=9, padding=4)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial feature extraction
        x1 = F.relu(self.conv_first(x))
        
        # Residual learning
        x = self.residual_blocks(x1)
        
        # Middle convolution with skip connection
        x = self.bn_mid(self.conv_mid(x))
        x = x + x1
        
        # Upsampling
        x = self.upsampling(x)
        
        # Final reconstruction
        x = self.conv_last(x)
        
        return x

# Test ResNet SR
resnet_sr = ResNet(num_channels=3, num_residual_blocks=8, upscale_factor=2)
test_resnet_sr = resnet_sr(test_lr)

print(f"ResNet SR output shape: {test_resnet_sr.shape}")

resnet_params = sum(p.numel() for p in resnet_sr.parameters())
print(f"ResNet SR parameters: {resnet_params:,}")

# 4. DENSE BLOCKS (DENSENET-STYLE)
print("\n4. DENSE BLOCKS (DENSENET-STYLE)")

class DenseLayer(nn.Module):
    """Dense layer for DenseNet-style networks"""
    
    def __init__(self, in_channels: int, growth_rate: int):
        super(DenseLayer, self).__init__()
        
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return torch.cat([x, out], dim=1)

class DenseBlock(nn.Module):
    """Dense block with multiple dense layers"""
    
    def __init__(self, in_channels: int, growth_rate: int, num_layers: int):
        super(DenseBlock, self).__init__()
        
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class RDN(nn.Module):
    """Residual Dense Network for super-resolution"""
    
    def __init__(self, num_channels: int = 3, num_features: int = 64, 
                 growth_rate: int = 32, num_blocks: int = 16, 
                 num_layers: int = 8, upscale_factor: int = 2):
        super(RDN, self).__init__()
        
        # Shallow feature extraction
        self.conv_first = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)
        
        # Residual dense blocks
        self.rdb_blocks = nn.ModuleList([
            DenseBlock(num_features, growth_rate, num_layers) for _ in range(num_blocks)
        ])
        
        # Local feature fusion
        self.lff = nn.Conv2d(num_features + num_layers * growth_rate, num_features, kernel_size=1)
        
        # Global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(num_features * num_blocks, num_features, kernel_size=1),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )
        
        # Upsampling
        self.upsampling = nn.Sequential(
            nn.Conv2d(num_features, num_features * (upscale_factor ** 2), kernel_size=3, padding=1),
            PixelShuffle(upscale_factor)
        )
        
        # Reconstruction
        self.conv_last = nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shallow feature extraction
        f_1 = self.conv_first(x)
        
        # Residual dense blocks
        local_features = []
        rdb_out = f_1
        
        for rdb in self.rdb_blocks:
            rdb_out = rdb(rdb_out)
            local_features.append(self.lff(rdb_out))
            rdb_out = f_1  # Reset for next block
        
        # Global feature fusion
        global_features = torch.cat(local_features, dim=1)
        global_features = self.gff(global_features)
        global_features = global_features + f_1
        
        # Upsampling and reconstruction
        upsampled = self.upsampling(global_features)
        output = self.conv_last(upsampled)
        
        return output

# Test RDN (with fewer blocks for demo)
rdn_sr = RDN(num_blocks=4, num_layers=4, upscale_factor=2)
test_rdn_sr = rdn_sr(test_lr)

print(f"RDN output shape: {test_rdn_sr.shape}")

rdn_params = sum(p.numel() for p in rdn_sr.parameters())
print(f"RDN parameters: {rdn_params:,}")

# 5. GENERATIVE ADVERSARIAL NETWORKS FOR SR
print("\n5. GENERATIVE ADVERSARIAL NETWORKS FOR SR")

class SRResidualBlock(nn.Module):
    """Residual block for SRGAN generator"""
    
    def __init__(self, num_features: int = 64):
        super(SRResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual

class SRGANGenerator(nn.Module):
    """Generator network for SRGAN"""
    
    def __init__(self, num_channels: int = 3, num_features: int = 64, 
                 num_residual_blocks: int = 16, upscale_factor: int = 4):
        super(SRGANGenerator, self).__init__()
        
        # Initial convolution
        self.conv_first = nn.Sequential(
            nn.Conv2d(num_channels, num_features, kernel_size=9, padding=4),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(*[
            SRResidualBlock(num_features) for _ in range(num_residual_blocks)
        ])
        
        # Middle convolution
        self.conv_mid = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features)
        )
        
        # Upsampling blocks
        upsampling = []
        for _ in range(int(math.log(upscale_factor, 2))):
            upsampling.extend([
                nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                PixelShuffle(2),
                nn.ReLU(inplace=True)
            ])
        
        self.upsampling = nn.Sequential(*upsampling)
        
        # Final convolution
        self.conv_last = nn.Conv2d(num_features, num_channels, kernel_size=9, padding=4)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial features
        x1 = self.conv_first(x)
        
        # Residual learning
        x = self.residual_blocks(x1)
        x = self.conv_mid(x) + x1
        
        # Upsampling
        x = self.upsampling(x)
        
        # Final output
        return torch.sigmoid(self.conv_last(x))

class SRGANDiscriminator(nn.Module):
    """Discriminator network for SRGAN"""
    
    def __init__(self, num_channels: int = 3):
        super(SRGANDiscriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, stride=1, normalize=True):
            """Discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(num_channels, 64, normalize=False),
            *discriminator_block(64, 64, stride=2),
            *discriminator_block(64, 128),
            *discriminator_block(128, 128, stride=2),
            *discriminator_block(128, 256),
            *discriminator_block(256, 256, stride=2),
            *discriminator_block(256, 512),
            *discriminator_block(512, 512, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).view(x.size(0), -1)

# Test SRGAN components
srgan_gen = SRGANGenerator(upscale_factor=4)
srgan_disc = SRGANDiscriminator()

test_lr_small = torch.randn(1, 3, 32, 32)
test_sr_gan = srgan_gen(test_lr_small)
test_disc_out = srgan_disc(test_sr_gan)

print(f"SRGAN Generator output: {test_sr_gan.shape}")
print(f"SRGAN Discriminator output: {test_disc_out.shape}")

# 6. LOSS FUNCTIONS FOR SUPER-RESOLUTION
print("\n6. LOSS FUNCTIONS FOR SUPER-RESOLUTION")

class SRLoss:
    """Loss functions for super-resolution"""
    
    @staticmethod
    def mse_loss(sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """Mean Squared Error loss"""
        return F.mse_loss(sr, hr)
    
    @staticmethod
    def l1_loss(sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """L1 loss (Mean Absolute Error)"""
        return F.l1_loss(sr, hr)
    
    @staticmethod
    def charbonnier_loss(sr: torch.Tensor, hr: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
        """Charbonnier loss (smooth L1)"""
        diff = sr - hr
        return torch.mean(torch.sqrt(diff ** 2 + epsilon ** 2))
    
    @staticmethod
    def ssim_loss(sr: torch.Tensor, hr: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        """Structural Similarity Index loss (simplified)"""
        # Simplified SSIM implementation
        mu1 = F.avg_pool2d(sr, window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool2d(hr, window_size, stride=1, padding=window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(sr * sr, window_size, stride=1, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(hr * hr, window_size, stride=1, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(sr * hr, window_size, stride=1, padding=window_size//2) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    
    def __init__(self, layer_weights: Dict[str, float] = None):
        super(PerceptualLoss, self).__init__()
        
        if layer_weights is None:
            layer_weights = {'conv1_2': 0.1, 'conv2_2': 0.1, 'conv3_3': 1.0, 'conv4_3': 1.0, 'conv5_3': 1.0}
        
        self.layer_weights = layer_weights
        
        # Load pretrained VGG19
        vgg = torchvision.models.vgg19(pretrained=True).features
        
        # Extract relevant layers
        self.vgg_layers = {}
        layer_names = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3', 'conv5_3']
        layer_indices = [3, 8, 17, 26, 35]
        
        current_layer = 0
        for i, (name, idx) in enumerate(zip(layer_names, layer_indices)):
            layers = nn.Sequential()
            for j in range(current_layer, idx + 1):
                layers.add_module(str(j), vgg[j])
            self.vgg_layers[name] = layers
            current_layer = idx + 1
        
        # Freeze VGG parameters
        for layer in self.vgg_layers.values():
            for param in layer.parameters():
                param.requires_grad = False
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input for VGG"""
        return (x - self.mean) / self.std
    
    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss"""
        sr_norm = self.normalize(sr)
        hr_norm = self.normalize(hr)
        
        total_loss = 0
        
        for layer_name, weight in self.layer_weights.items():
            sr_features = self.vgg_layers[layer_name](sr_norm)
            hr_features = self.vgg_layers[layer_name](hr_norm)
            
            loss = F.mse_loss(sr_features, hr_features)
            total_loss += weight * loss
        
        return total_loss

# Test loss functions
dummy_sr = torch.rand(1, 3, 128, 128)
dummy_hr = torch.rand(1, 3, 128, 128)

mse_loss = SRLoss.mse_loss(dummy_sr, dummy_hr)
l1_loss = SRLoss.l1_loss(dummy_sr, dummy_hr)
charbonnier_loss = SRLoss.charbonnier_loss(dummy_sr, dummy_hr)
ssim_loss = SRLoss.ssim_loss(dummy_sr, dummy_hr)

print(f"MSE loss: {mse_loss.item():.6f}")
print(f"L1 loss: {l1_loss.item():.6f}")
print(f"Charbonnier loss: {charbonnier_loss.item():.6f}")
print(f"SSIM loss: {ssim_loss.item():.6f}")

# Test perceptual loss
perceptual_loss = PerceptualLoss()
perc_loss = perceptual_loss(dummy_sr, dummy_hr)
print(f"Perceptual loss: {perc_loss.item():.6f}")

# 7. TRAINING UTILITIES
print("\n7. TRAINING UTILITIES")

class SRTrainer:
    """Training class for super-resolution models"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu', loss_type: str = 'l1'):
        self.model = model.to(device)
        self.device = device
        self.loss_type = loss_type
        
        # Initialize loss function
        if loss_type == 'mse':
            self.criterion = SRLoss.mse_loss
        elif loss_type == 'l1':
            self.criterion = SRLoss.l1_loss
        elif loss_type == 'charbonnier':
            self.criterion = SRLoss.charbonnier_loss
        elif loss_type == 'perceptual':
            self.criterion = PerceptualLoss().to(device)
        else:
            self.criterion = SRLoss.l1_loss
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Training history
        self.train_losses = []
        
    def train_step(self, lr_batch: torch.Tensor, hr_batch: torch.Tensor) -> float:
        """Single training step"""
        lr_batch = lr_batch.to(self.device)
        hr_batch = hr_batch.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass
        sr_batch = self.model(lr_batch)
        
        # Compute loss
        if self.loss_type == 'perceptual':
            loss = self.criterion(sr_batch, hr_batch)
        else:
            loss = self.criterion(sr_batch, hr_batch)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (lr_batch, hr_batch) in enumerate(dataloader):
            loss = self.train_step(lr_batch, hr_batch)
            epoch_loss += loss
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}: Loss: {loss:.6f}')
        
        avg_loss = epoch_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        
        total_psnr = 0.0
        total_ssim = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for lr_batch, hr_batch in dataloader:
                lr_batch = lr_batch.to(self.device)
                hr_batch = hr_batch.to(self.device)
                
                sr_batch = self.model(lr_batch)
                
                # Calculate metrics
                psnr = self.calculate_psnr(sr_batch, hr_batch)
                ssim = self.calculate_ssim(sr_batch, hr_batch)
                
                total_psnr += psnr * lr_batch.size(0)
                total_ssim += ssim * lr_batch.size(0)
                num_samples += lr_batch.size(0)
        
        return {
            'psnr': total_psnr / num_samples,
            'ssim': total_ssim / num_samples
        }
    
    def calculate_psnr(self, sr: torch.Tensor, hr: torch.Tensor) -> float:
        """Calculate PSNR (Peak Signal-to-Noise Ratio)"""
        mse = F.mse_loss(sr, hr)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr.item()
    
    def calculate_ssim(self, sr: torch.Tensor, hr: torch.Tensor) -> float:
        """Calculate SSIM (simplified)"""
        ssim_loss = SRLoss.ssim_loss(sr, hr)
        return (1 - ssim_loss).item()

# 8. DATASET FOR SUPER-RESOLUTION
print("\n8. DATASET FOR SUPER-RESOLUTION")

class SRDataset(Dataset):
    """Super-resolution dataset"""
    
    def __init__(self, size: int = 1000, hr_size: int = 128, scale_factor: int = 2):
        self.size = size
        self.hr_size = hr_size
        self.lr_size = hr_size // scale_factor
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Generate high-resolution image
        hr_image = torch.rand(3, self.hr_size, self.hr_size)
        
        # Generate low-resolution image by downsampling
        lr_image = F.interpolate(hr_image.unsqueeze(0), size=(self.lr_size, self.lr_size), 
                               mode='bicubic', align_corners=False).squeeze(0)
        
        return lr_image, hr_image

# 9. TRAINING EXAMPLE
print("\n9. TRAINING EXAMPLE")

# Create dataset and dataloader
sr_dataset = SRDataset(size=200, hr_size=128, scale_factor=2)
sr_dataloader = DataLoader(sr_dataset, batch_size=4, shuffle=True)

# Initialize trainer with SRCNN
sr_trainer = SRTrainer(SRCNN(upscale_factor=2), device='cpu', loss_type='l1')

print("Training SRCNN for super-resolution...")
for epoch in range(2):
    avg_loss = sr_trainer.train_epoch(sr_dataloader)
    print(f'Epoch {epoch+1}: Average Loss: {avg_loss:.6f}')

# Evaluate model
val_dataset = SRDataset(size=50, hr_size=128, scale_factor=2)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

metrics = sr_trainer.evaluate(val_dataloader)
print(f"Validation PSNR: {metrics['psnr']:.2f} dB")
print(f"Validation SSIM: {metrics['ssim']:.4f}")

# 10. MULTI-SCALE SUPER-RESOLUTION
print("\n10. MULTI-SCALE SUPER-RESOLUTION")

class MultiScaleSR(nn.Module):
    """Multi-scale super-resolution network"""
    
    def __init__(self, num_channels: int = 3, num_features: int = 64):
        super(MultiScaleSR, self).__init__()
        
        # Shared feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            *[ResidualBlock(num_features, use_batch_norm=False) for _ in range(4)]
        )
        
        # Scale-specific heads
        self.scale_2x = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
            PixelShuffle(2),
            nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)
        )
        
        self.scale_4x = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
            PixelShuffle(2),
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
            PixelShuffle(2),
            nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)
        )
        
        self.scale_8x = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
            PixelShuffle(2),
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
            PixelShuffle(2),
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
            PixelShuffle(2),
            nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, scale: int = 2) -> torch.Tensor:
        # Extract features
        features = self.feature_extractor(x)
        
        # Apply scale-specific head
        if scale == 2:
            return torch.sigmoid(self.scale_2x(features))
        elif scale == 4:
            return torch.sigmoid(self.scale_4x(features))
        elif scale == 8:
            return torch.sigmoid(self.scale_8x(features))
        else:
            raise ValueError(f"Unsupported scale factor: {scale}")

# Test multi-scale SR
multi_scale_sr = MultiScaleSR()
test_input = torch.rand(1, 3, 32, 32)

for scale in [2, 4, 8]:
    output = multi_scale_sr(test_input, scale=scale)
    print(f"Multi-scale SR {scale}x output: {output.shape}")

print("\n=== IMAGE SUPER-RESOLUTION COMPLETE ===")
print("Key concepts covered:")
print("- Basic SRCNN implementation")
print("- Efficient sub-pixel convolution (pixel shuffle)")
print("- Residual blocks for deep networks")
print("- Dense blocks (RDN-style architecture)")
print("- Generative adversarial networks (SRGAN)")
print("- Multiple loss functions (MSE, L1, Charbonnier, SSIM, Perceptual)")
print("- Training utilities and evaluation metrics")
print("- Super-resolution datasets")
print("- Multi-scale super-resolution")
print("- PSNR and SSIM evaluation metrics")