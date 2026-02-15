"""
PyTorch Neural Style Transfer - Artistic Style Transfer Implementation
Comprehensive guide to implementing neural style transfer in PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, List, Optional, Dict, Union
import copy

print("=== NEURAL STYLE TRANSFER ===")

# 1. VGG FEATURE EXTRACTOR
print("\n1. VGG FEATURE EXTRACTOR")

class VGGFeatureExtractor(nn.Module):
    """VGG19 feature extractor for style transfer"""
    
    def __init__(self, layer_names: List[str] = None):
        super(VGGFeatureExtractor, self).__init__()
        
        if layer_names is None:
            # Default layers for content and style
            layer_names = [
                'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1'
            ]
        
        self.layer_names = layer_names
        
        # Load pretrained VGG19
        vgg = models.vgg19(pretrained=True).features
        
        # Create feature extractor
        self.features = nn.Sequential()
        
        # VGG layer mapping
        vgg_layers = {
            'conv1_1': 0, 'conv1_2': 2,
            'conv2_1': 5, 'conv2_2': 7,
            'conv3_1': 10, 'conv3_2': 12, 'conv3_3': 14, 'conv3_4': 16,
            'conv4_1': 19, 'conv4_2': 21, 'conv4_3': 23, 'conv4_4': 25,
            'conv5_1': 28, 'conv5_2': 30, 'conv5_3': 32, 'conv5_4': 34
        }
        
        # Add layers up to the deepest required layer
        max_layer = max([vgg_layers[name] for name in layer_names])
        
        for i in range(max_layer + 1):
            self.features.add_module(str(i), vgg[i])
        
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Store layer indices
        self.layer_indices = {name: vgg_layers[name] for name in layer_names}
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from specified layers"""
        features = {}
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            # Check if this layer output should be saved
            for name, idx in self.layer_indices.items():
                if i == idx:
                    features[name] = x
        
        return features

# Test feature extractor
feature_extractor = VGGFeatureExtractor()
test_input = torch.randn(1, 3, 224, 224)
features = feature_extractor(test_input)

print("VGG feature shapes:")
for name, feature in features.items():
    print(f"  {name}: {feature.shape}")

# 2. LOSS FUNCTIONS FOR STYLE TRANSFER
print("\n2. LOSS FUNCTIONS FOR STYLE TRANSFER")

class StyleTransferLoss:
    """Loss functions for neural style transfer"""
    
    @staticmethod
    def content_loss(generated_features: torch.Tensor, content_features: torch.Tensor) -> torch.Tensor:
        """Content loss (MSE between feature maps)"""
        return F.mse_loss(generated_features, content_features)
    
    @staticmethod
    def gram_matrix(features: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix for style representation"""
        batch_size, channels, height, width = features.size()
        
        # Reshape features to (batch_size, channels, height*width)
        features = features.view(batch_size, channels, height * width)
        
        # Compute Gram matrix
        gram = torch.bmm(features, features.transpose(1, 2))
        
        # Normalize by number of elements
        gram = gram / (channels * height * width)
        
        return gram
    
    @staticmethod
    def style_loss(generated_features: torch.Tensor, style_features: torch.Tensor) -> torch.Tensor:
        """Style loss (MSE between Gram matrices)"""
        generated_gram = StyleTransferLoss.gram_matrix(generated_features)
        style_gram = StyleTransferLoss.gram_matrix(style_features)
        
        return F.mse_loss(generated_gram, style_gram)
    
    @staticmethod
    def total_variation_loss(generated_image: torch.Tensor) -> torch.Tensor:
        """Total variation loss for smoothness"""
        batch_size, channels, height, width = generated_image.size()
        
        # Horizontal total variation
        tv_h = torch.mean(torch.abs(generated_image[:, :, 1:, :] - generated_image[:, :, :-1, :]))
        
        # Vertical total variation
        tv_w = torch.mean(torch.abs(generated_image[:, :, :, 1:] - generated_image[:, :, :, :-1]))
        
        return tv_h + tv_w

# Test loss functions
dummy_features = torch.randn(1, 512, 28, 28)
dummy_content = torch.randn(1, 512, 28, 28)
dummy_style = torch.randn(1, 512, 28, 28)
dummy_image = torch.randn(1, 3, 224, 224)

content_loss = StyleTransferLoss.content_loss(dummy_features, dummy_content)
style_loss = StyleTransferLoss.style_loss(dummy_features, dummy_style)
tv_loss = StyleTransferLoss.total_variation_loss(dummy_image)

print(f"Content loss: {content_loss.item():.6f}")
print(f"Style loss: {style_loss.item():.6f}")
print(f"Total variation loss: {tv_loss.item():.6f}")

# Test Gram matrix
gram = StyleTransferLoss.gram_matrix(dummy_features)
print(f"Gram matrix shape: {gram.shape}")

# 3. GATYS STYLE TRANSFER (OPTIMIZATION-BASED)
print("\n3. GATYS STYLE TRANSFER (OPTIMIZATION-BASED)")

class GatysStyleTransfer:
    """Original Gatys et al. style transfer implementation"""
    
    def __init__(self, content_layers: List[str] = ['conv4_2'], 
                 style_layers: List[str] = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
                 device: str = 'cpu'):
        
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.device = device
        
        # Feature extractor
        all_layers = list(set(content_layers + style_layers))
        self.feature_extractor = VGGFeatureExtractor(all_layers).to(device)
        
        # Loss weights
        self.content_weight = 1.0
        self.style_weight = 1000.0
        self.tv_weight = 1e-4
        
    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """Preprocess image for VGG"""
        # Normalize using ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        return (image - mean) / std
    
    def deprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """Deprocess image from VGG format"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        image = image * std + mean
        return torch.clamp(image, 0, 1)
    
    def extract_features(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract content and style features"""
        preprocessed = self.preprocess_image(image)
        return self.feature_extractor(preprocessed)
    
    def compute_loss(self, generated_image: torch.Tensor, 
                    content_features: Dict[str, torch.Tensor],
                    style_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute total loss for style transfer"""
        
        # Extract features from generated image
        generated_features = self.extract_features(generated_image)
        
        # Content loss
        content_loss = 0
        for layer in self.content_layers:
            content_loss += StyleTransferLoss.content_loss(
                generated_features[layer], content_features[layer]
            )
        
        # Style loss
        style_loss = 0
        for layer in self.style_layers:
            style_loss += StyleTransferLoss.style_loss(
                generated_features[layer], style_features[layer]
            )
        
        # Total variation loss
        tv_loss = StyleTransferLoss.total_variation_loss(generated_image)
        
        # Weighted total loss
        total_loss = (self.content_weight * content_loss + 
                     self.style_weight * style_loss + 
                     self.tv_weight * tv_loss)
        
        return {
            'total_loss': total_loss,
            'content_loss': content_loss,
            'style_loss': style_loss,
            'tv_loss': tv_loss
        }
    
    def transfer_style(self, content_image: torch.Tensor, style_image: torch.Tensor,
                      num_steps: int = 300, lr: float = 0.01) -> torch.Tensor:
        """Perform style transfer optimization"""
        
        # Move images to device
        content_image = content_image.to(self.device)
        style_image = style_image.to(self.device)
        
        # Extract target features
        content_features = self.extract_features(content_image)
        style_features = self.extract_features(style_image)
        
        # Initialize generated image with content image
        generated_image = content_image.clone().requires_grad_(True)
        
        # Optimizer
        optimizer = optim.Adam([generated_image], lr=lr)
        
        print(f"Starting style transfer optimization for {num_steps} steps...")
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Compute loss
            losses = self.compute_loss(generated_image, content_features, style_features)
            
            # Backward pass
            losses['total_loss'].backward()
            optimizer.step()
            
            # Clamp values to valid range
            with torch.no_grad():
                generated_image.clamp_(0, 1)
            
            if step % 50 == 0:
                print(f"Step {step}: Total Loss: {losses['total_loss'].item():.4f}, "
                      f"Content: {losses['content_loss'].item():.4f}, "
                      f"Style: {losses['style_loss'].item():.4f}")
        
        return generated_image.detach()

# Test Gatys style transfer (simplified)
print("Testing Gatys style transfer...")
gatys_transfer = GatysStyleTransfer(device='cpu')

# Create dummy images
dummy_content = torch.rand(1, 3, 224, 224)
dummy_style = torch.rand(1, 3, 224, 224)

# Quick test with fewer steps
result = gatys_transfer.transfer_style(dummy_content, dummy_style, num_steps=10, lr=0.1)
print(f"Style transfer result shape: {result.shape}")

# 4. FAST NEURAL STYLE TRANSFER
print("\n4. FAST NEURAL STYLE TRANSFER")

class TransformNet(nn.Module):
    """Transformation network for fast style transfer"""
    
    def __init__(self, in_channels: int = 3):
        super(TransformNet, self).__init__()
        
        # Initial convolution layers
        self.conv1 = ConvLayer(in_channels, 32, kernel_size=9, stride=1)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        
        # Upsampling layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.deconv3 = ConvLayer(32, in_channels, kernel_size=9, stride=1, normalize=False, relu=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.deconv1(y)
        y = self.deconv2(y)
        y = self.deconv3(y)
        
        return torch.sigmoid(y)  # Output in [0, 1] range

class ConvLayer(nn.Module):
    """Convolution layer with instance normalization"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, normalize: bool = True, relu: bool = True):
        super(ConvLayer, self).__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
        self.normalize = normalize
        self.relu = relu
        
        if normalize:
            self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        
        if relu:
            self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        
        if self.normalize:
            y = self.norm(y)
        
        if self.relu:
            y = self.activation(y)
        
        return y

class ResidualBlock(nn.Module):
    """Residual block for transformation network"""
    
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, relu=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.conv1(x)
        y = self.conv2(y)
        return y + residual

class UpsampleConvLayer(nn.Module):
    """Upsampling convolution layer"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int, upsample: int = None):
        super(UpsampleConvLayer, self).__init__()
        
        self.upsample = upsample
        
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=upsample, mode='nearest')
        
        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = self.upsample_layer(x)
        return self.conv(x)

# Test transformation network
transform_net = TransformNet(in_channels=3)
test_input = torch.randn(1, 3, 256, 256)
transformed = transform_net(test_input)

print(f"Transform network input: {test_input.shape}")
print(f"Transform network output: {transformed.shape}")

# Count parameters
total_params = sum(p.numel() for p in transform_net.parameters())
print(f"Transform network parameters: {total_params:,}")

# 5. PERCEPTUAL LOSS NETWORK
print("\n5. PERCEPTUAL LOSS NETWORK")

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    
    def __init__(self, content_layers: List[str] = ['conv4_2'],
                 style_layers: List[str] = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1'],
                 content_weight: float = 1.0, style_weight: float = 1000.0):
        super(PerceptualLoss, self).__init__()
        
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_weight = content_weight
        self.style_weight = style_weight
        
        # Feature extractor
        all_layers = list(set(content_layers + style_layers))
        self.feature_extractor = VGGFeatureExtractor(all_layers)
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input for VGG"""
        return (x - self.mean) / self.std
    
    def forward(self, generated: torch.Tensor, target: torch.Tensor, 
                style_target: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Compute perceptual loss"""
        
        # Extract features
        generated_features = self.feature_extractor(self.normalize(generated))
        target_features = self.feature_extractor(self.normalize(target))
        
        # Content loss
        content_loss = 0
        for layer in self.content_layers:
            content_loss += F.mse_loss(generated_features[layer], target_features[layer])
        
        # Style loss
        style_loss = 0
        if style_target is not None:
            style_features = self.feature_extractor(self.normalize(style_target))
            
            for layer in self.style_layers:
                generated_gram = StyleTransferLoss.gram_matrix(generated_features[layer])
                style_gram = StyleTransferLoss.gram_matrix(style_features[layer])
                style_loss += F.mse_loss(generated_gram, style_gram)
        
        # Total loss
        total_loss = self.content_weight * content_loss + self.style_weight * style_loss
        
        return {
            'total_loss': total_loss,
            'content_loss': content_loss,
            'style_loss': style_loss
        }

# Test perceptual loss
perceptual_loss = PerceptualLoss()

dummy_generated = torch.rand(1, 3, 224, 224)
dummy_target = torch.rand(1, 3, 224, 224)
dummy_style_target = torch.rand(1, 3, 224, 224)

losses = perceptual_loss(dummy_generated, dummy_target, dummy_style_target)
print("Perceptual losses:")
for loss_name, loss_value in losses.items():
    print(f"  {loss_name}: {loss_value.item():.6f}")

# 6. FAST STYLE TRANSFER TRAINER
print("\n6. FAST STYLE TRANSFER TRAINER")

class FastStyleTransferTrainer:
    """Trainer for fast neural style transfer"""
    
    def __init__(self, style_image: torch.Tensor, device: str = 'cpu'):
        self.device = device
        self.style_image = style_image.to(device)
        
        # Initialize transformation network
        self.transform_net = TransformNet().to(device)
        
        # Initialize perceptual loss
        self.perceptual_loss = PerceptualLoss(
            content_weight=1.0,
            style_weight=10.0
        ).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.transform_net.parameters(), lr=1e-3)
        
        # Training history
        self.train_losses = []
        
    def train_step(self, content_batch: torch.Tensor) -> Dict[str, float]:
        """Single training step"""
        content_batch = content_batch.to(self.device)
        batch_size = content_batch.size(0)
        
        # Generate stylized images
        stylized_batch = self.transform_net(content_batch)
        
        # Expand style image to match batch size
        style_batch = self.style_image.expand(batch_size, -1, -1, -1)
        
        # Compute perceptual loss
        losses = self.perceptual_loss(stylized_batch, content_batch, style_batch)
        
        # Add total variation loss for smoothness
        tv_loss = StyleTransferLoss.total_variation_loss(stylized_batch)
        total_loss = losses['total_loss'] + 1e-6 * tv_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'content_loss': losses['content_loss'].item(),
            'style_loss': losses['style_loss'].item(),
            'tv_loss': tv_loss.item()
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.transform_net.train()
        
        epoch_losses = {'total_loss': 0, 'content_loss': 0, 'style_loss': 0, 'tv_loss': 0}
        num_batches = 0
        
        for batch_idx, (content_batch, _) in enumerate(dataloader):
            losses = self.train_step(content_batch)
            
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}: Total: {losses["total_loss"]:.4f}, '
                      f'Content: {losses["content_loss"]:.4f}, '
                      f'Style: {losses["style_loss"]:.4f}')
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        self.train_losses.append(epoch_losses['total_loss'])
        
        return epoch_losses
    
    def stylize(self, content_image: torch.Tensor) -> torch.Tensor:
        """Apply style transfer to content image"""
        self.transform_net.eval()
        
        with torch.no_grad():
            content_image = content_image.to(self.device)
            stylized = self.transform_net(content_image)
        
        return stylized

# 7. REAL-TIME STYLE TRANSFER
print("\n7. REAL-TIME STYLE TRANSFER")

class RealTimeStyleTransfer:
    """Real-time style transfer for video/webcam"""
    
    def __init__(self, model_path: str = None, device: str = 'cpu'):
        self.device = device
        self.transform_net = TransformNet().to(device)
        
        if model_path:
            self.load_model(model_path)
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        self.postprocess = transforms.ToPILImage()
    
    def load_model(self, model_path: str):
        """Load pretrained transformation network"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.transform_net.load_state_dict(checkpoint)
        self.transform_net.eval()
    
    def save_model(self, model_path: str):
        """Save transformation network"""
        torch.save(self.transform_net.state_dict(), model_path)
    
    def process_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Process a single frame"""
        with torch.no_grad():
            frame = frame.to(self.device)
            stylized = self.transform_net(frame)
        return stylized
    
    def process_batch(self, frames: torch.Tensor) -> torch.Tensor:
        """Process batch of frames"""
        with torch.no_grad():
            frames = frames.to(self.device)
            stylized = self.transform_net(frames)
        return stylized

# Test real-time style transfer
rt_style_transfer = RealTimeStyleTransfer(device='cpu')

# Process single frame
test_frame = torch.rand(1, 3, 256, 256)
stylized_frame = rt_style_transfer.process_frame(test_frame)
print(f"Real-time stylized frame shape: {stylized_frame.shape}")

# Process batch of frames
test_batch = torch.rand(4, 3, 256, 256)
stylized_batch = rt_style_transfer.process_batch(test_batch)
print(f"Real-time stylized batch shape: {stylized_batch.shape}")

# 8. MULTI-STYLE TRANSFER
print("\n8. MULTI-STYLE TRANSFER")

class MultiStyleTransformNet(nn.Module):
    """Multi-style transformation network"""
    
    def __init__(self, num_styles: int = 4):
        super(MultiStyleTransformNet, self).__init__()
        
        self.num_styles = num_styles
        
        # Shared encoder
        self.encoder = nn.Sequential(
            ConvLayer(3, 32, kernel_size=9, stride=1),
            ConvLayer(32, 64, kernel_size=3, stride=2),
            ConvLayer(64, 128, kernel_size=3, stride=2)
        )
        
        # Style-specific residual blocks
        self.style_residuals = nn.ModuleList([
            nn.Sequential(*[ResidualBlock(128) for _ in range(5)])
            for _ in range(num_styles)
        ])
        
        # Shared decoder
        self.decoder = nn.Sequential(
            UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2),
            UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2),
            ConvLayer(32, 3, kernel_size=9, stride=1, normalize=False, relu=False)
        )
    
    def forward(self, x: torch.Tensor, style_id: int) -> torch.Tensor:
        """Forward pass with specific style"""
        # Encode
        features = self.encoder(x)
        
        # Apply style-specific residual blocks
        features = self.style_residuals[style_id](features)
        
        # Decode
        output = self.decoder(features)
        
        return torch.sigmoid(output)
    
    def forward_interpolate(self, x: torch.Tensor, style_weights: torch.Tensor) -> torch.Tensor:
        """Forward pass with style interpolation"""
        # Encode
        features = self.encoder(x)
        
        # Weighted combination of style residuals
        styled_features = 0
        for i, weight in enumerate(style_weights):
            if weight > 0:
                styled_features += weight * self.style_residuals[i](features)
        
        # Decode
        output = self.decoder(styled_features)
        
        return torch.sigmoid(output)

# Test multi-style network
multi_style_net = MultiStyleTransformNet(num_styles=4)
test_input = torch.rand(1, 3, 256, 256)

# Style transfer with specific style
stylized_0 = multi_style_net(test_input, style_id=0)
print(f"Multi-style output (style 0): {stylized_0.shape}")

# Style interpolation
style_weights = torch.tensor([0.5, 0.3, 0.2, 0.0])
interpolated = multi_style_net.forward_interpolate(test_input, style_weights)
print(f"Multi-style interpolated output: {interpolated.shape}")

# 9. TRAINING EXAMPLE
print("\n9. TRAINING EXAMPLE")

class DummyImageDataset(Dataset):
    """Dummy dataset for style transfer training"""
    
    def __init__(self, size: int = 1000, img_size: int = 256):
        self.size = size
        self.img_size = img_size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Generate random content image
        img = torch.rand(3, self.img_size, self.img_size)
        return img, 0  # Dummy label

# Training example
print("Training fast style transfer...")

# Create dummy style image
dummy_style = torch.rand(1, 3, 224, 224)

# Create trainer
trainer = FastStyleTransferTrainer(dummy_style, device='cpu')

# Create dataset
dummy_dataset = DummyImageDataset(size=100, img_size=256)
dummy_loader = DataLoader(dummy_dataset, batch_size=4, shuffle=True)

# Train for a few epochs
for epoch in range(2):
    losses = trainer.train_epoch(dummy_loader)
    print(f'Epoch {epoch+1}: Total: {losses["total_loss"]:.4f}, '
          f'Content: {losses["content_loss"]:.4f}, '
          f'Style: {losses["style_loss"]:.4f}')

# Test stylization
test_content = torch.rand(1, 3, 256, 256)
stylized_result = trainer.stylize(test_content)
print(f"Stylized result shape: {stylized_result.shape}")

# 10. STYLE TRANSFER UTILITIES
print("\n10. STYLE TRANSFER UTILITIES")

class StyleTransferUtils:
    """Utility functions for style transfer"""
    
    @staticmethod
    def resize_to_match(source: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """Resize image to match target size"""
        return F.interpolate(source, size=target_size, mode='bilinear', align_corners=False)
    
    @staticmethod
    def prepare_image(image: torch.Tensor, size: int = 512) -> torch.Tensor:
        """Prepare image for style transfer"""
        # Resize to target size while maintaining aspect ratio
        _, _, h, w = image.shape
        
        if h > w:
            new_h, new_w = size, int(size * w / h)
        else:
            new_h, new_w = int(size * h / w), size
        
        resized = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        # Pad to square if needed
        pad_h = (size - new_h) // 2
        pad_w = (size - new_w) // 2
        
        padded = F.pad(resized, (pad_w, pad_w, pad_h, pad_h))
        
        return padded
    
    @staticmethod
    def blend_images(image1: torch.Tensor, image2: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
        """Blend two images"""
        return alpha * image1 + (1 - alpha) * image2
    
    @staticmethod
    def apply_color_transfer(content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Transfer color statistics from style to content"""
        # Convert to YUV color space for better color transfer
        # This is a simplified version
        
        # Calculate mean and std for each channel
        content_mean = content.mean(dim=(2, 3), keepdim=True)
        content_std = content.std(dim=(2, 3), keepdim=True)
        
        style_mean = style.mean(dim=(2, 3), keepdim=True)
        style_std = style.std(dim=(2, 3), keepdim=True)
        
        # Normalize content and apply style statistics
        normalized = (content - content_mean) / (content_std + 1e-8)
        transferred = normalized * style_std + style_mean
        
        return torch.clamp(transferred, 0, 1)

# Test utilities
test_img1 = torch.rand(1, 3, 400, 300)
test_img2 = torch.rand(1, 3, 300, 400)

prepared = StyleTransferUtils.prepare_image(test_img1, size=512)
print(f"Prepared image shape: {prepared.shape}")

blended = StyleTransferUtils.blend_images(test_img1[:, :, :300, :300], test_img2[:, :, :300, :300], alpha=0.7)
print(f"Blended image shape: {blended.shape}")

color_transferred = StyleTransferUtils.apply_color_transfer(test_img1, test_img2)
print(f"Color transferred shape: {color_transferred.shape}")

print("\n=== NEURAL STYLE TRANSFER COMPLETE ===")
print("Key concepts covered:")
print("- VGG feature extractor for perceptual features")
print("- Style transfer loss functions (content, style, TV)")
print("- Gatys optimization-based style transfer")
print("- Fast neural style transfer with transformation networks")
print("- Perceptual loss for training feed-forward networks")
print("- Real-time style transfer implementation")
print("- Multi-style transfer and style interpolation")
print("- Training pipeline and utilities")
print("- Color transfer and image blending techniques")