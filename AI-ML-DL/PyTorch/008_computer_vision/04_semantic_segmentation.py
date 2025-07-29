"""
PyTorch Semantic Segmentation - U-Net, FCN and Advanced Architectures
Comprehensive guide to semantic segmentation implementation in PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import math

print("=== SEMANTIC SEGMENTATION ===")

# 1. BASIC SEGMENTATION BUILDING BLOCKS
print("\n1. BASIC SEGMENTATION BUILDING BLOCKS")

class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 use_batchnorm: bool = True, dropout: float = 0.0):
        super(DoubleConv, self).__init__()
        
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
            
        layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        self.double_conv = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class DownBlock(nn.Module):
    """Downsampling block for encoder"""
    
    def __init__(self, in_channels: int, out_channels: int, use_pooling: bool = True):
        super(DownBlock, self).__init__()
        
        if use_pooling:
            self.downsample = nn.MaxPool2d(2)
        else:
            self.downsample = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)
            
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        return self.conv(x)

class UpBlock(nn.Module):
    """Upsampling block for decoder"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 upsample_mode: str = 'transpose'):
        super(UpBlock, self).__init__()
        
        if upsample_mode == 'transpose':
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        elif upsample_mode == 'bilinear':
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            raise ValueError(f"Unknown upsample mode: {upsample_mode}")
            
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        
        # Handle size mismatch
        if x.shape != skip_connection.shape:
            diffY = skip_connection.size()[2] - x.size()[2]
            diffX = skip_connection.size()[3] - x.size()[3]
            
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([skip_connection, x], dim=1)
        return self.conv(x)

# Test building blocks
test_input = torch.randn(2, 3, 64, 64)
double_conv = DoubleConv(3, 64)
down_block = DownBlock(64, 128)
up_block = UpBlock(128, 64)

print(f"Input shape: {test_input.shape}")
conv_out = double_conv(test_input)
print(f"DoubleConv output: {conv_out.shape}")
down_out = down_block(conv_out)
print(f"DownBlock output: {down_out.shape}")
up_out = up_block(down_out, conv_out)
print(f"UpBlock output: {up_out.shape}")

# 2. U-NET IMPLEMENTATION
print("\n2. U-NET IMPLEMENTATION")

class UNet(nn.Module):
    """U-Net architecture for semantic segmentation"""
    
    def __init__(self, in_channels: int = 3, num_classes: int = 21, 
                 features: List[int] = [64, 128, 256, 512],
                 dropout: float = 0.2):
        super(UNet, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = features
        
        # Encoder (downsampling)
        self.encoder = nn.ModuleList()
        self.encoder.append(DoubleConv(in_channels, features[0], dropout=dropout))
        
        for i in range(len(features) - 1):
            self.encoder.append(DownBlock(features[i], features[i + 1]))
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout=dropout)
        
        # Decoder (upsampling)
        self.decoder = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.decoder.append(UpBlock(features[i] * 2, features[i - 1]))
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], num_classes, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store skip connections
        skip_connections = []
        
        # Encoder path
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        skip_connections = skip_connections[::-1]  # Reverse order
        
        for i, layer in enumerate(self.decoder):
            skip_connection = skip_connections[i]
            x = layer(x, skip_connection)
        
        # Final prediction
        return self.final_conv(x)

# Test U-Net
unet = UNet(in_channels=3, num_classes=21, features=[64, 128, 256, 512])
test_input = torch.randn(2, 3, 256, 256)
output = unet(test_input)
print(f"U-Net input: {test_input.shape}")
print(f"U-Net output: {output.shape}")

# Calculate parameters
total_params = sum(p.numel() for p in unet.parameters())
print(f"U-Net parameters: {total_params:,}")

# 3. FCN (FULLY CONVOLUTIONAL NETWORK)
print("\n3. FCN (FULLY CONVOLUTIONAL NETWORK)")

class FCN(nn.Module):
    """Fully Convolutional Network for semantic segmentation"""
    
    def __init__(self, num_classes: int = 21, backbone: str = 'vgg16'):
        super(FCN, self).__init__()
        
        self.num_classes = num_classes
        
        if backbone == 'vgg16':
            # Load VGG16 backbone
            vgg = torchvision.models.vgg16(pretrained=True)
            
            # Extract features (remove classifier)
            self.features = vgg.features
            
            # Replace final maxpool with conv for spatial preservation
            self.features[30] = nn.Conv2d(512, 512, 3, padding=1)  # Replace last maxpool
            
            # Add classifier as convolutional layers
            self.classifier = nn.Sequential(
                nn.Conv2d(512, 4096, 7),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.5),
                nn.Conv2d(4096, 4096, 1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.5),
                nn.Conv2d(4096, num_classes, 1)
            )
            
            # Upsampling layers
            self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8, padding=4)
            
        else:
            raise ValueError(f"Backbone {backbone} not implemented")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        x = self.features(x)
        
        # Classify
        x = self.classifier(x)
        
        # Upsample to original size
        x = self.upsample_8x(x)
        
        return x

class FCNWithSkips(nn.Module):
    """FCN with skip connections (FCN-8s style)"""
    
    def __init__(self, num_classes: int = 21):
        super(FCNWithSkips, self).__init__()
        
        # VGG16 backbone
        vgg = torchvision.models.vgg16(pretrained=True)
        
        # Split VGG into different stages
        self.stage1 = nn.Sequential(*list(vgg.features.children())[:17])  # pool3
        self.stage2 = nn.Sequential(*list(vgg.features.children())[17:24])  # pool4
        self.stage3 = nn.Sequential(*list(vgg.features.children())[24:31])  # pool5
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(4096, num_classes, 1)
        )
        
        # Skip connection classifiers
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)
        
        # Upsampling layers
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, padding=1)
        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8, padding=4)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1: up to pool3
        pool3 = self.stage1(x)
        
        # Stage 2: up to pool4
        pool4 = self.stage2(pool3)
        
        # Stage 3: up to pool5
        pool5 = self.stage3(pool4)
        
        # Main classifier
        score = self.classifier(pool5)
        
        # Upsample and add skip connections
        upscore2 = self.upsample_2x(score)
        score_pool4c = self.score_pool4(pool4)
        
        # Add pool4 skip connection
        fuse_pool4 = upscore2 + score_pool4c
        
        # Upsample again
        upscore_pool4 = self.upsample_2x(fuse_pool4)
        score_pool3c = self.score_pool3(pool3)
        
        # Add pool3 skip connection
        fuse_pool3 = upscore_pool4 + score_pool3c
        
        # Final upsampling
        output = self.upsample_8x(fuse_pool3)
        
        return output

# Test FCN models
fcn_simple = FCN(num_classes=21)
fcn_skips = FCNWithSkips(num_classes=21)

test_input = torch.randn(1, 3, 224, 224)
fcn_output = fcn_simple(test_input)
fcn_skips_output = fcn_skips(test_input)

print(f"FCN simple output: {fcn_output.shape}")
print(f"FCN with skips output: {fcn_skips_output.shape}")

# 4. ADVANCED SEGMENTATION ARCHITECTURES
print("\n4. ADVANCED SEGMENTATION ARCHITECTURES")

class ResidualBlock(nn.Module):
    """Residual block for segmentation networks"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        return F.relu(out)

class AttentionGate(nn.Module):
    """Attention gate for feature refinement"""
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Upsample g1 to match x1 size if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        psi = F.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

class AttentionUNet(nn.Module):
    """U-Net with attention gates"""
    
    def __init__(self, in_channels: int = 3, num_classes: int = 21,
                 features: List[int] = [64, 128, 256, 512]):
        super(AttentionUNet, self).__init__()
        
        self.features = features
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.encoder.append(DoubleConv(in_channels, features[0]))
        
        for i in range(len(features) - 1):
            self.encoder.append(DownBlock(features[i], features[i + 1]))
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Attention gates
        self.attention_gates = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.attention_gates.append(
                AttentionGate(features[i] * 2 if i == len(features) - 1 else features[i],
                             features[i - 1], features[i - 1] // 2)
            )
        
        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.decoder.append(UpBlock(features[i] * 2, features[i - 1]))
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], num_classes, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store skip connections
        skip_connections = []
        
        # Encoder path
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with attention
        skip_connections = skip_connections[::-1]  # Reverse order
        
        for i, (attention, decoder) in enumerate(zip(self.attention_gates, self.decoder)):
            skip_connection = skip_connections[i]
            
            # Apply attention gate
            attended_skip = attention(x, skip_connection)
            
            # Decoder step
            x = decoder(x, attended_skip)
        
        return self.final_conv(x)

# Test Attention U-Net
attention_unet = AttentionUNet(in_channels=3, num_classes=21)
test_input = torch.randn(1, 3, 256, 256)
attention_output = attention_unet(test_input)
print(f"Attention U-Net output: {attention_output.shape}")

# 5. SEGMENTATION LOSS FUNCTIONS
print("\n5. SEGMENTATION LOSS FUNCTIONS")

class SegmentationLoss(nn.Module):
    """Collection of segmentation loss functions"""
    
    def __init__(self, num_classes: int, ignore_index: int = -1):
        super(SegmentationLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
    def cross_entropy_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Standard cross-entropy loss"""
        return F.cross_entropy(pred, target, ignore_index=self.ignore_index)
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        """Dice loss for segmentation"""
        pred_soft = F.softmax(pred, dim=1)
        
        # Convert target to one-hot
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Calculate Dice coefficient
        intersection = torch.sum(pred_soft * target_one_hot, dim=(2, 3))
        union = torch.sum(pred_soft, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
        
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                   alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
        """Focal loss for handling class imbalance"""
        ce_loss = F.cross_entropy(pred, target, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def iou_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        """IoU (Jaccard) loss"""
        pred_soft = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        intersection = torch.sum(pred_soft * target_one_hot, dim=(2, 3))
        union = torch.sum(pred_soft + target_one_hot - pred_soft * target_one_hot, dim=(2, 3))
        
        iou = (intersection + smooth) / (union + smooth)
        return 1 - iou.mean()
    
    def combined_loss(self, pred: torch.Tensor, target: torch.Tensor,
                     ce_weight: float = 1.0, dice_weight: float = 1.0) -> torch.Tensor:
        """Combined cross-entropy and Dice loss"""
        ce_loss = self.cross_entropy_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        
        return ce_weight * ce_loss + dice_weight * dice_loss

# Test segmentation losses
loss_fn = SegmentationLoss(num_classes=21)

# Create dummy predictions and targets
dummy_pred = torch.randn(2, 21, 64, 64)
dummy_target = torch.randint(0, 21, (2, 64, 64))

print("Testing segmentation losses:")
print(f"Cross-entropy loss: {loss_fn.cross_entropy_loss(dummy_pred, dummy_target):.4f}")
print(f"Dice loss: {loss_fn.dice_loss(dummy_pred, dummy_target):.4f}")
print(f"Focal loss: {loss_fn.focal_loss(dummy_pred, dummy_target):.4f}")
print(f"IoU loss: {loss_fn.iou_loss(dummy_pred, dummy_target):.4f}")
print(f"Combined loss: {loss_fn.combined_loss(dummy_pred, dummy_target):.4f}")

# 6. EVALUATION METRICS
print("\n6. EVALUATION METRICS")

class SegmentationMetrics:
    """Evaluation metrics for semantic segmentation"""
    
    def __init__(self, num_classes: int, ignore_index: int = -1):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
    def pixel_accuracy(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate pixel accuracy"""
        pred_labels = torch.argmax(pred, dim=1)
        
        if self.ignore_index >= 0:
            mask = target != self.ignore_index
            correct = (pred_labels == target) & mask
            total = mask.sum()
        else:
            correct = (pred_labels == target)
            total = target.numel()
        
        return correct.sum().float() / total.float()
    
    def mean_pixel_accuracy(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate mean pixel accuracy per class"""
        pred_labels = torch.argmax(pred, dim=1)
        class_accuracies = []
        
        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue
                
            cls_mask = target == cls
            if cls_mask.sum() == 0:
                continue
                
            cls_correct = (pred_labels == target) & cls_mask
            cls_accuracy = cls_correct.sum().float() / cls_mask.sum().float()
            class_accuracies.append(cls_accuracy)
        
        return torch.stack(class_accuracies).mean() if class_accuracies else torch.tensor(0.0)
    
    def intersection_over_union(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Calculate IoU metrics"""
        pred_labels = torch.argmax(pred, dim=1)
        
        class_ious = []
        
        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue
                
            pred_cls = (pred_labels == cls)
            target_cls = (target == cls)
            
            intersection = (pred_cls & target_cls).sum().float()
            union = (pred_cls | target_cls).sum().float()
            
            if union == 0:
                continue
                
            iou = intersection / union
            class_ious.append(iou)
        
        if class_ious:
            mean_iou = torch.stack(class_ious).mean()
            return {
                'mean_iou': mean_iou.item(),
                'class_ious': [iou.item() for iou in class_ious]
            }
        else:
            return {'mean_iou': 0.0, 'class_ious': []}
    
    def dice_coefficient(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Calculate Dice coefficient"""
        pred_labels = torch.argmax(pred, dim=1)
        
        class_dice = []
        
        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue
                
            pred_cls = (pred_labels == cls)
            target_cls = (target == cls)
            
            intersection = (pred_cls & target_cls).sum().float()
            total = pred_cls.sum() + target_cls.sum()
            
            if total == 0:
                continue
                
            dice = (2 * intersection) / total
            class_dice.append(dice)
        
        if class_dice:
            mean_dice = torch.stack(class_dice).mean()
            return {
                'mean_dice': mean_dice.item(),
                'class_dice': [dice.item() for dice in class_dice]
            }
        else:
            return {'mean_dice': 0.0, 'class_dice': []}

# Test evaluation metrics
metrics = SegmentationMetrics(num_classes=21)

dummy_pred = torch.randn(2, 21, 32, 32)
dummy_target = torch.randint(0, 21, (2, 32, 32))

print("Testing segmentation metrics:")
print(f"Pixel accuracy: {metrics.pixel_accuracy(dummy_pred, dummy_target):.4f}")
print(f"Mean pixel accuracy: {metrics.mean_pixel_accuracy(dummy_pred, dummy_target):.4f}")

iou_results = metrics.intersection_over_union(dummy_pred, dummy_target)
print(f"Mean IoU: {iou_results['mean_iou']:.4f}")

dice_results = metrics.dice_coefficient(dummy_pred, dummy_target)
print(f"Mean Dice: {dice_results['mean_dice']:.4f}")

# 7. TRAINING UTILITIES
print("\n7. TRAINING UTILITIES")

class SegmentationTrainer:
    """Training utilities for segmentation models"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.training_history = {
            'train_loss': [], 'val_loss': [],
            'train_iou': [], 'val_iou': []
        }
        
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                   loss_fn: SegmentationLoss, metrics: SegmentationMetrics) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_iou = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            
            # Resize output to match target if needed
            if output.shape[2:] != target.shape[1:]:
                output = F.interpolate(output, size=target.shape[1:], mode='bilinear', align_corners=True)
            
            loss = loss_fn.combined_loss(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate IoU
            iou_results = metrics.intersection_over_union(output, target)
            total_iou += iou_results['mean_iou']
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}: Loss: {loss.item():.4f}, IoU: {iou_results["mean_iou"]:.4f}')
        
        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches
        
        return {'loss': avg_loss, 'iou': avg_iou}

# Create dummy dataset for demonstration
class DummySegmentationDataset(Dataset):
    def __init__(self, size: int = 100, image_size: Tuple[int, int] = (256, 256), num_classes: int = 21):
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Random image and segmentation mask
        image = torch.randn(3, *self.image_size)
        mask = torch.randint(0, self.num_classes, self.image_size)
        return image, mask

# Test training setup
dummy_dataset = DummySegmentationDataset(size=50)
dummy_loader = DataLoader(dummy_dataset, batch_size=2, shuffle=True)

trainer = SegmentationTrainer(unet)
optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
loss_fn = SegmentationLoss(num_classes=21)
metrics = SegmentationMetrics(num_classes=21)

print("\nTesting training epoch...")
train_results = trainer.train_epoch(dummy_loader, optimizer, loss_fn, metrics)
print(f"Training results: Loss = {train_results['loss']:.4f}, IoU = {train_results['iou']:.4f}")

print("\n=== SEMANTIC SEGMENTATION COMPLETE ===")
print("Key concepts covered:")
print("- Basic segmentation building blocks")
print("- U-Net implementation with skip connections")
print("- FCN (Fully Convolutional Network)")
print("- Advanced architectures (Attention U-Net)")
print("- Segmentation loss functions (CE, Dice, Focal, IoU)")
print("- Evaluation metrics (Pixel accuracy, IoU, Dice)")
print("- Training utilities and pipeline")