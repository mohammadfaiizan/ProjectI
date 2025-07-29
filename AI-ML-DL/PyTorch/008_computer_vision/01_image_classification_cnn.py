"""
PyTorch CNN Image Classification - Building CNNs from Scratch
Comprehensive guide to implementing Convolutional Neural Networks for image classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import time

print("=== CNN IMAGE CLASSIFICATION ===")

# 1. BASIC CNN BUILDING BLOCKS
print("\n1. BASIC CNN BUILDING BLOCKS")

class ConvBlock(nn.Module):
    """Basic convolutional block with conv + batch norm + relu"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 use_batchnorm: bool = True, activation: str = 'relu'):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for efficient models"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                 padding=padding, groups=in_channels)
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Test building blocks
test_input = torch.randn(2, 3, 32, 32)
conv_block = ConvBlock(3, 64)
depthwise_conv = DepthwiseSeparableConv(3, 64)

print(f"Input shape: {test_input.shape}")
print(f"ConvBlock output: {conv_block(test_input).shape}")
print(f"DepthwiseSeparableConv output: {depthwise_conv(test_input).shape}")

# 2. SIMPLE CNN ARCHITECTURE
print("\n2. SIMPLE CNN ARCHITECTURE")

class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10 classification"""
    
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            
            ConvBlock(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
            
            ConvBlock(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),  # 8x8 -> 4x4
            
            ConvBlock(128, 256, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# Test SimpleCNN
simple_cnn = SimpleCNN(num_classes=10)
test_output = simple_cnn(test_input)
print(f"SimpleCNN output shape: {test_output.shape}")

# Calculate model parameters
total_params = sum(p.numel() for p in simple_cnn.parameters())
trainable_params = sum(p.numel() for p in simple_cnn.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# 3. ADVANCED CNN ARCHITECTURES
print("\n3. ADVANCED CNN ARCHITECTURES")

class ResidualBlock(nn.Module):
    """Residual block for ResNet-style architectures"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class InceptionBlock(nn.Module):
    """Inception-style block with multiple parallel paths"""
    
    def __init__(self, in_channels: int, out_1x1: int, out_3x3_reduce: int, 
                 out_3x3: int, out_5x5_reduce: int, out_5x5: int, out_pool: int):
        super(InceptionBlock, self).__init__()
        
        # 1x1 convolution branch
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1, padding=0)
        
        # 3x3 convolution branch
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, out_3x3_reduce, kernel_size=1, padding=0),
            ConvBlock(out_3x3_reduce, out_3x3, kernel_size=3, padding=1)
        )
        
        # 5x5 convolution branch
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, out_5x5_reduce, kernel_size=1, padding=0),
            ConvBlock(out_5x5_reduce, out_5x5, kernel_size=5, padding=2)
        )
        
        # Max pooling branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            ConvBlock(in_channels, out_pool, kernel_size=1, padding=0)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)

# Test advanced blocks
residual_block = ResidualBlock(64, 64)
inception_block = InceptionBlock(64, 16, 32, 64, 8, 16, 16)

test_feature = torch.randn(2, 64, 16, 16)
print(f"ResidualBlock output: {residual_block(test_feature).shape}")
print(f"InceptionBlock output: {inception_block(test_feature).shape}")

# 4. CUSTOM RESNET IMPLEMENTATION
print("\n4. CUSTOM RESNET IMPLEMENTATION")

class CustomResNet(nn.Module):
    """Custom ResNet implementation for image classification"""
    
    def __init__(self, num_classes: int = 10, layers: List[int] = [2, 2, 2, 2]):
        super(CustomResNet, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Residual layers
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Test CustomResNet
custom_resnet = CustomResNet(num_classes=10, layers=[1, 1, 1, 1])  # Smaller for demo
test_large_input = torch.randn(2, 3, 224, 224)
resnet_output = custom_resnet(test_large_input)
print(f"CustomResNet output shape: {resnet_output.shape}")

# 5. EFFICIENT CNN ARCHITECTURES
print("\n5. EFFICIENT CNN ARCHITECTURES")

class MobileNetBlock(nn.Module):
    """MobileNet-style block with depthwise separable convolutions"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(MobileNetBlock, self).__init__()
        
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class EfficientCNN(nn.Module):
    """Efficient CNN using depthwise separable convolutions"""
    
    def __init__(self, num_classes: int = 10):
        super(EfficientCNN, self).__init__()
        
        # Initial convolution
        self.conv1 = ConvBlock(3, 32, stride=2)
        
        # MobileNet blocks
        self.blocks = nn.Sequential(
            MobileNetBlock(32, 64),
            MobileNetBlock(64, 128, stride=2),
            MobileNetBlock(128, 128),
            MobileNetBlock(128, 256, stride=2),
            MobileNetBlock(256, 256),
            MobileNetBlock(256, 512, stride=2),
            MobileNetBlock(512, 512),
        )
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Test EfficientCNN
efficient_cnn = EfficientCNN(num_classes=10)
efficient_output = efficient_cnn(test_input)
print(f"EfficientCNN output shape: {efficient_output.shape}")

# Compare model sizes
simple_params = sum(p.numel() for p in simple_cnn.parameters())
efficient_params = sum(p.numel() for p in efficient_cnn.parameters())
print(f"SimpleCNN parameters: {simple_params:,}")
print(f"EfficientCNN parameters: {efficient_params:,}")
print(f"Parameter reduction: {(1 - efficient_params/simple_params)*100:.1f}%")

# 6. TRAINING INFRASTRUCTURE
print("\n6. TRAINING INFRASTRUCTURE")

class CNNTrainer:
    """Training class for CNN models"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.train_accuracies = []
        
    def train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}: Loss: {loss.item():.4f}')
                
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        
        return avg_loss, accuracy
        
    def evaluate(self, dataloader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy

# 7. DATA LOADING AND PREPROCESSING
print("\n7. DATA LOADING AND PREPROCESSING")

def get_cifar10_loaders(batch_size: int = 64, download: bool = False) -> Tuple[DataLoader, DataLoader]:
    """Get CIFAR-10 data loaders with preprocessing"""
    
    # Data transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Datasets
    if download:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform_test)
    else:
        # Create dummy datasets for demonstration
        class DummyCIFAR10(Dataset):
            def __init__(self, size=1000, transform=None):
                self.size = size
                self.transform = transform
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                # Generate random image and label
                image = torch.randn(3, 32, 32)
                if self.transform:
                    image = self.transform(image)
                label = torch.randint(0, 10, (1,)).item()
                return image, label
                
        trainset = DummyCIFAR10(1000, transform_train)
        testset = DummyCIFAR10(200, transform_test)
    
    # Data loaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return trainloader, testloader

# Get data loaders
trainloader, testloader = get_cifar10_loaders(batch_size=32, download=False)
print(f"Training batches: {len(trainloader)}")
print(f"Test batches: {len(testloader)}")

# 8. COMPLETE TRAINING EXAMPLE
print("\n8. COMPLETE TRAINING EXAMPLE")

def train_cnn_model(model: nn.Module, epochs: int = 3, learning_rate: float = 0.001):
    """Complete training example"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Initialize trainer
    trainer = CNNTrainer(model, device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training
        train_loss, train_acc = trainer.train_epoch(trainloader, optimizer, criterion)
        
        # Validation
        val_loss, val_acc = trainer.evaluate(testloader, criterion)
        
        # Update learning rate
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Time: {epoch_time:.2f}s')
        print('-' * 50)
        
    return trainer

# Train a simple model
print("Training SimpleCNN...")
simple_model = SimpleCNN(num_classes=10)
trainer = train_cnn_model(simple_model, epochs=2, learning_rate=0.001)

# 9. MODEL ANALYSIS AND VISUALIZATION
print("\n9. MODEL ANALYSIS AND VISUALIZATION")

def analyze_model(model: nn.Module, input_shape: Tuple[int, ...] = (1, 3, 32, 32)):
    """Analyze model architecture and parameters"""
    
    print("Model Architecture Analysis:")
    print("=" * 50)
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")
    
    # Layer-wise analysis
    print("\nLayer-wise Parameter Count:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:50} {param.numel():>10,} parameters")
    
    # Forward pass analysis
    model.eval()
    dummy_input = torch.randn(*input_shape)
    
    # Hook to capture intermediate outputs
    layer_outputs = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                layer_outputs[name] = output.shape
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    print(f"\nOutput shape: {output.shape}")
    print("\nIntermediate layer shapes:")
    for name, shape in layer_outputs.items():
        if name:  # Skip empty names
            print(f"{name:50} {str(shape):>20}")

# Analyze the trained model
analyze_model(simple_model, input_shape=(1, 3, 32, 32))

print("\n=== CNN IMAGE CLASSIFICATION COMPLETE ===")
print("Key concepts covered:")
print("- Basic CNN building blocks (ConvBlock, DepthwiseSeparableConv)")
print("- Simple and advanced CNN architectures")
print("- Residual and Inception blocks")
print("- Custom ResNet implementation")
print("- Efficient architectures (MobileNet-style)")
print("- Complete training infrastructure")
print("- Data loading and preprocessing")
print("- Model analysis and visualization")