"""
ERA 4: EFFICIENCY AND MOBILE OPTIMIZATION - MobileNet Efficiency
===============================================================

Year: 2017
Paper: "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" (Howard et al., 2017)
Innovation: Depthwise separable convolutions, mobile-optimized architecture
Previous Limitation: Heavy computational cost, large model size for mobile deployment
Performance Gain: 8-9x computation reduction, 25-50x parameter reduction with minimal accuracy loss
Impact: Enabled deep learning on mobile devices, established mobile-first AI paradigm

This file implements MobileNet architecture that revolutionized mobile computer vision through
depthwise separable convolutions and efficient architecture design for resource-constrained devices.
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
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# HISTORICAL CONTEXT & MOTIVATION
# ============================================================================

YEAR = "2017"
INNOVATION = "Depthwise separable convolutions for mobile-optimized efficiency"
PREVIOUS_LIMITATION = "Heavy computation and large models unsuitable for mobile devices"
IMPACT = "Enabled mobile AI, established efficiency as key design principle"

print(f"=== MobileNet Efficiency ({YEAR}) ===")
print(f"Innovation: {INNOVATION}")
print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
print(f"Impact: {IMPACT}")
print("="*60)

# ============================================================================
# STANDARDIZED DATASET LOADING (CIFAR-10)
# ============================================================================

def load_cifar10_dataset():
    """Load CIFAR-10 dataset with MobileNet-style preprocessing"""
    print("Loading CIFAR-10 dataset for MobileNet efficiency study...")
    
    # MobileNet-style preprocessing
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # CIFAR-10 class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print(f"Classes: {len(classes)}")
    
    return train_loader, test_loader, classes

# ============================================================================
# DEPTHWISE SEPARABLE CONVOLUTION
# ============================================================================

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution - Core innovation of MobileNet
    
    Traditional Convolution: D_K × D_K × M × N
    Depthwise + Pointwise: D_K × D_K × M + M × N
    
    Computational Reduction: 1/N + 1/D_K²
    For 3x3 conv: ~8-9x reduction in computation
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Depthwise convolution (spatial filtering)
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=in_channels,  # Key: groups=in_channels
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Pointwise convolution (channel mixing)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        print(f"  DepthwiseSeparable: {in_channels}→{out_channels}, {kernel_size}x{kernel_size}")
        print(f"    Computational saving: {self._compute_efficiency():.1f}x")
    
    def _compute_efficiency(self):
        """Calculate computational efficiency gain"""
        # Traditional convolution operations
        traditional_ops = self.kernel_size * self.kernel_size * self.in_channels * self.out_channels
        
        # Depthwise separable operations
        depthwise_ops = self.kernel_size * self.kernel_size * self.in_channels
        pointwise_ops = self.in_channels * self.out_channels
        separable_ops = depthwise_ops + pointwise_ops
        
        return traditional_ops / separable_ops
    
    def forward(self, x):
        """Forward pass through depthwise separable convolution"""
        # Depthwise convolution
        x = F.relu(self.bn1(self.depthwise(x)))
        
        # Pointwise convolution
        x = F.relu(self.bn2(self.pointwise(x)))
        
        return x
    
    def get_computational_analysis(self, input_size):
        """Analyze computational complexity"""
        H, W = input_size[2], input_size[3]
        
        # Traditional convolution
        traditional_flops = (
            H * W * self.kernel_size * self.kernel_size * 
            self.in_channels * self.out_channels
        )
        
        # Depthwise separable convolution
        depthwise_flops = H * W * self.kernel_size * self.kernel_size * self.in_channels
        pointwise_flops = H * W * self.in_channels * self.out_channels
        separable_flops = depthwise_flops + pointwise_flops
        
        return {
            'traditional_flops': traditional_flops,
            'separable_flops': separable_flops,
            'efficiency_ratio': traditional_flops / separable_flops,
            'flop_reduction': (1 - separable_flops / traditional_flops) * 100
        }

# ============================================================================
# MOBILENET ARCHITECTURE
# ============================================================================

class MobileNet_Efficiency(nn.Module):
    """
    MobileNet architecture optimized for mobile devices
    
    Key Innovations:
    - Depthwise separable convolutions throughout
    - Width multiplier for model scaling
    - Resolution multiplier for input scaling
    - Global average pooling
    - Minimal fully connected layers
    """
    
    def __init__(self, num_classes=10, width_mult=1.0, resolution_mult=1.0):
        super(MobileNet_Efficiency, self).__init__()
        
        self.width_mult = width_mult
        self.resolution_mult = resolution_mult
        
        print(f"Building MobileNet with width_mult={width_mult}, resolution_mult={resolution_mult}")
        
        # Helper function to make channels divisible by 8
        def _make_divisible(v, divisor=8):
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
        
        # Initial convolution
        input_channel = _make_divisible(32 * width_mult)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True)
        )
        
        # MobileNet configuration: [input_channels, output_channels, stride]
        mobile_config = [
            [input_channel, 64, 1],
            [64, 128, 2],
            [128, 128, 1],
            [128, 256, 2],
            [256, 256, 1],
            [256, 512, 2],
            [512, 512, 1],
            [512, 512, 1],
            [512, 512, 1],
            [512, 512, 1],
            [512, 512, 1],
            [512, 1024, 2],
            [1024, 1024, 1],
        ]
        
        # Build depthwise separable convolutions
        self.features = nn.ModuleList()
        for in_ch, out_ch, stride in mobile_config:
            in_ch = _make_divisible(in_ch * width_mult)
            out_ch = _make_divisible(out_ch * width_mult)
            
            # Use depthwise separable convolution
            block = DepthwiseSeparableConv(in_ch, out_ch, stride=stride)
            self.features.append(block)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        last_channel = _make_divisible(1024 * width_mult)
        self.classifier = nn.Linear(last_channel, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        # Calculate efficiency metrics
        total_params = sum(p.numel() for p in self.parameters())
        print(f"MobileNet Architecture Summary:")
        print(f"  Width Multiplier: {width_mult}")
        print(f"  Resolution Multiplier: {resolution_mult}")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Depthwise Separable Blocks: {len(self.features)}")
        print(f"  Key Innovation: Mobile-optimized efficiency")
    
    def _initialize_weights(self):
        """Initialize weights for MobileNet"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass through MobileNet"""
        # Initial convolution
        x = self.conv1(x)
        
        # Depthwise separable convolutions
        for block in self.features:
            x = block(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def get_efficiency_analysis(self, input_size=(1, 3, 32, 32)):
        """Analyze overall efficiency of MobileNet"""
        total_traditional_flops = 0
        total_separable_flops = 0
        
        # Analyze each depthwise separable block
        current_size = list(input_size)
        
        for block in self.features:
            if isinstance(block, DepthwiseSeparableConv):
                analysis = block.get_computational_analysis(current_size)
                total_traditional_flops += analysis['traditional_flops']
                total_separable_flops += analysis['separable_flops']
                
                # Update size for next layer (rough estimation)
                if hasattr(block.depthwise, 'stride') and block.depthwise.stride[0] == 2:
                    current_size[2] //= 2
                    current_size[3] //= 2
                current_size[1] = block.out_channels
        
        return {
            'total_traditional_flops': total_traditional_flops,
            'total_separable_flops': total_separable_flops,
            'overall_efficiency': total_traditional_flops / total_separable_flops,
            'flop_reduction_percent': (1 - total_separable_flops / total_traditional_flops) * 100
        }

# ============================================================================
# COMPARISON: STANDARD CNN vs MOBILENET
# ============================================================================

class StandardCNN(nn.Module):
    """Standard CNN for comparison with MobileNet"""
    
    def __init__(self, num_classes=10):
        super(StandardCNN, self).__init__()
        
        print("Building Standard CNN for comparison...")
        
        # Standard convolutions (no depthwise separable)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, 3, stride=2, padding=1), nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, num_classes)
        
        # Initialize weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ============================================================================
# EFFICIENCY ANALYSIS
# ============================================================================

def analyze_width_multiplier_effects():
    """Analyze effects of different width multipliers"""
    print("\nAnalyzing Width Multiplier Effects...")
    
    width_multipliers = [0.25, 0.5, 0.75, 1.0]
    results = {}
    
    for width_mult in width_multipliers:
        model = MobileNet_Efficiency(width_mult=width_mult)
        
        # Calculate model statistics
        total_params = sum(p.numel() for p in model.parameters())
        efficiency_analysis = model.get_efficiency_analysis()
        
        results[width_mult] = {
            'parameters': total_params,
            'flop_reduction': efficiency_analysis['flop_reduction_percent'],
            'efficiency_ratio': efficiency_analysis['overall_efficiency']
        }
        
        print(f"Width Multiplier {width_mult}:")
        print(f"  Parameters: {total_params:,}")
        print(f"  FLOP Reduction: {efficiency_analysis['flop_reduction_percent']:.1f}%")
        print(f"  Efficiency Ratio: {efficiency_analysis['overall_efficiency']:.1f}x")
    
    return results

def compare_depthwise_vs_standard():
    """Compare depthwise separable vs standard convolutions"""
    print("\nComparing Depthwise Separable vs Standard Convolutions...")
    
    # Test different configurations
    configs = [
        (64, 128, 3),   # (in_channels, out_channels, kernel_size)
        (128, 256, 3),
        (256, 512, 3),
        (512, 1024, 3)
    ]
    
    results = {}
    
    for in_ch, out_ch, k_size in configs:
        config_name = f"{in_ch}→{out_ch}"
        
        # Create depthwise separable conv
        dw_conv = DepthwiseSeparableConv(in_ch, out_ch, k_size)
        
        # Calculate parameters
        dw_params = sum(p.numel() for p in dw_conv.parameters())
        
        # Standard conv parameters
        standard_params = k_size * k_size * in_ch * out_ch + out_ch  # +bias
        
        # Computational analysis
        input_size = (1, in_ch, 32, 32)
        comp_analysis = dw_conv.get_computational_analysis(input_size)
        
        results[config_name] = {
            'dw_params': dw_params,
            'standard_params': standard_params,
            'param_reduction': (1 - dw_params / standard_params) * 100,
            'flop_reduction': comp_analysis['flop_reduction'],
            'efficiency_ratio': comp_analysis['efficiency_ratio']
        }
        
        print(f"{config_name}: Param Reduction: {results[config_name]['param_reduction']:.1f}%, "
              f"FLOP Reduction: {comp_analysis['flop_reduction']:.1f}%")
    
    return results

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_mobilenet_efficiency(model, train_loader, test_loader, epochs=60, learning_rate=0.1):
    """Train MobileNet with efficiency-focused training"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Training configuration optimized for mobile models
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=4e-5  # Lower weight decay for mobile models
    )
    
    # Learning rate scheduling
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 45], gamma=0.1)
    
    criterion = nn.CrossEntropyLoss()
    
    # Training tracking
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    model_name = model.__class__.__name__
    print(f"Training {model_name} on device: {device}")
    
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()
            
            if batch_idx % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.4f}, LR: {current_lr:.6f}')
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_train_acc = 100. * correct_train / total_train
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_train_acc)
        
        # Test evaluation
        test_acc = evaluate_model(model, test_loader, device)
        test_accuracies.append(test_acc)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), f'AI-ML-DL/Models/CNN/mobilenet_efficiency_best.pth')
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {epoch_loss:.4f}, '
              f'Train Acc: {epoch_train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        # Early stopping
        if test_acc > 90.0:
            print(f"Good performance reached at epoch {epoch+1}")
            break
    
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    return train_losses, train_accuracies, test_accuracies

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

# ============================================================================
# PERFORMANCE TRACKING
# ============================================================================

def track_training_metrics(model_name, train_function, *args):
    """Track computational metrics during training"""
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024
    
    start_time = time.time()
    result = train_function(*args)
    training_time = time.time() - start_time
    
    memory_after = process.memory_info().rss / 1024 / 1024
    memory_used = memory_after - memory_before
    
    return {
        'model_name': model_name,
        'training_time_minutes': training_time / 60,
        'memory_usage_mb': memory_used,
        'result': result
    }

# ============================================================================
# MAIN EXECUTION (PARTIAL DUE TO LENGTH)
# ============================================================================

def main():
    print(f"=== MobileNet Efficiency Implementation ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load CIFAR-10 dataset
    train_loader, test_loader, classes = load_cifar10_dataset()
    
    # Initialize models
    mobilenet = MobileNet_Efficiency(width_mult=1.0)
    standard_cnn = StandardCNN()
    
    # Analyze efficiency
    width_mult_analysis = analyze_width_multiplier_effects()
    depthwise_comparison = compare_depthwise_vs_standard()
    
    print(f"\nMobileNet revolutionized mobile AI through:")
    print(f"• Depthwise separable convolutions: 8-9x FLOP reduction")
    print(f"• Width multiplier scaling: 0.25x to 1.0x model sizes")
    print(f"• Global average pooling: Minimal parameters")
    print(f"• Mobile-first architecture design")
    
    return {
        'model': 'MobileNet Efficiency',
        'year': YEAR,
        'innovation': INNOVATION,
        'width_mult_analysis': width_mult_analysis,
        'depthwise_comparison': depthwise_comparison
    }

if __name__ == "__main__":
    results = main()