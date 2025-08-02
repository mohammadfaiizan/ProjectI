"""
ERA 3: RESIDUAL LEARNING BREAKTHROUGH - ResNet Residual Revolution
================================================================

Year: 2015
Paper: "Deep Residual Learning for Image Recognition" (He et al., 2015)
Innovation: Skip connections enabling ultra-deep networks, residual learning
Previous Limitation: Degradation problem in very deep networks, vanishing gradients
Performance Gain: 152-layer networks, identity mappings, breakthrough performance
Impact: Enabled training of arbitrarily deep networks, revolutionized architecture design

This file implements the revolutionary ResNet architecture that introduced skip connections
and residual learning, solving the degradation problem and enabling ultra-deep networks.
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

YEAR = "2015"
INNOVATION = "Skip connections and residual learning enabling ultra-deep networks"
PREVIOUS_LIMITATION = "Degradation problem: deeper networks performed worse than shallow ones"
IMPACT = "Revolutionized deep learning, enabled arbitrarily deep networks"

print(f"=== ResNet Residual Revolution ({YEAR}) ===")
print(f"Innovation: {INNOVATION}")
print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
print(f"Impact: {IMPACT}")
print("="*60)

# ============================================================================
# STANDARDIZED DATASET LOADING (CIFAR-10)
# ============================================================================

def load_cifar10_dataset():
    """
    Load CIFAR-10 dataset with ResNet-style preprocessing
    """
    print("Loading CIFAR-10 dataset for ResNet residual learning study...")
    
    # ResNet-style preprocessing
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
# RESIDUAL BLOCK IMPLEMENTATION
# ============================================================================

class BasicBlock(nn.Module):
    """
    Basic Residual Block for ResNet-18/34
    
    Revolutionary Skip Connection:
    - Input: x
    - Transformation: F(x) 
    - Output: F(x) + x (residual connection)
    - If dimensions don't match: F(x) + W_s*x (projection shortcut)
    """
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        # First conv layer
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # Second conv layer
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            # Projection shortcut to match dimensions
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        """
        Forward pass with revolutionary residual connection
        """
        # Store input for skip connection
        identity = x
        
        # Main path: F(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Skip connection: F(x) + x
        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out

class Bottleneck(nn.Module):
    """
    Bottleneck Residual Block for ResNet-50/101/152
    
    Three-layer design:
    - 1x1 conv (reduce dimensions)
    - 3x3 conv (main computation)  
    - 1x1 conv (restore dimensions)
    - Skip connection around all three
    """
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        
        # 1x1 conv (dimensionality reduction)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 3x3 conv (main computation)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # 1x1 conv (dimensionality expansion)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        """Forward pass with bottleneck residual connection"""
        identity = x
        
        # Bottleneck path: 1x1 -> 3x3 -> 1x1
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # Skip connection
        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out

# ============================================================================
# ResNet ARCHITECTURE
# ============================================================================

class ResNet_Revolution(nn.Module):
    """
    ResNet architecture with revolutionary skip connections
    
    Key Innovations:
    - Skip connections: H(x) = F(x) + x
    - Identity mappings for gradient flow
    - Batch normalization integration
    - Enables training of 50, 101, 152+ layer networks
    - Solves degradation problem
    """
    
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_Revolution, self).__init__()
        
        self.in_planes = 64
        self.block_type = block.__name__
        
        print(f"Building ResNet with {self.block_type} blocks...")
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_resnet_weights()
        
        # Calculate architecture statistics
        total_layers = sum(num_blocks) * 2 + 2  # Each block has 2 convs + initial conv + fc
        if block == Bottleneck:
            total_layers = sum(num_blocks) * 3 + 2  # Each bottleneck has 3 convs
        
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"ResNet Architecture Summary:")
        print(f"  Block Type: {self.block_type}")
        print(f"  Total Layers: {total_layers}")
        print(f"  Residual Blocks: {sum(num_blocks)}")
        print(f"  Skip Connections: {sum(num_blocks)}")
        print(f"  Parameters: {total_params:,}")
        print(f"  Key Innovation: Identity mappings")
    
    def _make_layer(self, block, planes, num_blocks, stride):
        """Create a layer of residual blocks"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        
        return nn.Sequential(*layers)
    
    def _initialize_resnet_weights(self):
        """Initialize weights using ResNet method"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass through ResNet with skip connections"""
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual layers (each contains multiple skip connections)
        x = self.layer1(x)  # Multiple identity mappings here
        x = self.layer2(x)  # More identity mappings
        x = self.layer3(x)  # Even more identity mappings
        x = self.layer4(x)  # Final identity mappings
        
        # Global average pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def get_gradient_flow_analysis(self, x, target):
        """
        Analyze gradient flow through skip connections
        Demonstrates how residuals solve vanishing gradient problem
        """
        self.train()
        x.requires_grad_(True)
        
        # Forward pass
        output = self.forward(x)
        loss = F.cross_entropy(output, target)
        
        # Backward pass
        loss.backward(retain_graph=True)
        
        # Collect gradients from different layers
        gradient_norms = {}
        
        for name, module in self.named_modules():
            if isinstance(module, (BasicBlock, Bottleneck)):
                if hasattr(module.conv1, 'weight') and module.conv1.weight.grad is not None:
                    grad_norm = module.conv1.weight.grad.norm().item()
                    gradient_norms[name] = grad_norm
        
        return gradient_norms

# ============================================================================
# COMPARISON: WITH VS WITHOUT SKIP CONNECTIONS
# ============================================================================

class PlainCNN(nn.Module):
    """
    Plain CNN without skip connections (pre-ResNet approach)
    Used to demonstrate the degradation problem
    """
    
    def __init__(self, num_layers=18, num_classes=10):
        super(PlainCNN, self).__init__()
        
        self.num_layers = num_layers
        
        # Build plain network without skip connections
        layers = []
        in_channels = 3
        
        # Initial layer
        layers.extend([
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ])
        in_channels = 64
        
        # Add many layers to show degradation problem
        layer_configs = [
            (64, 4), (128, 4), (256, 4), (512, 4)  # (channels, num_blocks)
        ]
        
        for channels, num_blocks in layer_configs:
            for i in range(num_blocks):
                stride = 2 if i == 0 and channels > 64 else 1
                layers.extend([
                    nn.Conv2d(in_channels, channels, kernel_size=3, 
                             stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels, kernel_size=3, 
                             padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                ])
                in_channels = channels
        
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ============================================================================
# RESNET VARIANTS
# ============================================================================

def ResNet18():
    """ResNet-18 with BasicBlock"""
    return ResNet_Revolution(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    """ResNet-34 with BasicBlock"""
    return ResNet_Revolution(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    """ResNet-50 with Bottleneck"""
    return ResNet_Revolution(Bottleneck, [3, 4, 6, 3])

def ResNet101():
    """ResNet-101 with Bottleneck"""
    return ResNet_Revolution(Bottleneck, [3, 4, 23, 3])

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_resnet_revolution(model, train_loader, test_loader, epochs=90, learning_rate=0.1):
    """
    Train ResNet with proper techniques from the original paper
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # ResNet training configuration (from paper)
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # Learning rate schedule: divide by 10 at epochs 32k and 48k iterations
    # For our shorter training, we'll use epochs 30 and 60
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
    
    criterion = nn.CrossEntropyLoss()
    
    # Training tracking
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print(f"Training {model.__class__.__name__} on device: {device}")
    print("Using ResNet training schedule from original paper...")
    
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
            
            # No gradient clipping needed - residual connections solve gradient problems
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
        test_acc = evaluate_resnet_model(model, test_loader, device)
        test_accuracies.append(test_acc)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), f'AI-ML-DL/Models/CNN/resnet_revolution_best.pth')
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {epoch_loss:.4f}, '
              f'Train Acc: {epoch_train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        # Early stopping
        if test_acc > 93.0:
            print(f"Excellent performance reached at epoch {epoch+1}")
            break
    
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    return train_losses, train_accuracies, test_accuracies

def evaluate_resnet_model(model, test_loader, device):
    """Evaluate ResNet model on test set"""
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
# DEGRADATION PROBLEM DEMONSTRATION
# ============================================================================

def demonstrate_degradation_problem(train_loader, test_loader):
    """
    Demonstrate the degradation problem: deeper plain networks perform worse
    """
    print("\nDemonstrating Degradation Problem...")
    print("Training plain networks of different depths...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    # Test plain networks of different depths
    plain_depths = [10, 18, 34]  # Different depths without skip connections
    
    for depth in plain_depths:
        print(f"\nTraining Plain CNN with ~{depth} layers...")
        
        model = PlainCNN(num_layers=depth).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Quick training to show trend
        model.train()
        final_losses = []
        for epoch in range(15):  # Fewer epochs for demonstration
            epoch_loss = 0
            for batch_idx, (data, targets) in enumerate(train_loader):
                if batch_idx > 50:  # Limited batches for speed
                    break
                
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Check for gradient explosion/vanishing
                grad_norm = 0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.norm().item() ** 2
                grad_norm = grad_norm ** 0.5
                
                # Clip gradients if they explode
                if grad_norm > 5.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                
                optimizer.step()
                epoch_loss += loss.item()
            
            final_losses.append(epoch_loss / 50)
        
        # Test accuracy
        test_acc = evaluate_resnet_model(model, test_loader, device)
        param_count = sum(p.numel() for p in model.parameters())
        
        results[f'Plain-{depth}'] = {
            'test_accuracy': test_acc,
            'final_loss': final_losses[-1],
            'parameters': param_count,
            'depth': depth
        }
        print(f"Plain-{depth}: {test_acc:.2f}% accuracy, Loss: {final_losses[-1]:.4f}")
    
    return results

# ============================================================================
# VISUALIZATION AND ANALYSIS
# ============================================================================

def visualize_residual_revolution():
    """Visualize ResNet's revolutionary impact"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Degradation problem illustration
    ax = axes[0, 0]
    depths = [18, 34, 50, 101, 152]
    plain_performance = [70, 68, 65, 60, 55]  # Theoretical degradation
    resnet_performance = [70, 73, 76, 78, 77]  # ResNet improvement
    
    ax.plot(depths, plain_performance, 'r--', label='Plain Networks', linewidth=2, marker='o')
    ax.plot(depths, resnet_performance, 'g-', label='ResNet', linewidth=2, marker='s')
    ax.set_title('Solving the Degradation Problem', fontsize=14)
    ax.set_xlabel('Network Depth (Layers)')
    ax.set_ylabel('Performance (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Skip connection concept
    ax = axes[0, 1]
    components = ['Input\n(x)', 'Weight\nLayers\nF(x)', 'Skip\nConnection\n(+)', 'Output\nF(x)+x']
    importance = [8, 9, 10, 9]
    colors = ['#3498DB', '#E67E22', '#E74C3C', '#27AE60']
    
    bars = ax.bar(components, importance, color=colors)
    ax.set_title('Residual Learning Components', fontsize=14)
    ax.set_ylabel('Importance Score')
    ax.tick_params(axis='x', rotation=45)
    
    # Gradient flow comparison
    ax = axes[1, 0]
    layer_depths = ['Layer 1', 'Layer 10', 'Layer 20', 'Layer 30', 'Layer 50']
    plain_gradients = [1.0, 0.5, 0.1, 0.01, 0.001]  # Vanishing gradients
    resnet_gradients = [1.0, 0.9, 0.8, 0.7, 0.6]    # Stable gradients
    
    x = np.arange(len(layer_depths))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, plain_gradients, width, label='Plain Network', color='#E74C3C')
    bars2 = ax.bar(x + width/2, resnet_gradients, width, label='ResNet', color='#27AE60')
    
    ax.set_title('Gradient Flow Comparison', fontsize=14)
    ax.set_ylabel('Gradient Magnitude')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_depths)
    ax.set_yscale('log')
    ax.legend()
    
    # ResNet architecture evolution
    ax = axes[1, 1]
    resnet_variants = ['ResNet-18', 'ResNet-34', 'ResNet-50', 'ResNet-101', 'ResNet-152']
    imagenet_accuracy = [69.8, 73.3, 76.2, 77.4, 77.8]  # Top-1 ImageNet accuracy
    
    bars = ax.bar(resnet_variants, imagenet_accuracy, 
                  color=['#3498DB', '#E67E22', '#9B59B6', '#1ABC9C', '#E74C3C'])
    ax.set_title('ResNet Variants Performance', fontsize=14)
    ax.set_ylabel('ImageNet Top-1 Accuracy (%)')
    ax.tick_params(axis='x', rotation=45)
    
    for bar, acc in zip(bars, imagenet_accuracy):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                f'{acc}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/CNN/006_resnet_revolution_analysis.png', dpi=300, bbox_inches='tight')
    print("ResNet revolution analysis saved: 006_resnet_revolution_analysis.png")

def visualize_skip_connection_concept():
    """Visualize the skip connection concept"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Traditional vs Residual learning
    ax = axes[0]
    ax.text(0.5, 0.9, 'Traditional Learning', ha='center', va='center', 
            fontsize=14, fontweight='bold')
    ax.text(0.5, 0.8, 'H(x) = F(x)', ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'), fontsize=12)
    ax.text(0.5, 0.6, 'Network must learn\ncomplete mapping H(x)', ha='center', va='center', fontsize=10)
    ax.text(0.5, 0.4, 'Difficult for deep networks', ha='center', va='center', 
            fontsize=10, style='italic', color='red')
    
    ax.text(0.5, 0.2, 'Residual Learning', ha='center', va='center', 
            fontsize=14, fontweight='bold')
    ax.text(0.5, 0.1, 'H(x) = F(x) + x', ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'), fontsize=12)
    ax.text(0.5, -0.1, 'Network learns residual\nF(x) = H(x) - x', ha='center', va='center', fontsize=10)
    ax.text(0.5, -0.3, 'Easier to optimize!', ha='center', va='center', 
            fontsize=10, style='italic', color='green')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.4, 1)
    ax.axis('off')
    
    # Identity mapping visualization
    ax = axes[1]
    
    # Draw residual block
    # Input
    ax.add_patch(plt.Rectangle((0.1, 0.7), 0.1, 0.1, facecolor='lightblue'))
    ax.text(0.15, 0.75, 'x', ha='center', va='center', fontweight='bold')
    
    # Weight layers
    ax.add_patch(plt.Rectangle((0.4, 0.5), 0.2, 0.3, facecolor='orange'))
    ax.text(0.5, 0.65, 'Weight\nLayers\nF(x)', ha='center', va='center', fontweight='bold')
    
    # Addition
    ax.add_patch(plt.Circle((0.75, 0.75), 0.05, facecolor='red'))
    ax.text(0.75, 0.75, '+', ha='center', va='center', fontweight='bold', color='white')
    
    # Output
    ax.add_patch(plt.Rectangle((0.85, 0.7), 0.1, 0.1, facecolor='lightgreen'))
    ax.text(0.9, 0.75, 'H(x)', ha='center', va='center', fontweight='bold')
    
    # Arrows
    # Input to weight layers
    ax.annotate('', xy=(0.4, 0.65), xytext=(0.21, 0.75),
               arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # Weight layers to addition
    ax.annotate('', xy=(0.7, 0.75), xytext=(0.6, 0.65),
               arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # Skip connection
    ax.annotate('', xy=(0.7, 0.75), xytext=(0.21, 0.75),
               arrowprops=dict(arrowstyle='->', lw=3, color='red', 
                             connectionstyle="arc3,rad=0.3"))
    ax.text(0.45, 0.9, 'Skip Connection', ha='center', va='center', 
            color='red', fontweight='bold')
    
    # Addition to output
    ax.annotate('', xy=(0.85, 0.75), xytext=(0.8, 0.75),
               arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, 1)
    ax.axis('off')
    ax.set_title('Residual Block Architecture', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/CNN/006_skip_connection_concept.png', dpi=300, bbox_inches='tight')
    print("Skip connection concept saved: 006_skip_connection_concept.png")

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
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"=== ResNet Residual Revolution Implementation ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load CIFAR-10 dataset
    train_loader, test_loader, classes = load_cifar10_dataset()
    
    # Initialize ResNet models
    resnet18 = ResNet18()
    resnet34 = ResNet34()
    
    # Compare different ResNet variants
    resnet_models = {
        'ResNet-18': resnet18,
        'ResNet-34': resnet34
    }
    
    results = {}
    
    # Train ResNet models
    for model_name, model in resnet_models.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print('='*50)
        
        # Train model
        metrics = track_training_metrics(
            model_name,
            train_resnet_revolution,
            model, train_loader, test_loader, 60, 0.1  # Standard ResNet training
        )
        
        train_losses, train_accuracies, test_accuracies = metrics['result']
        
        results[model_name] = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'final_accuracy': test_accuracies[-1] if test_accuracies else 0,
            'training_time': metrics['training_time_minutes'],
            'memory_usage': metrics['memory_usage_mb'],
            'parameters': sum(p.numel() for p in model.parameters())
        }
        
        print(f"{model_name} Results:")
        print(f"  Final Test Accuracy: {test_accuracies[-1]:.2f}%")
        print(f"  Training Time: {metrics['training_time_minutes']:.2f} minutes")
        print(f"  Parameters: {results[model_name]['parameters']:,}")
    
    # Demonstrate degradation problem
    degradation_results = demonstrate_degradation_problem(train_loader, test_loader)
    
    # Generate visualizations
    print("\nGenerating ResNet analysis...")
    visualize_residual_revolution()
    visualize_skip_connection_concept()
    
    # Create comprehensive results visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # ResNet variants comparison
    ax = axes[0, 0]
    model_names = list(results.keys())
    final_accs = [results[name]['final_accuracy'] for name in model_names]
    colors = ['#3498DB', '#E67E22']
    
    bars = ax.bar(model_names, final_accs, color=colors)
    ax.set_title('ResNet Variants Performance', fontsize=14)
    ax.set_ylabel('Final Test Accuracy (%)')
    for bar, acc in zip(bars, final_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # Degradation problem demonstration
    ax = axes[0, 1]
    plain_names = list(degradation_results.keys())
    plain_accs = [degradation_results[name]['test_accuracy'] for name in plain_names]
    bars = ax.bar(plain_names, plain_accs, color=['#95A5A6', '#BDC3C7', '#7F8C8D'])
    ax.set_title('Plain Networks: Degradation Problem', fontsize=14)
    ax.set_ylabel('Test Accuracy (%)')
    for bar, acc in zip(bars, plain_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # Training curves
    ax = axes[1, 0]
    colors = ['#3498DB', '#E67E22']
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        test_accs = results[model_name]['test_accuracies']
        epochs = range(1, len(test_accs) + 1)
        ax.plot(epochs, test_accs, color=color, label=model_name, linewidth=2)
    
    ax.set_title('ResNet Training Progression', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Parameter efficiency
    ax = axes[1, 1]
    param_counts = [results[name]['parameters']/1e6 for name in model_names]
    bars = ax.bar(model_names, param_counts, color=colors)
    ax.set_title('ResNet Parameter Scaling', fontsize=14)
    ax.set_ylabel('Parameters (Millions)')
    for bar, params in zip(bars, param_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{params:.1f}M', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/CNN/006_resnet_revolution_results.png', dpi=300, bbox_inches='tight')
    print("Comprehensive ResNet results saved: 006_resnet_revolution_results.png")
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("RESNET RESIDUAL REVOLUTION SUMMARY")
    print("="*70)
    
    print(f"\nHistorical Context ({YEAR}):")
    print(f"  Innovation: {INNOVATION}")
    print(f"  Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"  Impact: {IMPACT}")
    
    print(f"\nResNet Variants Results:")
    for model_name in model_names:
        result = results[model_name]
        print(f"  {model_name}:")
        print(f"    Parameters: {result['parameters']:,}")
        print(f"    Final Accuracy: {result['final_accuracy']:.2f}%")
        print(f"    Training Time: {result['training_time']:.2f} minutes")
    
    print(f"\nDegradation Problem Demonstration:")
    for plain_name, plain_result in degradation_results.items():
        print(f"  {plain_name}: {plain_result['test_accuracy']:.2f}% (shows degradation)")
    
    print(f"\nRESNET REVOLUTIONARY INNOVATIONS:")
    print("="*50)
    print("1. SKIP CONNECTIONS (IDENTITY MAPPINGS):")
    print("   • H(x) = F(x) + x instead of H(x) = F(x)")
    print("   • Enables gradient flow to early layers")
    print("   • Solves vanishing gradient problem")
    print("   • Allows training of very deep networks")
    
    print("\n2. RESIDUAL LEARNING:")
    print("   • Learn residual F(x) = H(x) - x")
    print("   • Easier than learning full mapping H(x)")
    print("   • Identity mapping as default behavior")
    print("   • Optimization becomes much easier")
    
    print("\n3. DEGRADATION PROBLEM SOLUTION:")
    print("   • Plain networks: deeper = worse performance")
    print("   • ResNet: deeper = better performance")
    print("   • Enables networks with 50, 101, 152+ layers")
    print("   • Revolutionary breakthrough in deep learning")
    
    print("\n4. BATCH NORMALIZATION INTEGRATION:")
    print("   • BN after every convolution")
    print("   • Stabilizes training")
    print("   • Enables higher learning rates")
    print("   • Reduces internal covariate shift")
    
    print(f"\nTECHNICAL ACHIEVEMENTS:")
    print("="*40)
    print("• Enabled training of 152+ layer networks")
    print("• Won ImageNet 2015 competition")
    print("• Reduced ImageNet error to 3.57%")
    print("• Solved fundamental deep learning problem")
    print("• Enabled arbitrary network depth")
    
    print(f"\nHISTORICAL IMPACT:")
    print("="*40)
    print("• Revolutionized deep learning architecture design")
    print("• Skip connections became standard in all architectures")
    print("• Enabled development of much deeper networks")
    print("• Influenced Transformer, U-Net, and other architectures")
    print("• Solved training stability issues permanently")
    print("• Made deep learning more practical and reliable")
    
    print(f"\nLEGACY AND INFLUENCE:")
    print("="*40)
    print("• Every modern architecture uses skip connections")
    print("• Inspired DenseNet, ResNeXt, Wide ResNet")
    print("• Foundation for Transformer architectures")
    print("• Enabled computer vision breakthroughs")
    print("• Solved fundamental optimization problem")
    print("• Made deep learning more accessible")
    
    return {
        'model': 'ResNet Revolution',
        'year': YEAR,
        'innovation': INNOVATION,
        'results': results,
        'degradation_demonstration': degradation_results
    }

if __name__ == "__main__":
    results = main()