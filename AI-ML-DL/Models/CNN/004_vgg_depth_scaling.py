"""
ERA 2: IMAGENET REVOLUTION - VGG Depth Scaling
==============================================

Year: 2014
Paper: "Very Deep Convolutional Networks for Large-Scale Image Recognition" (Simonyan & Zisserman, 2014)
Innovation: Systematic depth scaling with small 3x3 filters, very deep networks (16-19 layers)
Previous Limitation: Limited network depth, large filter sizes, unclear scaling principles
Performance Gain: Showed depth importance, established small filter paradigm
Impact: Proved "deeper is better" principle, influenced all subsequent architectures

This file implements the VGG architecture that systematically studied network depth and
established the paradigm of using small 3x3 convolutional filters in very deep networks.
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

YEAR = "2014"
INNOVATION = "Systematic depth scaling with small 3x3 filters, very deep networks"
PREVIOUS_LIMITATION = "Unclear scaling principles, large filters, limited depth understanding"
IMPACT = "Established 'deeper is better' and small filter paradigms"

print(f"=== VGG Depth Scaling ({YEAR}) ===")
print(f"Innovation: {INNOVATION}")
print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
print(f"Impact: {IMPACT}")
print("="*60)

# ============================================================================
# STANDARDIZED DATASET LOADING (CIFAR-10)
# ============================================================================

def load_cifar10_dataset():
    """
    Load CIFAR-10 dataset with VGG-style preprocessing
    Emphasizes the role of proper preprocessing in very deep networks
    """
    print("Loading CIFAR-10 dataset for VGG depth scaling study...")
    
    # VGG-style preprocessing with strong augmentation
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
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
# VGG ARCHITECTURE CONFIGURATIONS
# ============================================================================

# VGG configurations (adapted for CIFAR-10)
VGG_CONFIGS = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG_DepthScaling(nn.Module):
    """
    VGG architecture with systematic depth scaling study
    
    Key VGG Innovations:
    - Very small (3x3) convolutional filters throughout
    - Systematic depth scaling (11, 13, 16, 19 layers)
    - Homogeneous architecture with simple design rules
    - Demonstrated that depth matters more than filter size
    - Established receptive field equivalence principle
    """
    
    def __init__(self, vgg_type='VGG16', num_classes=10):
        super(VGG_DepthScaling, self).__init__()
        
        self.vgg_type = vgg_type
        print(f"Building VGG {vgg_type} Architecture...")
        
        # Build feature extraction layers
        self.features = self._make_layers(VGG_CONFIGS[vgg_type])
        
        # Adaptive pooling to handle CIFAR-10 size
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # VGG classifier (3 FC layers with dropout)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
        
        # Initialize weights using VGG method
        self._initialize_vgg_weights()
        
        # Calculate depth statistics
        conv_layers = sum(1 for x in VGG_CONFIGS[vgg_type] if isinstance(x, int))
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"VGG {vgg_type} Architecture Summary:")
        print(f"  Input: 32x32x3 (CIFAR-10 RGB)")
        print(f"  Convolutional Layers: {conv_layers}")
        print(f"  All Conv Filters: 3x3 (Revolutionary small size)")
        print(f"  Max Pooling Layers: {VGG_CONFIGS[vgg_type].count('M')}")
        print(f"  Fully Connected Layers: 3")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Key Innovation: Systematic depth scaling")
    
    def _make_layers(self, config):
        """Build VGG feature layers from configuration"""
        layers = []
        in_channels = 3
        
        for x in config:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # All conv layers use 3x3 filters (VGG innovation)
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        
        return nn.Sequential(*layers)
    
    def _initialize_vgg_weights(self):
        """Initialize weights using VGG method"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Xavier initialization for conv layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Normal initialization for FC layers
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 1)
    
    def forward(self, x):
        """Forward pass through VGG architecture"""
        # Feature extraction with 3x3 conv layers
        x = self.features(x)
        
        # Adaptive pooling
        x = self.avgpool(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def get_depth_analysis(self):
        """Analyze network depth and receptive field"""
        conv_count = 0
        receptive_field = 1
        stride_product = 1
        
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                conv_count += 1
                # 3x3 conv with padding=1 increases receptive field by 2
                receptive_field += 2 * stride_product
            elif isinstance(layer, nn.MaxPool2d):
                # 2x2 max pool doubles the stride
                stride_product *= 2
        
        return {
            'conv_layers': conv_count,
            'max_pools': VGG_CONFIGS[self.vgg_type].count('M'),
            'receptive_field': receptive_field,
            'effective_stride': stride_product
        }

# ============================================================================
# DEPTH SCALING COMPARISON
# ============================================================================

def compare_filter_sizes(train_loader, test_loader):
    """
    Compare 3x3 filters vs larger filters to show VGG innovation
    Demonstrates why small filters are better
    """
    print("\nComparing Filter Sizes: VGG 3x3 vs Traditional Large Filters...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    # Test different filter configurations
    class FilterTestNet(nn.Module):
        def __init__(self, filter_config):
            super().__init__()
            
            if filter_config == '3x3_stack':
                # VGG approach: stack of 3x3 filters
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(True),
                    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True),
                    nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True),
                    nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True),
                    nn.MaxPool2d(2),
                )
                classifier_input = 256 * 4 * 4
            else:
                # Traditional approach: larger filters
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 7, padding=3), nn.ReLU(True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 5, padding=2), nn.ReLU(True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 5, padding=2), nn.ReLU(True),
                    nn.MaxPool2d(2),
                )
                classifier_input = 256 * 4 * 4
            
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input, 512),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(512, 10)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
    
    filter_configs = {
        'VGG 3x3 Stack': '3x3_stack',
        'Traditional Large': 'large_filters'
    }
    
    for config_name, config_type in filter_configs.items():
        print(f"\nTesting {config_name}...")
        
        model = FilterTestNet(config_type).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        # Quick training comparison
        model.train()
        total_loss = 0
        for epoch in range(8):
            for batch_idx, (data, targets) in enumerate(train_loader):
                if batch_idx > 25:
                    break
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        # Test accuracy
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
        
        test_acc = 100. * correct / total
        param_count = sum(p.numel() for p in model.parameters())
        
        results[config_name] = {
            'test_accuracy': test_acc,
            'parameters': param_count,
            'avg_loss': total_loss / (8 * 25)
        }
        print(f"{config_name}: {test_acc:.2f}% accuracy, {param_count:,} parameters")
    
    return results

# ============================================================================
# TRAINING FUNCTION WITH VGG TECHNIQUES
# ============================================================================

def train_vgg_depth_scaling(model, train_loader, test_loader, epochs=80, learning_rate=0.01):
    """
    Train VGG with systematic depth scaling approach
    Uses techniques from the original VGG paper
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # VGG training configuration
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=5e-4
    )
    
    # Multi-step learning rate decay
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
    
    criterion = nn.CrossEntropyLoss()
    
    # Training tracking
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print(f"Training VGG {model.vgg_type} on device: {device}")
    print("Using VGG systematic training approach...")
    
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
            
            # Gradient clipping for very deep networks
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_train_acc = 100. * correct_train / total_train
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_train_acc)
        
        # Test evaluation
        test_acc = evaluate_vgg_model(model, test_loader, device)
        test_accuracies.append(test_acc)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), f'AI-ML-DL/Models/CNN/vgg_{model.vgg_type.lower()}_best.pth')
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {epoch_loss:.4f}, '
              f'Train Acc: {epoch_train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        # Early stopping for very deep networks
        if epoch > 40 and test_acc > 91.0:
            print(f"Excellent performance reached at epoch {epoch+1}")
            break
    
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    return train_losses, train_accuracies, test_accuracies

def evaluate_vgg_model(model, test_loader, device):
    """Evaluate VGG model on test set"""
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
# VISUALIZATION AND ANALYSIS
# ============================================================================

def visualize_depth_scaling_study():
    """Visualize VGG's systematic depth scaling study"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # VGG depth progression
    ax = axes[0, 0]
    vgg_variants = ['VGG-11', 'VGG-13', 'VGG-16', 'VGG-19']
    conv_layers = [8, 10, 13, 16]  # Actual conv layer counts
    colors = ['#3498DB', '#E67E22', '#E74C3C', '#9B59B6']
    
    bars = ax.bar(vgg_variants, conv_layers, color=colors)
    ax.set_title('VGG Systematic Depth Scaling', fontsize=14)
    ax.set_ylabel('Convolutional Layers')
    for bar, layers in zip(bars, conv_layers):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                f'{layers}', ha='center', va='bottom')
    
    # Filter size comparison
    ax = axes[0, 1]
    filter_approaches = ['Large Filters\n(7x7, 5x5)', 'VGG Small Filters\n(3x3 only)']
    receptive_fields = [7, 7]  # Same receptive field
    parameter_efficiency = [1.0, 0.7]  # VGG is more efficient
    
    x = np.arange(len(filter_approaches))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, receptive_fields, width, label='Receptive Field', color='#3498DB')
    bars2 = ax.bar(x + width/2, [p*10 for p in parameter_efficiency], width, 
                   label='Parameter Efficiency (×10)', color='#E74C3C')
    
    ax.set_title('Filter Size Innovation', fontsize=14)
    ax.set_ylabel('Value')
    ax.set_xticks(x)
    ax.set_xticklabels(filter_approaches)
    ax.legend()
    
    # Depth vs Performance (theoretical)
    ax = axes[1, 0]
    network_depths = [8, 11, 16, 19, 25]  # Including deeper theoretical
    imagenet_performance = [85, 88, 92, 92.7, 92]  # Performance plateau
    
    ax.plot(network_depths, imagenet_performance, 'ro-', linewidth=2, markersize=8)
    ax.set_title('Depth vs Performance Relationship', fontsize=14)
    ax.set_xlabel('Network Depth (Conv Layers)')
    ax.set_ylabel('ImageNet Top-5 Accuracy (%)')
    ax.grid(True, alpha=0.3)
    
    # Highlight VGG sweet spot
    ax.axvspan(13, 19, alpha=0.3, color='green', label='VGG Sweet Spot')
    ax.legend()
    
    # Historical impact timeline
    ax = axes[1, 1]
    years = ['2012\nAlexNet', '2013\nZFNet', '2014\nVGG', '2014\nGoogLeNet']
    depth_innovation = [8, 8, 19, 22]
    bars = ax.bar(years, depth_innovation, color=['#95A5A6', '#BDC3C7', '#E74C3C', '#27AE60'])
    ax.set_title('Deep Learning Depth Evolution', fontsize=14)
    ax.set_ylabel('Network Depth (Layers)')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/CNN/004_vgg_depth_scaling_analysis.png', dpi=300, bbox_inches='tight')
    print("VGG depth scaling analysis saved: 004_vgg_depth_scaling_analysis.png")

def analyze_receptive_field_equivalence():
    """Analyze VGG's receptive field equivalence principle"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Single 7x7 vs Two 3x3 layers
    configurations = [
        {'name': 'Single 7×7 Conv', 'layers': [7], 'params': 7*7, 'nonlinearities': 1},
        {'name': 'Two 3×3 Conv', 'layers': [3, 3], 'params': 2*3*3, 'nonlinearities': 2},
        {'name': 'Three 3×3 Conv', 'layers': [3, 3, 3], 'params': 3*3*3, 'nonlinearities': 3}
    ]
    
    # Receptive field comparison
    ax = axes[0]
    names = [config['name'] for config in configurations]
    receptive_fields = [7, 7, 7]  # All have same receptive field
    bars = ax.bar(names, receptive_fields, color=['#95A5A6', '#3498DB', '#E74C3C'])
    ax.set_title('Receptive Field Equivalence', fontsize=14)
    ax.set_ylabel('Effective Receptive Field')
    ax.tick_params(axis='x', rotation=45)
    
    # Parameter comparison
    ax = axes[1]
    param_counts = [config['params'] for config in configurations]
    bars = ax.bar(names, param_counts, color=['#95A5A6', '#3498DB', '#E74C3C'])
    ax.set_title('Parameter Count Comparison', fontsize=14)
    ax.set_ylabel('Parameters per Channel')
    ax.tick_params(axis='x', rotation=45)
    
    for bar, params in zip(bars, param_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{params}', ha='center', va='bottom')
    
    # Non-linearity comparison
    ax = axes[2]
    nonlinearities = [config['nonlinearities'] for config in configurations]
    bars = ax.bar(names, nonlinearities, color=['#95A5A6', '#3498DB', '#E74C3C'])
    ax.set_title('Non-linearity Functions', fontsize=14)
    ax.set_ylabel('Number of ReLU Activations')
    ax.tick_params(axis='x', rotation=45)
    
    for bar, nl in zip(bars, nonlinearities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{nl}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/CNN/004_vgg_receptive_field_analysis.png', dpi=300, bbox_inches='tight')
    print("VGG receptive field analysis saved: 004_vgg_receptive_field_analysis.png")

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
    print(f"=== VGG Depth Scaling Implementation ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load CIFAR-10 dataset
    train_loader, test_loader, classes = load_cifar10_dataset()
    
    # Test different VGG depths
    vgg_variants = ['VGG11', 'VGG16']  # Test two variants for demonstration
    results = {}
    
    for vgg_type in vgg_variants:
        print(f"\n{'='*50}")
        print(f"Training {vgg_type}")
        print('='*50)
        
        # Initialize VGG model
        model = VGG_DepthScaling(vgg_type=vgg_type, num_classes=10)
        
        # Analyze depth characteristics
        depth_analysis = model.get_depth_analysis()
        print(f"{vgg_type} Depth Analysis:")
        print(f"  Convolutional Layers: {depth_analysis['conv_layers']}")
        print(f"  Max Pool Layers: {depth_analysis['max_pools']}")
        print(f"  Receptive Field: {depth_analysis['receptive_field']}")
        
        # Train model
        metrics = track_training_metrics(
            vgg_type,
            train_vgg_depth_scaling,
            model, train_loader, test_loader, 40, 0.01  # Reduced epochs for demo
        )
        
        train_losses, train_accuracies, test_accuracies = metrics['result']
        
        results[vgg_type] = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'final_accuracy': test_accuracies[-1] if test_accuracies else 0,
            'training_time': metrics['training_time_minutes'],
            'memory_usage': metrics['memory_usage_mb'],
            'parameters': sum(p.numel() for p in model.parameters()),
            'depth_analysis': depth_analysis
        }
        
        print(f"{vgg_type} Results:")
        print(f"  Final Test Accuracy: {test_accuracies[-1]:.2f}%")
        print(f"  Training Time: {metrics['training_time_minutes']:.2f} minutes")
        print(f"  Parameters: {results[vgg_type]['parameters']:,}")
    
    # Compare filter sizes
    filter_results = compare_filter_sizes(train_loader, test_loader)
    
    # Generate visualizations
    print("\nGenerating VGG analysis visualizations...")
    visualize_depth_scaling_study()
    analyze_receptive_field_equivalence()
    
    # Create comprehensive results visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # VGG depth comparison
    ax = axes[0, 0]
    vgg_names = list(results.keys())
    final_accs = [results[name]['final_accuracy'] for name in vgg_names]
    colors = ['#3498DB', '#E74C3C']
    
    bars = ax.bar(vgg_names, final_accs, color=colors)
    ax.set_title('VGG Depth Scaling Results', fontsize=14)
    ax.set_ylabel('Final Test Accuracy (%)')
    for bar, acc in zip(bars, final_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # Parameter scaling
    ax = axes[0, 1]
    param_counts = [results[name]['parameters']/1e6 for name in vgg_names]
    bars = ax.bar(vgg_names, param_counts, color=colors)
    ax.set_title('VGG Parameter Scaling', fontsize=14)
    ax.set_ylabel('Parameters (Millions)')
    for bar, params in zip(bars, param_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{params:.1f}M', ha='center', va='bottom')
    
    # Training curves comparison
    ax = axes[1, 0]
    for vgg_name, color in zip(vgg_names, colors):
        test_accs = results[vgg_name]['test_accuracies']
        epochs = range(1, len(test_accs) + 1)
        ax.plot(epochs, test_accs, color=color, label=vgg_name, linewidth=2)
    
    ax.set_title('VGG Training Progression', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Filter size comparison
    ax = axes[1, 1]
    filter_names = list(filter_results.keys())
    filter_accs = [filter_results[name]['test_accuracy'] for name in filter_names]
    bars = ax.bar(filter_names, filter_accs, color=['#27AE60', '#95A5A6'])
    ax.set_title('Filter Size Innovation Impact', fontsize=14)
    ax.set_ylabel('Test Accuracy (%)')
    for bar, acc in zip(bars, filter_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                f'{acc:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/CNN/004_vgg_depth_scaling_results.png', dpi=300, bbox_inches='tight')
    print("Comprehensive VGG results saved: 004_vgg_depth_scaling_results.png")
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("VGG DEPTH SCALING SUMMARY")
    print("="*70)
    
    print(f"\nHistorical Context ({YEAR}):")
    print(f"  Innovation: {INNOVATION}")
    print(f"  Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"  Impact: {IMPACT}")
    
    print(f"\nDepth Scaling Results:")
    for vgg_name in vgg_names:
        result = results[vgg_name]
        print(f"  {vgg_name}:")
        print(f"    Conv Layers: {result['depth_analysis']['conv_layers']}")
        print(f"    Parameters: {result['parameters']:,}")
        print(f"    Final Accuracy: {result['final_accuracy']:.2f}%")
        print(f"    Training Time: {result['training_time']:.2f} minutes")
    
    print(f"\nFilter Size Innovation Results:")
    for filter_name, filter_result in filter_results.items():
        print(f"  {filter_name}:")
        print(f"    Test Accuracy: {filter_result['test_accuracy']:.2f}%")
        print(f"    Parameters: {filter_result['parameters']:,}")
    
    print(f"\nVGG KEY INNOVATIONS:")
    print("="*40)
    print("1. SYSTEMATIC DEPTH SCALING:")
    print("   • Studied effect of network depth systematically")
    print("   • Variants: VGG-11, VGG-13, VGG-16, VGG-19")
    print("   • Proved 'deeper is better' principle")
    print("   • Established optimal depth ranges")
    
    print("\n2. SMALL FILTER PARADIGM:")
    print("   • Exclusive use of 3×3 convolutional filters")
    print("   • Replaced large 7×7 and 5×5 filters")
    print("   • Same receptive field with fewer parameters")
    print("   • More non-linearities for same receptive field")
    
    print("\n3. RECEPTIVE FIELD EQUIVALENCE:")
    print("   • Two 3×3 conv = One 7×7 conv (same receptive field)")
    print("   • But 2×(3²) = 18 params vs 7² = 49 params")
    print("   • More non-linear transformations")
    print("   • Better representational capacity")
    
    print("\n4. HOMOGENEOUS ARCHITECTURE:")
    print("   • Simple and uniform design rules")
    print("   • All conv layers use 3×3 filters")
    print("   • Systematic channel doubling")
    print("   • Easy to understand and reproduce")
    
    print(f"\nTECHNICAL INSIGHTS:")
    print("="*40)
    print("• Small filters are more parameter efficient")
    print("• Depth increases representational power")
    print("• More non-linearities improve learning")
    print("• Systematic scaling enables fair comparison")
    print("• Homogeneous design aids understanding")
    
    print(f"\nHISTORICAL IMPACT:")
    print("="*40)
    print("• Established systematic architecture research")
    print("• Proved importance of network depth")
    print("• Popularized 3×3 filter standard")
    print("• Influenced all subsequent CNN designs")
    print("• Enabled deeper network development")
    print("• Simplified architecture design principles")
    
    print(f"\nLIMITATIONS (addressed by later work):")
    print("="*40)
    print("• Very deep networks still hard to train")
    print("• Vanishing gradient problem persists")
    print("• No skip connections")
    print("• Computationally expensive")
    print("• Limited by gradient flow")
    
    return {
        'model': 'VGG Depth Scaling',
        'year': YEAR,
        'innovation': INNOVATION,
        'results': results,
        'filter_comparison': filter_results
    }

if __name__ == "__main__":
    results = main()