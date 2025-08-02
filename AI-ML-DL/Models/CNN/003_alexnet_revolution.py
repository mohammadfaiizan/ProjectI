"""
ERA 2: IMAGENET REVOLUTION - AlexNet Revolution
===============================================

Year: 2012
Paper: "ImageNet Classification with Deep Convolutional Neural Networks" (Krizhevsky et al., 2012)
Innovation: Deep CNN with ReLU activations, dropout, GPU training, data augmentation
Previous Limitation: Shallow networks, saturating activations, limited computational power
Performance Gain: Massive improvement on ImageNet, sparked deep learning revolution
Impact: Demonstrated viability of deep learning, launched modern AI era

This file implements the revolutionary AlexNet architecture that won ImageNet 2012 and
sparked the deep learning revolution by solving key training challenges of early deep networks.
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

YEAR = "2012"
INNOVATION = "Deep CNN with ReLU, dropout, GPU training, breakthrough on ImageNet"
PREVIOUS_LIMITATION = "Shallow networks, saturating activations, limited scale, vanishing gradients"
IMPACT = "Launched deep learning revolution, proved scalability of deep CNNs"

print(f"=== AlexNet Revolution ({YEAR}) ===")
print(f"Innovation: {INNOVATION}")
print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
print(f"Impact: {IMPACT}")
print("="*60)

# ============================================================================
# STANDARDIZED DATASET LOADING (CIFAR-10)
# ============================================================================

def load_cifar10_dataset():
    """
    Load CIFAR-10 dataset with AlexNet-style preprocessing
    Includes aggressive data augmentation that was revolutionary in 2012
    """
    print("Loading CIFAR-10 dataset with AlexNet-style augmentation...")
    
    # AlexNet-style aggressive data augmentation
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
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
    print("Revolutionary data augmentation applied!")
    
    return train_loader, test_loader, classes

# ============================================================================
# AlexNet ARCHITECTURE (ADAPTED FOR CIFAR-10)
# ============================================================================

class AlexNet_Revolution(nn.Module):
    """
    AlexNet architecture adapted for CIFAR-10 (32x32 RGB → 10 classes)
    
    Original AlexNet was designed for 224x224 ImageNet images
    This adaptation maintains the revolutionary innovations while
    working with smaller CIFAR-10 images
    
    Revolutionary Innovations:
    - ReLU activations (instead of tanh/sigmoid)
    - Dropout regularization
    - Local Response Normalization (LRN)
    - Large convolutional filters
    - Deep architecture (8 layers)
    - GPU-optimized training
    """
    
    def __init__(self, num_classes=10):
        super(AlexNet_Revolution, self).__init__()
        
        print("Building AlexNet Revolution Architecture...")
        
        # Feature extraction layers (5 convolutional layers)
        self.features = nn.Sequential(
            # Conv1: Large receptive field for feature detection
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # Adapted for 32x32
            nn.ReLU(inplace=True),  # REVOLUTIONARY: ReLU instead of tanh
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),  # LRN
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2: Increase depth
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3: Deeper feature extraction
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4: More complex features
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5: High-level features
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Adaptive pooling to handle different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Classification layers (3 fully connected layers)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # REVOLUTIONARY: Dropout regularization
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(0.5),  # Heavy dropout for regularization
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes),  # Output layer
        )
        
        # Initialize weights using AlexNet method
        self._initialize_alexnet_weights()
        
        print(f"AlexNet Architecture Summary:")
        print(f"  Input: 32x32x3 (CIFAR-10 RGB)")
        print(f"  Convolutional Layers: 5")
        print(f"  Fully Connected Layers: 3")
        print(f"  Activation: ReLU (Revolutionary!)")
        print(f"  Regularization: Dropout + LRN")
        print(f"  Output: {num_classes} classes")
    
    def _initialize_alexnet_weights(self):
        """Initialize weights using AlexNet's method (improved from early networks)"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Xavier/Glorot initialization for ReLU
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Normal initialization with smaller std for FC layers
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 1)  # Bias = 1 for ReLU
    
    def forward(self, x):
        """
        Forward pass through AlexNet architecture
        Demonstrates revolutionary deep CNN with ReLU activations
        """
        # Feature extraction
        x = self.features(x)
        
        # Adaptive pooling
        x = self.avgpool(x)
        
        # Flatten for classifier
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def get_feature_maps(self, x):
        """
        Extract feature maps at different layers for visualization
        Shows the depth and complexity of learned features
        """
        features = {}
        
        # Process through feature layers
        x = x
        layer_idx = 0
        
        for name, layer in self.features.named_children():
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                features[f'conv_{layer_idx}'] = x.clone()
                layer_idx += 1
            elif isinstance(layer, nn.ReLU):
                features[f'relu_{layer_idx-1}'] = x.clone()
        
        return features

# ============================================================================
# COMPARISON MODEL: PRE-ALEXNET BASELINE
# ============================================================================

class PreAlexNetBaseline(nn.Module):
    """
    Baseline model representing pre-AlexNet approaches
    Uses old techniques to show the revolutionary impact
    """
    
    def __init__(self, num_classes=10):
        super(PreAlexNetBaseline, self).__init__()
        
        # Shallow architecture with old techniques
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.Tanh(),  # Old activation
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 5, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.Tanh(),
            nn.Linear(512, num_classes)
        )
        
        # Old initialization
        self._initialize_old_method()
    
    def _initialize_old_method(self):
        """Old initialization method"""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(module.weight, 0, 0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ============================================================================
# TRAINING FUNCTION WITH ALEXNET INNOVATIONS
# ============================================================================

def train_alexnet_revolution(model, train_loader, test_loader, epochs=100, learning_rate=0.01):
    """
    Train AlexNet with revolutionary techniques from 2012
    Includes innovations like adaptive learning rate, momentum, and weight decay
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # AlexNet optimizer configuration (revolutionary for 2012)
    optimizer = optim.SGD(
        model.parameters(), 
        lr=learning_rate, 
        momentum=0.9,      # Heavy momentum
        weight_decay=5e-4  # L2 regularization
    )
    
    # Learning rate scheduling (adaptive LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training tracking
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    learning_rates = []
    
    print(f"Training AlexNet Revolution on device: {device}")
    print("Using revolutionary 2012 techniques: ReLU + Dropout + Momentum + LR scheduling")
    
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (stability improvement)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            # Update weights
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
        current_lr = optimizer.param_groups[0]['lr']
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_train_acc)
        learning_rates.append(current_lr)
        
        # Test evaluation
        test_acc = evaluate_alexnet_revolution(model, test_loader, device)
        test_accuracies.append(test_acc)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'AI-ML-DL/Models/CNN/alexnet_revolution_best.pth')
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {epoch_loss:.4f}, '
              f'Train Acc: {epoch_train_acc:.2f}%, Test Acc: {test_acc:.2f}%, LR: {current_lr:.6f}')
        
        # Early stopping if reaching high accuracy
        if test_acc > 90.0:  # AlexNet can achieve >90% on CIFAR-10
            print(f"Excellent performance reached at epoch {epoch+1}")
            break
    
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    return train_losses, train_accuracies, test_accuracies, learning_rates

def evaluate_alexnet_revolution(model, test_loader, device):
    """Evaluate AlexNet on test set"""
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
# REVOLUTIONARY INNOVATIONS ANALYSIS
# ============================================================================

def compare_activations_impact(train_loader, test_loader):
    """
    Compare ReLU vs Tanh to show revolutionary impact
    """
    print("\nComparing Revolutionary ReLU vs Traditional Tanh...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    # Test different activations
    class ActivationTestNet(nn.Module):
        def __init__(self, activation_fn):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                activation_fn,
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                activation_fn,
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                activation_fn,
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(256 * 4 * 4, 512),
                activation_fn,
                nn.Linear(512, 10)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
    
    activations = {
        'ReLU (Revolutionary)': nn.ReLU(inplace=True),
        'Tanh (Traditional)': nn.Tanh()
    }
    
    for activation_name, activation_fn in activations.items():
        print(f"\nTesting {activation_name}...")
        
        model = ActivationTestNet(activation_fn).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        # Quick training comparison
        model.train()
        total_loss = 0
        for epoch in range(5):  # Quick test
            for batch_idx, (data, targets) in enumerate(train_loader):
                if batch_idx > 20:  # Limited batches
                    break
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        # Test accuracy
        test_acc = evaluate_alexnet_revolution(model, test_loader, device)
        results[activation_name] = {
            'test_accuracy': test_acc,
            'avg_loss': total_loss / (5 * 20)
        }
        print(f"{activation_name}: {test_acc:.2f}% accuracy")
    
    return results

def analyze_dropout_impact(train_loader, test_loader):
    """
    Analyze the revolutionary impact of dropout regularization
    """
    print("\nAnalyzing Revolutionary Dropout Impact...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    # Test with and without dropout
    dropout_configs = {
        'With Dropout (Revolutionary)': 0.5,
        'Without Dropout (Traditional)': 0.0
    }
    
    for config_name, dropout_rate in dropout_configs.items():
        print(f"\nTesting {config_name}...")
        
        class DropoutTestNet(nn.Module):
            def __init__(self, dropout_rate):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
                self.classifier = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(128 * 8 * 8, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_rate),
                    nn.Linear(512, 10)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x
        
        model = DropoutTestNet(dropout_rate).to(device)
        
        # Quick training
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        train_accs = []
        for epoch in range(10):
            model.train()
            correct = 0
            total = 0
            for batch_idx, (data, targets) in enumerate(train_loader):
                if batch_idx > 30:
                    break
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_acc = 100. * correct / total
            train_accs.append(train_acc)
        
        test_acc = evaluate_alexnet_revolution(model, test_loader, device)
        overfitting = max(train_accs) - test_acc
        
        results[config_name] = {
            'test_accuracy': test_acc,
            'max_train_accuracy': max(train_accs),
            'overfitting_gap': overfitting
        }
        print(f"{config_name}: Test {test_acc:.2f}%, Overfitting Gap: {overfitting:.2f}%")
    
    return results

# ============================================================================
# VISUALIZATION AND ANALYSIS
# ============================================================================

def visualize_alexnet_revolution():
    """Visualize AlexNet's revolutionary impact"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # ImageNet revolution timeline
    ax = axes[0, 0]
    years = ['2010\n(Traditional)', '2011\n(Pre-AlexNet)', '2012\n(AlexNet)', '2013\n(Post-Revolution)']
    imagenet_errors = [28.2, 25.8, 15.3, 11.7]  # Top-5 error rates
    colors = ['#95A5A6', '#BDC3C7', '#E74C3C', '#27AE60']
    
    bars = ax.bar(years, imagenet_errors, color=colors)
    ax.set_title('ImageNet Revolution: Error Rate Drop', fontsize=14)
    ax.set_ylabel('Top-5 Error Rate (%)')
    for bar, error in zip(bars, imagenet_errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{error}%', ha='center', va='bottom')
    
    # AlexNet innovations
    ax = axes[0, 1]
    innovations = ['ReLU\nActivation', 'Dropout\nRegularization', 'Data\nAugmentation', 
                   'GPU\nTraining', 'Deep\nArchitecture']
    impact_scores = [10, 9, 8, 8, 9]
    bars = ax.bar(innovations, impact_scores, color=['#3498DB', '#E67E22', '#9B59B6', '#1ABC9C', '#E74C3C'])
    ax.set_title('AlexNet Revolutionary Innovations', fontsize=14)
    ax.set_ylabel('Impact Score (1-10)')
    ax.tick_params(axis='x', rotation=45)
    
    # Training difficulty comparison
    ax = axes[1, 0]
    approaches = ['Pre-2012\n(Traditional)', 'AlexNet\n(Revolutionary)', 'Post-2012\n(Modern)']
    training_difficulty = [9, 4, 2]
    bars = ax.bar(approaches, training_difficulty, color=['#E74C3C', '#F39C12', '#27AE60'])
    ax.set_title('Training Difficulty Reduction', fontsize=14)
    ax.set_ylabel('Training Difficulty (1-10)')
    
    # Performance scaling
    ax = axes[1, 1]
    model_sizes = ['LeNet\n(60K)', 'Early Deep\n(200K)', 'AlexNet\n(60M)', 'Modern\n(25M+)']
    performance_scores = [6, 6.5, 8.5, 9.5]
    bars = ax.bar(model_sizes, performance_scores, color=['#3498DB', '#E67E22', '#E74C3C', '#27AE60'])
    ax.set_title('Model Scale vs Performance', fontsize=14)
    ax.set_ylabel('Performance Score (1-10)')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/CNN/003_alexnet_revolution_impact.png', dpi=300, bbox_inches='tight')
    print("AlexNet revolution impact analysis saved: 003_alexnet_revolution_impact.png")

def visualize_feature_progression(model, test_loader, device, classes):
    """Visualize feature learning progression in AlexNet"""
    model.eval()
    
    # Get a batch of test images
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    # Get feature maps
    with torch.no_grad():
        features = model.get_feature_maps(images[:1])
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Original image
    img = images[0].cpu()
    img = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1) + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    img = torch.clamp(img, 0, 1)
    
    axes[0, 0].imshow(img.permute(1, 2, 0))
    axes[0, 0].set_title(f'Input Image\nClass: {classes[labels[0]]}', fontsize=12)
    axes[0, 0].axis('off')
    
    # Feature maps at different layers
    feature_layers = ['conv_0', 'conv_1', 'conv_2', 'conv_3', 'conv_4']
    cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    
    for i, (layer_name, cmap) in enumerate(zip(feature_layers, cmaps)):
        if layer_name in features:
            feature_map = features[layer_name][0].cpu()
            
            # Show first few channels
            for j in range(min(3, feature_map.size(0))):
                row = (i + 1) // 4
                col = (i + 1) % 4
                if row < 3 and col < 4:
                    if j == 0:  # Only show first channel per layer
                        axes[row, col].imshow(feature_map[j], cmap=cmap)
                        axes[row, col].set_title(f'Layer {i+1} Features', fontsize=10)
                        axes[row, col].axis('off')
    
    # Fill remaining subplots with layer information
    layer_info = [
        "Conv1: Large filters\nEdge detection",
        "Conv2: Corner/texture\ndetection", 
        "Conv3: Shape/pattern\nrecognition",
        "Conv4: Object parts\nidentification",
        "Conv5: High-level\nfeatures"
    ]
    
    for i, info in enumerate(layer_info):
        row = 2
        col = i % 4
        if row < 3 and col < 4 and i >= 3:
            axes[row, col].text(0.5, 0.5, info, ha='center', va='center', 
                               transform=axes[row, col].transAxes, fontsize=10)
            axes[row, col].axis('off')
    
    # Remove empty subplots
    for i in range(3):
        for j in range(4):
            if i * 4 + j >= len(feature_layers) + 1:
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/CNN/003_alexnet_feature_progression.png', dpi=300, bbox_inches='tight')
    print("AlexNet feature progression saved: 003_alexnet_feature_progression.png")

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
    print(f"=== AlexNet Revolution Implementation ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load CIFAR-10 dataset
    train_loader, test_loader, classes = load_cifar10_dataset()
    
    # Initialize AlexNet model
    alexnet_model = AlexNet_Revolution(num_classes=10)
    
    # Initialize baseline for comparison
    baseline_model = PreAlexNetBaseline(num_classes=10)
    
    # Compare model complexities
    alexnet_params = sum(p.numel() for p in alexnet_model.parameters())
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    
    print(f"\nModel Complexity Comparison:")
    print(f"  AlexNet Parameters: {alexnet_params:,}")
    print(f"  Pre-AlexNet Baseline: {baseline_params:,}")
    print(f"  Parameter Increase: {alexnet_params/baseline_params:.1f}x")
    
    # Train AlexNet
    print("\nTraining AlexNet Revolution...")
    alexnet_metrics = track_training_metrics(
        'AlexNet Revolution',
        train_alexnet_revolution,
        alexnet_model, train_loader, test_loader, 50, 0.01
    )
    
    alexnet_train_losses, alexnet_train_accs, alexnet_test_accs, alexnet_lrs = alexnet_metrics['result']
    
    # Train baseline for comparison
    print("\nTraining Pre-AlexNet Baseline...")
    baseline_metrics = track_training_metrics(
        'Pre-AlexNet Baseline',
        train_alexnet_revolution,  # Same training function, different model
        baseline_model, train_loader, test_loader, 50, 0.01
    )
    
    baseline_train_losses, baseline_train_accs, baseline_test_accs, baseline_lrs = baseline_metrics['result']
    
    # Revolutionary innovations analysis
    activation_results = compare_activations_impact(train_loader, test_loader)
    dropout_results = analyze_dropout_impact(train_loader, test_loader)
    
    # Create visualizations
    print("\nGenerating revolution analysis...")
    visualize_alexnet_revolution()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visualize_feature_progression(alexnet_model, test_loader, device, classes)
    
    # Create comprehensive results visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training comparison
    ax = axes[0, 0]
    epochs = range(1, len(alexnet_test_accs) + 1)
    ax.plot(epochs, alexnet_test_accs, 'r-', label='AlexNet (Revolutionary)', linewidth=3)
    epochs_baseline = range(1, len(baseline_test_accs) + 1)
    ax.plot(epochs_baseline, baseline_test_accs, 'b--', label='Pre-AlexNet Baseline', linewidth=2)
    ax.set_title('Revolutionary Performance Improvement', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Innovation impact comparison
    ax = axes[0, 1]
    innovations = ['ReLU vs\nTanh', 'With vs\nWithout Dropout']
    relu_improvement = activation_results['ReLU (Revolutionary)']['test_accuracy'] - activation_results['Tanh (Traditional)']['test_accuracy']
    dropout_improvement = dropout_results['With Dropout (Revolutionary)']['test_accuracy'] - dropout_results['Without Dropout (Traditional)']['test_accuracy']
    improvements = [relu_improvement, dropout_improvement]
    
    bars = ax.bar(innovations, improvements, color=['#E74C3C', '#3498DB'])
    ax.set_title('Revolutionary Innovations Impact', fontsize=14)
    ax.set_ylabel('Accuracy Improvement (%)')
    for bar, imp in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'+{imp:.1f}%', ha='center', va='bottom')
    
    # Learning rate scheduling effect
    ax = axes[1, 0]
    epochs_lr = range(1, len(alexnet_lrs) + 1)
    ax.plot(epochs_lr, alexnet_lrs, 'g-', linewidth=2)
    ax.set_title('Revolutionary Learning Rate Scheduling', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Final performance comparison
    ax = axes[1, 1]
    models = ['Pre-AlexNet\nBaseline', 'AlexNet\nRevolution']
    final_accs = [baseline_test_accs[-1], alexnet_test_accs[-1]]
    bars = ax.bar(models, final_accs, color=['#95A5A6', '#E74C3C'])
    ax.set_title('Final Performance Comparison', fontsize=14)
    ax.set_ylabel('Test Accuracy (%)')
    for bar, acc in zip(bars, final_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/CNN/003_alexnet_revolution_results.png', dpi=300, bbox_inches='tight')
    print("Comprehensive AlexNet results saved: 003_alexnet_revolution_results.png")
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("ALEXNET REVOLUTION SUMMARY")
    print("="*70)
    
    print(f"\nHistorical Context ({YEAR}):")
    print(f"  Innovation: {INNOVATION}")
    print(f"  Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"  Impact: {IMPACT}")
    
    print(f"\nPerformance Comparison:")
    print(f"  AlexNet Final Accuracy: {alexnet_test_accs[-1]:.2f}%")
    print(f"  Pre-AlexNet Baseline: {baseline_test_accs[-1]:.2f}%")
    print(f"  Improvement: +{alexnet_test_accs[-1] - baseline_test_accs[-1]:.2f}%")
    
    print(f"\nComputational Analysis:")
    print(f"  AlexNet Parameters: {alexnet_params:,}")
    print(f"  Training Time: {alexnet_metrics['training_time_minutes']:.2f} minutes")
    print(f"  Memory Usage: {alexnet_metrics['memory_usage_mb']:.2f} MB")
    
    print(f"\nRevolutionary Innovation Results:")
    print(f"  ReLU vs Tanh Improvement: +{relu_improvement:.2f}%")
    print(f"  Dropout Regularization Benefit: +{dropout_improvement:.2f}%")
    print(f"  Overfitting Reduction: {dropout_results['Without Dropout (Traditional)']['overfitting_gap'] - dropout_results['With Dropout (Revolutionary)']['overfitting_gap']:.2f}%")
    
    print(f"\nALEXNET REVOLUTIONARY INNOVATIONS:")
    print("="*50)
    print("1. RELU ACTIVATION FUNCTION:")
    print("   • Solves vanishing gradient problem")
    print("   • Non-saturating activation")
    print("   • Faster training convergence")
    print("   • Enables deeper networks")
    
    print("\n2. DROPOUT REGULARIZATION:")
    print("   • Prevents overfitting in large networks")
    print("   • Random neuron deactivation during training")
    print("   • Improves generalization")
    print("   • Enables training of very large models")
    
    print("\n3. DATA AUGMENTATION:")
    print("   • Artificial dataset expansion")
    print("   • Rotation, flipping, color jittering")
    print("   • Reduces overfitting")
    print("   • Improves robustness")
    
    print("\n4. GPU ACCELERATION:")
    print("   • Parallel computation")
    print("   • Faster training of large models")
    print("   • Enabled deep learning revolution")
    print("   • Made large-scale experiments feasible")
    
    print("\n5. DEEP ARCHITECTURE:")
    print("   • 8 layers (5 conv + 3 FC)")
    print("   • Hierarchical feature learning")
    print("   • Complex pattern recognition")
    print("   • Representation learning")
    
    print(f"\nHISTORICAL IMPACT:")
    print("="*40)
    print("• Sparked the deep learning revolution")
    print("• Proved scalability of deep CNNs")
    print("• Demonstrated GPU training effectiveness")
    print("• Established ReLU as standard activation")
    print("• Popularized dropout regularization")
    print("• Launched era of data-driven AI")
    print("• Inspired thousand of follow-up papers")
    
    print(f"\nTECHNICAL BREAKTHROUGHS:")
    print("="*40)
    print("• First successful very deep CNN (8 layers)")
    print("• Revolutionary 15.3% ImageNet error (vs 25.8%)")
    print("• Demonstrated importance of architecture design")
    print("• Showed value of computational scale")
    print("• Proved end-to-end learning effectiveness")
    
    print(f"\nLEGACY AND INFLUENCE:")
    print("="*40)
    print("• Foundation for all modern CNNs")
    print("• Techniques still used today")
    print("• Inspired VGG, GoogLeNet, ResNet")
    print("• Democratized computer vision")
    print("• Launched AI startup ecosystem")
    print("• Changed academic research direction")
    
    return {
        'model': 'AlexNet Revolution',
        'year': YEAR,
        'innovation': INNOVATION,
        'final_accuracy': alexnet_test_accs[-1],
        'improvement_over_baseline': alexnet_test_accs[-1] - baseline_test_accs[-1],
        'parameters': alexnet_params,
        'training_time': alexnet_metrics['training_time_minutes'],
        'relu_improvement': relu_improvement,
        'dropout_benefit': dropout_improvement
    }

if __name__ == "__main__":
    results = main()