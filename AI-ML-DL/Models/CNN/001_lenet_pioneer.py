"""
ERA 1: PIONEERING DEEP NETWORKS - LeNet-5 Pioneer
==================================================

Year: 1998
Paper: "Gradient-Based Learning Applied to Document Recognition" (LeCun et al., 1998)
Innovation: First successful CNN architecture with end-to-end gradient-based learning
Previous Limitation: Hand-crafted features, template matching, shallow learning approaches
Performance Gain: End-to-end feature learning, spatial hierarchy understanding
Impact: Established CNNs as viable approach for computer vision, founded modern deep learning

This file implements the pioneering LeNet-5 architecture, adapted for CIFAR-10 to demonstrate
the foundational concepts of convolutional neural networks in a standardized evaluation framework.
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

YEAR = "1998"
INNOVATION = "First successful CNN with gradient-based learning"
PREVIOUS_LIMITATION = "Hand-crafted features, template matching, no end-to-end learning"
IMPACT = "Established CNN paradigm, foundational architecture for computer vision"

print(f"=== LeNet-5 Pioneer ({YEAR}) ===")
print(f"Innovation: {INNOVATION}")
print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
print(f"Impact: {IMPACT}")
print("="*60)

# ============================================================================
# STANDARDIZED DATASET LOADING (CIFAR-10)
# ============================================================================

def load_cifar10_dataset():
    """
    Load CIFAR-10 dataset with standardized preprocessing
    Adapted for LeNet which was originally designed for 32x32 grayscale
    """
    print("Loading CIFAR-10 dataset (adapted for LeNet)...")
    
    # LeNet-compatible transforms
    # Convert to grayscale to match original LeNet design, then back to RGB for consistency
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
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
# LeNet-5 ARCHITECTURE (ADAPTED FOR CIFAR-10)
# ============================================================================

class LeNet5_Pioneer(nn.Module):
    """
    LeNet-5 architecture adapted for CIFAR-10 (32x32 RGB → 10 classes)
    
    Original LeNet-5 was designed for 32x32 grayscale images (MNIST)
    This adaptation maintains the core architectural principles while
    handling RGB input and 10-class classification
    
    Architecture Innovation:
    - Alternating convolution and pooling layers
    - Spatial feature hierarchy (local → global)
    - Shared weights and translation invariance
    - End-to-end gradient-based learning
    """
    
    def __init__(self, num_classes=10):
        super(LeNet5_Pioneer, self).__init__()
        
        print("Building LeNet-5 Pioneer Architecture...")
        
        # Feature extraction layers (convolutional)
        # C1: Convolution layer 1
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=0)  # 32x32x3 → 28x28x6
        # S2: Subsampling layer 1 (pooling)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)      # 28x28x6 → 14x14x6
        
        # C3: Convolution layer 2  
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0) # 14x14x6 → 10x10x16
        # S4: Subsampling layer 2 (pooling)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)      # 10x10x16 → 5x5x16
        
        # Classification layers (fully connected)
        # C5: Fully connected layer 1 (convolution-like in original)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5x16 = 400 → 120
        # F6: Fully connected layer 2
        self.fc2 = nn.Linear(120, 84)           # 120 → 84
        # Output: Classification layer
        self.fc3 = nn.Linear(84, num_classes)   # 84 → 10 classes
        
        # Initialize weights using historical method
        self._initialize_weights()
        
        print(f"LeNet-5 Architecture Summary:")
        print(f"  Input: 32x32x3 (CIFAR-10 RGB)")
        print(f"  Conv1: 28x28x6 (5x5 conv, 6 filters)")
        print(f"  Pool1: 14x14x6 (2x2 avg pool)")
        print(f"  Conv2: 10x10x16 (5x5 conv, 16 filters)")
        print(f"  Pool2: 5x5x16 (2x2 avg pool)")
        print(f"  FC1: 120 neurons")
        print(f"  FC2: 84 neurons") 
        print(f"  Output: {num_classes} classes")
    
    def _initialize_weights(self):
        """Initialize weights using historical methods similar to original LeNet"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Xavier/Glorot initialization (used in early deep learning)
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass through LeNet-5 architecture
        Demonstrates the original CNN paradigm of alternating conv-pool layers
        """
        # Feature extraction stage
        # Conv1 + Pool1
        x = self.pool1(torch.tanh(self.conv1(x)))  # Original used tanh activation
        
        # Conv2 + Pool2  
        x = self.pool2(torch.tanh(self.conv2(x)))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten: batch_size x (16*5*5)
        
        # Classification stage
        x = torch.tanh(self.fc1(x))  # FC1 with tanh
        x = torch.tanh(self.fc2(x))  # FC2 with tanh
        x = self.fc3(x)              # Output layer (no activation, will use CrossEntropy)
        
        return x
    
    def get_feature_maps(self, x):
        """
        Extract feature maps at different layers for visualization
        Helps understand the spatial hierarchy learning
        """
        features = {}
        
        # After Conv1
        conv1_out = torch.tanh(self.conv1(x))
        features['conv1'] = conv1_out
        
        # After Pool1
        pool1_out = self.pool1(conv1_out)
        features['pool1'] = pool1_out
        
        # After Conv2
        conv2_out = torch.tanh(self.conv2(pool1_out))
        features['conv2'] = conv2_out
        
        # After Pool2
        pool2_out = self.pool2(conv2_out)
        features['pool2'] = pool2_out
        
        return features

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_lenet_pioneer(model, train_loader, test_loader, epochs=100, learning_rate=0.001):
    """
    Train LeNet-5 with historical training approach
    Uses techniques available in the late 1990s
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Historical optimizer (SGD was primary choice in 1998)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    
    # Learning rate scheduler (simple step decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training tracking
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print(f"Training LeNet-5 Pioneer on device: {device}")
    print("Using historical training approach: SGD + momentum")
    
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
            
            # Gradient clipping (modern addition for stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
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
        test_acc = evaluate_lenet_pioneer(model, test_loader, device)
        test_accuracies.append(test_acc)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'AI-ML-DL/Models/CNN/lenet_pioneer_best.pth')
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {epoch_loss:.4f}, '
              f'Train Acc: {epoch_train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        # Early stopping if converged
        if epoch > 20 and test_acc > 85.0:  # LeNet can achieve ~85% on CIFAR-10
            print(f"Early stopping at epoch {epoch+1} with test accuracy {test_acc:.2f}%")
            break
    
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    return train_losses, train_accuracies, test_accuracies

def evaluate_lenet_pioneer(model, test_loader, device):
    """Evaluate LeNet-5 on test set"""
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

def visualize_lenet_architecture():
    """Visualize LeNet-5 architecture and its innovation"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Architecture diagram (conceptual)
    ax = axes[0, 0]
    layers = ['Input\n32×32×3', 'Conv1\n28×28×6', 'Pool1\n14×14×6', 
              'Conv2\n10×10×16', 'Pool2\n5×5×16', 'FC1\n120', 'FC2\n84', 'Output\n10']
    x_pos = range(len(layers))
    sizes = [32*32*3, 28*28*6, 14*14*6, 10*10*16, 5*5*16, 120, 84, 10]
    
    bars = ax.bar(x_pos, [s/1000 for s in sizes], color=['#3498DB', '#E67E22', '#E74C3C', 
                                                         '#9B59B6', '#1ABC9C', '#F39C12', 
                                                         '#95A5A6', '#34495E'])
    ax.set_title('LeNet-5 Architecture Layer Sizes (×1000)', fontsize=14)
    ax.set_ylabel('Parameters (thousands)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(layers, rotation=45)
    
    # Historical context
    ax = axes[0, 1]
    years = ['Pre-1990\n(Hand-crafted)', '1998\n(LeNet-5)', '2012\n(AlexNet)', '2015\n(ResNet)']
    performance = [60, 75, 85, 95]  # Approximate accuracy progression
    colors = ['#BDC3C7', '#3498DB', '#E67E22', '#E74C3C']
    
    bars = ax.bar(years, performance, color=colors)
    ax.set_title('CNN Evolution: Accuracy Progression', fontsize=14)
    ax.set_ylabel('CIFAR-10 Accuracy (%)')
    ax.set_ylim(50, 100)
    
    # Add value labels on bars
    for bar, val in zip(bars, performance):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val}%', ha='center', va='bottom')
    
    # LeNet innovations
    ax = axes[1, 0]
    innovations = ['Convolution\nLayers', 'Pooling\nOperations', 'Shared\nWeights', 
                   'End-to-End\nLearning', 'Spatial\nHierarchy']
    importance = [10, 9, 8, 10, 9]
    bars = ax.bar(innovations, importance, color=['#3498DB', '#E67E22', '#9B59B6', '#E74C3C', '#1ABC9C'])
    ax.set_title('LeNet-5 Key Innovations', fontsize=14)
    ax.set_ylabel('Innovation Impact (1-10)')
    ax.tick_params(axis='x', rotation=45)
    
    # Computational comparison
    ax = axes[1, 1]
    methods = ['Template\nMatching', 'Hand-crafted\nFeatures', 'LeNet-5\nCNN']
    params = [0, 100, 60000]  # Approximate parameter counts
    bars = ax.bar(methods, [p/1000 for p in params], color=['#95A5A6', '#BDC3C7', '#3498DB'])
    ax.set_title('Parameter Count Comparison (×1000)', fontsize=14)
    ax.set_ylabel('Parameters (thousands)')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/CNN/001_lenet_pioneer_analysis.png', dpi=300, bbox_inches='tight')
    print("\nArchitecture analysis saved: 001_lenet_pioneer_analysis.png")

def visualize_feature_maps(model, test_loader, device, classes):
    """Visualize feature maps to understand spatial hierarchy learning"""
    model.eval()
    
    # Get a batch of test images
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    # Get feature maps for first image
    with torch.no_grad():
        features = model.get_feature_maps(images[:1])
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original image
    img = images[0].cpu()
    img = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1) + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    img = torch.clamp(img, 0, 1)
    
    axes[0, 0].imshow(img.permute(1, 2, 0))
    axes[0, 0].set_title(f'Input Image\nClass: {classes[labels[0]]}', fontsize=12)
    axes[0, 0].axis('off')
    
    # Conv1 feature maps (show first 6 channels)
    conv1_features = features['conv1'][0].cpu()
    for i in range(min(3, conv1_features.size(0))):
        row = 0 if i < 3 else 1
        col = i + 1 if i < 3 else i - 2
        if col < 4:
            axes[row, col].imshow(conv1_features[i], cmap='viridis')
            axes[row, col].set_title(f'Conv1 Feature {i+1}', fontsize=10)
            axes[row, col].axis('off')
    
    # Conv2 feature maps (show first few channels)
    conv2_features = features['conv2'][0].cpu()
    for i in range(min(4, conv2_features.size(0))):
        axes[1, i].imshow(conv2_features[i], cmap='plasma')
        axes[1, i].set_title(f'Conv2 Feature {i+1}', fontsize=10)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/CNN/001_lenet_feature_maps.png', dpi=300, bbox_inches='tight')
    print("Feature maps visualization saved: 001_lenet_feature_maps.png")

def analyze_computational_metrics(model):
    """Analyze computational requirements of LeNet-5"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate FLOPs (approximate)
    # For LeNet-5: Conv1(5x5x3x6) + Conv2(5x5x6x16) + FC layers
    conv1_flops = 28 * 28 * 5 * 5 * 3 * 6  # 28x28 output, 5x5x3 kernel, 6 filters
    conv2_flops = 10 * 10 * 5 * 5 * 6 * 16  # 10x10 output, 5x5x6 kernel, 16 filters
    fc1_flops = 400 * 120
    fc2_flops = 120 * 84
    fc3_flops = 84 * 10
    
    total_flops = conv1_flops + conv2_flops + fc1_flops + fc2_flops + fc3_flops
    
    print(f"\nLeNet-5 Computational Analysis:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Estimated FLOPs: {total_flops:,}")
    print(f"  Model Size: ~{total_params * 4 / 1024:.1f} KB (FP32)")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'flops': total_flops,
        'model_size_kb': total_params * 4 / 1024
    }

# ============================================================================
# ABLATION STUDIES
# ============================================================================

def ablation_study_activations(train_loader, test_loader):
    """
    Study the impact of activation functions
    Compare tanh (original) vs ReLU (modern)
    """
    print("\nAblation Study: Activation Functions")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    # Test different activation functions
    activations = {
        'tanh': torch.tanh,
        'relu': F.relu,
        'sigmoid': torch.sigmoid
    }
    
    for activation_name, activation_fn in activations.items():
        print(f"\nTesting {activation_name.upper()} activation...")
        
        # Create modified LeNet with different activation
        class LeNetWithActivation(LeNet5_Pioneer):
            def __init__(self, activation_fn):
                super().__init__()
                self.activation = activation_fn
            
            def forward(self, x):
                x = self.pool1(self.activation(self.conv1(x)))
                x = self.pool2(self.activation(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = self.activation(self.fc1(x))
                x = self.activation(self.fc2(x))
                x = self.fc3(x)
                return x
        
        model = LeNetWithActivation(activation_fn).to(device)
        
        # Quick training (fewer epochs for ablation)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(10):  # Quick training
            for batch_idx, (data, targets) in enumerate(train_loader):
                if batch_idx > 50:  # Limited batches for speed
                    break
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        # Test accuracy
        test_acc = evaluate_lenet_pioneer(model, test_loader, device)
        results[activation_name] = test_acc
        print(f"{activation_name.upper()} Test Accuracy: {test_acc:.2f}%")
    
    return results

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
    print(f"=== LeNet-5 Pioneer Implementation ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load CIFAR-10 dataset
    train_loader, test_loader, classes = load_cifar10_dataset()
    
    # Initialize LeNet-5 model
    model = LeNet5_Pioneer(num_classes=10)
    
    # Analyze computational requirements
    comp_metrics = analyze_computational_metrics(model)
    
    # Visualize architecture
    visualize_lenet_architecture()
    
    # Train the model
    print("\nStarting LeNet-5 Pioneer Training...")
    metrics = track_training_metrics(
        'LeNet-5 Pioneer',
        train_lenet_pioneer,
        model, train_loader, test_loader, 50, 0.01  # 50 epochs, higher LR for faster convergence
    )
    
    train_losses, train_accuracies, test_accuracies = metrics['result']
    
    # Visualize feature maps
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visualize_feature_maps(model, test_loader, device, classes)
    
    # Ablation study
    ablation_results = ablation_study_activations(train_loader, test_loader)
    
    # Create comprehensive results visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training curves
    ax = axes[0, 0]
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax.set_title('LeNet-5 Training Loss', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax = axes[0, 1]
    ax.plot(epochs, train_accuracies, 'g-', label='Training Accuracy', linewidth=2)
    ax.plot(epochs, test_accuracies, 'r-', label='Test Accuracy', linewidth=2)
    ax.set_title('LeNet-5 Accuracy Progression', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Ablation study results
    ax = axes[1, 0]
    activations = list(ablation_results.keys())
    accuracies = list(ablation_results.values())
    bars = ax.bar(activations, accuracies, color=['#3498DB', '#E67E22', '#9B59B6'])
    ax.set_title('Activation Function Ablation Study', fontsize=14)
    ax.set_ylabel('Test Accuracy (%)')
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # Computational metrics
    ax = axes[1, 1]
    metrics_names = ['Parameters\n(thousands)', 'FLOPs\n(millions)', 'Model Size\n(KB)', 'Training Time\n(minutes)']
    metrics_values = [
        comp_metrics['total_params'] / 1000,
        comp_metrics['flops'] / 1e6,
        comp_metrics['model_size_kb'],
        metrics['training_time_minutes']
    ]
    bars = ax.bar(metrics_names, metrics_values, color=['#1ABC9C', '#F39C12', '#E74C3C', '#9B59B6'])
    ax.set_title('LeNet-5 Computational Metrics', fontsize=14)
    ax.set_ylabel('Values (various units)')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/CNN/001_lenet_pioneer_results.png', dpi=300, bbox_inches='tight')
    print("\nComprehensive results saved: 001_lenet_pioneer_results.png")
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("LENET-5 PIONEER SUMMARY")
    print("="*70)
    
    print(f"\nHistorical Context ({YEAR}):")
    print(f"  Innovation: {INNOVATION}")
    print(f"  Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"  Impact: {IMPACT}")
    
    print(f"\nArchitectural Details:")
    print(f"  Total Parameters: {comp_metrics['total_params']:,}")
    print(f"  Model Size: {comp_metrics['model_size_kb']:.1f} KB")
    print(f"  Estimated FLOPs: {comp_metrics['flops']:,}")
    
    print(f"\nTraining Results:")
    print(f"  Final Training Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"  Final Test Accuracy: {test_accuracies[-1]:.2f}%")
    print(f"  Training Time: {metrics['training_time_minutes']:.2f} minutes")
    print(f"  Memory Usage: {metrics['memory_usage_mb']:.2f} MB")
    
    print(f"\nAblation Study Results:")
    for activation, accuracy in ablation_results.items():
        print(f"  {activation.upper()}: {accuracy:.2f}%")
    
    print(f"\nLeNet-5 KEY INNOVATIONS:")
    print("="*40)
    print("1. CONVOLUTIONAL LAYERS:")
    print("   • Shared weights across spatial locations")
    print("   • Translation invariance")
    print("   • Local connectivity patterns")
    
    print("\n2. POOLING OPERATIONS:")
    print("   • Spatial downsampling")
    print("   • Invariance to small translations")
    print("   • Computational efficiency")
    
    print("\n3. HIERARCHICAL FEATURE LEARNING:")
    print("   • Low-level to high-level features")
    print("   • Spatial feature hierarchy")
    print("   • End-to-end optimization")
    
    print("\n4. GRADIENT-BASED LEARNING:")
    print("   • Backpropagation through conv layers")
    print("   • Automatic feature discovery")
    print("   • No hand-crafted feature engineering")
    
    print(f"\nHISTORICAL SIGNIFICANCE:")
    print("="*40)
    print("• First successful deep CNN architecture")
    print("• Established conv-pool paradigm")
    print("• Proved viability of end-to-end learning")
    print("• Foundation for modern computer vision")
    print("• Inspired decades of CNN research")
    
    print(f"\nLIMITATIONS (addressed by later architectures):")
    print("="*40)
    print("• Limited depth (vanishing gradients)")
    print("• Small receptive fields")
    print("• No batch normalization")
    print("• Simple pooling operations")
    print("• Fixed architecture design")
    
    return {
        'model': 'LeNet-5 Pioneer',
        'year': YEAR,
        'innovation': INNOVATION,
        'final_accuracy': test_accuracies[-1],
        'parameters': comp_metrics['total_params'],
        'training_time': metrics['training_time_minutes'],
        'memory_usage': metrics['memory_usage_mb']
    }

if __name__ == "__main__":
    results = main()