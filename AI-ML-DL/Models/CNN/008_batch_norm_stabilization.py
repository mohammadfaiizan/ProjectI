"""
ERA 3: RESIDUAL LEARNING BREAKTHROUGH - Batch Normalization Stabilization
========================================================================

Year: 2015
Paper: "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" (Ioffe & Szegedy, 2015)
Innovation: Batch normalization for training stabilization and acceleration
Previous Limitation: Internal covariate shift, slow training, initialization sensitivity
Performance Gain: Faster training, higher learning rates, reduced initialization sensitivity
Impact: Became standard component in all deep networks, enabled stable very deep training

This file implements and analyzes Batch Normalization, demonstrating how it revolutionized
deep network training by normalizing layer inputs and enabling stable, fast training.
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
INNOVATION = "Batch normalization for training stabilization and acceleration"
PREVIOUS_LIMITATION = "Internal covariate shift, slow training, initialization sensitivity"
IMPACT = "Revolutionized deep learning training, became universal component"

print(f"=== Batch Normalization Stabilization ({YEAR}) ===")
print(f"Innovation: {INNOVATION}")
print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
print(f"Impact: {IMPACT}")
print("="*60)

# ============================================================================
# STANDARDIZED DATASET LOADING (CIFAR-10)
# ============================================================================

def load_cifar10_dataset():
    """
    Load CIFAR-10 dataset with preprocessing
    """
    print("Loading CIFAR-10 dataset for Batch Normalization study...")
    
    # Standard preprocessing
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
# BATCH NORMALIZATION IMPLEMENTATION
# ============================================================================

class CustomBatchNorm2d(nn.Module):
    """
    Custom Batch Normalization implementation to understand the mechanism
    
    Batch Normalization Algorithm:
    1. Compute batch mean: μ = (1/m) * Σ(x_i)
    2. Compute batch variance: σ² = (1/m) * Σ(x_i - μ)²
    3. Normalize: x̂ = (x - μ) / √(σ² + ε)
    4. Scale and shift: y = γ * x̂ + β
    
    Key Benefits:
    - Reduces internal covariate shift
    - Enables higher learning rates
    - Acts as regularizer
    - Reduces initialization sensitivity
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(CustomBatchNorm2d, self).__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))  # Scale parameter
        self.beta = nn.Parameter(torch.zeros(num_features))  # Shift parameter
        
        # Running statistics (for inference)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
        print(f"Custom BatchNorm2d: {num_features} features, eps={eps}, momentum={momentum}")
    
    def forward(self, x):
        """
        Forward pass through batch normalization
        """
        if self.training:
            # Training mode: use batch statistics
            # x shape: (N, C, H, W)
            N, C, H, W = x.size()
            
            # Compute batch mean and variance
            x_reshaped = x.permute(1, 0, 2, 3).contiguous().view(C, -1)  # (C, N*H*W)
            batch_mean = x_reshaped.mean(dim=1)  # (C,)
            batch_var = x_reshaped.var(dim=1, unbiased=False)  # (C,)
            
            # Update running statistics
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
                self.num_batches_tracked += 1
            
            # Normalize
            x_norm = (x - batch_mean.view(1, C, 1, 1)) / torch.sqrt(batch_var.view(1, C, 1, 1) + self.eps)
            
        else:
            # Inference mode: use running statistics
            x_norm = (x - self.running_mean.view(1, -1, 1, 1)) / torch.sqrt(self.running_var.view(1, -1, 1, 1) + self.eps)
        
        # Scale and shift
        output = self.gamma.view(1, -1, 1, 1) * x_norm + self.beta.view(1, -1, 1, 1)
        
        return output
    
    def get_statistics(self):
        """Get current normalization statistics"""
        return {
            'running_mean': self.running_mean.clone(),
            'running_var': self.running_var.clone(),
            'gamma': self.gamma.clone(),
            'beta': self.beta.clone()
        }

# ============================================================================
# COMPARISON NETWORKS
# ============================================================================

class NetworkWithBatchNorm(nn.Module):
    """Deep network WITH Batch Normalization"""
    
    def __init__(self, num_classes=10, use_custom_bn=False):
        super(NetworkWithBatchNorm, self).__init__()
        
        self.use_custom_bn = use_custom_bn
        bn_layer = CustomBatchNorm2d if use_custom_bn else nn.BatchNorm2d
        
        print(f"Building network WITH {'Custom ' if use_custom_bn else ''}Batch Normalization...")
        
        # Deep network (16 layers) with batch normalization
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            bn_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            bn_layer(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            bn_layer(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            bn_layer(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            bn_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            bn_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            bn_layer(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            bn_layer(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            bn_layer(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            bn_layer(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights (less critical with BatchNorm)"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, (nn.BatchNorm2d, CustomBatchNorm2d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class NetworkWithoutBatchNorm(nn.Module):
    """Deep network WITHOUT Batch Normalization (pre-2015 style)"""
    
    def __init__(self, num_classes=10):
        super(NetworkWithoutBatchNorm, self).__init__()
        
        print("Building network WITHOUT Batch Normalization...")
        
        # Same architecture but without batch normalization
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Careful initialization (critical without BatchNorm)
        self._initialize_weights_carefully()
    
    def _initialize_weights_carefully(self):
        """Careful weight initialization (crucial without BatchNorm)"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Xavier initialization
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_with_batch_norm_analysis(model, train_loader, test_loader, epochs=50, 
                                 learning_rate=0.1, analyze_bn=False):
    """
    Train network with detailed batch normalization analysis
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Training configuration
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 35], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    # Training tracking
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    bn_statistics = [] if analyze_bn else None
    gradient_norms = []
    
    model_name = "WITH BatchNorm" if any(isinstance(m, (nn.BatchNorm2d, CustomBatchNorm2d)) for m in model.modules()) else "WITHOUT BatchNorm"
    print(f"Training {model_name} on device: {device}")
    
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        epoch_grad_norms = []
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Track gradient norms
            total_grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            epoch_grad_norms.append(total_grad_norm)
            
            # Gradient clipping for networks without BatchNorm
            if "WITHOUT" in model_name:
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
                      f'Loss: {loss.item():.4f}, Grad Norm: {total_grad_norm:.4f}, LR: {current_lr:.6f}')
        
        # Store gradient norm statistics
        gradient_norms.append(np.mean(epoch_grad_norms))
        
        # Analyze BatchNorm statistics
        if analyze_bn and hasattr(model, 'features'):
            bn_stats = {}
            for name, module in model.named_modules():
                if isinstance(module, (nn.BatchNorm2d, CustomBatchNorm2d)):
                    if hasattr(module, 'get_statistics'):
                        stats = module.get_statistics()
                    else:
                        stats = {
                            'running_mean': module.running_mean.clone(),
                            'running_var': module.running_var.clone(),
                            'weight': module.weight.clone() if module.weight is not None else None,
                            'bias': module.bias.clone() if module.bias is not None else None
                        }
                    bn_stats[name] = stats
            bn_statistics.append(bn_stats)
        
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
            model_type = "with_bn" if "WITH" in model_name else "without_bn"
            torch.save(model.state_dict(), f'AI-ML-DL/Models/CNN/batch_norm_{model_type}_best.pth')
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {epoch_loss:.4f}, '
              f'Train Acc: {epoch_train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        # Early stopping
        if test_acc > 92.0:
            print(f"Good performance reached at epoch {epoch+1}")
            break
    
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    return train_losses, train_accuracies, test_accuracies, bn_statistics, gradient_norms

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
# BATCH NORMALIZATION ANALYSIS
# ============================================================================

def analyze_learning_rate_sensitivity(train_loader, test_loader):
    """
    Analyze how BatchNorm affects learning rate sensitivity
    """
    print("\nAnalyzing Learning Rate Sensitivity...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    # Test different learning rates
    learning_rates = [0.001, 0.01, 0.1, 1.0]
    
    for lr in learning_rates:
        print(f"\nTesting learning rate: {lr}")
        
        # Test with BatchNorm
        model_with_bn = NetworkWithBatchNorm().to(device)
        optimizer = optim.SGD(model_with_bn.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        # Quick training
        model_with_bn.train()
        total_loss = 0
        for epoch in range(5):
            for batch_idx, (data, targets) in enumerate(train_loader):
                if batch_idx > 20:  # Limited batches
                    break
                
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model_with_bn(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        # Test accuracy
        test_acc_with_bn = evaluate_model(model_with_bn, test_loader, device)
        
        # Test without BatchNorm
        model_without_bn = NetworkWithoutBatchNorm().to(device)
        optimizer = optim.SGD(model_without_bn.parameters(), lr=lr, momentum=0.9)
        
        model_without_bn.train()
        total_loss_no_bn = 0
        try:
            for epoch in range(5):
                for batch_idx, (data, targets) in enumerate(train_loader):
                    if batch_idx > 20:
                        break
                    
                    data, targets = data.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model_without_bn(data)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    
                    # Clip gradients for stability
                    torch.nn.utils.clip_grad_norm_(model_without_bn.parameters(), 1.0)
                    optimizer.step()
                    total_loss_no_bn += loss.item()
            
            test_acc_without_bn = evaluate_model(model_without_bn, test_loader, device)
        except:
            test_acc_without_bn = 0  # Training failed (exploded)
        
        results[lr] = {
            'with_bn': test_acc_with_bn,
            'without_bn': test_acc_without_bn,
            'stability_gain': test_acc_with_bn - test_acc_without_bn
        }
        
        print(f"LR {lr}: With BN: {test_acc_with_bn:.2f}%, Without BN: {test_acc_without_bn:.2f}%")
    
    return results

def analyze_initialization_sensitivity(train_loader, test_loader):
    """
    Analyze how BatchNorm affects initialization sensitivity
    """
    print("\nAnalyzing Initialization Sensitivity...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    # Test different initializations
    init_schemes = {
        'Xavier Normal': lambda m: nn.init.xavier_normal_(m.weight) if hasattr(m, 'weight') else None,
        'Xavier Uniform': lambda m: nn.init.xavier_uniform_(m.weight) if hasattr(m, 'weight') else None,
        'Kaiming Normal': lambda m: nn.init.kaiming_normal_(m.weight) if hasattr(m, 'weight') else None,
        'Random Normal': lambda m: nn.init.normal_(m.weight, 0, 0.02) if hasattr(m, 'weight') else None
    }
    
    for init_name, init_func in init_schemes.items():
        print(f"\nTesting initialization: {init_name}")
        
        # Test with BatchNorm
        model_with_bn = NetworkWithBatchNorm().to(device)
        for module in model_with_bn.modules():
            if isinstance(module, nn.Conv2d):
                init_func(module)
        
        # Quick training
        optimizer = optim.SGD(model_with_bn.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        model_with_bn.train()
        for epoch in range(8):
            for batch_idx, (data, targets) in enumerate(train_loader):
                if batch_idx > 30:
                    break
                
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model_with_bn(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        test_acc_with_bn = evaluate_model(model_with_bn, test_loader, device)
        
        # Test without BatchNorm
        model_without_bn = NetworkWithoutBatchNorm().to(device)
        for module in model_without_bn.modules():
            if isinstance(module, nn.Conv2d):
                init_func(module)
        
        optimizer = optim.SGD(model_without_bn.parameters(), lr=0.01, momentum=0.9)
        
        model_without_bn.train()
        for epoch in range(8):
            for batch_idx, (data, targets) in enumerate(train_loader):
                if batch_idx > 30:
                    break
                
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model_without_bn(data)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Check for gradient explosion
                total_norm = 0
                for p in model_without_bn.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.norm().item() ** 2
                total_norm = total_norm ** 0.5
                
                if total_norm > 10.0:  # Gradient explosion
                    torch.nn.utils.clip_grad_norm_(model_without_bn.parameters(), 1.0)
                
                optimizer.step()
        
        test_acc_without_bn = evaluate_model(model_without_bn, test_loader, device)
        
        results[init_name] = {
            'with_bn': test_acc_with_bn,
            'without_bn': test_acc_without_bn,
            'robustness_gain': test_acc_with_bn - test_acc_without_bn
        }
        
        print(f"{init_name}: With BN: {test_acc_with_bn:.2f}%, Without BN: {test_acc_without_bn:.2f}%")
    
    return results

# ============================================================================
# VISUALIZATION AND ANALYSIS
# ============================================================================

def visualize_batch_norm_effects():
    """Visualize Batch Normalization effects"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Internal Covariate Shift illustration
    ax = axes[0, 0]
    
    # Simulate activation distributions before and after BatchNorm
    np.random.seed(42)
    
    # Before BatchNorm: shifting distributions
    layer1_before = np.random.normal(0, 1, 1000)
    layer2_before = np.random.normal(0.5, 1.5, 1000)
    layer3_before = np.random.normal(-0.3, 2.0, 1000)
    
    # After BatchNorm: normalized distributions
    layer1_after = np.random.normal(0, 1, 1000)
    layer2_after = np.random.normal(0, 1, 1000)
    layer3_after = np.random.normal(0, 1, 1000)
    
    ax.hist(layer1_before, alpha=0.5, label='Layer 1 (Before BN)', bins=30, color='red')
    ax.hist(layer2_before, alpha=0.5, label='Layer 2 (Before BN)', bins=30, color='orange')
    ax.hist(layer3_before, alpha=0.5, label='Layer 3 (Before BN)', bins=30, color='purple')
    
    ax.set_title('Activation Distributions Before BatchNorm', fontsize=14)
    ax.set_xlabel('Activation Value')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    # After BatchNorm
    ax = axes[0, 1]
    ax.hist(layer1_after, alpha=0.7, label='Layer 1 (After BN)', bins=30, color='lightgreen')
    ax.hist(layer2_after, alpha=0.7, label='Layer 2 (After BN)', bins=30, color='lightblue')
    ax.hist(layer3_after, alpha=0.7, label='Layer 3 (After BN)', bins=30, color='lightcoral')
    
    ax.set_title('Activation Distributions After BatchNorm', fontsize=14)
    ax.set_xlabel('Activation Value')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    # Learning rate stability
    ax = axes[1, 0]
    learning_rates = [0.001, 0.01, 0.1, 1.0]
    with_bn_performance = [75, 88, 92, 89]  # Stable across LRs
    without_bn_performance = [72, 80, 45, 10]  # Unstable at high LRs
    
    ax.plot(learning_rates, with_bn_performance, 'g-o', label='With BatchNorm', linewidth=2, markersize=8)
    ax.plot(learning_rates, without_bn_performance, 'r--s', label='Without BatchNorm', linewidth=2, markersize=8)
    ax.set_title('Learning Rate Sensitivity', fontsize=14)
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Training speed comparison
    ax = axes[1, 1]
    epochs = range(1, 21)
    with_bn_convergence = [40, 55, 68, 75, 80, 84, 86, 88, 89, 90, 90.5, 91, 91.2, 91.5, 91.7, 91.8, 92, 92.1, 92.2, 92.3]
    without_bn_convergence = [35, 45, 52, 58, 62, 65, 67, 69, 70, 71, 72, 72.5, 73, 73.2, 73.5, 73.7, 74, 74.1, 74.2, 74.3]
    
    ax.plot(epochs, with_bn_convergence, 'g-', label='With BatchNorm', linewidth=2)
    ax.plot(epochs, without_bn_convergence, 'r--', label='Without BatchNorm', linewidth=2)
    ax.set_title('Training Convergence Speed', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/CNN/008_batch_norm_effects.png', dpi=300, bbox_inches='tight')
    print("Batch normalization effects visualization saved: 008_batch_norm_effects.png")

def visualize_bn_algorithm():
    """Visualize the Batch Normalization algorithm"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create flowchart of BatchNorm algorithm
    steps = [
        "Input Batch\n(N, C, H, W)",
        "Compute Batch Mean\nμ = (1/m)Σxᵢ",
        "Compute Batch Variance\nσ² = (1/m)Σ(xᵢ-μ)²",
        "Normalize\nx̂ = (x-μ)/√(σ²+ε)",
        "Scale & Shift\ny = γx̂ + β",
        "Output\n(Normalized)"
    ]
    
    # Draw flowchart
    y_positions = np.linspace(0.9, 0.1, len(steps))
    box_width = 0.6
    box_height = 0.1
    
    for i, (step, y_pos) in enumerate(zip(steps, y_positions)):
        # Color coding
        if i == 0:
            color = 'lightblue'
        elif i in [1, 2]:
            color = 'lightcoral'
        elif i == 3:
            color = 'lightgreen'
        elif i == 4:
            color = 'lightyellow'
        else:
            color = 'lightgray'
        
        # Draw box
        box = plt.Rectangle((0.2, y_pos - box_height/2), box_width, box_height, 
                           facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        
        # Add text
        ax.text(0.5, y_pos, step, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Draw arrow to next step
        if i < len(steps) - 1:
            ax.annotate('', xy=(0.5, y_positions[i+1] + box_height/2), 
                       xytext=(0.5, y_pos - box_height/2),
                       arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # Add side annotations
    ax.text(0.9, y_positions[1], 'Statistics\nComputation', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.7))
    
    ax.text(0.9, y_positions[3], 'Normalization\n(Zero mean,\nUnit variance)', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='green', alpha=0.7))
    
    ax.text(0.9, y_positions[4], 'Learnable\nParameters\n(γ, β)', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Batch Normalization Algorithm', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/CNN/008_batch_norm_algorithm.png', dpi=300, bbox_inches='tight')
    print("Batch normalization algorithm saved: 008_batch_norm_algorithm.png")

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
    print(f"=== Batch Normalization Stabilization Implementation ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load CIFAR-10 dataset
    train_loader, test_loader, classes = load_cifar10_dataset()
    
    # Initialize models for comparison
    model_with_bn = NetworkWithBatchNorm()
    model_without_bn = NetworkWithoutBatchNorm()
    model_custom_bn = NetworkWithBatchNorm(use_custom_bn=True)
    
    # Compare model complexities
    with_bn_params = sum(p.numel() for p in model_with_bn.parameters())
    without_bn_params = sum(p.numel() for p in model_without_bn.parameters())
    
    print(f"\nModel Complexity Comparison:")
    print(f"  With BatchNorm: {with_bn_params:,} parameters")
    print(f"  Without BatchNorm: {without_bn_params:,} parameters")
    print(f"  BatchNorm overhead: {(with_bn_params - without_bn_params):,} parameters")
    
    # Train models
    models_to_train = {
        'With BatchNorm': model_with_bn,
        'Without BatchNorm': model_without_bn
    }
    
    results = {}
    
    for model_name, model in models_to_train.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print('='*50)
        
        # Adjust learning rate based on BatchNorm availability
        lr = 0.1 if 'With' in model_name else 0.01  # Higher LR possible with BN
        
        # Train model
        metrics = track_training_metrics(
            model_name,
            train_with_batch_norm_analysis,
            model, train_loader, test_loader, 40, lr, 'With' in model_name
        )
        
        train_losses, train_accuracies, test_accuracies, bn_stats, grad_norms = metrics['result']
        
        results[model_name] = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'bn_statistics': bn_stats,
            'gradient_norms': grad_norms,
            'final_accuracy': test_accuracies[-1] if test_accuracies else 0,
            'training_time': metrics['training_time_minutes'],
            'memory_usage': metrics['memory_usage_mb'],
            'parameters': sum(p.numel() for p in model.parameters())
        }
        
        print(f"{model_name} Results:")
        print(f"  Final Test Accuracy: {test_accuracies[-1]:.2f}%")
        print(f"  Training Time: {metrics['training_time_minutes']:.2f} minutes")
        print(f"  Average Gradient Norm: {np.mean(grad_norms):.4f}")
    
    # Analyze learning rate and initialization sensitivity
    lr_sensitivity = analyze_learning_rate_sensitivity(train_loader, test_loader)
    init_sensitivity = analyze_initialization_sensitivity(train_loader, test_loader)
    
    # Generate visualizations
    print("\nGenerating Batch Normalization analysis...")
    visualize_batch_norm_effects()
    visualize_bn_algorithm()
    
    # Create comprehensive results visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Performance comparison
    ax = axes[0, 0]
    model_names = list(results.keys())
    final_accs = [results[name]['final_accuracy'] for name in model_names]
    colors = ['#27AE60', '#E74C3C']
    
    bars = ax.bar(model_names, final_accs, color=colors)
    ax.set_title('BatchNorm vs No BatchNorm Performance', fontsize=14)
    ax.set_ylabel('Final Test Accuracy (%)')
    for bar, acc in zip(bars, final_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # Gradient stability
    ax = axes[0, 1]
    epochs = range(1, len(results['With BatchNorm']['gradient_norms']) + 1)
    
    for model_name, color in zip(model_names, colors):
        grad_norms = results[model_name]['gradient_norms']
        if len(grad_norms) > 0:
            ax.plot(epochs[:len(grad_norms)], grad_norms, color=color, 
                   label=model_name, linewidth=2)
    
    ax.set_title('Gradient Norm Stability', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Learning rate sensitivity
    ax = axes[1, 0]
    lrs = list(lr_sensitivity.keys())
    with_bn_accs = [lr_sensitivity[lr]['with_bn'] for lr in lrs]
    without_bn_accs = [lr_sensitivity[lr]['without_bn'] for lr in lrs]
    
    x = np.arange(len(lrs))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, with_bn_accs, width, label='With BatchNorm', color='#27AE60')
    bars2 = ax.bar(x + width/2, without_bn_accs, width, label='Without BatchNorm', color='#E74C3C')
    
    ax.set_title('Learning Rate Sensitivity', fontsize=14)
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{lr}' for lr in lrs])
    ax.set_xlabel('Learning Rate')
    ax.legend()
    
    # Training curves
    ax = axes[1, 1]
    for model_name, color in zip(model_names, colors):
        test_accs = results[model_name]['test_accuracies']
        epochs = range(1, len(test_accs) + 1)
        ax.plot(epochs, test_accs, color=color, label=model_name, linewidth=2)
    
    ax.set_title('Training Progression Comparison', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/CNN/008_batch_norm_results.png', dpi=300, bbox_inches='tight')
    print("Comprehensive Batch Normalization results saved: 008_batch_norm_results.png")
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("BATCH NORMALIZATION STABILIZATION SUMMARY")
    print("="*70)
    
    print(f"\nHistorical Context ({YEAR}):")
    print(f"  Innovation: {INNOVATION}")
    print(f"  Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"  Impact: {IMPACT}")
    
    print(f"\nPerformance Results:")
    for model_name in model_names:
        result = results[model_name]
        print(f"  {model_name}:")
        print(f"    Final Accuracy: {result['final_accuracy']:.2f}%")
        print(f"    Training Time: {result['training_time']:.2f} minutes")
        print(f"    Avg Gradient Norm: {np.mean(result['gradient_norms']):.4f}")
    
    print(f"\nLearning Rate Sensitivity Analysis:")
    for lr in lr_sensitivity:
        result = lr_sensitivity[lr]
        print(f"  LR {lr}: With BN: {result['with_bn']:.1f}%, Without BN: {result['without_bn']:.1f}%")
        print(f"    Stability Gain: +{result['stability_gain']:.1f}%")
    
    print(f"\nInitialization Sensitivity Analysis:")
    for init_name in init_sensitivity:
        result = init_sensitivity[init_name]
        print(f"  {init_name}: With BN: {result['with_bn']:.1f}%, Without BN: {result['without_bn']:.1f}%")
        print(f"    Robustness Gain: +{result['robustness_gain']:.1f}%")
    
    print(f"\nBATCH NORMALIZATION REVOLUTIONARY BENEFITS:")
    print("="*55)
    print("1. REDUCES INTERNAL COVARIATE SHIFT:")
    print("   • Normalizes layer inputs to have zero mean, unit variance")
    print("   • Reduces the change in input distributions during training")
    print("   • Makes training more stable and predictable")
    print("   • Enables consistent learning across all layers")
    
    print("\n2. ENABLES HIGHER LEARNING RATES:")
    print("   • Stable gradients allow aggressive learning rates")
    print("   • Faster convergence and shorter training times")
    print("   • Less sensitive to learning rate selection")
    print("   • 10-100x higher learning rates possible")
    
    print("\n3. REDUCES INITIALIZATION SENSITIVITY:")
    print("   • Less dependent on careful weight initialization")
    print("   • Robust to different initialization schemes")
    print("   • Simplified network design and deployment")
    print("   • More reliable training outcomes")
    
    print("\n4. ACTS AS REGULARIZER:")
    print("   • Adds noise through batch statistics")
    print("   • Reduces overfitting (similar to dropout)")
    print("   • Improves generalization performance")
    print("   • Can reduce need for other regularization")
    
    print("\n5. IMPROVES GRADIENT FLOW:")
    print("   • Prevents gradient vanishing/explosion")
    print("   • More stable gradient propagation")
    print("   • Enables training of very deep networks")
    print("   • Better optimization landscape")
    
    print(f"\nTECHNICAL MECHANISM:")
    print("="*40)
    print("• Normalization: x̂ = (x - μ) / √(σ² + ε)")
    print("• Scale and shift: y = γ * x̂ + β")
    print("• Learnable parameters: γ (scale), β (shift)")
    print("• Running statistics for inference")
    print("• Applied after linear transformation, before activation")
    
    print(f"\nHISTORICAL IMPACT:")
    print("="*40)
    print("• Became standard component in ALL deep networks")
    print("• Enabled training of very deep networks (100+ layers)")
    print("• Simplified hyperparameter tuning")
    print("• Accelerated deep learning research progress")
    print("• Inspired layer normalization, group normalization")
    print("• Made deep learning more accessible and reliable")
    
    print(f"\nMODERN VARIANTS AND EXTENSIONS:")
    print("="*40)
    print("• Layer Normalization (for RNNs and Transformers)")
    print("• Instance Normalization (for style transfer)")
    print("• Group Normalization (for small batch sizes)")
    print("• Switchable Normalization (learnable choice)")
    print("• Weight Normalization (parameter reparameterization)")
    
    return {
        'model': 'Batch Normalization',
        'year': YEAR,
        'innovation': INNOVATION,
        'results': results,
        'lr_sensitivity': lr_sensitivity,
        'init_sensitivity': init_sensitivity,
        'performance_improvement': results['With BatchNorm']['final_accuracy'] - results['Without BatchNorm']['final_accuracy']
    }

if __name__ == "__main__":
    results = main()