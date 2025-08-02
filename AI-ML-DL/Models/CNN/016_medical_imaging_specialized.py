"""
ERA 6: SPECIALIZED APPLICATIONS - Medical Imaging Specialized Networks
======================================================================

Year: 2018-Present
Innovation: Specialized CNN architectures for medical image analysis with domain-specific adaptations
Previous Limitation: General-purpose CNNs not optimized for medical imaging challenges
Performance Gain: Superior diagnostic accuracy, robust to medical imaging artifacts, clinically validated
Impact: Revolutionized medical diagnosis, enabled AI-assisted healthcare, improved patient outcomes

This file implements specialized CNN architectures for medical imaging that address unique challenges
in medical diagnosis through domain-specific architectural innovations and training strategies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
from collections import defaultdict
import math
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# HISTORICAL CONTEXT & MOTIVATION
# ============================================================================

YEAR = "2018-Present"
INNOVATION = "Specialized CNN architectures for medical image analysis with domain-specific adaptations"
PREVIOUS_LIMITATION = "General CNNs not optimized for medical challenges, poor handling of artifacts"
IMPACT = "Revolutionized medical diagnosis, enabled AI-assisted healthcare"

print(f"=== Medical Imaging Specialized Networks ({YEAR}) ===")
print(f"Innovation: {INNOVATION}")
print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
print(f"Impact: {IMPACT}")
print("="*60)

# ============================================================================
# MEDICAL DATASET SIMULATION (USING CIFAR-10 AS PROXY)
# ============================================================================

class MedicalImagingDataset(Dataset):
    """
    Simulated medical imaging dataset using CIFAR-10 as proxy
    In practice, this would be real medical datasets like:
    - Chest X-rays (NIH, ChestX-ray14)
    - Skin lesions (ISIC, HAM10000)
    - Retinal images (MESSIDOR, EyePACS)
    - Brain MRI (BraTS, ADNI)
    """
    
    def __init__(self, cifar_dataset, medical_task='chest_xray'):
        self.cifar_dataset = cifar_dataset
        self.medical_task = medical_task
        
        # Define medical imaging task mappings
        self.task_configs = {
            'chest_xray': {
                'name': 'Chest X-ray Pneumonia Detection',
                'classes': ['Normal', 'Pneumonia'],
                'class_mapping': self._binary_mapping,
                'modality': 'X-ray',
                'challenge': 'Subtle pathological patterns'
            },
            'skin_cancer': {
                'name': 'Skin Lesion Classification',
                'classes': ['Benign', 'Malignant'],
                'class_mapping': self._binary_mapping,
                'modality': 'Dermoscopy',
                'challenge': 'Fine-grained texture analysis'
            },
            'retinal_disease': {
                'name': 'Diabetic Retinopathy Detection',
                'classes': ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR'],
                'class_mapping': self._multiclass_severity_mapping,
                'modality': 'Fundus Photography',
                'challenge': 'Multi-scale lesion detection'
            },
            'brain_tumor': {
                'name': 'Brain Tumor Classification',
                'classes': ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary'],
                'class_mapping': self._tumor_mapping,
                'modality': 'MRI',
                'challenge': 'Cross-sectional anatomy'
            }
        }
        
        self.config = self.task_configs[medical_task]
        self.num_classes = len(self.config['classes'])
        
        print(f"  Medical Dataset Configuration:")
        print(f"    Task: {self.config['name']}")
        print(f"    Modality: {self.config['modality']}")
        print(f"    Classes: {self.config['classes']}")
        print(f"    Challenge: {self.config['challenge']}")
        print(f"    Samples: {len(cifar_dataset):,}")
    
    def _binary_mapping(self, cifar_label):
        """Map CIFAR-10 to binary medical classification"""
        return 1 if cifar_label >= 5 else 0  # Split into two groups
    
    def _multiclass_severity_mapping(self, cifar_label):
        """Map CIFAR-10 to severity-based classification"""
        # Map to 5 severity levels
        return cifar_label // 2
    
    def _tumor_mapping(self, cifar_label):
        """Map CIFAR-10 to tumor type classification"""
        if cifar_label < 3:
            return 0  # No tumor
        elif cifar_label < 6:
            return 1  # Glioma
        elif cifar_label < 8:
            return 2  # Meningioma
        else:
            return 3  # Pituitary
    
    def __len__(self):
        return len(self.cifar_dataset)
    
    def __getitem__(self, idx):
        image, cifar_label = self.cifar_dataset[idx]
        medical_label = self.config['class_mapping'](cifar_label)
        return image, medical_label

def create_medical_imaging_dataset(medical_task='chest_xray'):
    """Create medical imaging dataset for training and testing"""
    print(f"Creating medical imaging dataset for {medical_task}...")
    
    # Medical imaging specific transforms
    transform_train = transforms.Compose([
        transforms.Resize(224),  # Medical images often need higher resolution
        transforms.RandomRotation(10),  # Limited rotation for medical images
        transforms.RandomHorizontalFlip(p=0.3),  # Careful with anatomical flips
        transforms.ColorJitter(brightness=0.1, contrast=0.2),  # Medical contrast adjustment
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load CIFAR-10 as base
    cifar_train = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    cifar_test = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Create medical datasets
    train_dataset = MedicalImagingDataset(cifar_train, medical_task)
    test_dataset = MedicalImagingDataset(cifar_test, medical_task)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    print(f"Medical imaging dataset created:")
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Test samples: {len(test_dataset):,}")
    print(f"  Number of classes: {train_dataset.num_classes}")
    print(f"  Task type: {train_dataset.config['name']}")
    
    return train_loader, test_loader, train_dataset.num_classes, train_dataset.config

# ============================================================================
# ATTENTION MECHANISMS FOR MEDICAL IMAGING
# ============================================================================

class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    Focus on informative channels for medical diagnosis
    """
    
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    Focus on relevant spatial regions for pathology detection
    """
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    Combines channel and spatial attention for medical imaging
    """
    
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
        print(f"    CBAM Attention: {in_channels} channels, kernel {kernel_size}")
    
    def forward(self, x):
        # Channel attention
        x = x * self.channel_attention(x)
        
        # Spatial attention
        x = x * self.spatial_attention(x)
        
        return x

# ============================================================================
# MEDICAL IMAGING SPECIFIC BLOCKS
# ============================================================================

class MedicalResidualBlock(nn.Module):
    """
    Medical-specific Residual Block with Attention
    Enhanced for medical imaging pathology detection
    """
    
    def __init__(self, in_channels, out_channels, stride=1, use_attention=True):
        super(MedicalResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Attention mechanism
        self.attention = CBAM(out_channels) if use_attention else nn.Identity()
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply attention
        out = self.attention(out)
        
        # Skip connection
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out

class MultiScaleFeatureExtractor(nn.Module):
    """
    Multi-scale feature extraction for medical lesion detection
    Captures features at different scales simultaneously
    """
    
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFeatureExtractor, self).__init__()
        
        # Different kernel sizes for multi-scale features
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, 5, padding=2),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        print(f"    Multi-scale Extractor: {in_channels}→{out_channels} (4 scales)")
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        
        outputs = torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], dim=1)
        return outputs

# ============================================================================
# MEDICAL IMAGING CNN ARCHITECTURE
# ============================================================================

class MedicalImaging_SpecializedCNN(nn.Module):
    """
    Specialized CNN for Medical Imaging
    
    Innovations for Medical Imaging:
    - Multi-scale feature extraction for lesion detection
    - Attention mechanisms for relevant region focus
    - Medical-specific data augmentation handling
    - Robust to imaging artifacts and variations
    - Interpretable feature representations
    """
    
    def __init__(self, num_classes=2, medical_task='chest_xray', use_attention=True):
        super(MedicalImaging_SpecializedCNN, self).__init__()
        
        self.num_classes = num_classes
        self.medical_task = medical_task
        self.use_attention = use_attention
        
        print(f"Building Specialized Medical Imaging CNN...")
        print(f"  Medical task: {medical_task}")
        print(f"  Use attention: {use_attention}")
        
        # Initial feature extraction
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Multi-scale feature extraction stages
        self.stage1 = nn.Sequential(
            MedicalResidualBlock(64, 64, use_attention=use_attention),
            MedicalResidualBlock(64, 64, use_attention=use_attention)
        )
        
        self.stage2 = nn.Sequential(
            MedicalResidualBlock(64, 128, stride=2, use_attention=use_attention),
            MultiScaleFeatureExtractor(128, 128),
            MedicalResidualBlock(128, 128, use_attention=use_attention)
        )
        
        self.stage3 = nn.Sequential(
            MedicalResidualBlock(128, 256, stride=2, use_attention=use_attention),
            MultiScaleFeatureExtractor(256, 256),
            MedicalResidualBlock(256, 256, use_attention=use_attention)
        )
        
        self.stage4 = nn.Sequential(
            MedicalResidualBlock(256, 512, stride=2, use_attention=use_attention),
            MultiScaleFeatureExtractor(512, 512),
            MedicalResidualBlock(512, 512, use_attention=use_attention)
        )
        
        # Global feature aggregation
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Medical-specific classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Calculate statistics
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"Medical CNN Architecture Summary:")
        print(f"  Medical task: {medical_task}")
        print(f"  Number of classes: {num_classes}")
        print(f"  Use attention: {use_attention}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Key features: Multi-scale + Attention + Medical-specific")
    
    def _initialize_weights(self):
        """Initialize weights for medical imaging CNN"""
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
        """Forward pass through medical imaging CNN"""
        # Initial feature extraction
        x = self.initial_conv(x)
        
        # Multi-scale feature extraction stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    
    def extract_features(self, x, layer='stage4'):
        """Extract intermediate features for visualization/analysis"""
        self.eval()
        
        with torch.no_grad():
            # Forward through initial layers
            x = self.initial_conv(x)
            stage1_features = self.stage1(x)
            
            if layer == 'stage1':
                return stage1_features
            
            stage2_features = self.stage2(stage1_features)
            
            if layer == 'stage2':
                return stage2_features
            
            stage3_features = self.stage3(stage2_features)
            
            if layer == 'stage3':
                return stage3_features
            
            stage4_features = self.stage4(stage3_features)
            
            return stage4_features
    
    def get_attention_maps(self, x):
        """Extract attention maps for interpretability"""
        attention_maps = []
        
        def hook_attention(module, input, output):
            if hasattr(module, 'channel_attention') or hasattr(module, 'spatial_attention'):
                attention_maps.append(output.detach())
        
        # Register hooks for attention modules
        hooks = []
        for module in self.modules():
            if isinstance(module, CBAM):
                hooks.append(module.register_forward_hook(hook_attention))
        
        # Forward pass
        _ = self.forward(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_maps
    
    def get_medical_analysis(self):
        """Analyze medical imaging specific capabilities"""
        return {
            'medical_task': self.medical_task,
            'num_classes': self.num_classes,
            'attention_enabled': self.use_attention,
            'multi_scale_features': True,
            'interpretability': 'Attention maps + Feature visualization',
            'medical_adaptations': [
                'Multi-scale lesion detection',
                'Attention-based region focus',
                'Robust to imaging artifacts',
                'Medical-specific augmentation handling'
            ],
            'clinical_applications': [
                'Computer-aided diagnosis',
                'Screening and early detection',
                'Treatment planning support',
                'Medical education and training'
            ]
        }

# ============================================================================
# MEDICAL LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for Medical Imaging
    Addresses class imbalance common in medical datasets
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        print(f"  Focal Loss: alpha={alpha}, gamma={gamma} (for class imbalance)")
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    """
    Dice Loss for Medical Segmentation Tasks
    Commonly used in medical image segmentation
    """
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
        print(f"  Dice Loss: smooth={smooth} (for segmentation)")
    
    def forward(self, inputs, targets):
        # Convert to probabilities
        inputs = torch.softmax(inputs, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1))
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # Calculate Dice coefficient
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_medical_imaging(model, train_loader, test_loader, epochs=60, learning_rate=1e-3, 
                         use_focal_loss=True):
    """Train medical imaging specialized CNN"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Medical imaging training configuration
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )
    
    # Learning rate scheduling for medical imaging
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Loss function selection
    if use_focal_loss:
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
    else:
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()
            
            if batch_idx % 300 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.4f}, LR: {current_lr:.6f}')
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_train_acc = 100. * correct_train / total_train
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_train_acc)
        
        # Test evaluation
        test_acc = evaluate_medical_model(model, test_loader, device)
        test_accuracies.append(test_acc)
        
        # Learning rate scheduling
        scheduler.step(epoch_loss)
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), f'AI-ML-DL/Models/CNN/medical_imaging_best.pth')
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {epoch_loss:.4f}, '
              f'Train Acc: {epoch_train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        # Early stopping for demonstration
        if test_acc > 90.0:
            print(f"Excellent medical performance reached at epoch {epoch+1}")
            break
    
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    return train_losses, train_accuracies, test_accuracies

def evaluate_medical_model(model, test_loader, device):
    """Evaluate medical imaging model with detailed metrics"""
    model.eval()
    correct = 0
    total = 0
    
    # For medical imaging, we often need detailed per-class metrics
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Per-class statistics
            for i in range(targets.size(0)):
                label = targets[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1
    
    overall_accuracy = 100. * correct / total
    
    # Print per-class accuracies (important for medical imaging)
    print(f"  Per-class accuracies:")
    for class_id in sorted(class_total.keys()):
        if class_total[class_id] > 0:
            class_acc = 100. * class_correct[class_id] / class_total[class_id]
            print(f"    Class {class_id}: {class_acc:.2f}% ({class_correct[class_id]}/{class_total[class_id]})")
    
    return overall_accuracy

# ============================================================================
# VISUALIZATION AND ANALYSIS
# ============================================================================

def visualize_medical_innovations():
    """Visualize medical imaging specialized innovations"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Medical imaging challenges
    ax = axes[0, 0]
    ax.set_title('Medical Imaging Challenges', fontsize=14)
    
    challenges = ['Class\nImbalance', 'Subtle\nPathology', 'Imaging\nArtifacts', 
                 'Inter-observer\nVariability', 'Limited\nData']
    difficulty = [9, 8, 7, 6, 8]
    colors = ['#E74C3C', '#E67E22', '#F39C12', '#F1C40F', '#E74C3C']
    
    bars = ax.bar(challenges, difficulty, color=colors)
    ax.set_ylabel('Difficulty Level (1-10)')
    ax.set_ylim(0, 10)
    
    for bar, diff in zip(bars, difficulty):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{diff}', ha='center', va='bottom', fontweight='bold')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Specialized architecture components
    ax = axes[0, 1]
    ax.set_title('Medical CNN Specialized Components', fontsize=14)
    
    # Draw architecture blocks
    components = [
        ('Multi-scale\nFeature Extraction', 0.2, 0.8, '#3498DB'),
        ('Channel & Spatial\nAttention (CBAM)', 0.2, 0.6, '#E67E22'),
        ('Medical Residual\nBlocks', 0.2, 0.4, '#27AE60'),
        ('Focal Loss\n(Class Imbalance)', 0.2, 0.2, '#9B59B6')
    ]
    
    for comp, x, y, color in components:
        rect = plt.Rectangle((x, y-0.05), 0.6, 0.1, 
                           facecolor=color, alpha=0.7, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x + 0.3, y, comp, ha='center', va='center', 
               fontweight='bold', color='white', fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Medical applications performance
    ax = axes[1, 0]
    applications = ['Chest X-ray\nPneumonia', 'Skin Cancer\nDetection', 
                   'Diabetic\nRetinopathy', 'Brain Tumor\nClassification']
    ai_accuracy = [94.5, 91.2, 89.8, 96.1]  # Example accuracies
    human_accuracy = [87.2, 86.5, 82.3, 88.7]  # Radiologist baseline
    
    x = np.arange(len(applications))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, human_accuracy, width, label='Human Expert', color='#95A5A6')
    bars2 = ax.bar(x + width/2, ai_accuracy, width, label='Medical AI', color='#27AE60')
    
    ax.set_title('AI vs Human Expert Performance', fontsize=14)
    ax.set_ylabel('Diagnostic Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(applications)
    ax.legend()
    
    # Add accuracy labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, 
                   f'{height:.1f}%', ha='center', va='bottom')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Attention mechanism visualization
    ax = axes[1, 1]
    ax.set_title('Medical Attention Mechanism', fontsize=14)
    
    # Simulate medical image with attention
    # Create a simple representation
    image_grid = np.random.random((10, 10)) * 0.3
    
    # Add "pathological" regions with higher attention
    pathology_regions = [(2, 3), (7, 8), (5, 2)]
    for x, y in pathology_regions:
        # Add pathology
        image_grid[max(0, y-1):min(10, y+2), max(0, x-1):min(10, x+2)] += 0.7
    
    # Show image with attention overlay
    im = ax.imshow(image_grid, cmap='hot', alpha=0.8)
    
    # Mark attention regions
    for x, y in pathology_regions:
        circle = plt.Circle((x, y), 1, fill=False, color='cyan', linewidth=3)
        ax.add_patch(circle)
        ax.text(x, y, '⚠', ha='center', va='center', fontsize=20, color='cyan')
    
    ax.set_title('Attention Focus on Pathology')
    ax.axis('off')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Attention Weight')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/CNN/016_medical_innovations.png', dpi=300, bbox_inches='tight')
    print("Medical imaging innovations visualization saved: 016_medical_innovations.png")

def visualize_medical_workflow():
    """Visualize medical AI diagnostic workflow"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Medical AI pipeline
    ax = axes[0]
    ax.set_title('Medical AI Diagnostic Pipeline', fontsize=14, fontweight='bold')
    
    # Pipeline stages
    stages = [
        ('Image\nAcquisition', 0.1, 0.8, '#3498DB'),
        ('Preprocessing\n& Quality Check', 0.1, 0.6, '#E67E22'),
        ('Feature Extraction\n(CNN Backbone)', 0.1, 0.4, '#27AE60'),
        ('Attention\nMechanisms', 0.1, 0.2, '#9B59B6'),
        ('Classification\n& Diagnosis', 0.6, 0.8, '#E74C3C'),
        ('Confidence\nAssessment', 0.6, 0.6, '#F39C12'),
        ('Clinical\nInterpretation', 0.6, 0.4, '#1ABC9C'),
        ('Treatment\nRecommendation', 0.6, 0.2, '#34495E')
    ]
    
    for stage, x, y, color in stages:
        rect = plt.Rectangle((x, y-0.05), 0.3, 0.1, 
                           facecolor=color, alpha=0.7, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x + 0.15, y, stage, ha='center', va='center', 
               fontweight='bold', color='white', fontsize=9)
    
    # Draw connections
    connections = [
        ((0.25, 0.8), (0.25, 0.65)),  # Acquisition to Preprocessing
        ((0.25, 0.6), (0.25, 0.45)),  # Preprocessing to Feature Extraction
        ((0.25, 0.4), (0.25, 0.25)),  # Feature Extraction to Attention
        ((0.4, 0.2), (0.6, 0.8)),     # Attention to Classification
        ((0.75, 0.8), (0.75, 0.65)),  # Classification to Confidence
        ((0.75, 0.6), (0.75, 0.45)),  # Confidence to Interpretation
        ((0.75, 0.4), (0.75, 0.25))   # Interpretation to Treatment
    ]
    
    for (x1, y1), (x2, y2) in connections:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Clinical validation process
    ax = axes[1]
    ax.set_title('Clinical Validation & Deployment', fontsize=14, fontweight='bold')
    
    # Validation phases
    validation_phases = [
        ('Retrospective\nValidation', 0.2, 0.8, '#3498DB'),
        ('Prospective\nClinical Trial', 0.2, 0.6, '#E67E22'),
        ('Regulatory\nApproval', 0.2, 0.4, '#27AE60'),
        ('Clinical\nDeployment', 0.2, 0.2, '#E74C3C')
    ]
    
    for phase, x, y, color in validation_phases:
        circle = plt.Circle((x, y), 0.08, facecolor=color, alpha=0.7, edgecolor='black')
        ax.add_patch(circle)
        ax.text(x, y, phase, ha='center', va='center', 
               fontweight='bold', color='white', fontsize=8)
    
    # Success metrics
    metrics = [
        ('Sensitivity: 95%+', 0.5, 0.8),
        ('Specificity: 90%+', 0.5, 0.7),
        ('PPV: 85%+', 0.5, 0.6),
        ('NPV: 98%+', 0.5, 0.5),
        ('AUC: 0.95+', 0.5, 0.4),
        ('Clinical Utility', 0.5, 0.3),
        ('Cost Effectiveness', 0.5, 0.2)
    ]
    
    for metric, x, y in metrics:
        ax.text(x, y, metric, ha='left', va='center', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue', alpha=0.7))
    
    # Draw validation flow
    for i in range(len(validation_phases) - 1):
        y1 = validation_phases[i][2]
        y2 = validation_phases[i+1][2]
        ax.annotate('', xy=(0.2, y2 + 0.08), xytext=(0.2, y1 - 0.08),
                   arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/CNN/016_medical_workflow.png', dpi=300, bbox_inches='tight')
    print("Medical AI workflow saved: 016_medical_workflow.png")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"=== Medical Imaging Specialized Networks Implementation ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Create medical imaging datasets for different tasks
    chest_train, chest_test, chest_classes, chest_config = create_medical_imaging_dataset('chest_xray')
    skin_train, skin_test, skin_classes, skin_config = create_medical_imaging_dataset('skin_cancer')
    
    # Initialize specialized medical models
    medical_cnn_attention = MedicalImaging_SpecializedCNN(
        num_classes=chest_classes, medical_task='chest_xray', use_attention=True
    )
    
    medical_cnn_baseline = MedicalImaging_SpecializedCNN(
        num_classes=chest_classes, medical_task='chest_xray', use_attention=False
    )
    
    # Compare model complexities
    attention_params = sum(p.numel() for p in medical_cnn_attention.parameters())
    baseline_params = sum(p.numel() for p in medical_cnn_baseline.parameters())
    
    print(f"\nModel Complexity Comparison:")
    print(f"  Medical CNN with Attention: {attention_params:,} parameters")
    print(f"  Medical CNN Baseline: {baseline_params:,} parameters")
    print(f"  Attention overhead: {(attention_params-baseline_params):,} parameters")
    
    # Analyze medical imaging capabilities
    medical_analysis = medical_cnn_attention.get_medical_analysis()
    
    print(f"\nMedical Imaging Analysis:")
    for key, value in medical_analysis.items():
        if isinstance(value, list):
            print(f"  {key.replace('_', ' ').title()}:")
            for item in value:
                print(f"    • {item}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Generate visualizations
    print("\nGenerating Medical Imaging analysis...")
    visualize_medical_innovations()
    visualize_medical_workflow()
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("MEDICAL IMAGING SPECIALIZED NETWORKS SUMMARY")
    print("="*70)
    
    print(f"\nHistorical Context ({YEAR}):")
    print(f"  Innovation: {INNOVATION}")
    print(f"  Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"  Impact: {IMPACT}")
    
    print(f"\nMEDICAL IMAGING INNOVATIONS:")
    print("="*50)
    print("1. DOMAIN-SPECIFIC ARCHITECTURES:")
    print("   • Multi-scale feature extraction for lesion detection")
    print("   • Attention mechanisms for pathology focus")
    print("   • Medical-specific residual blocks")
    print("   • Robust to imaging artifacts and variations")
    
    print("\n2. SPECIALIZED ATTENTION MECHANISMS:")
    print("   • Channel attention for informative feature selection")
    print("   • Spatial attention for relevant region focus")
    print("   • CBAM (Convolutional Block Attention Module)")
    print("   • Interpretable attention maps for clinical use")
    
    print("\n3. MEDICAL LOSS FUNCTIONS:")
    print("   • Focal Loss for class imbalance handling")
    print("   • Dice Loss for segmentation tasks")
    print("   • Weighted loss for rare disease detection")
    print("   • Multi-task learning for comprehensive diagnosis")
    
    print("\n4. CLINICAL VALIDATION & DEPLOYMENT:")
    print("   • Rigorous clinical trial validation")
    print("   • FDA/regulatory approval processes")
    print("   • Integration with hospital workflows")
    print("   • Continuous learning and adaptation")
    
    print(f"\nTECHNICAL ACHIEVEMENTS:")
    print("="*40)
    print("• Superior diagnostic accuracy vs human experts")
    print("• Robust handling of medical imaging challenges")
    print("• Interpretable AI for clinical decision support")
    print("• Scalable deployment in healthcare systems")
    print("• Reduced diagnostic errors and improved outcomes")
    
    print(f"\nMEDICAL APPLICATIONS:")
    print("="*40)
    print("• Radiology: Chest X-rays, CT, MRI analysis")
    print("• Dermatology: Skin cancer and lesion detection")
    print("• Ophthalmology: Diabetic retinopathy screening")
    print("• Pathology: Histopathology slide analysis")
    print("• Cardiology: ECG analysis and cardiac imaging")
    print("• Oncology: Tumor detection and classification")
    
    print(f"\nCLINICAL IMPACT:")
    print("="*40)
    print("• Early disease detection and screening")
    print("• Reduced diagnostic errors and misinterpretations")
    print("• Increased diagnostic speed and efficiency")
    print("• Democratized access to expert-level diagnosis")
    print("• Support for under-resourced healthcare settings")
    print("• Improved patient outcomes and survival rates")
    
    print(f"\nCHALLENGES ADDRESSED:")
    print("="*40)
    print("• Class imbalance in medical datasets")
    print("• Subtle pathological pattern detection")
    print("• Imaging artifact and noise robustness")
    print("• Inter-observer variability reduction")
    print("• Limited annotated medical data")
    print("• Regulatory and safety requirements")
    
    print(f"\nFUTURE DIRECTIONS:")
    print("="*40)
    print("• Multimodal medical AI (imaging + clinical data)")
    print("• Federated learning for privacy-preserving training")
    print("• Explainable AI for clinical transparency")
    print("• Real-time diagnostic support systems")
    print("• Personalized medicine and treatment planning")
    print("• Global health applications and accessibility")
    
    # Update TODO status
    print("\n" + "="*70)
    print("ERA 6: SPECIALIZED APPLICATIONS COMPLETED")
    print("="*70)
    print("• YOLO Object Detection: Real-time single-shot detection")
    print("• ArcFace Face Recognition: Angular margin loss for superior discrimination")
    print("• Medical Imaging CNNs: Specialized architectures for healthcare")
    print("• Established domain-specific AI applications")
    print("• Demonstrated practical deployment in critical domains")
    print("• Enabled AI adoption in specialized professional fields")
    
    return {
        'model': 'Medical Imaging Specialized Networks',
        'year': YEAR,
        'innovation': INNOVATION,
        'medical_analysis': medical_analysis,
        'parameter_comparison': {
            'attention_model': attention_params,
            'baseline_model': baseline_params
        },
        'medical_tasks': [chest_config, skin_config]
    }

if __name__ == "__main__":
    results = main()