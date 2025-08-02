"""
ERA 6: SPECIALIZED APPLICATIONS - Face Recognition with ArcFace
===============================================================

Year: 2018-Present
Paper: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (Deng et al., 2018)
Innovation: Angular margin loss for enhanced face recognition with superior discrimination
Previous Limitation: Softmax loss insufficient for face verification, poor feature separability
Performance Gain: State-of-the-art face verification accuracy, superior feature embedding quality
Impact: Revolutionized face recognition, enabled high-security applications, improved biometric systems

This file implements ArcFace face recognition that achieved breakthrough performance through
angular margin loss and deep metric learning for robust face identification and verification.
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
INNOVATION = "Angular margin loss for enhanced face recognition with superior discrimination"
PREVIOUS_LIMITATION = "Softmax loss insufficient for verification, poor intra-class compactness"
IMPACT = "Revolutionized face recognition, enabled high-security applications"

print(f"=== ArcFace Face Recognition ({YEAR}) ===")
print(f"Innovation: {INNOVATION}")
print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
print(f"Impact: {IMPACT}")
print("="*60)

# ============================================================================
# FACE DATASET SIMULATION (USING CIFAR-10 AS PROXY)
# ============================================================================

class FaceDataset(Dataset):
    """
    Simulated face recognition dataset using CIFAR-10 as proxy
    In practice, this would be a real face dataset like LFW, CFP-FP, etc.
    """
    
    def __init__(self, cifar_dataset, num_identities=100, samples_per_identity=50):
        self.cifar_dataset = cifar_dataset
        self.num_identities = num_identities
        self.samples_per_identity = samples_per_identity
        
        # Create identity mapping (simulate multiple images per person)
        self.identity_mapping = self._create_identity_mapping()
        
        print(f"  Face Dataset Simulation:")
        print(f"    Total identities: {num_identities}")
        print(f"    Samples per identity: {samples_per_identity}")
        print(f"    Total samples: {len(self.identity_mapping)}")
    
    def _create_identity_mapping(self):
        """Create mapping from images to identities"""
        mapping = []
        
        # Group CIFAR samples into identities
        samples_used = 0
        for identity_id in range(self.num_identities):
            for sample_idx in range(self.samples_per_identity):
                if samples_used < len(self.cifar_dataset):
                    cifar_idx = samples_used
                    mapping.append((cifar_idx, identity_id))
                    samples_used += 1
                else:
                    break
            if samples_used >= len(self.cifar_dataset):
                break
        
        return mapping
    
    def __len__(self):
        return len(self.identity_mapping)
    
    def __getitem__(self, idx):
        cifar_idx, identity_id = self.identity_mapping[idx]
        image, _ = self.cifar_dataset[cifar_idx]  # Ignore CIFAR label
        return image, identity_id

def create_face_recognition_dataset():
    """Create face recognition dataset for training and testing"""
    print("Creating face recognition dataset from CIFAR-10...")
    
    # Face recognition specific transforms
    transform_train = transforms.Compose([
        transforms.Resize(112),  # Standard face recognition input size
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1] normalization
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(112),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load CIFAR-10 as base
    cifar_train = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    cifar_test = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Create face datasets
    train_dataset = FaceDataset(cifar_train, num_identities=100, samples_per_identity=50)
    test_dataset = FaceDataset(cifar_test, num_identities=50, samples_per_identity=20)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"Face recognition dataset created:")
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Test samples: {len(test_dataset):,}")
    print(f"  Training identities: {train_dataset.num_identities}")
    print(f"  Test identities: {test_dataset.num_identities}")
    print(f"  Image size: 112x112 (face recognition standard)")
    
    return train_loader, test_loader, train_dataset.num_identities

# ============================================================================
# ARCFACE LOSS FUNCTION
# ============================================================================

class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss - Additive Angular Margin Loss
    
    Key Innovation: Angular margin in hypersphere
    - Projects features to hypersphere (L2 normalization)
    - Adds angular margin to target class
    - Enhances intra-class compactness and inter-class discrepancy
    
    Formula: cos(θ + m) where θ is angle between feature and weight
    """
    
    def __init__(self, embedding_dim=512, num_classes=100, margin=0.5, scale=64.0):
        super(ArcFaceLoss, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin  # Angular margin (m)
        self.scale = scale    # Scaling factor (s)
        
        # Weight matrix (represents class centers on hypersphere)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        # Precompute margin cosine and sine
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        
        print(f"  ArcFace Loss Configuration:")
        print(f"    Embedding dimension: {embedding_dim}")
        print(f"    Number of classes: {num_classes}")
        print(f"    Angular margin (m): {margin}")
        print(f"    Scale factor (s): {scale}")
        print(f"    Threshold: {self.threshold:.4f}")
    
    def forward(self, embeddings, labels):
        """
        Forward pass of ArcFace loss
        
        Args:
            embeddings: Feature embeddings (B, embedding_dim)
            labels: Ground truth labels (B,)
            
        Returns:
            ArcFace loss value
        """
        # L2 normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity (cos θ)
        cosine = F.linear(embeddings, weight_norm)
        
        # Calculate sin θ
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # cos(θ + m) = cos θ cos m - sin θ sin m
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Handle numerical instability
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
        
        # One-hot encoding for target class
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply angular margin only to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Scale the output
        output *= self.scale
        
        # Cross-entropy loss
        loss = F.cross_entropy(output, labels)
        
        return loss, output
    
    def get_angular_statistics(self, embeddings, labels):
        """Analyze angular statistics for insights"""
        with torch.no_grad():
            embeddings = F.normalize(embeddings, p=2, dim=1)
            weight_norm = F.normalize(self.weight, p=2, dim=1)
            
            # Cosine similarities
            cosine = F.linear(embeddings, weight_norm)
            
            # Convert to angles
            angles = torch.acos(torch.clamp(cosine, -1+1e-7, 1-1e-7))
            
            # Target class angles
            target_angles = angles.gather(1, labels.view(-1, 1))
            
            # Non-target class angles
            mask = torch.ones_like(angles)
            mask.scatter_(1, labels.view(-1, 1), 0)
            non_target_angles = angles[mask.bool()].view(angles.size(0), -1)
            
            return {
                'target_angles_mean': target_angles.mean().item(),
                'target_angles_std': target_angles.std().item(),
                'non_target_angles_mean': non_target_angles.mean().item(),
                'non_target_angles_std': non_target_angles.std().item(),
                'angular_margin_applied': self.margin
            }

# ============================================================================
# FACE RECOGNITION BACKBONE
# ============================================================================

class ResNetBackbone(nn.Module):
    """
    ResNet Backbone for Face Recognition
    Modified ResNet with appropriate output dimensions
    """
    
    def __init__(self, depth=50, embedding_dim=512):
        super(ResNetBackbone, self).__init__()
        
        self.depth = depth
        self.embedding_dim = embedding_dim
        
        print(f"Building ResNet-{depth} backbone for face recognition...")
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        if depth == 18:
            layers = [2, 2, 2, 2]
            block = BasicBlock
        elif depth == 34:
            layers = [3, 4, 6, 3]
            block = BasicBlock
        elif depth == 50:
            layers = [3, 4, 6, 3]
            block = Bottleneck
        elif depth == 101:
            layers = [3, 4, 23, 3]
            block = Bottleneck
        else:
            raise ValueError(f"Unsupported ResNet depth: {depth}")
        
        self.layer1 = self._make_layer(block, 64, 64, layers[0])
        self.layer2 = self._make_layer(block, 64 * block.expansion, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128 * block.expansion, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256 * block.expansion, 512, layers[3], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature embedding layer
        final_dim = 512 * block.expansion
        self.fc = nn.Linear(final_dim, embedding_dim)
        self.bn_fc = nn.BatchNorm1d(embedding_dim)
        
        print(f"  ResNet-{depth} backbone configured for {embedding_dim}D embeddings")
    
    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        """Create ResNet layer"""
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = [block(in_channels, out_channels, stride, downsample)]
        in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through ResNet backbone"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.bn_fc(x)
        
        return x

class BasicBlock(nn.Module):
    """Basic ResNet block"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class Bottleneck(nn.Module):
    """Bottleneck ResNet block"""
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

# ============================================================================
# ARCFACE MODEL ARCHITECTURE
# ============================================================================

class ArcFace_FaceRecognition(nn.Module):
    """
    ArcFace Face Recognition Model
    
    Revolutionary Innovation:
    - Angular margin loss for enhanced discrimination
    - Hypersphere feature embedding
    - Superior face verification performance
    - Robust to intra-class variations
    """
    
    def __init__(self, num_classes=100, embedding_dim=512, backbone_depth=50, 
                 margin=0.5, scale=64.0):
        super(ArcFace_FaceRecognition, self).__init__()
        
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.scale = scale
        
        print(f"Building ArcFace Face Recognition Model...")
        
        # Backbone network
        self.backbone = ResNetBackbone(depth=backbone_depth, embedding_dim=embedding_dim)
        
        # ArcFace loss (used during training)
        self.arcface_loss = ArcFaceLoss(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            margin=margin,
            scale=scale
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Calculate statistics
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"ArcFace Architecture Summary:")
        print(f"  Backbone: ResNet-{backbone_depth}")
        print(f"  Embedding dimension: {embedding_dim}")
        print(f"  Number of identities: {num_classes}")
        print(f"  Angular margin: {margin}")
        print(f"  Scale factor: {scale}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Key innovation: Angular margin loss on hypersphere")
    
    def _initialize_weights(self):
        """Initialize ArcFace weights"""
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
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, labels=None):
        """Forward pass through ArcFace model"""
        # Extract features
        embeddings = self.backbone(x)
        
        if self.training and labels is not None:
            # Training mode: compute ArcFace loss
            loss, logits = self.arcface_loss(embeddings, labels)
            return embeddings, loss, logits
        else:
            # Inference mode: return normalized embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings
    
    def extract_features(self, x):
        """Extract normalized face embeddings for verification"""
        self.eval()
        
        with torch.no_grad():
            embeddings = self.backbone(x)
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def verify_faces(self, face1, face2, threshold=0.6):
        """
        Verify if two face images belong to the same person
        
        Args:
            face1, face2: Face images
            threshold: Cosine similarity threshold
            
        Returns:
            Boolean verification result and similarity score
        """
        # Extract features
        feat1 = self.extract_features(face1.unsqueeze(0))
        feat2 = self.extract_features(face2.unsqueeze(0))
        
        # Cosine similarity
        similarity = torch.cosine_similarity(feat1, feat2).item()
        
        # Verification decision
        is_same = similarity > threshold
        
        return is_same, similarity
    
    def get_embedding_analysis(self):
        """Analyze embedding space properties"""
        return {
            'embedding_dimension': self.embedding_dim,
            'normalization': 'L2 (unit hypersphere)',
            'angular_margin': self.margin,
            'scale_factor': self.scale,
            'metric': 'Cosine similarity',
            'verification_threshold': 'Typically 0.6-0.8',
            'feature_space': 'Hypersphere',
            'discrimination_method': 'Angular margin enhancement'
        }

# ============================================================================
# FACE VERIFICATION EVALUATION
# ============================================================================

def evaluate_face_verification(model, test_loader, device, num_pairs=1000):
    """
    Evaluate face verification performance
    Generate positive and negative pairs for ROC analysis
    """
    model.eval()
    
    print("Evaluating face verification performance...")
    
    # Collect embeddings and labels
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            embeddings = model.extract_features(data)
            
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)
    
    # Concatenate all embeddings and labels
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Generate verification pairs
    positive_pairs = []
    negative_pairs = []
    similarities_pos = []
    similarities_neg = []
    
    # Generate positive pairs (same identity)
    unique_labels = torch.unique(all_labels)
    pos_count = 0
    
    for label in unique_labels:
        indices = torch.where(all_labels == label)[0]
        if len(indices) >= 2:
            # Sample pairs from same identity
            for i in range(min(len(indices), 10)):  # Limit pairs per identity
                for j in range(i+1, min(len(indices), 10)):
                    if pos_count < num_pairs // 2:
                        emb1 = all_embeddings[indices[i]]
                        emb2 = all_embeddings[indices[j]]
                        similarity = torch.cosine_similarity(emb1, emb2, dim=0).item()
                        
                        positive_pairs.append((indices[i].item(), indices[j].item()))
                        similarities_pos.append(similarity)
                        pos_count += 1
    
    # Generate negative pairs (different identities)
    neg_count = 0
    while neg_count < num_pairs // 2:
        idx1 = torch.randint(0, len(all_labels), (1,)).item()
        idx2 = torch.randint(0, len(all_labels), (1,)).item()
        
        if all_labels[idx1] != all_labels[idx2]:
            emb1 = all_embeddings[idx1]
            emb2 = all_embeddings[idx2]
            similarity = torch.cosine_similarity(emb1, emb2, dim=0).item()
            
            negative_pairs.append((idx1, idx2))
            similarities_neg.append(similarity)
            neg_count += 1
    
    # Calculate verification metrics
    all_similarities = similarities_pos + similarities_neg
    all_labels_ver = [1] * len(similarities_pos) + [0] * len(similarities_neg)
    
    # ROC analysis at different thresholds
    thresholds = np.linspace(0.0, 1.0, 100)
    tpr_scores = []
    fpr_scores = []
    
    for threshold in thresholds:
        tp = sum(1 for sim, label in zip(all_similarities, all_labels_ver) 
                if sim >= threshold and label == 1)
        fp = sum(1 for sim, label in zip(all_similarities, all_labels_ver) 
                if sim >= threshold and label == 0)
        fn = sum(1 for sim, label in zip(all_similarities, all_labels_ver) 
                if sim < threshold and label == 1)
        tn = sum(1 for sim, label in zip(all_similarities, all_labels_ver) 
                if sim < threshold and label == 0)
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_scores.append(tpr)
        fpr_scores.append(fpr)
    
    # Find best threshold (closest to top-left corner)
    distances = [(tpr - 1)**2 + fpr**2 for tpr, fpr in zip(tpr_scores, fpr_scores)]
    best_idx = np.argmin(distances)
    best_threshold = thresholds[best_idx]
    best_tpr = tpr_scores[best_idx]
    best_fpr = fpr_scores[best_idx]
    
    print(f"Face Verification Results:")
    print(f"  Positive pairs: {len(similarities_pos)}")
    print(f"  Negative pairs: {len(similarities_neg)}")
    print(f"  Average positive similarity: {np.mean(similarities_pos):.4f}")
    print(f"  Average negative similarity: {np.mean(similarities_neg):.4f}")
    print(f"  Best threshold: {best_threshold:.4f}")
    print(f"  True Positive Rate: {best_tpr:.4f}")
    print(f"  False Positive Rate: {best_fpr:.4f}")
    print(f"  Accuracy at best threshold: {(best_tpr + (1-best_fpr))/2:.4f}")
    
    return {
        'positive_similarities': similarities_pos,
        'negative_similarities': similarities_neg,
        'best_threshold': best_threshold,
        'tpr': best_tpr,
        'fpr': best_fpr,
        'roc_curve': (fpr_scores, tpr_scores, thresholds)
    }

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_arcface_recognition(model, train_loader, test_loader, epochs=60, learning_rate=1e-3):
    """Train ArcFace face recognition model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # ArcFace training configuration
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=5e-4
    )
    
    # Learning rate scheduling
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[20, 35, 50], gamma=0.1
    )
    
    # Training tracking
    train_losses = []
    train_accuracies = []
    angular_stats = []
    
    model_name = model.__class__.__name__
    print(f"Training {model_name} on device: {device}")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        epoch_angular_stats = []
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            embeddings, loss, logits = model(data, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = logits.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
            
            # Collect angular statistics
            if batch_idx % 100 == 0:
                stats = model.arcface_loss.get_angular_statistics(embeddings, labels)
                epoch_angular_stats.append(stats)
            
            if batch_idx % 200 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.4f}, LR: {current_lr:.6f}')
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_train_acc = 100. * correct_train / total_train
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_train_acc)
        
        # Average angular statistics
        if epoch_angular_stats:
            avg_stats = {}
            for key in epoch_angular_stats[0].keys():
                avg_stats[key] = np.mean([stats[key] for stats in epoch_angular_stats])
            angular_stats.append(avg_stats)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), f'AI-ML-DL/Models/CNN/arcface_recognition_best.pth')
        
        print(f'Epoch {epoch+1}/{epochs}: Loss: {epoch_loss:.4f}, '
              f'Train Acc: {epoch_train_acc:.2f}%')
        
        if epoch_angular_stats:
            print(f'  Target angle: {avg_stats["target_angles_mean"]:.4f} ± {avg_stats["target_angles_std"]:.4f}')
            print(f'  Non-target angle: {avg_stats["non_target_angles_mean"]:.4f} ± {avg_stats["non_target_angles_std"]:.4f}')
        
        # Early stopping for demonstration
        if epoch_loss < 0.1:
            print(f"Convergence reached at epoch {epoch+1}")
            break
    
    print(f"Best training loss: {best_loss:.4f}")
    return train_losses, train_accuracies, angular_stats

# ============================================================================
# VISUALIZATION AND ANALYSIS
# ============================================================================

def visualize_arcface_innovations():
    """Visualize ArcFace's face recognition innovations"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss function comparison
    ax = axes[0, 0]
    ax.set_title('Loss Function Evolution', fontsize=14)
    
    # Simulate loss landscapes
    theta = np.linspace(0, np.pi, 100)
    
    # Softmax loss (no margin)
    softmax_loss = -np.log(np.cos(theta))
    
    # ArcFace loss (with angular margin)
    margin = 0.5
    arcface_theta = theta + margin
    arcface_theta = np.clip(arcface_theta, 0, np.pi)
    arcface_loss = -np.log(np.cos(arcface_theta))
    
    ax.plot(theta, softmax_loss, label='Softmax Loss', linewidth=2, color='blue')
    ax.plot(theta, arcface_loss, label='ArcFace Loss', linewidth=2, color='red')
    
    ax.axvline(x=margin, color='red', linestyle='--', alpha=0.7, label=f'Angular Margin ({margin})')
    ax.set_xlabel('Angle θ (radians)')
    ax.set_ylabel('Loss Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Feature distribution on hypersphere
    ax = axes[0, 1]
    ax.set_title('Feature Distribution on Hypersphere', fontsize=14)
    
    # Simulate feature points for different classes
    np.random.seed(42)
    n_classes = 3
    colors = ['red', 'blue', 'green']
    
    for i, color in enumerate(colors):
        # Generate points around class center
        center_angle = i * 2 * np.pi / n_classes
        angles = np.random.normal(center_angle, 0.3, 20)
        
        # Without margin (more spread)
        x_no_margin = np.cos(angles) + np.random.normal(0, 0.1, 20)
        y_no_margin = np.sin(angles) + np.random.normal(0, 0.1, 20)
        
        # With margin (more compact)
        x_margin = 0.8 * np.cos(angles) + np.random.normal(0, 0.05, 20)
        y_margin = 0.8 * np.sin(angles) + np.random.normal(0, 0.05, 20)
        
        if i == 0:  # Only show legend for first class
            ax.scatter(x_no_margin, y_no_margin, c=color, alpha=0.5, s=50, 
                      marker='o', label='Without ArcFace')
            ax.scatter(x_margin, y_margin, c=color, alpha=0.8, s=50, 
                      marker='s', label='With ArcFace')
        else:
            ax.scatter(x_no_margin, y_no_margin, c=color, alpha=0.5, s=50, marker='o')
            ax.scatter(x_margin, y_margin, c=color, alpha=0.8, s=50, marker='s')
    
    # Draw unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--')
    ax.add_patch(circle)
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0, -1.3, 'Unit Hypersphere', ha='center', fontweight='bold')
    
    # Angular margin visualization
    ax = axes[1, 0]
    ax.set_title('Angular Margin Effect', fontsize=14)
    
    # Draw angle representation
    theta_target = np.pi/4  # Target class angle
    margin_val = 0.5
    
    # Original decision boundary
    x_orig = [0, np.cos(theta_target)]
    y_orig = [0, np.sin(theta_target)]
    ax.plot(x_orig, y_orig, 'b-', linewidth=3, label='Original Target')
    
    # With angular margin
    x_margin = [0, np.cos(theta_target + margin_val)]
    y_margin = [0, np.sin(theta_target + margin_val)]
    ax.plot(x_margin, y_margin, 'r-', linewidth=3, label='With Angular Margin')
    
    # Draw the margin arc
    arc_angles = np.linspace(theta_target, theta_target + margin_val, 20)
    arc_x = 0.5 * np.cos(arc_angles)
    arc_y = 0.5 * np.sin(arc_angles)
    ax.plot(arc_x, arc_y, 'g-', linewidth=4, label=f'Margin ({margin_val:.1f} rad)')
    
    # Add annotations
    ax.annotate('', xy=(np.cos(theta_target), np.sin(theta_target)), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax.annotate('', xy=(np.cos(theta_target + margin_val), np.sin(theta_target + margin_val)), 
               xytext=(0, 0), arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Performance comparison
    ax = axes[1, 1]
    methods = ['Softmax', 'Center Loss', 'SphereFace', 'CosFace', 'ArcFace']
    accuracy = [94.2, 97.1, 98.5, 99.1, 99.4]  # Example LFW accuracies
    
    bars = ax.bar(methods, accuracy, color=['#95A5A6', '#3498DB', '#E67E22', '#9B59B6', '#E74C3C'])
    ax.set_title('Face Recognition Accuracy Comparison', fontsize=14)
    ax.set_ylabel('LFW Accuracy (%)')
    
    # Add accuracy labels
    for bar, acc in zip(bars, accuracy):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{acc}%', ha='center', va='bottom')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/CNN/015_arcface_innovations.png', dpi=300, bbox_inches='tight')
    print("ArcFace innovations visualization saved: 015_arcface_innovations.png")

def visualize_face_verification_roc(verification_results):
    """Visualize ROC curve for face verification"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC Curve
    ax = axes[0]
    fpr_scores, tpr_scores, thresholds = verification_results['roc_curve']
    
    ax.plot(fpr_scores, tpr_scores, 'b-', linewidth=2, label='ArcFace ROC')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Random Classifier')
    
    # Mark best operating point
    best_fpr = verification_results['fpr']
    best_tpr = verification_results['tpr']
    ax.plot(best_fpr, best_tpr, 'ro', markersize=8, label=f'Best Point (TPR={best_tpr:.3f})')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Face Verification ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Similarity distribution
    ax = axes[1]
    pos_sim = verification_results['positive_similarities']
    neg_sim = verification_results['negative_similarities']
    
    ax.hist(neg_sim, bins=30, alpha=0.7, color='red', label='Different Persons', density=True)
    ax.hist(pos_sim, bins=30, alpha=0.7, color='green', label='Same Person', density=True)
    
    # Mark threshold
    threshold = verification_results['best_threshold']
    ax.axvline(x=threshold, color='blue', linestyle='--', linewidth=2, 
              label=f'Threshold ({threshold:.3f})')
    
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.set_title('Similarity Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/CNN/015_face_verification_roc.png', dpi=300, bbox_inches='tight')
    print("Face verification ROC analysis saved: 015_face_verification_roc.png")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"=== ArcFace Face Recognition Implementation ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Create face recognition dataset
    train_loader, test_loader, num_identities = create_face_recognition_dataset()
    
    # Initialize ArcFace models
    arcface_512 = ArcFace_FaceRecognition(
        num_classes=num_identities, embedding_dim=512, backbone_depth=50
    )
    
    arcface_256 = ArcFace_FaceRecognition(
        num_classes=num_identities, embedding_dim=256, backbone_depth=18
    )
    
    # Compare model complexities
    arcface_512_params = sum(p.numel() for p in arcface_512.parameters())
    arcface_256_params = sum(p.numel() for p in arcface_256.parameters())
    
    print(f"\nModel Complexity Comparison:")
    print(f"  ArcFace-512D (ResNet-50): {arcface_512_params:,} parameters")
    print(f"  ArcFace-256D (ResNet-18): {arcface_256_params:,} parameters")
    print(f"  Parameter ratio: {arcface_512_params/arcface_256_params:.2f}x")
    
    # Analyze embedding properties
    embedding_analysis = arcface_512.get_embedding_analysis()
    
    print(f"\nArcFace Embedding Analysis:")
    for key, value in embedding_analysis.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Generate visualizations
    print("\nGenerating ArcFace analysis...")
    visualize_arcface_innovations()
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("ARCFACE FACE RECOGNITION SUMMARY")
    print("="*70)
    
    print(f"\nHistorical Context ({YEAR}):")
    print(f"  Innovation: {INNOVATION}")
    print(f"  Previous Limitation: {PREVIOUS_LIMITATION}")
    print(f"  Impact: {IMPACT}")
    
    print(f"\nARCFACE REVOLUTIONARY INNOVATIONS:")
    print("="*50)
    print("1. ANGULAR MARGIN LOSS:")
    print("   • Additive angular margin in hypersphere")
    print("   • Enhanced intra-class compactness")
    print("   • Improved inter-class separability")
    print("   • cos(θ + m) formulation for target class")
    
    print("\n2. HYPERSPHERE FEATURE EMBEDDING:")
    print("   • L2 normalization projects features to unit sphere")
    print("   • Angular distance as similarity metric")
    print("   • Stable feature space geometry")
    print("   • Consistent feature magnitudes")
    
    print("\n3. SUPERIOR DISCRIMINATION:")
    print("   • Better handling of intra-class variations")
    print("   • Robust to pose, illumination, expression changes")
    print("   • High-quality feature representations")
    print("   • Improved verification performance")
    
    print("\n4. PRACTICAL DEPLOYMENT:")
    print("   • Efficient inference with cosine similarity")
    print("   • Scalable to large identity databases")
    print("   • High-security biometric applications")
    print("   • Real-world face recognition systems")
    
    print(f"\nTECHNICAL ACHIEVEMENTS:")
    print("="*40)
    print("• State-of-the-art face verification accuracy")
    print("• Superior embedding quality and discrimination")
    print("• Robust to challenging conditions")
    print("• Practical deployment in security systems")
    print("• Influenced numerous metric learning methods")
    
    print(f"\nEMBEDDING SPACE PROPERTIES:")
    for key, value in embedding_analysis.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print(f"\nARCFACE VS TRADITIONAL METHODS:")
    print("="*40)
    print("• Softmax: Classification loss, poor verification")
    print("• ArcFace: Angular margin loss, superior verification")
    print("• Traditional: Euclidean distance metrics")
    print("• ArcFace: Cosine similarity on hypersphere")
    print("• Traditional: Ad-hoc feature normalization")
    print("• ArcFace: Principled hypersphere embedding")
    
    print(f"\nFACE RECOGNITION APPLICATIONS:")
    print("="*40)
    print("• Security and surveillance systems")
    print("• Mobile device authentication")
    print("• Border control and identity verification")
    print("• Social media photo tagging")
    print("• Access control systems")
    print("• Law enforcement identification")
    
    print(f"\nMARGIN-BASED LOSS EVOLUTION:")
    print("="*40)
    print("• SphereFace: Multiplicative angular margin")
    print("• CosFace: Additive cosine margin")
    print("• ArcFace: Additive angular margin (most effective)")
    print("• Each improves upon previous limitations")
    print("• ArcFace achieves best discrimination")
    
    print(f"\nHISTORICAL IMPACT:")
    print("="*40)
    print("• Revolutionized face recognition field")
    print("• Established angular margin as standard")
    print("• Enabled high-security biometric systems")
    print("• Influenced deep metric learning research")
    print("• Set new benchmarks on face verification datasets")
    print("• Made face recognition practical for real applications")
    
    return {
        'model': 'ArcFace Face Recognition',
        'year': YEAR,
        'innovation': INNOVATION,
        'embedding_analysis': embedding_analysis,
        'parameter_comparison': {
            'arcface_512': arcface_512_params,
            'arcface_256': arcface_256_params
        }
    }

if __name__ == "__main__":
    results = main()