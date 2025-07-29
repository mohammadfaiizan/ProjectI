#!/usr/bin/env python3
"""PyTorch Classification Loss Functions - CrossEntropy, NLL, Binary CE syntax"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

print("=== Classification Loss Functions Overview ===")

print("Common classification losses:")
print("1. Cross-Entropy Loss (multi-class)")
print("2. Binary Cross-Entropy Loss")
print("3. Negative Log-Likelihood Loss")
print("4. Focal Loss (for imbalanced datasets)")
print("5. Label Smoothing Loss")
print("6. Weighted Cross-Entropy")
print("7. Multi-label losses")
print("8. Confidence-based losses")

print("\n=== Cross-Entropy Loss ===")

# Cross-Entropy Loss - most common classification loss
ce_loss = nn.CrossEntropyLoss()
ce_loss_no_reduction = nn.CrossEntropyLoss(reduction='none')
ce_loss_sum = nn.CrossEntropyLoss(reduction='sum')

# Sample data for multi-class classification
batch_size, num_classes = 10, 5
logits = torch.randn(batch_size, num_classes)
targets = torch.randint(0, num_classes, (batch_size,))

print(f"Logits shape: {logits.shape}")
print(f"Targets shape: {targets.shape}")
print(f"Sample targets: {targets[:5]}")

# Different CE computations
ce_mean = ce_loss(logits, targets)
ce_none = ce_loss_no_reduction(logits, targets)
ce_sum = ce_loss_sum(logits, targets)

print(f"CE Loss (mean): {ce_mean.item():.6f}")
print(f"CE Loss (none) shape: {ce_none.shape}")
print(f"CE Loss (sum): {ce_sum.item():.6f}")

# Manual Cross-Entropy computation
log_softmax = F.log_softmax(logits, dim=1)
ce_manual = F.nll_loss(log_softmax, targets)
print(f"Manual CE (via log_softmax + nll): {ce_manual.item():.6f}")

# Using softmax + log + nll manually
softmax_probs = F.softmax(logits, dim=1)
log_probs = torch.log(softmax_probs)
ce_manual2 = F.nll_loss(log_probs, targets)
print(f"Manual CE (softmax + log + nll): {ce_manual2.item():.6f}")

# Functional API
ce_functional = F.cross_entropy(logits, targets)
print(f"Functional CE: {ce_functional.item():.6f}")

print("\n=== Negative Log-Likelihood Loss ===")

# NLL Loss - expects log-probabilities as input
nll_loss = nn.NLLLoss()
nll_loss_no_reduction = nn.NLLLoss(reduction='none')

# Convert logits to log-probabilities
log_probs = F.log_softmax(logits, dim=1)

# NLL computations
nll_mean = nll_loss(log_probs, targets)
nll_none = nll_loss_no_reduction(log_probs, targets)

print(f"NLL Loss (mean): {nll_mean.item():.6f}")
print(f"NLL Loss (none) shape: {nll_none.shape}")

# Manual NLL computation
nll_manual = -log_probs.gather(1, targets.unsqueeze(1)).squeeze().mean()
print(f"Manual NLL: {nll_manual.item():.6f}")

# Functional API
nll_functional = F.nll_loss(log_probs, targets)
print(f"Functional NLL: {nll_functional.item():.6f}")

print(f"CE and NLL equivalence: {torch.allclose(ce_mean, nll_mean)}")

print("\n=== Binary Cross-Entropy Loss ===")

# Binary Cross-Entropy for binary classification
bce_loss = nn.BCELoss()
bce_loss_no_reduction = nn.BCELoss(reduction='none')

# BCE with logits (more numerically stable)
bce_logits_loss = nn.BCEWithLogitsLoss()
bce_logits_no_reduction = nn.BCEWithLogitsLoss(reduction='none')

# Sample binary data
binary_logits = torch.randn(10, 1)
binary_targets = torch.randint(0, 2, (10, 1)).float()

print(f"Binary logits shape: {binary_logits.shape}")
print(f"Binary targets: {binary_targets.squeeze()}")

# BCE with logits (recommended)
bce_logits_mean = bce_logits_loss(binary_logits, binary_targets)
bce_logits_none = bce_logits_no_reduction(binary_logits, binary_targets)

print(f"BCE with Logits (mean): {bce_logits_mean.item():.6f}")
print(f"BCE with Logits (none) shape: {bce_logits_none.shape}")

# Manual BCE with logits
sigmoid_probs = torch.sigmoid(binary_logits)
bce_manual = F.binary_cross_entropy(sigmoid_probs, binary_targets)
print(f"Manual BCE (sigmoid + bce): {bce_manual.item():.6f}")

# More numerically stable manual computation
bce_manual_stable = F.binary_cross_entropy_with_logits(binary_logits, binary_targets)
print(f"Manual BCE with logits: {bce_manual_stable.item():.6f}")

# Multi-label binary classification
multi_label_logits = torch.randn(10, 5)  # 5 binary labels per sample
multi_label_targets = torch.randint(0, 2, (10, 5)).float()

multi_bce = bce_logits_loss(multi_label_logits, multi_label_targets)
print(f"Multi-label BCE: {multi_bce.item():.6f}")

print("\n=== Weighted Cross-Entropy Loss ===")

# Weighted CE for imbalanced datasets
class_counts = torch.tensor([1000, 500, 100, 50, 10])  # Imbalanced classes
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_weights)

weighted_ce_loss = nn.CrossEntropyLoss(weight=class_weights)
weighted_ce = weighted_ce_loss(logits, targets)

print(f"Class counts: {class_counts}")
print(f"Class weights: {class_weights}")
print(f"Regular CE: {ce_mean.item():.6f}")
print(f"Weighted CE: {weighted_ce.item():.6f}")

# Class-specific weight analysis
ce_per_class = ce_loss_no_reduction(logits, targets)
for i in range(num_classes):
    mask = targets == i
    if mask.any():
        avg_loss = ce_per_class[mask].mean()
        count = mask.sum()
        print(f"Class {i}: {count} samples, avg loss: {avg_loss:.6f}")

print("\n=== Focal Loss ===")

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BinaryFocalLoss(nn.Module):
    """Focal Loss for binary classification"""
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        
        # Apply alpha weighting
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_weight * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Test Focal Loss
focal_loss = FocalLoss(alpha=1, gamma=2)
binary_focal_loss = BinaryFocalLoss(alpha=0.25, gamma=2)

focal_result = focal_loss(logits, targets)
binary_focal_result = binary_focal_loss(binary_logits, binary_targets)

print(f"Regular CE: {ce_mean.item():.6f}")
print(f"Focal Loss (γ=2): {focal_result.item():.6f}")
print(f"Binary Focal Loss: {binary_focal_result.item():.6f}")

# Test with different gamma values
for gamma in [0, 1, 2, 5]:
    focal_gamma = FocalLoss(gamma=gamma)
    result = focal_gamma(logits, targets)
    print(f"Focal Loss (γ={gamma}): {result.item():.6f}")

print("\n=== Label Smoothing Loss ===")

class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Cross-Entropy Loss"""
    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Create smoothed targets
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = -smooth_targets * log_probs
        
        if self.reduction == 'mean':
            return loss.sum(dim=1).mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.sum(dim=1)

# Test Label Smoothing
label_smooth_loss = LabelSmoothingLoss(smoothing=0.1)
smooth_result = label_smooth_loss(logits, targets)

print(f"Regular CE: {ce_mean.item():.6f}")
print(f"Label Smoothing (0.1): {smooth_result.item():.6f}")

# Test with different smoothing values
for smoothing in [0.0, 0.05, 0.1, 0.2]:
    smooth_loss = LabelSmoothingLoss(smoothing=smoothing)
    result = smooth_loss(logits, targets)
    print(f"Label Smoothing ({smoothing}): {result.item():.6f}")

print("\n=== Multi-Label Classification Losses ===")

class MultiLabelSoftMarginLoss(nn.Module):
    """Multi-label soft margin loss"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # inputs: [batch_size, num_labels]
        # targets: [batch_size, num_labels] (0 or 1)
        loss = torch.log(1 + torch.exp(-inputs * (2 * targets - 1)))
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification"""
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, reduction='mean'):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Probability
        probs = torch.sigmoid(inputs)
        
        # Asymmetric clipping
        if self.clip is not None and self.clip > 0:
            probs = torch.clamp(probs, self.clip, 1 - self.clip)
        
        # Calculate loss
        pos_loss = targets * torch.log(probs)
        neg_loss = (1 - targets) * torch.log(1 - probs)
        
        # Asymmetric focusing
        pos_loss = pos_loss * (1 - probs) ** self.gamma_pos
        neg_loss = neg_loss * probs ** self.gamma_neg
        
        loss = -(pos_loss + neg_loss)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Test multi-label losses
ml_soft_margin = MultiLabelSoftMarginLoss()
asymmetric_loss = AsymmetricLoss()

ml_soft_result = ml_soft_margin(multi_label_logits, multi_label_targets)
asymmetric_result = asymmetric_loss(multi_label_logits, multi_label_targets)

print(f"Multi-label BCE: {multi_bce.item():.6f}")
print(f"Multi-label Soft Margin: {ml_soft_result.item():.6f}")
print(f"Asymmetric Loss: {asymmetric_result.item():.6f}")

# Built-in PyTorch multi-label soft margin
builtin_ml_loss = nn.MultiLabelSoftMarginLoss()
builtin_result = builtin_ml_loss(multi_label_logits, multi_label_targets)
print(f"Built-in Multi-label Soft Margin: {builtin_result.item():.6f}")

print("\n=== Confidence-Based Losses ===")

class ConfidencePenaltyLoss(nn.Module):
    """Penalize overconfident predictions"""
    def __init__(self, lambda_reg=0.1, reduction='mean'):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        
        # Confidence penalty (entropy regularization)
        probs = F.softmax(inputs, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        confidence_penalty = -self.lambda_reg * entropy
        
        total_loss = ce_loss + confidence_penalty
        
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss

class TemperatureScaledLoss(nn.Module):
    """Temperature-scaled cross-entropy for calibration"""
    def __init__(self, temperature=1.0, reduction='mean'):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        scaled_logits = inputs / self.temperature
        loss = F.cross_entropy(scaled_logits, targets, reduction=self.reduction)
        return loss

# Test confidence-based losses
confidence_loss = ConfidencePenaltyLoss(lambda_reg=0.1)
temperature_loss = TemperatureScaledLoss(temperature=2.0)

confidence_result = confidence_loss(logits, targets)
temperature_result = temperature_loss(logits, targets)

print(f"Regular CE: {ce_mean.item():.6f}")
print(f"Confidence Penalty Loss: {confidence_result.item():.6f}")
print(f"Temperature Scaled Loss: {temperature_result.item():.6f}")
print(f"Learned temperature: {temperature_loss.temperature.item():.6f}")

print("\n=== Advanced Classification Losses ===")

class TripletLoss(nn.Module):
    """Triplet Loss for metric learning"""
    def __init__(self, margin=1.0, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class CenterLoss(nn.Module):
    """Center Loss for face recognition"""
    def __init__(self, num_classes, feature_dim, alpha=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.alpha = alpha
        
        # Centers for each class
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))
    
    def forward(self, features, targets):
        batch_size = features.size(0)
        
        # Compute distances to centers
        centers_batch = self.centers[targets]  # [batch_size, feature_dim]
        loss = F.mse_loss(features, centers_batch)
        
        return loss

class ArcFaceLoss(nn.Module):
    """ArcFace Loss for face recognition"""
    def __init__(self, feature_dim, num_classes, margin=0.5, scale=64):
        super().__init__()
        self.margin = margin
        self.scale = scale
        
        self.weight = nn.Parameter(torch.randn(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, features, targets):
        # Normalize features and weights
        features = F.normalize(features, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(features, weight)
        
        # Add margin to target class
        phi = cosine - self.margin
        
        # One-hot encoding for targets
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Apply margin only to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale
        
        return F.cross_entropy(output, targets)

# Test advanced losses
embedding_dim = 128
num_classes = 10

# Generate sample embeddings
anchor_emb = torch.randn(10, embedding_dim)
positive_emb = torch.randn(10, embedding_dim)
negative_emb = torch.randn(10, embedding_dim)
features = torch.randn(10, embedding_dim)

triplet_loss = TripletLoss(margin=1.0)
center_loss = CenterLoss(num_classes, embedding_dim)
arcface_loss = ArcFaceLoss(embedding_dim, num_classes)

triplet_result = triplet_loss(anchor_emb, positive_emb, negative_emb)
center_result = center_loss(features, targets)
arcface_result = arcface_loss(features, targets)

print(f"Triplet Loss: {triplet_result.item():.6f}")
print(f"Center Loss: {center_result.item():.6f}")
print(f"ArcFace Loss: {arcface_result.item():.6f}")

print("\n=== Loss Combination Strategies ===")

class CombinedClassificationLoss(nn.Module):
    """Combine multiple classification losses"""
    def __init__(self, ce_weight=1.0, focal_weight=0.5, smoothing_weight=0.3):
        super().__init__()
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.smoothing_weight = smoothing_weight
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss(gamma=2)
        self.smooth_loss = LabelSmoothingLoss(smoothing=0.1)
    
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        focal_loss = self.focal_loss(inputs, targets)
        smooth_loss = self.smooth_loss(inputs, targets)
        
        total_loss = (self.ce_weight * ce_loss + 
                     self.focal_weight * focal_loss + 
                     self.smoothing_weight * smooth_loss)
        
        return total_loss, {
            'ce': ce_loss,
            'focal': focal_loss,
            'smooth': smooth_loss,
            'total': total_loss
        }

# Test combined loss
combined_cls_loss = CombinedClassificationLoss()
combined_result, loss_components = combined_cls_loss(logits, targets)

print(f"Combined Classification Loss: {combined_result.item():.6f}")
print("Components:")
for name, value in loss_components.items():
    print(f"  {name}: {value.item():.6f}")

print("\n=== Classification Loss Best Practices ===")

print("Loss Selection Guidelines:")
print("1. Cross-Entropy: Standard multi-class classification")
print("2. Binary CE with Logits: Binary classification (more stable)")
print("3. Focal Loss: Imbalanced datasets, hard example mining")
print("4. Label Smoothing: Prevent overconfidence, improve calibration")
print("5. Weighted CE: Class imbalance with known class frequencies")
print("6. Multi-label losses: Multiple independent binary classifications")

print("\nImplementation Tips:")
print("1. Use BCEWithLogitsLoss instead of BCE for numerical stability")
print("2. Apply label smoothing to prevent overconfidence")
print("3. Use class weights for imbalanced datasets")
print("4. Consider focal loss for extreme class imbalance")
print("5. Monitor per-class accuracy and loss during training")
print("6. Use temperature scaling for calibration")

print("\nCommon Issues:")
print("1. Vanishing gradients with saturated predictions")
print("2. Exploding gradients with very confident wrong predictions")
print("3. Class imbalance leading to biased predictions")
print("4. Overconfidence in predictions")
print("5. Poor calibration (confidence ≠ accuracy)")

print("\nDebugging Classification Losses:")
print("1. Check prediction distribution across classes")
print("2. Monitor per-class loss values")
print("3. Visualize confusion matrix")
print("4. Check gradient norms")
print("5. Verify data loading and target format")
print("6. Test with perfect predictions (should give ~0 loss)")

print("\n=== Classification Losses Complete ===")

# Memory cleanup
del logits, targets, binary_logits, binary_targets
del multi_label_logits, multi_label_targets, features
del anchor_emb, positive_emb, negative_emb