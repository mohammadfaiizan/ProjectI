#!/usr/bin/env python3
"""PyTorch Loss Functions - All loss functions and custom implementations"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== Classification Loss Functions ===")

# Cross Entropy Loss
ce_loss = nn.CrossEntropyLoss()
logits = torch.randn(32, 10)  # batch_size=32, num_classes=10
targets = torch.randint(0, 10, (32,))  # class indices

ce_output = ce_loss(logits, targets)
ce_func = F.cross_entropy(logits, targets)

print(f"CrossEntropy loss: {ce_output.item():.4f}")
print(f"Functional CE equal: {torch.allclose(ce_output, ce_func)}")

# Negative Log Likelihood Loss
log_probs = F.log_softmax(logits, dim=1)
nll_loss = nn.NLLLoss()
nll_output = nll_loss(log_probs, targets)
nll_func = F.nll_loss(log_probs, targets)

print(f"NLL loss: {nll_output.item():.4f}")
print(f"NLL equals CE: {torch.allclose(ce_output, nll_output)}")  # Should be True

# Binary Cross Entropy
binary_logits = torch.randn(32, 1)
binary_targets = torch.randint(0, 2, (32, 1)).float()

bce_loss = nn.BCELoss()
bce_sigmoid = nn.BCEWithLogitsLoss()

binary_probs = torch.sigmoid(binary_logits)
bce_output = bce_loss(binary_probs, binary_targets)
bce_logits_output = bce_sigmoid(binary_logits, binary_targets)

print(f"BCE loss: {bce_output.item():.4f}")
print(f"BCE with logits: {bce_logits_output.item():.4f}")

print("\n=== Regression Loss Functions ===")

# Mean Squared Error
mse_loss = nn.MSELoss()
predictions = torch.randn(32, 1)
targets_reg = torch.randn(32, 1)

mse_output = mse_loss(predictions, targets_reg)
mse_func = F.mse_loss(predictions, targets_reg)

print(f"MSE loss: {mse_output.item():.4f}")
print(f"MSE functional equal: {torch.allclose(mse_output, mse_func)}")

# Mean Absolute Error (L1 Loss)
mae_loss = nn.L1Loss()
mae_output = mae_loss(predictions, targets_reg)
mae_func = F.l1_loss(predictions, targets_reg)

print(f"MAE/L1 loss: {mae_output.item():.4f}")
print(f"MAE functional equal: {torch.allclose(mae_output, mae_func)}")

# Smooth L1 Loss (Huber Loss)
smooth_l1_loss = nn.SmoothL1Loss(beta=1.0)
smooth_l1_output = smooth_l1_loss(predictions, targets_reg)
smooth_l1_func = F.smooth_l1_loss(predictions, targets_reg, beta=1.0)

print(f"Smooth L1 loss: {smooth_l1_output.item():.4f}")
print(f"Huber loss: {F.huber_loss(predictions, targets_reg, delta=1.0).item():.4f}")

print("\n=== Advanced Loss Functions ===")

# Focal Loss (custom implementation)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

focal_loss = FocalLoss(alpha=1, gamma=2)
focal_output = focal_loss(logits, targets)
print(f"Focal loss: {focal_output.item():.4f}")

# Label Smoothing Cross Entropy
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, inputs, targets):
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        targets_one_hot = F.one_hot(targets, num_classes).float()
        targets_smooth = (1 - self.epsilon) * targets_one_hot + self.epsilon / num_classes
        loss = -(targets_smooth * log_probs).sum(dim=-1).mean()
        return loss

label_smooth_loss = LabelSmoothingCrossEntropy(epsilon=0.1)
label_smooth_output = label_smooth_loss(logits, targets)
print(f"Label smoothing CE: {label_smooth_output.item():.4f}")

# Dice Loss (for segmentation)
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice_coeff = (2 * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1 - dice_coeff
        return dice_loss

# Test Dice loss
seg_logits = torch.randn(4, 1, 64, 64)
seg_targets = torch.randint(0, 2, (4, 1, 64, 64)).float()

dice_loss = DiceLoss()
dice_output = dice_loss(seg_logits, seg_targets)
print(f"Dice loss: {dice_output.item():.4f}")

print("\n=== Multi-task Loss Functions ===")

# Weighted Multi-task Loss
class MultiTaskLoss(nn.Module):
    def __init__(self, loss_weights=None):
        super().__init__()
        self.loss_weights = loss_weights or [1.0, 1.0]
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, cls_logits, cls_targets, reg_preds, reg_targets):
        cls_loss = self.ce_loss(cls_logits, cls_targets)
        reg_loss = self.mse_loss(reg_preds, reg_targets)
        
        total_loss = (self.loss_weights[0] * cls_loss + 
                     self.loss_weights[1] * reg_loss)
        
        return total_loss, cls_loss, reg_loss

# Test multi-task loss
cls_logits = torch.randn(32, 10)
cls_targets = torch.randint(0, 10, (32,))
reg_preds = torch.randn(32, 1)
reg_targets = torch.randn(32, 1)

multitask_loss = MultiTaskLoss(loss_weights=[1.0, 0.5])
total_loss, cls_loss, reg_loss = multitask_loss(cls_logits, cls_targets, reg_preds, reg_targets)

print(f"Multi-task total loss: {total_loss.item():.4f}")
print(f"Classification loss: {cls_loss.item():.4f}")
print(f"Regression loss: {reg_loss.item():.4f}")

print("\n=== Ranking and Similarity Losses ===")

# Triplet Loss
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, reduction='mean')
anchor = torch.randn(32, 128)
positive = torch.randn(32, 128)
negative = torch.randn(32, 128)

triplet_output = triplet_loss(anchor, positive, negative)
triplet_func = F.triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2)

print(f"Triplet loss: {triplet_output.item():.4f}")
print(f"Triplet functional equal: {torch.allclose(triplet_output, triplet_func)}")

# Margin Ranking Loss
margin_ranking_loss = nn.MarginRankingLoss(margin=0.0)
input1 = torch.randn(32, 1)
input2 = torch.randn(32, 1)
target_ranking = torch.randint(-1, 2, (32,), dtype=torch.float)  # -1, 0, or 1

margin_output = margin_ranking_loss(input1, input2, target_ranking)
print(f"Margin ranking loss: {margin_output.item():.4f}")

# Cosine Embedding Loss
cosine_embedding_loss = nn.CosineEmbeddingLoss(margin=0.0)
embedding1 = torch.randn(32, 128)
embedding2 = torch.randn(32, 128)
target_cosine = torch.randint(-1, 2, (32,), dtype=torch.float, step=2)  # -1 or 1

cosine_output = cosine_embedding_loss(embedding1, embedding2, target_cosine)
print(f"Cosine embedding loss: {cosine_output.item():.4f}")

print("\n=== Contrastive Learning Losses ===")

# InfoNCE Loss (Contrastive Loss)
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features):
        batch_size = features.size(0)
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create labels (positive pairs are diagonal)
        labels = torch.arange(batch_size, device=features.device)
        
        # Compute loss
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

# Test InfoNCE
features_contrastive = torch.randn(64, 256)
infonce_loss = InfoNCELoss(temperature=0.1)
infonce_output = infonce_loss(features_contrastive)
print(f"InfoNCE loss: {infonce_output.item():.4f}")

# Supervised Contrastive Loss
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        batch_size = features.size(0)
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove diagonal
        mask = mask - torch.eye(batch_size, device=features.device)
        
        # Compute loss
        exp_sim = torch.exp(similarity_matrix)
        sum_exp_sim = exp_sim.sum(dim=1, keepdim=True)
        
        log_prob = similarity_matrix - torch.log(sum_exp_sim)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)
        
        loss = -mean_log_prob_pos.mean()
        return loss

# Test Supervised Contrastive
features_supcon = torch.randn(32, 256)
labels_supcon = torch.randint(0, 5, (32,))
supcon_loss = SupConLoss(temperature=0.1)
supcon_output = supcon_loss(features_supcon, labels_supcon)
print(f"Supervised contrastive loss: {supcon_output.item():.4f}")

print("\n=== Loss Function Reduction ===")

# Different reduction methods
mse_none = nn.MSELoss(reduction='none')
mse_mean = nn.MSELoss(reduction='mean')
mse_sum = nn.MSELoss(reduction='sum')

pred_red = torch.randn(10, 5)
target_red = torch.randn(10, 5)

loss_none = mse_none(pred_red, target_red)
loss_mean = mse_mean(pred_red, target_red)
loss_sum = mse_sum(pred_red, target_red)

print(f"Loss none shape: {loss_none.shape}")
print(f"Loss mean: {loss_mean.item():.4f}")
print(f"Loss sum: {loss_sum.item():.4f}")
print(f"Manual mean: {loss_none.mean().item():.4f}")
print(f"Manual sum: {loss_none.sum().item():.4f}")

print("\n=== Weighted Loss Functions ===")

# Class weights for imbalanced datasets
class_weights = torch.tensor([1.0, 2.0, 3.0, 1.5, 2.5, 1.0, 1.0, 1.0, 1.0, 1.0])
weighted_ce = nn.CrossEntropyLoss(weight=class_weights)

weighted_output = weighted_ce(logits, targets)
print(f"Weighted CE loss: {weighted_output.item():.4f}")

# Sample weights
ce_reduction_none = nn.CrossEntropyLoss(reduction='none')
sample_weights = torch.rand(32)

loss_per_sample = ce_reduction_none(logits, targets)
weighted_loss = (loss_per_sample * sample_weights).mean()

print(f"Sample weighted loss: {weighted_loss.item():.4f}")

print("\n=== Custom Loss Function Combinations ===")

# Combined Loss (e.g., for segmentation)
class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, logits, targets):
        # Convert targets for dice loss
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).permute(0, 3, 1, 2).float()
        
        ce = self.ce_loss(logits, targets)
        dice = 0
        for i in range(logits.size(1)):
            dice += self.dice_loss(logits[:, i:i+1], targets_one_hot[:, i:i+1])
        dice /= logits.size(1)
        
        combined = self.ce_weight * ce + self.dice_weight * dice
        return combined, ce, dice

# Test combined loss
seg_logits_multi = torch.randn(4, 3, 32, 32)
seg_targets_multi = torch.randint(0, 3, (4, 32, 32))

combined_loss = CombinedLoss(ce_weight=1.0, dice_weight=0.5)
combined_output, ce_part, dice_part = combined_loss(seg_logits_multi, seg_targets_multi)

print(f"Combined loss: {combined_output.item():.4f}")
print(f"CE component: {ce_part.item():.4f}")
print(f"Dice component: {dice_part.item():.4f}")

print("\n=== Loss Function Utilities ===")

# Loss function with ignore index
ce_ignore = nn.CrossEntropyLoss(ignore_index=-1)
targets_ignore = targets.clone()
targets_ignore[:5] = -1  # Ignore first 5 samples

loss_ignore = ce_ignore(logits, targets_ignore)
print(f"Loss with ignore index: {loss_ignore.item():.4f}")

# Soft targets (knowledge distillation)
def soft_cross_entropy(pred, soft_targets, temperature=1.0):
    logsoftmax = F.log_softmax(pred / temperature, dim=1)
    return torch.mean(torch.sum(-soft_targets * logsoftmax, dim=1))

soft_targets = F.softmax(torch.randn(32, 10), dim=1)
soft_ce_loss = soft_cross_entropy(logits, soft_targets, temperature=3.0)
print(f"Soft cross entropy: {soft_ce_loss.item():.4f}")

print("\n=== Loss Function Best Practices ===")

print("Loss Function Guidelines:")
print("1. Classification: CrossEntropyLoss (multi-class), BCEWithLogitsLoss (binary)")
print("2. Regression: MSELoss (smooth), L1Loss (robust), SmoothL1Loss (hybrid)")
print("3. Imbalanced data: Use class weights or focal loss")
print("4. Segmentation: Combined CE + Dice loss")
print("5. Contrastive learning: InfoNCE, Supervised Contrastive")
print("6. Knowledge distillation: Soft cross entropy with temperature")
print("7. Multi-task: Weighted combination with proper scaling")
print("8. Always use reduction='none' for custom weighting")

print("\nCommon Pitfalls:")
print("- Don't apply softmax before CrossEntropyLoss")
print("- Use BCEWithLogitsLoss instead of BCE + Sigmoid")
print("- Be careful with loss scaling in multi-task learning")
print("- Consider label smoothing for better generalization")
print("- Use appropriate loss for your task (ranking, similarity, etc.)")

print("\n=== Loss Functions Complete ===") 