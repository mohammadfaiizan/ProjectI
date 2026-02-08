#!/usr/bin/env python3
"""PyTorch Loss Layers Syntax - Loss functions as layers"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== Loss Functions as Layers Overview ===")

print("Loss layers provide:")
print("1. Modular loss computation")
print("2. Integration with nn.Module")
print("3. Custom loss implementations")
print("4. Loss combination strategies")
print("5. Differentiable loss functions")

print("\n=== Classification Loss Functions ===")

# CrossEntropyLoss
ce_loss = nn.CrossEntropyLoss()
logits = torch.randn(4, 10)  # (batch_size, num_classes)
targets = torch.randint(0, 10, (4,))  # Class indices

ce_loss_value = ce_loss(logits, targets)
print(f"CrossEntropyLoss: {ce_loss_value.item():.4f}")

# NLLLoss (Negative Log Likelihood)
nll_loss = nn.NLLLoss()
log_probs = F.log_softmax(logits, dim=1)
nll_loss_value = nll_loss(log_probs, targets)
print(f"NLLLoss: {nll_loss_value.item():.4f}")

# Binary Cross Entropy
bce_loss = nn.BCELoss()
binary_probs = torch.sigmoid(torch.randn(8, 1))
binary_targets = torch.randint(0, 2, (8, 1)).float()
bce_loss_value = bce_loss(binary_probs, binary_targets)
print(f"BCELoss: {bce_loss_value.item():.4f}")

# Binary Cross Entropy with Logits
bce_logits_loss = nn.BCEWithLogitsLoss()
binary_logits = torch.randn(8, 1)
bce_logits_value = bce_logits_loss(binary_logits, binary_targets)
print(f"BCEWithLogitsLoss: {bce_logits_value.item():.4f}")

print("\n=== Regression Loss Functions ===")

# Mean Squared Error
mse_loss = nn.MSELoss()
predictions = torch.randn(10, 1)
regression_targets = torch.randn(10, 1)
mse_value = mse_loss(predictions, regression_targets)
print(f"MSELoss: {mse_value.item():.4f}")

# Mean Absolute Error
mae_loss = nn.L1Loss()
mae_value = mae_loss(predictions, regression_targets)
print(f"L1Loss (MAE): {mae_value.item():.4f}")

# Smooth L1 Loss (Huber Loss)
smooth_l1_loss = nn.SmoothL1Loss()
smooth_l1_value = smooth_l1_loss(predictions, regression_targets)
print(f"SmoothL1Loss: {smooth_l1_value.item():.4f}")

# Huber Loss with different delta
huber_loss = nn.HuberLoss(delta=1.0)
huber_value = huber_loss(predictions, regression_targets)
print(f"HuberLoss: {huber_value.item():.4f}")

print("\n=== Loss Reduction Options ===")

# Different reduction modes
reduction_modes = ['mean', 'sum', 'none']

for mode in reduction_modes:
    loss_fn = nn.CrossEntropyLoss(reduction=mode)
    loss_value = loss_fn(logits, targets)
    print(f"CrossEntropyLoss (reduction={mode}): {loss_value}")

print("\n=== Advanced Loss Functions ===")

# KL Divergence Loss
kl_loss = nn.KLDivLoss(reduction='batchmean')
input_probs = F.log_softmax(torch.randn(3, 5), dim=1)
target_probs = F.softmax(torch.randn(3, 5), dim=1)
kl_value = kl_loss(input_probs, target_probs)
print(f"KLDivLoss: {kl_value.item():.4f}")

# Cosine Embedding Loss
cosine_loss = nn.CosineEmbeddingLoss()
input1 = torch.randn(5, 10)
input2 = torch.randn(5, 10)
similarity_targets = torch.randint(-1, 2, (5,)).float()  # -1 or 1
cosine_value = cosine_loss(input1, input2, similarity_targets)
print(f"CosineEmbeddingLoss: {cosine_value.item():.4f}")

# Triplet Margin Loss
triplet_loss = nn.TripletMarginLoss(margin=1.0)
anchor = torch.randn(5, 10)
positive = torch.randn(5, 10)
negative = torch.randn(5, 10)
triplet_value = triplet_loss(anchor, positive, negative)
print(f"TripletMarginLoss: {triplet_value.item():.4f}")

print("\n=== Custom Loss Functions ===")

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
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

class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate Dice coefficient
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class ContrastiveLoss(nn.Module):
    """Contrastive Loss for siamese networks"""
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                         label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# Test custom loss functions
focal_loss = FocalLoss(alpha=0.25, gamma=2)
dice_loss = DiceLoss()
contrastive_loss = ContrastiveLoss(margin=2.0)

# Test Focal Loss
focal_value = focal_loss(logits, targets)
print(f"Focal Loss: {focal_value.item():.4f}")

# Test Dice Loss (for segmentation)
seg_preds = torch.sigmoid(torch.randn(2, 1, 32, 32))
seg_targets = torch.randint(0, 2, (2, 1, 32, 32)).float()
dice_value = dice_loss(seg_preds, seg_targets)
print(f"Dice Loss: {dice_value.item():.4f}")

# Test Contrastive Loss
feat1 = torch.randn(8, 128)
feat2 = torch.randn(8, 128)
pair_labels = torch.randint(0, 2, (8,)).float()
contrastive_value = contrastive_loss(feat1, feat2, pair_labels)
print(f"Contrastive Loss: {contrastive_value.item():.4f}")

print("\n=== Loss Combination Strategies ===")

class WeightedLossCombination(nn.Module):
    """Combine multiple losses with weights"""
    def __init__(self, losses, weights):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights
    
    def forward(self, *args, **kwargs):
        total_loss = 0
        individual_losses = {}
        
        for i, (loss_fn, weight) in enumerate(zip(self.losses, self.weights)):
            loss_value = loss_fn(*args, **kwargs)
            total_loss += weight * loss_value
            individual_losses[f'loss_{i}'] = loss_value.item()
        
        return total_loss, individual_losses

class AdaptiveLossWeighting(nn.Module):
    """Adaptive loss weighting with learnable parameters"""
    def __init__(self, losses, initial_weights=None):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        
        if initial_weights is None:
            initial_weights = [1.0] * len(losses)
        
        # Learnable loss weights (log-space for numerical stability)
        self.log_weights = nn.Parameter(torch.log(torch.tensor(initial_weights, dtype=torch.float32)))
    
    def forward(self, *args, **kwargs):
        weights = torch.exp(self.log_weights)
        total_loss = 0
        individual_losses = {}
        
        for i, (loss_fn, weight) in enumerate(zip(self.losses, weights)):
            loss_value = loss_fn(*args, **kwargs)
            total_loss += weight * loss_value
            individual_losses[f'loss_{i}'] = loss_value.item()
            individual_losses[f'weight_{i}'] = weight.item()
        
        return total_loss, individual_losses

# Test loss combination
loss_list = [nn.MSELoss(), nn.L1Loss()]
weights = [0.7, 0.3]

weighted_loss = WeightedLossCombination(loss_list, weights)
adaptive_loss = AdaptiveLossWeighting(loss_list)

pred = torch.randn(5, 1)
target = torch.randn(5, 1)

combined_value, individual = weighted_loss(pred, target)
adaptive_value, adaptive_individual = adaptive_loss(pred, target)

print(f"Weighted Loss: {combined_value.item():.4f}")
print(f"Individual losses: {individual}")
print(f"Adaptive Loss: {adaptive_value.item():.4f}")
print(f"Adaptive weights: {adaptive_individual}")

print("\n=== Multi-task Loss Functions ===")

class MultiTaskLoss(nn.Module):
    """Multi-task loss with uncertainty weighting"""
    def __init__(self, num_tasks, initial_log_vars=None):
        super().__init__()
        
        if initial_log_vars is None:
            initial_log_vars = [0.0] * num_tasks
        
        # Learnable uncertainty parameters
        self.log_vars = nn.Parameter(torch.tensor(initial_log_vars, dtype=torch.float32))
    
    def forward(self, losses):
        """
        losses: list of individual task losses
        """
        total_loss = 0
        precision_weights = []
        
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]
            total_loss += weighted_loss
            precision_weights.append(precision.item())
        
        return total_loss, precision_weights

# Test multi-task loss
task_losses = [
    torch.tensor(0.5),  # Classification loss
    torch.tensor(1.2),  # Regression loss
    torch.tensor(0.8)   # Auxiliary loss
]

multitask_loss = MultiTaskLoss(num_tasks=3)
mt_loss_value, precisions = multitask_loss(task_losses)

print(f"Multi-task Loss: {mt_loss_value.item():.4f}")
print(f"Task precisions: {precisions}")

print("\n=== Loss Functions for Different Tasks ===")

class PerceptualLoss(nn.Module):
    """Perceptual loss using pretrained features"""
    def __init__(self, feature_layers=None):
        super().__init__()
        # Simplified - would use actual pretrained network
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        self.mse_loss = nn.MSELoss()
    
    def forward(self, input_img, target_img):
        input_features = self.feature_extractor(input_img)
        target_features = self.feature_extractor(target_img)
        return self.mse_loss(input_features, target_features)

class SSIMLoss(nn.Module):
    """Structural Similarity Index Loss"""
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
    
    def forward(self, img1, img2):
        # Simplified SSIM computation
        mu1 = F.avg_pool2d(img1, self.window_size, stride=1, padding=self.window_size//2)
        mu2 = F.avg_pool2d(img2, self.window_size, stride=1, padding=self.window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 * img1, self.window_size, stride=1, padding=self.window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, self.window_size, stride=1, padding=self.window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, self.window_size, stride=1, padding=self.window_size//2) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)

# Test specialized losses
perceptual_loss = PerceptualLoss()
ssim_loss = SSIMLoss()

img1 = torch.randn(2, 3, 64, 64)
img2 = torch.randn(2, 3, 64, 64)

perceptual_value = perceptual_loss(img1, img2)
ssim_value = ssim_loss(img1, img2)

print(f"Perceptual Loss: {perceptual_value.item():.4f}")
print(f"SSIM Loss: {ssim_value.item():.4f}")

print("\n=== Regularization as Loss Terms ===")

class L1RegularizationLoss(nn.Module):
    """L1 regularization as a loss term"""
    def __init__(self, weight=1e-4):
        super().__init__()
        self.weight = weight
    
    def forward(self, model):
        l1_loss = 0
        for param in model.parameters():
            l1_loss += torch.norm(param, 1)
        return self.weight * l1_loss

class L2RegularizationLoss(nn.Module):
    """L2 regularization as a loss term"""
    def __init__(self, weight=1e-4):
        super().__init__()
        self.weight = weight
    
    def forward(self, model):
        l2_loss = 0
        for param in model.parameters():
            l2_loss += torch.norm(param, 2) ** 2
        return self.weight * l2_loss

class OrthogonalRegularizationLoss(nn.Module):
    """Orthogonal regularization for weight matrices"""
    def __init__(self, weight=1e-4):
        super().__init__()
        self.weight = weight
    
    def forward(self, weight_matrix):
        w = weight_matrix.view(weight_matrix.size(0), -1)
        sym = torch.mm(w, w.t())
        sym -= torch.eye(sym.size(0), device=sym.device)
        return self.weight * sym.abs().sum()

print("\n=== Loss Scheduling and Annealing ===")

class AnnealedLoss(nn.Module):
    """Loss with annealing schedule"""
    def __init__(self, base_loss, annealing_schedule='linear', total_steps=1000):
        super().__init__()
        self.base_loss = base_loss
        self.annealing_schedule = annealing_schedule
        self.total_steps = total_steps
        self.current_step = 0
    
    def forward(self, *args, **kwargs):
        base_loss_value = self.base_loss(*args, **kwargs)
        
        # Calculate annealing factor
        progress = min(self.current_step / self.total_steps, 1.0)
        
        if self.annealing_schedule == 'linear':
            factor = progress
        elif self.annealing_schedule == 'cosine':
            factor = 0.5 * (1 - torch.cos(torch.tensor(progress * 3.14159)))
        elif self.annealing_schedule == 'exponential':
            factor = 1 - torch.exp(torch.tensor(-5 * progress))
        else:
            factor = 1.0
        
        self.current_step += 1
        return factor * base_loss_value

print("\n=== Functional vs Module Loss Usage ===")

# Functional interface
functional_ce = F.cross_entropy(logits, targets)
functional_mse = F.mse_loss(predictions, regression_targets)

print(f"Functional CrossEntropy: {functional_ce.item():.4f}")
print(f"Functional MSE: {functional_mse.item():.4f}")

# Module interface with state
class StatefulLoss(nn.Module):
    """Loss function that maintains state"""
    def __init__(self):
        super().__init__()
        self.register_buffer('running_loss', torch.tensor(0.0))
        self.register_buffer('num_updates', torch.tensor(0))
        self.momentum = 0.9
    
    def forward(self, predictions, targets):
        current_loss = F.mse_loss(predictions, targets)
        
        # Update running average
        if self.num_updates == 0:
            self.running_loss.copy_(current_loss)
        else:
            self.running_loss.mul_(self.momentum).add_(current_loss, alpha=1-self.momentum)
        
        self.num_updates += 1
        
        return current_loss
    
    def get_running_loss(self):
        return self.running_loss.item()

stateful_loss = StatefulLoss()
for i in range(5):
    pred = torch.randn(4, 1)
    target = torch.randn(4, 1)
    loss_val = stateful_loss(pred, target)
    print(f"Step {i+1}: Loss = {loss_val.item():.4f}, Running = {stateful_loss.get_running_loss():.4f}")

print("\n=== Loss Function Best Practices ===")

print("Loss Function Guidelines:")
print("1. Choose appropriate loss for your task type")
print("2. Consider class imbalance (use Focal Loss, class weights)")
print("3. Use reduction='none' for custom aggregation")
print("4. Combine losses carefully with proper weighting")
print("5. Monitor individual loss components in multi-task")
print("6. Use numerical stable implementations")
print("7. Consider loss annealing for training stability")

print("\nTask-Specific Recommendations:")
print("- Classification: CrossEntropyLoss, Focal Loss")
print("- Regression: MSELoss, L1Loss, HuberLoss")
print("- Segmentation: DiceLoss, IoULoss, CrossEntropyLoss")
print("- Object Detection: Focal Loss, IoULoss, SmoothL1Loss")
print("- GANs: BCELoss, Wasserstein Loss, LSGANLoss")
print("- Metric Learning: TripletLoss, ContrastiveLoss")

print("\nImplementation Tips:")
print("- Use LogSoftmax + NLLLoss instead of manual CrossEntropy")
print("- BCEWithLogitsLoss is more stable than Sigmoid + BCELoss")
print("- Clamp extreme values to prevent numerical issues")
print("- Use appropriate reduction for your use case")
print("- Consider label smoothing for regularization")
print("- Profile loss computation for performance")

print("\nCommon Pitfalls:")
print("- Wrong tensor shapes for loss computation")
print("- Forgetting to handle class imbalance")
print("- Not considering numerical stability")
print("- Inappropriate loss weighting in multi-task")
print("- Using wrong reduction mode")
print("- Not monitoring individual loss components")

print("\n=== Loss Layers Complete ===") 