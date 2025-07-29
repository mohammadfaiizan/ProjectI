#!/usr/bin/env python3
"""PyTorch Custom Loss Functions - Creating custom loss functions"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=== Custom Loss Functions Overview ===")

print("Custom loss patterns:")
print("1. Basic custom loss classes")
print("2. Function-based custom losses")
print("3. Differentiable custom operations")
print("4. Composite and weighted losses")
print("5. Domain-specific losses")
print("6. Learnable loss parameters")
print("7. Dynamic loss functions")
print("8. Multi-objective losses")

print("\n=== Basic Custom Loss Classes ===")

class BasicCustomLoss(nn.Module):
    """Template for basic custom loss function"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        # Implement your custom loss logic here
        loss = (predictions - targets) ** 2  # Example: MSE-like
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

class WeightedMSELoss(nn.Module):
    """MSE loss with per-sample weights"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, predictions, targets, weights=None):
        loss = (predictions - targets) ** 2
        
        if weights is not None:
            loss = loss * weights
        
        if self.reduction == 'mean':
            if weights is not None:
                return loss.sum() / weights.sum()
            else:
                return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class AdaptiveLoss(nn.Module):
    """Loss that adapts based on prediction confidence"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        # Basic loss
        mse_loss = (predictions - targets) ** 2
        
        # Confidence-based weighting
        prediction_confidence = torch.sigmoid(torch.abs(predictions))
        adaptive_weight = 1.0 + (1.0 - prediction_confidence)
        
        loss = mse_loss * adaptive_weight
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Test basic custom losses
predictions = torch.randn(10, 5)
targets = torch.randn(10, 5)
weights = torch.rand(10, 5)

basic_loss = BasicCustomLoss()
weighted_mse = WeightedMSELoss()
adaptive_loss = AdaptiveLoss()

basic_result = basic_loss(predictions, targets)
weighted_result = weighted_mse(predictions, targets, weights)
adaptive_result = adaptive_loss(predictions, targets)

print(f"Basic Custom Loss: {basic_result.item():.6f}")
print(f"Weighted MSE Loss: {weighted_result.item():.6f}")
print(f"Adaptive Loss: {adaptive_result.item():.6f}")

print("\n=== Function-Based Custom Losses ===")

def custom_mae_loss(predictions, targets, reduction='mean'):
    """Function-based custom MAE loss"""
    loss = torch.abs(predictions - targets)
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def wing_loss(predictions, targets, omega=10.0, epsilon=2.0, reduction='mean'):
    """Wing Loss for robust regression"""
    diff = torch.abs(predictions - targets)
    
    c = omega - omega * math.log(1 + omega / epsilon)
    
    loss = torch.where(
        diff < omega,
        omega * torch.log(1 + diff / epsilon),
        diff - c
    )
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def focal_regression_loss(predictions, targets, alpha=2.0, gamma=1.0, reduction='mean'):
    """Focal loss adapted for regression"""
    diff = torch.abs(predictions - targets)
    
    # Modulating factor
    modulating_factor = torch.pow(diff, gamma)
    
    # Base loss (can be MAE, MSE, etc.)
    base_loss = alpha * diff
    
    loss = modulating_factor * base_loss
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

# Test function-based losses
func_mae = custom_mae_loss(predictions, targets)
wing_result = wing_loss(predictions, targets, omega=5.0, epsilon=1.0)
focal_reg = focal_regression_loss(predictions, targets, alpha=1.0, gamma=2.0)

print(f"Function MAE Loss: {func_mae.item():.6f}")
print(f"Wing Loss: {wing_result.item():.6f}")
print(f"Focal Regression Loss: {focal_reg.item():.6f}")

print("\n=== Differentiable Custom Operations ===")

class DifferentiableThreshold(torch.autograd.Function):
    """Custom differentiable threshold operation"""
    
    @staticmethod
    def forward(ctx, input_tensor, threshold):
        ctx.save_for_backward(input_tensor)
        ctx.threshold = threshold
        return (input_tensor > threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        threshold = ctx.threshold
        
        # Straight-through estimator
        grad_input = grad_output.clone()
        grad_input[input_tensor <= threshold] = 0
        
        return grad_input, None

class CustomSigmoidLoss(nn.Module):
    """Loss using custom differentiable operations"""
    def __init__(self, threshold=0.5, reduction='mean'):
        super().__init__()
        self.threshold = threshold
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        # Apply custom threshold operation
        thresholded = DifferentiableThreshold.apply(predictions, self.threshold)
        
        # Compute loss
        loss = F.binary_cross_entropy(thresholded, targets)
        
        return loss

# Test differentiable operations
binary_preds = torch.randn(10, 1)
binary_targets = torch.randint(0, 2, (10, 1)).float()

custom_sigmoid_loss = CustomSigmoidLoss(threshold=0.0)
custom_sig_result = custom_sigmoid_loss(torch.sigmoid(binary_preds), binary_targets)

print(f"Custom Sigmoid Loss: {custom_sig_result.item():.6f}")

print("\n=== Composite and Weighted Losses ===")

class MultiComponentLoss(nn.Module):
    """Combines multiple loss components"""
    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            self.weights = {'mse': 1.0, 'mae': 0.5, 'smooth_l1': 0.3}
        else:
            self.weights = weights
        
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
    
    def forward(self, predictions, targets):
        mse = self.mse_loss(predictions, targets)
        mae = self.mae_loss(predictions, targets)
        smooth_l1 = self.smooth_l1_loss(predictions, targets)
        
        total_loss = (self.weights['mse'] * mse + 
                     self.weights['mae'] * mae + 
                     self.weights['smooth_l1'] * smooth_l1)
        
        return total_loss, {
            'mse': mse,
            'mae': mae,
            'smooth_l1': smooth_l1,
            'total': total_loss
        }

class BalancedLoss(nn.Module):
    """Automatically balances multiple loss components"""
    def __init__(self, num_losses, alpha=0.1):
        super().__init__()
        self.num_losses = num_losses
        self.alpha = alpha
        
        # Learnable balancing weights
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
    
    def forward(self, loss_components):
        # loss_components: list of individual losses
        balanced_loss = 0
        
        for i, loss in enumerate(loss_components):
            precision = torch.exp(-self.log_vars[i])
            balanced_loss += precision * loss + self.log_vars[i]
        
        return balanced_loss

# Test composite losses
multi_comp_loss = MultiComponentLoss()
balanced_loss = BalancedLoss(num_losses=3)

multi_result, components = multi_comp_loss(predictions, targets)
loss_list = [components['mse'], components['mae'], components['smooth_l1']]
balanced_result = balanced_loss(loss_list)

print(f"Multi-component Loss: {multi_result.item():.6f}")
print("Components:")
for name, value in components.items():
    if name != 'total':
        print(f"  {name}: {value.item():.6f}")

print(f"Balanced Loss: {balanced_result.item():.6f}")
print(f"Learned weights: {balanced_loss.log_vars.data}")

print("\n=== Domain-Specific Losses ===")

class PerceptualLoss(nn.Module):
    """Perceptual loss using feature differences"""
    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions, targets):
        # Extract features
        pred_features = self.feature_extractor(predictions)
        target_features = self.feature_extractor(targets)
        
        # Compute feature-space loss
        loss = self.mse_loss(pred_features, target_features)
        
        return loss

class TemporalConsistencyLoss(nn.Module):
    """Loss for temporal consistency in sequences"""
    def __init__(self, alpha=1.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, sequence_predictions, sequence_targets):
        # sequence_predictions: [batch, time, features]
        # sequence_targets: [batch, time, features]
        
        # Frame-wise loss
        frame_loss = F.mse_loss(sequence_predictions, sequence_targets, reduction='none')
        frame_loss = frame_loss.mean(dim=-1)  # Average over features
        
        # Temporal consistency loss
        pred_diff = sequence_predictions[:, 1:] - sequence_predictions[:, :-1]
        target_diff = sequence_targets[:, 1:] - sequence_targets[:, :-1]
        temporal_loss = F.mse_loss(pred_diff, target_diff, reduction='none')
        temporal_loss = temporal_loss.mean(dim=-1)  # Average over features
        
        # Combine losses
        total_frame_loss = frame_loss.mean(dim=1)  # Average over time
        total_temporal_loss = temporal_loss.mean(dim=1)  # Average over time
        
        total_loss = total_frame_loss + self.alpha * total_temporal_loss
        
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss

class ContrastiveDivergenceLoss(nn.Module):
    """Contrastive divergence loss for energy-based models"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pos_energy, neg_energy):
        # Positive examples should have low energy
        # Negative examples should have high energy
        
        loss = pos_energy - neg_energy
        loss = F.softplus(loss)  # Smooth approximation to max(0, .)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Test domain-specific losses
# Temporal sequence data
seq_predictions = torch.randn(5, 10, 8)  # batch=5, time=10, features=8
seq_targets = torch.randn(5, 10, 8)

temporal_loss = TemporalConsistencyLoss(alpha=0.5)
temporal_result = temporal_loss(seq_predictions, seq_targets)

# Contrastive divergence
pos_energy = torch.randn(10)
neg_energy = torch.randn(10)

cd_loss = ContrastiveDivergenceLoss()
cd_result = cd_loss(pos_energy, neg_energy)

print(f"Temporal Consistency Loss: {temporal_result.item():.6f}")
print(f"Contrastive Divergence Loss: {cd_result.item():.6f}")

print("\n=== Learnable Loss Parameters ===")

class LearnableLoss(nn.Module):
    """Loss with learnable parameters"""
    def __init__(self, init_alpha=1.0, init_beta=1.0):
        super().__init__()
        # Learnable loss parameters
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.beta = nn.Parameter(torch.tensor(init_beta))
        self.gamma = nn.Parameter(torch.tensor(2.0))
    
    def forward(self, predictions, targets):
        diff = torch.abs(predictions - targets)
        
        # Learnable loss function
        loss = self.alpha * torch.pow(diff, self.gamma) + self.beta * diff
        
        return loss.mean()

class AdaptiveMarginLoss(nn.Module):
    """Margin loss with learnable margin"""
    def __init__(self, init_margin=1.0):
        super().__init__()
        self.margin = nn.Parameter(torch.tensor(init_margin))
    
    def forward(self, positive_scores, negative_scores):
        loss = F.relu(self.margin - positive_scores + negative_scores)
        return loss.mean()

class TemperatureScaledCrossEntropy(nn.Module):
    """Cross-entropy with learnable temperature"""
    def __init__(self, init_temperature=1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temperature))
    
    def forward(self, logits, targets):
        scaled_logits = logits / torch.clamp(self.temperature, min=0.1)
        return F.cross_entropy(scaled_logits, targets)

# Test learnable losses
learnable_loss = LearnableLoss(init_alpha=0.5, init_beta=0.3)
adaptive_margin = AdaptiveMarginLoss(init_margin=0.5)
temp_scaled_ce = TemperatureScaledCrossEntropy(init_temperature=2.0)

learnable_result = learnable_loss(predictions, targets)

# Margin loss test data
pos_scores = torch.randn(10)
neg_scores = torch.randn(10)
margin_result = adaptive_margin(pos_scores, neg_scores)

# Temperature scaled CE test data
logits = torch.randn(10, 5)
ce_targets = torch.randint(0, 5, (10,))
temp_ce_result = temp_scaled_ce(logits, ce_targets)

print(f"Learnable Loss: {learnable_result.item():.6f}")
print(f"  Learned alpha: {learnable_loss.alpha.item():.6f}")
print(f"  Learned beta: {learnable_loss.beta.item():.6f}")
print(f"  Learned gamma: {learnable_loss.gamma.item():.6f}")

print(f"Adaptive Margin Loss: {margin_result.item():.6f}")
print(f"  Learned margin: {adaptive_margin.margin.item():.6f}")

print(f"Temperature Scaled CE: {temp_ce_result.item():.6f}")
print(f"  Learned temperature: {temp_scaled_ce.temperature.item():.6f}")

print("\n=== Dynamic Loss Functions ===")

class CurriculumLoss(nn.Module):
    """Loss that changes difficulty over time"""
    def __init__(self):
        super().__init__()
        self.current_epoch = 0
        self.total_epochs = 100
    
    def set_epoch(self, epoch, total_epochs=100):
        self.current_epoch = epoch
        self.total_epochs = total_epochs
    
    def forward(self, predictions, targets):
        # Progress from 0 to 1
        progress = min(self.current_epoch / self.total_epochs, 1.0)
        
        # Start with easier loss (MAE), gradually move to harder loss (MSE)
        mae_loss = F.l1_loss(predictions, targets)
        mse_loss = F.mse_loss(predictions, targets)
        
        # Interpolate between losses
        loss = (1 - progress) * mae_loss + progress * mse_loss
        
        return loss

class SelfPacedLoss(nn.Module):
    """Self-paced learning loss"""
    def __init__(self, lambda_param=1.0):
        super().__init__()
        self.lambda_param = lambda_param
    
    def forward(self, predictions, targets, age=None):
        # Compute base loss for each sample
        base_loss = F.mse_loss(predictions, targets, reduction='none')
        base_loss = base_loss.mean(dim=1)  # Per-sample loss
        
        # Self-paced weighting (smaller losses get higher weights)
        weights = torch.where(base_loss < 1.0 / self.lambda_param,
                             torch.ones_like(base_loss),
                             torch.zeros_like(base_loss))
        
        # Weighted loss
        weighted_loss = weights * base_loss
        
        return weighted_loss.mean()

class UncertaintyWeightedLoss(nn.Module):
    """Loss weighted by model uncertainty"""
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets, uncertainty):
        # Base loss
        base_loss = F.mse_loss(predictions, targets, reduction='none')
        
        # Weight by inverse uncertainty (higher uncertainty = lower weight)
        weights = 1.0 / (uncertainty + 1e-8)
        weights = weights / weights.mean()  # Normalize
        
        weighted_loss = weights * base_loss
        
        return weighted_loss.mean()

# Test dynamic losses
curriculum_loss = CurriculumLoss()
self_paced_loss = SelfPacedLoss(lambda_param=0.5)
uncertainty_loss = UncertaintyWeightedLoss()

# Test at different epochs
curriculum_loss.set_epoch(0, 100)
curriculum_early = curriculum_loss(predictions, targets)

curriculum_loss.set_epoch(50, 100)
curriculum_mid = curriculum_loss(predictions, targets)

curriculum_loss.set_epoch(100, 100)
curriculum_late = curriculum_loss(predictions, targets)

# Self-paced loss
self_paced_result = self_paced_loss(predictions, targets)

# Uncertainty weighted loss
uncertainty = torch.rand(10, 5)  # Simulated uncertainty
uncertainty_result = uncertainty_loss(predictions, targets, uncertainty)

print(f"Curriculum Loss:")
print(f"  Epoch 0: {curriculum_early.item():.6f}")
print(f"  Epoch 50: {curriculum_mid.item():.6f}")
print(f"  Epoch 100: {curriculum_late.item():.6f}")

print(f"Self-Paced Loss: {self_paced_result.item():.6f}")
print(f"Uncertainty Weighted Loss: {uncertainty_result.item():.6f}")

print("\n=== Multi-Objective Losses ===")

class MultiObjectiveLoss(nn.Module):
    """Loss for multi-objective optimization"""
    def __init__(self, objectives, method='weighted_sum'):
        super().__init__()
        self.objectives = objectives  # List of loss functions
        self.method = method
        
        if method == 'weighted_sum':
            self.weights = nn.Parameter(torch.ones(len(objectives)))
        elif method == 'adaptive':
            self.log_vars = nn.Parameter(torch.zeros(len(objectives)))
    
    def forward(self, predictions, targets):
        losses = []
        
        # Compute individual objective losses
        for i, objective in enumerate(self.objectives):
            if isinstance(predictions, (list, tuple)):
                loss = objective(predictions[i], targets[i])
            else:
                loss = objective(predictions, targets)
            losses.append(loss)
        
        if self.method == 'weighted_sum':
            # Weighted sum approach
            weights = F.softmax(self.weights, dim=0)
            total_loss = sum(w * l for w, l in zip(weights, losses))
        
        elif self.method == 'adaptive':
            # Adaptive weighting based on uncertainty
            total_loss = 0
            for i, loss in enumerate(losses):
                precision = torch.exp(-self.log_vars[i])
                total_loss += precision * loss + self.log_vars[i]
        
        else:
            # Simple average
            total_loss = sum(losses) / len(losses)
        
        return total_loss, losses

class ParetoLoss(nn.Module):
    """Pareto-optimal multi-objective loss"""
    def __init__(self, objectives, alpha=0.5):
        super().__init__()
        self.objectives = objectives
        self.alpha = alpha  # Trade-off parameter
    
    def forward(self, predictions, targets):
        losses = []
        
        for objective in self.objectives:
            loss = objective(predictions, targets)
            losses.append(loss)
        
        # Normalize losses
        normalized_losses = []
        for loss in losses:
            normalized_losses.append(loss / (loss.detach() + 1e-8))
        
        # Pareto weighting
        if len(normalized_losses) == 2:
            total_loss = (self.alpha * normalized_losses[0] + 
                         (1 - self.alpha) * normalized_losses[1])
        else:
            # For more than 2 objectives, use equal weighting
            total_loss = sum(normalized_losses) / len(normalized_losses)
        
        return total_loss, losses

# Test multi-objective losses
objectives = [nn.MSELoss(), nn.L1Loss()]
multi_obj_loss = MultiObjectiveLoss(objectives, method='adaptive')
pareto_loss = ParetoLoss(objectives, alpha=0.7)

multi_obj_result, multi_obj_components = multi_obj_loss(predictions, targets)
pareto_result, pareto_components = pareto_loss(predictions, targets)

print(f"Multi-Objective Loss: {multi_obj_result.item():.6f}")
print(f"  Component losses: {[l.item() for l in multi_obj_components]}")
print(f"  Learned weights: {multi_obj_loss.log_vars.data}")

print(f"Pareto Loss: {pareto_result.item():.6f}")
print(f"  Component losses: {[l.item() for l in pareto_components]}")

print("\n=== Custom Loss Best Practices ===")

print("Design Guidelines:")
print("1. Ensure differentiability for gradient-based optimization")
print("2. Handle edge cases and numerical stability")
print("3. Implement proper reduction strategies")
print("4. Consider computational efficiency")
print("5. Validate loss behavior with simple test cases")
print("6. Use appropriate parameter initialization")
print("7. Document loss function motivation and usage")

print("\nImplementation Tips:")
print("1. Inherit from nn.Module for learnable parameters")
print("2. Use torch.autograd.Function for custom backward passes")
print("3. Implement reduction='none' for flexibility")
print("4. Add numerical stability with small epsilon values")
print("5. Test gradients with torch.autograd.gradcheck")
print("6. Consider loss scaling for mixed precision training")

print("\nCommon Pitfalls:")
print("1. Non-differentiable operations breaking gradients")
print("2. Numerical instability with extreme values")
print("3. Incorrect gradient computation in custom functions")
print("4. Poor scaling between different loss components")
print("5. Memory leaks from improper tensor handling")
print("6. Loss not decreasing due to implementation errors")

print("\nDebugging Custom Losses:")
print("1. Test with known analytical solutions")
print("2. Verify gradients with finite differences")
print("3. Check loss values for sanity")
print("4. Monitor individual loss components")
print("5. Validate on simple synthetic data")
print("6. Compare with existing implementations when possible")

print("\n=== Custom Loss Functions Complete ===")

# Memory cleanup
del predictions, targets, weights
del seq_predictions, seq_targets
del pos_energy, neg_energy, uncertainty