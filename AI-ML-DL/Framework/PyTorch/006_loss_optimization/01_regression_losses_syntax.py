#!/usr/bin/env python3
"""PyTorch Regression Loss Functions - MSE, MAE, Huber, SmoothL1Loss syntax"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

print("=== Regression Loss Functions Overview ===")

print("Common regression losses:")
print("1. Mean Squared Error (MSE/L2 Loss)")
print("2. Mean Absolute Error (MAE/L1 Loss)")
print("3. Smooth L1 Loss (Huber-like)")
print("4. Huber Loss")
print("5. Log-Cosh Loss")
print("6. Quantile Loss")
print("7. RMSE Loss")
print("8. Relative losses")

print("\n=== Mean Squared Error (MSE) ===")

# MSE Loss - most common regression loss
mse_loss = nn.MSELoss()
mse_loss_no_reduction = nn.MSELoss(reduction='none')
mse_loss_sum = nn.MSELoss(reduction='sum')

# Sample data
predictions = torch.randn(10, 5)
targets = torch.randn(10, 5)

# Different MSE computations
mse_mean = mse_loss(predictions, targets)
mse_none = mse_loss_no_reduction(predictions, targets)
mse_sum = mse_loss_sum(predictions, targets)

print(f"MSE (mean reduction): {mse_mean.item():.6f}")
print(f"MSE (no reduction) shape: {mse_none.shape}")
print(f"MSE (sum reduction): {mse_sum.item():.6f}")

# Manual MSE computation
mse_manual = torch.mean((predictions - targets) ** 2)
print(f"Manual MSE: {mse_manual.item():.6f}")
print(f"MSE matches manual: {torch.allclose(mse_mean, mse_manual)}")

# Functional API
mse_functional = F.mse_loss(predictions, targets)
print(f"Functional MSE: {mse_functional.item():.6f}")

print("\n=== Mean Absolute Error (MAE) ===")

# MAE Loss - robust to outliers
mae_loss = nn.L1Loss()
mae_loss_no_reduction = nn.L1Loss(reduction='none')
mae_loss_sum = nn.L1Loss(reduction='sum')

# MAE computations
mae_mean = mae_loss(predictions, targets)
mae_none = mae_loss_no_reduction(predictions, targets)
mae_sum = mae_loss_sum(predictions, targets)

print(f"MAE (mean reduction): {mae_mean.item():.6f}")
print(f"MAE (no reduction) shape: {mae_none.shape}")
print(f"MAE (sum reduction): {mae_sum.item():.6f}")

# Manual MAE computation
mae_manual = torch.mean(torch.abs(predictions - targets))
print(f"Manual MAE: {mae_manual.item():.6f}")
print(f"MAE matches manual: {torch.allclose(mae_mean, mae_manual)}")

# Functional API
mae_functional = F.l1_loss(predictions, targets)
print(f"Functional MAE: {mae_functional.item():.6f}")

print("\n=== Smooth L1 Loss ===")

# Smooth L1 Loss - combines MSE and MAE benefits
smooth_l1_loss = nn.SmoothL1Loss()
smooth_l1_loss_beta = nn.SmoothL1Loss(beta=2.0)  # Different beta parameter
smooth_l1_loss_no_reduction = nn.SmoothL1Loss(reduction='none')

# Smooth L1 computations
smooth_l1_mean = smooth_l1_loss(predictions, targets)
smooth_l1_beta = smooth_l1_loss_beta(predictions, targets)
smooth_l1_none = smooth_l1_loss_no_reduction(predictions, targets)

print(f"Smooth L1 (beta=1.0): {smooth_l1_mean.item():.6f}")
print(f"Smooth L1 (beta=2.0): {smooth_l1_beta.item():.6f}")
print(f"Smooth L1 (no reduction) shape: {smooth_l1_none.shape}")

# Manual Smooth L1 computation
def smooth_l1_manual(input_tensor, target, beta=1.0):
    diff = torch.abs(input_tensor - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return torch.mean(loss)

smooth_l1_manual_result = smooth_l1_manual(predictions, targets, beta=1.0)
print(f"Manual Smooth L1: {smooth_l1_manual_result.item():.6f}")

# Functional API
smooth_l1_functional = F.smooth_l1_loss(predictions, targets)
print(f"Functional Smooth L1: {smooth_l1_functional.item():.6f}")

print("\n=== Huber Loss ===")

# Huber Loss - robust regression loss
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0, reduction='mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
    
    def forward(self, input_tensor, target):
        diff = torch.abs(input_tensor - target)
        loss = torch.where(diff <= self.delta,
                          0.5 * diff ** 2,
                          self.delta * (diff - 0.5 * self.delta))
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

# Test Huber loss with different deltas
huber_loss_1 = HuberLoss(delta=1.0)
huber_loss_2 = HuberLoss(delta=2.0)
huber_loss_05 = HuberLoss(delta=0.5)

huber_1 = huber_loss_1(predictions, targets)
huber_2 = huber_loss_2(predictions, targets)
huber_05 = huber_loss_05(predictions, targets)

print(f"Huber Loss (delta=1.0): {huber_1.item():.6f}")
print(f"Huber Loss (delta=2.0): {huber_2.item():.6f}")
print(f"Huber Loss (delta=0.5): {huber_05.item():.6f}")

print("\n=== Advanced Regression Losses ===")

class LogCoshLoss(nn.Module):
    """Log-Cosh Loss - smooth approximation to MAE"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input_tensor, target):
        diff = input_tensor - target
        loss = torch.log(torch.cosh(diff))
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

class QuantileLoss(nn.Module):
    """Quantile Loss for quantile regression"""
    def __init__(self, quantile=0.5, reduction='mean'):
        super().__init__()
        self.quantile = quantile
        self.reduction = reduction
    
    def forward(self, input_tensor, target):
        diff = target - input_tensor
        loss = torch.where(diff >= 0,
                          self.quantile * diff,
                          (self.quantile - 1) * diff)
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

class RMSELoss(nn.Module):
    """Root Mean Squared Error Loss"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input_tensor, target):
        mse = torch.mean((input_tensor - target) ** 2, dim=-1)
        rmse = torch.sqrt(mse)
        
        if self.reduction == 'mean':
            return torch.mean(rmse)
        elif self.reduction == 'sum':
            return torch.sum(rmse)
        else:
            return rmse

# Test advanced losses
log_cosh_loss = LogCoshLoss()
quantile_loss_median = QuantileLoss(quantile=0.5)
quantile_loss_q75 = QuantileLoss(quantile=0.75)
rmse_loss = RMSELoss()

log_cosh = log_cosh_loss(predictions, targets)
quantile_median = quantile_loss_median(predictions, targets)
quantile_75 = quantile_loss_q75(predictions, targets)
rmse = rmse_loss(predictions, targets)

print(f"Log-Cosh Loss: {log_cosh.item():.6f}")
print(f"Quantile Loss (0.5): {quantile_median.item():.6f}")
print(f"Quantile Loss (0.75): {quantile_75.item():.6f}")
print(f"RMSE Loss: {rmse.item():.6f}")

print("\n=== Relative and Scaled Losses ===")

class MAPELoss(nn.Module):
    """Mean Absolute Percentage Error"""
    def __init__(self, epsilon=1e-8, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, input_tensor, target):
        # Avoid division by zero
        denominator = torch.abs(target) + self.epsilon
        loss = torch.abs((target - input_tensor) / denominator) * 100
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

class MSLELoss(nn.Module):
    """Mean Squared Logarithmic Error"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input_tensor, target):
        # Ensure positive values
        input_pos = torch.clamp(input_tensor, min=0)
        target_pos = torch.clamp(target, min=0)
        
        loss = (torch.log(input_pos + 1) - torch.log(target_pos + 1)) ** 2
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

class RelativeL2Loss(nn.Module):
    """Relative L2 Loss"""
    def __init__(self, epsilon=1e-8, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, input_tensor, target):
        numerator = torch.sum((input_tensor - target) ** 2, dim=-1)
        denominator = torch.sum(target ** 2, dim=-1) + self.epsilon
        loss = numerator / denominator
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

# Test relative losses with positive data
positive_predictions = torch.abs(predictions) + 0.1
positive_targets = torch.abs(targets) + 0.1

mape_loss = MAPELoss()
msle_loss = MSLELoss()
relative_l2_loss = RelativeL2Loss()

mape = mape_loss(positive_predictions, positive_targets)
msle = msle_loss(positive_predictions, positive_targets)
relative_l2 = relative_l2_loss(positive_predictions, positive_targets)

print(f"MAPE Loss: {mape.item():.6f}%")
print(f"MSLE Loss: {msle.item():.6f}")
print(f"Relative L2 Loss: {relative_l2.item():.6f}")

print("\n=== Robust Regression Losses ===")

class TukeyBiweightLoss(nn.Module):
    """Tukey Biweight Loss - very robust to outliers"""
    def __init__(self, c=4.685, reduction='mean'):
        super().__init__()
        self.c = c
        self.reduction = reduction
    
    def forward(self, input_tensor, target):
        diff = torch.abs(input_tensor - target)
        
        # Tukey biweight function
        condition = diff <= self.c
        loss = torch.where(condition,
                          (self.c ** 2 / 6) * (1 - (1 - (diff / self.c) ** 2) ** 3),
                          self.c ** 2 / 6)
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

class AdaptiveRobustLoss(nn.Module):
    """Adaptive Robust Loss that learns robustness parameter"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        # Learnable shape parameter (alpha)
        self.alpha = nn.Parameter(torch.tensor(2.0))
    
    def forward(self, input_tensor, target):
        diff = input_tensor - target
        
        # Adaptive robust loss function
        loss = (torch.abs(self.alpha - 2) / self.alpha) * \
               ((torch.abs(diff / self.alpha) + 1) ** self.alpha - 1)
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

# Test robust losses
tukey_loss = TukeyBiweightLoss()
adaptive_loss = AdaptiveRobustLoss()

tukey = tukey_loss(predictions, targets)
adaptive = adaptive_loss(predictions, targets)

print(f"Tukey Biweight Loss: {tukey.item():.6f}")
print(f"Adaptive Robust Loss: {adaptive.item():.6f}")
print(f"Learned alpha parameter: {adaptive_loss.alpha.item():.6f}")

print("\n=== Multi-Task Regression Losses ===")

class MultiTaskMSELoss(nn.Module):
    """Multi-task MSE with learned task weights"""
    def __init__(self, num_tasks, reduction='mean'):
        super().__init__()
        self.num_tasks = num_tasks
        self.reduction = reduction
        # Learnable task weights (log-variance)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, predictions, targets):
        # predictions and targets: [batch_size, num_tasks]
        losses = []
        
        for i in range(self.num_tasks):
            pred_i = predictions[:, i]
            target_i = targets[:, i]
            
            # Weighted loss based on learned uncertainty
            precision = torch.exp(-self.log_vars[i])
            loss_i = precision * F.mse_loss(pred_i, target_i, reduction='none') + self.log_vars[i]
            losses.append(loss_i)
        
        total_loss = torch.stack(losses, dim=1)
        
        if self.reduction == 'mean':
            return torch.mean(total_loss)
        elif self.reduction == 'sum':
            return torch.sum(total_loss)
        else:
            return total_loss

# Test multi-task loss
num_tasks = 3
batch_size = 10
multi_predictions = torch.randn(batch_size, num_tasks)
multi_targets = torch.randn(batch_size, num_tasks)

multi_task_loss = MultiTaskMSELoss(num_tasks)
multi_loss = multi_task_loss(multi_predictions, multi_targets)

print(f"Multi-task MSE Loss: {multi_loss.item():.6f}")
print(f"Learned log-variances: {multi_task_loss.log_vars.data}")

print("\n=== Loss Comparison and Visualization ===")

def compare_losses_with_outliers():
    """Compare different losses with outlier data"""
    # Create data with outliers
    clean_data = torch.randn(100)
    outlier_data = clean_data.clone()
    outlier_data[90:] = clean_data[90:] + 10  # Add outliers
    
    targets = torch.zeros_like(clean_data)
    
    losses = {
        'MSE': nn.MSELoss(),
        'MAE': nn.L1Loss(),
        'Huber (δ=1)': HuberLoss(delta=1.0),
        'Huber (δ=2)': HuberLoss(delta=2.0),
        'Smooth L1': nn.SmoothL1Loss(),
        'Log-Cosh': LogCoshLoss()
    }
    
    print("Loss comparison (clean vs outlier data):")
    for name, loss_fn in losses.items():
        clean_loss = loss_fn(clean_data, targets)
        outlier_loss = loss_fn(outlier_data, targets)
        ratio = outlier_loss / clean_loss
        
        print(f"{name:12}: Clean={clean_loss:.4f}, Outlier={outlier_loss:.4f}, Ratio={ratio:.2f}")

compare_losses_with_outliers()

print("\n=== Custom Loss Combinations ===")

class CombinedLoss(nn.Module):
    """Combination of multiple losses"""
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        self.alpha = alpha  # MSE weight
        self.beta = beta    # MAE weight
        self.gamma = gamma  # Smooth L1 weight
        
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.smooth_l1 = nn.SmoothL1Loss()
    
    def forward(self, input_tensor, target):
        mse_loss = self.mse(input_tensor, target)
        mae_loss = self.mae(input_tensor, target)
        smooth_l1_loss = self.smooth_l1(input_tensor, target)
        
        total_loss = (self.alpha * mse_loss + 
                     self.beta * mae_loss + 
                     self.gamma * smooth_l1_loss)
        
        return total_loss, {
            'mse': mse_loss,
            'mae': mae_loss,
            'smooth_l1': smooth_l1_loss,
            'total': total_loss
        }

class GraduallyRobustLoss(nn.Module):
    """Loss that becomes more robust over time"""
    def __init__(self, initial_beta=0.1, final_beta=2.0):
        super().__init__()
        self.initial_beta = initial_beta
        self.final_beta = final_beta
        self.current_epoch = 0
        self.max_epochs = 100
    
    def set_epoch(self, epoch, max_epochs=100):
        self.current_epoch = epoch
        self.max_epochs = max_epochs
    
    def forward(self, input_tensor, target):
        # Gradually increase beta (robustness)
        progress = min(self.current_epoch / self.max_epochs, 1.0)
        current_beta = self.initial_beta + progress * (self.final_beta - self.initial_beta)
        
        diff = torch.abs(input_tensor - target)
        loss = torch.where(diff < current_beta,
                          0.5 * diff ** 2 / current_beta,
                          diff - 0.5 * current_beta)
        
        return torch.mean(loss)

# Test combined losses
combined_loss = CombinedLoss()
gradual_loss = GraduallyRobustLoss()

combined_result, loss_components = combined_loss(predictions, targets)
print(f"\nCombined Loss: {combined_result.item():.6f}")
print("Components:")
for name, value in loss_components.items():
    print(f"  {name}: {value.item():.6f}")

# Test gradual loss at different epochs
gradual_loss.set_epoch(0, 100)
gradual_early = gradual_loss(predictions, targets)

gradual_loss.set_epoch(50, 100)
gradual_mid = gradual_loss(predictions, targets)

gradual_loss.set_epoch(100, 100)
gradual_late = gradual_loss(predictions, targets)

print(f"\nGradually Robust Loss:")
print(f"  Epoch 0: {gradual_early.item():.6f}")
print(f"  Epoch 50: {gradual_mid.item():.6f}")
print(f"  Epoch 100: {gradual_late.item():.6f}")

print("\n=== Regression Loss Best Practices ===")

print("Loss Selection Guidelines:")
print("1. MSE: When you want to penalize large errors heavily")
print("2. MAE: When you want robustness to outliers")
print("3. Huber/Smooth L1: Balance between MSE and MAE benefits")
print("4. Quantile Loss: When you need prediction intervals")
print("5. MAPE: When relative errors are more important")
print("6. Log-Cosh: Smooth approximation to MAE")

print("\nImplementation Tips:")
print("1. Always check for numerical stability (avoid division by zero)")
print("2. Consider data preprocessing (normalization/standardization)")
print("3. Use appropriate reduction method for your use case")
print("4. Monitor loss values during training for debugging")
print("5. Consider combining multiple losses for better performance")
print("6. Use robust losses when dealing with noisy data")

print("\nCommon Issues:")
print("1. Exploding gradients with MSE on large values")
print("2. Vanishing gradients with saturated activation functions")
print("3. Scale sensitivity - normalize targets when possible")
print("4. Outlier sensitivity - consider robust alternatives")
print("5. Multi-scale problems - use relative losses")

print("\n=== Regression Losses Complete ===")

# Memory cleanup
del predictions, targets, positive_predictions, positive_targets
del multi_predictions, multi_targets