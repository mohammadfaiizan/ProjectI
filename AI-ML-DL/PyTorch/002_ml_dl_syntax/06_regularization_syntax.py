#!/usr/bin/env python3
"""PyTorch Regularization Techniques - Dropout, weight decay, L1/L2 regularization"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== Dropout Regularization ===")

# Basic Dropout
dropout = nn.Dropout(p=0.5)
input_tensor = torch.randn(32, 128)

# Training mode - random zeroing
dropout.train()
output_train = dropout(input_tensor)
print(f"Training mode - input mean: {input_tensor.mean():.4f}")
print(f"Training mode - output mean: {output_train.mean():.4f}")

# Evaluation mode - no dropout
dropout.eval()
output_eval = dropout(input_tensor)
print(f"Eval mode - output mean: {output_eval.mean():.4f}")
print(f"Eval mode - input equals output: {torch.allclose(input_tensor, output_eval)}")

# Different dropout probabilities
dropout_rates = [0.1, 0.25, 0.5, 0.75]
for rate in dropout_rates:
    drop_layer = nn.Dropout(p=rate)
    drop_layer.train()
    output = drop_layer(input_tensor)
    zeros_percentage = (output == 0).float().mean() * 100
    print(f"Dropout p={rate}: {zeros_percentage:.1f}% zeros")

print("\n=== Dropout Variants ===")

# 2D Dropout (for CNNs)
dropout2d = nn.Dropout2d(p=0.25)
conv_input = torch.randn(16, 64, 32, 32)
dropout2d.train()
conv_output = dropout2d(conv_input)

print(f"Dropout2d input shape: {conv_input.shape}")
print(f"Dropout2d output shape: {conv_output.shape}")

# Check channel-wise dropout
channels_dropped = (conv_output.sum(dim=[2, 3]) == 0).sum(dim=1)
print(f"Channels dropped per sample: {channels_dropped[:5]}")  # First 5 samples

# 3D Dropout
dropout3d = nn.Dropout3d(p=0.3)
input_3d = torch.randn(8, 32, 16, 16, 16)
dropout3d.train()
output_3d = dropout3d(input_3d)
print(f"Dropout3d: {input_3d.shape} -> {output_3d.shape}")

# Alpha Dropout (for SELU networks)
alpha_dropout = nn.AlphaDropout(p=0.5)
selu_input = torch.randn(32, 128)
alpha_output = alpha_dropout(selu_input)
print(f"AlphaDropout preserves mean: {torch.allclose(selu_input.mean(), alpha_output.mean(), atol=0.1)}")

print("\n=== Functional Dropout ===")

# Functional dropout with manual control
input_func = torch.randn(32, 256)

# Training with dropout
func_output_train = F.dropout(input_func, p=0.5, training=True)
print(f"Functional dropout training: {func_output_train.mean():.4f}")

# Inference without dropout
func_output_infer = F.dropout(input_func, p=0.5, training=False)
print(f"Functional dropout inference: {func_output_infer.mean():.4f}")

# In-place dropout
input_inplace = torch.randn(32, 128)
original_mean = input_inplace.mean().item()
F.dropout(input_inplace, p=0.5, training=True, inplace=True)
print(f"In-place dropout: {original_mean:.4f} -> {input_inplace.mean():.4f}")

print("\n=== Weight Decay Regularization ===")

# Weight decay in optimizer
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Optimizer with weight decay
optimizer_wd = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
print(f"Weight decay: {optimizer_wd.param_groups[0]['weight_decay']}")

# Different weight decay for different parameter groups
conv_params = []
fc_params = []

for name, param in model.named_parameters():
    if 'weight' in name:
        if len(param.shape) > 2:  # Conv weights
            conv_params.append(param)
        else:  # FC weights
            fc_params.append(param)

optimizer_groups = torch.optim.Adam([
    {'params': conv_params, 'weight_decay': 1e-3},
    {'params': fc_params, 'weight_decay': 1e-4}
])

print(f"Conv weight decay: {optimizer_groups.param_groups[0]['weight_decay']}")
print(f"FC weight decay: {optimizer_groups.param_groups[1]['weight_decay']}")

print("\n=== Manual L1 and L2 Regularization ===")

# L2 regularization (manual implementation)
def l2_regularization(model, lambda_reg=1e-4):
    l2_loss = 0
    for param in model.parameters():
        if param.requires_grad:
            l2_loss += torch.norm(param, p=2)**2
    return lambda_reg * l2_loss

# L1 regularization (manual implementation)  
def l1_regularization(model, lambda_reg=1e-4):
    l1_loss = 0
    for param in model.parameters():
        if param.requires_grad:
            l1_loss += torch.norm(param, p=1)
    return lambda_reg * l1_loss

# Calculate regularization losses
l2_loss = l2_regularization(model, lambda_reg=1e-4)
l1_loss = l1_regularization(model, lambda_reg=1e-5)

print(f"L2 regularization loss: {l2_loss.item():.6f}")
print(f"L1 regularization loss: {l1_loss.item():.6f}")

# Add to main loss
dummy_input = torch.randn(32, 784)
dummy_target = torch.randint(0, 10, (32,))
criterion = nn.CrossEntropyLoss()

output = model(dummy_input)
main_loss = criterion(output, dummy_target)
total_loss = main_loss + l2_loss + l1_loss

print(f"Main loss: {main_loss.item():.4f}")
print(f"Total loss with regularization: {total_loss.item():.4f}")

print("\n=== Elastic Net Regularization ===")

# Combination of L1 and L2
def elastic_net_regularization(model, l1_lambda=1e-5, l2_lambda=1e-4, alpha=0.5):
    l1_loss = l1_regularization(model, l1_lambda)
    l2_loss = l2_regularization(model, l2_lambda)
    return alpha * l1_loss + (1 - alpha) * l2_loss

elastic_loss = elastic_net_regularization(model, l1_lambda=1e-5, l2_lambda=1e-4, alpha=0.7)
print(f"Elastic net regularization: {elastic_loss.item():.6f}")

print("\n=== Batch Normalization as Regularization ===")

# Model with and without batch normalization
class ModelWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x

class ModelWithoutBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

model_with_bn = ModelWithBN()
model_without_bn = ModelWithoutBN()

# Compare parameter count
with_bn_params = sum(p.numel() for p in model_with_bn.parameters())
without_bn_params = sum(p.numel() for p in model_without_bn.parameters())

print(f"Model with BN parameters: {with_bn_params}")
print(f"Model without BN parameters: {without_bn_params}")
print(f"BN adds {with_bn_params - without_bn_params} parameters")

print("\n=== Data Augmentation as Regularization ===")

# Simulate data augmentation effects
import torch.nn.functional as F

def random_noise_augmentation(x, noise_std=0.1):
    """Add random noise as augmentation"""
    noise = torch.randn_like(x) * noise_std
    return x + noise

def random_masking(x, mask_prob=0.1):
    """Random feature masking"""
    mask = torch.rand_like(x) > mask_prob
    return x * mask.float()

# Apply augmentations
original_input = torch.randn(32, 784)
augmented_1 = random_noise_augmentation(original_input, 0.05)
augmented_2 = random_masking(original_input, 0.1)

print(f"Original input mean: {original_input.mean():.4f}")
print(f"Noise augmented mean: {augmented_1.mean():.4f}")
print(f"Masked input mean: {augmented_2.mean():.4f}")

print("\n=== Early Stopping Regularization ===")

# Early stopping implementation
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                self.restore_checkpoint(model)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = {name: param.clone() for name, param in model.named_parameters()}
    
    def restore_checkpoint(self, model):
        if self.best_weights:
            for name, param in model.named_parameters():
                param.data.copy_(self.best_weights[name])

# Test early stopping
early_stopping = EarlyStopping(patience=3)
test_model = nn.Linear(10, 1)

# Simulate training with validation losses
val_losses = [1.0, 0.8, 0.6, 0.65, 0.63, 0.64, 0.66, 0.67]
for epoch, val_loss in enumerate(val_losses):
    should_stop = early_stopping(val_loss, test_model)
    print(f"Epoch {epoch}: val_loss={val_loss:.3f}, stop={should_stop}")
    if should_stop:
        break

print("\n=== Label Smoothing Regularization ===")

# Label smoothing implementation
class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * F.log_softmax(pred, dim=1), dim=1))

# Test label smoothing
smoothing_loss = LabelSmoothingLoss(num_classes=10, smoothing=0.1)
logits = torch.randn(32, 10)
targets = torch.randint(0, 10, (32,))

smooth_loss = smoothing_loss(logits, targets)
ce_loss = F.cross_entropy(logits, targets)

print(f"Cross entropy loss: {ce_loss.item():.4f}")
print(f"Label smoothing loss: {smooth_loss.item():.4f}")

print("\n=== Mixup Regularization ===")

# Mixup data augmentation
def mixup_data(x, y, alpha=1.0):
    """Mixup data augmentation"""
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample()
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Test mixup
x_mixup = torch.randn(32, 784)
y_mixup = torch.randint(0, 10, (32,))

mixed_x, y_a, y_b, lam = mixup_data(x_mixup, y_mixup, alpha=0.2)
print(f"Mixup lambda: {lam:.3f}")
print(f"Original input mean: {x_mixup.mean():.4f}")
print(f"Mixed input mean: {mixed_x.mean():.4f}")

print("\n=== Cutout Regularization ===")

# Cutout augmentation for images
def cutout(img, mask_size=16):
    """Apply cutout to image"""
    _, h, w = img.shape
    y = torch.randint(0, h, (1,)).item()
    x = torch.randint(0, w, (1,)).item()
    
    y1 = max(0, y - mask_size // 2)
    y2 = min(h, y + mask_size // 2)
    x1 = max(0, x - mask_size // 2)
    x2 = min(w, x + mask_size // 2)
    
    img_cutout = img.clone()
    img_cutout[:, y1:y2, x1:x2] = 0
    
    return img_cutout

# Test cutout
img_test = torch.randn(3, 64, 64)
img_cutout = cutout(img_test, mask_size=16)

non_zero_original = (img_test != 0).sum()
non_zero_cutout = (img_cutout != 0).sum()

print(f"Original non-zero pixels: {non_zero_original}")
print(f"Cutout non-zero pixels: {non_zero_cutout}")
print(f"Pixels masked: {non_zero_original - non_zero_cutout}")

print("\n=== Regularization Comparison ===")

# Compare different regularization effects
class RegularizedModel(nn.Module):
    def __init__(self, use_dropout=True, use_bn=True, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
        self.use_dropout = use_dropout
        self.use_bn = use_bn
        
        if use_bn:
            self.bn1 = nn.BatchNorm1d(256)
            self.bn2 = nn.BatchNorm1d(128)
        
        if use_dropout:
            self.dropout = nn.Dropout(dropout_rate)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        if self.use_dropout:
            x = self.dropout(x)
        
        x = self.fc2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.relu(x)
        if self.use_dropout:
            x = self.dropout(x)
        
        x = self.fc3(x)
        return x

# Test different configurations
configs = [
    {'use_dropout': False, 'use_bn': False, 'name': 'No regularization'},
    {'use_dropout': True, 'use_bn': False, 'name': 'Dropout only'},
    {'use_dropout': False, 'use_bn': True, 'name': 'BatchNorm only'},
    {'use_dropout': True, 'use_bn': True, 'name': 'Dropout + BatchNorm'},
]

test_input = torch.randn(32, 784)

for config in configs:
    model_test = RegularizedModel(**{k: v for k, v in config.items() if k != 'name'})
    model_test.train()
    
    output1 = model_test(test_input)
    output2 = model_test(test_input)
    
    # Check reproducibility (should be different with dropout)
    different = not torch.allclose(output1, output2, atol=1e-6)
    print(f"{config['name']:>20}: Outputs different = {different}")

print("\n=== Regularization Best Practices ===")

print("Regularization Guidelines:")
print("1. Start with dropout (0.2-0.5) for fully connected layers")
print("2. Use BatchNorm for better convergence and implicit regularization")
print("3. Apply weight decay (1e-4 to 1e-3) in optimizer")
print("4. Consider data augmentation as primary regularization")
print("5. Use early stopping to prevent overfitting")
print("6. Label smoothing for better generalization")
print("7. Mixup/Cutmix for strong data augmentation")

print("\nDropout Tips:")
print("- Higher rates (0.5-0.8) for larger networks")
print("- Lower rates (0.1-0.3) for smaller networks or CNNs")
print("- No dropout in final layer usually")
print("- Use Dropout2d for convolutional layers")

print("\nWeight Decay:")
print("- Typically 1e-4 for Adam, 5e-4 for SGD")
print("- Don't apply to bias terms")
print("- Different rates for different layer types")
print("- Monitor if weights become too small")

print("\n=== Regularization Complete ===") 