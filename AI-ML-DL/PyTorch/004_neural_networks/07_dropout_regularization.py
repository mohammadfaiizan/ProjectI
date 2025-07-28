#!/usr/bin/env python3
"""PyTorch Dropout Regularization - All dropout variants and usage"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== Dropout Regularization Overview ===")

print("Dropout layers provide:")
print("1. Regularization to prevent overfitting")
print("2. Ensemble effect by training multiple sub-networks")
print("3. Reduced co-adaptation of neurons")
print("4. Different variants for various data types")
print("5. Active only during training mode")

print("\n=== Basic Dropout (nn.Dropout) ===")

# Basic Dropout for fully connected layers
dropout_layer = nn.Dropout(p=0.5) # Dropout probability of 0.5

# Input: (batch_size, features)
input_tensor = torch.randn(4, 10)

# Set model to training mode (dropout is active)
dropout_layer.train()
output_train = dropout_layer(input_tensor)

# Set model to evaluation mode (dropout is inactive)
dropout_layer.eval()
output_eval = dropout_layer(input_tensor)

print(f"Input tensor:\n{input_tensor}")
print(f"Output (training mode):\n{output_train}")
print(f"Output (evaluation mode):\n{output_eval}")

print(f"Number of zeros in training output: {(output_train == 0).sum().item()}")
print(f"Number of zeros in evaluation output: {(output_eval == 0).sum().item()}")

# Dropout with different probabilities
p_values = [0.1, 0.5, 0.8]
for p in p_values:
    do = nn.Dropout(p=p)
    do.train()
    out = do(input_tensor)
    print(f"\nDropout with p={p}:")
    print(f"  Input mean: {input_tensor.mean():.4f}, std: {input_tensor.std():.4f}")
    print(f"  Output mean: {out.mean():.4f}, std: {out.std():.4f}")
    print(f"  Zeroed elements: {(out == 0).sum().item() / out.numel():.2%}")

print("\n=== Dropout2d (Spatial Dropout) ===")

# Dropout2d for convolutional layers (drops entire feature maps)
dropout2d_layer = nn.Dropout2d(p=0.3)

# Input: (batch_size, channels, height, width)
input_2d = torch.randn(2, 3, 5, 5)

dropout2d_layer.train()
output_2d_train = dropout2d_layer(input_2d)

print(f"Input 2D shape: {input_2d.shape}")
print(f"Output 2D (training mode) shape: {output_2d_train.shape}")
print(f"Original channel 0:\n{input_2d[0, 0]}")
print(f"Dropped channel 0 (training):\n{output_2d_train[0, 0]}")

# Check if entire channels are zeroed out
zeroed_channels = (output_2d_train == 0).all(dim=[2, 3]).sum(dim=0)
print(f"Channels completely zeroed out per batch: {zeroed_channels}")

print("\n=== Dropout3d (Volumetric Dropout) ===")

# Dropout3d for 3D convolutional layers
dropout3d_layer = nn.Dropout3d(p=0.2)

# Input: (batch_size, channels, depth, height, width)
input_3d = torch.randn(1, 2, 3, 4, 4)

dropout3d_layer.train()
output_3d_train = dropout3d_layer(input_3d)

print(f"\nInput 3D shape: {input_3d.shape}")
print(f"Output 3D (training mode) shape: {output_3d_train.shape}")

print("\n=== Alpha Dropout ===")

# Alpha Dropout for SELU activations (preserves mean and variance)
alpha_dropout = nn.AlphaDropout(p=0.1)

# Input should be from SELU
input_alpha = F.selu(torch.randn(4, 10))

alpha_dropout.train()
output_alpha_train = alpha_dropout(input_alpha)

print(f"Alpha Dropout input mean: {input_alpha.mean():.4f}, std: {input_alpha.std():.4f}")
print(f"Alpha Dropout output mean: {output_alpha_train.mean():.4f}, std: {output_alpha_train.std():.4f}")

print("\n=== Feature Dropout (nn.FeatureDropout) ===")

# nn.FeatureDropout (deprecated in recent PyTorch versions, but conceptually useful)
# This is similar to Dropout1d applied to features, not sequence length.
# For demonstration, we'll use a custom implementation or simulate its effect.

class CustomFeatureDropout(nn.Module):
    """Simulates nn.FeatureDropout by dropping entire feature dimensions."""
    def __init__(self, p=0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p

    def forward(self, input):
        if not self.training or self.p == 0:
            return input
        
        # Create a mask for features (last dimension)
        num_features = input.size(-1)
        mask = torch.ones(num_features, device=input.device)
        num_dropped = int(self.p * num_features)
        
        # Randomly select features to drop
        indices_to_drop = torch.randperm(num_features)[:num_dropped]
        mask[indices_to_drop] = 0
        
        # Apply mask and scale
        return input * mask * (1.0 / (1.0 - self.p))

feature_dropout = CustomFeatureDropout(p=0.5)
input_feature = torch.randn(4, 5, 10) # (batch, sequence, features)

feature_dropout.train()
output_feature_train = feature_dropout(input_feature)

print(f"\nFeature Dropout input shape: {input_feature.shape}")
print(f"Feature Dropout output shape: {output_feature_train.shape}")
print(f"Original feature 0 (first sample):\n{input_feature[0, :, 0]}")
print(f"Dropped feature 0 (first sample):\n{output_feature_train[0, :, 0]}")

print("\n=== Functional Dropout ===")

# Using functional interface for dropout
input_func = torch.randn(4, 10)

# Functional dropout (requires manual handling of training mode and scaling)
output_func_train = F.dropout(input_func, p=0.5, training=True)
output_func_eval = F.dropout(input_func, p=0.5, training=False)

print(f"Functional dropout (training=True):\n{output_func_train}")
print(f"Functional dropout (training=False):\n{output_func_eval}")

# Functional dropout2d
input_func_2d = torch.randn(2, 3, 5, 5)
output_func_2d = F.dropout2d(input_func_2d, p=0.3, training=True)
print(f"Functional dropout2d output shape: {output_func_2d.shape}")

print("\n=== Dropout in Neural Networks ===")

class DropoutNet(nn.Module):
    """Simple network demonstrating dropout usage"""
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x) # Dropout applied after activation
        x = self.fc2(x)
        return x

# Create and test network with dropout
net_with_dropout = DropoutNet(10, 20, 1)
input_net = torch.randn(5, 10)

# Training mode
net_with_dropout.train()
output_net_train = net_with_dropout(input_net)
print(f"Network output (training mode): {output_net_train.shape}")

# Evaluation mode
net_with_dropout.eval()
output_net_eval = net_with_dropout(input_net)
print(f"Network output (evaluation mode): {output_net_eval.shape}")

# Verify different outputs in train/eval
print(f"Outputs are different in train/eval: {not torch.allclose(output_net_train, output_net_eval)}")

print("\n=== Dropout for Regularization Strength ===")

# Dropout as a hyperparameter
class RegularizedNet(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout2d = nn.Dropout2d(dropout_rate)
        self.fc = nn.Linear(16 * 14 * 14, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout2d(x) # Spatial dropout
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Test with different dropout rates
input_reg = torch.randn(4, 1, 28, 28)
for rate in [0.0, 0.2, 0.5]:
    reg_net = RegularizedNet(rate)
    reg_net.train()
    output_reg = reg_net(input_reg)
    print(f"\nRegularizedNet with dropout_rate={rate}:")
    print(f"  Output shape: {output_reg.shape}")
    # Check if dropout is active (by checking for zeros in intermediate layers)
    if rate > 0:
        # A simple way to check if dropout is active is to see if some values are zeroed out
        # This is not a perfect check but gives an idea
        temp_output = reg_net.pool(F.relu(reg_net.conv1(input_reg)))
        dropped_output = reg_net.dropout2d(temp_output)
        num_zeros = (dropped_output == 0).sum().item()
        print(f"  Number of zeros after dropout: {num_zeros}")

print("\n=== DropConnect Implementation ===")

class DropConnect(nn.Module):
    """DropConnect: drops connections instead of neurons"""
    def __init__(self, in_features, out_features, p=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p = p
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
    
    def forward(self, x):
        if self.training and self.p > 0:
            # Create binary mask for weights
            mask = torch.bernoulli(torch.full_like(self.weight, 1 - self.p))
            dropped_weight = self.weight * mask / (1 - self.p)
        else:
            dropped_weight = self.weight
        
        return F.linear(x, dropped_weight, self.bias)

# Test DropConnect
dropconnect_layer = DropConnect(10, 5, p=0.3)
input_dc = torch.randn(4, 10)

dropconnect_layer.train()
output_dc_train = dropconnect_layer(input_dc)
print(f"\nDropConnect output (training): {output_dc_train.shape}")

dropconnect_layer.eval()
output_dc_eval = dropconnect_layer(input_dc)
print(f"DropConnect output (evaluation): {output_dc_eval.shape}")

print("\n=== Advanced Dropout Techniques ===")

class DropBlock2D(nn.Module):
    """DropBlock for 2D feature maps - drops contiguous regions"""
    def __init__(self, drop_rate, block_size):
        super().__init__()
        self.drop_rate = drop_rate
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_rate == 0:
            return x
        
        # Calculate gamma (number of activation units to drop)
        gamma = self.drop_rate / (self.block_size ** 2)
        
        # Sample mask
        mask = (torch.rand(x.shape[0], x.shape[1], x.shape[2] - self.block_size + 1, 
                          x.shape[3] - self.block_size + 1, device=x.device) < gamma).float()
        
        # Expand mask to cover block regions
        mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, 
                           padding=self.block_size // 2)
        
        # Invert mask (1 where we keep, 0 where we drop)
        mask = 1 - mask
        
        # Normalize
        mask = mask * mask.numel() / mask.sum()
        
        return x * mask

# Test DropBlock
dropblock = DropBlock2D(drop_rate=0.1, block_size=3)
input_db = torch.randn(2, 4, 8, 8)

dropblock.train()
output_db = dropblock(input_db)
print(f"\nDropBlock input shape: {input_db.shape}")
print(f"DropBlock output shape: {output_db.shape}")

print("\n=== Dropout Best Practices ===")

print("Dropout Guidelines:")
print("1. Apply dropout during training only (use model.train()/model.eval())")
print("2. Common dropout rates: 0.2-0.5 for hidden layers")
print("3. Use Dropout2d/3d for convolutional layers (drops entire feature maps)")
print("4. Alpha Dropout for networks with SELU activations")
print("5. Apply dropout after activation functions (or after linear/conv and before activation)")
print("6. Don't use dropout on input or output layers (usually)")
print("7. Experiment with dropout rates as a hyperparameter")

print("\nPlacement of Dropout:")
print("- After activation: `Linear -> ReLU -> Dropout -> Linear` (common)")
print("- Before activation: `Linear -> Dropout -> ReLU -> Linear` (less common, but sometimes used)")
print("- After pooling layers in CNNs")
print("- Between recurrent layers in RNNs (but not within the cell)")

print("\nVariants and Use Cases:")
print("- `nn.Dropout`: Standard for dense layers")
print("- `nn.Dropout2d`/`nn.Dropout3d`: For convolutional layers, drops entire channels/planes")
print("- `nn.AlphaDropout`: For self-normalizing networks (SELU)")
print("- `DropConnect`: Drops individual weights instead of activations")
print("- `DropBlock`: Structured dropout for contiguous regions in feature maps")

print("\nTraining Considerations:")
print("- Dropout scales activations during training to compensate for dropped units.")
print("- During inference, dropout is turned off, and weights are effectively scaled by `1-p`.")
print("- Ensure your training loop correctly sets `model.train()` and `model.eval()`.")
print("- Dropout can slow down convergence but improves generalization.")
print("- Combine with other regularization techniques like L1/L2 regularization.")

print("\nCommon Pitfalls:")
print("- Forgetting to switch to `eval()` mode during inference/validation.")
print("- Applying dropout to input or output layers unnecessarily.")
print("- Using too high a dropout rate, leading to underfitting.")
print("- Not scaling activations correctly if implementing custom dropout.")
print("- Applying dropout within RNN cells (use `dropout` parameter in `nn.LSTM`/`nn.GRU` instead).")

print("\n=== Dropout Regularization Complete ===") 