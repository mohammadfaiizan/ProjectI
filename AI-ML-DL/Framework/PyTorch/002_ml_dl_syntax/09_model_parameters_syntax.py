#!/usr/bin/env python3
"""PyTorch Model Parameters - Parameter access, freezing, sharing"""

import torch
import torch.nn as nn

print("=== Basic Parameter Access ===")

# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleModel()

# Access all parameters
print("All model parameters:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

print("\n=== Parameter Groups ===")

# Group parameters by type
conv_params = []
bn_params = []
fc_params = []

for name, param in model.named_parameters():
    if 'conv' in name:
        conv_params.append(param)
    elif 'bn' in name:
        bn_params.append(param)
    elif 'fc' in name:
        fc_params.append(param)

print(f"Conv parameters: {sum(p.numel() for p in conv_params):,}")
print(f"BatchNorm parameters: {sum(p.numel() for p in bn_params):,}")
print(f"FC parameters: {sum(p.numel() for p in fc_params):,}")

# Access specific layer parameters
conv1_weight = model.conv1.weight
conv1_bias = model.conv1.bias

print(f"\nConv1 weight shape: {conv1_weight.shape}")
print(f"Conv1 bias shape: {conv1_bias.shape}")
print(f"Conv1 weight requires_grad: {conv1_weight.requires_grad}")

print("\n=== Parameter Freezing ===")

# Freeze specific parameters
def freeze_layer(layer):
    """Freeze all parameters in a layer"""
    for param in layer.parameters():
        param.requires_grad = False

# Freeze conv1 layer
freeze_layer(model.conv1)
print(f"Conv1 weight requires_grad after freezing: {model.conv1.weight.requires_grad}")

# Freeze by name pattern
def freeze_parameters_by_name(model, patterns):
    """Freeze parameters whose names match patterns"""
    frozen_count = 0
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in patterns):
            param.requires_grad = False
            frozen_count += 1
            print(f"Frozen: {name}")
    return frozen_count

# Freeze all batch norm parameters
frozen = freeze_parameters_by_name(model, ['bn'])
print(f"Frozen {frozen} batch norm parameters")

# Unfreeze parameters
def unfreeze_layer(layer):
    """Unfreeze all parameters in a layer"""
    for param in layer.parameters():
        param.requires_grad = True

unfreeze_layer(model.conv1)
print(f"Conv1 weight requires_grad after unfreezing: {model.conv1.weight.requires_grad}")

print("\n=== Parameter Statistics ===")

def analyze_parameters(model):
    """Analyze parameter statistics"""
    stats = {}
    
    for name, param in model.named_parameters():
        stats[name] = {
            'shape': param.shape,
            'numel': param.numel(),
            'mean': param.data.mean().item(),
            'std': param.data.std().item(),
            'min': param.data.min().item(),
            'max': param.data.max().item(),
            'requires_grad': param.requires_grad
        }
    
    return stats

param_stats = analyze_parameters(model)

# Print stats for first few parameters
for name, stats in list(param_stats.items())[:3]:
    print(f"\n{name}:")
    print(f"  Shape: {stats['shape']}")
    print(f"  Count: {stats['numel']:,}")
    print(f"  Mean: {stats['mean']:.6f}")
    print(f"  Std: {stats['std']:.6f}")
    print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")

print("\n=== Parameter Initialization ===")

# Initialize specific parameters
def init_weights(m):
    """Initialize weights of different layer types"""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

# Apply initialization
model.apply(init_weights)
print("Model weights initialized")

# Initialize specific layer manually
nn.init.normal_(model.fc1.weight, mean=0, std=0.01)
nn.init.constant_(model.fc1.bias, 0)
print("FC1 layer manually initialized")

print("\n=== Parameter Sharing ===")

# Shared parameter example
class ModelWithSharedParams(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_fc = nn.Linear(128, 64)
        self.fc1 = nn.Linear(256, 128)
        # Share weights between two layers
        self.fc2 = nn.Linear(128, 64)
        self.fc2.weight = self.shared_fc.weight
        self.fc2.bias = self.shared_fc.bias
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.shared_fc(x)  # Use shared parameters
        x = torch.relu(x)
        x = self.fc3(x)
        return x

shared_model = ModelWithSharedParams()

# Check parameter sharing
print(f"Shared FC weight id: {id(shared_model.shared_fc.weight)}")
print(f"FC2 weight id: {id(shared_model.fc2.weight)}")
print(f"Weights are shared: {shared_model.shared_fc.weight is shared_model.fc2.weight}")

# Count unique parameters
unique_params = set()
for param in shared_model.parameters():
    unique_params.add(id(param))

all_params = sum(p.numel() for p in shared_model.parameters())
unique_param_count = len(unique_params)

print(f"All parameter count: {all_params}")
print(f"Unique parameter objects: {unique_param_count}")

print("\n=== Parameter Gradients ===")

# Check gradient status
input_data = torch.randn(4, 3, 32, 32)
output = model(input_data)
loss = output.sum()
loss.backward()

print("Parameter gradients:")
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"  {name}: grad_norm={grad_norm:.6f}")
    else:
        print(f"  {name}: No gradient")

# Zero gradients
model.zero_grad()
print("\nGradients zeroed")

# Check which parameters have gradients after zeroing
has_grad = sum(1 for p in model.parameters() if p.grad is not None)
print(f"Parameters with gradients after zero_grad(): {has_grad}")

print("\n=== Parameter State Dict Operations ===")

# Get parameter state dict
state_dict = model.state_dict()
print(f"State dict keys: {len(state_dict)}")

# Modify parameter values
original_conv1_weight = model.conv1.weight.clone()
model.conv1.weight.data.fill_(0.5)

print(f"Original conv1 weight mean: {original_conv1_weight.mean():.6f}")
print(f"Modified conv1 weight mean: {model.conv1.weight.mean():.6f}")

# Restore from state dict
model.load_state_dict(state_dict)
print(f"Restored conv1 weight mean: {model.conv1.weight.mean():.6f}")

print("\n=== Parameter Buffers ===")

# Access buffers (non-trainable parameters like BatchNorm running stats)
print("Model buffers:")
for name, buffer in model.named_buffers():
    print(f"  {name}: {buffer.shape}")

# Register custom buffer
model.register_buffer('custom_buffer', torch.randn(10))
print(f"Custom buffer registered: {model.custom_buffer.shape}")

# Buffers are included in state_dict but not in parameters()
buffer_count = len(list(model.buffers()))
param_count = len(list(model.parameters()))
state_dict_count = len(model.state_dict())

print(f"Buffers: {buffer_count}")
print(f"Parameters: {param_count}")
print(f"State dict entries: {state_dict_count}")

print("\n=== Parameter Transfer ===")

# Transfer parameters between models
source_model = SimpleModel()
target_model = SimpleModel()

# Copy specific layer parameters
def copy_layer_params(source_layer, target_layer):
    """Copy parameters from source to target layer"""
    with torch.no_grad():
        for (src_name, src_param), (tgt_name, tgt_param) in zip(
            source_layer.named_parameters(), target_layer.named_parameters()
        ):
            if src_param.shape == tgt_param.shape:
                tgt_param.copy_(src_param)
                print(f"Copied {src_name} -> {tgt_name}")

# Copy conv1 parameters
copy_layer_params(source_model.conv1, target_model.conv1)

# Verify copy
params_equal = torch.allclose(source_model.conv1.weight, target_model.conv1.weight)
print(f"Conv1 parameters copied successfully: {params_equal}")

print("\n=== Parameter Monitoring ===")

class ParameterMonitor:
    """Monitor parameter changes during training"""
    
    def __init__(self, model):
        self.model = model
        self.initial_params = {}
        self.save_initial_params()
    
    def save_initial_params(self):
        """Save initial parameter values"""
        for name, param in self.model.named_parameters():
            self.initial_params[name] = param.clone().detach()
    
    def compute_parameter_changes(self):
        """Compute how much parameters have changed"""
        changes = {}
        for name, param in self.model.named_parameters():
            if name in self.initial_params:
                initial = self.initial_params[name]
                current = param.detach()
                change = torch.norm(current - initial).item()
                relative_change = change / torch.norm(initial).item()
                changes[name] = {
                    'absolute_change': change,
                    'relative_change': relative_change
                }
        return changes

# Test parameter monitoring
monitor = ParameterMonitor(model)

# Simulate training step
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
dummy_input = torch.randn(4, 3, 32, 32)
dummy_target = torch.randint(0, 10, (4,))

optimizer.zero_grad()
output = model(dummy_input)
loss = nn.CrossEntropyLoss()(output, dummy_target)
loss.backward()
optimizer.step()

# Check parameter changes
changes = monitor.compute_parameter_changes()
print("\nParameter changes after one training step:")
for name, change_info in list(changes.items())[:3]:
    print(f"  {name}:")
    print(f"    Absolute change: {change_info['absolute_change']:.6f}")
    print(f"    Relative change: {change_info['relative_change']:.6f}")

print("\n=== Advanced Parameter Operations ===")

# Parameter regularization
def l1_regularization(model, lambda_reg=1e-5):
    """Compute L1 regularization for model parameters"""
    l1_loss = 0
    for param in model.parameters():
        if param.requires_grad:
            l1_loss += torch.norm(param, p=1)
    return lambda_reg * l1_loss

def l2_regularization(model, lambda_reg=1e-4):
    """Compute L2 regularization for model parameters"""
    l2_loss = 0
    for param in model.parameters():
        if param.requires_grad:
            l2_loss += torch.norm(param, p=2) ** 2
    return lambda_reg * l2_loss

l1_reg = l1_regularization(model)
l2_reg = l2_regularization(model)

print(f"L1 regularization: {l1_reg.item():.6f}")
print(f"L2 regularization: {l2_reg.item():.6f}")

# Parameter clipping
def clip_parameters(model, max_norm=1.0):
    """Clip parameter values to max norm"""
    clipped_count = 0
    for param in model.parameters():
        if param.requires_grad:
            param_norm = torch.norm(param)
            if param_norm > max_norm:
                param.data = param.data * (max_norm / param_norm)
                clipped_count += 1
    return clipped_count

clipped = clip_parameters(model, max_norm=2.0)
print(f"Clipped {clipped} parameters")

print("\n=== Parameter Best Practices ===")

print("Parameter Management Guidelines:")
print("1. Access parameters through .named_parameters() for inspection")
print("2. Use .requires_grad=False to freeze parameters")
print("3. Group parameters by type for different optimizers")
print("4. Monitor parameter statistics during training")
print("5. Use parameter sharing for memory efficiency")
print("6. Initialize parameters appropriately for your architecture")
print("7. Save/load state_dict for model persistence")

print("\nCommon Patterns:")
print("- Feature extraction: Freeze backbone, train classifier")
print("- Fine-tuning: Lower LR for pretrained params")
print("- Parameter sharing: RNN cells, Siamese networks")
print("- Regularization: L1/L2 on specific parameter groups")

print("\nDebugging Tips:")
print("- Check requires_grad status if gradients are missing")
print("- Monitor parameter norms to detect exploding/vanishing")
print("- Verify parameter sharing with 'is' operator")
print("- Use parameter hooks for detailed gradient monitoring")

print("\n=== Model Parameters Complete ===") 