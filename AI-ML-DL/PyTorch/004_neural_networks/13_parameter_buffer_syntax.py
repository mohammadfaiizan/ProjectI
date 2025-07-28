#!/usr/bin/env python3
"""PyTorch Parameter Buffer Syntax - Parameters, buffers, non-trainable params"""

import torch
import torch.nn as nn

print("=== Parameters and Buffers Overview ===")

print("Parameters and buffers provide:")
print("1. Trainable parameter management")
print("2. Non-trainable state storage")
print("3. Automatic device movement")
print("4. State dictionary integration")
print("5. Gradient computation control")

print("\n=== Basic Parameters ===")

# Creating parameters
param1 = nn.Parameter(torch.randn(3, 4))
param2 = nn.Parameter(torch.zeros(10), requires_grad=False)

print(f"Parameter 1 shape: {param1.shape}")
print(f"Parameter 1 requires_grad: {param1.requires_grad}")
print(f"Parameter 2 requires_grad: {param2.requires_grad}")

# Parameters in modules
class SimpleParameterModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(5, 3))
        self.bias = nn.Parameter(torch.zeros(5))
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        return torch.matmul(x, self.weight.t()) + self.bias * self.scale

param_module = SimpleParameterModule()
print(f"\nModule parameters:")
for name, param in param_module.named_parameters():
    print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")

print("\n=== Register Buffer ===")

class BufferModule(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Trainable parameters
        self.weight = nn.Parameter(torch.randn(4, 4))
        
        # Non-trainable buffers
        self.register_buffer('running_mean', torch.zeros(4))
        self.register_buffer('running_var', torch.ones(4))
        self.register_buffer('num_batches_tracked', torch.tensor(0))
        
        # Persistent vs non-persistent buffers
        self.register_buffer('persistent_buffer', torch.randn(4), persistent=True)
        self.register_buffer('non_persistent_buffer', torch.randn(4), persistent=False)
    
    def forward(self, x):
        # Update running statistics
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            
            # Exponential moving average
            momentum = 0.1
            self.running_mean.mul_(1 - momentum).add_(batch_mean, alpha=momentum)
            self.running_var.mul_(1 - momentum).add_(batch_var, alpha=momentum)
            self.num_batches_tracked += 1
        
        # Normalize using running statistics
        normalized = (x - self.running_mean) / torch.sqrt(self.running_var + 1e-5)
        
        return torch.matmul(normalized, self.weight)

buffer_module = BufferModule()

print("Module buffers:")
for name, buffer in buffer_module.named_buffers():
    print(f"  {name}: {buffer.shape}")

print("\nModule parameters vs buffers:")
print(f"  Parameters: {len(list(buffer_module.parameters()))}")
print(f"  Buffers: {len(list(buffer_module.buffers()))}")
print(f"  Total tensors: {len(list(buffer_module.state_dict()))}")

print("\n=== Parameter Properties ===")

class ParameterPropertiesModule(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Different parameter configurations
        self.trainable_weight = nn.Parameter(torch.randn(3, 3))
        self.frozen_weight = nn.Parameter(torch.randn(3, 3))
        self.frozen_weight.requires_grad = False
        
        # Buffer that moves with the module
        self.register_buffer('constant_tensor', torch.eye(3))
    
    def freeze_parameter(self, param_name):
        """Freeze a specific parameter"""
        if hasattr(self, param_name):
            getattr(self, param_name).requires_grad = False
    
    def unfreeze_parameter(self, param_name):
        """Unfreeze a specific parameter"""
        if hasattr(self, param_name):
            getattr(self, param_name).requires_grad = True

prop_module = ParameterPropertiesModule()

print("Parameter properties:")
for name, param in prop_module.named_parameters():
    print(f"  {name}: requires_grad={param.requires_grad}, shape={param.shape}")

# Test freezing/unfreezing
prop_module.freeze_parameter('trainable_weight')
print(f"\nAfter freezing trainable_weight:")
print(f"  trainable_weight.requires_grad: {prop_module.trainable_weight.requires_grad}")

prop_module.unfreeze_parameter('trainable_weight')
print(f"After unfreezing:")
print(f"  trainable_weight.requires_grad: {prop_module.trainable_weight.requires_grad}")

print("\n=== State Dictionary Operations ===")

class StateDictModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 3)
        self.bn = nn.BatchNorm1d(3)
        
        # Custom parameters and buffers
        self.custom_param = nn.Parameter(torch.randn(3))
        self.register_buffer('custom_buffer', torch.ones(3))
        
        # Non-persistent buffer (won't be saved)
        self.register_buffer('temp_buffer', torch.zeros(3), persistent=False)

state_module = StateDictModule()

# Get state dictionary
state_dict = state_module.state_dict()
print("State dictionary keys:")
for key in state_dict.keys():
    tensor = state_dict[key]
    print(f"  {key}: {tensor.shape} ({tensor.dtype})")

# Save and load state
torch.save(state_dict, 'model_state.pth')
loaded_state = torch.load('model_state.pth')

print(f"\nLoaded state keys: {list(loaded_state.keys())}")
print(f"temp_buffer in saved state: {'temp_buffer' in loaded_state}")

# Load state into module
new_module = StateDictModule()
new_module.load_state_dict(loaded_state)
print("State loaded successfully")

print("\n=== Custom Parameter Classes ===")

class ConstrainedParameter(nn.Parameter):
    """Parameter with value constraints"""
    def __new__(cls, data, constraint='none'):
        instance = super().__new__(cls, data)
        instance.constraint = constraint
        return instance
    
    def apply_constraint(self):
        """Apply constraint to parameter values"""
        with torch.no_grad():
            if self.constraint == 'positive':
                self.data.clamp_(min=0)
            elif self.constraint == 'unit_norm':
                self.data.div_(self.data.norm() + 1e-8)
            elif self.constraint == 'orthogonal':
                u, _, v = torch.svd(self.data)
                self.data.copy_(torch.matmul(u, v))

class ScaledParameter(nn.Parameter):
    """Parameter with automatic scaling"""
    def __new__(cls, data, scale_factor=1.0):
        instance = super().__new__(cls, data)
        instance.scale_factor = scale_factor
        return instance
    
    def scaled_value(self):
        """Get scaled parameter value"""
        return self.data * self.scale_factor

# Test custom parameters
constrained_param = ConstrainedParameter(torch.randn(3, 3), constraint='positive')
scaled_param = ScaledParameter(torch.randn(4), scale_factor=0.1)

print(f"Constrained parameter before: {constrained_param.min().item():.4f}")
constrained_param.apply_constraint()
print(f"Constrained parameter after: {constrained_param.min().item():.4f}")

print(f"Scaled parameter original: {scaled_param.norm():.4f}")
print(f"Scaled parameter scaled: {scaled_param.scaled_value().norm():.4f}")

print("\n=== Parameter Sharing ===")

class SharedParameterModule(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Shared weight matrix
        self.shared_weight = nn.Parameter(torch.randn(4, 4))
        
        # Multiple layers using the same weight
        self.layer1 = nn.Linear(4, 4)
        self.layer2 = nn.Linear(4, 4)
        self.layer3 = nn.Linear(4, 4)
        
        # Share the weight parameter
        self.layer1.weight = self.shared_weight
        self.layer2.weight = self.shared_weight
        self.layer3.weight = self.shared_weight
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

shared_module = SharedParameterModule()

print("Parameter sharing:")
print(f"  Shared weight id: {id(shared_module.shared_weight)}")
print(f"  Layer1 weight id: {id(shared_module.layer1.weight)}")
print(f"  Layer2 weight id: {id(shared_module.layer2.weight)}")
print(f"  Weights are same object: {shared_module.shared_weight is shared_module.layer1.weight}")

# Count unique parameters
unique_params = set()
for param in shared_module.parameters():
    unique_params.add(id(param))

print(f"  Total parameter tensors: {len(list(shared_module.parameters()))}")
print(f"  Unique parameter tensors: {len(unique_params)}")

print("\n=== Buffer Management ===")

class AdvancedBufferModule(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Different types of buffers
        self.register_buffer('counts', torch.zeros(5, dtype=torch.long))
        self.register_buffer('weights', torch.ones(5))
        self.register_buffer('mask', torch.ones(5, dtype=torch.bool))
        
        # Conditional buffer registration
        self.use_running_stats = True
        if self.use_running_stats:
            self.register_buffer('running_sum', torch.zeros(5))
            self.register_buffer('running_count', torch.tensor(0, dtype=torch.long))
    
    def update_stats(self, data):
        """Update running statistics"""
        if self.use_running_stats and self.training:
            self.running_sum += data.sum(dim=0)
            self.running_count += data.size(0)
            self.counts += 1
    
    def get_running_mean(self):
        """Get running mean"""
        if self.running_count > 0:
            return self.running_sum / self.running_count
        return self.running_sum
    
    def reset_stats(self):
        """Reset running statistics"""
        if hasattr(self, 'running_sum'):
            self.running_sum.zero_()
            self.running_count.zero_()
            self.counts.zero_()

buffer_mgmt_module = AdvancedBufferModule()

# Test buffer operations
test_data = torch.randn(10, 5)
buffer_mgmt_module.update_stats(test_data)

print("Buffer management:")
print(f"  Running count: {buffer_mgmt_module.running_count.item()}")
print(f"  Running mean: {buffer_mgmt_module.get_running_mean()}")
print(f"  Update counts: {buffer_mgmt_module.counts}")

print("\n=== Parameter Initialization ===")

class InitializationModule(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 30)
        self.linear3 = nn.Linear(30, 1)
        
        # Custom parameters
        self.custom_weight = nn.Parameter(torch.empty(5, 5))
        self.custom_bias = nn.Parameter(torch.empty(5))
        
        # Initialize parameters
        self.init_parameters()
    
    def init_parameters(self):
        """Initialize all parameters"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize custom parameters
        nn.init.orthogonal_(self.custom_weight)
        nn.init.normal_(self.custom_bias, std=0.01)

init_module = InitializationModule()

print("Parameter initialization:")
for name, param in init_module.named_parameters():
    stats = {
        'mean': param.mean().item(),
        'std': param.std().item(),
        'norm': param.norm().item()
    }
    print(f"  {name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, norm={stats['norm']:.4f}")

print("\n=== Parameter Groups ===")

class ParameterGroupModule(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Different parameter groups
        self.backbone = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        self.classifier = nn.Linear(10, 5)
        
        # Special parameters
        self.attention_weights = nn.Parameter(torch.randn(10))
    
    def get_parameter_groups(self):
        """Get parameter groups for different learning rates"""
        return [
            {
                'params': self.backbone.parameters(),
                'lr': 1e-3,
                'name': 'backbone'
            },
            {
                'params': self.classifier.parameters(),
                'lr': 1e-2,
                'name': 'classifier'
            },
            {
                'params': [self.attention_weights],
                'lr': 1e-4,
                'name': 'attention'
            }
        ]

param_group_module = ParameterGroupModule()
param_groups = param_group_module.get_parameter_groups()

print("Parameter groups:")
for group in param_groups:
    param_count = sum(p.numel() for p in group['params'])
    print(f"  {group['name']}: {param_count} parameters, lr={group['lr']}")

print("\n=== Device Management ===")

class DeviceAwareModule(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.weight = nn.Parameter(torch.randn(3, 3))
        self.register_buffer('buffer', torch.ones(3))
        
        # Store device info
        self.register_buffer('device_info', torch.tensor([0], dtype=torch.long))
    
    def to(self, device):
        """Override to method to track device changes"""
        result = super().to(device)
        
        # Update device info
        if isinstance(device, torch.device):
            device_idx = device.index if device.index is not None else 0
        elif isinstance(device, str):
            device_idx = 0
        else:
            device_idx = device
        
        result.device_info.fill_(device_idx)
        return result
    
    def get_device(self):
        """Get current device"""
        return next(self.parameters()).device

device_module = DeviceAwareModule()
print(f"Initial device: {device_module.get_device()}")

# Move to different device (if available)
if torch.cuda.is_available():
    device_module = device_module.cuda()
    print(f"After cuda(): {device_module.get_device()}")
    print(f"Device info buffer: {device_module.device_info.item()}")

print("\n=== Parameter and Buffer Utilities ===")

def count_parameters(module, trainable_only=True):
    """Count parameters in a module"""
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())

def count_buffers(module):
    """Count buffers in a module"""
    return sum(b.numel() for b in module.buffers())

def get_parameter_info(module):
    """Get detailed parameter information"""
    info = {
        'trainable_params': 0,
        'frozen_params': 0,
        'buffers': 0,
        'total_memory_mb': 0
    }
    
    for param in module.parameters():
        if param.requires_grad:
            info['trainable_params'] += param.numel()
        else:
            info['frozen_params'] += param.numel()
        
        # Calculate memory (4 bytes per float32)
        info['total_memory_mb'] += param.numel() * 4 / (1024 * 1024)
    
    for buffer in module.buffers():
        info['buffers'] += buffer.numel()
        info['total_memory_mb'] += buffer.numel() * 4 / (1024 * 1024)
    
    return info

# Test utilities
test_module = StateDictModule()
param_count = count_parameters(test_module)
buffer_count = count_buffers(test_module)
param_info = get_parameter_info(test_module)

print("Parameter utilities:")
print(f"  Trainable parameters: {param_count}")
print(f"  Buffer elements: {buffer_count}")
print(f"  Parameter info: {param_info}")

print("\n=== Named Parameters and Buffers ===")

def analyze_module_components(module, prefix=''):
    """Analyze all components of a module"""
    print(f"\nAnalyzing {type(module).__name__}:")
    
    # Parameters
    print("Parameters:")
    for name, param in module.named_parameters(prefix=prefix):
        print(f"  {name}: {param.shape}, grad={param.requires_grad}")
    
    # Buffers
    print("Buffers:")
    for name, buffer in module.named_buffers(prefix=prefix):
        print(f"  {name}: {buffer.shape}, dtype={buffer.dtype}")
    
    # Children
    print("Children:")
    for name, child in module.named_children():
        print(f"  {name}: {type(child).__name__}")

# Analyze a complex module
complex_module = nn.Sequential(
    nn.Conv2d(3, 16, 3),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(16, 10)
)

analyze_module_components(complex_module)

print("\n=== Parameter and Buffer Best Practices ===")

print("Parameter Guidelines:")
print("1. Use nn.Parameter for trainable parameters")
print("2. Use register_buffer for non-trainable state")
print("3. Set requires_grad=False for frozen parameters")
print("4. Use proper initialization schemes")
print("5. Consider parameter sharing for efficiency")
print("6. Group parameters for different learning rates")
print("7. Monitor parameter norms during training")

print("\nBuffer Guidelines:")
print("1. Use buffers for running statistics")
print("2. Set persistent=False for temporary buffers")
print("3. Buffers automatically move with the module")
print("4. Don't forget to update buffers in forward pass")
print("5. Use appropriate data types for buffers")
print("6. Reset buffers when needed")

print("\nState Management:")
print("- Save/load state_dict for model persistence")
print("- Use strict=False for partial loading")
print("- Handle missing/unexpected keys appropriately")
print("- Consider version compatibility")
print("- Validate loaded parameters")

print("\nCommon Patterns:")
print("- Running statistics in normalization layers")
print("- Lookup tables and embeddings")
print("- Masks and attention weights")
print("- Configuration and hyperparameters")
print("- Cached computations")

print("\n=== Parameters and Buffers Complete ===") 