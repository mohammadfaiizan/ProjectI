#!/usr/bin/env python3
"""PyTorch nn.Module Fundamentals - nn.Module basics, custom layers"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== nn.Module Fundamentals Overview ===")

print("nn.Module provides:")
print("1. Base class for all neural network components")
print("2. Automatic parameter tracking")
print("3. GPU/CPU device management")
print("4. Training/evaluation mode switching")
print("5. Hook registration capabilities")

print("\n=== Basic nn.Module Structure ===")

class SimpleModule(nn.Module):
    """Basic nn.Module example"""
    def __init__(self, input_size, output_size):
        super().__init__()  # Always call parent constructor
        self.linear = nn.Linear(input_size, output_size)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        """Forward pass - must be implemented"""
        x = self.linear(x)
        x = self.activation(x)
        return x

# Create and test basic module
simple_model = SimpleModule(10, 5)
input_tensor = torch.randn(3, 10)
output = simple_model(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
print(f"Model parameters: {sum(p.numel() for p in simple_model.parameters())}")

print("\n=== Custom Layer Implementation ===")

class CustomLinear(nn.Module):
    """Custom linear layer implementation"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Register parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        """Forward pass"""
        return F.linear(x, self.weight, self.bias)
    
    def extra_repr(self):
        """String representation for print()"""
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

# Test custom linear layer
custom_linear = CustomLinear(8, 4)
test_input = torch.randn(2, 8)
custom_output = custom_linear(test_input)

print(f"Custom linear layer: {custom_linear}")
print(f"Custom output shape: {custom_output.shape}")

# Compare with built-in linear
builtin_linear = nn.Linear(8, 4)
builtin_output = builtin_linear(test_input)
print(f"Built-in output shape: {builtin_output.shape}")

print("\n=== Parameter Management ===")

class ParameterExample(nn.Module):
    """Demonstrate parameter management"""
    def __init__(self):
        super().__init__()
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.randn(5, 3))
        self.bias = nn.Parameter(torch.zeros(5))
        
        # Non-learnable buffers
        self.register_buffer('running_mean', torch.zeros(5))
        self.register_buffer('running_var', torch.ones(5))
        
        # Constants (not registered)
        self.scale_factor = 2.0
    
    def forward(self, x):
        # Use parameters and buffers
        x = torch.matmul(x, self.weight.t()) + self.bias
        
        # Update running statistics (example)
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            
            # Exponential moving average
            momentum = 0.1
            self.running_mean.mul_(1 - momentum).add_(batch_mean, alpha=momentum)
            self.running_var.mul_(1 - momentum).add_(batch_var, alpha=momentum)
        
        # Normalize using running stats
        normalized = (x - self.running_mean) / torch.sqrt(self.running_var + 1e-5)
        
        return normalized * self.scale_factor

param_model = ParameterExample()

# Inspect parameters and buffers
print("Model parameters:")
for name, param in param_model.named_parameters():
    print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")

print("\nModel buffers:")
for name, buffer in param_model.named_buffers():
    print(f"  {name}: {buffer.shape}")

print(f"\nTotal parameters: {sum(p.numel() for p in param_model.parameters())}")
print(f"Trainable parameters: {sum(p.numel() for p in param_model.parameters() if p.requires_grad)}")

print("\n=== Module Hierarchy and Submodules ===")

class ComplexModel(nn.Module):
    """Complex model with submodules"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        # Sequential submodule
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Custom submodule
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)
        
        # Self-attention (batch_first=True)
        if len(encoded.shape) == 2:
            encoded = encoded.unsqueeze(1)  # Add sequence dimension
        
        attended, attention_weights = self.attention(encoded, encoded, encoded)
        attended = attended.squeeze(1)  # Remove sequence dimension
        
        # Classification
        output = self.classifier(attended)
        
        return output, attention_weights

complex_model = ComplexModel(20, 64, 10)

# Explore module hierarchy
print("Model structure:")
for name, module in complex_model.named_modules():
    if name:  # Skip root module
        print(f"  {name}: {type(module).__name__}")

print(f"\nModel has {len(list(complex_model.modules()))} modules")
print(f"Model has {len(list(complex_model.children()))} direct children")

# Test the model
complex_input = torch.randn(5, 20)
complex_output, attn_weights = complex_model(complex_input)
print(f"Complex model output shape: {complex_output.shape}")

print("\n=== Training and Evaluation Modes ===")

class ModeAwareModule(nn.Module):
    """Module that behaves differently in train/eval mode"""
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.linear = nn.Linear(features, features)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm1d(features)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)  # Only active in training mode
        return x

mode_model = ModeAwareModule(16)
mode_input = torch.randn(8, 16)

# Training mode
mode_model.train()
print(f"Training mode: {mode_model.training}")
train_output = mode_model(mode_input)
print(f"Training output std: {train_output.std():.4f}")

# Evaluation mode
mode_model.eval()
print(f"Evaluation mode: {mode_model.training}")
eval_output = mode_model(mode_input)
print(f"Evaluation output std: {eval_output.std():.4f}")

print("\n=== Device Management ===")

class DeviceAwareModule(nn.Module):
    """Module with device management utilities"""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)
    
    def get_device(self):
        """Get the device of the model parameters"""
        return next(self.parameters()).device
    
    def count_parameters(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

device_model = DeviceAwareModule(10, 5)
print(f"Model device: {device_model.get_device()}")
print(f"Total parameters: {device_model.count_parameters()}")

# Move to GPU if available
if torch.cuda.is_available():
    device_model = device_model.cuda()
    print(f"Model moved to: {device_model.get_device()}")
    
    # Move input to same device
    device_input = torch.randn(3, 10).cuda()
    device_output = device_model(device_input)
    print(f"Output device: {device_output.device}")

print("\n=== Hook Registration ===")

class HookedModule(nn.Module):
    """Module with forward and backward hooks"""
    def __init__(self, features):
        super().__init__()
        self.linear1 = nn.Linear(features, features)
        self.linear2 = nn.Linear(features, features)
        self.activations = {}
        self.gradients = {}
        
        # Register hooks
        self.register_hooks()
    
    def register_hooks(self):
        """Register forward and backward hooks"""
        def save_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach().clone()
            return hook
        
        def save_gradient(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    self.gradients[name] = grad_output[0].detach().clone()
            return hook
        
        # Register hooks
        self.linear1.register_forward_hook(save_activation('linear1'))
        self.linear1.register_backward_hook(save_gradient('linear1'))
        self.linear2.register_forward_hook(save_activation('linear2'))
        self.linear2.register_backward_hook(save_gradient('linear2'))
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

hooked_model = HookedModule(8)
hook_input = torch.randn(4, 8, requires_grad=True)

# Forward pass
hook_output = hooked_model(hook_input)
print(f"Activations captured: {list(hooked_model.activations.keys())}")

# Backward pass
loss = hook_output.sum()
loss.backward()
print(f"Gradients captured: {list(hooked_model.gradients.keys())}")

# Check captured data
for name, activation in hooked_model.activations.items():
    print(f"{name} activation shape: {activation.shape}")

print("\n=== Module State Management ===")

class StatefulModule(nn.Module):
    """Module with custom state management"""
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.linear = nn.Linear(features, features)
        
        # Register custom state
        self.register_buffer('step_count', torch.tensor(0))
        self.register_buffer('loss_history', torch.zeros(100))
        
        # Custom attributes
        self.custom_config = {'learning_rate': 0.01, 'momentum': 0.9}
    
    def forward(self, x):
        self.step_count += 1
        return self.linear(x)
    
    def update_loss_history(self, loss_value):
        """Update loss history buffer"""
        idx = (self.step_count - 1) % 100
        self.loss_history[idx] = loss_value
    
    def get_average_loss(self):
        """Get average loss from history"""
        valid_steps = min(self.step_count.item(), 100)
        if valid_steps == 0:
            return 0.0
        return self.loss_history[:valid_steps].mean().item()
    
    def state_dict(self, *args, **kwargs):
        """Custom state dict that includes custom config"""
        state = super().state_dict(*args, **kwargs)
        state['custom_config'] = self.custom_config
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        """Custom load state dict"""
        # Extract custom config
        if 'custom_config' in state_dict:
            self.custom_config = state_dict.pop('custom_config')
        
        return super().load_state_dict(state_dict, strict)

stateful_model = StatefulModule(12)

# Simulate training steps
for step in range(5):
    step_input = torch.randn(2, 12)
    step_output = stateful_model(step_input)
    step_loss = step_output.sum()
    
    stateful_model.update_loss_history(step_loss.item())
    print(f"Step {step + 1}: loss={step_loss.item():.4f}, avg_loss={stateful_model.get_average_loss():.4f}")

print(f"Total steps: {stateful_model.step_count.item()}")

# Save and load state
state = stateful_model.state_dict()
print(f"State dict keys: {list(state.keys())}")

# Create new model and load state
new_stateful_model = StatefulModule(12)
new_stateful_model.load_state_dict(state)
print(f"Loaded step count: {new_stateful_model.step_count.item()}")
print(f"Loaded config: {new_stateful_model.custom_config}")

print("\n=== Module Utilities and Helpers ===")

class UtilityModule(nn.Module):
    """Module with utility methods"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.linear = nn.Linear(16, 10)
    
    def forward(self, x):
        if len(x.shape) == 4:  # Image input
            x = F.adaptive_avg_pool2d(self.conv(x), (1, 1))
            x = x.view(x.size(0), -1)
        return self.linear(x)
    
    def summary(self):
        """Print model summary"""
        print(f"Model: {self.__class__.__name__}")
        print("Layers:")
        for name, module in self.named_modules():
            if name and len(list(module.children())) == 0:  # Leaf modules
                params = sum(p.numel() for p in module.parameters())
                print(f"  {name}: {type(module).__name__} ({params} params)")
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
    
    def freeze_layer(self, layer_name):
        """Freeze specific layer"""
        for name, param in self.named_parameters():
            if layer_name in name:
                param.requires_grad = False
                print(f"Frozen: {name}")
    
    def unfreeze_layer(self, layer_name):
        """Unfreeze specific layer"""
        for name, param in self.named_parameters():
            if layer_name in name:
                param.requires_grad = True
                print(f"Unfrozen: {name}")
    
    def get_layer_outputs(self, x, layer_names):
        """Get intermediate layer outputs"""
        outputs = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                outputs[name] = output.detach()
            return hook
        
        hooks = []
        for name, module in self.named_modules():
            if name in layer_names:
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass
        final_output = self(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return final_output, outputs

utility_model = UtilityModule()

# Model summary
utility_model.summary()

# Test freezing
utility_model.freeze_layer('conv')
print(f"\nTrainable params after freezing conv: {sum(p.numel() for p in utility_model.parameters() if p.requires_grad)}")

utility_model.unfreeze_layer('conv')
print(f"Trainable params after unfreezing: {sum(p.numel() for p in utility_model.parameters() if p.requires_grad)}")

# Get intermediate outputs
test_input = torch.randn(1, 3, 32, 32)
final_out, intermediate_outs = utility_model.get_layer_outputs(test_input, ['conv'])
print(f"\nIntermediate outputs: {list(intermediate_outs.keys())}")
if 'conv' in intermediate_outs:
    print(f"Conv output shape: {intermediate_outs['conv'].shape}")

print("\n=== nn.Module Best Practices ===")

print("nn.Module Best Practices:")
print("1. Always call super().__init__() first")
print("2. Register all parameters and buffers properly")
print("3. Implement forward() method")
print("4. Use meaningful names for submodules")
print("5. Handle train/eval modes appropriately")
print("6. Implement extra_repr() for better printing")
print("7. Use hooks sparingly and remove them when done")

print("\nParameter Management:")
print("- Use nn.Parameter for learnable parameters")
print("- Use register_buffer for non-learnable state")
print("- Initialize parameters in __init__ or reset_parameters")
print("- Use proper initialization schemes")

print("\nDevice Management:")
print("- Use .to(device) to move modules")
print("- Check parameter devices before operations")
print("- Ensure input and model are on same device")
print("- Use .cuda() and .cpu() for simple cases")

print("\nDebugging Tips:")
print("- Use hooks to inspect intermediate values")
print("- Print module structure with named_modules()")
print("- Check parameter counts and shapes")
print("- Verify gradient flow with requires_grad")
print("- Use model.eval() for inference")

print("\n=== nn.Module Fundamentals Complete ===") 