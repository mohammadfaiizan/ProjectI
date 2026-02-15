#!/usr/bin/env python3
"""PyTorch Forward and Backward Pass Mechanics"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== Basic Forward Pass ===")

# Simple model for demonstration
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNet()
input_tensor = torch.randn(32, 10, requires_grad=True)

# Forward pass
output = model(input_tensor)
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
print(f"Output requires_grad: {output.requires_grad}")

print("\n=== Manual Forward Pass ===")

# Manual forward computation
x = input_tensor
print(f"Step 1 - Input: {x.shape}")

# First linear layer
w1 = model.fc1.weight
b1 = model.fc1.bias
z1 = torch.matmul(x, w1.t()) + b1
print(f"Step 2 - Linear 1: {z1.shape}")

# ReLU activation
a1 = torch.relu(z1)
print(f"Step 3 - ReLU: {a1.shape}")

# Second linear layer
w2 = model.fc2.weight
b2 = model.fc2.bias
z2 = torch.matmul(a1, w2.t()) + b2
print(f"Step 4 - Linear 2: {z2.shape}")

# Verify manual computation matches model
manual_output = z2
print(f"Manual equals model: {torch.allclose(manual_output, output)}")

print("\n=== Basic Backward Pass ===")

# Create loss
target = torch.randint(0, 5, (32,))
criterion = nn.CrossEntropyLoss()
loss = criterion(output, target)

print(f"Loss: {loss.item():.4f}")
print(f"Loss requires_grad: {loss.requires_grad}")

# Backward pass
loss.backward()

# Check gradients
print(f"Input gradient shape: {input_tensor.grad.shape}")
print(f"FC1 weight gradient shape: {model.fc1.weight.grad.shape}")
print(f"FC1 bias gradient shape: {model.fc1.bias.grad.shape}")

print("\n=== Gradient Flow Analysis ===")

# Analyze gradient magnitudes
def analyze_gradients(model):
    """Analyze gradient flow through model"""
    print("Gradient Analysis:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            param_norm = param.norm().item()
            relative_grad = grad_norm / param_norm if param_norm > 0 else 0
            print(f"  {name:>12}: grad_norm={grad_norm:.6f}, rel_grad={relative_grad:.6f}")
        else:
            print(f"  {name:>12}: No gradient")

analyze_gradients(model)

print("\n=== Intermediate Activations ===")

# Hook to capture intermediate activations
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks
model.fc1.register_forward_hook(get_activation('fc1'))
model.relu.register_forward_hook(get_activation('relu'))
model.fc2.register_forward_hook(get_activation('fc2'))

# Forward pass with hooks
model.zero_grad()
output_with_hooks = model(input_tensor)

print("Captured activations:")
for name, activation in activations.items():
    print(f"  {name}: {activation.shape}, mean={activation.mean():.4f}")

print("\n=== Custom Forward/Backward Functions ===")

class CustomSquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Save input for backward pass
        ctx.save_for_backward(input)
        return input.pow(2)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved input
        input, = ctx.saved_tensors
        # Compute gradient: d/dx(x^2) = 2x
        grad_input = 2 * input * grad_output
        return grad_input

# Test custom function
custom_square = CustomSquareFunction.apply
x_custom = torch.randn(5, requires_grad=True)
y_custom = custom_square(x_custom)
loss_custom = y_custom.sum()
loss_custom.backward()

print(f"Custom function input: {x_custom}")
print(f"Custom function output: {y_custom}")
print(f"Custom function gradient: {x_custom.grad}")

# Verify with analytical gradient (2x)
expected_grad = 2 * x_custom.detach()
print(f"Expected gradient: {expected_grad}")
print(f"Gradients match: {torch.allclose(x_custom.grad, expected_grad)}")

print("\n=== Multiple Backward Passes ===")

# Create computation graph
x_multi = torch.randn(3, requires_grad=True)
y_multi = x_multi.pow(2)
z_multi = y_multi.sum()

# First backward pass
grad_x = torch.autograd.grad(z_multi, x_multi, retain_graph=True)[0]
print(f"First backward - x_grad: {grad_x}")

# Second backward pass (higher order gradients)
grad_x_sum = grad_x.sum()
grad2_x = torch.autograd.grad(grad_x_sum, x_multi)[0]
print(f"Second backward - x_grad2: {grad2_x}")

print("\n=== Gradient Checkpointing ===")

def checkpoint_sequential(*functions):
    """Simplified gradient checkpointing for sequential functions"""
    def checkpoint_forward(input):
        # Forward pass without storing intermediate activations
        with torch.no_grad():
            for fn in functions:
                input = fn(input)
        return input
    
    class CheckpointFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.functions = functions
            ctx.save_for_backward(input)
            return checkpoint_forward(input)
        
        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            # Recompute forward pass with gradients
            current_input = input.detach().requires_grad_(True)
            
            with torch.enable_grad():
                for fn in ctx.functions:
                    current_input = fn(current_input)
            
            grad_input = torch.autograd.grad(current_input, input, grad_output)[0]
            return grad_input
    
    return CheckpointFunction.apply

# Test checkpointing
def layer1(x): return torch.relu(x @ torch.randn(10, 20))
def layer2(x): return torch.relu(x @ torch.randn(20, 15))
def layer3(x): return x @ torch.randn(15, 5)

checkpoint_model = checkpoint_sequential(layer1, layer2, layer3)
x_checkpoint = torch.randn(32, 10, requires_grad=True)
y_checkpoint = checkpoint_model(x_checkpoint)
loss_checkpoint = y_checkpoint.sum()
loss_checkpoint.backward()

print(f"Checkpointed forward: {y_checkpoint.shape}")
print(f"Checkpointed gradient: {x_checkpoint.grad.shape}")

print("\n=== Forward/Backward Hooks ===")

# Gradient hooks
gradient_norms = {}

def grad_hook(name):
    def hook(grad):
        gradient_norms[name] = grad.norm().item()
        return grad
    return hook

# Register gradient hooks
for name, param in model.named_parameters():
    if param.requires_grad:
        param.register_hook(grad_hook(name))

# Forward and backward with gradient hooks
model.zero_grad()
output_hook = model(input_tensor)
loss_hook = criterion(output_hook, target)
loss_hook.backward()

print("Gradient norms captured by hooks:")
for name, norm in gradient_norms.items():
    print(f"  {name}: {norm:.6f}")

print("\n=== Backward Pass Debugging ===")

def debug_backward_pass(model, input_data, target_data, criterion):
    """Debug backward pass for issues"""
    model.zero_grad()
    
    # Forward pass
    output = model(input_data)
    loss = criterion(output, target_data)
    
    print(f"Loss: {loss.item():.6f}")
    
    # Check for NaN in forward pass
    if torch.isnan(output).any():
        print("WARNING: NaN detected in forward pass output")
    
    if torch.isnan(loss):
        print("WARNING: NaN detected in loss")
    
    # Backward pass
    try:
        loss.backward()
    except RuntimeError as e:
        print(f"Backward pass error: {e}")
        return False
    
    # Check gradients
    gradient_issues = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                gradient_issues.append(f"No gradient for {name}")
            elif torch.isnan(param.grad).any():
                gradient_issues.append(f"NaN gradient for {name}")
            elif torch.isinf(param.grad).any():
                gradient_issues.append(f"Inf gradient for {name}")
            elif param.grad.norm() > 100:
                gradient_issues.append(f"Large gradient for {name}: {param.grad.norm():.2f}")
    
    if gradient_issues:
        print("Gradient issues found:")
        for issue in gradient_issues:
            print(f"  {issue}")
    else:
        print("No gradient issues detected")
    
    return True

# Test debugging
debug_success = debug_backward_pass(model, input_tensor, target, criterion)

print("\n=== Memory Efficient Forward/Backward ===")

class MemoryEfficientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(100, 100) for _ in range(10)
        ])
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Process in chunks to save memory
        for layer in self.layers:
            x = self.relu(layer(x))
            # Could add checkpointing here for deeper networks
        return x

# Memory monitoring
if torch.cuda.is_available():
    def get_memory_usage():
        return torch.cuda.memory_allocated() / 1e6  # MB
    
    mem_model = MemoryEfficientModel().cuda()
    mem_input = torch.randn(1000, 100, device='cuda', requires_grad=True)
    
    initial_memory = get_memory_usage()
    mem_output = mem_model(mem_input)
    forward_memory = get_memory_usage()
    
    mem_loss = mem_output.sum()
    mem_loss.backward()
    backward_memory = get_memory_usage()
    
    print(f"Memory usage:")
    print(f"  Initial: {initial_memory:.2f} MB")
    print(f"  After forward: {forward_memory:.2f} MB")
    print(f"  After backward: {backward_memory:.2f} MB")

print("\n=== Forward/Backward Best Practices ===")

print("Forward Pass Guidelines:")
print("1. Keep forward() method clean and readable")
print("2. Use torch.no_grad() for inference to save memory")
print("3. Handle different input shapes gracefully")
print("4. Add input validation for robustness")
print("5. Use hooks to monitor intermediate activations")

print("\nBackward Pass Guidelines:")
print("1. Always call zero_grad() before backward()")
print("2. Check for NaN/Inf gradients during debugging")
print("3. Use gradient clipping for stability")
print("4. Monitor gradient magnitudes for vanishing/exploding")
print("5. Use retain_graph=True for multiple backward passes")

print("\nMemory Optimization:")
print("- Use gradient checkpointing for deep networks")
print("- Clear intermediate variables when possible")
print("- Consider mixed precision training")
print("- Use torch.cuda.empty_cache() to free GPU memory")

print("\nDebugging Tips:")
print("- Register hooks to capture intermediate values")
print("- Use torch.autograd.detect_anomaly() for NaN detection")
print("- Print gradient norms to diagnose training issues")
print("- Visualize computation graph with torchviz")

print("\n=== Forward/Backward Pass Complete ===") 