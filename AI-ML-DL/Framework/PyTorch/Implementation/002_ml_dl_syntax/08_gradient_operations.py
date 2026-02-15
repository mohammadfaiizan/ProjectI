#!/usr/bin/env python3
"""PyTorch Gradient Operations - Gradient computation, manipulation, zeroing"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== Basic Gradient Operations ===")

# Create tensors with gradient tracking
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = torch.tensor([1.0, 4.0], requires_grad=True)

print(f"x requires_grad: {x.requires_grad}")
print(f"y requires_grad: {y.requires_grad}")

# Forward computation
z = x * y + x**2
loss = z.sum()

print(f"z: {z}")
print(f"loss: {loss}")

# Backward pass
loss.backward()

print(f"x.grad: {x.grad}")
print(f"y.grad: {y.grad}")

print("\n=== Manual Gradient Computation ===")

# Manual gradient computation using autograd.grad
x_manual = torch.tensor([2.0, 3.0], requires_grad=True)
y_manual = torch.tensor([1.0, 4.0], requires_grad=True)

z_manual = x_manual * y_manual + x_manual**2
loss_manual = z_manual.sum()

# Compute gradients manually
grad_x, grad_y = torch.autograd.grad(loss_manual, [x_manual, y_manual])

print(f"Manual grad_x: {grad_x}")
print(f"Manual grad_y: {grad_y}")

# Gradient with respect to intermediate variables
grad_z = torch.autograd.grad(loss_manual, z_manual, retain_graph=True)[0]
print(f"Gradient w.r.t z: {grad_z}")

print("\n=== Gradient Accumulation ===")

# Multiple backward passes accumulate gradients
x_accum = torch.tensor([1.0, 2.0], requires_grad=True)

# First computation
loss1 = (x_accum**2).sum()
loss1.backward()
print(f"After first backward: {x_accum.grad}")

# Second computation (gradients accumulate)
loss2 = (x_accum**3).sum()
loss2.backward()
print(f"After second backward: {x_accum.grad}")

# Zero gradients manually
x_accum.grad.zero_()
print(f"After zero_grad(): {x_accum.grad}")

print("\n=== Gradient Manipulation ===")

# Gradient clipping
x_clip = torch.randn(5, requires_grad=True)
y_clip = (x_clip**4).sum()
y_clip.backward()

print(f"Original gradient: {x_clip.grad}")

# Clip gradient norm
torch.nn.utils.clip_grad_norm_([x_clip], max_norm=1.0)
print(f"After clipping: {x_clip.grad}")

# Gradient scaling
x_scale = torch.randn(3, requires_grad=True)
y_scale = (x_scale**2).sum()
y_scale.backward()

original_grad = x_scale.grad.clone()
x_scale.grad *= 0.5  # Scale gradients
print(f"Original: {original_grad}")
print(f"Scaled: {x_scale.grad}")

print("\n=== Higher-Order Gradients ===")

# Second-order gradients
x_second = torch.tensor([2.0], requires_grad=True)
y_second = x_second**3

# First-order gradient
grad_first = torch.autograd.grad(y_second, x_second, create_graph=True)[0]
print(f"First-order gradient: {grad_first}")

# Second-order gradient
grad_second = torch.autograd.grad(grad_first, x_second)[0]
print(f"Second-order gradient: {grad_second}")

# Using higher-order derivatives
def second_derivative(func, x):
    """Compute second derivative of func at x"""
    x = x.requires_grad_(True)
    y = func(x)
    grad_first = torch.autograd.grad(y, x, create_graph=True)[0]
    grad_second = torch.autograd.grad(grad_first, x)[0]
    return grad_second

x_test = torch.tensor([2.0])
second_deriv = second_derivative(lambda x: x**4, x_test)
print(f"Second derivative of x^4 at x=2: {second_deriv}")

print("\n=== Gradient Context Managers ===")

# torch.no_grad() context
x_no_grad = torch.tensor([1.0, 2.0], requires_grad=True)

with torch.no_grad():
    y_no_grad = x_no_grad**2
    print(f"Inside no_grad - y requires_grad: {y_no_grad.requires_grad}")

# torch.enable_grad() context
with torch.no_grad():
    with torch.enable_grad():
        x_enable = torch.tensor([1.0, 2.0], requires_grad=True)
        y_enable = x_enable**2
        print(f"Inside enable_grad - y requires_grad: {y_enable.requires_grad}")

# Temporarily disable gradients
x_temp = torch.tensor([1.0, 2.0], requires_grad=True)
x_temp.requires_grad_(False)
y_temp = x_temp**2
print(f"After requires_grad_(False): {y_temp.requires_grad}")

x_temp.requires_grad_(True)
y_temp2 = x_temp**2
print(f"After requires_grad_(True): {y_temp2.requires_grad}")

print("\n=== Gradient Flow Analysis ===")

# Analyze gradient flow in neural networks
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 10)
        self.layer3 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

model = SimpleNet()
input_data = torch.randn(32, 10)
target = torch.randn(32, 1)

# Forward pass
output = model(input_data)
loss = F.mse_loss(output, target)

# Backward pass
loss.backward()

# Analyze gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm()
        print(f"{name:>15} grad norm: {grad_norm:.6f}")

print("\n=== Custom Gradient Functions ===")

# Custom autograd function
class SquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input**2
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = 2 * input * grad_output
        return grad_input

# Use custom function
square_func = SquareFunction.apply
x_custom = torch.tensor([3.0, 4.0], requires_grad=True)
y_custom = square_func(x_custom)
y_custom.sum().backward()

print(f"Custom function input: {x_custom}")
print(f"Custom function output: {y_custom}")
print(f"Custom function gradient: {x_custom.grad}")

print("\n=== Gradient Hooks ===")

# Register hooks to monitor gradients
x_hook = torch.tensor([1.0, 2.0], requires_grad=True)

def gradient_hook(grad):
    print(f"Gradient hook called: {grad}")
    return grad * 2  # Modify gradient

# Register hook
hook_handle = x_hook.register_hook(gradient_hook)

y_hook = (x_hook**3).sum()
y_hook.backward()

print(f"Final gradient with hook: {x_hook.grad}")

# Remove hook
hook_handle.remove()

print("\n=== Module Hooks for Gradients ===")

# Hook for module gradients
class GradientMonitor:
    def __init__(self):
        self.gradients = {}
    
    def __call__(self, module, grad_input, grad_output):
        module_name = module.__class__.__name__
        if grad_output[0] is not None:
            self.gradients[module_name] = grad_output[0].norm().item()

# Register hooks on model
monitor = GradientMonitor()
hooks = []

for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        hook = module.register_backward_hook(monitor)
        hooks.append(hook)

# Run forward and backward
model.zero_grad()
output = model(input_data)
loss = F.mse_loss(output, target)
loss.backward()

print("Module gradient norms:")
for module_name, grad_norm in monitor.gradients.items():
    print(f"{module_name:>10}: {grad_norm:.6f}")

# Clean up hooks
for hook in hooks:
    hook.remove()

print("\n=== Gradient Checkpointing ===")

# Gradient checkpointing for memory efficiency
def checkpoint_forward(func, *args):
    """Simple gradient checkpointing implementation"""
    class CheckpointFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *inputs):
            ctx.func = func
            with torch.no_grad():
                outputs = func(*inputs)
            ctx.save_for_backward(*inputs)
            return outputs
        
        @staticmethod
        def backward(ctx, *grad_outputs):
            inputs = ctx.saved_tensors
            with torch.enable_grad():
                inputs = [inp.detach().requires_grad_(True) for inp in inputs]
                outputs = ctx.func(*inputs)
            
            return torch.autograd.grad(outputs, inputs, grad_outputs)
    
    return CheckpointFunction.apply(*args)

# Example usage of checkpointing
def expensive_operation(x):
    return torch.sin(x).exp().cos()

x_checkpoint = torch.randn(1000, 1000, requires_grad=True)
y_checkpoint = checkpoint_forward(expensive_operation, x_checkpoint)
loss_checkpoint = y_checkpoint.sum()
loss_checkpoint.backward()

print(f"Checkpoint gradient computed, shape: {x_checkpoint.grad.shape}")

print("\n=== Gradient Debugging ===")

# Debug gradient computation
def debug_gradients(model, input_data, target):
    """Debug gradient flow in model"""
    model.zero_grad()
    
    # Forward pass
    output = model(input_data)
    loss = F.mse_loss(output, target)
    
    # Backward pass
    loss.backward()
    
    # Check for gradient issues
    gradient_issues = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            
            # Check for NaN gradients
            if torch.isnan(grad).any():
                gradient_issues[name] = "NaN gradients"
            
            # Check for exploding gradients
            elif grad.norm() > 10.0:
                gradient_issues[name] = f"Large gradient norm: {grad.norm():.2f}"
            
            # Check for vanishing gradients
            elif grad.norm() < 1e-6:
                gradient_issues[name] = f"Small gradient norm: {grad.norm():.2e}"
    
    return gradient_issues

# Debug the model
issues = debug_gradients(model, input_data, target)
if issues:
    print("Gradient issues found:")
    for param_name, issue in issues.items():
        print(f"  {param_name}: {issue}")
else:
    print("No gradient issues detected")

print("\n=== Gradient Accumulation for Large Batches ===")

# Simulate large batch training with gradient accumulation
def train_with_gradient_accumulation(model, data_loader, optimizer, accumulation_steps=4):
    """Train with gradient accumulation"""
    model.train()
    optimizer.zero_grad()
    
    accumulated_loss = 0
    for i, (data, target) in enumerate(data_loader):
        # Forward pass
        output = model(data)
        loss = F.mse_loss(output, target) / accumulation_steps
        
        # Backward pass
        loss.backward()
        accumulated_loss += loss.item()
        
        # Update parameters every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    return accumulated_loss

# Simulate data loader
class FakeDataLoader:
    def __init__(self, batch_size, num_batches):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.current = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current < self.num_batches:
            self.current += 1
            data = torch.randn(self.batch_size, 10)
            target = torch.randn(self.batch_size, 1)
            return data, target
        else:
            self.current = 0
            raise StopIteration

# Test gradient accumulation
fake_loader = FakeDataLoader(batch_size=8, num_batches=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

accumulated_loss = train_with_gradient_accumulation(model, fake_loader, optimizer, accumulation_steps=2)
print(f"Accumulated loss: {accumulated_loss:.6f}")

print("\n=== Gradient Operations Best Practices ===")

print("Gradient Management Guidelines:")
print("1. Always call optimizer.zero_grad() before backward()")
print("2. Use gradient clipping for RNNs and deep networks")
print("3. Monitor gradient norms during training")
print("4. Use torch.no_grad() for inference to save memory")
print("5. Consider gradient accumulation for large effective batch sizes")
print("6. Use gradient checkpointing for memory-constrained training")
print("7. Debug NaN and exploding gradients early")

print("\nCommon Issues:")
print("- Forgetting to zero gradients (accumulation)")
print("- Not using retain_graph=True when needed")
print("- Memory issues with large computation graphs")
print("- Gradient explosion in deep networks")
print("- Vanishing gradients in very deep networks")

print("\nDebugging Tips:")
print("- Check gradient norms regularly")
print("- Use gradient hooks for monitoring")
print("- Verify gradients with finite differences")
print("- Use smaller learning rates if gradients explode")
print("- Consider residual connections for deep networks")

print("\n=== Gradient Operations Complete ===") 