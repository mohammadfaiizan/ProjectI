#!/usr/bin/env python3
"""PyTorch Custom Backward Hooks - Implementing custom backward hooks"""

import torch
import torch.nn as nn

print("=== Custom Backward Hooks Overview ===")

print("Backward hooks enable:")
print("1. Gradient monitoring and logging")
print("2. Gradient modification during backprop")
print("3. Debugging gradient flow")
print("4. Custom gradient processing")
print("5. Gradient-based optimization tricks")

print("\n=== Basic Tensor Hooks ===")

# Simple tensor hook
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

def simple_hook(grad):
    print(f"Gradient received: {grad}")
    # Hook can modify gradient by returning new value
    return grad * 2  # Double the gradient

# Register hook
hook_handle = x.register_hook(simple_hook)

# Forward and backward
y = (x**2).sum()
y.backward()

print(f"Original x: {x}")
print(f"Modified gradient: {x.grad}")

# Remove hook
hook_handle.remove()

print("\n=== Module Backward Hooks ===")

class SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 3)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

model = SimpleModule()

# Hook function for modules
def module_backward_hook(module, grad_input, grad_output):
    """
    grad_input: gradients w.r.t. inputs
    grad_output: gradients w.r.t. outputs
    """
    print(f"Module: {module.__class__.__name__}")
    print(f"Grad output shape: {grad_output[0].shape if grad_output[0] is not None else None}")
    print(f"Grad input shape: {grad_input[0].shape if grad_input[0] is not None else None}")
    
    # Can modify gradients by returning tuple
    # return modified_grad_input_tuple

# Register hook on linear layer
hook_linear = model.linear.register_backward_hook(module_backward_hook)

# Test the hook
input_data = torch.randn(4, 5, requires_grad=True)
output = model(input_data)
loss = output.sum()
loss.backward()

# Clean up
hook_linear.remove()

print("\n=== Gradient Monitoring Hooks ===")

class GradientMonitor:
    """Monitor gradients using hooks"""
    
    def __init__(self):
        self.gradients = {}
        self.hooks = []
    
    def register_tensor_hook(self, name, tensor):
        """Register hook for a tensor"""
        def hook_fn(grad):
            self.gradients[name] = {
                'norm': grad.norm().item(),
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'shape': grad.shape
            }
            return grad
        
        hook = tensor.register_hook(hook_fn)
        self.hooks.append(hook)
        return hook
    
    def register_module_hook(self, name, module):
        """Register hook for a module"""
        def hook_fn(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients[f"{name}_output"] = {
                    'norm': grad_output[0].norm().item(),
                    'mean': grad_output[0].mean().item(),
                    'std': grad_output[0].std().item(),
                    'shape': grad_output[0].shape
                }
        
        hook = module.register_backward_hook(hook_fn)
        self.hooks.append(hook)
        return hook
    
    def get_stats(self):
        """Get gradient statistics"""
        return self.gradients.copy()
    
    def clear(self):
        """Clear recorded gradients"""
        self.gradients.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

# Test gradient monitor
monitor = GradientMonitor()
model_monitor = SimpleModule()

# Register hooks
monitor.register_module_hook('linear', model_monitor.linear)
monitor.register_module_hook('relu', model_monitor.activation)

# Forward and backward
input_mon = torch.randn(8, 5, requires_grad=True)
output_mon = model_monitor(input_mon)
loss_mon = output_mon.sum()
loss_mon.backward()

# Check gradient statistics
stats = monitor.get_stats()
for name, stat in stats.items():
    print(f"{name}: norm={stat['norm']:.4f}, mean={stat['mean']:.4f}")

monitor.remove_hooks()

print("\n=== Gradient Modification Hooks ===")

class GradientClipper:
    """Clip gradients using hooks"""
    
    def __init__(self, max_norm=1.0):
        self.max_norm = max_norm
        self.hooks = []
    
    def clip_hook(self, grad):
        """Hook function that clips gradients"""
        grad_norm = grad.norm()
        if grad_norm > self.max_norm:
            # Clip gradient
            clipped_grad = grad * (self.max_norm / grad_norm)
            print(f"Gradient clipped: {grad_norm:.4f} -> {self.max_norm:.4f}")
            return clipped_grad
        return grad
    
    def register_model(self, model):
        """Register clipping hooks for all parameters"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(self.clip_hook)
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

# Test gradient clipping hooks
clipper = GradientClipper(max_norm=0.5)
model_clip = SimpleModule()

# Initialize with large weights to cause large gradients
for param in model_clip.parameters():
    nn.init.normal_(param, std=2.0)

clipper.register_model(model_clip)

# Training step that would produce large gradients
input_clip = torch.randn(4, 5) * 5  # Large input
output_clip = model_clip(input_clip)
loss_clip = output_clip.sum()
loss_clip.backward()

clipper.remove_hooks()

print("\n=== Gradient Accumulation Hooks ===")

class GradientAccumulator:
    """Accumulate gradients using hooks"""
    
    def __init__(self, accumulation_steps=4):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        self.accumulated_grads = {}
        self.hooks = []
    
    def accumulation_hook(self, name):
        """Create accumulation hook for specific parameter"""
        def hook_fn(grad):
            if name not in self.accumulated_grads:
                self.accumulated_grads[name] = torch.zeros_like(grad)
            
            # Accumulate gradients
            self.accumulated_grads[name] += grad / self.accumulation_steps
            
            # Return accumulated gradient when ready
            if self.current_step == self.accumulation_steps - 1:
                accumulated = self.accumulated_grads[name].clone()
                self.accumulated_grads[name].zero_()
                return accumulated
            else:
                # Return zero gradient for intermediate steps
                return torch.zeros_like(grad)
        
        return hook_fn
    
    def register_model(self, model):
        """Register accumulation hooks"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(self.accumulation_hook(name))
                self.hooks.append(hook)
    
    def step(self):
        """Increment accumulation step"""
        self.current_step = (self.current_step + 1) % self.accumulation_steps
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

# Test gradient accumulation hooks
accumulator = GradientAccumulator(accumulation_steps=3)
model_accum = SimpleModule()
optimizer_accum = torch.optim.SGD(model_accum.parameters(), lr=0.01)

accumulator.register_model(model_accum)

print("Gradient accumulation with hooks:")
for step in range(6):
    optimizer_accum.zero_grad()
    
    # Forward pass
    input_accum = torch.randn(4, 5)
    output_accum = model_accum(input_accum)
    loss_accum = output_accum.sum()
    
    # Backward pass
    loss_accum.backward()
    
    # Check if gradients are non-zero (indicates accumulation complete)
    total_grad_norm = sum(p.grad.norm().item() for p in model_accum.parameters() if p.grad is not None)
    print(f"Step {step}: total grad norm = {total_grad_norm:.6f}")
    
    # Update accumulator step
    accumulator.step()
    
    # Only step optimizer when gradients are accumulated
    if total_grad_norm > 0:
        optimizer_accum.step()

accumulator.remove_hooks()

print("\n=== Debugging Hooks ===")

class GradientDebugger:
    """Debug gradient flow using hooks"""
    
    def __init__(self):
        self.gradient_flow = []
        self.hooks = []
    
    def debug_hook(self, name):
        """Create debugging hook"""
        def hook_fn(grad):
            # Check for problematic gradients
            issues = []
            
            if torch.isnan(grad).any():
                issues.append("NaN")
            
            if torch.isinf(grad).any():
                issues.append("Inf")
            
            if grad.norm() > 1000:
                issues.append(f"Large norm: {grad.norm():.2e}")
            
            if grad.norm() < 1e-8:
                issues.append(f"Small norm: {grad.norm():.2e}")
            
            self.gradient_flow.append({
                'name': name,
                'shape': grad.shape,
                'norm': grad.norm().item(),
                'mean': grad.mean().item(),
                'issues': issues
            })
            
            if issues:
                print(f"WARNING in {name}: {', '.join(issues)}")
            
            return grad
        
        return hook_fn
    
    def register_model(self, model):
        """Register debugging hooks for all parameters"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(self.debug_hook(name))
                self.hooks.append(hook)
    
    def get_report(self):
        """Get debugging report"""
        return self.gradient_flow.copy()
    
    def clear(self):
        """Clear gradient flow history"""
        self.gradient_flow.clear()
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

# Test debugging hooks
debugger = GradientDebugger()
model_debug = SimpleModule()

# Create problematic scenario
for param in model_debug.parameters():
    nn.init.normal_(param, std=5.0)  # Large initialization

debugger.register_model(model_debug)

# Forward and backward that might cause issues
input_debug = torch.randn(4, 5) * 10
output_debug = model_debug(input_debug)
loss_debug = output_debug.sum()
loss_debug.backward()

# Get debug report
report = debugger.get_report()
print("\nGradient debug report:")
for entry in report:
    print(f"  {entry['name']}: norm={entry['norm']:.4f}, issues={entry['issues']}")

debugger.remove_hooks()

print("\n=== Hook-based Gradient Surgery ===")

class GradientSurgeon:
    """Perform gradient surgery using hooks"""
    
    def __init__(self):
        self.hooks = []
        self.surgery_rules = {}
    
    def add_rule(self, param_name, rule_fn):
        """Add surgery rule for specific parameter"""
        self.surgery_rules[param_name] = rule_fn
    
    def surgery_hook(self, param_name):
        """Create surgery hook"""
        def hook_fn(grad):
            if param_name in self.surgery_rules:
                return self.surgery_rules[param_name](grad)
            return grad
        
        return hook_fn
    
    def register_model(self, model):
        """Register surgery hooks"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(self.surgery_hook(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

# Test gradient surgery
surgeon = GradientSurgeon()
model_surgery = SimpleModule()

# Define surgery rules
def bias_surgery(grad):
    """Zero out bias gradients"""
    return torch.zeros_like(grad)

def weight_surgery(grad):
    """Apply L2 regularization to weights"""
    return grad + 0.01 * grad  # L2 penalty

# Add rules
surgeon.add_rule('linear.bias', bias_surgery)
surgeon.add_rule('linear.weight', weight_surgery)

surgeon.register_model(model_surgery)

# Test surgery
input_surgery = torch.randn(4, 5)
output_surgery = model_surgery(input_surgery)
loss_surgery = output_surgery.sum()
loss_surgery.backward()

print("Gradient surgery applied:")
for name, param in model_surgery.named_parameters():
    if param.grad is not None:
        print(f"  {name}: grad norm = {param.grad.norm():.6f}")

surgeon.remove_hooks()

print("\n=== Hook Best Practices ===")

print("Backward Hook Guidelines:")
print("1. Always remove hooks when done to prevent memory leaks")
print("2. Be careful with gradient modifications")
print("3. Use hooks for monitoring, not essential logic")
print("4. Handle None gradients in module hooks")
print("5. Keep hook functions lightweight")
print("6. Store hook handles for proper cleanup")
print("7. Consider hook execution order")

print("\nCommon Use Cases:")
print("- Gradient monitoring and logging")
print("- Gradient clipping and normalization")
print("- Debugging gradient flow")
print("- Implementing custom regularization")
print("- Gradient accumulation strategies")
print("- Research and experimentation")

print("\nPerformance Considerations:")
print("- Hooks add computational overhead")
print("- Avoid heavy computations in hooks")
print("- Remove unused hooks")
print("- Use hooks sparingly in production")
print("- Consider alternatives for performance-critical code")

print("\n=== Custom Backward Hooks Complete ===") 