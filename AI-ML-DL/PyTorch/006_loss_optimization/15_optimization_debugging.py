#!/usr/bin/env python3
"""PyTorch Optimization Debugging - Debugging optimization issues"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt

print("=== Optimization Debugging Overview ===")

print("Common optimization issues:")
print("1. Loss not decreasing")
print("2. Loss exploding or becoming NaN")
print("3. Slow convergence")
print("4. Oscillating loss")
print("5. Gradient vanishing/exploding")
print("6. Learning rate issues")
print("7. Optimizer state problems")
print("8. Memory and performance issues")

print("\n=== Debugging Tools and Utilities ===")

class OptimizationDebugger:
    """Comprehensive optimization debugging utility"""
    
    def __init__(self, model, optimizer, track_gradients=True, track_weights=True):
        self.model = model
        self.optimizer = optimizer
        self.track_gradients = track_gradients
        self.track_weights = track_weights
        
        # Statistics tracking
        self.loss_history = []
        self.gradient_norms = defaultdict(list)
        self.weight_norms = defaultdict(list)
        self.learning_rates = []
        self.step_count = 0
        
        # Gradient flow analysis
        self.gradient_flow = defaultdict(list)
        
        # Register hooks if needed
        if track_gradients:
            self._register_gradient_hooks()
    
    def _register_gradient_hooks(self):
        """Register hooks to track gradients"""
        def grad_hook(name):
            def hook(grad):
                if grad is not None:
                    norm = grad.norm().item()
                    self.gradient_flow[name].append(norm)
                    
                    # Check for problematic gradients
                    if torch.isnan(grad).any():
                        print(f"WARNING: NaN gradient detected in {name}")
                    if torch.isinf(grad).any():
                        print(f"WARNING: Inf gradient detected in {name}")
                return grad
            return hook
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.register_hook(grad_hook(name))
    
    def record_step(self, loss):
        """Record statistics for current step"""
        self.step_count += 1
        self.loss_history.append(loss)
        
        # Record learning rates
        lrs = [group['lr'] for group in self.optimizer.param_groups]
        self.learning_rates.append(lrs[0] if lrs else 0)
        
        # Record gradient and weight norms
        if self.track_gradients or self.track_weights:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if self.track_gradients and param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        self.gradient_norms[name].append(grad_norm)
                    
                    if self.track_weights:
                        weight_norm = param.norm().item()
                        self.weight_norms[name].append(weight_norm)
    
    def diagnose_optimization(self):
        """Comprehensive optimization diagnosis"""
        print("=== Optimization Diagnosis ===")
        
        if len(self.loss_history) < 2:
            print("Not enough data for diagnosis")
            return
        
        # Loss analysis
        self._diagnose_loss()
        
        # Gradient analysis
        if self.gradient_norms:
            self._diagnose_gradients()
        
        # Weight analysis
        if self.weight_norms:
            self._diagnose_weights()
        
        # Learning rate analysis
        self._diagnose_learning_rate()
    
    def _diagnose_loss(self):
        """Diagnose loss behavior"""
        print("\n--- Loss Analysis ---")
        
        recent_losses = self.loss_history[-10:]
        loss_change = (recent_losses[-1] - recent_losses[0]) / max(abs(recent_losses[0]), 1e-8)
        
        # Check for common loss issues
        if any(np.isnan(loss) or np.isinf(loss) for loss in recent_losses):
            print("❌ CRITICAL: Loss contains NaN or Inf values")
            print("   Possible causes: Learning rate too high, gradient explosion, numerical instability")
        
        elif loss_change > 0.1:
            print("⚠️  WARNING: Loss is increasing")
            print("   Possible causes: Learning rate too high, optimizer issues, data problems")
        
        elif abs(loss_change) < 1e-6:
            print("⚠️  WARNING: Loss has plateaued")
            print("   Possible causes: Learning rate too low, local minimum, vanishing gradients")
        
        elif loss_change < -0.01:
            print("✅ GOOD: Loss is decreasing steadily")
        
        else:
            print("ℹ️  INFO: Loss is changing slowly")
        
        # Loss statistics
        print(f"   Current loss: {self.loss_history[-1]:.6f}")
        print(f"   Loss change (last 10 steps): {loss_change:.6f}")
        print(f"   Loss variance (last 10 steps): {np.var(recent_losses):.6f}")
    
    def _diagnose_gradients(self):
        """Diagnose gradient behavior"""
        print("\n--- Gradient Analysis ---")
        
        # Compute gradient statistics
        all_grad_norms = []
        for name, norms in self.gradient_norms.items():
            if norms:
                all_grad_norms.extend(norms[-5:])  # Last 5 steps
        
        if not all_grad_norms:
            print("No gradient data available")
            return
        
        avg_grad_norm = np.mean(all_grad_norms)
        max_grad_norm = np.max(all_grad_norms)
        min_grad_norm = np.min(all_grad_norms)
        
        # Gradient magnitude analysis
        if avg_grad_norm > 10:
            print("⚠️  WARNING: Large gradients detected (possible explosion)")
            print(f"   Average gradient norm: {avg_grad_norm:.6f}")
            print("   Suggestions: Reduce learning rate, use gradient clipping")
        
        elif avg_grad_norm < 1e-6:
            print("⚠️  WARNING: Very small gradients (possible vanishing)")
            print(f"   Average gradient norm: {avg_grad_norm:.6f}")
            print("   Suggestions: Increase learning rate, check activation functions")
        
        else:
            print("✅ GOOD: Gradient magnitudes are reasonable")
            print(f"   Average gradient norm: {avg_grad_norm:.6f}")
        
        print(f"   Min/Max gradient norm: {min_grad_norm:.6f} / {max_grad_norm:.6f}")
        
        # Per-layer gradient analysis
        print("   Per-layer gradient norms (last step):")
        for name, norms in self.gradient_norms.items():
            if norms:
                print(f"     {name:20}: {norms[-1]:.6f}")
    
    def _diagnose_weights(self):
        """Diagnose weight behavior"""
        print("\n--- Weight Analysis ---")
        
        # Weight update analysis
        print("   Weight norms (current):")
        weight_changes = {}
        
        for name, norms in self.weight_norms.items():
            if len(norms) >= 2:
                current_norm = norms[-1]
                prev_norm = norms[-2]
                change = abs(current_norm - prev_norm) / max(abs(prev_norm), 1e-8)
                weight_changes[name] = change
                print(f"     {name:20}: {current_norm:.6f} (change: {change:.6f})")
        
        # Check for problematic weight updates
        if weight_changes:
            avg_change = np.mean(list(weight_changes.values()))
            if avg_change > 0.1:
                print("⚠️  WARNING: Large weight changes detected")
                print("   Suggestions: Reduce learning rate, check gradient clipping")
            elif avg_change < 1e-8:
                print("⚠️  WARNING: Very small weight changes")
                print("   Suggestions: Increase learning rate, check gradients")
            else:
                print("✅ GOOD: Weight changes are reasonable")
    
    def _diagnose_learning_rate(self):
        """Diagnose learning rate"""
        print("\n--- Learning Rate Analysis ---")
        
        current_lr = self.learning_rates[-1] if self.learning_rates else 0
        print(f"   Current learning rate: {current_lr:.6f}")
        
        if len(self.learning_rates) > 1:
            lr_trend = self.learning_rates[-1] - self.learning_rates[0]
            if lr_trend != 0:
                print(f"   Learning rate trend: {'decreasing' if lr_trend < 0 else 'increasing'}")
        
        # Learning rate recommendations based on loss and gradients
        if self.loss_history and self.gradient_norms:
            recent_losses = self.loss_history[-5:]
            if len(recent_losses) > 1:
                loss_increasing = recent_losses[-1] > recent_losses[0]
                
                avg_grad_norm = np.mean([norm for norms in self.gradient_norms.values() 
                                       for norm in norms[-3:] if norms])
                
                if loss_increasing and avg_grad_norm > 1:
                    print("   💡 SUGGESTION: Consider reducing learning rate")
                elif not loss_increasing and avg_grad_norm < 1e-3:
                    print("   💡 SUGGESTION: Consider increasing learning rate")

# Test optimization debugger
print("Testing optimization debugger:")

class ProblematicModel(nn.Module):
    """Model designed to showcase optimization issues"""
    
    def __init__(self, add_problems=False):
        super().__init__()
        self.add_problems = add_problems
        
        # Deep network that might have gradient issues
        layers = []
        sizes = [20, 100, 100, 50, 10]
        
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                # Use problematic activation if specified
                if add_problems and i == 1:
                    layers.append(nn.Sigmoid())  # Can cause vanishing gradients
                else:
                    layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Create model and data
debug_model = ProblematicModel(add_problems=True)
sample_input = torch.randn(32, 20)
sample_target = torch.randint(0, 10, (32,))
loss_fn = nn.CrossEntropyLoss()

# Use problematic optimizer settings
optimizer = optim.SGD(debug_model.parameters(), lr=10.0)  # Too high learning rate

# Create debugger
debugger = OptimizationDebugger(debug_model, optimizer)

print("\nTraining with problematic settings:")
for step in range(10):
    optimizer.zero_grad()
    
    outputs = debug_model(sample_input)
    loss = loss_fn(outputs, sample_target)
    
    debugger.record_step(loss.item())
    
    loss.backward()
    optimizer.step()
    
    if step % 3 == 0:
        print(f"  Step {step}: Loss = {loss.item():.6f}")

# Run diagnosis
debugger.diagnose_optimization()

print("\n=== Common Optimization Problems and Solutions ===")

def debug_learning_rate_issues():
    """Debug learning rate related issues"""
    print("\n--- Learning Rate Issues ---")
    
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    input_data = torch.randn(16, 10)
    target = torch.randn(16, 1)
    loss_fn = nn.MSELoss()
    
    # Test different learning rates
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    
    for lr in learning_rates:
        print(f"\nTesting LR = {lr}")
        
        # Reset model
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
        optimizer = optim.SGD(model.parameters(), lr=lr)
        
        losses = []
        for step in range(5):
            optimizer.zero_grad()
            outputs = model(input_data)
            loss = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Analyze loss behavior
        loss_change = (losses[-1] - losses[0]) / max(abs(losses[0]), 1e-8)
        
        if any(np.isnan(l) or np.isinf(l) for l in losses):
            print(f"  ❌ LR {lr}: Loss became NaN/Inf (too high)")
        elif loss_change > 0.1:
            print(f"  ⚠️  LR {lr}: Loss increasing (possibly too high)")
        elif abs(loss_change) < 1e-6:
            print(f"  ⚠️  LR {lr}: Loss not changing (possibly too low)")
        else:
            print(f"  ✅ LR {lr}: Loss decreasing properly")

debug_learning_rate_issues()

def debug_gradient_flow():
    """Debug gradient flow issues"""
    print("\n--- Gradient Flow Issues ---")
    
    class DeepModel(nn.Module):
        def __init__(self, depth=10, use_good_init=False):
            super().__init__()
            layers = []
            for i in range(depth):
                linear = nn.Linear(50, 50)
                
                # Apply different initializations
                if use_good_init:
                    nn.init.xavier_normal_(linear.weight)
                else:
                    nn.init.normal_(linear.weight, 0, 1)  # Bad initialization
                
                layers.extend([linear, nn.ReLU()])
            
            layers.append(nn.Linear(50, 1))
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)
    
    # Test gradient flow with different initializations
    configs = [
        ("Bad initialization", False),
        ("Good initialization", True)
    ]
    
    for config_name, use_good_init in configs:
        print(f"\n{config_name}:")
        
        model = DeepModel(depth=8, use_good_init=use_good_init)
        input_data = torch.randn(16, 50)
        target = torch.randn(16, 1)
        
        # Compute gradients
        optimizer = optim.Adam(model.parameters())
        optimizer.zero_grad()
        
        outputs = model(input_data)
        loss = nn.MSELoss()(outputs, target)
        loss.backward()
        
        # Analyze gradient magnitudes by layer
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None and 'weight' in name:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
        
        if grad_norms:
            first_layer_grad = grad_norms[0]
            last_layer_grad = grad_norms[-1]
            grad_ratio = first_layer_grad / max(last_layer_grad, 1e-8)
            
            print(f"  First layer gradient norm: {first_layer_grad:.6f}")
            print(f"  Last layer gradient norm: {last_layer_grad:.6f}")
            print(f"  Gradient ratio (first/last): {grad_ratio:.6f}")
            
            if grad_ratio > 100:
                print("  ⚠️  Possible gradient explosion")
            elif grad_ratio < 0.01:
                print("  ⚠️  Possible gradient vanishing")
            else:
                print("  ✅ Gradient flow looks good")

debug_gradient_flow()

def debug_optimizer_state():
    """Debug optimizer state issues"""
    print("\n--- Optimizer State Issues ---")
    
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    
    # Test different optimizers
    optimizers_to_test = {
        'SGD': optim.SGD(model.parameters(), lr=0.01),
        'SGD+Momentum': optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        'Adam': optim.Adam(model.parameters(), lr=0.001),
        'AdamW': optim.AdamW(model.parameters(), lr=0.001)
    }
    
    input_data = torch.randn(16, 10)
    target = torch.randn(16, 1)
    loss_fn = nn.MSELoss()
    
    for opt_name, optimizer in optimizers_to_test.items():
        print(f"\nTesting {opt_name}:")
        
        # Reset model
        for param in model.parameters():
            nn.init.normal_(param, 0, 0.1)
        
        losses = []
        for step in range(10):
            optimizer.zero_grad()
            outputs = model(input_data)
            loss = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Analyze convergence
        initial_loss = losses[0]
        final_loss = losses[-1]
        improvement = (initial_loss - final_loss) / initial_loss
        
        print(f"  Initial loss: {initial_loss:.6f}")
        print(f"  Final loss: {final_loss:.6f}")
        print(f"  Improvement: {improvement:.3f}")
        
        if improvement > 0.5:
            print("  ✅ Good convergence")
        elif improvement > 0.1:
            print("  ⚠️  Slow convergence")
        else:
            print("  ❌ Poor convergence")

debug_optimizer_state()

print("\n=== Optimization Debugging Tools ===")

def check_gradient_flow(model, input_data, target, loss_fn):
    """Check gradient flow through the model"""
    model.zero_grad()
    
    outputs = model(input_data)
    loss = loss_fn(outputs, target)
    loss.backward()
    
    # Collect gradient information
    gradient_info = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradient_info[name] = {
                'norm': param.grad.norm().item(),
                'mean': param.grad.mean().item(),
                'std': param.grad.std().item(),
                'max': param.grad.max().item(),
                'min': param.grad.min().item()
            }
    
    return gradient_info

def find_optimal_learning_rate(model, input_data, target, loss_fn, 
                              min_lr=1e-6, max_lr=1e-1, num_iterations=100):
    """Find optimal learning rate using learning rate range test"""
    model_copy = type(model)()
    model_copy.load_state_dict(model.state_dict())
    
    lrs = torch.logspace(np.log10(min_lr), np.log10(max_lr), num_iterations)
    losses = []
    
    optimizer = optim.SGD(model_copy.parameters(), lr=min_lr)
    
    for lr in lrs:
        # Set learning rate
        for group in optimizer.param_groups:
            group['lr'] = lr.item()
        
        # Training step
        optimizer.zero_grad()
        outputs = model_copy(input_data)
        loss = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Stop if loss explodes
        if len(losses) > 10 and loss.item() > 4 * min(losses):
            break
    
    # Find learning rate with steepest descent
    gradients = np.gradient(losses)
    best_lr_idx = np.argmin(gradients)
    
    return lrs[best_lr_idx].item(), losses

# Test debugging tools
print("\nTesting gradient flow checker:")
test_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
test_input = torch.randn(16, 10)
test_target = torch.randn(16, 1)

grad_info = check_gradient_flow(test_model, test_input, test_target, nn.MSELoss())
for name, info in grad_info.items():
    print(f"  {name}: norm={info['norm']:.6f}, mean={info['mean']:.6f}")

print("\nTesting learning rate finder:")
optimal_lr, lr_losses = find_optimal_learning_rate(test_model, test_input, test_target, nn.MSELoss())
print(f"  Suggested learning rate: {optimal_lr:.6f}")
print(f"  Loss range: {min(lr_losses):.6f} - {max(lr_losses):.6f}")

print("\n=== Optimization Debugging Best Practices ===")

print("Debugging Workflow:")
print("1. Start with simple baselines")
print("2. Monitor loss, gradients, and weights")
print("3. Check for numerical issues (NaN, Inf)")
print("4. Validate data and preprocessing")
print("5. Test different optimizers and learning rates")

print("\nEssential Monitoring:")
print("1. Loss curves (training and validation)")
print("2. Gradient norms per layer")
print("3. Weight update magnitudes")
print("4. Learning rate schedules")
print("5. Optimizer state statistics")

print("\nCommon Issues and Solutions:")
print("1. Loss not decreasing: Check LR, gradients, data")
print("2. Loss exploding: Reduce LR, clip gradients")
print("3. Slow convergence: Increase LR, change optimizer")
print("4. Oscillating loss: Reduce LR, add momentum")
print("5. Gradient vanishing: Check activations, initialization")

print("\nDebugging Tools:")
print("1. Gradient flow visualization")
print("2. Learning rate range tests")
print("3. Weight histogram analysis")
print("4. Loss landscape visualization")
print("5. Optimizer state inspection")

print("\nPreventive Measures:")
print("1. Proper weight initialization")
print("2. Gradient clipping")
print("3. Learning rate scheduling")
print("4. Batch normalization")
print("5. Skip connections in deep networks")

print("\n=== Optimization Debugging Complete ===")

# Memory cleanup
del debug_model, sample_input, sample_target