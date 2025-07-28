#!/usr/bin/env python3
"""PyTorch Autograd Anomaly Detection - Debugging gradient issues"""

import torch
import torch.nn as nn
import warnings

print("=== Autograd Anomaly Detection Overview ===")

print("Autograd anomaly detection helps find:")
print("1. NaN gradients")
print("2. Infinite gradients") 
print("3. Gradient computation errors")
print("4. Backward pass issues")
print("5. Memory access violations")

print("\n=== Basic Anomaly Detection ===")

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

try:
    # Create computation that will produce NaN
    x = torch.tensor([1.0, 0.0], requires_grad=True)
    y = torch.log(x)  # log(0) = -inf
    z = torch.exp(y)  # exp(-inf) = 0
    w = 1.0 / z      # 1/0 = inf
    
    print(f"x: {x}")
    print(f"y: {y}")
    print(f"z: {z}")
    print(f"w: {w}")
    
    # This will trigger anomaly detection
    loss = w.sum()
    loss.backward()
    
except RuntimeError as e:
    print(f"Anomaly detected: {str(e)[:100]}...")

# Disable anomaly detection
torch.autograd.set_detect_anomaly(False)

print("\n=== Context Manager for Anomaly Detection ===")

# Using context manager for specific operations
with torch.autograd.detect_anomaly():
    try:
        x_context = torch.tensor([2.0, -1.0], requires_grad=True)
        y_context = torch.sqrt(x_context)  # sqrt(-1) = NaN
        loss_context = y_context.sum()
        loss_context.backward()
    except RuntimeError as e:
        print(f"Context anomaly detected: {str(e)[:100]}...")

print("\n=== Identifying NaN Sources ===")

def create_nan_scenario():
    """Create a scenario that produces NaN gradients"""
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    # Division by zero scenario
    y = x / (x - 2.0)  # Will have inf when x=2
    z = y * 0          # inf * 0 = NaN
    
    return x, z

# Monitor for NaN without anomaly detection
x_nan, z_nan = create_nan_scenario()
loss_nan = z_nan.sum()

# Check for NaN before backward
if torch.isnan(loss_nan):
    print("NaN detected in loss before backward")
else:
    loss_nan.backward()
    
    # Check gradients for NaN
    if torch.isnan(x_nan.grad).any():
        print("NaN detected in gradients")
        print(f"NaN locations: {torch.isnan(x_nan.grad)}")

print("\n=== Debugging Infinite Gradients ===")

def create_inf_scenario():
    """Create scenario with infinite gradients"""
    x = torch.tensor([0.1, 0.01, 0.001], requires_grad=True)
    
    # Exponential explosion
    y = torch.exp(1.0 / x)  # exp(1/0.001) = exp(1000) = inf
    
    return x, y

x_inf, y_inf = create_inf_scenario()

# Check for infinite values
if torch.isinf(y_inf).any():
    print("Infinite values detected in forward pass")
    print(f"Infinite locations: {torch.isinf(y_inf)}")
    
    # Safe backward with anomaly detection
    with torch.autograd.detect_anomaly():
        try:
            loss_inf = y_inf.sum()
            loss_inf.backward()
        except RuntimeError as e:
            print(f"Infinite gradient error: {str(e)[:100]}...")

print("\n=== Advanced Debugging Techniques ===")

class DebuggingHook:
    """Hook for monitoring gradients during backward pass"""
    
    def __init__(self, name):
        self.name = name
        self.gradients = []
    
    def __call__(self, grad):
        # Store gradient statistics
        self.gradients.append({
            'mean': grad.mean().item(),
            'std': grad.std().item(),
            'max': grad.max().item(),
            'min': grad.min().item(),
            'has_nan': torch.isnan(grad).any().item(),
            'has_inf': torch.isinf(grad).any().item()
        })
        
        # Print warnings for problematic gradients
        if torch.isnan(grad).any():
            print(f"WARNING: NaN gradient in {self.name}")
        
        if torch.isinf(grad).any():
            print(f"WARNING: Infinite gradient in {self.name}")
        
        if grad.abs().max() > 1000:
            print(f"WARNING: Large gradient in {self.name}: {grad.abs().max():.2e}")
        
        return grad

# Test debugging hooks
class ProblematicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 1)
        # Initialize with problematic weights
        nn.init.constant_(self.fc1.weight, 10.0)  # Large weights
        nn.init.constant_(self.fc2.weight, 10.0)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net_debug = ProblematicNet()

# Register hooks
hook1 = DebuggingHook('fc1')
hook2 = DebuggingHook('fc2')

net_debug.fc1.weight.register_hook(hook1)
net_debug.fc2.weight.register_hook(hook2)

# Forward and backward pass
input_debug = torch.randn(3, 5) * 10  # Large input
output_debug = net_debug(input_debug)
loss_debug = output_debug.sum()

# This might produce large gradients
loss_debug.backward()

print(f"FC1 gradient stats: {hook1.gradients[-1] if hook1.gradients else 'No gradients'}")
print(f"FC2 gradient stats: {hook2.gradients[-1] if hook2.gradients else 'No gradients'}")

print("\n=== Model-Level Debugging ===")

def debug_model_gradients(model, threshold=1000):
    """Debug all gradients in a model"""
    gradient_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            stats = {
                'shape': grad.shape,
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'norm': grad.norm().item(),
                'max_abs': grad.abs().max().item(),
                'has_nan': torch.isnan(grad).any().item(),
                'has_inf': torch.isinf(grad).any().item(),
                'large_grad': (grad.abs() > threshold).any().item()
            }
            gradient_stats[name] = stats
    
    return gradient_stats

# Analyze model gradients
stats = debug_model_gradients(net_debug)
for name, stat in stats.items():
    print(f"\n{name}:")
    for key, value in stat.items():
        if key != 'shape':
            print(f"  {key}: {value}")

print("\n=== Gradient Explosion Detection ===")

def detect_gradient_explosion(model, threshold=1.0):
    """Detect exploding gradients"""
    total_norm = 0
    param_count = 0
    
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    total_norm = total_norm ** (1. / 2)
    
    return {
        'total_norm': total_norm,
        'is_exploding': total_norm > threshold,
        'param_count': param_count
    }

explosion_stats = detect_gradient_explosion(net_debug, threshold=10.0)
print(f"Gradient explosion stats: {explosion_stats}")

print("\n=== Vanishing Gradient Detection ===")

def detect_vanishing_gradients(model, threshold=1e-7):
    """Detect vanishing gradients"""
    vanishing_layers = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm < threshold:
                vanishing_layers.append((name, grad_norm))
    
    return vanishing_layers

# Test with a deeper network that might have vanishing gradients
class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for i in range(10):
            layers.extend([
                nn.Linear(50, 50),
                nn.Sigmoid()  # Sigmoid can cause vanishing gradients
            ])
        layers.append(nn.Linear(50, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

deep_net = DeepNet()
input_deep = torch.randn(5, 50)
output_deep = deep_net(input_deep)
loss_deep = output_deep.sum()
loss_deep.backward()

vanishing = detect_vanishing_gradients(deep_net)
if vanishing:
    print(f"Vanishing gradients detected in {len(vanishing)} layers:")
    for name, norm in vanishing[:3]:  # Show first 3
        print(f"  {name}: {norm:.2e}")

print("\n=== Custom Anomaly Detection ===")

class GradientAnomalyDetector:
    """Custom gradient anomaly detection"""
    
    def __init__(self, nan_threshold=0, inf_threshold=0, magnitude_threshold=1000):
        self.nan_threshold = nan_threshold
        self.inf_threshold = inf_threshold
        self.magnitude_threshold = magnitude_threshold
        self.anomalies = []
    
    def check_tensor(self, tensor, name="tensor"):
        """Check tensor for anomalies"""
        anomalies = []
        
        # Check for NaN
        nan_count = torch.isnan(tensor).sum().item()
        if nan_count > self.nan_threshold:
            anomalies.append(f"NaN values: {nan_count}")
        
        # Check for Inf
        inf_count = torch.isinf(tensor).sum().item()
        if inf_count > self.inf_threshold:
            anomalies.append(f"Inf values: {inf_count}")
        
        # Check for large magnitudes
        max_magnitude = tensor.abs().max().item()
        if max_magnitude > self.magnitude_threshold:
            anomalies.append(f"Large magnitude: {max_magnitude:.2e}")
        
        if anomalies:
            self.anomalies.append({
                'name': name,
                'anomalies': anomalies,
                'stats': {
                    'mean': tensor.mean().item(),
                    'std': tensor.std().item(),
                    'min': tensor.min().item(),
                    'max': tensor.max().item()
                }
            })
        
        return len(anomalies) == 0
    
    def report(self):
        """Report all detected anomalies"""
        if not self.anomalies:
            print("No anomalies detected")
            return
        
        print(f"Detected {len(self.anomalies)} anomalies:")
        for anomaly in self.anomalies:
            print(f"  {anomaly['name']}: {', '.join(anomaly['anomalies'])}")

# Test custom detector
detector = GradientAnomalyDetector(magnitude_threshold=100)

# Check model gradients
for name, param in net_debug.named_parameters():
    if param.grad is not None:
        detector.check_tensor(param.grad, name)

detector.report()

print("\n=== Practical Debugging Workflow ===")

def debug_training_step(model, data, target, criterion, optimizer):
    """Debug a single training step"""
    
    # Clear gradients
    optimizer.zero_grad()
    
    # Forward pass with anomaly detection
    with torch.autograd.detect_anomaly():
        try:
            output = model(data)
            
            # Check output for anomalies
            if torch.isnan(output).any():
                print("WARNING: NaN in model output")
                return False
            
            if torch.isinf(output).any():
                print("WARNING: Inf in model output")
                return False
            
            loss = criterion(output, target)
            
            # Check loss
            if torch.isnan(loss):
                print("WARNING: NaN loss")
                return False
            
            if torch.isinf(loss):
                print("WARNING: Inf loss")
                return False
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            max_grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    max_grad_norm = max(max_grad_norm, grad_norm)
                    
                    if torch.isnan(param.grad).any():
                        print("WARNING: NaN gradients detected")
                        return False
            
            print(f"Training step successful, max grad norm: {max_grad_norm:.4f}")
            return True
            
        except RuntimeError as e:
            print(f"Training step failed: {e}")
            return False

# Test debugging workflow
debug_model = nn.Sequential(
    nn.Linear(5, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

debug_data = torch.randn(3, 5)
debug_target = torch.randn(3, 1)
debug_criterion = nn.MSELoss()
debug_optimizer = torch.optim.SGD(debug_model.parameters(), lr=0.01)

success = debug_training_step(debug_model, debug_data, debug_target, 
                            debug_criterion, debug_optimizer)
print(f"Debug training step result: {success}")

print("\n=== Anomaly Detection Best Practices ===")

print("Gradient Debugging Guidelines:")
print("1. Enable anomaly detection during debugging only")
print("2. Check for NaN/Inf at each step")
print("3. Monitor gradient magnitudes")
print("4. Use hooks for detailed gradient tracking")
print("5. Implement custom anomaly detectors")
print("6. Test edge cases and extreme inputs")
print("7. Validate model architecture choices")

print("\nCommon Anomaly Sources:")
print("- Division by zero or near-zero values")
print("- Logarithm of negative numbers")
print("- Square root of negative numbers")
print("- Exponential overflow")
print("- Poor weight initialization")
print("- Inappropriate learning rates")
print("- Numerical instability in custom functions")

print("\nDebugging Strategies:")
print("- Start with small, simple examples")
print("- Add gradual complexity")
print("- Use gradient clipping as temporary fix")
print("- Check data preprocessing")
print("- Validate loss function implementation")
print("- Test with different optimizers")

print("\n=== Autograd Anomaly Detection Complete ===") 