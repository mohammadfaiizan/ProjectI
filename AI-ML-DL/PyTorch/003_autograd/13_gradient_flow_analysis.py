#!/usr/bin/env python3
"""PyTorch Gradient Flow Analysis - Analyzing gradient flow in networks"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

print("=== Gradient Flow Analysis Overview ===")

print("Gradient flow analysis helps detect:")
print("1. Vanishing gradients in deep networks")
print("2. Exploding gradients")
print("3. Dead neurons (zero gradients)")
print("4. Layer-wise gradient distribution")
print("5. Training bottlenecks")

print("\n=== Basic Gradient Flow Monitoring ===")

class GradientFlowTracker:
    """Track gradient flow through network layers"""
    
    def __init__(self):
        self.gradient_history = {}
        self.layer_names = []
    
    def track_model(self, model):
        """Start tracking gradients for a model"""
        self.layer_names = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.layer_names.append(name)
                if name not in self.gradient_history:
                    self.gradient_history[name] = []
    
    def record_gradients(self, model):
        """Record current gradient norms"""
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                self.gradient_history[name].append(grad_norm)
    
    def get_statistics(self):
        """Get gradient flow statistics"""
        stats = {}
        for name, norms in self.gradient_history.items():
            if norms:
                stats[name] = {
                    'mean': sum(norms) / len(norms),
                    'max': max(norms),
                    'min': min(norms),
                    'latest': norms[-1],
                    'count': len(norms)
                }
        return stats
    
    def clear_history(self):
        """Clear gradient history"""
        for name in self.gradient_history:
            self.gradient_history[name].clear()

# Test gradient flow tracker
class DeepNetwork(nn.Module):
    def __init__(self, depth=8):
        super().__init__()
        layers = []
        for i in range(depth):
            if i == 0:
                layers.append(nn.Linear(100, 50))
            elif i == depth - 1:
                layers.append(nn.Linear(50, 1))
            else:
                layers.append(nn.Linear(50, 50))
            
            if i < depth - 1:
                layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Create deep network and tracker
deep_model = DeepNetwork(depth=10)
tracker = GradientFlowTracker()
tracker.track_model(deep_model)

# Simulate training steps
optimizer = torch.optim.Adam(deep_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for step in range(10):
    # Generate data
    input_data = torch.randn(32, 100)
    target = torch.randn(32, 1)
    
    # Forward and backward
    optimizer.zero_grad()
    output = deep_model(input_data)
    loss = criterion(output, target)
    loss.backward()
    
    # Record gradients
    tracker.record_gradients(deep_model)
    
    optimizer.step()

# Analyze gradient flow
stats = tracker.get_statistics()
print("Gradient flow statistics (first 3 layers):")
for i, (name, stat) in enumerate(list(stats.items())[:3]):
    print(f"  {name}:")
    print(f"    Mean norm: {stat['mean']:.6f}")
    print(f"    Latest norm: {stat['latest']:.6f}")

print("\n=== Detecting Vanishing Gradients ===")

def detect_vanishing_gradients(model, threshold=1e-6):
    """Detect layers with vanishing gradients"""
    vanishing_layers = []
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm < threshold:
                vanishing_layers.append((name, grad_norm))
    
    return vanishing_layers

# Test with problematic network (many layers + sigmoid)
class VanishingGradientNet(nn.Module):
    def __init__(self, depth=15):
        super().__init__()
        layers = []
        for i in range(depth):
            if i == 0:
                layers.extend([nn.Linear(50, 50), nn.Sigmoid()])
            elif i == depth - 1:
                layers.append(nn.Linear(50, 1))
            else:
                layers.extend([nn.Linear(50, 50), nn.Sigmoid()])
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

vanishing_model = VanishingGradientNet(depth=12)
optimizer_van = torch.optim.SGD(vanishing_model.parameters(), lr=0.1)

# Single training step
input_van = torch.randn(16, 50)
target_van = torch.randn(16, 1)

optimizer_van.zero_grad()
output_van = vanishing_model(input_van)
loss_van = criterion(output_van, target_van)
loss_van.backward()

# Check for vanishing gradients
vanishing = detect_vanishing_gradients(vanishing_model, threshold=1e-4)
print(f"Detected {len(vanishing)} layers with vanishing gradients:")
for name, norm in vanishing[:3]:  # Show first 3
    print(f"  {name}: {norm:.2e}")

print("\n=== Detecting Exploding Gradients ===")

def detect_exploding_gradients(model, threshold=10.0):
    """Detect layers with exploding gradients"""
    exploding_layers = []
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > threshold:
                exploding_layers.append((name, grad_norm))
    
    return exploding_layers

# Test with problematic initialization
class ExplodingGradientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(50, 50) for _ in range(8)
        ])
        self.output = nn.Linear(50, 1)
        
        # Bad initialization - large weights
        for layer in self.layers:
            nn.init.normal_(layer.weight, std=3.0)
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)

exploding_model = ExplodingGradientNet()
optimizer_exp = torch.optim.SGD(exploding_model.parameters(), lr=0.1)

# Training step
input_exp = torch.randn(16, 50) * 2  # Large input
target_exp = torch.randn(16, 1)

optimizer_exp.zero_grad()
output_exp = exploding_model(input_exp)
loss_exp = criterion(output_exp, target_exp)
loss_exp.backward()

# Check for exploding gradients
exploding = detect_exploding_gradients(exploding_model, threshold=5.0)
print(f"Detected {len(exploding)} layers with exploding gradients:")
for name, norm in exploding[:3]:
    print(f"  {name}: {norm:.2f}")

print("\n=== Layer-wise Gradient Analysis ===")

def analyze_gradient_distribution(model):
    """Analyze gradient distribution across layers"""
    analysis = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad = param.grad
            analysis[name] = {
                'shape': grad.shape,
                'norm': grad.norm().item(),
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'min': grad.min().item(),
                'max': grad.max().item(),
                'zero_fraction': (grad == 0).float().mean().item(),
                'abs_mean': grad.abs().mean().item()
            }
    
    return analysis

# Analyze gradient distribution
grad_analysis = analyze_gradient_distribution(deep_model)

print("Layer-wise gradient analysis:")
for name, stats in list(grad_analysis.items())[:3]:
    print(f"  {name}:")
    print(f"    Norm: {stats['norm']:.6f}")
    print(f"    Mean: {stats['mean']:.6f}")
    print(f"    Zero fraction: {stats['zero_fraction']:.3f}")

print("\n=== Dead Neuron Detection ===")

def detect_dead_neurons(model, activation_threshold=1e-6):
    """Detect dead neurons (always zero activation)"""
    dead_neurons = {}
    
    # Hook to capture activations
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.LeakyReLU)):
            hook = module.register_forward_hook(get_activation(name))
            hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        test_input = torch.randn(32, 100)
        _ = model(test_input)
    
    # Analyze activations
    for name, activation in activations.items():
        # Check how many neurons are consistently zero
        zero_neurons = (activation.abs() < activation_threshold).all(dim=0)
        dead_count = zero_neurons.sum().item()
        total_neurons = zero_neurons.numel()
        
        if dead_count > 0:
            dead_neurons[name] = {
                'dead_count': dead_count,
                'total_neurons': total_neurons,
                'dead_fraction': dead_count / total_neurons
            }
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    return dead_neurons

# Test dead neuron detection
dead_info = detect_dead_neurons(deep_model)
if dead_info:
    print("Dead neurons detected:")
    for name, info in dead_info.items():
        print(f"  {name}: {info['dead_count']}/{info['total_neurons']} ({info['dead_fraction']:.1%})")
else:
    print("No dead neurons detected")

print("\n=== Gradient Flow Visualization ===")

def plot_gradient_flow(gradient_history, layer_names=None):
    """Plot gradient flow over training steps"""
    if not gradient_history:
        print("No gradient history to plot")
        return
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    
    # Plot gradient norms for each layer
    for name, norms in gradient_history.items():
        if norms and (layer_names is None or name in layer_names):
            steps = range(len(norms))
            plt.plot(steps, norms, label=name, marker='o', markersize=3)
    
    plt.xlabel('Training Step')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Flow Analysis')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.yscale('log')  # Log scale for better visualization
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot (optional)
    # plt.savefig('gradient_flow.png', dpi=150, bbox_inches='tight')
    plt.show()

# Generate more data for visualization
tracker.clear_history()
for step in range(20):
    input_data = torch.randn(32, 100)
    target = torch.randn(32, 1)
    
    optimizer.zero_grad()
    output = deep_model(input_data)
    loss = criterion(output, target)
    loss.backward()
    
    tracker.record_gradients(deep_model)
    optimizer.step()

# Select subset of layers for visualization
selected_layers = [name for name in tracker.layer_names if 'weight' in name][:5]
print(f"Plotting gradient flow for {len(selected_layers)} layers")

print("\n=== Gradient Ratio Analysis ===")

def analyze_gradient_ratios(model):
    """Analyze ratios between consecutive layer gradients"""
    layer_grads = []
    layer_names = []
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None and 'weight' in name:
            layer_grads.append(param.grad.norm().item())
            layer_names.append(name)
    
    # Calculate ratios
    ratios = []
    for i in range(len(layer_grads) - 1):
        if layer_grads[i+1] > 0:
            ratio = layer_grads[i] / layer_grads[i+1]
            ratios.append(ratio)
    
    return ratios, layer_names[:-1]

# Analyze gradient ratios
ratios, ratio_names = analyze_gradient_ratios(deep_model)
print("Gradient ratios between consecutive layers:")
for i, (ratio, name) in enumerate(zip(ratios, ratio_names)):
    print(f"  {name} / next layer: {ratio:.3f}")

# Detect problematic ratios
problematic_ratios = [(i, r) for i, r in enumerate(ratios) if r > 10 or r < 0.1]
if problematic_ratios:
    print("Problematic gradient ratios detected:")
    for i, ratio in problematic_ratios:
        status = "exploding" if ratio > 10 else "vanishing"
        print(f"  Layer {i}: {ratio:.3f} ({status})")

print("\n=== Automated Gradient Flow Diagnosis ===")

def diagnose_gradient_flow(model, input_data, target, criterion):
    """Comprehensive gradient flow diagnosis"""
    # Forward and backward pass
    model.zero_grad()
    output = model(input_data)
    loss = criterion(output, target)
    loss.backward()
    
    diagnosis = {
        'vanishing_layers': [],
        'exploding_layers': [],
        'dead_gradients': [],
        'healthy_layers': [],
        'recommendations': []
    }
    
    # Check each layer
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            
            if grad_norm < 1e-6:
                diagnosis['vanishing_layers'].append((name, grad_norm))
            elif grad_norm > 100:
                diagnosis['exploding_layers'].append((name, grad_norm))
            elif grad_norm == 0:
                diagnosis['dead_gradients'].append(name)
            else:
                diagnosis['healthy_layers'].append((name, grad_norm))
    
    # Generate recommendations
    if diagnosis['vanishing_layers']:
        diagnosis['recommendations'].append("Consider: ReLU activations, residual connections, better initialization")
    
    if diagnosis['exploding_layers']:
        diagnosis['recommendations'].append("Consider: gradient clipping, smaller learning rate, better initialization")
    
    if diagnosis['dead_gradients']:
        diagnosis['recommendations'].append("Consider: check for zero inputs, activation functions, learning rate")
    
    if len(diagnosis['healthy_layers']) / (len(diagnosis['vanishing_layers']) + len(diagnosis['exploding_layers']) + len(diagnosis['healthy_layers'])) > 0.8:
        diagnosis['recommendations'].append("Gradient flow appears healthy")
    
    return diagnosis

# Test diagnosis
diagnosis = diagnose_gradient_flow(deep_model, input_data, target, criterion)

print("Gradient Flow Diagnosis:")
print(f"  Healthy layers: {len(diagnosis['healthy_layers'])}")
print(f"  Vanishing layers: {len(diagnosis['vanishing_layers'])}")
print(f"  Exploding layers: {len(diagnosis['exploding_layers'])}")
print(f"  Dead gradients: {len(diagnosis['dead_gradients'])}")

print("\nRecommendations:")
for rec in diagnosis['recommendations']:
    print(f"  - {rec}")

print("\n=== Gradient Flow Best Practices ===")

print("Gradient Flow Analysis Guidelines:")
print("1. Monitor gradient norms regularly during training")
print("2. Check for vanishing/exploding gradients in deep networks")
print("3. Use appropriate activation functions (ReLU family)")
print("4. Initialize weights properly (He/Xavier initialization)")
print("5. Consider residual connections for very deep networks")
print("6. Apply gradient clipping when needed")
print("7. Use batch normalization or layer normalization")

print("\nDiagnostic Indicators:")
print("- Vanishing: gradient norms < 1e-6")
print("- Exploding: gradient norms > 100")
print("- Dead neurons: zero activations consistently")
print("- Healthy: gradient norms in [1e-4, 10] range")

print("\nSolutions:")
print("- Vanishing: Better initialization, skip connections, normalization")
print("- Exploding: Gradient clipping, smaller learning rate")
print("- Dead neurons: Check initialization, avoid saturating activations")
print("- Architecture: Consider depth vs width trade-offs")

print("\n=== Gradient Flow Analysis Complete ===") 