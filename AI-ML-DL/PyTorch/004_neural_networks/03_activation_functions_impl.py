#!/usr/bin/env python3
"""PyTorch Activation Functions Implementation - All activation functions implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=== Activation Functions Overview ===")

print("Activation functions provide:")
print("1. Non-linearity to neural networks")
print("2. Different mathematical properties")
print("3. Various gradient characteristics")
print("4. Specialized behaviors for different tasks")
print("5. Both in-place and functional variants")

print("\n=== ReLU and Variants ===")

# ReLU (Rectified Linear Unit)
relu = nn.ReLU()
relu_inplace = nn.ReLU(inplace=True)

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"Input: {x}")
print(f"ReLU: {relu(x)}")
print(f"Functional ReLU: {F.relu(x)}")

# Leaky ReLU
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
print(f"Leaky ReLU (0.01): {leaky_relu(x)}")
print(f"Functional Leaky ReLU: {F.leaky_relu(x, negative_slope=0.01)}")

# Parametric ReLU (PReLU)
prelu = nn.PReLU(num_parameters=1)  # Single parameter for all channels
prelu_multi = nn.PReLU(num_parameters=5)  # Different parameter per channel

print(f"PReLU: {prelu(x)}")
print(f"PReLU parameter: {prelu.weight.item():.4f}")

# ELU (Exponential Linear Unit)
elu = nn.ELU(alpha=1.0)
print(f"ELU: {elu(x)}")
print(f"Functional ELU: {F.elu(x, alpha=1.0)}")

# SELU (Scaled Exponential Linear Unit)
selu = nn.SELU()
print(f"SELU: {selu(x)}")
print(f"Functional SELU: {F.selu(x)}")

# ReLU6
relu6 = nn.ReLU6()
x_large = torch.tensor([-1.0, 0.0, 3.0, 6.0, 8.0])
print(f"ReLU6 input: {x_large}")
print(f"ReLU6: {relu6(x_large)}")

print("\n=== Sigmoid and Tanh ===")

# Sigmoid
sigmoid = nn.Sigmoid()
x_sigmoid = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
print(f"Sigmoid input: {x_sigmoid}")
print(f"Sigmoid: {sigmoid(x_sigmoid)}")
print(f"Functional Sigmoid: {F.sigmoid(x_sigmoid)}")

# Tanh
tanh = nn.Tanh()
print(f"Tanh: {tanh(x_sigmoid)}")
print(f"Functional Tanh: {F.tanh(x_sigmoid)}")

# Hardtanh
hardtanh = nn.Hardtanh(min_val=-1.0, max_val=1.0)
print(f"Hardtanh: {hardtanh(x_sigmoid)}")

# Softsign
softsign = nn.Softsign()
print(f"Softsign: {softsign(x_sigmoid)}")

print("\n=== GELU Variants ===")

# GELU (Gaussian Error Linear Unit)
gelu = nn.GELU()
x_gelu = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"GELU input: {x_gelu}")
print(f"GELU: {gelu(x_gelu)}")
print(f"Functional GELU: {F.gelu(x_gelu)}")

# Approximate GELU
gelu_approx = nn.GELU(approximate='tanh')
print(f"GELU (approx): {gelu_approx(x_gelu)}")

# Manual GELU implementation
def manual_gelu(x):
    """Manual GELU implementation"""
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

def fast_gelu(x):
    """Fast GELU approximation"""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

print(f"Manual GELU: {manual_gelu(x_gelu)}")
print(f"Fast GELU: {fast_gelu(x_gelu)}")

print("\n=== Swish and SiLU ===")

# SiLU (Sigmoid Linear Unit) - same as Swish
silu = nn.SiLU()
print(f"SiLU: {silu(x_gelu)}")
print(f"Functional SiLU: {F.silu(x_gelu)}")

# Manual Swish implementation
def swish(x, beta=1.0):
    """Swish activation function"""
    return x * torch.sigmoid(beta * x)

print(f"Manual Swish: {swish(x_gelu)}")
print(f"Swish (beta=2): {swish(x_gelu, beta=2.0)}")

print("\n=== Mish Activation ===")

# Mish activation
class Mish(nn.Module):
    """Mish activation function"""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

mish = Mish()
print(f"Mish: {mish(x_gelu)}")

# Functional Mish
def mish_functional(x):
    """Functional Mish implementation"""
    return x * torch.tanh(F.softplus(x))

print(f"Functional Mish: {mish_functional(x_gelu)}")

print("\n=== Hardswish ===")

# Hardswish (mobile-optimized)
hardswish = nn.Hardswish()
print(f"Hardswish: {hardswish(x_gelu)}")
print(f"Functional Hardswish: {F.hardswish(x_gelu)}")

# Manual Hardswish
def manual_hardswish(x):
    """Manual Hardswish implementation"""
    return x * F.relu6(x + 3.0) / 6.0

print(f"Manual Hardswish: {manual_hardswish(x_gelu)}")

print("\n=== Softmax and LogSoftmax ===")

# Softmax
softmax = nn.Softmax(dim=-1)
x_softmax = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(f"Softmax input: {x_softmax}")
print(f"Softmax: {softmax(x_softmax)}")
print(f"Functional Softmax: {F.softmax(x_softmax, dim=-1)}")

# LogSoftmax
log_softmax = nn.LogSoftmax(dim=-1)
print(f"LogSoftmax: {log_softmax(x_softmax)}")
print(f"Functional LogSoftmax: {F.log_softmax(x_softmax, dim=-1)}")

# 2D Softmax
softmax_2d = nn.Softmax2d()
x_2d = torch.randn(2, 3, 4, 4)  # (batch, channels, height, width)
print(f"2D Softmax input shape: {x_2d.shape}")
print(f"2D Softmax output shape: {softmax_2d(x_2d).shape}")

print("\n=== Advanced Activations ===")

# GLU (Gated Linear Unit)
glu = nn.GLU(dim=-1)
x_glu = torch.randn(3, 8)  # Must be even dimension
print(f"GLU input shape: {x_glu.shape}")
print(f"GLU output shape: {glu(x_glu).shape}")

# Threshold
threshold = nn.Threshold(threshold=0.5, value=0.0)
x_thresh = torch.tensor([0.0, 0.3, 0.5, 0.7, 1.0])
print(f"Threshold input: {x_thresh}")
print(f"Threshold (0.5): {threshold(x_thresh)}")

# Tanhshrink
tanhshrink = nn.Tanhshrink()
print(f"Tanhshrink: {tanhshrink(x_thresh)}")

# Softshrink
softshrink = nn.Softshrink(lambd=0.5)
print(f"Softshrink: {softshrink(x_thresh)}")

# Hardshrink
hardshrink = nn.Hardshrink(lambd=0.5)
print(f"Hardshrink: {hardshrink(x_thresh)}")

print("\n=== Custom Activation Functions ===")

class ParametricActivation(nn.Module):
    """Custom parametric activation function"""
    def __init__(self, num_parameters=1):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(num_parameters))
        self.beta = nn.Parameter(torch.zeros(num_parameters))
    
    def forward(self, x):
        return self.alpha * torch.tanh(self.beta + x)

class AdaptiveActivation(nn.Module):
    """Adaptive activation that learns to combine multiple activations"""
    def __init__(self, num_activations=3):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_activations) / num_activations)
        
    def forward(self, x):
        activations = [
            torch.relu(x),
            torch.tanh(x),
            torch.sigmoid(x)
        ]
        
        # Weighted combination
        result = torch.zeros_like(x)
        weights_softmax = F.softmax(self.weights, dim=0)
        
        for i, activation in enumerate(activations):
            result += weights_softmax[i] * activation
            
        return result

class Swish_beta(nn.Module):
    """Swish with learnable beta parameter"""
    def __init__(self, num_features=1):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(num_features))
    
    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

# Test custom activations
param_act = ParametricActivation(1)
adaptive_act = AdaptiveActivation(3)
swish_beta = Swish_beta(1)

x_custom = torch.tensor([-1.0, 0.0, 1.0])

print(f"Parametric activation: {param_act(x_custom)}")
print(f"Adaptive activation: {adaptive_act(x_custom)}")
print(f"Swish with beta: {swish_beta(x_custom)}")

print(f"Adaptive weights: {F.softmax(adaptive_act.weights, dim=0)}")

print("\n=== Activation Function Properties ===")

def analyze_activation(activation_fn, x_range=(-3, 3), num_points=100):
    """Analyze activation function properties"""
    x = torch.linspace(x_range[0], x_range[1], num_points, requires_grad=True)
    y = activation_fn(x)
    
    # Compute derivative
    grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, x, grad_outputs=grad_outputs, create_graph=False)[0]
    
    # Statistics
    stats = {
        'output_range': (y.min().item(), y.max().item()),
        'output_mean': y.mean().item(),
        'output_std': y.std().item(),
        'grad_range': (grad.min().item(), grad.max().item()),
        'grad_mean': grad.mean().item(),
        'dead_neurons': (grad == 0).sum().item() / len(grad),
        'saturated_grad': ((grad.abs() < 0.01) & (grad != 0)).sum().item() / len(grad)
    }
    
    return stats

# Analyze common activations
activations_to_analyze = {
    'ReLU': F.relu,
    'Sigmoid': F.sigmoid,
    'Tanh': F.tanh,
    'GELU': F.gelu,
    'SiLU': F.silu,
    'ELU': F.elu
}

print("Activation function analysis:")
for name, activation in activations_to_analyze.items():
    stats = analyze_activation(activation)
    print(f"\n{name}:")
    print(f"  Output range: [{stats['output_range'][0]:.3f}, {stats['output_range'][1]:.3f}]")
    print(f"  Gradient range: [{stats['grad_range'][0]:.3f}, {stats['grad_range'][1]:.3f}]")
    print(f"  Dead neurons: {stats['dead_neurons']:.1%}")
    print(f"  Saturated gradients: {stats['saturated_grad']:.1%}")

print("\n=== Activation Functions in Networks ===")

class ActivationTestNetwork(nn.Module):
    """Network to test different activations"""
    def __init__(self, activation_type='relu'):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)
        
        # Choose activation
        if activation_type == 'relu':
            self.activation = nn.ReLU()
        elif activation_type == 'gelu':
            self.activation = nn.GELU()
        elif activation_type == 'silu':
            self.activation = nn.SiLU()
        elif activation_type == 'elu':
            self.activation = nn.ELU()
        elif activation_type == 'tanh':
            self.activation = nn.Tanh()
        elif activation_type == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.01)
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)  # No activation on output
        return x

# Test different activations in network
activation_types = ['relu', 'gelu', 'silu', 'elu', 'tanh', 'leaky_relu']

for act_type in activation_types:
    net = ActivationTestNetwork(act_type)
    test_input = torch.randn(32, 10)
    test_output = net(test_input)
    
    # Compute some statistics
    with torch.no_grad():
        hidden1 = net.activation(net.fc1(test_input))
        hidden2 = net.activation(net.fc2(hidden1))
    
    print(f"\n{act_type.upper()} network:")
    print(f"  Hidden layer 1 - mean: {hidden1.mean():.4f}, std: {hidden1.std():.4f}")
    print(f"  Hidden layer 2 - mean: {hidden2.mean():.4f}, std: {hidden2.std():.4f}")
    print(f"  Output - mean: {test_output.mean():.4f}, std: {test_output.std():.4f}")

print("\n=== Activation Function Combinations ===")

class MultiActivationBlock(nn.Module):
    """Block using multiple activations"""
    def __init__(self, features):
        super().__init__()
        self.features = features
        
        # Split features for different activations
        self.split_size = features // 3
        
        self.linear = nn.Linear(features, features)
        
        # Different activations for different parts
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.silu = nn.SiLU()
    
    def forward(self, x):
        x = self.linear(x)
        
        # Split tensor
        x1 = x[:, :self.split_size]
        x2 = x[:, self.split_size:2*self.split_size]
        x3 = x[:, 2*self.split_size:]
        
        # Apply different activations
        x1 = self.relu(x1)
        x2 = self.gelu(x2)
        x3 = self.silu(x3)
        
        # Concatenate back
        return torch.cat([x1, x2, x3], dim=1)

multi_act_block = MultiActivationBlock(30)
multi_input = torch.randn(5, 30)
multi_output = multi_act_block(multi_input)

print(f"Multi-activation block:")
print(f"  Input shape: {multi_input.shape}")
print(f"  Output shape: {multi_output.shape}")
print(f"  Output mean: {multi_output.mean():.4f}")

print("\n=== Performance Comparison ===")

def benchmark_activations(size=(1000, 1000), iterations=100):
    """Benchmark activation function performance"""
    import time
    
    x = torch.randn(size, requires_grad=True)
    
    activations = {
        'ReLU': F.relu,
        'GELU': F.gelu,
        'SiLU': F.silu,
        'Tanh': F.tanh,
        'Sigmoid': F.sigmoid,
        'ELU': F.elu,
    }
    
    results = {}
    
    for name, activation in activations.items():
        # Forward pass timing
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(iterations):
            y = activation(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        forward_time = time.time() - start_time
        
        # Backward pass timing
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(iterations):
            x.grad = None
            y = activation(x)
            loss = y.sum()
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        total_time = time.time() - start_time
        backward_time = total_time - forward_time
        
        results[name] = {
            'forward': forward_time / iterations,
            'backward': backward_time / iterations,
            'total': total_time / iterations
        }
    
    return results

# Benchmark activations
benchmark_results = benchmark_activations(size=(100, 100), iterations=20)

print("Activation function performance (per iteration):")
for name, times in benchmark_results.items():
    print(f"{name}:")
    print(f"  Forward: {times['forward']*1000:.3f}ms")
    print(f"  Backward: {times['backward']*1000:.3f}ms")
    print(f"  Total: {times['total']*1000:.3f}ms")

print("\n=== Activation Function Best Practices ===")

print("Activation Function Guidelines:")
print("1. ReLU: Default choice, fast, can cause dead neurons")
print("2. GELU: Good for transformers, smooth gradient")
print("3. SiLU/Swish: Good balance of properties")
print("4. ELU: Helps with vanishing gradients")
print("5. Tanh: Centered around zero, can saturate")
print("6. Sigmoid: Output bounded [0,1], can saturate")
print("7. LeakyReLU: Fixes dead neuron problem")

print("\nChoosing Activations:")
print("- Vision tasks: ReLU, ELU, SiLU")
print("- NLP tasks: GELU, SiLU")
print("- Output layers: None, Sigmoid (binary), Softmax (multi-class)")
print("- Hidden layers: ReLU, GELU, SiLU")
print("- Deep networks: ELU, SELU (self-normalizing)")

print("\nPerformance Considerations:")
print("- ReLU: Fastest, simple computation")
print("- GELU/SiLU: Slower but better gradients")
print("- Avoid sigmoid/tanh in deep networks")
print("- Use in-place operations when possible")
print("- Consider approximations for expensive functions")

print("\nImplementation Tips:")
print("- Use functional forms for flexibility")
print("- Implement custom activations as nn.Module")
print("- Consider learnable parameters")
print("- Test gradient flow with different activations")
print("- Profile performance for your use case")

print("\n=== Activation Functions Complete ===") 