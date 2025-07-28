#!/usr/bin/env python3
"""PyTorch Linear Layers Syntax - Linear layers, bias, weight access"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=== Linear Layers Overview ===")

print("Linear layers provide:")
print("1. Fully connected (dense) transformations")
print("2. Learnable weight matrices and bias vectors")
print("3. Flexible input/output dimensions")
print("4. Efficient matrix multiplication operations")
print("5. Foundation for most neural architectures")

print("\n=== Basic Linear Layer ===")

# Simple linear layer
linear_basic = nn.Linear(10, 5)  # 10 inputs, 5 outputs

print(f"Linear layer: {linear_basic}")
print(f"Weight shape: {linear_basic.weight.shape}")
print(f"Bias shape: {linear_basic.bias.shape}")
print(f"Parameters: {sum(p.numel() for p in linear_basic.parameters())}")

# Forward pass
input_tensor = torch.randn(3, 10)  # batch_size=3, features=10
output = linear_basic(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
print(f"Computation: output = input @ weight.T + bias")

print("\n=== Linear Layer Without Bias ===")

# Linear layer without bias
linear_no_bias = nn.Linear(8, 4, bias=False)

print(f"Linear (no bias): {linear_no_bias}")
print(f"Weight shape: {linear_no_bias.weight.shape}")
print(f"Bias: {linear_no_bias.bias}")
print(f"Parameters: {sum(p.numel() for p in linear_no_bias.parameters())}")

# Test
no_bias_input = torch.randn(2, 8)
no_bias_output = linear_no_bias(no_bias_input)
print(f"Output shape: {no_bias_output.shape}")

print("\n=== Weight and Bias Access ===")

# Access and modify weights and biases
linear_access = nn.Linear(6, 3)

print("Original weights:")
print(linear_access.weight.data)
print(f"Weight requires_grad: {linear_access.weight.requires_grad}")

print("\nOriginal bias:")
print(linear_access.bias.data)
print(f"Bias requires_grad: {linear_access.bias.requires_grad}")

# Modify weights
with torch.no_grad():
    linear_access.weight.fill_(0.1)  # Fill with constant
    linear_access.bias.zero_()       # Zero out bias

print("\nModified weights:")
print(linear_access.weight.data)
print("\nModified bias:")
print(linear_access.bias.data)

# Clone weights
weight_copy = linear_access.weight.data.clone()
bias_copy = linear_access.bias.data.clone()
print(f"Weight copy shape: {weight_copy.shape}")

print("\n=== Weight Initialization ===")

class LinearWithCustomInit(nn.Module):
    """Linear layer with custom initialization"""
    def __init__(self, in_features, out_features, init_type='xavier'):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.init_type = init_type
        self.init_weights()
    
    def init_weights(self):
        """Custom weight initialization"""
        if self.init_type == 'xavier':
            nn.init.xavier_uniform_(self.linear.weight)
        elif self.init_type == 'kaiming':
            nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        elif self.init_type == 'normal':
            nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        elif self.init_type == 'zeros':
            nn.init.zeros_(self.linear.weight)
        elif self.init_type == 'ones':
            nn.init.ones_(self.linear.weight)
        
        # Initialize bias
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        return self.linear(x)

# Test different initializations
init_types = ['xavier', 'kaiming', 'normal', 'zeros', 'ones']

for init_type in init_types:
    model = LinearWithCustomInit(4, 2, init_type)
    print(f"\n{init_type} initialization:")
    print(f"Weight stats - mean: {model.linear.weight.mean():.4f}, std: {model.linear.weight.std():.4f}")
    print(f"Weight range: [{model.linear.weight.min():.4f}, {model.linear.weight.max():.4f}]")

print("\n=== Advanced Weight Initialization ===")

def init_linear_layer(layer, init_scheme='he_normal'):
    """Advanced initialization schemes"""
    if isinstance(layer, nn.Linear):
        if init_scheme == 'he_normal':
            # He initialization for ReLU activations
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        elif init_scheme == 'he_uniform':
            nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
        elif init_scheme == 'xavier_normal':
            # Xavier/Glorot initialization for tanh/sigmoid
            nn.init.xavier_normal_(layer.weight)
        elif init_scheme == 'xavier_uniform':
            nn.init.xavier_uniform_(layer.weight)
        elif init_scheme == 'orthogonal':
            # Orthogonal initialization
            nn.init.orthogonal_(layer.weight)
        elif init_scheme == 'sparse':
            # Sparse initialization
            nn.init.sparse_(layer.weight, sparsity=0.1)
        
        # Bias initialization
        if layer.bias is not None:
            if init_scheme.startswith('he'):
                bound = 1 / math.sqrt(layer.weight.size(1))
                nn.init.uniform_(layer.bias, -bound, bound)
            else:
                nn.init.zeros_(layer.bias)

# Test advanced initialization
advanced_linear = nn.Linear(10, 5)
print(f"Before initialization - Weight std: {advanced_linear.weight.std():.4f}")

init_linear_layer(advanced_linear, 'he_normal')
print(f"After He normal - Weight std: {advanced_linear.weight.std():.4f}")

init_linear_layer(advanced_linear, 'orthogonal')
print(f"After orthogonal - Weight std: {advanced_linear.weight.std():.4f}")

print("\n=== Linear Layer Stacking ===")

class MultiLayerPerceptron(nn.Module):
    """Multi-layer perceptron with linear layers"""
    def __init__(self, layer_sizes, activation='relu', dropout=0.0):
        super().__init__()
        
        layers = []
        for i in range(len(layer_sizes) - 1):
            # Linear layer
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
            # Activation (except for last layer)
            if i < len(layer_sizes) - 2:
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
                
                # Dropout
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Create MLPs with different configurations
mlp_configs = [
    ([784, 256, 128, 10], 'relu', 0.2),
    ([100, 50, 25, 1], 'tanh', 0.1),
    ([20, 40, 20, 5], 'sigmoid', 0.0)
]

for i, (sizes, act, drop) in enumerate(mlp_configs):
    mlp = MultiLayerPerceptron(sizes, act, drop)
    test_input = torch.randn(4, sizes[0])
    test_output = mlp(test_input)
    
    print(f"\nMLP {i+1}: {sizes} with {act} activation")
    print(f"Input shape: {test_input.shape}, Output shape: {test_output.shape}")
    print(f"Parameters: {sum(p.numel() for p in mlp.parameters())}")

print("\n=== Linear Layer with Custom Forward ===")

class CustomLinearForward(nn.Module):
    """Linear layer with custom forward computation"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        # Initialize
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset parameters using standard initialization"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        # Manual matrix multiplication
        # x: (batch_size, in_features)
        # weight: (out_features, in_features)
        # output: (batch_size, out_features)
        
        # Method 1: Using torch.matmul
        output1 = torch.matmul(x, self.weight.t()) + self.bias
        
        # Method 2: Using einsum
        output2 = torch.einsum('bi,oi->bo', x, self.weight) + self.bias
        
        # Method 3: Using F.linear
        output3 = F.linear(x, self.weight, self.bias)
        
        # All methods should give same result
        assert torch.allclose(output1, output2, atol=1e-6)
        assert torch.allclose(output1, output3, atol=1e-6)
        
        return output1

custom_forward = CustomLinearForward(8, 4)
custom_input = torch.randn(3, 8)
custom_output = custom_forward(custom_input)

print(f"Custom forward output shape: {custom_output.shape}")

# Compare with standard linear
standard_linear = nn.Linear(8, 4)
standard_output = standard_linear(custom_input)
print(f"Standard linear output shape: {standard_output.shape}")

print("\n=== Batch Operations with Linear Layers ===")

# Linear layers handle batching automatically
batch_linear = nn.Linear(5, 3)

# Different batch sizes
batch_sizes = [1, 8, 32, 100]

for batch_size in batch_sizes:
    batch_input = torch.randn(batch_size, 5)
    batch_output = batch_linear(batch_input)
    print(f"Batch size {batch_size}: {batch_input.shape} -> {batch_output.shape}")

# Multiple dimensions
multi_dim_input = torch.randn(4, 10, 5)  # (batch, sequence, features)
multi_dim_output = batch_linear(multi_dim_input)
print(f"Multi-dimensional: {multi_dim_input.shape} -> {multi_dim_output.shape}")

print("\n=== Linear Layer Parameter Sharing ===")

class ParameterSharingExample(nn.Module):
    """Example of parameter sharing between layers"""
    def __init__(self, features):
        super().__init__()
        
        # Shared linear layer
        self.shared_linear = nn.Linear(features, features)
        
        # Additional layers
        self.input_proj = nn.Linear(features, features)
        self.output_proj = nn.Linear(features, features)
    
    def forward(self, x):
        # Project input
        x = self.input_proj(x)
        
        # Apply shared transformation multiple times
        x = self.shared_linear(x)
        x = F.relu(x)
        x = self.shared_linear(x)  # Same weights
        x = F.relu(x)
        
        # Project output
        x = self.output_proj(x)
        return x

shared_model = ParameterSharingExample(16)
shared_input = torch.randn(5, 16)
shared_output = shared_model(shared_input)

print(f"Parameter sharing model - Parameters: {sum(p.numel() for p in shared_model.parameters())}")
print(f"Output shape: {shared_output.shape}")

# Check that the same weights are used
print(f"Shared linear weight id: {id(shared_model.shared_linear.weight)}")

print("\n=== Linear Layer with Constraints ===")

class ConstrainedLinear(nn.Module):
    """Linear layer with weight constraints"""
    def __init__(self, in_features, out_features, constraint='none'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.constraint = constraint
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
    
    def get_constrained_weight(self):
        """Apply constraints to weights"""
        if self.constraint == 'none':
            return self.weight
        elif self.constraint == 'unit_norm':
            # Normalize each row to unit norm
            return F.normalize(self.weight, p=2, dim=1)
        elif self.constraint == 'orthogonal':
            # Orthogonalize weight matrix (approximate)
            u, s, v = torch.svd(self.weight)
            return torch.matmul(u, v)
        elif self.constraint == 'positive':
            # Ensure all weights are positive
            return F.relu(self.weight)
        else:
            return self.weight
    
    def forward(self, x):
        constrained_weight = self.get_constrained_weight()
        return F.linear(x, constrained_weight, self.bias)

# Test different constraints
constraints = ['none', 'unit_norm', 'orthogonal', 'positive']

for constraint in constraints:
    constrained_layer = ConstrainedLinear(6, 4, constraint)
    constraint_input = torch.randn(3, 6)
    constraint_output = constrained_layer(constraint_input)
    
    weight = constrained_layer.get_constrained_weight()
    
    print(f"\n{constraint} constraint:")
    print(f"Weight shape: {weight.shape}")
    print(f"Weight norm per row: {weight.norm(dim=1).mean():.4f}")
    print(f"Output shape: {constraint_output.shape}")

print("\n=== Linear Layer Performance Tips ===")

def benchmark_linear_operations(size, iterations=1000):
    """Benchmark different linear operations"""
    import time
    
    # Create data
    input_data = torch.randn(32, size)
    weight = torch.randn(size, size)
    bias = torch.randn(size)
    
    # F.linear
    start = time.time()
    for _ in range(iterations):
        _ = F.linear(input_data, weight, bias)
    f_linear_time = time.time() - start
    
    # Manual matmul + bias
    start = time.time()
    for _ in range(iterations):
        _ = torch.matmul(input_data, weight.t()) + bias
    manual_time = time.time() - start
    
    # nn.Linear
    linear_layer = nn.Linear(size, size)
    linear_layer.weight.data = weight
    linear_layer.bias.data = bias
    
    start = time.time()
    for _ in range(iterations):
        _ = linear_layer(input_data)
    nn_linear_time = time.time() - start
    
    return f_linear_time, manual_time, nn_linear_time

# Benchmark
f_time, manual_time, nn_time = benchmark_linear_operations(128, 100)

print("Linear operation benchmarks:")
print(f"F.linear: {f_time:.4f}s")
print(f"Manual matmul: {manual_time:.4f}s")
print(f"nn.Linear: {nn_time:.4f}s")

print("\n=== Linear Layer Debugging ===")

class DebuggableLinear(nn.Module):
    """Linear layer with debugging capabilities"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.debug_mode = False
        self.forward_count = 0
    
    def forward(self, x):
        self.forward_count += 1
        
        if self.debug_mode:
            print(f"\nForward pass #{self.forward_count}")
            print(f"Input - shape: {x.shape}, mean: {x.mean():.4f}, std: {x.std():.4f}")
            print(f"Weight - shape: {self.linear.weight.shape}, norm: {self.linear.weight.norm():.4f}")
            print(f"Bias - shape: {self.linear.bias.shape}, norm: {self.linear.bias.norm():.4f}")
        
        output = self.linear(x)
        
        if self.debug_mode:
            print(f"Output - shape: {output.shape}, mean: {output.mean():.4f}, std: {output.std():.4f}")
        
        return output
    
    def get_weight_stats(self):
        """Get weight statistics"""
        return {
            'weight_mean': self.linear.weight.mean().item(),
            'weight_std': self.linear.weight.std().item(),
            'weight_norm': self.linear.weight.norm().item(),
            'bias_mean': self.linear.bias.mean().item(),
            'bias_std': self.linear.bias.std().item(),
            'bias_norm': self.linear.bias.norm().item(),
        }

debug_linear = DebuggableLinear(10, 5)
debug_linear.debug_mode = True

debug_input = torch.randn(3, 10)
debug_output = debug_linear(debug_input)

stats = debug_linear.get_weight_stats()
print(f"\nWeight statistics: {stats}")

print("\n=== Linear Layer Best Practices ===")

print("Linear Layer Best Practices:")
print("1. Choose appropriate initialization scheme")
print("2. Consider bias necessity for your use case")
print("3. Use proper weight constraints when needed")
print("4. Monitor weight and gradient magnitudes")
print("5. Consider parameter sharing for weight efficiency")
print("6. Use F.linear for custom implementations")
print("7. Be aware of memory usage with large layers")

print("\nInitialization Guidelines:")
print("- Xavier/Glorot for tanh/sigmoid activations")
print("- He/Kaiming for ReLU activations")
print("- Orthogonal for RNNs and avoiding vanishing gradients")
print("- Custom initialization for specific requirements")

print("\nPerformance Tips:")
print("- Use in-place operations when safe")
print("- Consider quantization for inference")
print("- Batch operations efficiently")
print("- Profile different implementations")
print("- Use appropriate data types (float16 vs float32)")

print("\nDebugging:")
print("- Check input/output shapes")
print("- Monitor weight statistics")
print("- Verify gradient flow")
print("- Use hooks for intermediate inspection")
print("- Test with known inputs/outputs")

print("\n=== Linear Layers Complete ===") 