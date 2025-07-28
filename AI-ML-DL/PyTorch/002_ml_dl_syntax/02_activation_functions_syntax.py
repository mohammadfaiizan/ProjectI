#!/usr/bin/env python3
"""PyTorch Activation Functions - All activation functions and usage patterns"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

print("=== Basic Activation Functions ===")

# Create sample input
x = torch.randn(10, 100)
print(f"Input shape: {x.shape}")

# ReLU - Rectified Linear Unit
relu = nn.ReLU()
relu_output = relu(x)
relu_func = F.relu(x)
print(f"ReLU output shape: {relu_output.shape}")
print(f"ReLU functional equal: {torch.equal(relu_output, relu_func)}")

# ReLU variants
relu6 = nn.ReLU6()
hardtanh = nn.Hardtanh(min_val=-1, max_val=1)
prelu = nn.PReLU(num_parameters=100)
rrelu = nn.RReLU(lower=0.1, upper=0.3)

relu6_out = relu6(x)
hardtanh_out = hardtanh(x)
prelu_out = prelu(x)
rrelu_out = rrelu(x)

print(f"ReLU6 output shape: {relu6_out.shape}")
print(f"Hardtanh output shape: {hardtanh_out.shape}")
print(f"PReLU output shape: {prelu_out.shape}")
print(f"RReLU output shape: {rrelu_out.shape}")

print("\n=== Sigmoid and Tanh Functions ===")

# Sigmoid
sigmoid = nn.Sigmoid()
sigmoid_output = sigmoid(x)
sigmoid_func = F.sigmoid(x)
print(f"Sigmoid output shape: {sigmoid_output.shape}")
print(f"Sigmoid range: [{sigmoid_output.min():.4f}, {sigmoid_output.max():.4f}]")

# Tanh
tanh = nn.Tanh()
tanh_output = tanh(x)
tanh_func = F.tanh(x)
print(f"Tanh output shape: {tanh_output.shape}")
print(f"Tanh range: [{tanh_output.min():.4f}, {tanh_output.max():.4f}]")

# Hard versions
hardsigmoid = nn.Hardsigmoid()
hardtanh = nn.Hardtanh()
hardsigmoid_out = hardsigmoid(x)
hardtanh_out = hardtanh(x)
print(f"Hard sigmoid shape: {hardsigmoid_out.shape}")
print(f"Hard tanh shape: {hardtanh_out.shape}")

print("\n=== Modern Activation Functions ===")

# Swish/SiLU
silu = nn.SiLU()
silu_output = silu(x)
swish_func = F.silu(x)  # SiLU and Swish are the same
print(f"SiLU/Swish output shape: {silu_output.shape}")

# GELU - Gaussian Error Linear Unit
gelu = nn.GELU()
gelu_output = gelu(x)
gelu_func = F.gelu(x)
print(f"GELU output shape: {gelu_output.shape}")

# Mish
mish = nn.Mish()
mish_output = mish(x)
print(f"Mish output shape: {mish_output.shape}")

# ELU - Exponential Linear Unit
elu = nn.ELU(alpha=1.0)
elu_output = elu(x)
elu_func = F.elu(x, alpha=1.0)
print(f"ELU output shape: {elu_output.shape}")

# SELU - Scaled Exponential Linear Unit
selu = nn.SELU()
selu_output = selu(x)
selu_func = F.selu(x)
print(f"SELU output shape: {selu_output.shape}")

print("\n=== Learnable Activation Functions ===")

# PReLU - Parametric ReLU
prelu_single = nn.PReLU(num_parameters=1)  # Single parameter
prelu_channel = nn.PReLU(num_parameters=100)  # Per-channel parameters

prelu_single_out = prelu_single(x)
prelu_channel_out = prelu_channel(x)

print(f"PReLU single param shape: {prelu_single_out.shape}")
print(f"PReLU channel param shape: {prelu_channel_out.shape}")
print(f"PReLU single weight: {prelu_single.weight}")
print(f"PReLU channel weights shape: {prelu_channel.weight.shape}")

# RReLU - Randomized ReLU
rrelu_train = nn.RReLU(lower=0.1, upper=0.3, inplace=False)
rrelu_eval = nn.RReLU(lower=0.1, upper=0.3, inplace=False)

rrelu_train.train()
rrelu_eval.eval()

rrelu_train_out = rrelu_train(x)
rrelu_eval_out = rrelu_eval(x)

print(f"RReLU training mode shape: {rrelu_train_out.shape}")
print(f"RReLU evaluation mode shape: {rrelu_eval_out.shape}")

print("\n=== Softmax and LogSoftmax ===")

# Softmax
softmax_dim1 = nn.Softmax(dim=1)
logsoftmax_dim1 = nn.LogSoftmax(dim=1)

softmax_output = softmax_dim1(x)
logsoftmax_output = logsoftmax_dim1(x)

print(f"Softmax output shape: {softmax_output.shape}")
print(f"Softmax sum per row: {softmax_output.sum(dim=1)[:5]}")  # Should be ~1.0
print(f"LogSoftmax output shape: {logsoftmax_output.shape}")

# Functional versions
softmax_func = F.softmax(x, dim=1)
logsoftmax_func = F.log_softmax(x, dim=1)

print(f"Functional softmax equal: {torch.allclose(softmax_output, softmax_func)}")
print(f"Functional logsoftmax equal: {torch.allclose(logsoftmax_output, logsoftmax_func)}")

# Gumbel Softmax
gumbel_softmax_out = F.gumbel_softmax(x, tau=1.0, hard=False, dim=1)
gumbel_softmax_hard = F.gumbel_softmax(x, tau=1.0, hard=True, dim=1)

print(f"Gumbel softmax soft shape: {gumbel_softmax_out.shape}")
print(f"Gumbel softmax hard shape: {gumbel_softmax_hard.shape}")

print("\n=== Specialized Activation Functions ===")

# LeakyReLU
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
leaky_relu_out = leaky_relu(x)
leaky_relu_func = F.leaky_relu(x, negative_slope=0.01)

print(f"LeakyReLU output shape: {leaky_relu_out.shape}")
print(f"LeakyReLU functional equal: {torch.equal(leaky_relu_out, leaky_relu_func)}")

# Threshold
threshold = nn.Threshold(threshold=0.1, value=0.0)
threshold_out = threshold(x)
threshold_func = F.threshold(x, threshold=0.1, value=0.0)

print(f"Threshold output shape: {threshold_out.shape}")

# Softplus
softplus = nn.Softplus(beta=1, threshold=20)
softplus_out = softplus(x)
softplus_func = F.softplus(x, beta=1, threshold=20)

print(f"Softplus output shape: {softplus_out.shape}")

# Softshrink
softshrink = nn.Softshrink(lambd=0.5)
softshrink_out = softshrink(x)
softshrink_func = F.softshrink(x, lambd=0.5)

print(f"Softshrink output shape: {softshrink_out.shape}")

# Hardshrink
hardshrink = nn.Hardshrink(lambd=0.5)
hardshrink_out = hardshrink(x)

print(f"Hardshrink output shape: {hardshrink_out.shape}")

print("\n=== Activation Functions in Neural Networks ===")

# Simple MLP with different activations
class MLPWithActivations(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'mish':
            self.activation = nn.Mish()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

# Test different activations
input_size, hidden_size, output_size = 100, 256, 10
test_input = torch.randn(32, input_size)

activations = ['relu', 'gelu', 'swish', 'mish', 'elu']
for act_name in activations:
    model = MLPWithActivations(input_size, hidden_size, output_size, act_name)
    output = model(test_input)
    print(f"MLP with {act_name}: {test_input.shape} -> {output.shape}")

print("\n=== Convolutional Networks with Activations ===")

class ConvNetWithActivations(nn.Module):
    def __init__(self, activation='relu'):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        return x

# Test CNN with different activations
conv_input = torch.randn(8, 3, 64, 64)
for act_name in ['relu', 'gelu', 'swish']:
    conv_model = ConvNetWithActivations(act_name)
    conv_output = conv_model(conv_input)
    print(f"CNN with {act_name}: {conv_input.shape} -> {conv_output.shape}")

print("\n=== Custom Activation Functions ===")

# Custom activation function as a module
class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

# Custom parametric activation
class ParametricReLU(nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.num_parameters = num_parameters
        self.weight = nn.Parameter(torch.ones(num_parameters) * init)
    
    def forward(self, x):
        return F.leaky_relu(x, self.weight)

# Test custom activations
custom_swish = Swish(beta=1.5)
custom_prelu = ParametricReLU(num_parameters=100, init=0.1)

swish_out = custom_swish(x)
custom_prelu_out = custom_prelu(x)

print(f"Custom Swish shape: {swish_out.shape}")
print(f"Custom PReLU shape: {custom_prelu_out.shape}")
print(f"Custom PReLU parameters: {custom_prelu.weight.shape}")

print("\n=== Activation Functions for Different Tasks ===")

# Classification head with appropriate activation
class ClassificationHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x, apply_softmax=False):
        logits = self.fc(x)
        if apply_softmax:
            return self.softmax(logits)
        return logits

# Regression head (usually no activation or specific activation)
class RegressionHead(nn.Module):
    def __init__(self, input_size, output_size, activation=None):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = None
    
    def forward(self, x):
        output = self.fc(x)
        if self.activation:
            output = self.activation(output)
        return output

# Test task-specific heads
feature_input = torch.randn(32, 512)

classifier = ClassificationHead(512, 10)
regressor_none = RegressionHead(512, 1, activation=None)
regressor_sigmoid = RegressionHead(512, 1, activation='sigmoid')

class_logits = classifier(feature_input)
class_probs = classifier(feature_input, apply_softmax=True)
reg_output = regressor_none(feature_input)
reg_sigmoid_output = regressor_sigmoid(feature_input)

print(f"Classification logits: {class_logits.shape}")
print(f"Classification probabilities: {class_probs.shape}")
print(f"Regression output: {reg_output.shape}")
print(f"Regression sigmoid output: {reg_sigmoid_output.shape}")

print("\n=== Activation Function Properties ===")

# Analyze activation function properties
test_range = torch.linspace(-5, 5, 1000)

activations_to_test = {
    'ReLU': F.relu,
    'GELU': F.gelu,
    'SiLU': F.silu,
    'Tanh': F.tanh,
    'Sigmoid': F.sigmoid,
    'ELU': lambda x: F.elu(x, alpha=1.0),
    'LeakyReLU': lambda x: F.leaky_relu(x, negative_slope=0.01)
}

for name, func in activations_to_test.items():
    output = func(test_range)
    print(f"{name}:")
    print(f"  Range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"  Zero-centered: {abs(output.mean()) < 0.1}")
    print(f"  Non-zero gradient at 0: {abs(torch.autograd.grad(func(torch.tensor(0.0, requires_grad=True)), torch.tensor(0.0, requires_grad=True))[0]) > 1e-6}")

print("\n=== Gradient Flow Analysis ===")

# Analyze gradient flow through different activations
def analyze_gradient_flow(activation_func, input_tensor):
    input_tensor.requires_grad_(True)
    output = activation_func(input_tensor)
    loss = output.sum()
    loss.backward()
    
    gradient_magnitude = input_tensor.grad.abs().mean()
    return gradient_magnitude

test_input = torch.randn(1000, requires_grad=True)

gradient_flows = {}
for name, func in activations_to_test.items():
    test_input.grad = None  # Reset gradients
    grad_mag = analyze_gradient_flow(func, test_input)
    gradient_flows[name] = grad_mag
    print(f"{name} gradient magnitude: {grad_mag:.6f}")

print("\n=== Activation Function Best Practices ===")

print("Activation Function Guidelines:")
print("1. ReLU: Good default choice, simple and effective")
print("2. GELU/SiLU: Better for Transformers and modern architectures")
print("3. Tanh: Good for RNNs, zero-centered")
print("4. Sigmoid: Use only for output layers (probability)")
print("5. LeakyReLU: When dying ReLU is a problem")
print("6. ELU: Good for deep networks, smooth")
print("7. Mish: Self-regularizing, good performance")
print("8. SELU: For self-normalizing networks")

print("\nActivation Placement:")
print("- Hidden layers: ReLU, GELU, SiLU, Mish")
print("- Output layers: Softmax (classification), Sigmoid (binary), None (regression)")
print("- After normalization: Experiment with placement")
print("- RNNs: Tanh traditionally, but GELU works well too")

print("\n=== Activation Functions Complete ===") 