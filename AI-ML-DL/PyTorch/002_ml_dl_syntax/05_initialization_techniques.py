#!/usr/bin/env python3
"""PyTorch Weight Initialization Techniques - All initialization methods"""

import torch
import torch.nn as nn
import torch.nn.init as init
import math

print("=== Basic Initialization Methods ===")

# Create sample layers for initialization
linear_layer = nn.Linear(256, 128)
conv_layer = nn.Conv2d(32, 64, 3, padding=1)
lstm_layer = nn.LSTM(128, 256, num_layers=2)

print(f"Linear layer weight shape: {linear_layer.weight.shape}")
print(f"Linear layer bias shape: {linear_layer.bias.shape}")
print(f"Conv layer weight shape: {conv_layer.weight.shape}")

# Default initialization (what PyTorch uses)
print(f"Linear weight default std: {linear_layer.weight.std():.6f}")
print(f"Conv weight default std: {conv_layer.weight.std():.6f}")

print("\n=== Uniform Initialization ===")

# Uniform initialization
uniform_layer = nn.Linear(256, 128)
init.uniform_(uniform_layer.weight, a=-0.1, b=0.1)
init.uniform_(uniform_layer.bias, a=-0.01, b=0.01)

print(f"Uniform weight range: [{uniform_layer.weight.min():.4f}, {uniform_layer.weight.max():.4f}]")
print(f"Uniform bias range: [{uniform_layer.bias.min():.4f}, {uniform_layer.bias.max():.4f}]")

# Kaiming uniform (He initialization)
kaiming_uniform_layer = nn.Linear(256, 128)
init.kaiming_uniform_(kaiming_uniform_layer.weight, mode='fan_in', nonlinearity='relu')
print(f"Kaiming uniform std: {kaiming_uniform_layer.weight.std():.6f}")

# Xavier uniform (Glorot initialization)
xavier_uniform_layer = nn.Linear(256, 128)
init.xavier_uniform_(xavier_uniform_layer.weight, gain=1.0)
print(f"Xavier uniform std: {xavier_uniform_layer.weight.std():.6f}")

print("\n=== Normal Initialization ===")

# Normal initialization
normal_layer = nn.Linear(256, 128)
init.normal_(normal_layer.weight, mean=0.0, std=0.02)
init.normal_(normal_layer.bias, mean=0.0, std=0.01)

print(f"Normal weight stats: mean={normal_layer.weight.mean():.6f}, std={normal_layer.weight.std():.6f}")
print(f"Normal bias stats: mean={normal_layer.bias.mean():.6f}, std={normal_layer.bias.std():.6f}")

# Kaiming normal (He normal)
kaiming_normal_layer = nn.Linear(256, 128)
init.kaiming_normal_(kaiming_normal_layer.weight, mode='fan_out', nonlinearity='relu')
print(f"Kaiming normal std: {kaiming_normal_layer.weight.std():.6f}")

# Xavier normal (Glorot normal)
xavier_normal_layer = nn.Linear(256, 128)
init.xavier_normal_(xavier_normal_layer.weight, gain=1.0)
print(f"Xavier normal std: {xavier_normal_layer.weight.std():.6f}")

print("\n=== Special Initialization Methods ===")

# Constant initialization
constant_layer = nn.Linear(256, 128)
init.constant_(constant_layer.weight, 0.1)
init.constant_(constant_layer.bias, 0.0)
print(f"Constant weight value: {constant_layer.weight[0, 0].item()}")

# Zeros and ones
zeros_layer = nn.Linear(256, 128)
ones_layer = nn.Linear(256, 128)
init.zeros_(zeros_layer.weight)
init.ones_(ones_layer.weight)
print(f"Zeros weight sum: {zeros_layer.weight.sum().item()}")
print(f"Ones weight sum: {ones_layer.weight.sum().item()}")

# Eye initialization (identity matrix for square matrices)
square_layer = nn.Linear(128, 128)
init.eye_(square_layer.weight)
print(f"Eye initialization trace: {torch.trace(square_layer.weight).item()}")

# Dirac initialization (for conv layers)
conv_dirac = nn.Conv2d(64, 64, 3, padding=1)
init.dirac_(conv_dirac.weight)
print(f"Dirac conv weight shape: {conv_dirac.weight.shape}")

print("\n=== Activation-Specific Initialization ===")

# Different gains for different activations
activations_gains = {
    'linear': 1.0,
    'sigmoid': 1.0,
    'tanh': 5/3,
    'relu': math.sqrt(2.0),
    'leaky_relu': math.sqrt(2.0 / (1 + 0.01**2)),
    'selu': 3/4
}

for activation, gain in activations_gains.items():
    layer = nn.Linear(256, 128)
    init.xavier_uniform_(layer.weight, gain=gain)
    print(f"{activation:>12} gain: {gain:.4f}, std: {layer.weight.std():.6f}")

print("\n=== Custom Initialization Functions ===")

# Custom initialization function
def custom_init(tensor, activation='relu'):
    """Custom initialization based on layer size and activation"""
    fan_in = tensor.size(-1)
    fan_out = tensor.size(0) if tensor.dim() > 1 else 1
    
    if activation == 'relu':
        # He initialization
        bound = math.sqrt(2.0 / fan_in)
    elif activation == 'tanh':
        # Xavier with tanh gain
        bound = math.sqrt(6.0 / (fan_in + fan_out)) * (5/3)
    else:
        # Xavier
        bound = math.sqrt(6.0 / (fan_in + fan_out))
    
    tensor.uniform_(-bound, bound)
    return tensor

# Apply custom initialization
custom_layer = nn.Linear(256, 128)
custom_init(custom_layer.weight, 'relu')
print(f"Custom init std: {custom_layer.weight.std():.6f}")

print("\n=== LSTM Initialization ===")

# LSTM-specific initialization
lstm_custom = nn.LSTM(128, 256, num_layers=2, bias=True)

def init_lstm(lstm_layer):
    """Initialize LSTM weights"""
    for name, param in lstm_layer.named_parameters():
        if 'weight_ih' in name:
            # Input to hidden weights
            init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:
            # Hidden to hidden weights
            init.orthogonal_(param.data)
        elif 'bias' in name:
            # Bias initialization
            param.data.fill_(0)
            # Set forget gate bias to 1
            n = param.size(0)
            param.data[n//4:n//2].fill_(1)

init_lstm(lstm_custom)
print("LSTM initialized with custom method")

# Check LSTM parameter initialization
for name, param in lstm_custom.named_parameters():
    if 'weight' in name:
        print(f"LSTM {name} std: {param.std():.6f}")

print("\n=== Convolutional Layer Initialization ===")

# Different conv initialization methods
conv_layers = {
    'kaiming_uniform': nn.Conv2d(32, 64, 3),
    'kaiming_normal': nn.Conv2d(32, 64, 3),
    'xavier_uniform': nn.Conv2d(32, 64, 3),
    'xavier_normal': nn.Conv2d(32, 64, 3),
}

# Apply different initializations
init.kaiming_uniform_(conv_layers['kaiming_uniform'].weight, nonlinearity='relu')
init.kaiming_normal_(conv_layers['kaiming_normal'].weight, nonlinearity='relu')
init.xavier_uniform_(conv_layers['xavier_uniform'].weight)
init.xavier_normal_(conv_layers['xavier_normal'].weight)

for name, layer in conv_layers.items():
    std = layer.weight.std()
    print(f"Conv {name:>15}: std={std:.6f}")

print("\n=== Batch Normalization Initialization ===")

# Batch normalization layer initialization
bn_layer = nn.BatchNorm2d(64)

# Default BN initialization
print(f"BN weight default: mean={bn_layer.weight.mean():.4f}, std={bn_layer.weight.std():.4f}")
print(f"BN bias default: mean={bn_layer.bias.mean():.4f}, std={bn_layer.bias.std():.4f}")

# Custom BN initialization
init.constant_(bn_layer.weight, 1.0)
init.constant_(bn_layer.bias, 0.0)
print("BN initialized: weight=1.0, bias=0.0")

print("\n=== Model-wide Initialization ===")

# Initialize entire model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model with custom function
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        init.constant_(m.bias, 0)

# Apply initialization
model = SimpleNet()
model.apply(init_weights)
print("Model initialized with custom function")

# Check initialization results
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        weight_std = module.weight.std()
        print(f"{name:>10} weight std: {weight_std:.6f}")

print("\n=== Pre-trained Model Initialization ===")

# Simulate loading pre-trained weights
def load_pretrained_weights(model, pretrained_dict, strict=False):
    """Load pre-trained weights into model"""
    model_dict = model.state_dict()
    
    # Filter out unnecessary keys
    if not strict:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and model_dict[k].shape == v.shape}
    
    # Update model dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    return len(pretrained_dict)

# Create fake pre-trained weights
pretrained_weights = {
    'conv1.weight': torch.randn_like(model.conv1.weight),
    'conv1.bias': torch.randn_like(model.conv1.bias),
    'bn1.weight': torch.ones_like(model.bn1.weight),
    'bn1.bias': torch.zeros_like(model.bn1.bias),
}

loaded_count = load_pretrained_weights(model, pretrained_weights)
print(f"Loaded {loaded_count} pre-trained parameters")

print("\n=== Initialization for Transfer Learning ===")

# Initialize only specific layers for fine-tuning
def init_for_transfer_learning(model, init_last_layer=True):
    """Initialize model for transfer learning"""
    for name, module in model.named_modules():
        if 'fc2' in name and init_last_layer:  # Last layer
            if isinstance(module, nn.Linear):
                init.xavier_normal_(module.weight)
                init.constant_(module.bias, 0)
                print(f"Re-initialized final layer: {name}")

init_for_transfer_learning(model)

print("\n=== Initialization Utilities ===")

# Calculate initialization bounds
def calculate_fan_in_fan_out(tensor):
    """Calculate fan_in and fan_out for a tensor"""
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    
    if dimensions == 2:  # Linear layer
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:  # Convolutional layer
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    
    return fan_in, fan_out

# Test fan calculation
linear_weight = torch.randn(128, 256)
conv_weight = torch.randn(64, 32, 3, 3)

linear_fan_in, linear_fan_out = calculate_fan_in_fan_out(linear_weight)
conv_fan_in, conv_fan_out = calculate_fan_in_fan_out(conv_weight)

print(f"Linear layer - fan_in: {linear_fan_in}, fan_out: {linear_fan_out}")
print(f"Conv layer - fan_in: {conv_fan_in}, fan_out: {conv_fan_out}")

# Initialization variance calculation
def calculate_init_variance(fan_in, fan_out, mode='fan_in', activation='relu'):
    """Calculate initialization variance"""
    if activation == 'relu':
        gain = math.sqrt(2.0)
    elif activation == 'tanh':
        gain = 5.0 / 3.0
    else:
        gain = 1.0
    
    if mode == 'fan_in':
        variance = gain / fan_in
    elif mode == 'fan_out':
        variance = gain / fan_out
    else:  # fan_avg
        variance = 2.0 * gain / (fan_in + fan_out)
    
    return variance

variance = calculate_init_variance(linear_fan_in, linear_fan_out, 'fan_in', 'relu')
print(f"Calculated variance for ReLU: {variance:.6f}")

print("\n=== Initialization Best Practices ===")

print("Initialization Guidelines:")
print("1. Use He (Kaiming) initialization for ReLU networks")
print("2. Use Xavier (Glorot) initialization for tanh/sigmoid networks")
print("3. Initialize biases to zero (except LSTM forget gates)")
print("4. Use orthogonal initialization for RNN hidden-to-hidden weights")
print("5. Initialize BatchNorm weights to 1, biases to 0")
print("6. Consider activation function when choosing initialization")
print("7. Use different initialization for different layer types")

print("\nCommon Patterns:")
print("- Conv layers: Kaiming normal with ReLU")
print("- Linear layers: Xavier or Kaiming based on activation")
print("- LSTM: Xavier for input weights, orthogonal for recurrent")
print("- BatchNorm: weights=1, bias=0")
print("- Final classification layer: Xavier normal")

print("\nTransfer Learning:")
print("- Keep pre-trained weights for feature layers")
print("- Re-initialize final classification layer")
print("- Use smaller learning rates for pre-trained layers")
print("- Consider layer-wise learning rate schedules")

print("\n=== Initialization Complete ===") 