#!/usr/bin/env python3
"""PyTorch Container Modules - Sequential, ModuleList, ModuleDict"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== Container Modules Overview ===")

print("Container modules provide:")
print("1. Organized module composition")
print("2. Dynamic model architecture")
print("3. Parameter registration and management")
print("4. Flexible model building patterns")
print("5. Modular design capabilities")

print("\n=== Sequential Container ===")

# Basic Sequential
sequential_model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 30),
    nn.ReLU(),
    nn.Linear(30, 1)
)

input_seq = torch.randn(5, 10)
output_seq = sequential_model(input_seq)

print(f"Sequential model input: {input_seq.shape}")
print(f"Sequential model output: {output_seq.shape}")
print(f"Sequential model: {sequential_model}")

# Sequential with OrderedDict
from collections import OrderedDict

sequential_ordered = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(10, 20)),
    ('relu1', nn.ReLU()),
    ('dropout1', nn.Dropout(0.2)),
    ('fc2', nn.Linear(20, 30)),
    ('relu2', nn.ReLU()),
    ('fc3', nn.Linear(30, 1))
]))

print(f"\nNamed Sequential layers:")
for name, module in sequential_ordered.named_children():
    print(f"  {name}: {module}")

# Accessing layers in Sequential
print(f"\nAccessing layers:")
print(f"First layer: {sequential_model[0]}")
print(f"Last layer: {sequential_model[-1]}")
print(f"Slice of layers: {sequential_model[1:3]}")

print("\n=== ModuleList Container ===")

# Basic ModuleList
module_list = nn.ModuleList([
    nn.Linear(10, 20),
    nn.Linear(20, 30),
    nn.Linear(30, 40)
])

print(f"ModuleList: {module_list}")
print(f"Number of modules: {len(module_list)}")

# Forward pass with ModuleList (manual)
x = torch.randn(3, 10)
for i, layer in enumerate(module_list):
    x = layer(x)
    if i < len(module_list) - 1:  # Apply ReLU except for last layer
        x = F.relu(x)

print(f"ModuleList output: {x.shape}")

# ModuleList with different layer types
mixed_module_list = nn.ModuleList([
    nn.Conv2d(3, 32, 3),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Conv2d(32, 64, 3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(64, 10)
])

# Using ModuleList in a custom module
class CustomModuleList(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

custom_model = CustomModuleList([
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 4)
])

custom_input = torch.randn(2, 8)
custom_output = custom_model(custom_input)
print(f"Custom ModuleList model: {custom_input.shape} -> {custom_output.shape}")

print("\n=== ModuleDict Container ===")

# Basic ModuleDict
module_dict = nn.ModuleDict({
    'encoder': nn.Linear(10, 20),
    'decoder': nn.Linear(20, 10),
    'classifier': nn.Linear(20, 5)
})

print(f"ModuleDict keys: {list(module_dict.keys())}")
print(f"Encoder layer: {module_dict['encoder']}")

# Adding modules to ModuleDict
module_dict['batch_norm'] = nn.BatchNorm1d(20)
module_dict.update({
    'dropout': nn.Dropout(0.3),
    'activation': nn.ReLU()
})

print(f"Updated ModuleDict keys: {list(module_dict.keys())}")

# Using ModuleDict in a custom module
class ConfigurableModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.layers = nn.ModuleDict()
        
        # Build layers based on config
        for layer_name, layer_config in config.items():
            if layer_config['type'] == 'linear':
                self.layers[layer_name] = nn.Linear(
                    layer_config['in_features'], 
                    layer_config['out_features']
                )
            elif layer_config['type'] == 'conv2d':
                self.layers[layer_name] = nn.Conv2d(
                    layer_config['in_channels'],
                    layer_config['out_channels'],
                    layer_config['kernel_size']
                )
            elif layer_config['type'] == 'activation':
                if layer_config['name'] == 'relu':
                    self.layers[layer_name] = nn.ReLU()
                elif layer_config['name'] == 'sigmoid':
                    self.layers[layer_name] = nn.Sigmoid()
        
        self.layer_order = list(config.keys())
    
    def forward(self, x):
        for layer_name in self.layer_order:
            x = self.layers[layer_name](x)
        return x

# Test configurable model
config = {
    'fc1': {'type': 'linear', 'in_features': 16, 'out_features': 32},
    'act1': {'type': 'activation', 'name': 'relu'},
    'fc2': {'type': 'linear', 'in_features': 32, 'out_features': 8},
    'act2': {'type': 'activation', 'name': 'sigmoid'}
}

configurable_model = ConfigurableModel(config)
config_input = torch.randn(4, 16)
config_output = configurable_model(config_input)

print(f"Configurable model: {config_input.shape} -> {config_output.shape}")

print("\n=== ParameterList and ParameterDict ===")

# ParameterList for learnable parameters
class ParameterListExample(nn.Module):
    def __init__(self, num_params):
        super().__init__()
        self.params = nn.ParameterList([
            nn.Parameter(torch.randn(2, 2)) for _ in range(num_params)
        ])
    
    def forward(self, x):
        result = x
        for param in self.params:
            result = torch.matmul(result, param)
        return result

param_list_model = ParameterListExample(3)
param_input = torch.randn(3, 2)
param_output = param_list_model(param_input)

print(f"ParameterList model: {param_input.shape} -> {param_output.shape}")
print(f"Number of parameters: {len(list(param_list_model.parameters()))}")

# ParameterDict for named parameters
class ParameterDictExample(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.ParameterDict({
            'weight_1': nn.Parameter(torch.randn(4, 4)),
            'weight_2': nn.Parameter(torch.randn(4, 4)),
            'bias': nn.Parameter(torch.randn(4))
        })
    
    def forward(self, x):
        x = torch.matmul(x, self.params['weight_1'])
        x = torch.matmul(x, self.params['weight_2'])
        x = x + self.params['bias']
        return x

param_dict_model = ParameterDictExample()
param_dict_input = torch.randn(2, 4)
param_dict_output = param_dict_model(param_dict_input)

print(f"ParameterDict model: {param_dict_input.shape} -> {param_dict_output.shape}")

print("\n=== Dynamic Container Usage ===")

class DynamicNetwork(nn.Module):
    """Network that builds itself dynamically"""
    def __init__(self, layer_configs):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleDict()
        
        for i, config in enumerate(layer_configs):
            # Add layer
            if config['type'] == 'linear':
                layer = nn.Linear(config['in_dim'], config['out_dim'])
            elif config['type'] == 'conv2d':
                layer = nn.Conv2d(
                    config['in_channels'], config['out_channels'], 
                    config['kernel_size'], config.get('stride', 1), 
                    config.get('padding', 0)
                )
            
            self.layers.append(layer)
            
            # Add activation if specified
            if 'activation' in config:
                act_name = f'act_{i}'
                if config['activation'] == 'relu':
                    self.activations[act_name] = nn.ReLU()
                elif config['activation'] == 'sigmoid':
                    self.activations[act_name] = nn.Sigmoid()
                elif config['activation'] == 'tanh':
                    self.activations[act_name] = nn.Tanh()
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Apply activation if exists
            act_name = f'act_{i}'
            if act_name in self.activations:
                x = self.activations[act_name](x)
        
        return x

# Test dynamic network
dynamic_config = [
    {'type': 'linear', 'in_dim': 20, 'out_dim': 50, 'activation': 'relu'},
    {'type': 'linear', 'in_dim': 50, 'out_dim': 30, 'activation': 'relu'},
    {'type': 'linear', 'in_dim': 30, 'out_dim': 10, 'activation': 'sigmoid'}
]

dynamic_net = DynamicNetwork(dynamic_config)
dynamic_input = torch.randn(8, 20)
dynamic_output = dynamic_net(dynamic_input)

print(f"Dynamic network: {dynamic_input.shape} -> {dynamic_output.shape}")
print(f"Number of layers: {len(dynamic_net.layers)}")
print(f"Number of activations: {len(dynamic_net.activations)}")

print("\n=== Nested Containers ===")

class NestedContainerModel(nn.Module):
    """Model with nested container structures"""
    def __init__(self):
        super().__init__()
        
        # Encoder with ModuleList of Sequential blocks
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64)
            ),
            nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32)
            )
        ])
        
        # Decoder with ModuleDict of different paths
        self.decoder = nn.ModuleDict({
            'path_a': nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 128)
            ),
            'path_b': nn.Sequential(
                nn.Linear(32, 128),
                nn.Sigmoid()
            )
        })
        
        # Classifier
        self.classifier = nn.Linear(128, 10)
    
    def forward(self, x, path='path_a'):
        # Encode
        for encoder_block in self.encoder:
            x = encoder_block(x)
        
        # Decode
        x = self.decoder[path](x)
        
        # Classify
        x = self.classifier(x)
        
        return x

nested_model = NestedContainerModel()
nested_input = torch.randn(4, 128)

# Test different paths
output_a = nested_model(nested_input, path='path_a')
output_b = nested_model(nested_input, path='path_b')

print(f"Nested model path A: {nested_input.shape} -> {output_a.shape}")
print(f"Nested model path B: {nested_input.shape} -> {output_b.shape}")

print("\n=== Container Iteration and Manipulation ===")

# Iterating through containers
print("Sequential iteration:")
for i, layer in enumerate(sequential_model):
    print(f"  Layer {i}: {type(layer).__name__}")

print("\nModuleList iteration:")
for i, module in enumerate(module_list):
    print(f"  Module {i}: {module}")

print("\nModuleDict iteration:")
for name, module in module_dict.items():
    print(f"  {name}: {type(module).__name__}")

# Adding and removing modules
print("\nModifying containers:")

# Add to ModuleList
module_list.append(nn.Linear(40, 50))
print(f"ModuleList length after append: {len(module_list)}")

# Add to ModuleDict
module_dict['new_layer'] = nn.Linear(50, 25)
print(f"ModuleDict keys after addition: {list(module_dict.keys())}")

# Remove from ModuleDict
del module_dict['new_layer']
print(f"ModuleDict keys after deletion: {list(module_dict.keys())}")

print("\n=== Container-based Model Factory ===")

class ModelFactory:
    """Factory for creating models with different container patterns"""
    
    @staticmethod
    def create_mlp(layer_sizes, activations=None, use_batch_norm=False, dropout_rate=0.0):
        """Create MLP using Sequential"""
        layers = []
        
        for i in range(len(layer_sizes) - 1):
            # Linear layer
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
            # Batch normalization
            if use_batch_norm and i < len(layer_sizes) - 2:
                layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
            
            # Activation
            if activations and i < len(activations):
                if activations[i] == 'relu':
                    layers.append(nn.ReLU())
                elif activations[i] == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif activations[i] == 'tanh':
                    layers.append(nn.Tanh())
            
            # Dropout
            if dropout_rate > 0 and i < len(layer_sizes) - 2:
                layers.append(nn.Dropout(dropout_rate))
        
        return nn.Sequential(*layers)
    
    @staticmethod
    def create_cnn_backbone(channels, kernels, pooling=True):
        """Create CNN backbone using ModuleList"""
        layers = nn.ModuleList()
        
        for i in range(len(channels) - 1):
            # Convolution
            conv = nn.Conv2d(channels[i], channels[i + 1], kernels[i], padding=kernels[i]//2)
            layers.append(conv)
            
            # Activation
            layers.append(nn.ReLU())
            
            # Pooling
            if pooling and i % 2 == 1:  # Pool every other layer
                layers.append(nn.MaxPool2d(2))
        
        return layers
    
    @staticmethod
    def create_multibranch_model(shared_layers, branches):
        """Create multi-branch model using ModuleDict"""
        model_dict = nn.ModuleDict()
        
        # Shared layers
        model_dict['shared'] = nn.Sequential(*shared_layers)
        
        # Branch-specific layers
        for branch_name, branch_layers in branches.items():
            model_dict[branch_name] = nn.Sequential(*branch_layers)
        
        return model_dict

# Test model factory
print("Model Factory examples:")

# MLP
mlp = ModelFactory.create_mlp(
    layer_sizes=[784, 256, 128, 10],
    activations=['relu', 'relu', None],
    use_batch_norm=True,
    dropout_rate=0.2
)
print(f"MLP: {mlp}")

# CNN backbone
cnn_layers = ModelFactory.create_cnn_backbone(
    channels=[3, 32, 64, 128],
    kernels=[3, 3, 3]
)
print(f"CNN backbone layers: {len(cnn_layers)}")

# Multi-branch model
shared = [nn.Linear(100, 50), nn.ReLU()]
branches = {
    'classification': [nn.Linear(50, 10)],
    'regression': [nn.Linear(50, 1)]
}
multibranch = ModelFactory.create_multibranch_model(shared, branches)
print(f"Multi-branch keys: {list(multibranch.keys())}")

print("\n=== Advanced Container Patterns ===")

class ResidualBlock(nn.Module):
    """Residual block using Sequential internally"""
    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.main_path = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features)
        )
        
        # Skip connection projection if dimensions don't match
        if in_features != out_features:
            self.skip_projection = nn.Linear(in_features, out_features)
        else:
            self.skip_projection = nn.Identity()
        
        self.final_activation = nn.ReLU()
    
    def forward(self, x):
        identity = self.skip_projection(x)
        out = self.main_path(x)
        out += identity
        return self.final_activation(out)

class ResNet(nn.Module):
    """ResNet using ModuleList of ResidualBlocks"""
    def __init__(self, input_size, hidden_sizes, num_classes):
        super().__init__()
        
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        
        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            block = ResidualBlock(hidden_sizes[i], hidden_sizes[i + 1])
            self.residual_blocks.append(block)
        
        self.output_layer = nn.Linear(hidden_sizes[-1], num_classes)
    
    def forward(self, x):
        x = self.input_layer(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        x = self.output_layer(x)
        return x

# Test ResNet
resnet = ResNet(input_size=64, hidden_sizes=[128, 256, 512], num_classes=10)
resnet_input = torch.randn(8, 64)
resnet_output = resnet(resnet_input)

print(f"ResNet: {resnet_input.shape} -> {resnet_output.shape}")
print(f"Number of residual blocks: {len(resnet.residual_blocks)}")

print("\n=== Container Module Best Practices ===")

print("Container Module Guidelines:")
print("1. Use Sequential for simple linear model flow")
print("2. Use ModuleList for dynamic or conditional execution")
print("3. Use ModuleDict for multiple execution paths")
print("4. Prefer containers over Python lists for modules")
print("5. Use ParameterList/ParameterDict for learnable parameters")
print("6. Organize complex models with nested containers")
print("7. Consider factory patterns for reusable architectures")

print("\nWhen to Use Each Container:")
print("- Sequential: Simple feed-forward models")
print("- ModuleList: Dynamic models, ResNets, variable depth")
print("- ModuleDict: Multi-task models, conditional execution")
print("- ParameterList: Multiple weight matrices")
print("- ParameterDict: Named learnable parameters")

print("\nDesign Patterns:")
print("- Factory methods for creating standard architectures")
print("- Nested containers for complex model organization")
print("- Dynamic containers for architecture search")
print("- Container composition for modular design")
print("- Configuration-driven model building")

print("\nPerformance Considerations:")
print("- Sequential is most efficient for linear flow")
print("- ModuleList allows selective execution")
print("- Avoid deep nesting for better readability")
print("- Use appropriate container for your use case")
print("- Consider memory implications of large containers")

print("\n=== Container Modules Complete ===") 