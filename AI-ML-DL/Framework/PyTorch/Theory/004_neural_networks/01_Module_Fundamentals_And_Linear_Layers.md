# Module Fundamentals and Linear Layers

## Table of Contents

- [nn.Module Lifecycle](#nnmodule-lifecycle)
- [nn.Linear](#nnlinear)
- [nn.Bilinear](#nnbilinear)
- [nn.Parameter vs register_buffer](#nnparameter-vs-register_buffer)
- [nn.Sequential](#nnsequential)
- [nn.ModuleList](#nnmodulelist)
- [nn.ModuleDict](#nnmoduledict)
- [nn.ParameterList and nn.ParameterDict](#nnparameterlist-and-nnparameterdict)
- [Model Composition Patterns](#model-composition-patterns)

---

## nn.Module Lifecycle

**nn.Module** is the base class for all neural network components in PyTorch. It provides automatic parameter tracking, GPU/CPU device management, training/evaluation mode switching, and hook registration.

### Initialization and Forward Pass

Every custom module must call `super().__init__()` and implement the `forward()` method. Parameters are registered via `nn.Parameter`, and non-learnable state via `register_buffer`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModule(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

model = SimpleModule(10, 5)
x = torch.randn(3, 10)
output = model(x)
```

### Parameters and state_dict

Use `parameters()` and `named_parameters()` to iterate over learnable parameters. Use `state_dict()` to obtain a dictionary of all parameters and buffers for saving/loading.

```python
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")

state = model.state_dict()
torch.save(state, 'model.pth')
```

### Training and Evaluation Modes

Call `model.train()` or `model.eval()` to switch modes. Layers like Dropout and BatchNorm behave differently in each mode.

```python
model.train()
train_output = model(x)

model.eval()
with torch.no_grad():
    eval_output = model(x)
```

---

## nn.Linear

**nn.Linear** performs a fully connected transformation: `output = input @ weight.T + bias`. It supports arbitrary batch dimensions and optional bias.

### Basic Usage

```python
linear = nn.Linear(10, 5)
x = torch.randn(3, 10)
output = linear(x)
print(output.shape)
```

| Parameter | Description |
|----------|-------------|
| in_features | Size of each input sample |
| out_features | Size of each output sample |
| bias | If True, adds learnable bias (default: True) |

### Weight and Bias Access

```python
linear = nn.Linear(6, 3)
print(linear.weight.shape)
print(linear.bias.shape)

with torch.no_grad():
    linear.weight.fill_(0.1)
    linear.bias.zero_()
```

### Without Bias

```python
linear_no_bias = nn.Linear(8, 4, bias=False)
```

### Weight Initialization

```python
nn.init.xavier_uniform_(linear.weight)
nn.init.kaiming_uniform_(linear.weight, a=math.sqrt(5), nonlinearity='relu')
nn.init.zeros_(linear.bias)
```

---

## nn.Bilinear

**nn.Bilinear** computes a bilinear transformation: `output = x1^T @ W @ x2 + b`. Useful for modeling interactions between two inputs.

```python
bilinear = nn.Bilinear(10, 20, 5)
x1 = torch.randn(3, 10)
x2 = torch.randn(3, 20)
output = bilinear(x1, x2)
```

---

## nn.Parameter vs register_buffer

### nn.Parameter

**nn.Parameter** wraps a tensor as a learnable parameter. It is included in `parameters()` and receives gradients during backpropagation.

```python
class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
```

### register_buffer

**register_buffer** registers a non-learnable tensor that moves with the module and is saved in `state_dict`. Use for running statistics, masks, or constants.

```python
class BufferExample(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))
        self.register_buffer('running_mean', torch.zeros(4))
        self.register_buffer('running_var', torch.ones(4))
```

| Feature | nn.Parameter | register_buffer |
|---------|-------------|-----------------|
| requires_grad | True by default | False |
| In optimizer | Yes | No |
| In state_dict | Yes | Yes (if persistent=True) |
| Use case | Weights, biases | Running stats, masks |

---

## nn.Sequential

**nn.Sequential** chains modules in sequence. Input is passed through each module in order.

```python
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 30),
    nn.ReLU(),
    nn.Linear(30, 1)
)

output = model(x)
```

### Named Sequential with OrderedDict

```python
from collections import OrderedDict

model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(10, 20)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(20, 1))
]))

print(model[0])
print(model.fc1)
```

---

## nn.ModuleList

**nn.ModuleList** holds submodules in a list. Unlike a Python list, modules are properly registered. You must implement the forward pass manually.

```python
layers = nn.ModuleList([
    nn.Linear(10, 20),
    nn.Linear(20, 30),
    nn.Linear(30, 40)
])

x = torch.randn(3, 10)
for i, layer in enumerate(layers):
    x = layer(x)
    if i < len(layers) - 1:
        x = F.relu(x)
```

### Use Case: Dynamic or Conditional Execution

```python
class CustomModuleList(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

---

## nn.ModuleDict

**nn.ModuleDict** holds submodules in a dictionary. Use for multiple execution paths or named branches.

```python
module_dict = nn.ModuleDict({
    'encoder': nn.Linear(10, 20),
    'decoder': nn.Linear(20, 10),
    'classifier': nn.Linear(20, 5)
})

x = module_dict['encoder'](x)
```

### Configurable Model

```python
class ConfigurableModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleDict()
        for name, cfg in config.items():
            if cfg['type'] == 'linear':
                self.layers[name] = nn.Linear(cfg['in_features'], cfg['out_features'])
        self.layer_order = list(config.keys())

    def forward(self, x):
        for name in self.layer_order:
            x = self.layers[name](x)
        return x
```

---

## nn.ParameterList and nn.ParameterDict

**nn.ParameterList** and **nn.ParameterDict** hold `nn.Parameter` objects for proper registration when you need multiple learnable tensors without wrapping them in modules.

```python
class ParameterListExample(nn.Module):
    def __init__(self, num_params):
        super().__init__()
        self.params = nn.ParameterList([
            nn.Parameter(torch.randn(2, 2)) for _ in range(num_params)
        ])

    def forward(self, x):
        for param in self.params:
            x = torch.matmul(x, param)
        return x

class ParameterDictExample(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.ParameterDict({
            'weight_1': nn.Parameter(torch.randn(4, 4)),
            'weight_2': nn.Parameter(torch.randn(4, 4)),
            'bias': nn.Parameter(torch.randn(4))
        })
```

---

## Model Composition Patterns

### Submodules and Nesting

Compose complex models from smaller building blocks. Use `named_modules()` and `children()` to traverse the hierarchy.

```python
class ComplexModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)
```

### Residual Block with Sequential

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features)
        )
        self.skip_projection = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        return F.relu(self.main_path(x) + self.skip_projection(x))
```

### Device Management

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = x.to(device)
output = model(x)
```
