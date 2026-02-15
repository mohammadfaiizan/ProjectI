#!/usr/bin/env python3
"""PyTorch Gradient Computation - Computing gradients, backward() mechanics"""

import torch
import torch.nn as nn

print("=== Gradient Computation Fundamentals ===")

# Basic backward() call
x = torch.tensor(2.0, requires_grad=True)
y = x**3 + 2*x**2 + x + 1

print(f"Function: y = x³ + 2x² + x + 1")
print(f"x = {x}")
print(f"y = {y}")

# Compute gradient
y.backward()
print(f"dy/dx = 3x² + 4x + 1")
print(f"At x=2: {x.grad}")
print(f"Expected: 3(4) + 4(2) + 1 = 21")

print("\n=== backward() Method Variants ===")

# backward() with gradient argument
x_grad = torch.tensor([1.0, 2.0], requires_grad=True)
y_grad = x_grad**2

# For non-scalar outputs, need gradient argument
grad_output = torch.tensor([1.0, 1.0])
y_grad.backward(grad_output)

print(f"x_grad: {x_grad}")
print(f"y_grad: {y_grad}")
print(f"Gradient with grad_output [1,1]: {x_grad.grad}")

# Reset and try different grad_output
x_grad.grad.zero_()
grad_output_weighted = torch.tensor([2.0, 0.5])
y_grad = x_grad**2  # Recompute
y_grad.backward(grad_output_weighted)

print(f"Gradient with grad_output [2,0.5]: {x_grad.grad}")

print("\n=== Manual Gradient Computation ===")

# Using torch.autograd.grad
x_manual = torch.tensor([1.0, 2.0], requires_grad=True)
y_manual = (x_manual**2).sum()

# Compute gradients manually
grads = torch.autograd.grad(y_manual, x_manual)
print(f"Manual gradient computation: {grads[0]}")

# Multiple inputs
a = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
c = a**2 + b**3

grads_multi = torch.autograd.grad(c, [a, b])
print(f"Gradient w.r.t a: {grads_multi[0]}")
print(f"Gradient w.r.t b: {grads_multi[1]}")

print("\n=== Gradient Computation Options ===")

# retain_graph option
x_retain = torch.tensor(2.0, requires_grad=True)
y_retain = x_retain**2
z_retain = y_retain**2

# First backward - retain graph
z_retain.backward(retain_graph=True)
grad_first = x_retain.grad.clone()

# Can backward again because graph is retained
x_retain.grad.zero_()
z_retain.backward()
grad_second = x_retain.grad.clone()

print(f"First backward: {grad_first}")
print(f"Second backward: {grad_second}")
print(f"Gradients equal: {torch.equal(grad_first, grad_second)}")

print("\n=== create_graph for Higher Order ===")

# Computing second-order derivatives
x_second = torch.tensor(2.0, requires_grad=True)
y_second = x_second**4

# First-order gradient with create_graph=True
grad_first_order = torch.autograd.grad(y_second, x_second, create_graph=True)[0]
print(f"First-order gradient: {grad_first_order}")

# Second-order gradient
grad_second_order = torch.autograd.grad(grad_first_order, x_second)[0]
print(f"Second-order gradient: {grad_second_order}")

# Analytical verification: y = x⁴, dy/dx = 4x³, d²y/dx² = 12x²
print(f"Expected second-order at x=2: 12*4 = 48")

print("\n=== Partial Derivatives ===")

# Function of multiple variables
x_partial = torch.tensor(3.0, requires_grad=True)
y_partial = torch.tensor(4.0, requires_grad=True)
z_partial = x_partial**2 + x_partial*y_partial + y_partial**2

print(f"Function: z = x² + xy + y²")
print(f"x = {x_partial}, y = {y_partial}")
print(f"z = {z_partial}")

# Compute partial derivatives
dz_dx = torch.autograd.grad(z_partial, x_partial, retain_graph=True)[0]
dz_dy = torch.autograd.grad(z_partial, y_partial)[0]

print(f"∂z/∂x = 2x + y = {dz_dx}")
print(f"∂z/∂y = x + 2y = {dz_dy}")
print(f"Expected: ∂z/∂x = 2(3) + 4 = 10, ∂z/∂y = 3 + 2(4) = 11")

print("\n=== Jacobian Computation ===")

# Computing Jacobian matrices
def compute_jacobian(func, inputs):
    """Compute Jacobian matrix"""
    inputs = inputs.requires_grad_(True)
    outputs = func(inputs)
    jacobian = torch.zeros(outputs.size(0), inputs.size(0))
    
    for i in range(outputs.size(0)):
        grad_outputs = torch.zeros_like(outputs)
        grad_outputs[i] = 1
        
        grad_inputs = torch.autograd.grad(
            outputs, inputs, grad_outputs=grad_outputs, 
            retain_graph=True, create_graph=False
        )[0]
        
        jacobian[i] = grad_inputs
    
    return jacobian

# Test function: f(x) = [x₁², x₁x₂, x₂²]
def test_function(x):
    return torch.stack([x[0]**2, x[0]*x[1], x[1]**2])

x_jac = torch.tensor([2.0, 3.0])
jacobian = compute_jacobian(test_function, x_jac)

print(f"Input: {x_jac}")
print(f"Function outputs: {test_function(x_jac)}")
print(f"Jacobian matrix:\n{jacobian}")

print("\n=== Batch Gradient Computation ===")

# Gradients for batched inputs
batch_size = 4
x_batch = torch.randn(batch_size, 2, requires_grad=True)
y_batch = (x_batch**2).sum(dim=1)  # Sum over features for each sample

print(f"Batch input shape: {x_batch.shape}")
print(f"Batch output shape: {y_batch.shape}")

# Compute gradients for each sample
grad_outputs_batch = torch.ones_like(y_batch)
y_batch.backward(grad_outputs_batch)

print(f"Batch gradients shape: {x_batch.grad.shape}")
print(f"Gradient for first sample: {x_batch.grad[0]}")

print("\n=== Neural Network Gradients ===")

# Gradients in neural networks
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 5)
        self.fc2 = nn.Linear(5, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = SimpleNet()
input_tensor = torch.randn(8, 3)
target = torch.randn(8, 2)

# Forward pass
output = net(input_tensor)
loss = nn.MSELoss()(output, target)

print(f"Network loss: {loss.item():.6f}")

# Compute gradients
loss.backward()

# Examine gradients
for name, param in net.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"{name} gradient norm: {grad_norm:.6f}")

print("\n=== Gradient Flow Analysis ===")

# Analyzing gradient flow through layers
def analyze_gradient_flow(model):
    """Analyze gradient magnitudes through network"""
    grad_info = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_info[name] = {
                'mean': param.grad.mean().item(),
                'std': param.grad.std().item(),
                'norm': param.grad.norm().item(),
                'max': param.grad.max().item(),
                'min': param.grad.min().item()
            }
    
    return grad_info

grad_analysis = analyze_gradient_flow(net)
for name, stats in grad_analysis.items():
    print(f"{name}:")
    print(f"  Mean: {stats['mean']:.6f}")
    print(f"  Norm: {stats['norm']:.6f}")

print("\n=== Custom Gradient Computation ===")

# Custom gradient computation with multiple outputs
x_custom = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

def custom_function(x):
    return torch.stack([
        x[0]**2 + x[1],
        x[1]**2 + x[2],
        x[2]**2 + x[0]
    ])

outputs = custom_function(x_custom)
print(f"Custom function outputs: {outputs}")

# Compute gradients for each output
for i, output in enumerate(outputs):
    x_custom.grad = None  # Clear previous gradients
    output.backward(retain_graph=True)
    print(f"Gradient for output {i}: {x_custom.grad}")

print("\n=== Gradient Checkpoint Integration ===")

# Using gradients with checkpointing
def checkpointed_function(x):
    """Function that could use gradient checkpointing"""
    h1 = torch.sin(x)
    h2 = torch.cos(h1)
    h3 = torch.exp(h2)
    return h3

x_checkpoint = torch.tensor(1.0, requires_grad=True)
result = checkpointed_function(x_checkpoint)

print(f"Checkpointed function result: {result}")

result.backward()
print(f"Gradient through checkpointed function: {x_checkpoint.grad}")

print("\n=== Memory Efficient Gradients ===")

# Memory considerations in gradient computation
import torch.utils.checkpoint as checkpoint

class MemoryEfficientBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        # Use checkpointing to save memory
        return checkpoint.checkpoint(self.layers, x)

# Test memory efficient gradients
mem_block = MemoryEfficientBlock(64)
x_mem = torch.randn(32, 64, requires_grad=True)

output_mem = mem_block(x_mem)
loss_mem = output_mem.sum()
loss_mem.backward()

print(f"Memory efficient block gradient computed")
print(f"Input gradient shape: {x_mem.grad.shape}")

print("\n=== Gradient Computation Best Practices ===")

print("Gradient Computation Guidelines:")
print("1. Use backward() for scalar outputs")
print("2. Provide grad_outputs for vector/tensor outputs")
print("3. Use retain_graph=True for multiple backwards")
print("4. Use create_graph=True for higher-order derivatives")
print("5. Clear gradients with zero_grad() before accumulation")
print("6. Use torch.autograd.grad for fine-grained control")
print("7. Monitor gradient magnitudes for debugging")

print("\nPerformance Tips:")
print("- Avoid unnecessary graph retention")
print("- Use gradient checkpointing for memory efficiency")
print("- Batch gradient computations when possible")
print("- Profile gradient computation overhead")

print("\nCommon Issues:")
print("- Forgetting grad_outputs for non-scalar backward")
print("- Graph freed errors from multiple backwards")
print("- Memory leaks from retained computation graphs")
print("- Gradient explosion/vanishing in deep networks")

print("\n=== Gradient Computation Complete ===") 