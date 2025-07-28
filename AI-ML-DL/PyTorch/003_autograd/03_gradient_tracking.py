#!/usr/bin/env python3
"""PyTorch Gradient Tracking - requires_grad, grad_fn, leaf tensors"""

import torch
import torch.nn as nn

print("=== Gradient Tracking Overview ===")

# requires_grad attribute
x = torch.tensor([1.0, 2.0])
y = torch.tensor([3.0, 4.0], requires_grad=True)

print(f"x.requires_grad: {x.requires_grad}")
print(f"y.requires_grad: {y.requires_grad}")

# Operations inherit requires_grad
z = x + y
w = x * 2  # x doesn't require grad

print(f"z = x + y, z.requires_grad: {z.requires_grad}")
print(f"w = x * 2, w.requires_grad: {w.requires_grad}")

print("\n=== Setting requires_grad ===")

# Different ways to set requires_grad
tensor1 = torch.randn(3, 3)
tensor1.requires_grad_(True)  # In-place
print(f"tensor1.requires_grad after requires_grad_(True): {tensor1.requires_grad}")

tensor2 = torch.randn(3, 3).requires_grad_(True)  # Chained
print(f"tensor2.requires_grad: {tensor2.requires_grad}")

tensor3 = torch.randn(3, 3, requires_grad=True)  # At creation
print(f"tensor3.requires_grad: {tensor3.requires_grad}")

# Disable requires_grad
tensor1.requires_grad_(False)
print(f"tensor1.requires_grad after requires_grad_(False): {tensor1.requires_grad}")

print("\n=== Leaf Tensors ===")

# Understanding leaf tensors
leaf_tensor = torch.tensor([1.0, 2.0], requires_grad=True)
non_leaf = leaf_tensor + 1
another_non_leaf = non_leaf * 2

print(f"leaf_tensor.is_leaf: {leaf_tensor.is_leaf}")
print(f"non_leaf.is_leaf: {non_leaf.is_leaf}")
print(f"another_non_leaf.is_leaf: {another_non_leaf.is_leaf}")

# User-created tensors are leaves
user_tensor = torch.randn(2, 2, requires_grad=True)
print(f"user_tensor.is_leaf: {user_tensor.is_leaf}")

# Model parameters are leaves
linear = nn.Linear(3, 2)
print(f"linear.weight.is_leaf: {linear.weight.is_leaf}")
print(f"linear.bias.is_leaf: {linear.bias.is_leaf}")

print("\n=== grad_fn Attribute ===")

# grad_fn tracks the function that created the tensor
a = torch.tensor(2.0, requires_grad=True)
b = a**2
c = b + 1
d = torch.sin(c)

print(f"a.grad_fn: {a.grad_fn}")  # None (leaf)
print(f"b.grad_fn: {b.grad_fn}")  # PowBackward
print(f"c.grad_fn: {c.grad_fn}")  # AddBackward
print(f"d.grad_fn: {d.grad_fn}")  # SinBackward

# Examining the computation graph
print(f"d.grad_fn.next_functions: {d.grad_fn.next_functions}")

print("\n=== Gradient Retention ===")

# Only leaf tensors retain gradients by default
x_retain = torch.tensor([1.0, 2.0], requires_grad=True)
y_retain = x_retain**2
z_retain = y_retain.sum()

z_retain.backward()

print(f"x_retain.grad: {x_retain.grad}")  # Available (leaf)
print(f"y_retain.grad: {y_retain.grad}")  # None (non-leaf)

# Force gradient retention
x_retain2 = torch.tensor([1.0, 2.0], requires_grad=True)
y_retain2 = x_retain2**2
y_retain2.retain_grad()  # Force retention
z_retain2 = y_retain2.sum()

z_retain2.backward()

print(f"x_retain2.grad: {x_retain2.grad}")
print(f"y_retain2.grad: {y_retain2.grad}")  # Now available

print("\n=== Tracking in Neural Networks ===")

class TrackingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        h1 = self.fc1(x)
        h1_activated = self.relu(h1)
        output = self.fc2(h1_activated)
        return output, h1, h1_activated

net = TrackingNet()
input_data = torch.randn(3, 4, requires_grad=True)

# Forward pass
output, h1, h1_activated = net(input_data)

print(f"input_data.requires_grad: {input_data.requires_grad}")
print(f"input_data.is_leaf: {input_data.is_leaf}")
print(f"h1.requires_grad: {h1.requires_grad}")
print(f"h1.is_leaf: {h1.is_leaf}")
print(f"h1.grad_fn: {h1.grad_fn}")
print(f"output.grad_fn: {output.grad_fn}")

# Model parameters
for name, param in net.named_parameters():
    print(f"{name}.requires_grad: {param.requires_grad}")
    print(f"{name}.is_leaf: {param.is_leaf}")

print("\n=== Conditional Gradient Tracking ===")

# Context-dependent gradient tracking
def conditional_forward(x, track_gradients=True):
    if track_gradients:
        y = x**2 + 2*x + 1
    else:
        with torch.no_grad():
            y = x**2 + 2*x + 1
    return y

x_cond = torch.tensor([1.0, 2.0], requires_grad=True)

# With gradient tracking
y_with_grad = conditional_forward(x_cond, True)
print(f"y_with_grad.requires_grad: {y_with_grad.requires_grad}")

# Without gradient tracking
y_no_grad = conditional_forward(x_cond, False)
print(f"y_no_grad.requires_grad: {y_no_grad.requires_grad}")

print("\n=== Detaching from Graph ===")

# Detaching tensors
x_detach = torch.tensor([1.0, 2.0], requires_grad=True)
y_detach = x_detach**2

# Detach y from the computation graph
y_detached = y_detach.detach()
z_detach = y_detached + 1

print(f"y_detach.requires_grad: {y_detach.requires_grad}")
print(f"y_detached.requires_grad: {y_detached.requires_grad}")
print(f"z_detach.requires_grad: {z_detach.requires_grad}")

# Gradients don't flow through detached tensors
loss = z_detach.sum()
loss.backward()
print(f"x_detach.grad: {x_detach.grad}")  # None

print("\n=== Data vs. Grad Separation ===")

# Accessing data without gradients
x_data = torch.tensor([1.0, 2.0], requires_grad=True)
y_data = x_data**2

print(f"y_data: {y_data}")
print(f"y_data.data: {y_data.data}")  # Just the data
print(f"y_data.data.requires_grad: {y_data.data.requires_grad}")

# .data shares storage but breaks gradient tracking
y_data_copy = y_data.data
y_data_copy[0] = 999
print(f"y_data after modifying .data: {y_data}")

print("\n=== Gradient Function Inspection ===")

# Detailed grad_fn examination
x_inspect = torch.tensor(2.0, requires_grad=True)
y_inspect = x_inspect**3
z_inspect = torch.log(y_inspect)
w_inspect = z_inspect * 2

print("Computation graph structure:")
current = w_inspect
depth = 0
while current.grad_fn is not None:
    print(f"  {'  ' * depth}{current.grad_fn}")
    if hasattr(current.grad_fn, 'next_functions'):
        next_fns = current.grad_fn.next_functions
        if next_fns and next_fns[0][0] is not None:
            current = type('MockTensor', (), {'grad_fn': next_fns[0][0]})()
            depth += 1
        else:
            break
    else:
        break

print("\n=== Memory and Performance Impact ===")

# Gradient tracking overhead
import time

def benchmark_tracking(size, iterations=1000):
    # With gradient tracking
    start = time.time()
    for _ in range(iterations):
        x = torch.randn(size, size, requires_grad=True)
        y = x.sum()
    time_with_grad = time.time() - start
    
    # Without gradient tracking
    start = time.time()
    for _ in range(iterations):
        x = torch.randn(size, size, requires_grad=False)
        y = x.sum()
    time_no_grad = time.time() - start
    
    return time_with_grad, time_no_grad

with_grad, no_grad = benchmark_tracking(50, 500)
print(f"With gradient tracking: {with_grad:.4f}s")
print(f"Without gradient tracking: {no_grad:.4f}s")
print(f"Overhead: {(with_grad - no_grad) / no_grad * 100:.1f}%")

print("\n=== Best Practices ===")

print("Gradient Tracking Best Practices:")
print("1. Only track gradients when necessary")
print("2. Use torch.no_grad() for inference")
print("3. Understand leaf vs non-leaf tensor behavior")
print("4. Use retain_grad() sparingly")
print("5. Detach tensors to break gradient flow when needed")
print("6. Be aware of .data usage breaking tracking")
print("7. Monitor memory usage with large computation graphs")

print("\nCommon Patterns:")
print("- Model parameters: requires_grad=True by default")
print("- Input data: set requires_grad=True for gradient w.r.t. input")
print("- Intermediate activations: inherit from inputs")
print("- Loss computation: typically requires gradients")

print("\n=== Gradient Tracking Complete ===") 