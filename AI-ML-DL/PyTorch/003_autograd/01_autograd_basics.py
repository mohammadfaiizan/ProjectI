#!/usr/bin/env python3
"""PyTorch Autograd Basics - Automatic differentiation fundamentals"""

import torch
import torch.nn as nn

print("=== Automatic Differentiation Overview ===")

# Basic concepts
print("Automatic Differentiation (Autograd) enables:")
print("1. Automatic computation of gradients")
print("2. Efficient backpropagation")
print("3. Dynamic computation graphs")
print("4. Higher-order derivatives")
print("5. Custom gradient functions")

print(f"PyTorch autograd enabled: {torch._C._get_tracing_state() is None}")

print("\n=== Basic Gradient Computation ===")

# Simple scalar example
x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3*x + 1

print(f"x = {x}")
print(f"y = x² + 3x + 1 = {y}")
print(f"x.requires_grad: {x.requires_grad}")
print(f"y.requires_grad: {y.requires_grad}")

# Compute gradient
y.backward()
print(f"dy/dx = 2x + 3 = {x.grad}")
print(f"Expected at x=2: 2(2) + 3 = 7")

print("\n=== Vector Operations ===")

# Vector gradients
x_vec = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y_vec = (x_vec**2).sum()

print(f"x_vec = {x_vec}")
print(f"y_vec = sum(x²) = {y_vec}")

y_vec.backward()
print(f"dy/dx = 2x = {x_vec.grad}")

print("\n=== Matrix Operations ===")

# Matrix gradients
x_mat = torch.randn(3, 2, requires_grad=True)
y_mat = torch.sum(x_mat**2)

print(f"x_mat shape: {x_mat.shape}")
print(f"y_mat (sum of squares): {y_mat}")

y_mat.backward()
print(f"Gradient shape: {x_mat.grad.shape}")
print(f"Gradient equals 2*x: {torch.allclose(x_mat.grad, 2*x_mat)}")

print("\n=== Computation Graph ===")

# Understanding the computation graph
a = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
c = a + b
d = c * 3
e = torch.sin(d)

print(f"a = {a}")
print(f"b = {b}")
print(f"c = a + b = {c}")
print(f"d = c * 3 = {d}")
print(f"e = sin(d) = {e}")

# Check computation graph
print(f"a.grad_fn: {a.grad_fn}")  # None (leaf node)
print(f"c.grad_fn: {c.grad_fn}")  # AddBackward
print(f"d.grad_fn: {d.grad_fn}")  # MulBackward
print(f"e.grad_fn: {e.grad_fn}")  # SinBackward

# Compute gradients
e.backward()
print(f"de/da = {a.grad}")
print(f"de/db = {b.grad}")

print("\n=== Leaf vs Non-leaf Tensors ===")

# Leaf tensors
leaf1 = torch.tensor([1.0, 2.0], requires_grad=True)
leaf2 = torch.tensor([3.0, 4.0], requires_grad=True)

# Non-leaf tensors
non_leaf1 = leaf1 + leaf2
non_leaf2 = non_leaf1 * 2

print(f"leaf1.is_leaf: {leaf1.is_leaf}")
print(f"leaf2.is_leaf: {leaf2.is_leaf}")
print(f"non_leaf1.is_leaf: {non_leaf1.is_leaf}")
print(f"non_leaf2.is_leaf: {non_leaf2.is_leaf}")

# Only leaf tensors retain gradients by default
loss = non_leaf2.sum()
loss.backward()

print(f"leaf1.grad: {leaf1.grad}")
print(f"leaf2.grad: {leaf2.grad}")
print(f"non_leaf1.grad: {non_leaf1.grad}")  # None by default

print("\n=== Retaining Gradients ===")

# Force gradient retention for non-leaf tensors
x_retain = torch.tensor([1.0, 2.0], requires_grad=True)
y_retain = x_retain**2
y_retain.retain_grad()  # Force gradient retention

z_retain = y_retain.sum()
z_retain.backward()

print(f"x_retain.grad: {x_retain.grad}")
print(f"y_retain.grad: {y_retain.grad}")  # Now available

print("\n=== Multiple Outputs ===")

# Multiple outputs from single input
x_multi = torch.tensor([1.0, 2.0], requires_grad=True)
y1_multi = x_multi[0]**2
y2_multi = x_multi[1]**3

print(f"x_multi: {x_multi}")
print(f"y1_multi: {y1_multi}")
print(f"y2_multi: {y2_multi}")

# Need to compute gradients separately or combine
combined_loss = y1_multi + y2_multi
combined_loss.backward()

print(f"Combined gradient: {x_multi.grad}")

print("\n=== Gradient Accumulation ===")

# Gradients accumulate across backward() calls
x_accum = torch.tensor([1.0, 2.0], requires_grad=True)

# First computation
y1_accum = (x_accum**2).sum()
y1_accum.backward()
grad_after_first = x_accum.grad.clone()

# Second computation (gradients accumulate)
y2_accum = (x_accum**3).sum()
y2_accum.backward()
grad_after_second = x_accum.grad.clone()

print(f"After first backward: {grad_after_first}")
print(f"After second backward: {grad_after_second}")
print(f"Gradients accumulated: {not torch.equal(grad_after_first, grad_after_second)}")

# Clear gradients
x_accum.grad.zero_()
print(f"After zero_grad(): {x_accum.grad}")

print("\n=== In-place Operations ===")

# In-place operations and autograd
x_inplace = torch.tensor([1.0, 2.0], requires_grad=True)
x_original_ptr = x_inplace.data_ptr()

# This works
y_inplace = x_inplace**2

# In-place operation on tensor with requires_grad=True
try:
    x_inplace += 1  # This might cause issues in some contexts
    print(f"In-place operation succeeded")
    print(f"Same memory pointer: {x_inplace.data_ptr() == x_original_ptr}")
except RuntimeError as e:
    print(f"In-place operation error: {e}")

print("\n=== Detaching from Graph ===")

# Detaching tensors from computation graph
x_detach = torch.tensor([1.0, 2.0], requires_grad=True)
y_detach = x_detach**2

# Detach y from graph
y_detached = y_detach.detach()
z_detach = y_detached * 3

print(f"y_detach.requires_grad: {y_detach.requires_grad}")
print(f"y_detached.requires_grad: {y_detached.requires_grad}")
print(f"z_detach.requires_grad: {z_detach.requires_grad}")

# Only gradients flow to x_detach, not through detached path
final_loss = z_detach.sum()
final_loss.backward()
print(f"x_detach.grad: {x_detach.grad}")  # Should be None

print("\n=== Neural Network Example ===")

# Simple neural network autograd
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create network and data
net = SimpleNN()
input_data = torch.randn(5, 2)
target = torch.randn(5, 1)

print(f"Input shape: {input_data.shape}")
print(f"Target shape: {target.shape}")

# Forward pass
output = net(input_data)
loss = nn.MSELoss()(output, target)

print(f"Output shape: {output.shape}")
print(f"Loss: {loss.item():.6f}")

# Check parameter gradients before backward
print(f"FC1 weight grad before backward: {net.fc1.weight.grad}")

# Backward pass
loss.backward()

# Check parameter gradients after backward
print(f"FC1 weight grad after backward: {net.fc1.weight.grad is not None}")
print(f"FC1 weight grad shape: {net.fc1.weight.grad.shape}")

print("\n=== Autograd Mechanics ===")

# Understanding autograd function calls
x_mech = torch.tensor(3.0, requires_grad=True)
y_mech = x_mech**2

print(f"x_mech: {x_mech}")
print(f"y_mech: {y_mech}")
print(f"y_mech.grad_fn: {y_mech.grad_fn}")
print(f"y_mech.grad_fn.next_functions: {y_mech.grad_fn.next_functions}")

# Multiple operations
z_mech = y_mech + 1
w_mech = z_mech * 2

print(f"z_mech.grad_fn: {z_mech.grad_fn}")
print(f"w_mech.grad_fn: {w_mech.grad_fn}")

print("\n=== Error Handling ===")

# Common autograd errors
try:
    # Trying to backward on non-scalar
    x_error = torch.tensor([1.0, 2.0], requires_grad=True)
    y_error = x_error**2
    y_error.backward()  # Error: grad can only be computed for scalar outputs
except RuntimeError as e:
    print(f"Backward error: {e}")

# Correct way with sum or specific gradient
x_correct = torch.tensor([1.0, 2.0], requires_grad=True)
y_correct = x_error**2
y_correct.sum().backward()  # Works
print(f"Correct gradient: {x_correct.grad}")

print("\n=== Performance Considerations ===")

# Gradient computation overhead
import time

def benchmark_autograd(size, iterations=1000):
    x = torch.randn(size, size, requires_grad=True)
    
    # With gradients
    start_time = time.time()
    for _ in range(iterations):
        y = (x**2).sum()
        y.backward()
        x.grad.zero_()
    with_grad_time = time.time() - start_time
    
    # Without gradients
    x_no_grad = torch.randn(size, size, requires_grad=False)
    start_time = time.time()
    for _ in range(iterations):
        y = (x_no_grad**2).sum()
    no_grad_time = time.time() - start_time
    
    return with_grad_time, no_grad_time

# Benchmark
with_grad, no_grad = benchmark_autograd(100, 100)
overhead = (with_grad - no_grad) / no_grad * 100

print(f"With autograd: {with_grad:.4f}s")
print(f"Without autograd: {no_grad:.4f}s")
print(f"Autograd overhead: {overhead:.1f}%")

print("\n=== Best Practices ===")

print("Autograd Best Practices:")
print("1. Only set requires_grad=True when needed")
print("2. Use torch.no_grad() for inference")
print("3. Clear gradients with optimizer.zero_grad()")
print("4. Be careful with in-place operations")
print("5. Use retain_grad() sparingly")
print("6. Detach tensors when breaking gradient flow")
print("7. Understand leaf vs non-leaf tensors")

print("\nCommon Pitfalls:")
print("- Forgetting to clear gradients")
print("- In-place operations breaking gradient flow")
print("- Trying to backward() on non-scalar tensors")
print("- Memory leaks from retaining computation graphs")
print("- Mixing requires_grad tensors inconsistently")

print("\n=== Autograd Basics Complete ===") 