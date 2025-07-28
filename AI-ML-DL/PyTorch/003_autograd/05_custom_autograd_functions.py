#!/usr/bin/env python3
"""PyTorch Custom Autograd Functions - Creating custom autograd functions"""

import torch
import torch.nn as nn
import math

print("=== Custom Autograd Functions Overview ===")

print("Custom autograd functions allow:")
print("1. Custom forward/backward operations")
print("2. Memory-efficient implementations")
print("3. Non-differentiable operations with custom gradients")
print("4. Integration of external libraries")
print("5. Debugging and gradient modification")

print("\n=== Basic Custom Function ===")

class SquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Save tensors for backward pass
        ctx.save_for_backward(input)
        return input.pow(2)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        input, = ctx.saved_tensors
        # Compute gradient: d/dx(x²) = 2x
        grad_input = 2 * input * grad_output
        return grad_input

# Use custom function
square = SquareFunction.apply
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = square(x)
loss = y.sum()

print(f"Input: {x}")
print(f"Output: {y}")

loss.backward()
print(f"Gradient: {x.grad}")
print(f"Expected: 2*x = {2*x.detach()}")

print("\n=== Multi-Input Custom Function ===")

class MultiplyAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, z):
        # Save all inputs for backward
        ctx.save_for_backward(x, y, z)
        # Compute x * y + z
        return x * y + z
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y, z = ctx.saved_tensors
        # Gradients: d/dx = y, d/dy = x, d/dz = 1
        grad_x = grad_output * y
        grad_y = grad_output * x
        grad_z = grad_output
        return grad_x, grad_y, grad_z

multiply_add = MultiplyAddFunction.apply
a = torch.tensor([2.0, 3.0], requires_grad=True)
b = torch.tensor([4.0, 5.0], requires_grad=True)
c = torch.tensor([1.0, 2.0], requires_grad=True)

result = multiply_add(a, b, c)
loss_multi = result.sum()
loss_multi.backward()

print(f"a.grad (should be b): {a.grad}")
print(f"b.grad (should be a): {b.grad}")
print(f"c.grad (should be [1,1]): {c.grad}")

print("\n=== Custom Function with Constants ===")

class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        # Save tensors and mark constants
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        # Gradient w.r.t input
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        
        # Gradient w.r.t weight
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        
        # Gradient w.r.t bias
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        
        return grad_input, grad_weight, grad_bias

# Test custom linear function
linear_custom = LinearFunction.apply
input_linear = torch.randn(5, 3, requires_grad=True)
weight_linear = torch.randn(4, 3, requires_grad=True)
bias_linear = torch.randn(4, requires_grad=True)

output_linear = linear_custom(input_linear, weight_linear, bias_linear)
loss_linear = output_linear.sum()
loss_linear.backward()

print(f"Custom linear output shape: {output_linear.shape}")
print(f"Input grad shape: {input_linear.grad.shape}")
print(f"Weight grad shape: {weight_linear.grad.shape}")
print(f"Bias grad shape: {bias_linear.grad.shape}")

print("\n=== Memory-Efficient Custom Function ===")

class MemoryEfficientActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Don't save intermediate results, recompute in backward
        ctx.save_for_backward(input)
        return torch.relu(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Recompute mask instead of saving it
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

memory_efficient_relu = MemoryEfficientActivation.apply
x_mem = torch.randn(1000, 1000, requires_grad=True)
y_mem = memory_efficient_relu(x_mem)
y_mem.sum().backward()

print(f"Memory-efficient activation gradient computed")

print("\n=== Non-Differentiable Operations ===")

class RoundSTE(torch.autograd.Function):
    """Round with Straight-Through Estimator"""
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: gradient passes unchanged
        return grad_output

class ClampedReLU(torch.autograd.Function):
    """ReLU with custom clamp behavior"""
    @staticmethod
    def forward(ctx, input, min_val, max_val):
        ctx.min_val = min_val
        ctx.max_val = max_val
        ctx.save_for_backward(input)
        return torch.clamp(torch.relu(input), min_val, max_val)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        # Zero gradient where input is negative or outside clamp range
        activated = torch.relu(input)
        grad_input[(input < 0) | (activated < ctx.min_val) | (activated > ctx.max_val)] = 0
        
        return grad_input, None, None

# Test non-differentiable operations
round_ste = RoundSTE.apply
clamped_relu = ClampedReLU.apply

x_round = torch.randn(5, requires_grad=True)
y_round = round_ste(x_round)
y_round.sum().backward()

print(f"Round STE input: {x_round}")
print(f"Round STE output: {y_round}")
print(f"Round STE gradient: {x_round.grad}")

print("\n=== Custom Function with Context ===")

class AttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, mask=None):
        # Attention computation
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        
        # Save for backward
        ctx.save_for_backward(query, key, value, attention_weights)
        ctx.mask = mask
        
        return output, attention_weights
    
    @staticmethod
    def backward(ctx, grad_output, grad_attention_weights):
        query, key, value, attention_weights = ctx.saved_tensors
        mask = ctx.mask
        
        # Gradient computations for attention
        grad_value = torch.matmul(attention_weights.transpose(-2, -1), grad_output)
        
        # Gradient through attention weights
        grad_scores = torch.matmul(grad_output, value.transpose(-2, -1))
        
        # Gradient through softmax
        grad_scores_softmax = attention_weights * (grad_scores - (attention_weights * grad_scores).sum(dim=-1, keepdim=True))
        
        if mask is not None:
            grad_scores_softmax = grad_scores_softmax.masked_fill(mask == 0, 0)
        
        # Gradients for query and key
        grad_query = torch.matmul(grad_scores_softmax, key)
        grad_key = torch.matmul(grad_scores_softmax.transpose(-2, -1), query)
        
        return grad_query, grad_key, grad_value, None

# Test custom attention
attention_func = AttentionFunction.apply
query = torch.randn(2, 4, 8, requires_grad=True)
key = torch.randn(2, 4, 8, requires_grad=True)
value = torch.randn(2, 4, 8, requires_grad=True)

output_att, weights_att = attention_func(query, key, value)
loss_att = output_att.sum()
loss_att.backward()

print(f"Custom attention output shape: {output_att.shape}")
print(f"Query gradient shape: {query.grad.shape}")

print("\n=== Custom Function with Multiple Outputs ===")

class SplitFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, split_size):
        ctx.split_size = split_size
        chunks = torch.split(input, split_size, dim=-1)
        return chunks
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        # Concatenate gradients back
        grad_input = torch.cat(grad_outputs, dim=-1)
        return grad_input, None

split_func = SplitFunction.apply
x_split = torch.randn(3, 12, requires_grad=True)
chunks = split_func(x_split, 4)

# Use chunks
loss_split = sum(chunk.sum() for chunk in chunks)
loss_split.backward()

print(f"Split function created {len(chunks)} chunks")
print(f"Original gradient shape: {x_split.grad.shape}")

print("\n=== Error Handling in Custom Functions ===")

class SafeDivisionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, numerator, denominator, eps=1e-8):
        # Safe division with epsilon
        safe_denominator = denominator + eps
        ctx.save_for_backward(numerator, safe_denominator)
        ctx.eps = eps
        return numerator / safe_denominator
    
    @staticmethod
    def backward(ctx, grad_output):
        numerator, safe_denominator = ctx.saved_tensors
        
        # Gradients for safe division
        grad_numerator = grad_output / safe_denominator
        grad_denominator = -grad_output * numerator / (safe_denominator ** 2)
        
        return grad_numerator, grad_denominator, None

safe_div = SafeDivisionFunction.apply
num = torch.randn(5, requires_grad=True)
denom = torch.randn(5, requires_grad=True)

result_safe = safe_div(num, denom)
result_safe.sum().backward()

print(f"Safe division completed without division by zero")

print("\n=== Debugging Custom Functions ===")

class DebuggingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        print(f"Forward - Input shape: {input.shape}, mean: {input.mean():.4f}")
        ctx.save_for_backward(input)
        return input ** 2
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = 2 * input * grad_output
        print(f"Backward - Grad output mean: {grad_output.mean():.4f}")
        print(f"Backward - Grad input mean: {grad_input.mean():.4f}")
        return grad_input

debug_func = DebuggingFunction.apply
x_debug = torch.randn(3, 3, requires_grad=True)
y_debug = debug_func(x_debug)
y_debug.sum().backward()

print("\n=== Performance Comparison ===")

# Compare custom vs native implementation
import time

def benchmark_custom_vs_native(size, iterations=1000):
    x = torch.randn(size, size, requires_grad=True)
    
    # Native implementation
    start = time.time()
    for _ in range(iterations):
        y = x ** 2
        loss = y.sum()
        x.grad = None
        loss.backward()
    native_time = time.time() - start
    
    # Custom implementation
    start = time.time()
    for _ in range(iterations):
        y = square(x)
        loss = y.sum()
        x.grad = None
        loss.backward()
    custom_time = time.time() - start
    
    return native_time, custom_time

native, custom = benchmark_custom_vs_native(100, 100)
print(f"Native implementation: {native:.4f}s")
print(f"Custom implementation: {custom:.4f}s")
print(f"Overhead: {(custom - native) / native * 100:.1f}%")

print("\n=== Custom Autograd Best Practices ===")

print("Custom Autograd Function Guidelines:")
print("1. Always implement both forward and backward")
print("2. Use ctx.save_for_backward for tensors")
print("3. Use ctx attributes for non-tensor data")
print("4. Handle ctx.needs_input_grad for optional gradients")
print("5. Return None for non-differentiable parameters")
print("6. Test gradients with torch.autograd.gradcheck")
print("7. Consider memory efficiency in backward pass")

print("\nCommon Use Cases:")
print("- Custom activation functions")
print("- Memory-efficient operations")
print("- Straight-through estimators")
print("- External library integration")
print("- Custom loss functions")
print("- Debugging gradient flow")

print("\nPerformance Tips:")
print("- Minimize tensor saves in forward")
print("- Recompute instead of save when memory-bound")
print("- Use in-place operations carefully")
print("- Profile custom functions vs alternatives")

print("\n=== Custom Autograd Functions Complete ===") 