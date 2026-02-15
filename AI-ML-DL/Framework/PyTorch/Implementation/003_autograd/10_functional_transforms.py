#!/usr/bin/env python3
"""PyTorch Functional Transforms - torch.func module, functional gradients"""

import torch
import torch.nn as nn
import warnings

print("=== Functional Transforms Overview ===")

print("torch.func enables:")
print("1. Functional gradient computation")
print("2. Vectorized computations (vmap)")
print("3. Jacobian computation (jacrev, jacfwd)")
print("4. Hessian computation")
print("5. Function composition and transformation")

# Check if torch.func is available (requires PyTorch 1.13+)
try:
    import torch.func as ft
    FUNC_AVAILABLE = True
    print("torch.func module available")
except ImportError:
    print("torch.func not available (requires PyTorch 1.13+)")
    FUNC_AVAILABLE = False
    # Create dummy implementations for demonstration
    class DummyFunc:
        @staticmethod
        def grad(func, argnums=0):
            def wrapper(*args, **kwargs):
                args = list(args)
                args[argnums] = args[argnums].requires_grad_(True)
                output = func(*args, **kwargs)
                return torch.autograd.grad(output, args[argnums])[0]
            return wrapper
        
        @staticmethod
        def vmap(func, in_dims=0, out_dims=0):
            def wrapper(*args, **kwargs):
                if isinstance(in_dims, int):
                    batched_arg = args[in_dims] if in_dims < len(args) else list(kwargs.values())[0]
                    batch_size = batched_arg.size(0)
                    results = []
                    for i in range(batch_size):
                        single_args = list(args)
                        if in_dims < len(args):
                            single_args[in_dims] = args[in_dims][i]
                        result = func(*single_args, **kwargs)
                        results.append(result)
                    return torch.stack(results, dim=out_dims)
                return func(*args, **kwargs)
            return wrapper
    
    ft = DummyFunc()

print("\n=== Basic Functional Gradients ===")

def simple_function(x):
    """Simple function for gradient computation"""
    return (x**2).sum()

# Traditional gradient computation
x_traditional = torch.randn(5, requires_grad=True)
y_traditional = simple_function(x_traditional)
grad_traditional = torch.autograd.grad(y_traditional, x_traditional)[0]

print(f"Traditional gradient: {grad_traditional}")

if FUNC_AVAILABLE:
    # Functional gradient computation
    grad_func = ft.grad(simple_function)
    x_functional = torch.randn(5)
    grad_functional = grad_func(x_functional)
    
    print(f"Functional gradient: {grad_functional}")
    print(f"Gradients match: {torch.allclose(grad_traditional, grad_functional)}")

print("\n=== Vectorized Map (vmap) ===")

def single_input_function(x):
    """Function that operates on single input"""
    return torch.sin(x).sum()

# Without vmap - manual batching
batch_inputs = torch.randn(10, 5)
manual_results = []
for i in range(batch_inputs.size(0)):
    result = single_input_function(batch_inputs[i])
    manual_results.append(result)
manual_batched = torch.stack(manual_results)

print(f"Manual batching result shape: {manual_batched.shape}")

if FUNC_AVAILABLE:
    # With vmap - automatic vectorization
    vmapped_func = ft.vmap(single_input_function)
    vmap_result = vmapped_func(batch_inputs)
    
    print(f"vmap result shape: {vmap_result.shape}")
    print(f"Results match: {torch.allclose(manual_batched, vmap_result)}")

print("\n=== Jacobian Computation ===")

def vector_function(x):
    """Vector-valued function for Jacobian"""
    return torch.stack([
        x[0]**2 + x[1],
        x[0] * x[1],
        x[1]**2
    ])

x_jac = torch.tensor([2.0, 3.0])

# Manual Jacobian computation
def manual_jacobian(func, inputs):
    """Manual Jacobian computation"""
    inputs = inputs.requires_grad_(True)
    outputs = func(inputs)
    jacobian = torch.zeros(outputs.size(0), inputs.size(0))
    
    for i in range(outputs.size(0)):
        grad_outputs = torch.zeros_like(outputs)
        grad_outputs[i] = 1.0
        grads = torch.autograd.grad(outputs, inputs, grad_outputs=grad_outputs, retain_graph=True)[0]
        jacobian[i] = grads
    
    return jacobian

manual_jac = manual_jacobian(vector_function, x_jac)
print(f"Manual Jacobian:\n{manual_jac}")

if FUNC_AVAILABLE and hasattr(ft, 'jacrev'):
    # Functional Jacobian (reverse mode)
    jacrev_func = ft.jacrev(vector_function)
    jac_rev = jacrev_func(x_jac)
    print(f"jacrev Jacobian:\n{jac_rev}")
    
    # Functional Jacobian (forward mode)
    if hasattr(ft, 'jacfwd'):
        jacfwd_func = ft.jacfwd(vector_function)
        jac_fwd = jacfwd_func(x_jac)
        print(f"jacfwd Jacobian:\n{jac_fwd}")

print("\n=== Hessian Computation ===")

def scalar_function_hessian(x):
    """Scalar function for Hessian computation"""
    return (x**4).sum() + (x[0] * x[1])**2

x_hess = torch.tensor([1.0, 2.0])

# Manual Hessian computation
def manual_hessian(func, inputs):
    """Manual Hessian computation"""
    def grad_func(x):
        x = x.requires_grad_(True)
        output = func(x)
        return torch.autograd.grad(output, x, create_graph=True)[0]
    
    return manual_jacobian(grad_func, inputs)

manual_hess = manual_hessian(scalar_function_hessian, x_hess)
print(f"Manual Hessian:\n{manual_hess}")

if FUNC_AVAILABLE:
    # Functional Hessian
    if hasattr(ft, 'hessian'):
        hess_func = ft.hessian(scalar_function_hessian)
        func_hess = hess_func(x_hess)
        print(f"Functional Hessian:\n{func_hess}")
    else:
        # Compose jacrev with grad for Hessian
        grad_func = ft.grad(scalar_function_hessian)
        hess_func = ft.jacrev(grad_func)
        func_hess = hess_func(x_hess)
        print(f"Composed Hessian:\n{func_hess}")

print("\n=== Neural Network Functional Gradients ===")

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 8)
        self.fc2 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def make_functional_net(net):
    """Convert nn.Module to functional form"""
    # Extract parameters
    params = list(net.parameters())
    
    def functional_net(params_tuple, x):
        # Temporarily assign parameters
        param_dict = {}
        param_idx = 0
        for name, param in net.named_parameters():
            param_dict[name] = params_tuple[param_idx]
            param_idx += 1
        
        # Create a copy of the network with new parameters
        with torch.no_grad():
            for name, param in net.named_parameters():
                param.copy_(param_dict[name])
        
        return net(x)
    
    return functional_net, tuple(params)

# Create functional network
net = SimpleNet()
func_net, params_tuple = make_functional_net(net)

# Test input
x_net = torch.randn(1, 3)

# Traditional gradient w.r.t. parameters
output_traditional = net(x_net)
loss_traditional = output_traditional.sum()
traditional_grads = torch.autograd.grad(loss_traditional, net.parameters())

print(f"Traditional parameter gradients:")
for i, grad in enumerate(traditional_grads):
    print(f"  Param {i} grad norm: {grad.norm():.6f}")

if FUNC_AVAILABLE:
    # Functional gradient w.r.t. parameters
    def loss_func(params_tuple):
        output = func_net(params_tuple, x_net)
        return output.sum()
    
    grad_func = ft.grad(loss_func)
    functional_grads = grad_func(params_tuple)
    
    print(f"Functional parameter gradients:")
    for i, grad in enumerate(functional_grads):
        print(f"  Param {i} grad norm: {grad.norm():.6f}")

print("\n=== Batch Processing with vmap ===")

if FUNC_AVAILABLE:
    # Single sample function
    def process_single_sample(params, x):
        """Process a single sample through the network"""
        return func_net(params, x.unsqueeze(0)).squeeze(0)
    
    # Batch of inputs
    batch_x = torch.randn(10, 3)
    
    # Manual batch processing
    manual_batch_results = []
    for i in range(batch_x.size(0)):
        result = process_single_sample(params_tuple, batch_x[i])
        manual_batch_results.append(result)
    manual_batch = torch.stack(manual_batch_results)
    
    # Vectorized batch processing
    vmap_batch_func = ft.vmap(process_single_sample, in_dims=(None, 0))
    vmap_batch_result = vmap_batch_func(params_tuple, batch_x)
    
    print(f"Manual batch shape: {manual_batch.shape}")
    print(f"vmap batch shape: {vmap_batch_result.shape}")
    print(f"Batch results match: {torch.allclose(manual_batch, vmap_batch_result)}")

print("\n=== Per-Sample Gradients ===")

if FUNC_AVAILABLE:
    # Function to compute loss for a single sample
    def single_sample_loss(params, x, y):
        """Compute loss for a single sample"""
        pred = func_net(params, x.unsqueeze(0))
        return ((pred - y)**2).sum()
    
    # Gradient function for single sample
    single_grad_func = ft.grad(single_sample_loss, argnums=0)
    
    # Vectorize to get per-sample gradients
    per_sample_grad_func = ft.vmap(single_grad_func, in_dims=(None, 0, 0))
    
    # Test data
    batch_x_grad = torch.randn(5, 3)
    batch_y_grad = torch.randn(5, 1)
    
    # Compute per-sample gradients
    per_sample_grads = per_sample_grad_func(params_tuple, batch_x_grad, batch_y_grad)
    
    print(f"Per-sample gradients computed for {len(per_sample_grads)} parameters")
    for i, grad_batch in enumerate(per_sample_grads):
        print(f"  Param {i}: shape {grad_batch.shape}, norm range [{grad_batch.norm(dim=tuple(range(1, grad_batch.ndim))).min():.6f}, {grad_batch.norm(dim=tuple(range(1, grad_batch.ndim))).max():.6f}]")

print("\n=== Function Composition ===")

def compose_functions(*funcs):
    """Compose multiple functions"""
    def composed(x):
        result = x
        for func in funcs:
            result = func(result)
        return result
    return composed

# Define component functions
def f1(x):
    return torch.sin(x)

def f2(x):
    return x**2

def f3(x):
    return torch.exp(x)

# Compose functions
composed_func = compose_functions(f1, f2, f3)

x_compose = torch.tensor(0.5)
result_compose = composed_func(x_compose)

print(f"Composed function result: {result_compose}")

if FUNC_AVAILABLE:
    # Gradient of composed function
    composed_grad_func = ft.grad(composed_func)
    grad_compose = composed_grad_func(x_compose)
    print(f"Composed function gradient: {grad_compose}")

print("\n=== Higher-Order Functional Derivatives ===")

def test_function_higher_order(x):
    """Function for higher-order derivatives"""
    return x**6 / 6

x_ho = torch.tensor(2.0)

if FUNC_AVAILABLE:
    # First derivative
    first_deriv_func = ft.grad(test_function_higher_order)
    first_deriv = first_deriv_func(x_ho)
    
    # Second derivative
    second_deriv_func = ft.grad(first_deriv_func)
    second_deriv = second_deriv_func(x_ho)
    
    # Third derivative
    third_deriv_func = ft.grad(second_deriv_func)
    third_deriv = third_deriv_func(x_ho)
    
    print(f"Function value: {test_function_higher_order(x_ho)}")
    print(f"First derivative: {first_deriv}")
    print(f"Second derivative: {second_deriv}")
    print(f"Third derivative: {third_deriv}")
    
    # Analytical verification: f(x) = x^6/6
    # f'(x) = x^5, f''(x) = 5x^4, f'''(x) = 20x^3
    print(f"Expected first: {x_ho**5}")
    print(f"Expected second: {5 * x_ho**4}")
    print(f"Expected third: {20 * x_ho**3}")

print("\n=== Functional Programming Patterns ===")

def curry_function(func):
    """Curry a function to enable partial application"""
    def curried(*args, **kwargs):
        if len(args) + len(kwargs) >= func.__code__.co_argcount:
            return func(*args, **kwargs)
        else:
            def partial(*more_args, **more_kwargs):
                return curried(*(args + more_args), **{**kwargs, **more_kwargs})
            return partial
    return curried

# Example: curried loss function
@curry_function
def mse_loss(pred, target, reduction='mean'):
    """Curried MSE loss"""
    loss = (pred - target)**2
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

# Partial application
mse_mean = mse_loss(reduction='mean')
mse_sum = mse_loss(reduction='sum')

pred_test = torch.tensor([1.0, 2.0, 3.0])
target_test = torch.tensor([1.1, 1.9, 3.2])

print(f"MSE (mean): {mse_mean(pred_test, target_test)}")
print(f"MSE (sum): {mse_sum(pred_test, target_test)}")

print("\n=== Performance Comparison ===")

import time

def benchmark_gradient_methods(func, inputs, iterations=100):
    """Benchmark different gradient computation methods"""
    
    # Traditional autograd
    start_time = time.time()
    for _ in range(iterations):
        inputs_copy = inputs.clone().requires_grad_(True)
        output = func(inputs_copy)
        grad = torch.autograd.grad(output, inputs_copy)[0]
    traditional_time = time.time() - start_time
    
    results = {'traditional': traditional_time}
    
    if FUNC_AVAILABLE:
        # Functional gradients
        grad_func = ft.grad(func)
        start_time = time.time()
        for _ in range(iterations):
            grad = grad_func(inputs)
        functional_time = time.time() - start_time
        results['functional'] = functional_time
    
    return results

# Benchmark
def benchmark_func(x):
    return (x**3 + torch.sin(x)).sum()

benchmark_input = torch.randn(100)
benchmark_results = benchmark_gradient_methods(benchmark_func, benchmark_input, 50)

print("Gradient Computation Benchmarks:")
for method, time_taken in benchmark_results.items():
    print(f"  {method}: {time_taken:.4f}s")

print("\n=== Advanced Functional Transforms ===")

if FUNC_AVAILABLE:
    # Custom transform: add noise to gradients
    def noisy_grad(func, noise_scale=0.01):
        """Add noise to gradients"""
        grad_func = ft.grad(func)
        
        def noisy_grad_func(x):
            grad = grad_func(x)
            noise = torch.randn_like(grad) * noise_scale
            return grad + noise
        
        return noisy_grad_func
    
    # Test noisy gradients
    clean_grad_func = ft.grad(benchmark_func)
    noisy_grad_func = noisy_grad(benchmark_func, 0.1)
    
    test_input = torch.randn(5)
    clean_grad = clean_grad_func(test_input)
    noisy_grad_result = noisy_grad_func(test_input)
    
    print(f"Clean gradient: {clean_grad}")
    print(f"Noisy gradient: {noisy_grad_result}")
    print(f"Noise magnitude: {(noisy_grad_result - clean_grad).norm():.6f}")

print("\n=== Functional Optimization ===")

if FUNC_AVAILABLE:
    # Functional implementation of gradient descent
    def functional_gd_step(loss_func, params, lr=0.01):
        """Single gradient descent step using functional gradients"""
        grad_func = ft.grad(loss_func)
        grads = grad_func(params)
        
        if isinstance(params, tuple):
            new_params = tuple(p - lr * g for p, g in zip(params, grads))
        else:
            new_params = params - lr * grads
        
        return new_params
    
    # Test functional optimization
    def quadratic_loss(x):
        """Simple quadratic loss"""
        return ((x - torch.tensor([1.0, 2.0]))**2).sum()
    
    # Optimize using functional gradient descent
    x_opt = torch.tensor([0.0, 0.0])
    loss_history = []
    
    for step in range(100):
        loss_val = quadratic_loss(x_opt)
        loss_history.append(loss_val.item())
        x_opt = functional_gd_step(quadratic_loss, x_opt, lr=0.1)
        
        if step % 20 == 0:
            print(f"Step {step}: x = {x_opt}, loss = {loss_val:.6f}")
    
    print(f"Final solution: {x_opt}")
    print(f"Target: [1.0, 2.0]")

print("\n=== Functional Transforms Best Practices ===")

print("Functional Transforms Guidelines:")
print("1. Use torch.func for composable gradient operations")
print("2. Leverage vmap for efficient batch processing")
print("3. Choose jacrev vs jacfwd based on problem dimensions")
print("4. Compose functions for complex transformations")
print("5. Consider performance implications of functional vs traditional")
print("6. Use functional programming patterns for modularity")
print("7. Test functional implementations against traditional ones")

print("\nWhen to Use Functional Transforms:")
print("- Per-sample gradients computation")
print("- Jacobian/Hessian computation for small inputs")
print("- Meta-learning applications")
print("- Research requiring flexible gradient computation")
print("- Batch processing of different functions")
print("- Higher-order derivative computation")

print("\nPerformance Considerations:")
print("- vmap can be faster than manual loops")
print("- jacfwd better for tall Jacobians (many inputs)")
print("- jacrev better for wide Jacobians (many outputs)")
print("- Functional transforms may have memory overhead")
print("- Profile before adopting in production")

print("\nLimitations:")
print("- Requires PyTorch 1.13+ for torch.func")
print("- May not support all operations")
print("- Memory usage can be higher")
print("- Learning curve for functional programming")
print("- Limited documentation for advanced use cases")

print("\n=== Functional Transforms Complete ===") 