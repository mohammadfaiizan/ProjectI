#!/usr/bin/env python3
"""PyTorch Autograd Performance - Optimizing autograd performance"""

import torch
import torch.nn as nn
import time
import gc

print("=== Autograd Performance Overview ===")

print("Autograd performance optimization focuses on:")
print("1. Memory efficiency")
print("2. Computation speed")
print("3. Graph optimization")
print("4. Gradient accumulation strategies")
print("5. Memory layout optimization")

print("\n=== Memory Optimization Strategies ===")

# Memory-efficient gradient computation
def memory_efficient_backward(model, inputs, targets, batch_size=32):
    """Memory-efficient backward pass with gradient accumulation"""
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0
    
    # Process in mini-batches to save memory
    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]
        
        # Forward pass
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        
        # Scale loss for accumulation
        loss = loss / (len(inputs) // batch_size + 1)
        
        # Backward pass
        loss.backward()
        total_loss += loss.item()
        
        # Clear intermediate activations
        del outputs, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return total_loss

# Test memory-efficient training
class LargeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

large_model = LargeModel()
large_inputs = torch.randn(256, 1024)
large_targets = torch.randn(256, 1)

# Memory usage comparison
if torch.cuda.is_available():
    large_model = large_model.cuda()
    large_inputs = large_inputs.cuda()
    large_targets = large_targets.cuda()
    
    # Standard backward
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    outputs = large_model(large_inputs)
    loss = nn.MSELoss()(outputs, large_targets)
    loss.backward()
    large_model.zero_grad()
    
    standard_memory = torch.cuda.max_memory_allocated() / 1e6
    
    # Memory-efficient backward
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    large_model.zero_grad()
    
    memory_efficient_backward(large_model, large_inputs, large_targets, batch_size=64)
    large_model.zero_grad()
    
    efficient_memory = torch.cuda.max_memory_allocated() / 1e6
    
    print(f"Standard memory usage: {standard_memory:.2f} MB")
    print(f"Memory-efficient usage: {efficient_memory:.2f} MB")
    print(f"Memory savings: {((standard_memory - efficient_memory) / standard_memory * 100):.1f}%")

print("\n=== In-Place Operations for Performance ===")

def compare_inplace_performance(size=1000, iterations=1000):
    """Compare in-place vs out-of-place operations"""
    
    # Out-of-place operations
    x = torch.randn(size, size, requires_grad=True)
    
    start_time = time.time()
    for _ in range(iterations):
        y = torch.relu(x)
        z = y + 1.0
        w = z * 2.0
        loss = w.sum()
        
        x.grad = None
        loss.backward()
    out_of_place_time = time.time() - start_time
    
    # In-place operations (where safe)
    x = torch.randn(size, size, requires_grad=True)
    
    start_time = time.time()
    for _ in range(iterations):
        y = torch.relu(x)
        y.add_(1.0)  # In-place addition
        y.mul_(2.0)  # In-place multiplication
        loss = y.sum()
        
        x.grad = None
        loss.backward()
    in_place_time = time.time() - start_time
    
    return out_of_place_time, in_place_time

out_place_t, in_place_t = compare_inplace_performance(100, 100)
print(f"Out-of-place time: {out_place_t:.4f}s")
print(f"In-place time: {in_place_t:.4f}s")
print(f"Speedup: {out_place_t / in_place_t:.2f}x")

print("\n=== Gradient Accumulation Optimization ===")

class OptimizedGradientAccumulator:
    """Optimized gradient accumulation"""
    def __init__(self, model, accumulation_steps=4):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
    
    def accumulate_gradients(self, loss):
        """Accumulate gradients efficiently"""
        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        
        self.step_count += 1
        
        # Return True when ready to step optimizer
        return self.step_count % self.accumulation_steps == 0
    
    def sync_gradients(self):
        """Synchronize gradients across devices if needed"""
        if torch.cuda.device_count() > 1:
            # In real distributed training, use appropriate reduction
            for param in self.model.parameters():
                if param.grad is not None:
                    # Simulate gradient synchronization
                    param.grad.data = param.grad.data.clone()
    
    def reset(self):
        """Reset accumulation counter"""
        self.step_count = 0

# Test gradient accumulation
accumulator = OptimizedGradientAccumulator(large_model, accumulation_steps=4)
optimizer = torch.optim.Adam(large_model.parameters())

print("Testing optimized gradient accumulation:")
for step in range(8):  # Simulate 8 mini-batches
    batch_inputs = torch.randn(32, 1024)
    batch_targets = torch.randn(32, 1)
    
    if torch.cuda.is_available():
        batch_inputs = batch_inputs.cuda()
        batch_targets = batch_targets.cuda()
    
    # Forward pass
    outputs = large_model(batch_inputs)
    loss = nn.MSELoss()(outputs, batch_targets)
    
    # Accumulate gradients
    should_step = accumulator.accumulate_gradients(loss)
    
    if should_step:
        # Synchronize gradients
        accumulator.sync_gradients()
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        accumulator.reset()
        
        print(f"  Optimizer step at mini-batch {step + 1}")

print("\n=== Graph Optimization Techniques ===")

def optimize_computation_graph():
    """Demonstrate computation graph optimizations"""
    
    # Inefficient graph structure
    def inefficient_computation(x):
        # Creates many intermediate nodes
        result = x
        for i in range(10):
            temp1 = torch.sin(result)
            temp2 = torch.cos(result)
            temp3 = temp1 + temp2
            result = temp3 * 0.1 + result * 0.9
        return result
    
    # Optimized graph structure
    def efficient_computation(x):
        # Fewer intermediate nodes
        result = x
        for i in range(10):
            sincos = torch.sin(result) + torch.cos(result)
            result = sincos * 0.1 + result * 0.9
        return result
    
    x = torch.randn(500, 500, requires_grad=True)
    
    # Benchmark inefficient version
    start_time = time.time()
    for _ in range(50):
        x.grad = None
        y = inefficient_computation(x)
        loss = y.sum()
        loss.backward()
    inefficient_time = time.time() - start_time
    
    # Benchmark efficient version
    start_time = time.time()
    for _ in range(50):
        x.grad = None
        y = efficient_computation(x)
        loss = y.sum()
        loss.backward()
    efficient_time = time.time() - start_time
    
    print(f"Inefficient graph time: {inefficient_time:.4f}s")
    print(f"Efficient graph time: {efficient_time:.4f}s")
    print(f"Speedup: {inefficient_time / efficient_time:.2f}x")

optimize_computation_graph()

print("\n=== Memory Layout Optimization ===")

def compare_memory_layouts():
    """Compare different memory layouts for performance"""
    
    # Contiguous vs non-contiguous tensors
    size = (100, 100, 100)
    
    # Contiguous tensor
    x_contiguous = torch.randn(*size, requires_grad=True)
    
    # Non-contiguous tensor (transposed)
    x_non_contiguous = torch.randn(size[2], size[1], size[0], requires_grad=True).transpose(0, 2)
    
    print(f"Contiguous tensor: {x_contiguous.is_contiguous()}")
    print(f"Non-contiguous tensor: {x_non_contiguous.is_contiguous()}")
    
    # Benchmark contiguous operations
    start_time = time.time()
    for _ in range(100):
        x_contiguous.grad = None
        y = torch.matmul(x_contiguous, x_contiguous.transpose(-2, -1))
        loss = y.sum()
        loss.backward()
    contiguous_time = time.time() - start_time
    
    # Benchmark non-contiguous operations
    start_time = time.time()
    for _ in range(100):
        x_non_contiguous.grad = None
        y = torch.matmul(x_non_contiguous, x_non_contiguous.transpose(-2, -1))
        loss = y.sum()
        loss.backward()
    non_contiguous_time = time.time() - start_time
    
    # Make contiguous and benchmark
    x_made_contiguous = x_non_contiguous.contiguous()
    start_time = time.time()
    for _ in range(100):
        x_made_contiguous.grad = None
        y = torch.matmul(x_made_contiguous, x_made_contiguous.transpose(-2, -1))
        loss = y.sum()
        loss.backward()
    made_contiguous_time = time.time() - start_time
    
    print(f"Contiguous tensor time: {contiguous_time:.4f}s")
    print(f"Non-contiguous tensor time: {non_contiguous_time:.4f}s")
    print(f"Made contiguous time: {made_contiguous_time:.4f}s")
    print(f"Contiguous speedup: {non_contiguous_time / contiguous_time:.2f}x")

compare_memory_layouts()

print("\n=== Efficient Activation Functions ===")

# Custom efficient activation functions
class EfficientActivations:
    """Efficient implementations of activation functions"""
    
    @staticmethod
    def fast_gelu(x):
        """Fast approximation of GELU"""
        return 0.5 * x * (1.0 + torch.tanh(0.797885 * (x + 0.044715 * torch.pow(x, 3))))
    
    @staticmethod
    def swish(x):
        """Swish activation (SiLU)"""
        return x * torch.sigmoid(x)
    
    @staticmethod
    def mish(x):
        """Mish activation"""
        return x * torch.tanh(torch.nn.functional.softplus(x))

def benchmark_activations():
    """Benchmark different activation functions"""
    x = torch.randn(1000, 1000, requires_grad=True)
    iterations = 100
    
    activations = {
        'ReLU': torch.relu,
        'GELU': torch.nn.functional.gelu,
        'Fast GELU': EfficientActivations.fast_gelu,
        'Swish': EfficientActivations.swish,
        'Mish': EfficientActivations.mish,
        'Tanh': torch.tanh
    }
    
    results = {}
    
    for name, activation in activations.items():
        x.grad = None
        
        start_time = time.time()
        for _ in range(iterations):
            x.grad = None
            y = activation(x)
            loss = y.sum()
            loss.backward()
        
        results[name] = time.time() - start_time
    
    print("Activation function benchmarks:")
    fastest_time = min(results.values())
    for name, time_taken in sorted(results.items(), key=lambda x: x[1]):
        speedup = fastest_time / time_taken
        print(f"  {name}: {time_taken:.4f}s (relative speed: {speedup:.2f}x)")

benchmark_activations()

print("\n=== Gradient Clipping Optimization ===")

def efficient_gradient_clipping(model, max_norm=1.0, norm_type=2.0):
    """Efficient gradient clipping implementation"""
    parameters = [p for p in model.parameters() if p.grad is not None]
    
    if len(parameters) == 0:
        return torch.tensor(0.0)
    
    device = parameters[0].grad.device
    
    if norm_type == torch.inf:
        # L-infinity norm
        norms = [p.grad.detach().abs().max() for p in parameters]
        total_norm = max(norms)
    else:
        # L2 or other Lp norms
        if norm_type == 2.0:
            # Optimized L2 norm computation
            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach()) for p in parameters])
            )
        else:
            # General Lp norm
            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
                norm_type
            )
    
    clip_coef = max_norm / (total_norm + 1e-6)
    
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)
    
    return total_norm

# Test efficient gradient clipping
test_model = LargeModel()
test_input = torch.randn(64, 1024)
test_target = torch.randn(64, 1)

# Create large gradients
output = test_model(test_input)
loss = nn.MSELoss()(output, test_target) * 1000  # Scale up loss
loss.backward()

grad_norm = efficient_gradient_clipping(test_model, max_norm=1.0)
print(f"Gradient norm before clipping: {grad_norm:.6f}")

# Check if gradients were clipped
new_grad_norm = torch.norm(
    torch.stack([torch.norm(p.grad.detach()) for p in test_model.parameters() if p.grad is not None])
)
print(f"Gradient norm after clipping: {new_grad_norm:.6f}")

print("\n=== Batch Size Optimization ===")

def find_optimal_batch_size(model, sample_input, max_memory_mb=1000):
    """Find optimal batch size for memory constraints"""
    model.eval()
    
    if not torch.cuda.is_available():
        print("CUDA not available, using fixed batch size")
        return 32
    
    model = model.cuda()
    sample_input = sample_input.cuda()
    
    batch_size = 1
    max_batch_size = 1
    
    while batch_size <= 1024:  # Reasonable upper limit
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Create batch
            batch = sample_input.unsqueeze(0).repeat(batch_size, 1)
            
            # Forward pass
            with torch.no_grad():
                output = model(batch)
            
            # Check memory usage
            memory_used = torch.cuda.max_memory_allocated() / 1e6
            
            if memory_used < max_memory_mb:
                max_batch_size = batch_size
                batch_size *= 2
            else:
                break
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                break
            else:
                raise e
    
    return max_batch_size

# Find optimal batch size
optimal_bs = find_optimal_batch_size(large_model, torch.randn(1024), max_memory_mb=500)
print(f"Optimal batch size for 500MB memory limit: {optimal_bs}")

print("\n=== Autograd Context Management ===")

class AutogradContextManager:
    """Efficient autograd context management"""
    
    def __init__(self):
        self.saved_states = []
    
    def save_state(self):
        """Save current autograd state"""
        state = {
            'grad_enabled': torch.is_grad_enabled(),
            'anomaly_enabled': torch.is_anomaly_enabled()
        }
        self.saved_states.append(state)
    
    def restore_state(self):
        """Restore previous autograd state"""
        if self.saved_states:
            state = self.saved_states.pop()
            torch.set_grad_enabled(state['grad_enabled'])
            # Note: anomaly detection state restoration is limited
    
    def __enter__(self):
        self.save_state()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore_state()

# Test context manager
with AutogradContextManager():
    torch.set_grad_enabled(False)
    # Perform inference operations
    test_output = large_model(torch.randn(1, 1024))
    print(f"Inference completed, grad enabled: {torch.is_grad_enabled()}")

print(f"Context restored, grad enabled: {torch.is_grad_enabled()}")

print("\n=== Performance Profiling Tools ===")

def profile_autograd_performance(model, input_data, target, iterations=10):
    """Comprehensive autograd performance profiling"""
    model.train()
    criterion = nn.MSELoss()
    
    # Warmup
    for _ in range(5):
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()
        model.zero_grad()
    
    # Profile forward pass
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    for _ in range(iterations):
        output = model(input_data)
        loss = criterion(output, target)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    forward_time = time.time() - start_time
    
    # Profile backward pass
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    for _ in range(iterations):
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()
        model.zero_grad()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    total_time = time.time() - start_time
    backward_time = total_time - forward_time
    
    # Memory profiling
    memory_stats = {}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        output = model(input_data)
        loss = criterion(output, target)
        memory_stats['forward'] = torch.cuda.memory_allocated() / 1e6
        
        loss.backward()
        memory_stats['backward'] = torch.cuda.max_memory_allocated() / 1e6
        
        model.zero_grad()
    
    return {
        'forward_time': forward_time / iterations,
        'backward_time': backward_time / iterations,
        'total_time': total_time / iterations,
        'memory_stats': memory_stats
    }

# Profile performance
perf_stats = profile_autograd_performance(large_model, large_inputs[:32], large_targets[:32])

print("Performance profiling results:")
print(f"  Forward time per iteration: {perf_stats['forward_time']:.6f}s")
print(f"  Backward time per iteration: {perf_stats['backward_time']:.6f}s")
print(f"  Total time per iteration: {perf_stats['total_time']:.6f}s")
print(f"  Backward/Forward ratio: {perf_stats['backward_time']/perf_stats['forward_time']:.2f}")

if perf_stats['memory_stats']:
    print(f"  Forward memory: {perf_stats['memory_stats']['forward']:.2f} MB")
    print(f"  Peak memory: {perf_stats['memory_stats']['backward']:.2f} MB")

print("\n=== Autograd Performance Best Practices ===")

print("Performance Optimization Guidelines:")
print("1. Use gradient accumulation for large effective batch sizes")
print("2. Minimize graph depth and complexity")
print("3. Use in-place operations where safe")
print("4. Ensure tensors are contiguous for better performance")
print("5. Use appropriate data types (float16 vs float32)")
print("6. Clear intermediate variables to save memory")
print("7. Profile before optimizing")

print("\nMemory Optimization:")
print("- Use gradient checkpointing for deep networks")
print("- Clear gradients promptly with zero_grad()")
print("- Use torch.no_grad() for inference")
print("- Monitor memory usage with profiling tools")
print("- Consider mixed precision training")
print("- Batch data efficiently")

print("\nComputation Optimization:")
print("- Choose efficient activation functions")
print("- Use vectorized operations over loops")
print("- Optimize batch sizes for your hardware")
print("- Use appropriate gradient clipping")
print("- Consider graph optimization techniques")
print("- Profile different autograd algorithms")

print("\nCommon Performance Issues:")
print("- Excessive graph depth")
print("- Memory fragmentation")
print("- Non-contiguous tensor operations")
print("- Inefficient gradient accumulation")
print("- Unnecessary gradient computation")
print("- Poor memory layout")

print("\nAdvanced Techniques:")
print("- Custom autograd functions for specialized operations")
print("- Gradient compression for distributed training")
print("- Dynamic computation graphs optimization")
print("- Hardware-specific optimizations")
print("- Kernel fusion for repeated operations")

print("\n=== Autograd Performance Complete ===") 