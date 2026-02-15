#!/usr/bin/env python3
"""PyTorch Autograd Profiling - Profiling gradient computation"""

import torch
import torch.nn as nn
import time
import gc

print("=== Autograd Profiling Overview ===")

print("Autograd profiling helps analyze:")
print("1. Gradient computation time")
print("2. Memory usage during backprop")
print("3. Bottlenecks in gradient flow")
print("4. Autograd overhead")
print("5. Optimization opportunities")

print("\n=== Basic Gradient Timing ===")

def time_gradient_computation(model, input_data, target, iterations=100):
    """Time gradient computation"""
    criterion = nn.MSELoss()
    
    # Warmup
    for _ in range(10):
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()
        model.zero_grad()
    
    # Time forward pass
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    for _ in range(iterations):
        output = model(input_data)
        loss = criterion(output, target)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    forward_time = time.time() - start_time
    
    # Time backward pass
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
    
    return forward_time, backward_time, total_time

# Test basic timing
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

model = SimpleModel()
input_data = torch.randn(64, 512, requires_grad=True)
target = torch.randn(64, 1)

forward_t, backward_t, total_t = time_gradient_computation(model, input_data, target, 50)

print(f"Forward time: {forward_t:.4f}s")
print(f"Backward time: {backward_t:.4f}s")
print(f"Total time: {total_t:.4f}s")
print(f"Backward/Forward ratio: {backward_t/forward_t:.2f}")

print("\n=== PyTorch Profiler for Autograd ===")

# PyTorch profiler for detailed analysis
from torch.profiler import profile, record_function, ProfilerActivity

def profile_gradient_computation(model, input_data, target):
    """Profile gradient computation with PyTorch profiler"""
    criterion = nn.MSELoss()
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        
        with record_function("forward_pass"):
            output = model(input_data)
            loss = criterion(output, target)
        
        with record_function("backward_pass"):
            loss.backward()
        
        with record_function("optimizer_step"):
            model.zero_grad()
    
    return prof

# Profile the model
profiler_result = profile_gradient_computation(model, input_data, target)

print("Profiler Summary:")
print(profiler_result.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# Memory profiling
if torch.cuda.is_available():
    print("\nMemory Summary:")
    print(profiler_result.key_averages().table(sort_by="cuda_memory_usage", row_limit=5))

print("\n=== Memory Profiling During Autograd ===")

def profile_memory_usage(model, input_data, target):
    """Profile memory usage during gradient computation"""
    criterion = nn.MSELoss()
    
    # Clear cache and reset stats
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    memory_stats = {}
    
    # Memory before forward
    if torch.cuda.is_available():
        memory_stats['before_forward'] = torch.cuda.memory_allocated() / 1e6
    
    # Forward pass
    output = model(input_data)
    loss = criterion(output, target)
    
    if torch.cuda.is_available():
        memory_stats['after_forward'] = torch.cuda.memory_allocated() / 1e6
    
    # Backward pass
    loss.backward()
    
    if torch.cuda.is_available():
        memory_stats['after_backward'] = torch.cuda.memory_allocated() / 1e6
        memory_stats['peak_memory'] = torch.cuda.max_memory_allocated() / 1e6
    
    model.zero_grad()
    
    if torch.cuda.is_available():
        memory_stats['after_cleanup'] = torch.cuda.memory_allocated() / 1e6
    
    return memory_stats

# Profile memory usage
if torch.cuda.is_available():
    model_cuda = model.cuda()
    input_cuda = input_data.cuda()
    target_cuda = target.cuda()
    
    mem_stats = profile_memory_usage(model_cuda, input_cuda, target_cuda)
    
    print("Memory Usage (MB):")
    for stage, memory in mem_stats.items():
        print(f"  {stage}: {memory:.2f}")
    
    forward_memory = mem_stats['after_forward'] - mem_stats['before_forward']
    backward_memory = mem_stats['after_backward'] - mem_stats['after_forward']
    
    print(f"\nMemory increases:")
    print(f"  Forward pass: {forward_memory:.2f} MB")
    print(f"  Backward pass: {backward_memory:.2f} MB")

print("\n=== Layer-wise Gradient Analysis ===")

class InstrumentedModel(nn.Module):
    """Model with gradient analysis hooks"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        
        self.gradient_stats = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to monitor gradients"""
        def make_hook(name):
            def hook(grad):
                self.gradient_stats[name] = {
                    'norm': grad.norm().item(),
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'max': grad.max().item(),
                    'min': grad.min().item()
                }
                return grad
            return hook
        
        self.fc1.weight.register_hook(make_hook('fc1_weight'))
        self.fc1.bias.register_hook(make_hook('fc1_bias'))
        self.fc2.weight.register_hook(make_hook('fc2_weight'))
        self.fc2.bias.register_hook(make_hook('fc2_bias'))
        self.fc3.weight.register_hook(make_hook('fc3_weight'))
        self.fc3.bias.register_hook(make_hook('fc3_bias'))
        self.fc4.weight.register_hook(make_hook('fc4_weight'))
        self.fc4.bias.register_hook(make_hook('fc4_bias'))
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def print_gradient_stats(self):
        """Print gradient statistics"""
        print("Layer-wise Gradient Statistics:")
        for name, stats in self.gradient_stats.items():
            print(f"  {name}:")
            print(f"    Norm: {stats['norm']:.6f}")
            print(f"    Mean: {stats['mean']:.6f}")
            print(f"    Std:  {stats['std']:.6f}")

# Test instrumented model
instrumented_model = InstrumentedModel()
instrumented_input = torch.randn(32, 256)
instrumented_target = torch.randn(32, 1)

output = instrumented_model(instrumented_input)
loss = nn.MSELoss()(output, instrumented_target)
loss.backward()

instrumented_model.print_gradient_stats()

print("\n=== Gradient Computation Graph Analysis ===")

def analyze_computation_graph(tensor):
    """Analyze the computation graph structure"""
    graph_info = {
        'nodes': 0,
        'leaf_nodes': 0,
        'function_types': {},
        'depth': 0
    }
    
    def traverse_graph(node, depth=0):
        if node is None:
            return
        
        graph_info['nodes'] += 1
        graph_info['depth'] = max(graph_info['depth'], depth)
        
        if hasattr(node, 'grad_fn') and node.grad_fn is not None:
            func_name = type(node.grad_fn).__name__
            graph_info['function_types'][func_name] = graph_info['function_types'].get(func_name, 0) + 1
            
            if hasattr(node.grad_fn, 'next_functions'):
                for next_fn, _ in node.grad_fn.next_functions:
                    if next_fn is not None:
                        # Create a mock tensor to continue traversal
                        mock_tensor = type('MockTensor', (), {'grad_fn': next_fn})()
                        traverse_graph(mock_tensor, depth + 1)
        else:
            graph_info['leaf_nodes'] += 1
    
    traverse_graph(tensor)
    return graph_info

# Analyze computation graph
complex_x = torch.randn(5, 5, requires_grad=True)
complex_y = torch.sin(complex_x)
complex_z = torch.matmul(complex_y, complex_y.t())
complex_w = torch.exp(complex_z).sum()

graph_analysis = analyze_computation_graph(complex_w)

print("Computation Graph Analysis:")
print(f"  Total nodes: {graph_analysis['nodes']}")
print(f"  Leaf nodes: {graph_analysis['leaf_nodes']}")
print(f"  Graph depth: {graph_analysis['depth']}")
print(f"  Function types: {graph_analysis['function_types']}")

print("\n=== Autograd Overhead Analysis ===")

def measure_autograd_overhead(func, inputs, iterations=1000):
    """Measure autograd overhead"""
    
    # Without gradients
    inputs_no_grad = [inp.detach() for inp in inputs]
    
    start_time = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            result = func(*inputs_no_grad)
            if isinstance(result, torch.Tensor):
                _ = result.sum()  # Ensure computation happens
    no_grad_time = time.time() - start_time
    
    # With gradients (forward only)
    start_time = time.time()
    for _ in range(iterations):
        result = func(*inputs)
        if isinstance(result, torch.Tensor):
            _ = result.sum()  # Ensure computation happens
    with_grad_time = time.time() - start_time
    
    # With gradients (forward + backward)
    start_time = time.time()
    for _ in range(iterations):
        for inp in inputs:
            if inp.grad is not None:
                inp.grad.zero_()
        
        result = func(*inputs)
        if isinstance(result, torch.Tensor):
            loss = result.sum()
            loss.backward()
    full_autograd_time = time.time() - start_time
    
    return no_grad_time, with_grad_time, full_autograd_time

# Test overhead
def test_function(x, y):
    return torch.matmul(x, y) + torch.sin(x).sum()

x_overhead = torch.randn(100, 100, requires_grad=True)
y_overhead = torch.randn(100, 100, requires_grad=True)

no_grad_t, with_grad_t, full_t = measure_autograd_overhead(
    test_function, [x_overhead, y_overhead], 100
)

print("Autograd Overhead Analysis:")
print(f"  No grad time: {no_grad_t:.4f}s")
print(f"  With grad time: {with_grad_t:.4f}s")
print(f"  Full autograd time: {full_t:.4f}s")
print(f"  Forward overhead: {((with_grad_t - no_grad_t) / no_grad_t * 100):.1f}%")
print(f"  Backward overhead: {((full_t - with_grad_t) / with_grad_t * 100):.1f}%")

print("\n=== Custom Profiling Tools ===")

class AutogradProfiler:
    """Custom profiler for autograd operations"""
    def __init__(self):
        self.events = []
        self.start_time = None
    
    def start(self):
        """Start profiling"""
        self.events = []
        self.start_time = time.time()
    
    def record_event(self, name):
        """Record a profiling event"""
        if self.start_time is not None:
            self.events.append({
                'name': name,
                'time': time.time() - self.start_time,
                'memory': torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
            })
    
    def stop(self):
        """Stop profiling and return results"""
        if self.start_time is not None:
            total_time = time.time() - self.start_time
            self.start_time = None
            return {
                'events': self.events,
                'total_time': total_time
            }
        return None
    
    def print_summary(self):
        """Print profiling summary"""
        if not self.events:
            print("No profiling events recorded")
            return
        
        print("Custom Profiling Summary:")
        prev_time = 0
        for event in self.events:
            duration = event['time'] - prev_time
            print(f"  {event['name']}: {duration:.6f}s, Memory: {event['memory']:.2f}MB")
            prev_time = event['time']

# Test custom profiler
profiler = AutogradProfiler()
profiler.start()

x_prof = torch.randn(200, 200, requires_grad=True)
profiler.record_event("input_created")

y_prof = torch.matmul(x_prof, x_prof.t())
profiler.record_event("matmul_completed")

z_prof = torch.sin(y_prof)
profiler.record_event("sin_completed")

loss_prof = z_prof.sum()
profiler.record_event("sum_completed")

loss_prof.backward()
profiler.record_event("backward_completed")

profiler.stop()
profiler.print_summary()

print("\n=== Gradient Computation Bottleneck Detection ===")

def detect_bottlenecks(model, input_data, target, threshold_ms=10):
    """Detect bottlenecks in gradient computation"""
    criterion = nn.MSELoss()
    bottlenecks = []
    
    # Hook to measure time for each operation
    def create_timing_hook(name):
        def hook(module, input, output):
            start_time = time.time()
            # The actual computation happens before this hook
            # So we'll measure the next operation indirectly
            pass
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            hooks.append(module.register_forward_hook(create_timing_hook(name)))
    
    # Time individual operations using profiler
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()
    
    # Analyze profiler results for bottlenecks
    events = prof.key_averages()
    for event in events:
        if event.cpu_time_total > threshold_ms * 1000:  # Convert ms to µs
            bottlenecks.append({
                'name': event.key,
                'cpu_time': event.cpu_time_total / 1000,  # Convert to ms
                'count': event.count
            })
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    return bottlenecks

# Detect bottlenecks
bottlenecks = detect_bottlenecks(model, input_data, target)

print("Gradient Computation Bottlenecks (>10ms):")
if bottlenecks:
    for bottleneck in sorted(bottlenecks, key=lambda x: x['cpu_time'], reverse=True):
        print(f"  {bottleneck['name']}: {bottleneck['cpu_time']:.2f}ms ({bottleneck['count']} calls)")
else:
    print("  No bottlenecks detected")

print("\n=== Gradient Checkpointing Profiling ===")

import torch.utils.checkpoint as checkpoint

def profile_checkpointing_overhead(func, inputs, use_checkpoint=True):
    """Profile checkpointing overhead"""
    
    # Warmup
    for _ in range(5):
        if use_checkpoint:
            output = checkpoint.checkpoint(func, *inputs)
        else:
            output = func(*inputs)
        loss = output.sum()
        loss.backward()
        for inp in inputs:
            if inp.grad is not None:
                inp.grad.zero_()
    
    # Profile
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        if use_checkpoint:
            output = checkpoint.checkpoint(func, *inputs)
        else:
            output = func(*inputs)
        loss = output.sum()
        loss.backward()
    
    return prof

def expensive_function(x):
    """Expensive function for checkpointing test"""
    for _ in range(10):
        x = torch.relu(x)
        x = torch.tanh(x)
    return x

checkpoint_input = torch.randn(100, 100, requires_grad=True)

# Profile without checkpointing
prof_no_cp = profile_checkpointing_overhead(expensive_function, [checkpoint_input], False)

# Profile with checkpointing
checkpoint_input.grad = None
prof_cp = profile_checkpointing_overhead(expensive_function, [checkpoint_input], True)

print("Checkpointing Profiling:")
print("\nWithout checkpointing:")
print(prof_no_cp.key_averages().table(sort_by="cpu_time_total", row_limit=5))

print("\nWith checkpointing:")
print(prof_cp.key_averages().table(sort_by="cpu_time_total", row_limit=5))

print("\n=== Autograd Profiling Best Practices ===")

print("Autograd Profiling Guidelines:")
print("1. Always warm up before profiling")
print("2. Use torch.cuda.synchronize() for accurate GPU timing")
print("3. Profile both forward and backward passes separately")
print("4. Monitor memory usage alongside timing")
print("5. Use PyTorch profiler for detailed analysis")
print("6. Profile with realistic batch sizes and model complexity")
print("7. Consider profiling different phases of training")

print("\nProfiling Tools:")
print("- torch.profiler for comprehensive analysis")
print("- time.time() for simple timing")
print("- torch.cuda.Event for GPU timing")
print("- Custom hooks for layer-wise analysis")
print("- Memory profiling with torch.cuda.memory_*")

print("\nCommon Bottlenecks:")
print("- Large matrix multiplications")
print("- Complex activation functions")
print("- Memory allocation/deallocation")
print("- Data transfer between CPU/GPU")
print("- Gradient accumulation overhead")
print("- Inefficient autograd graph structures")

print("\nOptimization Strategies:")
print("- Use gradient checkpointing for memory")
print("- Optimize autograd graph structure")
print("- Use in-place operations carefully")
print("- Consider mixed precision training")
print("- Profile different optimization algorithms")
print("- Batch operations when possible")

print("\n=== Autograd Profiling Complete ===") 