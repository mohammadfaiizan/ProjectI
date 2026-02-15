import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import gc
import psutil
import os
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import time
import threading

# Sample Models for Memory Analysis
class MemoryTestModel(nn.Module):
    """Model designed for memory testing"""
    
    def __init__(self, depth=5, width=128):
        super().__init__()
        
        layers = []
        in_features = 784  # 28x28 flattened
        
        for i in range(depth):
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_features = width
        
        layers.append(nn.Linear(width, 10))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)

class ConvMemoryModel(nn.Module):
    """Convolutional model for memory testing"""
    
    def __init__(self, base_channels=64, num_layers=8):
        super().__init__()
        
        layers = []
        in_channels = 3
        
        for i in range(num_layers):
            out_channels = base_channels * (2 ** min(i, 3))  # Cap at 8x base
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
            
            if i % 2 == 1:  # Downsample every other layer
                layers.append(nn.MaxPool2d(2, 2))
            
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Memory Profiler Class
class MemoryProfiler:
    """Comprehensive memory profiling utilities"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.memory_log = []
        self.monitoring = False
        self.monitor_thread = None
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory information"""
        info = {}
        
        # System memory
        process = psutil.Process(os.getpid())
        sys_memory = process.memory_info()
        info['system_rss_mb'] = sys_memory.rss / 1024**2
        info['system_vms_mb'] = sys_memory.vms / 1024**2
        
        # GPU memory (if available)
        if torch.cuda.is_available() and self.device.startswith('cuda'):
            info['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024**2
            info['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024**2
            info['gpu_max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024**2
            info['gpu_max_reserved_mb'] = torch.cuda.max_memory_reserved() / 1024**2
            
            # GPU properties
            props = torch.cuda.get_device_properties(0)
            info['gpu_total_mb'] = props.total_memory / 1024**2
        
        return info
    
    def reset_peak_memory_stats(self):
        """Reset peak memory statistics"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def print_memory_summary(self, title="Memory Summary"):
        """Print current memory usage summary"""
        info = self.get_memory_info()
        
        print(f"\n{title}")
        print("-" * len(title))
        
        print(f"System Memory:")
        print(f"  RSS: {info['system_rss_mb']:.1f} MB")
        print(f"  VMS: {info['system_vms_mb']:.1f} MB")
        
        if 'gpu_allocated_mb' in info:
            print(f"GPU Memory:")
            print(f"  Allocated: {info['gpu_allocated_mb']:.1f} MB")
            print(f"  Reserved: {info['gpu_reserved_mb']:.1f} MB")
            print(f"  Peak Allocated: {info['gpu_max_allocated_mb']:.1f} MB")
            print(f"  Peak Reserved: {info['gpu_max_reserved_mb']:.1f} MB")
            print(f"  Total GPU: {info['gpu_total_mb']:.1f} MB")
            print(f"  Utilization: {info['gpu_allocated_mb']/info['gpu_total_mb']*100:.1f}%")
    
    def start_continuous_monitoring(self, interval=0.5):
        """Start continuous memory monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.memory_log = []
        
        def monitor():
            while self.monitoring:
                timestamp = time.time()
                memory_info = self.get_memory_info()
                memory_info['timestamp'] = timestamp
                self.memory_log.append(memory_info)
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_continuous_monitoring(self):
        """Stop continuous memory monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def plot_memory_usage(self, save_path="memory_usage.png"):
        """Plot memory usage over time"""
        if not self.memory_log:
            print("No memory data to plot")
            return
        
        # Extract data
        timestamps = [entry['timestamp'] for entry in self.memory_log]
        start_time = timestamps[0]
        rel_times = [(t - start_time) for t in timestamps]
        
        sys_memory = [entry['system_rss_mb'] for entry in self.memory_log]
        
        # Create plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # System memory plot
        axes[0].plot(rel_times, sys_memory, 'b-', label='System RSS')
        axes[0].set_ylabel('Memory (MB)')
        axes[0].set_title('System Memory Usage Over Time')
        axes[0].grid(True)
        axes[0].legend()
        
        # GPU memory plot (if available)
        if 'gpu_allocated_mb' in self.memory_log[0]:
            gpu_allocated = [entry['gpu_allocated_mb'] for entry in self.memory_log]
            gpu_reserved = [entry['gpu_reserved_mb'] for entry in self.memory_log]
            
            axes[1].plot(rel_times, gpu_allocated, 'r-', label='GPU Allocated')
            axes[1].plot(rel_times, gpu_reserved, 'r--', label='GPU Reserved')
            axes[1].set_xlabel('Time (seconds)')
            axes[1].set_ylabel('GPU Memory (MB)')
            axes[1].set_title('GPU Memory Usage Over Time')
            axes[1].grid(True)
            axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, 'No GPU Memory Data', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('GPU Memory (Not Available)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return save_path

class LayerMemoryAnalyzer:
    """Analyze memory usage of individual layers"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.hooks = []
        self.layer_memory = {}
        self.activation_sizes = {}
    
    def _forward_hook(self, name):
        """Create forward hook to measure layer memory"""
        def hook(module, input, output):
            # Measure activation memory
            if isinstance(output, torch.Tensor):
                activation_memory = output.numel() * output.element_size() / 1024**2  # MB
                self.activation_sizes[name] = {
                    'shape': list(output.shape),
                    'memory_mb': activation_memory,
                    'dtype': str(output.dtype)
                }
            
            # GPU memory snapshot
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                self.layer_memory[name] = {
                    'gpu_allocated_mb': allocated,
                    'gpu_reserved_mb': reserved
                }
        
        return hook
    
    def register_hooks(self):
        """Register hooks for all layers"""
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                hook = self._forward_hook(name)
                handle = module.register_forward_hook(hook)
                self.hooks.append(handle)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def analyze_model(self, input_tensor: torch.Tensor):
        """Analyze memory usage of the model"""
        self.register_hooks()
        
        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        self.remove_hooks()
        
        return self.layer_memory, self.activation_sizes
    
    def print_layer_analysis(self):
        """Print layer-wise memory analysis"""
        print("\nLayer-wise Memory Analysis")
        print("-" * 50)
        print(f"{'Layer':<30} {'Shape':<20} {'Memory (MB)':<12} {'GPU Alloc (MB)':<15}")
        print("-" * 80)
        
        for name in self.activation_sizes:
            activation = self.activation_sizes[name]
            memory_info = self.layer_memory.get(name, {})
            
            shape_str = str(activation['shape'])[:18]
            print(f"{name[:28]:<30} {shape_str:<20} {activation['memory_mb']:<12.2f} "
                  f"{memory_info.get('gpu_allocated_mb', 0):<15.1f}")

class MemoryLeakDetector:
    """Detect memory leaks in PyTorch models"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.baseline_memory = None
        self.memory_history = []
    
    def set_baseline(self):
        """Set baseline memory usage"""
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        time.sleep(1)  # Allow memory to settle
        
        if torch.cuda.is_available():
            self.baseline_memory = torch.cuda.memory_allocated()
        else:
            process = psutil.Process(os.getpid())
            self.baseline_memory = process.memory_info().rss
        
        print(f"Baseline memory set: {self.baseline_memory / 1024**2:.1f} MB")
    
    def check_memory_leak(self, iteration: int, threshold_mb: float = 10.0):
        """Check for memory leaks"""
        # Force cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        time.sleep(0.1)
        
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated()
        else:
            process = psutil.Process(os.getpid())
            current_memory = process.memory_info().rss
        
        memory_increase = (current_memory - self.baseline_memory) / 1024**2
        
        self.memory_history.append({
            'iteration': iteration,
            'memory_mb': current_memory / 1024**2,
            'increase_mb': memory_increase
        })
        
        if memory_increase > threshold_mb:
            print(f"WARNING: Potential memory leak detected at iteration {iteration}")
            print(f"Memory increase: {memory_increase:.1f} MB (threshold: {threshold_mb} MB)")
            return True
        
        return False
    
    def plot_memory_trend(self):
        """Plot memory usage trend"""
        if not self.memory_history:
            print("No memory history to plot")
            return
        
        iterations = [entry['iteration'] for entry in self.memory_history]
        memory_usage = [entry['memory_mb'] for entry in self.memory_history]
        memory_increase = [entry['increase_mb'] for entry in self.memory_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Total memory usage
        ax1.plot(iterations, memory_usage, 'b-o', markersize=3)
        ax1.set_ylabel('Memory Usage (MB)')
        ax1.set_title('Total Memory Usage Over Iterations')
        ax1.grid(True)
        
        # Memory increase from baseline
        ax2.plot(iterations, memory_increase, 'r-o', markersize=3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Memory Increase (MB)')
        ax2.set_title('Memory Increase from Baseline')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('memory_leak_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()

class BatchSizeMemoryAnalyzer:
    """Analyze memory usage for different batch sizes"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def find_max_batch_size(self, input_shape: Tuple[int, ...], 
                          max_batch_size: int = 1024, 
                          memory_limit_mb: Optional[float] = None) -> int:
        """Find maximum batch size that fits in memory"""
        
        if memory_limit_mb is None and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            memory_limit_mb = props.total_memory / 1024**2 * 0.9  # Use 90% of GPU memory
        
        self.model.eval()
        
        low, high = 1, max_batch_size
        max_working_batch_size = 1
        
        while low <= high:
            mid = (low + high) // 2
            
            try:
                # Test batch size
                input_tensor = torch.randn(mid, *input_shape).to(self.device)
                
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                
                with torch.no_grad():
                    _ = self.model(input_tensor)
                
                # Check memory usage
                if torch.cuda.is_available():
                    memory_used = torch.cuda.max_memory_allocated() / 1024**2
                else:
                    memory_used = 0  # Cannot reliably measure CPU memory for single operation
                
                if memory_limit_mb is None or memory_used <= memory_limit_mb:
                    max_working_batch_size = mid
                    low = mid + 1
                else:
                    high = mid - 1
                
                # Cleanup
                del input_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    high = mid - 1
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise e
        
        return max_working_batch_size
    
    def analyze_batch_memory_scaling(self, input_shape: Tuple[int, ...], 
                                   batch_sizes: List[int] = None) -> Dict[int, Dict[str, float]]:
        """Analyze memory scaling with batch size"""
        
        if batch_sizes is None:
            max_bs = self.find_max_batch_size(input_shape, max_batch_size=256)
            batch_sizes = [2**i for i in range(0, int(np.log2(max_bs)) + 1)]
        
        results = {}
        
        for batch_size in batch_sizes:
            try:
                input_tensor = torch.randn(batch_size, *input_shape).to(self.device)
                
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                
                # Measure forward pass memory
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                
                if torch.cuda.is_available():
                    forward_memory = torch.cuda.max_memory_allocated() / 1024**2
                    torch.cuda.reset_peak_memory_stats()
                else:
                    forward_memory = 0
                
                # Measure backward pass memory (if possible)
                try:
                    self.model.train()
                    self.model.zero_grad()
                    outputs = self.model(input_tensor)
                    loss = outputs.sum()  # Dummy loss
                    loss.backward()
                    
                    if torch.cuda.is_available():
                        backward_memory = torch.cuda.max_memory_allocated() / 1024**2
                    else:
                        backward_memory = 0
                    
                except RuntimeError:
                    backward_memory = float('inf')  # OOM during backward
                
                results[batch_size] = {
                    'forward_memory_mb': forward_memory,
                    'backward_memory_mb': backward_memory,
                    'total_memory_mb': forward_memory + backward_memory
                }
                
                # Cleanup
                del input_tensor, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    results[batch_size] = {
                        'forward_memory_mb': float('inf'),
                        'backward_memory_mb': float('inf'),
                        'total_memory_mb': float('inf'),
                        'error': 'OOM'
                    }
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    break
                else:
                    raise e
        
        return results
    
    def plot_memory_scaling(self, results: Dict[int, Dict[str, float]]):
        """Plot memory scaling results"""
        valid_results = {bs: data for bs, data in results.items() 
                        if 'error' not in data and data['forward_memory_mb'] != float('inf')}
        
        if not valid_results:
            print("No valid results to plot")
            return
        
        batch_sizes = list(valid_results.keys())
        forward_memory = [valid_results[bs]['forward_memory_mb'] for bs in batch_sizes]
        backward_memory = [valid_results[bs]['backward_memory_mb'] for bs in batch_sizes]
        
        plt.figure(figsize=(10, 6))
        plt.plot(batch_sizes, forward_memory, 'b-o', label='Forward Pass')
        plt.plot(batch_sizes, backward_memory, 'r-o', label='Forward + Backward')
        
        plt.xlabel('Batch Size')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage vs Batch Size')
        plt.legend()
        plt.grid(True)
        plt.xscale('log', base=2)
        
        # Mark OOM point
        oom_batch_sizes = [bs for bs, data in results.items() if 'error' in data]
        if oom_batch_sizes:
            min_oom = min(oom_batch_sizes)
            plt.axvline(x=min_oom, color='red', linestyle='--', alpha=0.7, label=f'OOM at batch size {min_oom}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('batch_memory_scaling.png', dpi=150, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    print("Memory Profiling")
    print("=" * 20)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test models
    small_model = MemoryTestModel(depth=3, width=64).to(device)
    large_model = MemoryTestModel(depth=8, width=256).to(device)
    conv_model = ConvMemoryModel(base_channels=32, num_layers=6).to(device)
    
    print("\n1. Basic Memory Profiling")
    print("-" * 30)
    
    # Basic memory profiling
    profiler = MemoryProfiler(device)
    profiler.print_memory_summary("Initial State")
    
    # Load a model and check memory
    print("\nAfter loading models:")
    profiler.print_memory_summary("After Model Loading")
    
    print("\n2. Layer-wise Memory Analysis")
    print("-" * 35)
    
    # Analyze layer memory
    input_tensor = torch.randn(8, 28, 28).to(device)
    
    analyzer = LayerMemoryAnalyzer(small_model, device)
    layer_memory, activation_sizes = analyzer.analyze_model(input_tensor)
    analyzer.print_layer_analysis()
    
    print("\n3. Continuous Memory Monitoring")
    print("-" * 40)
    
    # Start continuous monitoring
    profiler.start_continuous_monitoring(interval=0.2)
    
    print("Running training simulation with memory monitoring...")
    
    # Simulate training with memory monitoring
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(small_model.parameters(), lr=0.001)
    
    small_model.train()
    for epoch in range(3):
        for batch in range(10):
            # Create batch data
            data = torch.randn(16, 28, 28).to(device)
            targets = torch.randint(0, 10, (16,)).to(device)
            
            # Training step
            optimizer.zero_grad()
            outputs = small_model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            time.sleep(0.1)  # Simulate processing time
    
    profiler.stop_continuous_monitoring()
    print("Memory monitoring completed!")
    
    # Plot memory usage
    profiler.plot_memory_usage("training_memory_usage.png")
    
    print("\n4. Memory Leak Detection")
    print("-" * 30)
    
    # Memory leak detection
    leak_detector = MemoryLeakDetector(device)
    leak_detector.set_baseline()
    
    print("Testing for memory leaks...")
    
    # Simulate potential memory leak scenario
    for i in range(20):
        # Create and process data
        data = torch.randn(32, 28, 28).to(device)
        
        with torch.no_grad():
            outputs = small_model(data)
            # Simulate forgetting to delete references
            if i % 5 == 0:  # Intentionally keep some references
                leaked_data = data.clone()
        
        # Check for leaks
        is_leak = leak_detector.check_memory_leak(i, threshold_mb=5.0)
        if is_leak and i > 10:  # Allow some warmup
            print(f"Memory leak detected at iteration {i}")
    
    # Plot memory trend
    leak_detector.plot_memory_trend()
    
    print("\n5. Batch Size Memory Analysis")
    print("-" * 35)
    
    # Analyze different batch sizes
    batch_analyzer = BatchSizeMemoryAnalyzer(conv_model, device)
    
    # Find maximum batch size
    max_batch_size = batch_analyzer.find_max_batch_size(
        input_shape=(3, 32, 32),
        max_batch_size=512
    )
    print(f"Maximum batch size for conv model: {max_batch_size}")
    
    # Analyze memory scaling
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    if max_batch_size < 64:
        batch_sizes = [bs for bs in batch_sizes if bs <= max_batch_size]
    
    scaling_results = batch_analyzer.analyze_batch_memory_scaling(
        input_shape=(3, 32, 32),
        batch_sizes=batch_sizes
    )
    
    print("\nBatch Size Memory Scaling:")
    print(f"{'Batch Size':<12} {'Forward (MB)':<15} {'Forward+Back (MB)':<18}")
    print("-" * 50)
    
    for bs, data in scaling_results.items():
        if 'error' not in data:
            print(f"{bs:<12} {data['forward_memory_mb']:<15.1f} {data['backward_memory_mb']:<18.1f}")
        else:
            print(f"{bs:<12} {'OOM':<15} {'OOM':<18}")
    
    # Plot scaling results
    batch_analyzer.plot_memory_scaling(scaling_results)
    
    print("\n6. Model Comparison")
    print("-" * 20)
    
    # Compare memory usage of different models
    models = {
        'Small Model': small_model,
        'Large Model': large_model,
        'Conv Model': conv_model
    }
    
    comparison_input = {
        'Small Model': torch.randn(16, 28, 28).to(device),
        'Large Model': torch.randn(16, 28, 28).to(device),
        'Conv Model': torch.randn(16, 3, 32, 32).to(device)
    }
    
    print("Model Memory Comparison:")
    print(f"{'Model':<15} {'Parameters':<12} {'Forward (MB)':<15} {'Peak (MB)':<12}")
    print("-" * 60)
    
    for name, model in models.items():
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        
        # Measure memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        model.eval()
        with torch.no_grad():
            _ = model(comparison_input[name])
        
        if torch.cuda.is_available():
            forward_memory = torch.cuda.memory_allocated() / 1024**2
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        else:
            forward_memory = 0
            peak_memory = 0
        
        print(f"{name:<15} {param_count:<12,} {forward_memory:<15.1f} {peak_memory:<12.1f}")
    
    print("\n7. Memory Optimization Tips")
    print("-" * 35)
    
    print("Memory Optimization Demonstrations:")
    
    # Gradient checkpointing example
    print("\n• Gradient Checkpointing:")
    
    class CheckpointModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(784, 512) for _ in range(10)
            ])
            self.output = nn.Linear(512, 10)
        
        def forward(self, x):
            x = x.view(x.size(0), -1)
            for layer in self.layers:
                x = torch.utils.checkpoint.checkpoint(layer, x)
            return self.output(x)
    
    checkpoint_model = CheckpointModel().to(device)
    
    # Compare with and without checkpointing
    test_input = torch.randn(64, 28, 28).to(device)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    checkpoint_model.train()
    outputs = checkpoint_model(test_input)
    loss = outputs.sum()
    loss.backward()
    
    if torch.cuda.is_available():
        checkpoint_memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  With gradient checkpointing: {checkpoint_memory:.1f} MB")
    
    # Mixed precision example
    print("\n• Mixed Precision Training:")
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
        scaler = torch.cuda.amp.GradScaler()
        
        small_model.train()
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = small_model(test_input[:32])  # Smaller batch for demo
            loss = criterion(outputs, torch.randint(0, 10, (32,)).to(device))
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        mixed_precision_memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  With mixed precision: {mixed_precision_memory:.1f} MB")
    
    print("\n• Memory Cleanup:")
    
    # Show memory cleanup effectiveness
    if torch.cuda.is_available():
        before_cleanup = torch.cuda.memory_allocated() / 1024**2
        print(f"  Before cleanup: {before_cleanup:.1f} MB")
        
        # Manual cleanup
        del test_input, outputs, loss
        gc.collect()
        torch.cuda.empty_cache()
        
        after_cleanup = torch.cuda.memory_allocated() / 1024**2
        print(f"  After cleanup: {after_cleanup:.1f} MB")
        print(f"  Memory freed: {before_cleanup - after_cleanup:.1f} MB")
    
    print("\nMemory profiling demonstrations completed!")
    print("Generated files:")
    print("  - training_memory_usage.png")
    print("  - memory_leak_analysis.png")
    print("  - batch_memory_scaling.png")