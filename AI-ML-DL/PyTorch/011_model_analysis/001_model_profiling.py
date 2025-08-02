import torch
import torch.nn as nn
import torch.profiler
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from collections import defaultdict
import psutil
import os

# Sample Models for Profiling
class SimpleCNN(nn.Module):
    """Simple CNN for profiling demonstrations"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)

class ResidualBlock(nn.Module):
    """Residual block for deeper network"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class DeepResNet(nn.Module):
    """Deeper ResNet for profiling"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Basic Profiling Tools
class SimpleProfiler:
    """Simple timing profiler"""
    
    def __init__(self):
        self.times = {}
        self.call_counts = {}
    
    def start_timer(self, name: str):
        """Start timing an operation"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.times[name] = time.time()
    
    def end_timer(self, name: str):
        """End timing an operation"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        if name in self.times:
            elapsed = time.time() - self.times[name]
            
            if name not in self.call_counts:
                self.call_counts[name] = []
            
            self.call_counts[name].append(elapsed)
            return elapsed
        return 0
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics"""
        stats = {}
        for name, times in self.call_counts.items():
            stats[name] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'count': len(times),
                'total': np.sum(times)
            }
        return stats
    
    def print_stats(self):
        """Print timing statistics"""
        stats = self.get_stats()
        print(f"{'Operation':<25} {'Count':<8} {'Mean (ms)':<12} {'Std (ms)':<12} {'Total (ms)':<12}")
        print("-" * 80)
        
        for name, stat in stats.items():
            print(f"{name:<25} {stat['count']:<8} {stat['mean']*1000:<12.3f} "
                  f"{stat['std']*1000:<12.3f} {stat['total']*1000:<12.3f}")

class ModelProfiler:
    """Comprehensive model profiler"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.hooks = []
        self.layer_times = defaultdict(list)
        self.layer_memory = defaultdict(list)
    
    def _forward_hook(self, name):
        """Create forward hook for timing layers"""
        def hook(module, input, output):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
                self.layer_memory[name].append(memory_used)
        
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
    
    def profile_forward_pass(self, input_tensor: torch.Tensor, num_runs: int = 10):
        """Profile forward pass"""
        self.model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = self.model(input_tensor)
        
        times = []
        memory_usage = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                
                start_time = time.time()
                _ = self.model(input_tensor)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
                
                if torch.cuda.is_available():
                    memory_usage.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB
        
        return {
            'forward_time_ms': {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times)
            },
            'memory_usage_mb': {
                'mean': np.mean(memory_usage) if memory_usage else 0,
                'std': np.std(memory_usage) if memory_usage else 0,
                'peak': np.max(memory_usage) if memory_usage else 0
            }
        }
    
    def profile_backward_pass(self, input_tensor: torch.Tensor, num_runs: int = 10):
        """Profile backward pass"""
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        
        # Create dummy targets
        batch_size = input_tensor.size(0)
        targets = torch.randint(0, 10, (batch_size,)).to(self.device)
        
        times = []
        
        for _ in range(num_runs):
            self.model.zero_grad()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Forward pass
            outputs = self.model(input_tensor)
            loss = criterion(outputs, targets)
            
            # Time backward pass
            start_time = time.time()
            loss.backward()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            'backward_time_ms': {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times)
            }
        }

# PyTorch Profiler Integration
class PyTorchProfiler:
    """PyTorch built-in profiler wrapper"""
    
    def __init__(self, output_dir: str = "./profiler_output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def profile_training_step(self, model: nn.Module, data_loader, 
                            optimizer, criterion, num_steps: int = 10):
        """Profile training steps using PyTorch profiler"""
        
        def trace_handler(prof):
            prof.export_chrome_trace(f"{self.output_dir}/trace.json")
            prof.export_stacks(f"{self.output_dir}/profiler_stacks.txt", "self_cuda_time_total")
        
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=3,
                repeat=2
            ),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as profiler:
            
            model.train()
            for step, (data, targets) in enumerate(data_loader):
                if step >= num_steps:
                    break
                
                data, targets = data.to(model.device if hasattr(model, 'device') else 'cuda'), \
                               targets.to(model.device if hasattr(model, 'device') else 'cuda')
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                profiler.step()
        
        return f"Profile saved to {self.output_dir}"
    
    def profile_inference(self, model: nn.Module, input_tensor: torch.Tensor, 
                         num_runs: int = 100):
        """Profile inference using PyTorch profiler"""
        
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
        ) as profiler:
            
            model.eval()
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model(input_tensor)
        
        # Print key stats
        print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        # Export detailed results
        profiler.export_chrome_trace(f"{self.output_dir}/inference_trace.json")
        
        return profiler.key_averages()

# Throughput Analysis
class ThroughputAnalyzer:
    """Analyze model throughput across different batch sizes"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
    
    def analyze_batch_sizes(self, input_shape: Tuple[int, ...], 
                          batch_sizes: List[int] = None,
                          num_runs: int = 20) -> Dict[int, Dict[str, float]]:
        """Analyze throughput for different batch sizes"""
        
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"Testing batch size: {batch_size}")
            
            # Create input tensor
            input_tensor = torch.randn(batch_size, *input_shape).to(self.device)
            
            try:
                # Profile forward pass
                forward_stats = self._profile_forward(input_tensor, num_runs)
                
                # Calculate throughput
                throughput = batch_size / (forward_stats['mean'] / 1000)  # samples/second
                
                results[batch_size] = {
                    'forward_time_ms': forward_stats['mean'],
                    'throughput_samples_per_sec': throughput,
                    'memory_mb': self._get_memory_usage(input_tensor)
                }
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM at batch size {batch_size}")
                    results[batch_size] = {
                        'forward_time_ms': float('inf'),
                        'throughput_samples_per_sec': 0,
                        'memory_mb': float('inf'),
                        'error': 'OOM'
                    }
                    break
                else:
                    raise e
        
        return results
    
    def _profile_forward(self, input_tensor: torch.Tensor, num_runs: int):
        """Profile forward pass for throughput analysis"""
        self.model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = self.model(input_tensor)
        
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.time()
                _ = self.model(input_tensor)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times)
        }
    
    def _get_memory_usage(self, input_tensor: torch.Tensor) -> float:
        """Get memory usage for given input"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                _ = self.model(input_tensor)
            
            return torch.cuda.max_memory_allocated() / 1024**2  # MB
        return 0
    
    def plot_throughput_analysis(self, results: Dict[int, Dict[str, float]]):
        """Plot throughput analysis results"""
        batch_sizes = []
        throughputs = []
        times = []
        
        for bs, stats in results.items():
            if 'error' not in stats:
                batch_sizes.append(bs)
                throughputs.append(stats['throughput_samples_per_sec'])
                times.append(stats['forward_time_ms'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Throughput plot
        ax1.plot(batch_sizes, throughputs, 'b-o')
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Throughput (samples/sec)')
        ax1.set_title('Throughput vs Batch Size')
        ax1.grid(True)
        
        # Latency plot
        ax2.plot(batch_sizes, times, 'r-o')
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Forward Time (ms)')
        ax2.set_title('Latency vs Batch Size')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('throughput_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()

# Resource Utilization Monitor
class ResourceMonitor:
    """Monitor system resource utilization during model execution"""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.monitoring = False
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        self.timestamps = []
    
    def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        self.timestamps = []
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
    
    def record_snapshot(self):
        """Record a single snapshot of resource usage"""
        if not self.monitoring:
            return
        
        # CPU and RAM
        self.cpu_usage.append(psutil.cpu_percent())
        memory = psutil.virtual_memory()
        self.memory_usage.append(memory.percent)
        
        # GPU (if available)
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
            
            self.gpu_memory.append(gpu_memory_percent)
            # Note: GPU utilization requires nvidia-ml-py package for accurate readings
            self.gpu_usage.append(0)  # Placeholder
        
        self.timestamps.append(time.time())
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get resource utilization statistics"""
        if not self.cpu_usage:
            return {}
        
        stats = {
            'cpu': {
                'mean': np.mean(self.cpu_usage),
                'max': np.max(self.cpu_usage),
                'min': np.min(self.cpu_usage),
                'std': np.std(self.cpu_usage)
            },
            'memory': {
                'mean': np.mean(self.memory_usage),
                'max': np.max(self.memory_usage),
                'min': np.min(self.memory_usage),
                'std': np.std(self.memory_usage)
            }
        }
        
        if self.gpu_memory:
            stats['gpu_memory'] = {
                'mean': np.mean(self.gpu_memory),
                'max': np.max(self.gpu_memory),
                'min': np.min(self.gpu_memory),
                'std': np.std(self.gpu_memory)
            }
        
        return stats
    
    def plot_resource_usage(self):
        """Plot resource usage over time"""
        if not self.timestamps:
            print("No monitoring data available")
            return
        
        # Convert timestamps to relative time
        start_time = self.timestamps[0]
        rel_times = [(t - start_time) for t in self.timestamps]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # CPU usage
        axes[0, 0].plot(rel_times, self.cpu_usage)
        axes[0, 0].set_title('CPU Usage (%)')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Usage %')
        axes[0, 0].grid(True)
        
        # Memory usage
        axes[0, 1].plot(rel_times, self.memory_usage)
        axes[0, 1].set_title('RAM Usage (%)')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Usage %')
        axes[0, 1].grid(True)
        
        # GPU memory
        if self.gpu_memory:
            axes[1, 0].plot(rel_times, self.gpu_memory)
            axes[1, 0].set_title('GPU Memory (%)')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Usage %')
            axes[1, 0].grid(True)
        else:
            axes[1, 0].text(0.5, 0.5, 'No GPU Data', ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Summary stats
        stats = self.get_stats()
        axes[1, 1].axis('off')
        stats_text = "Resource Usage Summary:\n\n"
        for resource, data in stats.items():
            stats_text += f"{resource.upper()}:\n"
            stats_text += f"  Mean: {data['mean']:.1f}%\n"
            stats_text += f"  Max: {data['max']:.1f}%\n\n"
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                       verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('resource_usage.png', dpi=150, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    print("Model Profiling")
    print("=" * 20)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create sample models
    simple_model = SimpleCNN(num_classes=10).to(device)
    deep_model = DeepResNet(num_classes=10).to(device)
    
    # Sample input
    batch_size = 8
    input_tensor = torch.randn(batch_size, 3, 32, 32).to(device)
    
    print("\n1. Simple Profiling")
    print("-" * 20)
    
    # Simple profiler demonstration
    profiler = SimpleProfiler()
    
    # Profile forward passes
    for i in range(10):
        profiler.start_timer("simple_forward")
        with torch.no_grad():
            _ = simple_model(input_tensor)
        profiler.end_timer("simple_forward")
        
        profiler.start_timer("deep_forward")
        with torch.no_grad():
            _ = deep_model(input_tensor)
        profiler.end_timer("deep_forward")
    
    print("Simple Profiler Results:")
    profiler.print_stats()
    
    print("\n2. Comprehensive Model Profiling")
    print("-" * 35)
    
    # Comprehensive profiling
    model_profiler = ModelProfiler(simple_model, device)
    
    # Profile forward pass
    forward_stats = model_profiler.profile_forward_pass(input_tensor, num_runs=20)
    print("Forward Pass Profiling:")
    print(f"  Mean time: {forward_stats['forward_time_ms']['mean']:.2f} ms")
    print(f"  Std time: {forward_stats['forward_time_ms']['std']:.2f} ms")
    print(f"  Peak memory: {forward_stats['memory_usage_mb']['peak']:.1f} MB")
    
    # Profile backward pass
    backward_stats = model_profiler.profile_backward_pass(input_tensor, num_runs=10)
    print("\nBackward Pass Profiling:")
    print(f"  Mean time: {backward_stats['backward_time_ms']['mean']:.2f} ms")
    print(f"  Std time: {backward_stats['backward_time_ms']['std']:.2f} ms")
    
    print("\n3. Throughput Analysis")
    print("-" * 25)
    
    # Throughput analysis
    throughput_analyzer = ThroughputAnalyzer(simple_model, device)
    
    # Test different batch sizes
    throughput_results = throughput_analyzer.analyze_batch_sizes(
        input_shape=(3, 32, 32),
        batch_sizes=[1, 2, 4, 8, 16, 32],
        num_runs=10
    )
    
    print("Throughput Analysis Results:")
    print(f"{'Batch Size':<12} {'Time (ms)':<12} {'Throughput':<15} {'Memory (MB)':<12}")
    print("-" * 60)
    
    for bs, stats in throughput_results.items():
        if 'error' not in stats:
            print(f"{bs:<12} {stats['forward_time_ms']:<12.2f} "
                  f"{stats['throughput_samples_per_sec']:<15.1f} {stats['memory_mb']:<12.1f}")
        else:
            print(f"{bs:<12} {'OOM':<12} {'OOM':<15} {'OOM':<12}")
    
    # Plot results
    throughput_analyzer.plot_throughput_analysis(throughput_results)
    
    print("\n4. PyTorch Profiler")
    print("-" * 22)
    
    # PyTorch profiler
    pytorch_profiler = PyTorchProfiler("./profiler_results")
    
    # Profile inference
    print("Profiling inference...")
    stats = pytorch_profiler.profile_inference(simple_model, input_tensor, num_runs=50)
    
    print("\nTop operations by CUDA time:")
    for item in stats[:5]:  # Top 5 operations
        print(f"  {item.key}: {item.cuda_time_total/1000:.2f} ms")
    
    print("\n5. Resource Monitoring")
    print("-" * 25)
    
    # Resource monitoring
    resource_monitor = ResourceMonitor(interval=0.1)
    
    print("Monitoring resources during training...")
    resource_monitor.start_monitoring()
    
    # Simulate training with resource monitoring
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    simple_model.train()
    for epoch in range(3):
        for batch in range(10):
            # Record resource snapshot
            resource_monitor.record_snapshot()
            
            # Simulate training step
            data = torch.randn(16, 3, 32, 32).to(device)
            targets = torch.randint(0, 10, (16,)).to(device)
            
            optimizer.zero_grad()
            outputs = simple_model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            time.sleep(0.1)  # Simulate processing time
    
    resource_monitor.stop_monitoring()
    
    # Display resource statistics
    resource_stats = resource_monitor.get_stats()
    print("\nResource Usage Statistics:")
    for resource, stats in resource_stats.items():
        print(f"{resource.upper()}:")
        print(f"  Mean usage: {stats['mean']:.1f}%")
        print(f"  Peak usage: {stats['max']:.1f}%")
    
    # Plot resource usage
    resource_monitor.plot_resource_usage()
    
    print("\n6. Model Comparison")
    print("-" * 20)
    
    # Compare different models
    models = {
        'SimpleCNN': simple_model,
        'DeepResNet': deep_model
    }
    
    comparison_results = {}
    
    for name, model in models.items():
        print(f"\nProfiling {name}...")
        
        # Profile forward pass
        profiler = ModelProfiler(model, device)
        forward_stats = profiler.profile_forward_pass(input_tensor, num_runs=20)
        
        comparison_results[name] = {
            'forward_time': forward_stats['forward_time_ms']['mean'],
            'memory_usage': forward_stats['memory_usage_mb']['peak'],
            'parameters': sum(p.numel() for p in model.parameters())
        }
    
    # Display comparison
    print("\nModel Comparison Results:")
    print(f"{'Model':<15} {'Forward (ms)':<15} {'Memory (MB)':<15} {'Parameters':<15}")
    print("-" * 65)
    
    for name, stats in comparison_results.items():
        print(f"{name:<15} {stats['forward_time']:<15.2f} "
              f"{stats['memory_usage']:<15.1f} {stats['parameters']:<15,}")
    
    # Calculate efficiency metrics
    print("\nEfficiency Metrics:")
    for name, stats in comparison_results.items():
        throughput = batch_size / (stats['forward_time'] / 1000)
        efficiency = throughput / stats['parameters'] * 1e6  # throughput per million params
        print(f"{name}: {efficiency:.2f} samples/sec/M-params")
    
    print("\nModel profiling demonstrations completed!")
    print(f"Results saved to: ./profiler_results/")
    print("Check generated plots: throughput_analysis.png, resource_usage.png")