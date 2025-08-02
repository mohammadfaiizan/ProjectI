import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import platform
from typing import Dict, List, Tuple, Optional, Any
import gc

# Sample Models for Benchmarking
class BenchmarkCNN(nn.Module):
    """CNN for benchmarking purposes"""
    
    def __init__(self, num_classes=10, width_multiplier=1.0):
        super().__init__()
        
        base_width = int(64 * width_multiplier)
        
        self.features = nn.Sequential(
            nn.Conv2d(3, base_width, 3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(base_width, base_width * 2, 3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(base_width * 2, base_width * 4, 3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_width * 4, base_width * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(base_width * 2, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class EfficientCNN(nn.Module):
    """Efficient CNN with depthwise separable convolutions"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Depthwise separable blocks
        self.blocks = nn.Sequential(
            self._make_depthwise_block(32, 64, 2),
            self._make_depthwise_block(64, 128, 2),
            self._make_depthwise_block(128, 256, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def _make_depthwise_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.classifier(x)
        return x

# Performance Benchmarking Suite
class PerformanceBenchmark:
    """Comprehensive performance benchmarking for PyTorch models"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.results = {}
        self.system_info = self._get_system_info()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
        }
        
        if torch.cuda.is_available():
            info.update({
                'cuda_available': True,
                'cuda_version': torch.version.cuda,
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3)
            })
        else:
            info['cuda_available'] = False
        
        return info
    
    def benchmark_inference(self, model: nn.Module, input_shape: Tuple[int, ...],
                          batch_sizes: List[int] = [1, 4, 8, 16, 32],
                          num_runs: int = 100, warmup_runs: int = 10) -> Dict[str, Any]:
        """Benchmark inference performance"""
        
        print(f"Benchmarking inference on {self.device}...")
        
        model = model.to(self.device)
        model.eval()
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size {batch_size}...")
            
            try:
                # Create input tensor
                input_tensor = torch.randn(batch_size, *input_shape).to(self.device)
                
                # Warmup
                with torch.no_grad():
                    for _ in range(warmup_runs):
                        _ = model(input_tensor)
                
                # Benchmark
                times = []
                
                with torch.no_grad():
                    for _ in range(num_runs):
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        start_time = time.perf_counter()
                        _ = model(input_tensor)
                        
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        end_time = time.perf_counter()
                        times.append((end_time - start_time) * 1000)  # Convert to ms
                
                # Calculate statistics
                times = np.array(times)
                
                results[batch_size] = {
                    'mean_time_ms': np.mean(times),
                    'std_time_ms': np.std(times),
                    'min_time_ms': np.min(times),
                    'max_time_ms': np.max(times),
                    'median_time_ms': np.median(times),
                    'p95_time_ms': np.percentile(times, 95),
                    'p99_time_ms': np.percentile(times, 99),
                    'throughput_samples_per_sec': batch_size / (np.mean(times) / 1000),
                    'throughput_batches_per_sec': 1000 / np.mean(times)
                }
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"    OOM at batch size {batch_size}")
                    break
                else:
                    raise e
        
        return results
    
    def benchmark_memory(self, model: nn.Module, input_shape: Tuple[int, ...],
                        batch_sizes: List[int] = [1, 4, 8, 16, 32]) -> Dict[str, Any]:
        """Benchmark memory usage"""
        
        print(f"Benchmarking memory usage on {self.device}...")
        
        model = model.to(self.device)
        model.eval()
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size {batch_size}...")
            
            try:
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                
                gc.collect()
                
                # Measure baseline memory
                if torch.cuda.is_available():
                    baseline_memory = torch.cuda.memory_allocated()
                else:
                    baseline_memory = 0
                
                # Create input and run model
                input_tensor = torch.randn(batch_size, *input_shape).to(self.device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                
                # Measure peak memory
                if torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated()
                    current_memory = torch.cuda.memory_allocated()
                else:
                    peak_memory = 0
                    current_memory = 0
                
                results[batch_size] = {
                    'baseline_memory_mb': baseline_memory / (1024**2),
                    'current_memory_mb': current_memory / (1024**2),
                    'peak_memory_mb': peak_memory / (1024**2),
                    'memory_per_sample_mb': (current_memory - baseline_memory) / (1024**2) / batch_size
                }
                
                # Clean up
                del input_tensor, output
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"    OOM at batch size {batch_size}")
                    break
                else:
                    raise e
        
        return results
    
    def benchmark_training(self, model: nn.Module, input_shape: Tuple[int, ...],
                          num_classes: int = 10, batch_size: int = 32,
                          num_steps: int = 100) -> Dict[str, Any]:
        """Benchmark training performance"""
        
        print(f"Benchmarking training on {self.device}...")
        
        model = model.to(self.device)
        model.train()
        
        # Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Create dummy data
        input_tensor = torch.randn(batch_size, *input_shape).to(self.device)
        targets = torch.randint(0, num_classes, (batch_size,)).to(self.device)
        
        # Warmup
        for _ in range(10):
            optimizer.zero_grad()
            outputs = model(input_tensor)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Benchmark training steps
        step_times = []
        
        for step in range(num_steps):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            optimizer.zero_grad()
            outputs = model(input_tensor)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            step_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        step_times = np.array(step_times)
        
        return {
            'mean_step_time_ms': np.mean(step_times),
            'std_step_time_ms': np.std(step_times),
            'min_step_time_ms': np.min(step_times),
            'max_step_time_ms': np.max(step_times),
            'samples_per_sec': batch_size / (np.mean(step_times) / 1000),
            'steps_per_sec': 1000 / np.mean(step_times)
        }
    
    def compare_models(self, models: Dict[str, nn.Module], input_shape: Tuple[int, ...],
                      test_inference: bool = True, test_memory: bool = True,
                      test_training: bool = True) -> Dict[str, Any]:
        """Compare multiple models"""
        
        print("Comparing models...")
        
        comparison_results = {}
        
        for model_name, model in models.items():
            print(f"\nBenchmarking {model_name}...")
            
            model_results = {
                'model_name': model_name,
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
            
            if test_inference:
                inference_results = self.benchmark_inference(model, input_shape, batch_sizes=[1, 8, 32])
                model_results['inference'] = inference_results
            
            if test_memory:
                memory_results = self.benchmark_memory(model, input_shape, batch_sizes=[1, 8, 32])
                model_results['memory'] = memory_results
            
            if test_training:
                training_results = self.benchmark_training(model, input_shape)
                model_results['training'] = training_results
            
            comparison_results[model_name] = model_results
        
        return comparison_results
    
    def plot_inference_benchmark(self, results: Dict[str, Any], title: str = "Inference Benchmark"):
        """Plot inference benchmark results"""
        
        batch_sizes = list(results.keys())
        mean_times = [results[bs]['mean_time_ms'] for bs in batch_sizes]
        std_times = [results[bs]['std_time_ms'] for bs in batch_sizes]
        throughputs = [results[bs]['throughput_samples_per_sec'] for bs in batch_sizes]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Latency plot
        ax1.errorbar(batch_sizes, mean_times, yerr=std_times, marker='o', capsize=5, linewidth=2)
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title('Inference Latency')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Throughput plot
        ax2.plot(batch_sizes, throughputs, marker='s', linewidth=2, markersize=6)
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Throughput (samples/sec)')
        ax2.set_title('Inference Throughput')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, comparison_results: Dict[str, Any]):
        """Plot model comparison results"""
        
        model_names = list(comparison_results.keys())
        
        # Extract data for batch size 8 (middle ground)
        inference_times = []
        throughputs = []
        memory_usage = []
        param_counts = []
        
        for name in model_names:
            result = comparison_results[name]
            
            # Inference (batch size 8)
            if 'inference' in result and 8 in result['inference']:
                inference_times.append(result['inference'][8]['mean_time_ms'])
                throughputs.append(result['inference'][8]['throughput_samples_per_sec'])
            else:
                inference_times.append(0)
                throughputs.append(0)
            
            # Memory (batch size 8)
            if 'memory' in result and 8 in result['memory']:
                memory_usage.append(result['memory'][8]['peak_memory_mb'])
            else:
                memory_usage.append(0)
            
            # Parameters
            param_counts.append(result['parameter_count'] / 1e6)  # Convert to millions
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Inference time comparison
        bars1 = axes[0, 0].bar(model_names, inference_times, alpha=0.7)
        axes[0, 0].set_ylabel('Inference Time (ms)')
        axes[0, 0].set_title('Inference Time Comparison (Batch Size 8)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, time_val in zip(bars1, inference_times):
            if time_val > 0:
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                               f'{time_val:.2f}', ha='center', va='bottom')
        
        # Throughput comparison
        bars2 = axes[0, 1].bar(model_names, throughputs, alpha=0.7, color='orange')
        axes[0, 1].set_ylabel('Throughput (samples/sec)')
        axes[0, 1].set_title('Throughput Comparison (Batch Size 8)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, throughput in zip(bars2, throughputs):
            if throughput > 0:
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                               f'{throughput:.0f}', ha='center', va='bottom')
        
        # Memory usage comparison
        bars3 = axes[1, 0].bar(model_names, memory_usage, alpha=0.7, color='green')
        axes[1, 0].set_ylabel('Peak Memory (MB)')
        axes[1, 0].set_title('Memory Usage Comparison (Batch Size 8)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, memory in zip(bars3, memory_usage):
            if memory > 0:
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                               f'{memory:.0f}', ha='center', va='bottom')
        
        # Parameter count comparison
        bars4 = axes[1, 1].bar(model_names, param_counts, alpha=0.7, color='red')
        axes[1, 1].set_ylabel('Parameters (Millions)')
        axes[1, 1].set_title('Model Size Comparison')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, params in zip(bars4, param_counts):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{params:.2f}M', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison_benchmark.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def generate_benchmark_report(self, comparison_results: Dict[str, Any]) -> str:
        """Generate comprehensive benchmark report"""
        
        report = "=" * 80 + "\n"
        report += "PYTORCH MODEL PERFORMANCE BENCHMARK REPORT\n"
        report += "=" * 80 + "\n\n"
        
        # System information
        report += "SYSTEM INFORMATION:\n"
        report += "-" * 20 + "\n"
        for key, value in self.system_info.items():
            report += f"{key.replace('_', ' ').title()}: {value}\n"
        report += "\n"
        
        # Model comparison summary
        report += "MODEL COMPARISON SUMMARY:\n"
        report += "-" * 28 + "\n"
        report += f"{'Model':<20} {'Params (M)':<12} {'Inf Time (ms)':<15} {'Throughput':<12} {'Memory (MB)':<12}\n"
        report += "-" * 80 + "\n"
        
        for model_name, results in comparison_results.items():
            params = results['parameter_count'] / 1e6
            
            # Get batch size 8 results or fallback
            inf_time = "N/A"
            throughput = "N/A"
            memory = "N/A"
            
            if 'inference' in results and 8 in results['inference']:
                inf_time = f"{results['inference'][8]['mean_time_ms']:.2f}"
                throughput = f"{results['inference'][8]['throughput_samples_per_sec']:.0f}"
            
            if 'memory' in results and 8 in results['memory']:
                memory = f"{results['memory'][8]['peak_memory_mb']:.0f}"
            
            report += f"{model_name:<20} {params:<12.2f} {inf_time:<15} {throughput:<12} {memory:<12}\n"
        
        report += "\n"
        
        # Detailed results for each model
        for model_name, results in comparison_results.items():
            report += f"DETAILED RESULTS: {model_name.upper()}\n"
            report += "-" * 40 + "\n"
            
            report += f"Parameters: {results['parameter_count']:,}\n"
            report += f"Trainable Parameters: {results['trainable_parameters']:,}\n\n"
            
            # Inference results
            if 'inference' in results:
                report += "Inference Performance:\n"
                for batch_size, inf_data in results['inference'].items():
                    report += f"  Batch Size {batch_size}:\n"
                    report += f"    Mean Time: {inf_data['mean_time_ms']:.2f} ± {inf_data['std_time_ms']:.2f} ms\n"
                    report += f"    Throughput: {inf_data['throughput_samples_per_sec']:.0f} samples/sec\n"
                    report += f"    P95 Latency: {inf_data['p95_time_ms']:.2f} ms\n"
                report += "\n"
            
            # Memory results
            if 'memory' in results:
                report += "Memory Usage:\n"
                for batch_size, mem_data in results['memory'].items():
                    report += f"  Batch Size {batch_size}:\n"
                    report += f"    Peak Memory: {mem_data['peak_memory_mb']:.1f} MB\n"
                    report += f"    Memory per Sample: {mem_data['memory_per_sample_mb']:.2f} MB\n"
                report += "\n"
            
            # Training results
            if 'training' in results:
                training_data = results['training']
                report += "Training Performance:\n"
                report += f"  Mean Step Time: {training_data['mean_step_time_ms']:.2f} ± {training_data['std_step_time_ms']:.2f} ms\n"
                report += f"  Training Throughput: {training_data['samples_per_sec']:.0f} samples/sec\n"
                report += f"  Steps per Second: {training_data['steps_per_sec']:.1f}\n"
                report += "\n"
        
        report += "=" * 80 + "\n"
        
        return report

# Specialized Benchmarks
class LatencyBenchmark:
    """Specialized latency benchmarking"""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def measure_layer_latency(self, model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, float]:
        """Measure latency of individual layers"""
        
        model = model.to(self.device)
        input_tensor = input_tensor.to(self.device)
        model.eval()
        
        layer_times = {}
        hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                layer_times[name] = (end_time - start_time) * 1000
            return hook
        
        # Register hooks
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                handle = module.register_forward_hook(make_hook(name))
                hooks.append(handle)
        
        # Forward pass with timing
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            _ = model(input_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return layer_times
    
    def percentile_latency_analysis(self, model: nn.Module, input_shape: Tuple[int, ...],
                                  batch_size: int = 1, num_runs: int = 1000) -> Dict[str, float]:
        """Detailed percentile latency analysis"""
        
        model = model.to(self.device)
        model.eval()
        
        input_tensor = torch.randn(batch_size, *input_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # Collect timing data
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                _ = model(input_tensor)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)
        
        times = np.array(times)
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'p50': np.percentile(times, 50),
            'p90': np.percentile(times, 90),
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99),
            'p99.9': np.percentile(times, 99.9)
        }

if __name__ == "__main__":
    print("Performance Benchmarking")
    print("=" * 28)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create benchmark suite
    benchmark = PerformanceBenchmark(device)
    
    print("\nSystem Information:")
    print("-" * 20)
    for key, value in benchmark.system_info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Create test models
    models = {
        'Standard CNN': BenchmarkCNN(num_classes=10, width_multiplier=1.0),
        'Wide CNN': BenchmarkCNN(num_classes=10, width_multiplier=2.0),
        'Efficient CNN': EfficientCNN(num_classes=10)
    }
    
    input_shape = (3, 32, 32)
    
    print("\n1. Model Comparison Benchmark")
    print("-" * 35)
    
    # Compare all models
    comparison_results = benchmark.compare_models(models, input_shape)
    
    # Plot results
    benchmark.plot_model_comparison(comparison_results)
    
    # Generate report
    report = benchmark.generate_benchmark_report(comparison_results)
    
    # Save report
    with open('benchmark_report.txt', 'w') as f:
        f.write(report)
    
    print("Benchmark report saved to 'benchmark_report.txt'")
    
    print("\n2. Detailed Inference Benchmark")
    print("-" * 40)
    
    # Detailed inference benchmark for one model
    test_model = models['Standard CNN']
    
    detailed_inference = benchmark.benchmark_inference(
        test_model, input_shape, 
        batch_sizes=[1, 2, 4, 8, 16, 32, 64],
        num_runs=200
    )
    
    benchmark.plot_inference_benchmark(detailed_inference, "Detailed Inference Benchmark")
    
    print("Inference Benchmark Results:")
    print("-" * 30)
    print(f"{'Batch Size':<12} {'Mean (ms)':<12} {'Std (ms)':<12} {'P95 (ms)':<12} {'Throughput':<12}")
    print("-" * 65)
    
    for batch_size, results in detailed_inference.items():
        mean_time = results['mean_time_ms']
        std_time = results['std_time_ms']
        p95_time = results['p95_time_ms']
        throughput = results['throughput_samples_per_sec']
        
        print(f"{batch_size:<12} {mean_time:<12.2f} {std_time:<12.2f} {p95_time:<12.2f} {throughput:<12.0f}")
    
    print("\n3. Latency Analysis")
    print("-" * 23)
    
    # Specialized latency analysis
    latency_benchmark = LatencyBenchmark(device)
    
    # Layer-wise latency
    sample_input = torch.randn(1, *input_shape)
    layer_latencies = latency_benchmark.measure_layer_latency(test_model, sample_input)
    
    print("Top 10 Slowest Layers:")
    print("-" * 25)
    sorted_layers = sorted(layer_latencies.items(), key=lambda x: x[1], reverse=True)
    
    for i, (layer_name, latency) in enumerate(sorted_layers[:10]):
        print(f"{i+1:2d}. {layer_name[:30]:<30} {latency:.3f} ms")
    
    # Percentile analysis
    percentile_results = latency_benchmark.percentile_latency_analysis(test_model, input_shape)
    
    print("\nPercentile Latency Analysis:")
    print("-" * 30)
    for percentile, latency in percentile_results.items():
        print(f"{percentile}: {latency:.3f} ms")
    
    print("\n4. Memory Benchmark")
    print("-" * 22)
    
    # Memory benchmark
    memory_results = benchmark.benchmark_memory(test_model, input_shape)
    
    print("Memory Usage by Batch Size:")
    print("-" * 30)
    print(f"{'Batch Size':<12} {'Peak (MB)':<12} {'Per Sample (MB)':<15}")
    print("-" * 40)
    
    for batch_size, mem_data in memory_results.items():
        peak_memory = mem_data['peak_memory_mb']
        per_sample = mem_data['memory_per_sample_mb']
        
        print(f"{batch_size:<12} {peak_memory:<12.1f} {per_sample:<15.3f}")
    
    print("\n5. Training Benchmark")
    print("-" * 24)
    
    # Training benchmark
    training_results = benchmark.benchmark_training(test_model, input_shape)
    
    print("Training Performance:")
    print("-" * 20)
    print(f"Mean Step Time: {training_results['mean_step_time_ms']:.2f} ± {training_results['std_step_time_ms']:.2f} ms")
    print(f"Training Throughput: {training_results['samples_per_sec']:.0f} samples/sec")
    print(f"Steps per Second: {training_results['steps_per_sec']:.1f}")
    
    print("\n6. Optimization Recommendations")
    print("-" * 40)
    
    recommendations = []
    
    # Analyze results for recommendations
    std_cnn_results = comparison_results['Standard CNN']
    
    # Check inference efficiency
    if 'inference' in std_cnn_results and 1 in std_cnn_results['inference']:
        single_inference_time = std_cnn_results['inference'][1]['mean_time_ms']
        if single_inference_time > 10:
            recommendations.append("Consider model optimization techniques (pruning, quantization)")
    
    # Check memory efficiency
    if 'memory' in std_cnn_results and 8 in std_cnn_results['memory']:
        memory_per_sample = std_cnn_results['memory'][8]['memory_per_sample_mb']
        if memory_per_sample > 10:
            recommendations.append("High memory usage per sample - consider reducing model size")
    
    # Check parameter efficiency
    param_count = std_cnn_results['parameter_count']
    if param_count > 5e6:
        recommendations.append("Large parameter count - consider using efficient architectures")
    
    # Compare with efficient model
    if 'Efficient CNN' in comparison_results:
        efficient_results = comparison_results['Efficient CNN']
        if ('inference' in both_results := (std_cnn_results, efficient_results) and 
            1 in std_cnn_results['inference'] and 1 in efficient_results['inference']):
            
            std_time = std_cnn_results['inference'][1]['mean_time_ms']
            eff_time = efficient_results['inference'][1]['mean_time_ms']
            
            if eff_time < std_time * 0.8:
                speedup = std_time / eff_time
                recommendations.append(f"Switch to efficient architecture for {speedup:.1f}x speedup")
    
    if recommendations:
        print("Performance Optimization Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("Model performance looks good!")
    
    print("\nPerformance benchmarking completed!")
    print("Generated files:")
    print("  - benchmark_report.txt (detailed benchmark report)")
    print("  - model_comparison_benchmark.png (comparison plots)")
    print("  - detailed_inference_benchmark.png (inference analysis)")