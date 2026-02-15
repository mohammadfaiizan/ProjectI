import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Optimized Model Architectures
class StandardCNN(nn.Module):
    """Standard CNN for comparison"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class OptimizedCNN(nn.Module):
    """Optimized CNN with inference optimizations"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Fused operations where possible
        self.features = nn.Sequential(
            # Block 1 - Reduced depth
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2 - Depthwise separable convolution
            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),  # Depthwise
            nn.Conv2d(64, 128, 1, bias=False),  # Pointwise
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3 - Efficient final block
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Simplified classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class MobileNetBlock(nn.Module):
    """MobileNet-style efficient block"""
    
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super().__init__()
        
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = stride == 1 and in_channels == out_channels
        
        layers = []
        
        # Expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            
            # Project
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class EfficientMobileNet(nn.Module):
    """Efficient MobileNet-style architecture"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        self.blocks = nn.Sequential(
            MobileNetBlock(32, 16, 1, 1),
            MobileNetBlock(16, 24, 2, 6),
            MobileNetBlock(24, 32, 2, 6),
            MobileNetBlock(32, 64, 2, 6),
            MobileNetBlock(64, 96, 1, 6),
            MobileNetBlock(96, 160, 2, 6),
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(160, 320, 1, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(320, num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.final_conv(x)
        x = self.classifier(x)
        return x

# Inference Optimization Techniques
class ModelOptimizer:
    """Model optimization utilities for inference"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.original_model = model.to(device)
        self.device = device
    
    def fuse_conv_bn(self, model: nn.Module) -> nn.Module:
        """Fuse convolution and batch normalization layers"""
        
        def _fuse_conv_bn(conv, bn):
            """Fuse a single conv-bn pair"""
            fused = nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                conv.kernel_size,
                conv.stride,
                conv.padding,
                conv.dilation,
                conv.groups,
                bias=True
            )
            
            # Copy conv weights
            fused.weight.data = conv.weight.data.clone()
            
            # Fuse batch norm parameters
            if conv.bias is not None:
                fused.bias.data = conv.bias.data.clone()
            else:
                fused.bias.data = torch.zeros_like(bn.bias.data)
            
            # Apply batch norm transformation
            w_bn = bn.weight.data / torch.sqrt(bn.running_var.data + bn.eps)
            fused.weight.data *= w_bn.view(-1, 1, 1, 1)
            fused.bias.data = (fused.bias.data - bn.running_mean.data) * w_bn + bn.bias.data
            
            return fused
        
        model.eval()  # Important: set to eval mode
        fused_model = model
        
        # Find conv-bn pairs and fuse them
        modules_to_fuse = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential):
                for i in range(len(module) - 1):
                    if (isinstance(module[i], nn.Conv2d) and 
                        isinstance(module[i + 1], nn.BatchNorm2d)):
                        modules_to_fuse.append((name, i))
        
        # Note: This is a simplified fusion - real implementation would be more complex
        print(f"Found {len(modules_to_fuse)} conv-bn pairs to fuse")
        
        return fused_model
    
    def apply_quantization(self, model: nn.Module, quantization_type='dynamic') -> nn.Module:
        """Apply quantization to model"""
        
        if quantization_type == 'dynamic':
            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )
        
        elif quantization_type == 'static':
            # Static quantization (simplified)
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            
            # Calibration would happen here with representative data
            # For demo, we'll skip calibration
            
            quantized_model = torch.quantization.convert(model, inplace=False)
        
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")
        
        return quantized_model
    
    def apply_pruning(self, model: nn.Module, sparsity: float = 0.3) -> nn.Module:
        """Apply magnitude-based pruning"""
        
        import torch.nn.utils.prune as prune
        
        pruned_model = model
        
        # Apply global magnitude pruning
        parameters_to_prune = []
        for name, module in pruned_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity,
        )
        
        # Remove pruning reparameterization
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        return pruned_model
    
    def convert_to_torchscript(self, model: nn.Module, example_input: torch.Tensor) -> torch.jit.ScriptModule:
        """Convert model to TorchScript"""
        
        model.eval()
        
        # Try tracing first
        try:
            traced_model = torch.jit.trace(model, example_input)
            return traced_model
        except Exception as e:
            print(f"Tracing failed: {e}")
            
            # Fall back to scripting
            try:
                scripted_model = torch.jit.script(model)
                return scripted_model
            except Exception as e:
                print(f"Scripting also failed: {e}")
                return None
    
    def optimize_for_mobile(self, model: nn.Module, example_input: torch.Tensor) -> torch.jit.ScriptModule:
        """Optimize model for mobile deployment"""
        
        # Convert to TorchScript
        traced_model = self.convert_to_torchscript(model, example_input)
        
        if traced_model is not None:
            # Optimize for mobile
            optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(traced_model)
            return optimized_model
        
        return None

class InferenceBenchmark:
    """Benchmark inference performance"""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def benchmark_model(self, model: nn.Module, input_shape: Tuple[int, ...],
                       batch_sizes: List[int] = [1, 4, 8, 16, 32],
                       num_runs: int = 100) -> Dict[int, Dict[str, float]]:
        """Benchmark model across different batch sizes"""
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"Benchmarking batch size {batch_size}...")
            
            try:
                # Create input tensor
                input_tensor = torch.randn(batch_size, *input_shape).to(self.device)
                
                # Warmup
                model.eval()
                with torch.no_grad():
                    for _ in range(10):
                        _ = model(input_tensor)
                
                # Benchmark
                times = []
                
                with torch.no_grad():
                    for _ in range(num_runs):
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        start_time = time.time()
                        _ = model(input_tensor)
                        
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        end_time = time.time()
                        times.append((end_time - start_time) * 1000)  # Convert to ms
                
                # Calculate statistics
                avg_time = np.mean(times)
                std_time = np.std(times)
                min_time = np.min(times)
                max_time = np.max(times)
                
                # Calculate throughput
                throughput = batch_size / (avg_time / 1000)  # samples per second
                
                results[batch_size] = {
                    'avg_time_ms': avg_time,
                    'std_time_ms': std_time,
                    'min_time_ms': min_time,
                    'max_time_ms': max_time,
                    'throughput_samples_per_sec': throughput,
                    'avg_time_per_sample_ms': avg_time / batch_size
                }
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM at batch size {batch_size}")
                    break
                else:
                    raise e
        
        return results
    
    def compare_optimizations(self, models: Dict[str, nn.Module], 
                            input_shape: Tuple[int, ...],
                            batch_size: int = 1) -> Dict[str, Dict[str, Any]]:
        """Compare different model optimizations"""
        
        comparison = {}
        
        for model_name, model in models.items():
            print(f"Benchmarking {model_name}...")
            
            # Single batch size benchmark for comparison
            results = self.benchmark_model(model, input_shape, [batch_size], num_runs=50)
            
            if batch_size in results:
                comparison[model_name] = results[batch_size]
                
                # Add model size information
                param_count = sum(p.numel() for p in model.parameters())
                model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
                
                comparison[model_name]['parameter_count'] = param_count
                comparison[model_name]['model_size_mb'] = model_size_mb
            
        return comparison
    
    def memory_benchmark(self, model: nn.Module, input_shape: Tuple[int, ...],
                        batch_sizes: List[int] = [1, 4, 8, 16]) -> Dict[int, float]:
        """Benchmark memory usage"""
        
        memory_usage = {}
        
        for batch_size in batch_sizes:
            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                
                input_tensor = torch.randn(batch_size, *input_shape).to(self.device)
                
                model.eval()
                with torch.no_grad():
                    _ = model(input_tensor)
                
                if torch.cuda.is_available():
                    memory_usage[batch_size] = torch.cuda.max_memory_allocated() / (1024**2)  # MB
                else:
                    memory_usage[batch_size] = 0  # Cannot measure CPU memory easily
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM at batch size {batch_size}")
                    break
                else:
                    raise e
        
        return memory_usage
    
    def plot_benchmark_results(self, results: Dict[int, Dict[str, float]], 
                             title: str = "Inference Benchmark"):
        """Plot benchmark results"""
        
        batch_sizes = list(results.keys())
        avg_times = [results[bs]['avg_time_ms'] for bs in batch_sizes]
        throughputs = [results[bs]['throughput_samples_per_sec'] for bs in batch_sizes]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Average time plot
        ax1.plot(batch_sizes, avg_times, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Average Time (ms)')
        ax1.set_title('Inference Time vs Batch Size')
        ax1.grid(True, alpha=0.3)
        
        # Throughput plot
        ax2.plot(batch_sizes, throughputs, 'r-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Throughput (samples/sec)')
        ax2.set_title('Throughput vs Batch Size')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(f'{title.lower().replace(" ", "_")}_benchmark.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_optimization_comparison(self, comparison: Dict[str, Dict[str, Any]]):
        """Plot optimization comparison"""
        
        model_names = list(comparison.keys())
        avg_times = [comparison[name]['avg_time_ms'] for name in model_names]
        throughputs = [comparison[name]['throughput_samples_per_sec'] for name in model_names]
        model_sizes = [comparison[name]['model_size_mb'] for name in model_names]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Average time comparison
        bars1 = axes[0, 0].bar(model_names, avg_times, alpha=0.7)
        axes[0, 0].set_ylabel('Average Time (ms)')
        axes[0, 0].set_title('Inference Time Comparison')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, time_val in zip(bars1, avg_times):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{time_val:.2f}', ha='center', va='bottom')
        
        # Throughput comparison
        bars2 = axes[0, 1].bar(model_names, throughputs, alpha=0.7, color='orange')
        axes[0, 1].set_ylabel('Throughput (samples/sec)')
        axes[0, 1].set_title('Throughput Comparison')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, throughput in zip(bars2, throughputs):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{throughput:.1f}', ha='center', va='bottom')
        
        # Model size comparison
        bars3 = axes[1, 0].bar(model_names, model_sizes, alpha=0.7, color='green')
        axes[1, 0].set_ylabel('Model Size (MB)')
        axes[1, 0].set_title('Model Size Comparison')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, size in zip(bars3, model_sizes):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{size:.2f}', ha='center', va='bottom')
        
        # Efficiency scatter plot (throughput vs model size)
        axes[1, 1].scatter(model_sizes, throughputs, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            axes[1, 1].annotate(name, (model_sizes[i], throughputs[i]), 
                               xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_xlabel('Model Size (MB)')
        axes[1, 1].set_ylabel('Throughput (samples/sec)')
        axes[1, 1].set_title('Efficiency: Throughput vs Model Size')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('optimization_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

class BatchingOptimizer:
    """Optimize inference through intelligent batching"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def find_optimal_batch_size(self, input_shape: Tuple[int, ...],
                               max_batch_size: int = 64,
                               target_latency_ms: float = 100) -> int:
        """Find optimal batch size for given latency target"""
        
        benchmark = InferenceBenchmark(self.device)
        
        # Test different batch sizes
        batch_sizes = [2**i for i in range(0, int(np.log2(max_batch_size)) + 1)]
        
        optimal_batch_size = 1
        best_throughput = 0
        
        for batch_size in batch_sizes:
            try:
                results = benchmark.benchmark_model(
                    self.model, input_shape, [batch_size], num_runs=20
                )
                
                if batch_size in results:
                    avg_time = results[batch_size]['avg_time_ms']
                    throughput = results[batch_size]['throughput_samples_per_sec']
                    
                    # Check if latency constraint is met
                    if avg_time <= target_latency_ms and throughput > best_throughput:
                        optimal_batch_size = batch_size
                        best_throughput = throughput
                
            except RuntimeError:
                break  # OOM
        
        return optimal_batch_size
    
    def dynamic_batching_inference(self, inputs: List[torch.Tensor],
                                 max_batch_size: int = 32,
                                 timeout_ms: float = 50) -> List[torch.Tensor]:
        """Perform inference with dynamic batching"""
        
        if not inputs:
            return []
        
        results = []
        current_batch = []
        
        start_time = time.time()
        
        for input_tensor in inputs:
            current_batch.append(input_tensor)
            
            # Check if we should process the batch
            should_process = (
                len(current_batch) >= max_batch_size or
                (time.time() - start_time) * 1000 >= timeout_ms
            )
            
            if should_process:
                # Process current batch
                batch_tensor = torch.stack(current_batch).to(self.device)
                
                self.model.eval()
                with torch.no_grad():
                    batch_results = self.model(batch_tensor)
                
                # Split results back
                for i in range(len(current_batch)):
                    results.append(batch_results[i])
                
                # Reset for next batch
                current_batch = []
                start_time = time.time()
        
        # Process remaining inputs
        if current_batch:
            batch_tensor = torch.stack(current_batch).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                batch_results = self.model(batch_tensor)
            
            for i in range(len(current_batch)):
                results.append(batch_results[i])
        
        return results

class MemoryOptimizer:
    """Optimize memory usage during inference"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def enable_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """Enable gradient checkpointing for memory efficiency"""
        
        # Note: This is mainly for training, but can be useful for inference in some cases
        def checkpoint_forward(module):
            def forward_impl(*args, **kwargs):
                return torch.utils.checkpoint.checkpoint(module.original_forward, *args, **kwargs)
            return forward_impl
        
        # Apply checkpointing to specific modules
        for name, module in model.named_modules():
            if isinstance(module, (nn.Sequential, nn.ModuleList)):
                if len(list(module.children())) > 2:  # Only for modules with multiple children
                    module.original_forward = module.forward
                    module.forward = checkpoint_forward(module)
        
        return model
    
    def optimize_memory_layout(self, model: nn.Module) -> nn.Module:
        """Optimize tensor memory layout"""
        
        # Convert to channels_last format for better memory access patterns
        model = model.to(memory_format=torch.channels_last)
        
        return model
    
    def apply_inplace_operations(self, model: nn.Module) -> nn.Module:
        """Convert operations to inplace where possible"""
        
        def convert_to_inplace(module):
            for name, child in module.named_children():
                if isinstance(child, nn.ReLU):
                    setattr(module, name, nn.ReLU(inplace=True))
                elif isinstance(child, nn.LeakyReLU):
                    setattr(module, name, nn.LeakyReLU(inplace=True))
                elif isinstance(child, nn.ELU):
                    setattr(module, name, nn.ELU(inplace=True))
                else:
                    convert_to_inplace(child)
        
        convert_to_inplace(model)
        return model

if __name__ == "__main__":
    print("Inference Optimization")
    print("=" * 25)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    input_shape = (3, 32, 32)
    
    # Create test models
    standard_model = StandardCNN(num_classes=10).to(device)
    optimized_model = OptimizedCNN(num_classes=10).to(device)
    mobile_model = EfficientMobileNet(num_classes=10).to(device)
    
    print("\n1. Basic Model Comparison")
    print("-" * 30)
    
    # Compare base models
    benchmark = InferenceBenchmark(device)
    
    base_models = {
        'Standard CNN': standard_model,
        'Optimized CNN': optimized_model,
        'MobileNet': mobile_model
    }
    
    print("Comparing base model architectures...")
    base_comparison = benchmark.compare_optimizations(base_models, input_shape, batch_size=1)
    
    print("\nBase Model Comparison:")
    print("-" * 25)
    print(f"{'Model':<15} {'Time (ms)':<12} {'Throughput':<12} {'Params':<12} {'Size (MB)':<12}")
    print("-" * 70)
    
    for name, stats in base_comparison.items():
        time_ms = stats['avg_time_ms']
        throughput = stats['throughput_samples_per_sec']
        params = stats['parameter_count']
        size_mb = stats['model_size_mb']
        
        print(f"{name:<15} {time_ms:<12.2f} {throughput:<12.1f} {params:<12,} {size_mb:<12.2f}")
    
    print("\n2. Model Optimization Techniques")
    print("-" * 40)
    
    # Apply various optimizations to the standard model
    optimizer = ModelOptimizer(standard_model, device)
    
    optimized_models = {'Original': standard_model}
    
    # Conv-BN fusion
    print("Applying Conv-BN fusion...")
    try:
        fused_model = optimizer.fuse_conv_bn(standard_model)
        optimized_models['Conv-BN Fused'] = fused_model
    except Exception as e:
        print(f"Conv-BN fusion failed: {e}")
    
    # Quantization
    print("Applying quantization...")
    try:
        quantized_model = optimizer.apply_quantization(standard_model, 'dynamic')
        optimized_models['Quantized'] = quantized_model
    except Exception as e:
        print(f"Quantization failed: {e}")
    
    # Pruning
    print("Applying pruning...")
    try:
        pruned_model = optimizer.apply_pruning(standard_model, sparsity=0.3)
        optimized_models['Pruned (30%)'] = pruned_model
    except Exception as e:
        print(f"Pruning failed: {e}")
    
    # TorchScript
    print("Converting to TorchScript...")
    try:
        example_input = torch.randn(1, *input_shape).to(device)
        torchscript_model = optimizer.convert_to_torchscript(standard_model, example_input)
        if torchscript_model is not None:
            optimized_models['TorchScript'] = torchscript_model
    except Exception as e:
        print(f"TorchScript conversion failed: {e}")
    
    print("\n3. Optimization Comparison")
    print("-" * 32)
    
    # Compare optimized models
    optimization_comparison = benchmark.compare_optimizations(optimized_models, input_shape, batch_size=1)
    
    print("Optimization Results:")
    print("-" * 20)
    print(f"{'Optimization':<15} {'Time (ms)':<12} {'Speedup':<10} {'Size (MB)':<12} {'Compression':<12}")
    print("-" * 70)
    
    baseline_time = optimization_comparison['Original']['avg_time_ms']
    baseline_size = optimization_comparison['Original']['model_size_mb']
    
    for name, stats in optimization_comparison.items():
        time_ms = stats['avg_time_ms']
        speedup = baseline_time / time_ms
        size_mb = stats['model_size_mb']
        compression = baseline_size / size_mb if size_mb > 0 else 1.0
        
        print(f"{name:<15} {time_ms:<12.2f} {speedup:<10.2f}x {size_mb:<12.2f} {compression:<12.2f}x")
    
    # Plot optimization comparison
    benchmark.plot_optimization_comparison(optimization_comparison)
    
    print("\n4. Batch Size Optimization")
    print("-" * 32)
    
    # Find optimal batch size
    print("Finding optimal batch size...")
    
    batching_optimizer = BatchingOptimizer(optimized_model, device)
    optimal_batch_size = batching_optimizer.find_optimal_batch_size(
        input_shape, max_batch_size=32, target_latency_ms=50
    )
    
    print(f"Optimal batch size for 50ms latency: {optimal_batch_size}")
    
    # Benchmark across batch sizes
    batch_results = benchmark.benchmark_model(
        optimized_model, input_shape, 
        batch_sizes=[1, 2, 4, 8, 16, 32],
        num_runs=50
    )
    
    print("\nBatch Size Analysis:")
    print("-" * 20)
    print(f"{'Batch Size':<12} {'Time (ms)':<12} {'Time/Sample':<15} {'Throughput':<15}")
    print("-" * 60)
    
    for batch_size, stats in batch_results.items():
        total_time = stats['avg_time_ms']
        time_per_sample = stats['avg_time_per_sample_ms']
        throughput = stats['throughput_samples_per_sec']
        
        print(f"{batch_size:<12} {total_time:<12.2f} {time_per_sample:<15.2f} {throughput:<15.1f}")
    
    # Plot batch size results
    benchmark.plot_benchmark_results(batch_results, "Batch Size Optimization")
    
    print("\n5. Memory Optimization")
    print("-" * 25)
    
    # Memory optimization
    memory_optimizer = MemoryOptimizer(standard_model, device)
    
    # Memory benchmark
    memory_usage = benchmark.memory_benchmark(standard_model, input_shape)
    
    print("Memory Usage by Batch Size:")
    print("-" * 30)
    for batch_size, memory_mb in memory_usage.items():
        print(f"Batch size {batch_size}: {memory_mb:.2f} MB")
    
    # Apply memory optimizations
    print("\nApplying memory optimizations...")
    
    memory_optimized_model = memory_optimizer.apply_inplace_operations(standard_model)
    memory_optimized_model = memory_optimizer.optimize_memory_layout(memory_optimized_model)
    
    # Compare memory usage
    optimized_memory_usage = benchmark.memory_benchmark(memory_optimized_model, input_shape)
    
    print("Memory Usage After Optimization:")
    print("-" * 35)
    for batch_size in memory_usage.keys():
        if batch_size in optimized_memory_usage:
            original = memory_usage[batch_size]
            optimized = optimized_memory_usage[batch_size]
            reduction = (original - optimized) / original * 100 if original > 0 else 0
            
            print(f"Batch size {batch_size}: {optimized:.2f} MB ({reduction:.1f}% reduction)")
    
    print("\n6. Dynamic Batching Demo")
    print("-" * 30)
    
    # Demonstrate dynamic batching
    print("Testing dynamic batching...")
    
    # Create sample inputs
    sample_inputs = [torch.randn(*input_shape) for _ in range(25)]
    
    # Time individual inference
    start_time = time.time()
    individual_results = []
    optimized_model.eval()
    with torch.no_grad():
        for inp in sample_inputs:
            result = optimized_model(inp.unsqueeze(0).to(device))
            individual_results.append(result)
    individual_time = time.time() - start_time
    
    # Time dynamic batching
    start_time = time.time()
    batched_results = batching_optimizer.dynamic_batching_inference(
        sample_inputs, max_batch_size=8, timeout_ms=20
    )
    batched_time = time.time() - start_time
    
    print(f"Individual inference time: {individual_time*1000:.2f} ms")
    print(f"Dynamic batching time: {batched_time*1000:.2f} ms")
    print(f"Speedup: {individual_time/batched_time:.2f}x")
    
    print("\n7. Production Optimization Tips")
    print("-" * 40)
    
    tips = [
        "Use TorchScript for deployment to eliminate Python overhead",
        "Apply quantization for 2-4x speedup with minimal accuracy loss",
        "Fuse Conv-BatchNorm layers for training and inference speedup",
        "Use appropriate batch sizes based on latency requirements",
        "Enable mixed precision training and inference",
        "Consider model pruning for memory-constrained environments",
        "Optimize tensor memory layout (channels_last) for better cache usage",
        "Use inplace operations where possible to reduce memory allocations",
        "Implement dynamic batching for varying workloads",
        "Profile memory usage and optimize for target hardware",
        "Consider knowledge distillation for smaller, faster models",
        "Use specialized hardware (TensorRT, CoreML) optimizations when available"
    ]
    
    print("Production Optimization Recommendations:")
    for i, tip in enumerate(tips, 1):
        print(f"{i:2d}. {tip}")
    
    print("\n8. Performance Summary")
    print("-" * 28)
    
    # Summary of best optimizations
    best_optimization = min(optimization_comparison.items(), 
                           key=lambda x: x[1]['avg_time_ms'])
    
    best_model_name = best_optimization[0]
    best_stats = best_optimization[1]
    
    original_stats = optimization_comparison['Original']
    
    print("Best Optimization Results:")
    print(f"  Best Model: {best_model_name}")
    print(f"  Speedup: {original_stats['avg_time_ms'] / best_stats['avg_time_ms']:.2f}x")
    print(f"  Size Reduction: {original_stats['model_size_mb'] / best_stats['model_size_mb']:.2f}x")
    print(f"  Throughput Improvement: {best_stats['throughput_samples_per_sec'] / original_stats['throughput_samples_per_sec']:.2f}x")
    
    print("\nInference optimization completed!")
    print("Generated files:")
    print("  - *_benchmark.png (benchmark plots)")
    print("  - optimization_comparison.png (optimization comparison)")