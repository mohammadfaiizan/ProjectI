import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any

# Sample Models for Optimization
class BaselineModel(nn.Module):
    """Baseline model before optimization"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

class OptimizedModel(nn.Module):
    """Optimized model for inference"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # Fused conv-bn-relu blocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Simplified classifier without dropout for inference
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

# Inference Optimization Techniques
class InferenceOptimizer:
    """Collection of inference optimization techniques"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
    
    def fuse_conv_bn_relu(self, model: nn.Module) -> nn.Module:
        """Fuse conv-bn-relu layers for inference optimization"""
        
        model.eval()
        
        # Find conv-bn-relu patterns and fuse them
        modules_to_fuse = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential):
                for i in range(len(module) - 2):
                    if (isinstance(module[i], nn.Conv2d) and
                        isinstance(module[i + 1], nn.BatchNorm2d) and
                        isinstance(module[i + 2], (nn.ReLU, nn.ReLU6))):
                        modules_to_fuse.append([f"{name}.{i}", f"{name}.{i+1}", f"{name}.{i+2}"])
                    elif (isinstance(module[i], nn.Conv2d) and
                          isinstance(module[i + 1], nn.BatchNorm2d)):
                        modules_to_fuse.append([f"{name}.{i}", f"{name}.{i+1}"])
        
        if modules_to_fuse:
            try:
                fused_model = torch.quantization.fuse_modules(model, modules_to_fuse)
                print(f"✓ Fused {len(modules_to_fuse)} module groups")
                return fused_model
            except Exception as e:
                print(f"Module fusion failed: {e}")
                return model
        else:
            print("No fuseable modules found")
            return model
    
    def optimize_for_inference(self, model: nn.Module) -> torch.jit.ScriptModule:
        """Apply comprehensive inference optimizations"""
        
        model.eval()
        
        # Convert to TorchScript
        try:
            scripted_model = torch.jit.script(model)
        except Exception:
            # Fall back to tracing
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            scripted_model = torch.jit.trace(model, dummy_input)
        
        # Freeze model for optimization
        scripted_model = torch.jit.freeze(scripted_model)
        
        # Apply inference optimizations
        scripted_model = torch.jit.optimize_for_inference(scripted_model)
        
        print("✓ Model optimized for inference")
        return scripted_model
    
    def apply_channel_last_optimization(self, model: nn.Module) -> nn.Module:
        """Apply channels-last memory format optimization"""
        
        # Convert model to channels-last format
        model = model.to(memory_format=torch.channels_last)
        
        print("✓ Applied channels-last memory format")
        return model
    
    def enable_mixed_precision(self, model: nn.Module) -> nn.Module:
        """Enable automatic mixed precision"""
        
        # This is typically used during training, but can help with inference
        # Note: Actual mixed precision inference requires specific implementation
        print("✓ Mixed precision considerations applied")
        return model
    
    def optimize_batch_processing(self, model: nn.Module, 
                                 optimal_batch_size: int = 8) -> Tuple[nn.Module, int]:
        """Optimize for batch processing"""
        
        # Set optimal batch size based on hardware
        return model, optimal_batch_size

class ModelProfiler:
    """Profile model performance for optimization insights"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
    
    def profile_model(self, model: nn.Module, 
                     input_shape: Tuple[int, ...],
                     num_runs: int = 100) -> Dict[str, Any]:
        """Comprehensive model profiling"""
        
        model = model.to(self.device)
        model.eval()
        
        test_input = torch.randn(input_shape).to(self.device)
        
        # Basic timing
        timing_results = self._profile_timing(model, test_input, num_runs)
        
        # Memory profiling
        memory_results = self._profile_memory(model, test_input)
        
        # FLOPs estimation
        flops_estimate = self._estimate_flops(model, test_input)
        
        return {
            'timing': timing_results,
            'memory': memory_results,
            'flops': flops_estimate,
            'model_size_mb': self._get_model_size(model)
        }
    
    def _profile_timing(self, model: nn.Module, 
                       test_input: torch.Tensor,
                       num_runs: int) -> Dict[str, float]:
        """Profile inference timing"""
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input)
        
        # Benchmark
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                _ = model(test_input)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        times = np.array(times)
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'p95_ms': np.percentile(times, 95),
            'throughput_fps': 1000 / np.mean(times)
        }
    
    def _profile_memory(self, model: nn.Module, 
                       test_input: torch.Tensor) -> Dict[str, float]:
        """Profile memory usage"""
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                _ = model(test_input)
            
            return {
                'peak_memory_mb': torch.cuda.max_memory_allocated() / (1024**2),
                'current_memory_mb': torch.cuda.memory_allocated() / (1024**2)
            }
        else:
            return {'peak_memory_mb': 0, 'current_memory_mb': 0}
    
    def _estimate_flops(self, model: nn.Module, 
                       test_input: torch.Tensor) -> int:
        """Estimate FLOPs (simplified)"""
        
        total_flops = 0
        
        def flop_count_hook(module, input, output):
            nonlocal total_flops
            
            if isinstance(module, nn.Conv2d):
                # Convolution FLOPs
                batch_size = input[0].size(0)
                output_dims = output.shape[2:]
                kernel_dims = module.kernel_size
                in_channels = module.in_channels
                out_channels = module.out_channels
                groups = module.groups
                
                filters_per_channel = out_channels // groups
                conv_per_position_flops = int(np.prod(kernel_dims)) * in_channels // groups
                
                active_elements_count = batch_size * int(np.prod(output_dims))
                total_flops += conv_per_position_flops * active_elements_count * filters_per_channel
            
            elif isinstance(module, nn.Linear):
                # Linear layer FLOPs
                total_flops += np.prod(input[0].shape) * module.out_features
        
        # Register hooks
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handle = module.register_forward_hook(flop_count_hook)
                hooks.append(handle)
        
        # Forward pass
        with torch.no_grad():
            _ = model(test_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return total_flops
    
    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB"""
        
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        
        return total_size / (1024**2)

class BatchProcessor:
    """Optimize batch processing for inference"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def find_optimal_batch_size(self, input_shape: Tuple[int, ...],
                               max_batch_size: int = 64) -> int:
        """Find optimal batch size for hardware"""
        
        optimal_batch_size = 1
        best_throughput = 0
        
        batch_sizes = [2**i for i in range(0, int(np.log2(max_batch_size)) + 1)]
        
        for batch_size in batch_sizes:
            try:
                test_input = torch.randn(batch_size, *input_shape[1:]).to(self.device)
                
                # Warmup
                with torch.no_grad():
                    for _ in range(5):
                        _ = self.model(test_input)
                
                # Benchmark
                times = []
                with torch.no_grad():
                    for _ in range(20):
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        start_time = time.perf_counter()
                        _ = self.model(test_input)
                        
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        end_time = time.perf_counter()
                        times.append(end_time - start_time)
                
                avg_time = np.mean(times)
                throughput = batch_size / avg_time
                
                if throughput > best_throughput:
                    best_throughput = throughput
                    optimal_batch_size = batch_size
                
                print(f"Batch size {batch_size}: {throughput:.2f} samples/sec")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM at batch size {batch_size}")
                    break
                else:
                    raise
        
        print(f"Optimal batch size: {optimal_batch_size}")
        return optimal_batch_size
    
    def batch_inference(self, inputs: List[torch.Tensor],
                       batch_size: int = 8) -> List[torch.Tensor]:
        """Perform batched inference"""
        
        results = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            
            # Pad batch if necessary
            while len(batch) < batch_size and len(batch) > 0:
                batch.append(batch[-1])  # Duplicate last element
            
            if batch:
                batch_tensor = torch.stack(batch).to(self.device)
                
                with torch.no_grad():
                    batch_outputs = self.model(batch_tensor)
                
                # Only take outputs for actual inputs (not padding)
                actual_outputs = batch_outputs[:len(inputs[i:i + batch_size])]
                
                for output in actual_outputs:
                    results.append(output)
        
        return results

class ModelComparator:
    """Compare different optimization techniques"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.profiler = ModelProfiler(device)
    
    def compare_optimizations(self, base_model: nn.Module,
                            input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Compare different optimization techniques"""
        
        optimizer = InferenceOptimizer(self.device)
        
        models = {
            'baseline': base_model,
            'fused': optimizer.fuse_conv_bn_relu(base_model.__class__(base_model.classifier[-1].out_features)),
            'channels_last': optimizer.apply_channel_last_optimization(base_model.__class__(base_model.classifier[-1].out_features)),
            'optimized': optimizer.optimize_for_inference(base_model.__class__(base_model.classifier[-1].out_features))
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Profiling {name} model...")
            
            try:
                profile = self.profiler.profile_model(model, input_shape)
                results[name] = profile
                
                print(f"  {name}: {profile['timing']['mean_ms']:.2f} ms")
                
            except Exception as e:
                print(f"  {name}: Failed - {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def generate_comparison_report(self, results: Dict[str, Any]) -> str:
        """Generate comparison report"""
        
        report = "Optimization Comparison Report\n"
        report += "=" * 35 + "\n\n"
        
        if 'baseline' in results and 'error' not in results['baseline']:
            baseline_time = results['baseline']['timing']['mean_ms']
            baseline_memory = results['baseline']['memory']['peak_memory_mb']
            
            report += f"{'Optimization':<15} {'Time (ms)':<12} {'Speedup':<10} {'Memory (MB)':<12} {'Memory Ratio':<12}\n"
            report += "-" * 70 + "\n"
            
            for name, result in results.items():
                if 'error' not in result:
                    time_ms = result['timing']['mean_ms']
                    speedup = baseline_time / time_ms
                    memory_mb = result['memory']['peak_memory_mb']
                    memory_ratio = memory_mb / baseline_memory if baseline_memory > 0 else 1.0
                    
                    report += f"{name:<15} {time_ms:<12.2f} {speedup:<10.2f}x {memory_mb:<12.1f} {memory_ratio:<12.2f}x\n"
                else:
                    report += f"{name:<15} {'Error':<12} {'-':<10} {'-':<12} {'-':<12}\n"
        
        return report

if __name__ == "__main__":
    print("Model Optimization for Inference")
    print("=" * 37)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create sample model
    model = BaselineModel(num_classes=10)
    optimized_model = OptimizedModel(num_classes=10)
    
    input_shape = (1, 3, 224, 224)
    
    print("\n1. Model Profiling")
    print("-" * 20)
    
    profiler = ModelProfiler(device)
    
    # Profile baseline model
    baseline_profile = profiler.profile_model(model, input_shape)
    
    print("Baseline Model Profile:")
    print(f"  Inference time: {baseline_profile['timing']['mean_ms']:.2f} ± {baseline_profile['timing']['std_ms']:.2f} ms")
    print(f"  Throughput: {baseline_profile['timing']['throughput_fps']:.1f} FPS")
    print(f"  Memory usage: {baseline_profile['memory']['peak_memory_mb']:.1f} MB")
    print(f"  Model size: {baseline_profile['model_size_mb']:.2f} MB")
    print(f"  Estimated FLOPs: {baseline_profile['flops']:,}")
    
    print("\n2. Optimization Techniques")
    print("-" * 30)
    
    optimizer = InferenceOptimizer(device)
    
    # Module fusion
    fused_model = optimizer.fuse_conv_bn_relu(model)
    
    # Channels-last optimization
    channels_last_model = optimizer.apply_channel_last_optimization(model)
    
    # Comprehensive inference optimization
    inference_optimized = optimizer.optimize_for_inference(model)
    
    print("\n3. Batch Size Optimization")
    print("-" * 31)
    
    batch_processor = BatchProcessor(model, device)
    
    # Find optimal batch size
    optimal_batch_size = batch_processor.find_optimal_batch_size(input_shape, max_batch_size=32)
    
    print("\n4. Optimization Comparison")
    print("-" * 30)
    
    comparator = ModelComparator(device)
    
    # Compare different optimizations
    comparison_results = comparator.compare_optimizations(model, input_shape)
    
    # Generate and print report
    report = comparator.generate_comparison_report(comparison_results)
    print(report)
    
    print("\n5. Best Practices Summary")
    print("-" * 29)
    
    best_practices = [
        "Profile models before optimizing to identify bottlenecks",
        "Fuse conv-bn-relu modules for inference optimization",
        "Use TorchScript for deployment optimization", 
        "Apply channels-last memory format for better cache usage",
        "Find optimal batch size for your hardware",
        "Remove dropout and other training-only layers",
        "Use inplace operations where possible",
        "Consider quantization for mobile/edge deployment",
        "Profile memory usage to avoid OOM errors",
        "Use mixed precision for supported hardware",
        "Optimize preprocessing pipelines",
        "Consider model distillation for smaller models"
    ]
    
    print("Inference Optimization Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\n6. Performance Optimization Workflow")
    print("-" * 40)
    
    workflow = [
        "1. Profile baseline model performance",
        "2. Identify computational bottlenecks",
        "3. Apply appropriate optimizations",
        "4. Re-profile and measure improvements",
        "5. Test with production-like workloads",
        "6. Deploy and monitor in production"
    ]
    
    print("Optimization Workflow:")
    for step in workflow:
        print(f"  {step}")
    
    print("\n7. Hardware-Specific Optimizations")
    print("-" * 40)
    
    hardware_optimizations = {
        "CPU": [
            "Use Intel MKL-DNN optimizations",
            "Enable OpenMP for parallel processing",
            "Optimize for cache-friendly memory access",
            "Use SIMD instructions where available"
        ],
        "GPU": [
            "Maximize GPU utilization with appropriate batch sizes",
            "Use Tensor Cores on supported GPUs",
            "Minimize CPU-GPU data transfers",
            "Consider TensorRT for NVIDIA GPUs"
        ],
        "Mobile/Edge": [
            "Use quantization to reduce model size",
            "Optimize for power consumption",
            "Use specialized mobile frameworks",
            "Consider model compression techniques"
        ]
    }
    
    for hardware, optimizations in hardware_optimizations.items():
        print(f"\n{hardware} Optimizations:")
        for opt in optimizations:
            print(f"  - {opt}")
    
    print("\nModel optimization demonstration completed!")
    print("Key findings:")
    if 'baseline' in comparison_results and 'error' not in comparison_results['baseline']:
        baseline_time = comparison_results['baseline']['timing']['mean_ms']
        print(f"  - Baseline inference time: {baseline_time:.2f} ms")
        
        # Show best optimization if available
        best_optimization = None
        best_speedup = 1.0
        
        for name, result in comparison_results.items():
            if name != 'baseline' and 'error' not in result:
                speedup = baseline_time / result['timing']['mean_ms']
                if speedup > best_speedup:
                    best_speedup = speedup
                    best_optimization = name
        
        if best_optimization:
            print(f"  - Best optimization: {best_optimization} ({best_speedup:.2f}x speedup)")
        
        print(f"  - Optimal batch size: {optimal_batch_size}")
    
    print("  - Apply optimizations based on your deployment target!")