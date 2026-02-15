import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
import warnings

# PyTorch Performance Optimization Guide
class PerformanceOptimizer:
    """Comprehensive PyTorch performance optimization utilities"""
    
    def __init__(self):
        self.optimization_history = []
    
    def optimize_data_loading(self, dataloader_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize data loading performance"""
        
        optimized_config = dataloader_config.copy()
        
        # Optimize num_workers
        if 'num_workers' not in optimized_config:
            import os
            optimized_config['num_workers'] = min(8, os.cpu_count())
        
        # Enable pin_memory for GPU training
        if torch.cuda.is_available() and 'pin_memory' not in optimized_config:
            optimized_config['pin_memory'] = True
        
        # Use persistent_workers for faster worker startup
        if optimized_config.get('num_workers', 0) > 0:
            optimized_config['persistent_workers'] = True
        
        # Optimize prefetch_factor
        if 'prefetch_factor' not in optimized_config:
            optimized_config['prefetch_factor'] = 2
        
        return optimized_config
    
    def optimize_model_memory(self, model: nn.Module) -> nn.Module:
        """Apply memory optimization techniques"""
        
        # Convert to half precision if supported
        if torch.cuda.is_available():
            model = model.half()
        
        # Enable memory efficient attention (if available)
        for module in model.modules():
            if hasattr(module, 'set_use_memory_efficient_attention_xformers'):
                module.set_use_memory_efficient_attention_xformers(True)
        
        return model
    
    def apply_mixed_precision(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> Tuple:
        """Setup mixed precision training"""
        
        try:
            from torch.cuda.amp import GradScaler, autocast
            scaler = GradScaler()
            
            def training_step(data, target):
                with autocast():
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                return loss
            
            return training_step, scaler
        except ImportError:
            warnings.warn("Mixed precision not available")
            return None, None
    
    def optimize_model_compilation(self, model: nn.Module) -> nn.Module:
        """Apply model compilation optimizations"""
        
        # Try TorchScript compilation
        try:
            model = torch.jit.script(model)
            print("✓ Model compiled with TorchScript")
        except Exception:
            try:
                # Try tracing instead
                dummy_input = torch.randn(1, 784)  # Adjust for your model
                model = torch.jit.trace(model, dummy_input)
                print("✓ Model traced with TorchScript")
            except Exception:
                print("⚠ TorchScript compilation failed")
        
        return model
    
    def get_performance_tips(self) -> Dict[str, List[str]]:
        """Get comprehensive performance optimization tips"""
        
        tips = {
            "Data Loading": [
                "Use appropriate num_workers (typically 4-8)",
                "Enable pin_memory for GPU training",
                "Use persistent_workers for faster startup",
                "Preprocess data offline when possible",
                "Use efficient data formats (HDF5, LMDB)",
                "Apply transforms on GPU when possible",
            ],
            
            "Model Optimization": [
                "Use mixed precision training (AMP)",
                "Apply gradient checkpointing for large models",
                "Use in-place operations where possible",
                "Fuse operations (BatchNorm + Conv)",
                "Use efficient attention mechanisms",
                "Consider model pruning and quantization",
            ],
            
            "Memory Management": [
                "Use gradient accumulation for large batches",
                "Clear cache with torch.cuda.empty_cache()",
                "Use smaller batch sizes if memory limited",
                "Enable memory mapping for large datasets",
                "Profile memory usage to identify leaks",
                "Use activation checkpointing",
            ],
            
            "CUDA Optimization": [
                "Use cuDNN for convolutions",
                "Enable tensor cores with half precision",
                "Minimize host-device transfers",
                "Use asynchronous operations",
                "Optimize kernel launch parameters",
                "Profile with NVIDIA tools",
            ],
            
            "Training Efficiency": [
                "Use learning rate scheduling",
                "Apply early stopping to save time",
                "Use distributed training for scale",
                "Cache preprocessing results",
                "Use efficient optimizers (AdamW vs Adam)",
                "Monitor training metrics efficiently",
            ]
        }
        
        return tips

class BenchmarkSuite:
    """Performance benchmarking utilities"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_operations(self) -> Dict[str, float]:
        """Benchmark common PyTorch operations"""
        
        operations = {
            "matrix_multiply": lambda: torch.mm(torch.randn(1000, 1000), torch.randn(1000, 1000)),
            "convolution": lambda: F.conv2d(torch.randn(1, 3, 224, 224), torch.randn(64, 3, 7, 7)),
            "batch_norm": lambda: F.batch_norm(torch.randn(32, 64, 56, 56), None, None),
            "relu_activation": lambda: F.relu(torch.randn(1000, 1000)),
            "softmax": lambda: F.softmax(torch.randn(1000, 1000), dim=1),
        }
        
        results = {}
        for name, op in operations.items():
            # Warmup
            for _ in range(10):
                _ = op()
            
            # Benchmark
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                for _ in range(100):
                    _ = op()
                end.record()
                torch.cuda.synchronize()
                
                results[name] = start.elapsed_time(end) / 100  # ms per operation
            else:
                start_time = time.time()
                for _ in range(100):
                    _ = op()
                end_time = time.time()
                results[name] = (end_time - start_time) * 1000 / 100  # ms per operation
        
        return results

def demonstrate_optimizations():
    """Demonstrate various optimization techniques"""
    
    optimizer = PerformanceOptimizer()
    
    print("PyTorch Performance Optimization")
    print("=" * 35)
    
    print("\n1. Data Loading Optimization")
    print("-" * 30)
    
    # Original config
    original_config = {"batch_size": 32, "shuffle": True}
    optimized_config = optimizer.optimize_data_loading(original_config)
    
    print(f"Original config: {original_config}")
    print(f"Optimized config: {optimized_config}")
    
    print("\n2. Model Memory Optimization")
    print("-" * 31)
    
    # Create sample model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    original_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Original model memory: {original_memory / 1024**2:.2f} MB")
    
    if torch.cuda.is_available():
        optimized_model = optimizer.optimize_model_memory(model.cuda())
        optimized_memory = sum(p.numel() * p.element_size() for p in optimized_model.parameters())
        print(f"Optimized model memory: {optimized_memory / 1024**2:.2f} MB")
        print(f"Memory reduction: {(1 - optimized_memory/original_memory)*100:.1f}%")
    
    print("\n3. Performance Tips")
    print("-" * 20)
    
    tips = optimizer.get_performance_tips()
    for category, tip_list in tips.items():
        print(f"\n{category}:")
        for tip in tip_list[:3]:  # Show first 3 tips
            print(f"  • {tip}")
    
    print("\n4. Operation Benchmarks")
    print("-" * 24)
    
    benchmark = BenchmarkSuite()
    results = benchmark.benchmark_operations()
    
    print("Operation benchmark results (ms per operation):")
    for op_name, time_ms in results.items():
        print(f"  {op_name:15s}: {time_ms:.3f} ms")
    
    print("\n5. Quick Optimization Checklist")
    print("-" * 34)
    
    checklist = [
        "✓ Use DataLoader with optimal num_workers",
        "✓ Enable mixed precision training (AMP)",
        "✓ Use pin_memory for GPU training",
        "✓ Apply model compilation (TorchScript)",
        "✓ Profile code to identify bottlenecks",
        "✓ Use efficient data formats and preprocessing",
        "✓ Monitor GPU utilization and memory usage",
        "✓ Use gradient accumulation for large effective batch sizes",
        "✓ Apply learning rate scheduling",
        "✓ Use distributed training for multi-GPU setups"
    ]
    
    for item in checklist:
        print(f"  {item}")

if __name__ == "__main__":
    demonstrate_optimizations()
    
    print("\n" + "="*50)
    print("PERFORMANCE OPTIMIZATION SUMMARY")
    print("="*50)
    
    print("\n🚀 KEY OPTIMIZATION AREAS:")
    print("1. Data Pipeline: Efficient loading and preprocessing")
    print("2. Model Architecture: Memory-efficient designs")
    print("3. Training Loop: Mixed precision and compilation")
    print("4. Memory Management: Gradient accumulation and caching")
    print("5. Hardware Utilization: GPU optimization and parallelization")
    
    print("\n⚡ QUICK WINS:")
    print("• Enable mixed precision training (+30-50% speedup)")
    print("• Optimize DataLoader settings (+20-40% speedup)")
    print("• Use TorchScript compilation (+10-20% speedup)")
    print("• Apply gradient accumulation (larger effective batches)")
    print("• Profile and eliminate bottlenecks")
    
    print("\n🔧 TOOLS TO USE:")
    print("• PyTorch Profiler for performance analysis")
    print("• TensorBoard for monitoring and visualization")
    print("• NVIDIA NSight for CUDA optimization")
    print("• torch.compile() for latest PyTorch versions")
    print("• Mixed precision training (torch.cuda.amp)")
    
    print("\n📊 MONITORING METRICS:")
    print("• GPU utilization percentage")
    print("• Memory usage and peak allocation")
    print("• Training throughput (samples/second)")
    print("• Time per batch and per epoch")
    print("• Model inference latency")
    
    print("\nPerformance optimization is an iterative process!")
    print("Always profile first, optimize, then measure improvements.")