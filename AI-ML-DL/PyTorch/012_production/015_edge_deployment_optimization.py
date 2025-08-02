import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any

# Edge-Optimized Models
class EdgeOptimizedCNN(nn.Module):
    """CNN optimized for edge deployment"""
    
    def __init__(self, num_classes: int = 10, width_multiplier: float = 0.5):
        super().__init__()
        
        # Efficient architecture with depthwise separable convolutions
        base_channels = int(32 * width_multiplier)
        
        self.features = nn.Sequential(
            # Standard convolution
            nn.Conv2d(3, base_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU6(inplace=True),
            
            # Depthwise separable blocks
            self._make_depthwise_block(base_channels, base_channels * 2, 1),
            self._make_depthwise_block(base_channels * 2, base_channels * 4, 2),
            self._make_depthwise_block(base_channels * 4, base_channels * 8, 2),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(base_channels * 8, num_classes)
        )
    
    def _make_depthwise_block(self, in_channels: int, out_channels: int, stride: int):
        """Create depthwise separable convolution block"""
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

class MicroNet(nn.Module):
    """Ultra-lightweight network for microcontrollers"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # Extremely small network
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 5, 2, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

# Edge Optimization Techniques
class EdgeOptimizer:
    """Optimize models for edge deployment"""
    
    def __init__(self):
        self.optimization_methods = []
    
    def apply_quantization(self, model: nn.Module, 
                          calibration_data: Optional[torch.utils.data.DataLoader] = None,
                          quantization_type: str = "dynamic") -> nn.Module:
        """Apply quantization for edge deployment"""
        
        model.eval()
        
        if quantization_type == "dynamic":
            # Dynamic quantization (post-training)
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )
            self.optimization_methods.append("dynamic_quantization")
            
        elif quantization_type == "static" and calibration_data:
            # Static quantization with calibration
            model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
            prepared_model = torch.quantization.prepare(model)
            
            # Calibrate
            with torch.no_grad():
                for batch_idx, (data, _) in enumerate(calibration_data):
                    if batch_idx >= 10:  # Use limited data for calibration
                        break
                    prepared_model(data)
            
            quantized_model = torch.quantization.convert(prepared_model)
            self.optimization_methods.append("static_quantization")
        
        else:
            quantized_model = model
            print("Warning: Quantization type not supported or missing calibration data")
        
        print(f"✓ Applied {quantization_type} quantization")
        return quantized_model
    
    def apply_pruning(self, model: nn.Module, 
                     sparsity: float = 0.3,
                     structured: bool = False) -> nn.Module:
        """Apply pruning to reduce model size"""
        
        if structured:
            # Structured pruning (remove entire channels/filters)
            self._structured_pruning(model, sparsity)
            self.optimization_methods.append("structured_pruning")
        else:
            # Unstructured pruning (remove individual weights)
            self._unstructured_pruning(model, sparsity)
            self.optimization_methods.append("unstructured_pruning")
        
        print(f"✓ Applied pruning with {sparsity*100:.1f}% sparsity")
        return model
    
    def _unstructured_pruning(self, model: nn.Module, sparsity: float):
        """Apply unstructured weight pruning"""
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Calculate threshold for pruning
                weights = module.weight.data.abs()
                threshold = torch.quantile(weights, sparsity)
                
                # Create mask
                mask = weights > threshold
                
                # Apply mask
                module.weight.data *= mask.float()
    
    def _structured_pruning(self, model: nn.Module, sparsity: float):
        """Apply structured channel pruning"""
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.out_channels > 1:
                # Calculate channel importance (L1 norm)
                channel_importance = torch.sum(torch.abs(module.weight.data), dim=(1, 2, 3))
                
                # Determine channels to prune
                num_prune = int(sparsity * module.out_channels)
                _, prune_indices = torch.topk(channel_importance, num_prune, largest=False)
                
                # Create mask for channels to keep
                keep_mask = torch.ones(module.out_channels, dtype=torch.bool)
                keep_mask[prune_indices] = False
                
                # Note: Actual structured pruning requires modifying the architecture
                # This is a simplified demonstration
                print(f"  Would prune {num_prune} channels from {name}")
    
    def apply_knowledge_distillation(self, student_model: nn.Module,
                                   teacher_model: nn.Module,
                                   train_loader: torch.utils.data.DataLoader,
                                   temperature: float = 4.0,
                                   alpha: float = 0.7,
                                   epochs: int = 10) -> nn.Module:
        """Apply knowledge distillation for model compression"""
        
        teacher_model.eval()
        student_model.train()
        
        optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
        criterion_hard = nn.CrossEntropyLoss()
        criterion_soft = nn.KLDivLoss(reduction='batchmean')
        
        print(f"Starting knowledge distillation for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                if batch_idx >= 10:  # Limit for demo
                    break
                
                optimizer.zero_grad()
                
                # Student predictions
                student_outputs = student_model(data)
                
                # Teacher predictions (no gradients)
                with torch.no_grad():
                    teacher_outputs = teacher_model(data)
                
                # Hard loss (student vs true labels)
                hard_loss = criterion_hard(student_outputs, targets)
                
                # Soft loss (student vs teacher)
                soft_targets = F.softmax(teacher_outputs / temperature, dim=1)
                soft_predictions = F.log_softmax(student_outputs / temperature, dim=1)
                soft_loss = criterion_soft(soft_predictions, soft_targets) * (temperature ** 2)
                
                # Combined loss
                total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
            
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                print(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
        
        self.optimization_methods.append("knowledge_distillation")
        print("✓ Knowledge distillation completed")
        return student_model
    
    def optimize_for_mobile(self, model: nn.Module) -> torch.jit.ScriptModule:
        """Optimize model specifically for mobile deployment"""
        
        model.eval()
        
        # Convert to TorchScript
        try:
            example_input = torch.randn(1, 3, 224, 224)
            scripted_model = torch.jit.trace(model, example_input)
        except:
            scripted_model = torch.jit.script(model)
        
        # Optimize for mobile
        optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(
            scripted_model,
            optimization_blocklist={
                "remove_dropout",  # Keep for consistency
                "fuse_add_relu"    # May cause issues on some devices
            }
        )
        
        self.optimization_methods.append("mobile_optimization")
        print("✓ Mobile optimization applied")
        return optimized_model

# Edge Performance Analyzer
class EdgePerformanceAnalyzer:
    """Analyze model performance for edge deployment"""
    
    def __init__(self):
        self.benchmarks = {}
    
    def analyze_model_complexity(self, model: nn.Module, 
                                input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> Dict[str, Any]:
        """Analyze model complexity metrics"""
        
        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Model size in MB
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        
        # FLOPs estimation (simplified)
        flops = self._estimate_flops(model, input_shape)
        
        # Memory requirements
        input_size_mb = np.prod(input_shape) * 4 / (1024**2)  # Assuming float32
        
        complexity = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'estimated_flops': flops,
            'input_memory_mb': input_size_mb,
            'parameters_per_class': total_params / getattr(model, 'num_classes', 10)
        }
        
        return complexity
    
    def _estimate_flops(self, model: nn.Module, input_shape: Tuple[int, ...]) -> int:
        """Estimate FLOPs for the model"""
        
        model.eval()
        flops = 0
        
        def flop_count_hook(module, input, output):
            nonlocal flops
            
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
                total_conv_flops = conv_per_position_flops * active_elements_count * filters_per_channel
                
                # Add bias FLOPs if present
                if module.bias is not None:
                    total_conv_flops += active_elements_count * out_channels
                
                flops += total_conv_flops
            
            elif isinstance(module, nn.Linear):
                # Linear layer FLOPs
                input_size = input[0].numel()
                flops += input_size * module.out_features
        
        # Register hooks
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handle = module.register_forward_hook(flop_count_hook)
                hooks.append(handle)
        
        # Forward pass
        with torch.no_grad():
            dummy_input = torch.randn(input_shape)
            model(dummy_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return flops
    
    def benchmark_inference_speed(self, model: nn.Module,
                                 input_shape: Tuple[int, ...] = (1, 3, 224, 224),
                                 num_runs: int = 100,
                                 device: str = 'cpu') -> Dict[str, float]:
        """Benchmark inference speed on target device"""
        
        model = model.to(device)
        model.eval()
        
        # Create test input
        test_input = torch.randn(input_shape).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input)
        
        # Benchmark
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                _ = model(test_input)
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        times = np.array(times)
        
        benchmark_results = {
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'p95_time_ms': np.percentile(times, 95),
            'p99_time_ms': np.percentile(times, 99),
            'throughput_fps': 1000 / np.mean(times)
        }
        
        return benchmark_results
    
    def analyze_memory_usage(self, model: nn.Module,
                           input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> Dict[str, float]:
        """Analyze memory usage patterns"""
        
        model.eval()
        
        # Calculate parameter memory
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        
        # Calculate activation memory (estimated)
        activation_memory = 0
        
        def memory_hook(module, input, output):
            nonlocal activation_memory
            if hasattr(output, 'numel'):
                activation_memory += output.numel() * 4 / (1024**2)  # Assuming float32
        
        # Register hooks
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU, nn.MaxPool2d)):
                handle = module.register_forward_hook(memory_hook)
                hooks.append(handle)
        
        # Forward pass
        with torch.no_grad():
            test_input = torch.randn(input_shape)
            _ = model(test_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Input memory
        input_memory = np.prod(input_shape) * 4 / (1024**2)
        
        total_memory = param_memory + activation_memory + input_memory
        
        return {
            'parameter_memory_mb': param_memory,
            'activation_memory_mb': activation_memory,
            'input_memory_mb': input_memory,
            'total_memory_mb': total_memory
        }

# Edge Device Targets
class EdgeDeviceTargets:
    """Define optimization targets for different edge devices"""
    
    DEVICE_SPECS = {
        'raspberry_pi_4': {
            'cpu_cores': 4,
            'memory_mb': 4096,
            'max_model_size_mb': 100,
            'target_inference_ms': 500,
            'power_budget_watts': 3
        },
        'jetson_nano': {
            'cpu_cores': 4,
            'memory_mb': 4096,
            'gpu_memory_mb': 2048,
            'max_model_size_mb': 200,
            'target_inference_ms': 100,
            'power_budget_watts': 10
        },
        'mobile_phone': {
            'cpu_cores': 8,
            'memory_mb': 6144,
            'max_model_size_mb': 50,
            'target_inference_ms': 200,
            'power_budget_watts': 2
        },
        'microcontroller': {
            'cpu_mhz': 168,
            'memory_kb': 512,
            'flash_kb': 2048,
            'max_model_size_kb': 500,
            'target_inference_ms': 1000,
            'power_budget_watts': 0.1
        }
    }
    
    @classmethod
    def get_optimization_strategy(cls, device_type: str) -> Dict[str, Any]:
        """Get optimization strategy for target device"""
        
        if device_type not in cls.DEVICE_SPECS:
            raise ValueError(f"Unknown device type: {device_type}")
        
        specs = cls.DEVICE_SPECS[device_type]
        
        if device_type == 'microcontroller':
            strategy = {
                'quantization': 'int8',
                'pruning_sparsity': 0.8,
                'knowledge_distillation': True,
                'architecture': 'micro',
                'max_parameters': 10000
            }
        elif device_type == 'mobile_phone':
            strategy = {
                'quantization': 'dynamic',
                'pruning_sparsity': 0.5,
                'mobile_optimization': True,
                'architecture': 'mobilenet',
                'max_parameters': 5000000
            }
        elif device_type == 'raspberry_pi_4':
            strategy = {
                'quantization': 'dynamic',
                'pruning_sparsity': 0.3,
                'architecture': 'efficient',
                'max_parameters': 10000000
            }
        else:  # jetson_nano
            strategy = {
                'quantization': 'dynamic',
                'pruning_sparsity': 0.2,
                'gpu_optimization': True,
                'architecture': 'standard',
                'max_parameters': 20000000
            }
        
        strategy['target_specs'] = specs
        return strategy

# Deployment Package Generator
class EdgeDeploymentPackage:
    """Generate deployment packages for edge devices"""
    
    def __init__(self, target_device: str):
        self.target_device = target_device
        self.optimization_strategy = EdgeDeviceTargets.get_optimization_strategy(target_device)
    
    def create_deployment_package(self, model: nn.Module, 
                                 package_name: str) -> str:
        """Create complete deployment package"""
        
        import os
        import tempfile
        import zipfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = os.path.join(temp_dir, package_name)
            os.makedirs(package_dir, exist_ok=True)
            
            # Save optimized model
            model_path = os.path.join(package_dir, "model.pth")
            torch.save(model.state_dict(), model_path)
            
            # Create deployment script
            deploy_script = self._create_deployment_script()
            script_path = os.path.join(package_dir, "deploy.py")
            with open(script_path, 'w') as f:
                f.write(deploy_script)
            
            # Create requirements
            requirements = self._create_requirements()
            req_path = os.path.join(package_dir, "requirements.txt")
            with open(req_path, 'w') as f:
                f.write(requirements)
            
            # Create configuration
            config = {
                'target_device': self.target_device,
                'optimization_strategy': self.optimization_strategy,
                'model_info': {
                    'class_name': model.__class__.__name__,
                    'parameters': sum(p.numel() for p in model.parameters())
                }
            }
            
            config_path = os.path.join(package_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Create README
            readme_content = self._create_readme()
            readme_path = os.path.join(package_dir, "README.md")
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            
            # Create package archive
            package_path = f"{package_name}_{self.target_device}.zip"
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(package_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zipf.write(file_path, arcname)
            
            print(f"✓ Deployment package created: {package_path}")
            return package_path
    
    def _create_deployment_script(self) -> str:
        """Create deployment script for target device"""
        
        script_content = f'''#!/usr/bin/env python3
"""
Edge deployment script for {self.target_device}
"""

import torch
import torch.nn as nn
import json
import time
from typing import Dict, Any

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

class ModelWrapper:
    def __init__(self, model_path: str):
        self.device = torch.device('cpu')  # Edge devices typically use CPU
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path: str):
        # Model loading logic specific to your architecture
        # This is a placeholder - replace with actual model class
        pass
        
    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
        return output
    
    def benchmark(self, num_runs: int = 100):
        # Benchmark inference speed
        test_input = torch.randn(1, 3, 224, 224)
        
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = self.predict(test_input)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)
        
        avg_time = sum(times) / len(times)
        print(f"Average inference time: {{avg_time:.2f}} ms")
        return avg_time

if __name__ == "__main__":
    model_wrapper = ModelWrapper("model.pth")
    model_wrapper.benchmark()
'''
        return script_content
    
    def _create_requirements(self) -> str:
        """Create requirements based on target device"""
        
        if self.target_device == 'microcontroller':
            return "# Microcontroller deployment - no Python dependencies"
        else:
            return '''torch>=1.12.0
numpy>=1.21.0
pillow>=8.3.0
'''
    
    def _create_readme(self) -> str:
        """Create README for deployment package"""
        
        return f'''# Edge Deployment Package for {self.target_device.title()}

## Overview
This package contains an optimized PyTorch model for deployment on {self.target_device}.

## Target Device Specifications
{json.dumps(self.optimization_strategy['target_specs'], indent=2)}

## Optimization Strategy
{json.dumps({k: v for k, v in self.optimization_strategy.items() if k != 'target_specs'}, indent=2)}

## Installation
1. Extract the package
2. Install requirements: `pip install -r requirements.txt`
3. Run deployment script: `python deploy.py`

## Usage
```python
from deploy import ModelWrapper

model = ModelWrapper("model.pth")
output = model.predict(input_tensor)
```

## Performance Notes
- Model optimized for {self.target_device}
- Expected inference time: {self.optimization_strategy['target_specs'].get('target_inference_ms', 'N/A')} ms
- Memory usage optimized for available resources
'''

if __name__ == "__main__":
    print("Edge Deployment Optimization")
    print("=" * 33)
    
    # Create sample models
    standard_model = EdgeOptimizedCNN(num_classes=10, width_multiplier=1.0)
    compact_model = EdgeOptimizedCNN(num_classes=10, width_multiplier=0.25)
    micro_model = MicroNet(num_classes=10)
    
    print("\n1. Model Complexity Analysis")
    print("-" * 32)
    
    analyzer = EdgePerformanceAnalyzer()
    
    models = {
        'Standard': standard_model,
        'Compact': compact_model,
        'Micro': micro_model
    }
    
    print("Model Complexity Comparison:")
    print("-" * 30)
    print(f"{'Model':<10} {'Params':<12} {'Size (MB)':<12} {'FLOPs':<15} {'Params/Class':<12}")
    print("-" * 70)
    
    for name, model in models.items():
        complexity = analyzer.analyze_model_complexity(model)
        
        print(f"{name:<10} {complexity['total_parameters']:<12,} "
              f"{complexity['model_size_mb']:<12.2f} "
              f"{complexity['estimated_flops']:<15,} "
              f"{complexity['parameters_per_class']:<12,.0f}")
    
    print("\n2. Edge Optimization")
    print("-" * 22)
    
    optimizer = EdgeOptimizer()
    
    # Apply quantization
    quantized_model = optimizer.apply_quantization(compact_model, quantization_type="dynamic")
    
    # Apply pruning
    pruned_model = optimizer.apply_pruning(compact_model, sparsity=0.5)
    
    # Create teacher-student for knowledge distillation demo
    print("\nKnowledge Distillation Demo:")
    
    # Create dummy training data
    dummy_data = torch.randn(50, 3, 224, 224)
    dummy_labels = torch.randint(0, 10, (50,))
    dummy_dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
    dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=10)
    
    # Apply knowledge distillation
    distilled_model = optimizer.apply_knowledge_distillation(
        student_model=micro_model,
        teacher_model=standard_model,
        train_loader=dummy_loader,
        epochs=3
    )
    
    print(f"Applied optimizations: {optimizer.optimization_methods}")
    
    print("\n3. Performance Benchmarking")
    print("-" * 31)
    
    # Benchmark inference speed
    test_models = {
        'Original': compact_model,
        'Quantized': quantized_model,
        'Distilled': distilled_model
    }
    
    print("Inference Speed Comparison:")
    print("-" * 28)
    print(f"{'Model':<12} {'Mean (ms)':<12} {'P95 (ms)':<12} {'FPS':<8}")
    print("-" * 45)
    
    for name, model in test_models.items():
        try:
            benchmark = analyzer.benchmark_inference_speed(model, num_runs=50)
            
            print(f"{name:<12} {benchmark['mean_time_ms']:<12.2f} "
                  f"{benchmark['p95_time_ms']:<12.2f} "
                  f"{benchmark['throughput_fps']:<8.1f}")
        except Exception as e:
            print(f"{name:<12} Error: {str(e)[:20]}")
    
    print("\n4. Memory Analysis")
    print("-" * 21)
    
    # Analyze memory usage
    memory_analysis = analyzer.analyze_memory_usage(compact_model)
    
    print("Memory Usage Breakdown:")
    print(f"  Parameters: {memory_analysis['parameter_memory_mb']:.2f} MB")
    print(f"  Activations: {memory_analysis['activation_memory_mb']:.2f} MB")
    print(f"  Input: {memory_analysis['input_memory_mb']:.2f} MB")
    print(f"  Total: {memory_analysis['total_memory_mb']:.2f} MB")
    
    print("\n5. Device-Specific Optimization")
    print("-" * 37)
    
    # Show optimization strategies for different devices
    devices = ['raspberry_pi_4', 'jetson_nano', 'mobile_phone', 'microcontroller']
    
    print("Device Optimization Strategies:")
    for device in devices:
        strategy = EdgeDeviceTargets.get_optimization_strategy(device)
        specs = strategy['target_specs']
        
        print(f"\n{device.replace('_', ' ').title()}:")
        print(f"  Memory: {specs.get('memory_mb', specs.get('memory_kb', 'N/A'))}")
        print(f"  Target latency: {specs['target_inference_ms']} ms")
        print(f"  Quantization: {strategy.get('quantization', 'none')}")
        print(f"  Pruning: {strategy.get('pruning_sparsity', 0)*100:.0f}%")
    
    print("\n6. Deployment Package Creation")
    print("-" * 36)
    
    # Create deployment packages for different devices
    target_devices = ['raspberry_pi_4', 'mobile_phone']
    
    for device in target_devices:
        package_generator = EdgeDeploymentPackage(device)
        package_path = package_generator.create_deployment_package(
            compact_model, 
            f"edge_model_{device}"
        )
    
    print("\n7. Edge Deployment Best Practices")
    print("-" * 39)
    
    best_practices = [
        "Profile models on actual target hardware",
        "Use device-specific optimization strategies",
        "Implement proper error handling for edge cases",
        "Monitor battery usage for battery-powered devices",
        "Implement model caching for repeated inferences",
        "Use appropriate data types (int8, float16)",
        "Optimize memory access patterns",
        "Consider thermal constraints",
        "Implement graceful degradation strategies",
        "Test across different environmental conditions",
        "Use hardware acceleration when available",
        "Implement efficient preprocessing pipelines"
    ]
    
    print("Edge Deployment Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\n8. Hardware Acceleration Options")
    print("-" * 37)
    
    acceleration_options = {
        "ARM NEON": "SIMD instructions for ARM processors",
        "Intel MKL-DNN": "Optimized primitives for Intel CPUs",
        "NVIDIA TensorRT": "GPU acceleration for NVIDIA devices",
        "OpenVINO": "Intel's inference optimization toolkit",
        "CoreML": "Apple's machine learning framework",
        "NNAPI": "Android Neural Networks API",
        "TensorFlow Lite": "Lightweight ML framework",
        "ONNX Runtime": "Cross-platform ML inferencing"
    }
    
    print("Hardware Acceleration Options:")
    for option, description in acceleration_options.items():
        print(f"  {option}: {description}")
    
    print("\n9. Troubleshooting Edge Deployment")
    print("-" * 38)
    
    troubleshooting = [
        "High latency: Check model complexity and device specifications",
        "Out of memory: Reduce model size or batch size",
        "Poor accuracy: Validate quantization and pruning effects",
        "Thermal throttling: Monitor device temperature",
        "Battery drain: Optimize inference frequency",
        "Compatibility issues: Test on target OS and hardware"
    ]
    
    print("Common Issues and Solutions:")
    for i, issue in enumerate(troubleshooting, 1):
        print(f"{i}. {issue}")
    
    print("\nEdge deployment optimization completed!")
    print("Generated deployment packages:")
    for device in target_devices:
        print(f"  - edge_model_{device}_{device}.zip")
    
    print("\nKey optimization techniques applied:")
    print("  - Model architecture optimization")
    print("  - Dynamic quantization")
    print("  - Weight pruning")
    print("  - Knowledge distillation")
    print("  - Device-specific configuration")
    
    print("\nNext steps:")
    print("1. Test on actual target hardware")
    print("2. Measure real-world performance")
    print("3. Implement monitoring and telemetry")
    print("4. Plan for model updates and deployment")