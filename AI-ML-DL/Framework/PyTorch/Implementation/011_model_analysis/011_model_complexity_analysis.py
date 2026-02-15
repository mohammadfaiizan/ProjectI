import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import OrderedDict
import time

# Sample Models for Complexity Analysis
class ComplexityTestCNN(nn.Module):
    """CNN for complexity analysis demonstration"""
    
    def __init__(self, num_classes=10, width_multiplier=1.0):
        super().__init__()
        
        base_width = int(64 * width_multiplier)
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, base_width, 3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(base_width, base_width * 2, 3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(base_width * 2, base_width * 4, 3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(base_width * 4, base_width * 8, 3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_width * 8, base_width * 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(base_width * 4, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class EfficientBlock(nn.Module):
    """Efficient building block with depthwise separable convolutions"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride, 1, 
                                  groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x

class EfficientNet(nn.Module):
    """Efficient network using depthwise separable convolutions"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.blocks = nn.Sequential(
            EfficientBlock(32, 64, stride=2),
            EfficientBlock(64, 128, stride=2),
            EfficientBlock(128, 256, stride=2),
            EfficientBlock(256, 512, stride=1),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.classifier(x)
        return x

# FLOPs Calculation
class FLOPsCalculator:
    """Calculate FLOPs (Floating Point Operations) for PyTorch models"""
    
    def __init__(self):
        self.flops = 0
        self.hooks = []
    
    def conv_flop_jit(self, inputs, outputs, kernel_size, groups=1):
        """Calculate FLOPs for convolution operation"""
        batch_size = inputs[0].size(0)
        output_dims = outputs.shape[2:]
        kernel_dims = kernel_size
        in_channels = inputs[0].size(1)
        out_channels = outputs.shape[1]
        
        filters_per_channel = out_channels // groups
        conv_per_position_flops = int(np.prod(kernel_dims)) * in_channels // groups
        
        active_elements_count = batch_size * int(np.prod(output_dims))
        overall_conv_flops = conv_per_position_flops * active_elements_count * filters_per_channel
        
        return overall_conv_flops
    
    def linear_flop_jit(self, inputs, outputs):
        """Calculate FLOPs for linear operation"""
        input_last_dim = inputs[0].size(-1)
        num_instances = np.prod(inputs[0].shape[:-1])
        output_last_dim = outputs.shape[-1]
        
        return num_instances * input_last_dim * output_last_dim
    
    def pool_flop_jit(self, inputs, outputs, kernel_size):
        """Calculate FLOPs for pooling operation"""
        return np.prod(inputs[0].shape) * kernel_size
    
    def bn_flop_jit(self, inputs, outputs):
        """Calculate FLOPs for batch normalization"""
        return np.prod(inputs[0].shape) * 2  # mean and variance
    
    def register_hooks(self, model):
        """Register hooks to calculate FLOPs"""
        
        def conv_hook(module, input, output):
            if isinstance(module, nn.Conv2d):
                flops = self.conv_flop_jit(input, output, module.kernel_size, module.groups)
                self.flops += flops
        
        def linear_hook(module, input, output):
            if isinstance(module, nn.Linear):
                flops = self.linear_flop_jit(input, output)
                self.flops += flops
        
        def pool_hook(module, input, output):
            if isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                kernel_size = module.kernel_size
                if isinstance(kernel_size, int):
                    kernel_size = (kernel_size, kernel_size)
                flops = self.pool_flop_jit(input, output, np.prod(kernel_size))
                self.flops += flops
        
        def bn_hook(module, input, output):
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                flops = self.bn_flop_jit(input, output)
                self.flops += flops
        
        # Register hooks for all supported layers
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                handle = module.register_forward_hook(conv_hook)
                self.hooks.append(handle)
            elif isinstance(module, nn.Linear):
                handle = module.register_forward_hook(linear_hook)
                self.hooks.append(handle)
            elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                handle = module.register_forward_hook(pool_hook)
                self.hooks.append(handle)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                handle = module.register_forward_hook(bn_hook)
                self.hooks.append(handle)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def calculate_flops(self, model, input_tensor):
        """Calculate total FLOPs for model"""
        self.flops = 0
        self.register_hooks(model)
        
        model.eval()
        with torch.no_grad():
            _ = model(input_tensor)
        
        self.remove_hooks()
        return self.flops

# Parameter Counting and Analysis
class ParameterAnalyzer:
    """Analyze model parameters in detail"""
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def count_parameters(self, trainable_only: bool = False) -> Dict[str, int]:
        """Count parameters by category"""
        
        param_counts = {
            'total': 0,
            'trainable': 0,
            'non_trainable': 0,
            'conv_layers': 0,
            'linear_layers': 0,
            'bn_layers': 0,
            'embedding_layers': 0
        }
        
        for name, param in self.model.named_parameters():
            param_count = param.numel()
            param_counts['total'] += param_count
            
            if param.requires_grad:
                param_counts['trainable'] += param_count
            else:
                param_counts['non_trainable'] += param_count
            
            # Categorize by layer type
            if 'conv' in name:
                param_counts['conv_layers'] += param_count
            elif 'linear' in name or 'fc' in name or 'classifier' in name:
                param_counts['linear_layers'] += param_count
            elif 'bn' in name or 'norm' in name:
                param_counts['bn_layers'] += param_count
            elif 'embedding' in name:
                param_counts['embedding_layers'] += param_count
        
        return param_counts
    
    def get_layer_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed parameter information for each layer"""
        
        layer_info = {}
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                info = {
                    'type': type(module).__name__,
                    'parameters': 0,
                    'trainable_parameters': 0,
                    'parameter_details': {}
                }
                
                for param_name, param in module.named_parameters():
                    param_count = param.numel()
                    info['parameters'] += param_count
                    
                    if param.requires_grad:
                        info['trainable_parameters'] += param_count
                    
                    info['parameter_details'][param_name] = {
                        'shape': list(param.shape),
                        'count': param_count,
                        'requires_grad': param.requires_grad,
                        'dtype': str(param.dtype)
                    }
                
                if info['parameters'] > 0:
                    layer_info[name] = info
        
        return layer_info
    
    def calculate_memory_usage(self, input_shape: Tuple[int, ...], 
                             dtype: torch.dtype = torch.float32) -> Dict[str, float]:
        """Calculate memory usage for model and activations"""
        
        # Parameter memory
        param_memory = 0
        for param in self.model.parameters():
            param_memory += param.numel() * param.element_size()
        
        # Gradient memory (same as parameters for trainable params)
        grad_memory = 0
        for param in self.model.parameters():
            if param.requires_grad:
                grad_memory += param.numel() * param.element_size()
        
        # Estimate activation memory by running a forward pass
        device = next(self.model.parameters()).device
        test_input = torch.randn(1, *input_shape, dtype=dtype).to(device)
        
        activation_memory = 0
        hooks = []
        
        def memory_hook(module, input, output):
            nonlocal activation_memory
            if isinstance(output, torch.Tensor):
                activation_memory += output.numel() * output.element_size()
            elif isinstance(output, (list, tuple)):
                for tensor in output:
                    if isinstance(tensor, torch.Tensor):
                        activation_memory += tensor.numel() * tensor.element_size()
        
        # Register hooks
        for module in self.model.modules():
            handle = module.register_forward_hook(memory_hook)
            hooks.append(handle)
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            _ = self.model(test_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Convert to MB
        memory_info = {
            'parameters_mb': param_memory / (1024**2),
            'gradients_mb': grad_memory / (1024**2),
            'activations_mb': activation_memory / (1024**2),
            'total_mb': (param_memory + grad_memory + activation_memory) / (1024**2)
        }
        
        return memory_info
    
    def analyze_parameter_efficiency(self) -> Dict[str, float]:
        """Analyze parameter efficiency metrics"""
        
        layer_info = self.get_layer_parameters()
        total_params = sum(info['parameters'] for info in layer_info.values())
        
        efficiency_metrics = {}
        
        # Parameter distribution
        for layer_name, info in layer_info.items():
            param_ratio = info['parameters'] / total_params
            efficiency_metrics[f'param_ratio_{layer_name}'] = param_ratio
        
        # Layer type efficiency
        layer_type_params = {}
        for info in layer_info.values():
            layer_type = info['type']
            if layer_type not in layer_type_params:
                layer_type_params[layer_type] = 0
            layer_type_params[layer_type] += info['parameters']
        
        for layer_type, params in layer_type_params.items():
            efficiency_metrics[f'type_ratio_{layer_type}'] = params / total_params
        
        return efficiency_metrics

class ModelComplexityAnalyzer:
    """Comprehensive model complexity analysis"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.flops_calculator = FLOPsCalculator()
        self.param_analyzer = ParameterAnalyzer(model)
    
    def comprehensive_analysis(self, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Perform comprehensive complexity analysis"""
        
        # Create test input
        test_input = torch.randn(1, *input_shape).to(self.device)
        
        analysis = {}
        
        # 1. Parameter analysis
        print("Analyzing parameters...")
        param_counts = self.param_analyzer.count_parameters()
        layer_params = self.param_analyzer.get_layer_parameters()
        memory_usage = self.param_analyzer.calculate_memory_usage(input_shape)
        efficiency_metrics = self.param_analyzer.analyze_parameter_efficiency()
        
        analysis['parameters'] = {
            'counts': param_counts,
            'layer_details': layer_params,
            'memory_usage': memory_usage,
            'efficiency': efficiency_metrics
        }
        
        # 2. FLOPs analysis
        print("Calculating FLOPs...")
        total_flops = self.flops_calculator.calculate_flops(self.model, test_input)
        
        analysis['flops'] = {
            'total': total_flops,
            'gflops': total_flops / 1e9,
            'mflops': total_flops / 1e6
        }
        
        # 3. Inference timing
        print("Measuring inference time...")
        timing_results = self._measure_inference_time(test_input)
        analysis['timing'] = timing_results
        
        # 4. Model size analysis
        print("Analyzing model size...")
        size_analysis = self._analyze_model_size()
        analysis['size'] = size_analysis
        
        # 5. Computational efficiency
        analysis['efficiency'] = self._calculate_efficiency_metrics(
            total_flops, param_counts['total'], timing_results['avg_time_ms']
        )
        
        return analysis
    
    def _measure_inference_time(self, test_input: torch.Tensor, 
                               num_runs: int = 100) -> Dict[str, float]:
        """Measure inference timing"""
        
        self.model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(test_input)
        
        # Timing runs
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.time()
                _ = self.model(test_input)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            'avg_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'throughput_fps': 1000 / np.mean(times)  # Frames per second
        }
    
    def _analyze_model_size(self) -> Dict[str, float]:
        """Analyze model size in different formats"""
        
        # Calculate size of state dict
        state_dict = self.model.state_dict()
        
        # Calculate sizes
        total_size_bytes = 0
        for tensor in state_dict.values():
            total_size_bytes += tensor.numel() * tensor.element_size()
        
        # Different precision sizes (theoretical)
        fp32_size_mb = total_size_bytes / (1024**2)
        fp16_size_mb = fp32_size_mb / 2
        int8_size_mb = fp32_size_mb / 4
        
        return {
            'fp32_size_mb': fp32_size_mb,
            'fp16_size_mb': fp16_size_mb,
            'int8_size_mb': int8_size_mb,
            'compression_ratio_fp16': fp32_size_mb / fp16_size_mb,
            'compression_ratio_int8': fp32_size_mb / int8_size_mb
        }
    
    def _calculate_efficiency_metrics(self, flops: int, params: int, 
                                    inference_time_ms: float) -> Dict[str, float]:
        """Calculate various efficiency metrics"""
        
        return {
            'flops_per_param': flops / params if params > 0 else 0,
            'params_per_flop': params / flops if flops > 0 else 0,
            'gflops_per_second': (flops / 1e9) / (inference_time_ms / 1000) if inference_time_ms > 0 else 0,
            'energy_efficiency': flops / (inference_time_ms * params) if inference_time_ms > 0 and params > 0 else 0
        }
    
    def compare_models(self, models: Dict[str, nn.Module], 
                      input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Compare complexity of multiple models"""
        
        comparison = {}
        
        for model_name, model in models.items():
            print(f"\nAnalyzing {model_name}...")
            
            # Temporarily switch model
            original_model = self.model
            self.model = model.to(self.device)
            self.param_analyzer = ParameterAnalyzer(self.model)
            
            # Analyze model
            analysis = self.comprehensive_analysis(input_shape)
            comparison[model_name] = analysis
            
            # Restore original model
            self.model = original_model
            self.param_analyzer = ParameterAnalyzer(self.model)
        
        return comparison
    
    def visualize_complexity(self, analysis: Dict[str, Any], save_path: str = None):
        """Visualize model complexity analysis"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Parameter distribution by layer type
        param_counts = analysis['parameters']['counts']
        layer_types = ['conv_layers', 'linear_layers', 'bn_layers']
        layer_params = [param_counts.get(lt, 0) for lt in layer_types]
        
        axes[0, 0].pie(layer_params, labels=['Conv', 'Linear', 'BatchNorm'], autopct='%1.1f%%')
        axes[0, 0].set_title('Parameter Distribution by Layer Type')
        
        # 2. Memory usage breakdown
        memory_usage = analysis['parameters']['memory_usage']
        memory_types = ['parameters_mb', 'gradients_mb', 'activations_mb']
        memory_values = [memory_usage[mt] for mt in memory_types]
        
        axes[0, 1].bar(['Parameters', 'Gradients', 'Activations'], memory_values)
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].set_title('Memory Usage Breakdown')
        
        # 3. Model size comparison
        size_info = analysis['size']
        precisions = ['fp32_size_mb', 'fp16_size_mb', 'int8_size_mb']
        sizes = [size_info[p] for p in precisions]
        
        axes[0, 2].bar(['FP32', 'FP16', 'INT8'], sizes)
        axes[0, 2].set_ylabel('Model Size (MB)')
        axes[0, 2].set_title('Model Size by Precision')
        
        # 4. Parameter count by layer
        layer_details = analysis['parameters']['layer_details']
        if layer_details:
            layer_names = list(layer_details.keys())[:10]  # Top 10 layers
            layer_param_counts = [layer_details[name]['parameters'] for name in layer_names]
            
            axes[1, 0].barh(range(len(layer_names)), layer_param_counts)
            axes[1, 0].set_yticks(range(len(layer_names)))
            axes[1, 0].set_yticklabels([name.split('.')[-1][:15] for name in layer_names])
            axes[1, 0].set_xlabel('Parameter Count')
            axes[1, 0].set_title('Parameters by Layer (Top 10)')
        
        # 5. Efficiency metrics
        efficiency = analysis['efficiency']
        metrics = ['flops_per_param', 'gflops_per_second']
        values = [efficiency[m] for m in metrics]
        
        axes[1, 1].bar(['FLOPs/Param', 'GFLOPs/sec'], values)
        axes[1, 1].set_ylabel('Efficiency')
        axes[1, 1].set_title('Efficiency Metrics')
        
        # 6. Timing analysis
        timing = analysis['timing']
        axes[1, 2].bar(['Avg', 'Min', 'Max'], 
                      [timing['avg_time_ms'], timing['min_time_ms'], timing['max_time_ms']])
        axes[1, 2].set_ylabel('Time (ms)')
        axes[1, 2].set_title('Inference Timing')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, analysis: Dict[str, Any]) -> str:
        """Generate detailed complexity analysis report"""
        
        report = "=" * 60 + "\n"
        report += "MODEL COMPLEXITY ANALYSIS REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Parameters section
        param_info = analysis['parameters']
        report += "PARAMETER ANALYSIS:\n"
        report += "-" * 20 + "\n"
        report += f"Total Parameters: {param_info['counts']['total']:,}\n"
        report += f"Trainable Parameters: {param_info['counts']['trainable']:,}\n"
        report += f"Non-trainable Parameters: {param_info['counts']['non_trainable']:,}\n"
        report += f"Conv Layer Parameters: {param_info['counts']['conv_layers']:,}\n"
        report += f"Linear Layer Parameters: {param_info['counts']['linear_layers']:,}\n"
        report += f"BatchNorm Parameters: {param_info['counts']['bn_layers']:,}\n\n"
        
        # Memory section
        memory_info = param_info['memory_usage']
        report += "MEMORY ANALYSIS:\n"
        report += "-" * 16 + "\n"
        report += f"Parameter Memory: {memory_info['parameters_mb']:.2f} MB\n"
        report += f"Gradient Memory: {memory_info['gradients_mb']:.2f} MB\n"
        report += f"Activation Memory: {memory_info['activations_mb']:.2f} MB\n"
        report += f"Total Memory: {memory_info['total_mb']:.2f} MB\n\n"
        
        # FLOPs section
        flops_info = analysis['flops']
        report += "COMPUTATIONAL COMPLEXITY:\n"
        report += "-" * 26 + "\n"
        report += f"Total FLOPs: {flops_info['total']:,}\n"
        report += f"GFLOPs: {flops_info['gflops']:.2f}\n"
        report += f"MFLOPs: {flops_info['mflops']:.2f}\n\n"
        
        # Timing section
        timing_info = analysis['timing']
        report += "INFERENCE TIMING:\n"
        report += "-" * 17 + "\n"
        report += f"Average Time: {timing_info['avg_time_ms']:.2f} ms\n"
        report += f"Throughput: {timing_info['throughput_fps']:.2f} FPS\n"
        report += f"Min Time: {timing_info['min_time_ms']:.2f} ms\n"
        report += f"Max Time: {timing_info['max_time_ms']:.2f} ms\n\n"
        
        # Size section
        size_info = analysis['size']
        report += "MODEL SIZE:\n"
        report += "-" * 11 + "\n"
        report += f"FP32 Size: {size_info['fp32_size_mb']:.2f} MB\n"
        report += f"FP16 Size: {size_info['fp16_size_mb']:.2f} MB\n"
        report += f"INT8 Size: {size_info['int8_size_mb']:.2f} MB\n\n"
        
        # Efficiency section
        efficiency_info = analysis['efficiency']
        report += "EFFICIENCY METRICS:\n"
        report += "-" * 19 + "\n"
        report += f"FLOPs per Parameter: {efficiency_info['flops_per_param']:.2f}\n"
        report += f"GFLOPs per Second: {efficiency_info['gflops_per_second']:.2f}\n"
        report += f"Energy Efficiency: {efficiency_info['energy_efficiency']:.2e}\n\n"
        
        report += "=" * 60 + "\n"
        
        return report

if __name__ == "__main__":
    print("Model Complexity Analysis")
    print("=" * 30)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test models
    models = {
        'Standard CNN': ComplexityTestCNN(num_classes=10, width_multiplier=1.0).to(device),
        'Wide CNN': ComplexityTestCNN(num_classes=10, width_multiplier=2.0).to(device),
        'EfficientNet': EfficientNet(num_classes=10).to(device)
    }
    
    input_shape = (3, 32, 32)
    
    print("\n1. Individual Model Analysis")
    print("-" * 35)
    
    # Analyze each model individually
    for model_name, model in models.items():
        print(f"\nAnalyzing {model_name}...")
        
        analyzer = ModelComplexityAnalyzer(model, device)
        analysis = analyzer.comprehensive_analysis(input_shape)
        
        # Print summary
        print(f"Results for {model_name}:")
        print(f"  Parameters: {analysis['parameters']['counts']['total']:,}")
        print(f"  GFLOPs: {analysis['flops']['gflops']:.2f}")
        print(f"  Memory: {analysis['parameters']['memory_usage']['total_mb']:.2f} MB")
        print(f"  Inference: {analysis['timing']['avg_time_ms']:.2f} ms")
        print(f"  Throughput: {analysis['timing']['throughput_fps']:.2f} FPS")
        
        # Generate and save report
        report = analyzer.generate_report(analysis)
        with open(f'complexity_report_{model_name.replace(" ", "_").lower()}.txt', 'w') as f:
            f.write(report)
        
        # Visualize complexity
        analyzer.visualize_complexity(analysis, f'complexity_{model_name.replace(" ", "_").lower()}.png')
    
    print("\n2. Model Comparison")
    print("-" * 22)
    
    # Compare all models
    base_analyzer = ModelComplexityAnalyzer(models['Standard CNN'], device)
    comparison = base_analyzer.compare_models(models, input_shape)
    
    # Create comparison table
    print("\nModel Comparison Table:")
    print("-" * 80)
    print(f"{'Model':<15} {'Params':<12} {'GFLOPs':<10} {'Memory(MB)':<12} {'Time(ms)':<10} {'FPS':<10}")
    print("-" * 80)
    
    for model_name, analysis in comparison.items():
        params = analysis['parameters']['counts']['total']
        gflops = analysis['flops']['gflops']
        memory = analysis['parameters']['memory_usage']['total_mb']
        time_ms = analysis['timing']['avg_time_ms']
        fps = analysis['timing']['throughput_fps']
        
        print(f"{model_name:<15} {params:<12,} {gflops:<10.2f} {memory:<12.1f} {time_ms:<10.2f} {fps:<10.1f}")
    
    print("\n3. Efficiency Analysis")
    print("-" * 25)
    
    # Calculate efficiency ratios
    print("Efficiency Comparison (relative to Standard CNN):")
    print("-" * 50)
    
    baseline = comparison['Standard CNN']
    baseline_params = baseline['parameters']['counts']['total']
    baseline_flops = baseline['flops']['gflops']
    baseline_time = baseline['timing']['avg_time_ms']
    
    for model_name, analysis in comparison.items():
        if model_name == 'Standard CNN':
            continue
        
        params_ratio = analysis['parameters']['counts']['total'] / baseline_params
        flops_ratio = analysis['flops']['gflops'] / baseline_flops
        speed_ratio = baseline_time / analysis['timing']['avg_time_ms']  # Higher is better
        
        print(f"{model_name}:")
        print(f"  Parameter Efficiency: {1/params_ratio:.2f}x")
        print(f"  Computational Efficiency: {1/flops_ratio:.2f}x")
        print(f"  Speed Improvement: {speed_ratio:.2f}x")
        
        # Overall efficiency score (simplified)
        efficiency_score = (1/params_ratio + 1/flops_ratio + speed_ratio) / 3
        print(f"  Overall Efficiency Score: {efficiency_score:.2f}")
        print()
    
    print("\n4. Parameter Breakdown Analysis")
    print("-" * 35)
    
    # Detailed parameter analysis for one model
    test_model = models['Standard CNN']
    param_analyzer = ParameterAnalyzer(test_model)
    
    layer_details = param_analyzer.get_layer_parameters()
    
    print("Top 10 Layers by Parameter Count:")
    print("-" * 40)
    
    # Sort layers by parameter count
    sorted_layers = sorted(layer_details.items(), 
                          key=lambda x: x[1]['parameters'], 
                          reverse=True)
    
    print(f"{'Layer':<25} {'Type':<15} {'Parameters':<12} {'% of Total':<12}")
    print("-" * 70)
    
    total_params = sum(info['parameters'] for _, info in layer_details.items())
    
    for i, (layer_name, info) in enumerate(sorted_layers[:10]):
        param_count = info['parameters']
        param_percentage = (param_count / total_params) * 100
        layer_type = info['type']
        
        print(f"{layer_name[:23]:<25} {layer_type:<15} {param_count:<12,} {param_percentage:<12.1f}%")
    
    print("\n5. FLOPs Breakdown")
    print("-" * 20)
    
    # Detailed FLOPs analysis
    flops_calc = FLOPsCalculator()
    
    # Manual calculation for demonstration
    test_input = torch.randn(1, *input_shape).to(device)
    
    # Hook to capture layer-wise FLOPs
    layer_flops = {}
    
    def flops_hook(name):
        def hook(module, input, output):
            if isinstance(module, nn.Conv2d):
                flops = flops_calc.conv_flop_jit(input, output, module.kernel_size, module.groups)
                layer_flops[name] = flops
            elif isinstance(module, nn.Linear):
                flops = flops_calc.linear_flop_jit(input, output)
                layer_flops[name] = flops
        return hook
    
    # Register hooks
    hooks = []
    for name, module in test_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handle = module.register_forward_hook(flops_hook(name))
            hooks.append(handle)
    
    # Forward pass
    test_model.eval()
    with torch.no_grad():
        _ = test_model(test_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Display FLOPs breakdown
    print("FLOPs by Layer:")
    print("-" * 15)
    
    total_layer_flops = sum(layer_flops.values())
    sorted_flops = sorted(layer_flops.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'Layer':<25} {'FLOPs':<15} {'% of Total':<12}")
    print("-" * 55)
    
    for layer_name, flops in sorted_flops[:10]:
        flops_percentage = (flops / total_layer_flops) * 100 if total_layer_flops > 0 else 0
        print(f"{layer_name[:23]:<25} {flops:<15,} {flops_percentage:<12.1f}%")
    
    print("\n6. Optimization Recommendations")
    print("-" * 40)
    
    recommendations = []
    
    # Analyze Standard CNN for recommendations
    std_analysis = comparison['Standard CNN']
    
    # Check parameter efficiency
    conv_params = std_analysis['parameters']['counts']['conv_layers']
    linear_params = std_analysis['parameters']['counts']['linear_layers']
    total_params = std_analysis['parameters']['counts']['total']
    
    if linear_params / total_params > 0.7:
        recommendations.append("High ratio of linear layer parameters - consider using global average pooling")
    
    if conv_params / total_params < 0.3:
        recommendations.append("Low convolutional parameters - model might benefit from more conv layers")
    
    # Check computational efficiency
    flops_per_param = std_analysis['efficiency']['flops_per_param']
    if flops_per_param < 1:
        recommendations.append("Low FLOPs per parameter - consider depthwise separable convolutions")
    
    # Check memory usage
    activation_memory = std_analysis['parameters']['memory_usage']['activations_mb']
    param_memory = std_analysis['parameters']['memory_usage']['parameters_mb']
    
    if activation_memory > param_memory * 2:
        recommendations.append("High activation memory - consider reducing input resolution or using checkpointing")
    
    # Check inference speed
    fps = std_analysis['timing']['throughput_fps']
    if fps < 30:
        recommendations.append("Low inference speed - consider model pruning or quantization")
    
    print("Optimization Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    if not recommendations:
        print("No major optimization issues detected!")
    
    print("\nModel complexity analysis completed!")
    print("Generated files:")
    print("  - complexity_report_*.txt (detailed reports)")
    print("  - complexity_*.png (visualization plots)")