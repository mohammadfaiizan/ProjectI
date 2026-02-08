import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Any

# Note: ONNX operations require the onnx package
# Install with: pip install onnx onnxruntime onnxruntime-tools

try:
    import onnx
    import onnxruntime as ort
    from onnx import helper, checker, shape_inference
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX packages not available. Install with:")
    print("pip install onnx onnxruntime onnxruntime-tools")

# Sample Models for ONNX Export
class ONNXCompatibleCNN(nn.Module):
    """CNN designed to be ONNX compatible"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

class ONNXResNetBlock(nn.Module):
    """ONNX-compatible ResNet block"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)  # Avoid inplace operations for ONNX
        out = F.relu(out)
        return out

class ONNXResNet(nn.Module):
    """ONNX-compatible ResNet"""
    
    def __init__(self, num_classes: int = 10, layers: List[int] = [2, 2, 2, 2]):
        super().__init__()
        
        self.in_channels = 64
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        layers.append(ONNXResNetBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(ONNXResNetBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ONNX Export Utilities
class ONNXExporter:
    """Utilities for exporting PyTorch models to ONNX"""
    
    def __init__(self, export_dir: str = "onnx_models"):
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)
    
    def export_model(self, model: nn.Module, 
                    example_input: torch.Tensor,
                    filename: str,
                    input_names: List[str] = ['input'],
                    output_names: List[str] = ['output'],
                    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
                    opset_version: int = 11) -> str:
        """Export PyTorch model to ONNX format"""
        
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX packages not available")
        
        filepath = os.path.join(self.export_dir, filename)
        
        # Set model to evaluation mode
        model.eval()
        
        # Export the model
        with torch.no_grad():
            torch.onnx.export(
                model,
                example_input,
                filepath,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False
            )
        
        print(f"✓ Model exported to: {filepath}")
        return filepath
    
    def export_with_dynamic_shapes(self, model: nn.Module,
                                  example_input: torch.Tensor,
                                  filename: str) -> str:
        """Export model with dynamic batch size and input dimensions"""
        
        dynamic_axes = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size'}
        }
        
        return self.export_model(
            model, example_input, filename,
            dynamic_axes=dynamic_axes
        )
    
    def export_traced_model(self, model: nn.Module,
                           example_input: torch.Tensor,
                           filename: str) -> str:
        """Export traced model for better ONNX compatibility"""
        
        model.eval()
        
        # Trace the model first
        traced_model = torch.jit.trace(model, example_input)
        
        # Export traced model
        return self.export_model(traced_model, example_input, filename)
    
    def validate_export(self, model: nn.Module,
                       example_input: torch.Tensor,
                       onnx_path: str,
                       tolerance: float = 1e-5) -> bool:
        """Validate ONNX export by comparing outputs"""
        
        if not ONNX_AVAILABLE:
            return False
        
        # Get PyTorch output
        model.eval()
        with torch.no_grad():
            pytorch_output = model(example_input).numpy()
        
        # Get ONNX output
        ort_session = ort.InferenceSession(onnx_path)
        input_name = ort_session.get_inputs()[0].name
        onnx_output = ort_session.run(None, {input_name: example_input.numpy()})[0]
        
        # Compare outputs
        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        
        is_valid = max_diff < tolerance
        print(f"Validation: {'✓ PASS' if is_valid else '✗ FAIL'} (max diff: {max_diff:.2e})")
        
        return is_valid

# ONNX Model Analysis
class ONNXAnalyzer:
    """Analyze ONNX models"""
    
    def __init__(self):
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX packages not available")
    
    def analyze_model(self, onnx_path: str) -> Dict[str, Any]:
        """Analyze ONNX model structure and properties"""
        
        # Load ONNX model
        model = onnx.load(onnx_path)
        
        # Check model validity
        checker.check_model(model)
        
        # Infer shapes
        model = shape_inference.infer_shapes(model)
        
        analysis = {
            'model_info': {
                'ir_version': model.ir_version,
                'opset_version': model.opset_import[0].version if model.opset_import else None,
                'producer_name': model.producer_name,
                'producer_version': model.producer_version
            },
            'graph_info': {
                'num_nodes': len(model.graph.node),
                'num_inputs': len(model.graph.input),
                'num_outputs': len(model.graph.output),
                'num_initializers': len(model.graph.initializer)
            },
            'inputs': [],
            'outputs': [],
            'nodes': []
        }
        
        # Analyze inputs
        for input_tensor in model.graph.input:
            input_info = {
                'name': input_tensor.name,
                'type': input_tensor.type.tensor_type.elem_type,
                'shape': [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
            }
            analysis['inputs'].append(input_info)
        
        # Analyze outputs
        for output_tensor in model.graph.output:
            output_info = {
                'name': output_tensor.name,
                'type': output_tensor.type.tensor_type.elem_type,
                'shape': [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
            }
            analysis['outputs'].append(output_info)
        
        # Analyze nodes (operators)
        op_counts = {}
        for node in model.graph.node:
            op_type = node.op_type
            op_counts[op_type] = op_counts.get(op_type, 0) + 1
            
            analysis['nodes'].append({
                'name': node.name,
                'op_type': op_type,
                'inputs': list(node.input),
                'outputs': list(node.output)
            })
        
        analysis['operator_counts'] = op_counts
        
        return analysis
    
    def print_model_info(self, analysis: Dict[str, Any]):
        """Print model analysis in readable format"""
        
        print("ONNX Model Analysis")
        print("=" * 25)
        
        # Model info
        print(f"IR Version: {analysis['model_info']['ir_version']}")
        print(f"Opset Version: {analysis['model_info']['opset_version']}")
        print(f"Producer: {analysis['model_info']['producer_name']} v{analysis['model_info']['producer_version']}")
        
        # Graph info
        print(f"\nGraph Structure:")
        print(f"  Nodes: {analysis['graph_info']['num_nodes']}")
        print(f"  Inputs: {analysis['graph_info']['num_inputs']}")
        print(f"  Outputs: {analysis['graph_info']['num_outputs']}")
        print(f"  Initializers: {analysis['graph_info']['num_initializers']}")
        
        # Inputs
        print(f"\nInputs:")
        for i, input_info in enumerate(analysis['inputs']):
            print(f"  {i+1}. {input_info['name']}: {input_info['shape']} (type: {input_info['type']})")
        
        # Outputs
        print(f"\nOutputs:")
        for i, output_info in enumerate(analysis['outputs']):
            print(f"  {i+1}. {output_info['name']}: {output_info['shape']} (type: {output_info['type']})")
        
        # Operator counts
        print(f"\nOperator Counts:")
        for op_type, count in sorted(analysis['operator_counts'].items()):
            print(f"  {op_type}: {count}")

# ONNX Runtime Utilities
class ONNXInference:
    """Utilities for ONNX model inference"""
    
    def __init__(self, onnx_path: str, providers: Optional[List[str]] = None):
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX packages not available")
        
        if providers is None:
            providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        print(f"✓ ONNX model loaded with providers: {providers}")
    
    def run_inference(self, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Run inference on ONNX model"""
        
        return self.session.run(self.output_names, inputs)
    
    def run_inference_tensor(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Run inference with PyTorch tensor input"""
        
        input_dict = {self.input_names[0]: input_tensor.numpy()}
        outputs = self.run_inference(input_dict)
        return outputs[0]
    
    def benchmark_inference(self, input_tensor: torch.Tensor, 
                          num_runs: int = 100) -> float:
        """Benchmark ONNX model inference speed"""
        
        import time
        
        input_dict = {self.input_names[0]: input_tensor.numpy()}
        
        # Warmup
        for _ in range(10):
            self.session.run(self.output_names, input_dict)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            self.session.run(self.output_names, input_dict)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return np.mean(times)

# Model Optimization
class ONNXOptimizer:
    """Optimize ONNX models for deployment"""
    
    def __init__(self):
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX packages not available")
    
    def optimize_model(self, onnx_path: str, 
                      optimized_path: str,
                      optimization_level: str = 'basic') -> str:
        """Optimize ONNX model"""
        
        try:
            from onnxruntime.tools import optimizer
            
            # Load model
            model = onnx.load(onnx_path)
            
            # Apply optimizations
            if optimization_level == 'basic':
                optimized_model = optimizer.optimize_model(
                    model,
                    optimization_options=['eliminate_nop', 'eliminate_identity', 'eliminate_dropout']
                )
            elif optimization_level == 'extended':
                optimized_model = optimizer.optimize_model(
                    model,
                    optimization_options=[
                        'eliminate_nop', 'eliminate_identity', 'eliminate_dropout',
                        'fuse_consecutive_transposes', 'fuse_add_bias_into_conv',
                        'fuse_bn_into_conv', 'fuse_relu_into_conv'
                    ]
                )
            else:
                optimized_model = model
            
            # Save optimized model
            onnx.save(optimized_model, optimized_path)
            print(f"✓ Optimized model saved to: {optimized_path}")
            
            return optimized_path
        
        except ImportError:
            print("ONNX optimization tools not available")
            return onnx_path

if __name__ == "__main__":
    print("ONNX Export and Import")
    print("=" * 25)
    
    if not ONNX_AVAILABLE:
        print("ONNX packages not available. Please install:")
        print("pip install onnx onnxruntime onnxruntime-tools")
        exit(1)
    
    # Create sample models
    cnn_model = ONNXCompatibleCNN(num_classes=10)
    resnet_model = ONNXResNet(num_classes=10, layers=[1, 1, 1, 1])  # Smaller for demo
    
    # Example inputs
    example_input = torch.randn(1, 3, 32, 32)
    batch_input = torch.randn(4, 3, 32, 32)
    
    print("\n1. Basic ONNX Export")
    print("-" * 23)
    
    exporter = ONNXExporter("demo_onnx_models")
    
    # Export CNN model
    cnn_path = exporter.export_model(cnn_model, example_input, "cnn_model.onnx")
    
    # Validate export
    is_valid = exporter.validate_export(cnn_model, example_input, cnn_path)
    
    print("\n2. Dynamic Shape Export")
    print("-" * 27)
    
    # Export with dynamic shapes
    dynamic_path = exporter.export_with_dynamic_shapes(
        cnn_model, example_input, "cnn_dynamic.onnx"
    )
    
    print("\n3. Traced Model Export")
    print("-" * 26)
    
    # Export traced model
    traced_path = exporter.export_traced_model(
        resnet_model, example_input, "resnet_traced.onnx"
    )
    
    print("\n4. Model Analysis")
    print("-" * 20)
    
    analyzer = ONNXAnalyzer()
    
    # Analyze CNN model
    cnn_analysis = analyzer.analyze_model(cnn_path)
    analyzer.print_model_info(cnn_analysis)
    
    print("\n5. ONNX Runtime Inference")
    print("-" * 31)
    
    # Test ONNX inference
    onnx_inference = ONNXInference(cnn_path)
    
    # Run inference
    onnx_output = onnx_inference.run_inference_tensor(example_input)
    
    # Compare with PyTorch
    cnn_model.eval()
    with torch.no_grad():
        pytorch_output = cnn_model(example_input).numpy()
    
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    print(f"Max difference PyTorch vs ONNX: {max_diff:.2e}")
    
    # Benchmark inference speed
    onnx_time = onnx_inference.benchmark_inference(example_input)
    print(f"ONNX inference time: {onnx_time:.2f} ms")
    
    print("\n6. Model Optimization")
    print("-" * 24)
    
    optimizer = ONNXOptimizer()
    
    # Optimize model
    optimized_path = optimizer.optimize_model(
        cnn_path, 
        "demo_onnx_models/cnn_optimized.onnx",
        optimization_level='extended'
    )
    
    # Compare original vs optimized
    try:
        optimized_inference = ONNXInference(optimized_path)
        optimized_time = optimized_inference.benchmark_inference(example_input)
        
        speedup = onnx_time / optimized_time
        print(f"Original: {onnx_time:.2f} ms")
        print(f"Optimized: {optimized_time:.2f} ms")
        print(f"Speedup: {speedup:.2f}x")
    
    except Exception as e:
        print(f"Optimization comparison failed: {e}")
    
    print("\n7. Different Input Shapes")
    print("-" * 29)
    
    # Test dynamic shapes
    try:
        dynamic_inference = ONNXInference(dynamic_path)
        
        test_shapes = [(1, 3, 32, 32), (2, 3, 64, 64), (1, 3, 128, 128)]
        
        for shape in test_shapes:
            test_input = torch.randn(shape)
            output = dynamic_inference.run_inference_tensor(test_input)
            print(f"Input {shape} -> Output {output.shape}")
    
    except Exception as e:
        print(f"Dynamic shape testing failed: {e}")
    
    print("\n8. Deployment Considerations")
    print("-" * 33)
    
    deployment_tips = [
        "Use ONNX Runtime for cross-platform deployment",
        "Optimize models before deployment for better performance",
        "Test with representative input shapes and data",
        "Consider quantization for edge deployment",
        "Use appropriate execution providers (CPU, CUDA, TensorRT)",
        "Validate model outputs after export",
        "Handle dynamic shapes for flexible input sizes",
        "Monitor inference latency in production",
        "Use batching for higher throughput",
        "Consider model versioning for updates"
    ]
    
    print("ONNX Deployment Best Practices:")
    for i, tip in enumerate(deployment_tips, 1):
        print(f"{i:2d}. {tip}")
    
    print("\n9. Supported Operations")
    print("-" * 27)
    
    # Show operator support
    supported_ops = [
        "Conv2d, Linear, BatchNorm2d, ReLU, MaxPool2d",
        "AdaptiveAvgPool2d, Dropout, Flatten",
        "Add, Mul, Concat, Reshape, Transpose",
        "Softmax, Sigmoid, Tanh, GELU",
        "LSTM, GRU (with limitations)",
        "Attention mechanisms (newer opsets)"
    ]
    
    print("Common Supported Operations:")
    for i, ops in enumerate(supported_ops, 1):
        print(f"{i}. {ops}")
    
    print("\n10. Common Issues and Solutions")
    print("-" * 37)
    
    issues = [
        "Unsupported operations: Use alternative implementations",
        "Dynamic shapes: Export with dynamic_axes parameter",
        "Control flow: May require scripting before export",
        "Custom operators: Implement custom ONNX operators",
        "Version compatibility: Match PyTorch and ONNX versions",
        "Inference differences: Validate outputs after export"
    ]
    
    print("Common Issues:")
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
    
    print("\nONNX export/import demonstration completed!")
    print("Generated files in demo_onnx_models/:")
    print("  - cnn_model.onnx (basic export)")
    print("  - cnn_dynamic.onnx (dynamic shapes)")
    print("  - resnet_traced.onnx (traced model)")
    print("  - cnn_optimized.onnx (optimized model)")