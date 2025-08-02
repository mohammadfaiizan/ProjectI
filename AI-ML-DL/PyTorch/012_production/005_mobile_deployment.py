import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional, Any

# Sample Models for Mobile Deployment
class MobileCompatibleCNN(nn.Module):
    """CNN designed for mobile deployment"""
    
    def __init__(self, num_classes: int = 10, width_multiplier: float = 1.0):
        super().__init__()
        
        # Efficient mobile-friendly architecture
        base_channels = int(32 * width_multiplier)
        
        self.features = nn.Sequential(
            # Stem
            nn.Conv2d(3, base_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU6(inplace=True),
            
            # Depthwise separable blocks
            self._make_depthwise_block(base_channels, base_channels * 2, 1),
            self._make_depthwise_block(base_channels * 2, base_channels * 4, 2),
            self._make_depthwise_block(base_channels * 4, base_channels * 8, 2),
            
            # Final layers
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        self.classifier = nn.Linear(base_channels * 8, num_classes)
    
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

class EfficientMobileNet(nn.Module):
    """Efficient MobileNet-style model for mobile deployment"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # Quantization stubs for mobile quantization
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        # Efficient backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            
            # Inverted residual blocks (simplified)
            self._make_inverted_block(32, 16, 1, 1),
            self._make_inverted_block(16, 24, 6, 2),
            self._make_inverted_block(24, 32, 6, 2),
            self._make_inverted_block(32, 96, 6, 1),
            
            nn.Conv2d(96, 320, 1, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(320, num_classes)
        )
    
    def _make_inverted_block(self, in_channels: int, out_channels: int, 
                           expand_ratio: int, stride: int):
        """Create inverted residual block"""
        hidden_dim = in_channels * expand_ratio
        use_residual = stride == 1 and in_channels == out_channels
        
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
        
        block = nn.Sequential(*layers)
        
        if use_residual:
            return ResidualBlock(block)
        else:
            return block
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.backbone(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

class ResidualBlock(nn.Module):
    """Residual connection wrapper"""
    
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
        self.add = torch.nn.quantized.FloatFunctional()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.add.add(x, self.block(x))

# Mobile Optimization Utilities
class MobileOptimizer:
    """Utilities for optimizing models for mobile deployment"""
    
    def __init__(self):
        self.optimizations_applied = []
    
    def optimize_for_mobile(self, model: nn.Module, 
                           example_input: torch.Tensor,
                           quantize: bool = True,
                           fuse_modules: bool = True) -> torch.jit.ScriptModule:
        """Comprehensive mobile optimization pipeline"""
        
        print("Starting mobile optimization pipeline...")
        
        # 1. Set model to evaluation mode
        model.eval()
        
        # 2. Fuse modules if requested
        if fuse_modules:
            model = self._fuse_modules(model)
            self.optimizations_applied.append("module_fusion")
        
        # 3. Convert to TorchScript
        try:
            scripted_model = torch.jit.trace(model, example_input)
            self.optimizations_applied.append("torchscript_tracing")
        except Exception as e:
            print(f"Tracing failed: {e}, trying script...")
            scripted_model = torch.jit.script(model)
            self.optimizations_applied.append("torchscript_scripting")
        
        # 4. Apply quantization if requested
        if quantize:
            scripted_model = self._apply_mobile_quantization(scripted_model, example_input)
            self.optimizations_applied.append("quantization")
        
        # 5. Optimize for mobile
        mobile_model = torch.utils.mobile_optimizer.optimize_for_mobile(
            scripted_model,
            optimization_blocklist={
                # Remove optimizations that might cause issues
                "remove_dropout",
                "fuse_add_relu"
            }
        )
        self.optimizations_applied.append("mobile_optimization")
        
        print(f"✓ Mobile optimization complete. Applied: {', '.join(self.optimizations_applied)}")
        return mobile_model
    
    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """Fuse conv-bn-relu modules for better performance"""
        
        # Find fuseable module sequences
        modules_to_fuse = []
        
        def find_fuseable_modules(module, prefix=""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                if isinstance(child, nn.Sequential):
                    # Look for conv-bn-relu patterns
                    for i in range(len(child) - 1):
                        if (isinstance(child[i], nn.Conv2d) and 
                            isinstance(child[i + 1], nn.BatchNorm2d)):
                            
                            # Check if next is ReLU
                            if (i + 2 < len(child) and 
                                isinstance(child[i + 2], (nn.ReLU, nn.ReLU6))):
                                modules_to_fuse.append([
                                    f"{full_name}.{i}",
                                    f"{full_name}.{i+1}",
                                    f"{full_name}.{i+2}"
                                ])
                            else:
                                modules_to_fuse.append([
                                    f"{full_name}.{i}",
                                    f"{full_name}.{i+1}"
                                ])
                else:
                    find_fuseable_modules(child, full_name)
        
        find_fuseable_modules(model)
        
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
    
    def _apply_mobile_quantization(self, scripted_model: torch.jit.ScriptModule,
                                  example_input: torch.Tensor) -> torch.jit.ScriptModule:
        """Apply quantization optimized for mobile"""
        
        try:
            # Set mobile-optimized quantization backend
            torch.backends.quantized.engine = 'qnnpack'
            
            # Create quantized model
            quantized_model = torch.quantization.quantize_dynamic(
                scripted_model, 
                {torch.nn.Linear}, 
                dtype=torch.qint8
            )
            
            print("✓ Mobile quantization applied")
            return quantized_model
        
        except Exception as e:
            print(f"Mobile quantization failed: {e}")
            return scripted_model
    
    def analyze_mobile_compatibility(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze model compatibility with mobile deployment"""
        
        analysis = {
            'compatible_ops': [],
            'incompatible_ops': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Mobile-friendly operations
        mobile_friendly_ops = {
            nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.ReLU6,
            nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
            nn.Flatten, nn.Dropout
        }
        
        # Potentially problematic operations
        problematic_ops = {
            nn.LSTM, nn.GRU, nn.RNN,  # RNNs can be problematic
            nn.MultiheadAttention,    # Complex attention
            nn.Transformer,           # Transformer blocks
        }
        
        # Analyze model operations
        for name, module in model.named_modules():
            module_type = type(module)
            
            if module_type in mobile_friendly_ops:
                analysis['compatible_ops'].append((name, module_type.__name__))
            elif module_type in problematic_ops:
                analysis['incompatible_ops'].append((name, module_type.__name__))
                analysis['warnings'].append(f"{name}: {module_type.__name__} may not be mobile-friendly")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Generate recommendations
        if total_params > 10e6:  # 10M parameters
            analysis['recommendations'].append("Consider model compression - large parameter count")
        
        if len(analysis['incompatible_ops']) > 0:
            analysis['recommendations'].append("Replace incompatible operations with mobile-friendly alternatives")
        
        if not any(isinstance(m, (nn.ReLU6,)) for m in model.modules()):
            analysis['recommendations'].append("Consider using ReLU6 instead of ReLU for better mobile performance")
        
        return analysis

# Mobile Model Manager
class MobileModelManager:
    """Manage mobile model deployment and versioning"""
    
    def __init__(self, model_dir: str = "mobile_models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
    
    def save_mobile_model(self, mobile_model: torch.jit.ScriptModule,
                         model_name: str,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save mobile model with metadata"""
        
        model_path = os.path.join(self.model_dir, f"{model_name}.ptl")
        
        # Save mobile model
        mobile_model.save(model_path)
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'model_name': model_name,
            'file_path': model_path,
            'file_size_mb': os.path.getsize(model_path) / (1024 * 1024),
            'created_at': torch.datetime.now().isoformat() if hasattr(torch, 'datetime') else 'unknown'
        })
        
        metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Mobile model saved: {model_path}")
        print(f"✓ Model size: {metadata['file_size_mb']:.2f} MB")
        
        return model_path
    
    def load_mobile_model(self, model_name: str) -> torch.jit.ScriptModule:
        """Load mobile model"""
        
        model_path = os.path.join(self.model_dir, f"{model_name}.ptl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Mobile model not found: {model_path}")
        
        mobile_model = torch.jit.load(model_path, map_location='cpu')
        print(f"✓ Mobile model loaded: {model_path}")
        
        return mobile_model
    
    def load_metadata(self, model_name: str) -> Dict[str, Any]:
        """Load model metadata"""
        
        metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        else:
            return {}
    
    def list_mobile_models(self) -> List[Dict[str, Any]]:
        """List all available mobile models"""
        
        models = []
        
        for file in os.listdir(self.model_dir):
            if file.endswith('.ptl'):
                model_name = file[:-4]  # Remove .ptl extension
                metadata = self.load_metadata(model_name)
                models.append(metadata)
        
        return models
    
    def benchmark_mobile_model(self, model_name: str,
                              input_shape: Tuple[int, ...],
                              num_runs: int = 100) -> Dict[str, float]:
        """Benchmark mobile model performance"""
        
        mobile_model = self.load_mobile_model(model_name)
        test_input = torch.randn(input_shape)
        
        # Warmup
        mobile_model.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = mobile_model(test_input)
        
        # Benchmark
        import time
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = mobile_model(test_input)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'throughput_fps': 1000 / np.mean(times)
        }

# Android/iOS Deployment Helpers
class MobileDeploymentHelper:
    """Helper utilities for mobile platform deployment"""
    
    @staticmethod
    def generate_android_integration_code(model_name: str) -> str:
        """Generate Android integration code"""
        
        android_code = f"""
// Android Java code for PyTorch Mobile integration
import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;

public class ModelInference {{
    private Module model;
    
    public ModelInference(String modelPath) {{
        // Load the mobile model
        model = LiteModuleLoader.load(modelPath);
    }}
    
    public float[] predict(float[] inputArray, int[] inputShape) {{
        // Create input tensor
        Tensor inputTensor = Tensor.fromBlob(inputArray, inputShape);
        
        // Run inference
        IValue output = model.forward(IValue.from(inputTensor));
        
        // Extract output
        Tensor outputTensor = output.toTensor();
        return outputTensor.getDataAsFloatArray();
    }}
}}

// Usage example:
// ModelInference inference = new ModelInference("assets/{model_name}.ptl");
// float[] result = inference.predict(inputData, inputShape);
"""
        return android_code
    
    @staticmethod
    def generate_ios_integration_code(model_name: str) -> str:
        """Generate iOS integration code"""
        
        ios_code = f"""
// iOS Swift code for PyTorch Mobile integration
import LibTorch

class ModelInference {{
    private var module: TorchModule
    
    init(modelPath: String) throws {{
        // Load the mobile model
        module = try TorchModule(contentsOf: URL(fileURLWithPath: modelPath))
    }}
    
    func predict(inputArray: [Float], inputShape: [Int]) throws -> [Float] {{
        // Create input tensor
        let inputTensor = try Tensor(data: inputArray, shape: inputShape)
        
        // Run inference
        let outputTensor = try module.forward([inputTensor])
        
        // Extract output
        guard let output = outputTensor[0].floatArray else {{
            throw InferenceError.invalidOutput
        }}
        
        return output
    }}
}}

// Usage example:
// let inference = try ModelInference(modelPath: "{model_name}.ptl")
// let result = try inference.predict(inputArray: inputData, inputShape: inputShape)
"""
        return ios_code
    
    @staticmethod
    def create_deployment_guide(model_name: str, 
                               input_shape: Tuple[int, ...],
                               output_shape: Tuple[int, ...]) -> str:
        """Create deployment guide"""
        
        guide = f"""
# Mobile Deployment Guide for {model_name}

## Model Information
- Model Name: {model_name}
- Input Shape: {input_shape}
- Output Shape: {output_shape}
- Format: PyTorch Mobile (.ptl)

## Android Deployment

### 1. Add PyTorch Mobile to your project
```gradle
implementation 'org.pytorch:pytorch_android_lite:1.12.2'
implementation 'org.pytorch:pytorch_android_torchvision_lite:1.12.2'
```

### 2. Place model file
- Copy `{model_name}.ptl` to `app/src/main/assets/`

### 3. Integration
- Use the provided Java/Kotlin code
- Handle input preprocessing (normalization, resizing)
- Process model outputs according to your use case

## iOS Deployment

### 1. Add PyTorch Mobile to your project
```ruby
# In Podfile
pod 'LibTorch-Lite'
```

### 2. Place model file
- Add `{model_name}.ptl` to your app bundle

### 3. Integration
- Use the provided Swift/Objective-C code
- Implement proper input preprocessing
- Handle model outputs appropriately

## Performance Considerations

1. **Model Size**: Keep model under 50MB for mobile apps
2. **Inference Speed**: Target <100ms on mid-range devices
3. **Memory Usage**: Monitor peak memory consumption
4. **Battery Impact**: Consider inference frequency

## Testing Checklist

- [ ] Test on various device types and OS versions
- [ ] Validate model accuracy vs server version
- [ ] Measure inference speed and memory usage
- [ ] Test edge cases and error handling
- [ ] Verify model loading and initialization

## Troubleshooting

1. **Model loading fails**: Check file path and permissions
2. **Slow inference**: Ensure model is optimized for mobile
3. **High memory usage**: Consider quantization or model compression
4. **Accuracy issues**: Validate preprocessing pipeline
"""
        return guide

if __name__ == "__main__":
    print("PyTorch Mobile Deployment")
    print("=" * 30)
    
    # Create sample models
    cnn_model = MobileCompatibleCNN(num_classes=10, width_multiplier=0.5)
    mobilenet_model = EfficientMobileNet(num_classes=10)
    
    example_input = torch.randn(1, 3, 224, 224)
    
    print("\n1. Mobile Compatibility Analysis")
    print("-" * 38)
    
    optimizer = MobileOptimizer()
    
    # Analyze mobile compatibility
    cnn_analysis = optimizer.analyze_mobile_compatibility(cnn_model)
    
    print("Mobile Compatibility Analysis:")
    print(f"Compatible operations: {len(cnn_analysis['compatible_ops'])}")
    print(f"Incompatible operations: {len(cnn_analysis['incompatible_ops'])}")
    
    if cnn_analysis['warnings']:
        print("Warnings:")
        for warning in cnn_analysis['warnings']:
            print(f"  - {warning}")
    
    if cnn_analysis['recommendations']:
        print("Recommendations:")
        for rec in cnn_analysis['recommendations']:
            print(f"  - {rec}")
    
    print("\n2. Mobile Optimization")
    print("-" * 25)
    
    # Optimize CNN for mobile
    mobile_cnn = optimizer.optimize_for_mobile(
        cnn_model, example_input, quantize=True, fuse_modules=True
    )
    
    # Optimize MobileNet for mobile
    optimizer_mobilenet = MobileOptimizer()
    mobile_mobilenet = optimizer_mobilenet.optimize_for_mobile(
        mobilenet_model, example_input, quantize=True, fuse_modules=True
    )
    
    print("\n3. Model Management")
    print("-" * 22)
    
    manager = MobileModelManager("demo_mobile_models")
    
    # Save mobile models
    cnn_metadata = {
        'description': 'Lightweight CNN for mobile classification',
        'input_shape': list(example_input.shape),
        'num_classes': 10,
        'optimizations': optimizer.optimizations_applied
    }
    
    mobilenet_metadata = {
        'description': 'Efficient MobileNet for mobile deployment',
        'input_shape': list(example_input.shape),
        'num_classes': 10,
        'optimizations': optimizer_mobilenet.optimizations_applied
    }
    
    cnn_path = manager.save_mobile_model(mobile_cnn, "mobile_cnn", cnn_metadata)
    mobilenet_path = manager.save_mobile_model(mobile_mobilenet, "mobile_mobilenet", mobilenet_metadata)
    
    # List available models
    available_models = manager.list_mobile_models()
    print(f"\nAvailable mobile models: {len(available_models)}")
    for model_info in available_models:
        print(f"  - {model_info.get('model_name', 'unknown')}: {model_info.get('file_size_mb', 0):.2f} MB")
    
    print("\n4. Performance Benchmarking")
    print("-" * 32)
    
    # Benchmark mobile models
    input_shape = (1, 3, 224, 224)
    
    models_to_benchmark = ['mobile_cnn', 'mobile_mobilenet']
    
    print("Mobile Model Performance:")
    print("-" * 28)
    print(f"{'Model':<15} {'Time (ms)':<12} {'FPS':<8} {'Size (MB)':<12}")
    print("-" * 50)
    
    for model_name in models_to_benchmark:
        try:
            benchmark_results = manager.benchmark_mobile_model(model_name, input_shape)
            metadata = manager.load_metadata(model_name)
            
            print(f"{model_name:<15} {benchmark_results['mean_time_ms']:<12.2f} "
                  f"{benchmark_results['throughput_fps']:<8.1f} "
                  f"{metadata.get('file_size_mb', 0):<12.2f}")
        
        except Exception as e:
            print(f"{model_name:<15} Error: {str(e)[:20]}")
    
    print("\n5. Mobile Platform Integration")
    print("-" * 36)
    
    deployment_helper = MobileDeploymentHelper()
    
    # Generate platform-specific code
    android_code = deployment_helper.generate_android_integration_code("mobile_cnn")
    ios_code = deployment_helper.generate_ios_integration_code("mobile_cnn")
    
    # Save integration code
    with open("demo_mobile_models/android_integration.java", "w") as f:
        f.write(android_code)
    
    with open("demo_mobile_models/ios_integration.swift", "w") as f:
        f.write(ios_code)
    
    print("✓ Android integration code generated")
    print("✓ iOS integration code generated")
    
    # Generate deployment guide
    deployment_guide = deployment_helper.create_deployment_guide(
        "mobile_cnn", (1, 3, 224, 224), (1, 10)
    )
    
    with open("demo_mobile_models/deployment_guide.md", "w") as f:
        f.write(deployment_guide)
    
    print("✓ Deployment guide generated")
    
    print("\n6. Model Validation")
    print("-" * 22)
    
    # Validate mobile models work correctly
    test_input = torch.randn(1, 3, 224, 224)
    
    for model_name in models_to_benchmark:
        try:
            mobile_model = manager.load_mobile_model(model_name)
            
            with torch.no_grad():
                output = mobile_model(test_input)
                print(f"✓ {model_name}: Output shape {tuple(output.shape)}")
        
        except Exception as e:
            print(f"✗ {model_name}: Validation failed - {e}")
    
    print("\n7. Mobile Deployment Best Practices")
    print("-" * 42)
    
    best_practices = [
        "Keep model size under 50MB for mobile apps",
        "Target inference time <100ms on mid-range devices",
        "Use quantization to reduce model size and improve speed",
        "Apply module fusion before mobile optimization",
        "Test on actual mobile devices, not just emulators",
        "Monitor memory usage during inference",
        "Implement proper error handling for model loading",
        "Use batch size of 1 for mobile inference",
        "Consider preprocessing on device vs server",
        "Profile app performance with model integrated",
        "Use ReLU6 activation for better mobile performance",
        "Avoid operations not supported on mobile backends"
    ]
    
    print("Mobile Deployment Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\n8. Platform-Specific Considerations")
    print("-" * 40)
    
    considerations = {
        "Android": [
            "Use PyTorch Android Lite runtime",
            "Target Android API level 21+",
            "Consider APK size impact",
            "Test on different Android versions",
            "Handle device capabilities gracefully"
        ],
        "iOS": [
            "Use PyTorch iOS framework",
            "Target iOS 12.0+",
            "Consider app store size limits",
            "Test on different iOS devices",
            "Handle memory warnings properly"
        ]
    }
    
    for platform, items in considerations.items():
        print(f"\n{platform} Considerations:")
        for item in items:
            print(f"  - {item}")
    
    print("\nPyTorch Mobile deployment demonstration completed!")
    print("Generated files:")
    print("  - demo_mobile_models/ (mobile models and metadata)")
    print("  - android_integration.java (Android code)")
    print("  - ios_integration.swift (iOS code)")
    print("  - deployment_guide.md (deployment instructions)")