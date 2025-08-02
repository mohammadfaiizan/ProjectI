import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import jit
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any

# Sample Models for TorchScript Demo
class ScriptableCNN(nn.Module):
    """CNN designed to be compatible with TorchScript"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

class ConditionalModel(nn.Module):
    """Model with conditional logic for TorchScript"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        self.classifier = nn.Linear(64, num_classes)
        self.aux_classifier = nn.Linear(64, num_classes)
    
    def forward(self, x: torch.Tensor, use_aux: bool = False) -> torch.Tensor:
        features = self.backbone(x)
        
        if use_aux:
            return self.aux_classifier(features)
        else:
            return self.classifier(features)

class ControlFlowModel(nn.Module):
    """Model demonstrating control flow in TorchScript"""
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # TorchScript-compatible loops
        outputs = []
        for i in range(batch_size):
            single_input = x[i:i+1]  # Keep batch dimension
            features = self.conv(single_input)
            pooled = self.pool(features)
            flattened = torch.flatten(pooled, 1)
            output = self.fc(flattened)
            outputs.append(output)
        
        return torch.cat(outputs, dim=0)

# TorchScript Compilation Utilities
class TorchScriptCompiler:
    """Utilities for compiling models to TorchScript"""
    
    def __init__(self):
        self.compiled_models = {}
    
    def trace_model(self, model: nn.Module, example_inputs: torch.Tensor,
                   strict: bool = True) -> torch.jit.ScriptModule:
        """Compile model using tracing"""
        
        model.eval()
        
        try:
            traced_model = torch.jit.trace(model, example_inputs, strict=strict)
            print("✓ Model tracing successful")
            return traced_model
        except Exception as e:
            print(f"✗ Model tracing failed: {e}")
            raise
    
    def script_model(self, model: nn.Module) -> torch.jit.ScriptModule:
        """Compile model using scripting"""
        
        try:
            scripted_model = torch.jit.script(model)
            print("✓ Model scripting successful")
            return scripted_model
        except Exception as e:
            print(f"✗ Model scripting failed: {e}")
            raise
    
    def trace_with_annotations(self, model: nn.Module, 
                              example_inputs: torch.Tensor) -> torch.jit.ScriptModule:
        """Trace model with type annotations"""
        
        # Add type annotations to model if not present
        self._add_type_annotations(model)
        
        return self.trace_model(model, example_inputs)
    
    def _add_type_annotations(self, model: nn.Module):
        """Add type annotations to model methods"""
        
        # This is a simplified example - real implementation would be more complex
        if not hasattr(model.forward, '__annotations__'):
            # Add basic annotations
            pass
    
    def hybrid_compilation(self, model: nn.Module, 
                          example_inputs: torch.Tensor) -> torch.jit.ScriptModule:
        """Try tracing first, fall back to scripting"""
        
        try:
            return self.trace_model(model, example_inputs)
        except Exception as trace_error:
            print(f"Tracing failed: {trace_error}")
            print("Falling back to scripting...")
            return self.script_model(model)
    
    def optimize_script(self, scripted_model: torch.jit.ScriptModule,
                       optimization_level: str = "default") -> torch.jit.ScriptModule:
        """Apply optimizations to scripted model"""
        
        if optimization_level == "aggressive":
            # Freeze the model
            scripted_model = torch.jit.freeze(scripted_model)
            
            # Optimize for inference
            scripted_model = torch.jit.optimize_for_inference(scripted_model)
        
        elif optimization_level == "mobile":
            # Optimize for mobile
            scripted_model = torch.utils.mobile_optimizer.optimize_for_mobile(scripted_model)
        
        print(f"✓ Model optimized with {optimization_level} settings")
        return scripted_model

# Custom TorchScript Operations
@torch.jit.script
def custom_activation(x: torch.Tensor, alpha: float = 0.2) -> torch.Tensor:
    """Custom activation function compatible with TorchScript"""
    return torch.where(x > 0, x, alpha * x)

@torch.jit.script
def custom_pooling(x: torch.Tensor, kernel_size: int = 2) -> torch.Tensor:
    """Custom pooling operation"""
    return F.max_pool2d(x, kernel_size=kernel_size, stride=kernel_size)

class CustomModule(torch.nn.Module):
    """Module with custom TorchScript operations"""
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.alpha = 0.2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = custom_activation(x, self.alpha)
        x = custom_pooling(x, 2)
        return x

# Performance Comparison
class PerformanceComparer:
    """Compare performance between eager and scripted models"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
    
    def benchmark_model(self, model: nn.Module, input_tensor: torch.Tensor,
                       num_runs: int = 100, warmup_runs: int = 10) -> Dict[str, float]:
        """Benchmark model performance"""
        
        model = model.to(self.device)
        input_tensor = input_tensor.to(self.device)
        model.eval()
        
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
        
        times = np.array(times)
        
        return {
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'throughput_fps': 1000 / np.mean(times)
        }
    
    def compare_eager_vs_script(self, eager_model: nn.Module,
                               input_tensor: torch.Tensor) -> Dict[str, Any]:
        """Compare eager mode vs TorchScript performance"""
        
        # Benchmark eager model
        eager_results = self.benchmark_model(eager_model, input_tensor)
        
        # Create TorchScript version
        compiler = TorchScriptCompiler()
        try:
            scripted_model = compiler.trace_model(eager_model, input_tensor)
            script_results = self.benchmark_model(scripted_model, input_tensor)
            
            # Calculate speedup
            speedup = eager_results['mean_time_ms'] / script_results['mean_time_ms']
            
            return {
                'eager': eager_results,
                'scripted': script_results,
                'speedup': speedup,
                'success': True
            }
        
        except Exception as e:
            return {
                'eager': eager_results,
                'scripted': None,
                'speedup': None,
                'success': False,
                'error': str(e)
            }

# Model Serialization and Deployment
class ScriptModelManager:
    """Manage TorchScript model serialization and deployment"""
    
    def __init__(self, save_dir: str = "scripted_models"):
        self.save_dir = save_dir
        import os
        os.makedirs(save_dir, exist_ok=True)
    
    def save_scripted_model(self, scripted_model: torch.jit.ScriptModule,
                           filename: str, metadata: Optional[Dict] = None):
        """Save TorchScript model to disk"""
        
        filepath = f"{self.save_dir}/{filename}"
        
        # Save the model
        scripted_model.save(filepath)
        
        # Save metadata separately
        if metadata:
            import json
            metadata_path = f"{filepath}.metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"Scripted model saved to: {filepath}")
    
    def load_scripted_model(self, filename: str) -> torch.jit.ScriptModule:
        """Load TorchScript model from disk"""
        
        filepath = f"{self.save_dir}/{filename}"
        
        try:
            model = torch.jit.load(filepath, map_location='cpu')
            print(f"Scripted model loaded from: {filepath}")
            return model
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
    
    def load_model_metadata(self, filename: str) -> Optional[Dict]:
        """Load model metadata"""
        
        import json
        metadata_path = f"{self.save_dir}/{filename}.metadata.json"
        
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
    
    def convert_and_save(self, model: nn.Module, filename: str,
                        example_input: torch.Tensor,
                        optimization_level: str = "default") -> bool:
        """Convert model to TorchScript and save"""
        
        try:
            compiler = TorchScriptCompiler()
            
            # Try hybrid compilation
            scripted_model = compiler.hybrid_compilation(model, example_input)
            
            # Apply optimizations
            scripted_model = compiler.optimize_script(scripted_model, optimization_level)
            
            # Create metadata
            metadata = {
                'model_class': model.__class__.__name__,
                'input_shape': list(example_input.shape),
                'optimization_level': optimization_level,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save model and metadata
            self.save_scripted_model(scripted_model, filename, metadata)
            
            return True
        
        except Exception as e:
            print(f"Conversion failed: {e}")
            return False

# Advanced TorchScript Features
class AdvancedScriptFeatures:
    """Demonstrate advanced TorchScript features"""
    
    @staticmethod
    def create_quantized_script_model(model: nn.Module, 
                                    example_input: torch.Tensor) -> torch.jit.ScriptModule:
        """Create quantized TorchScript model"""
        
        # Prepare model for quantization
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare for static quantization
        prepared_model = torch.quantization.prepare(model)
        
        # Calibrate with example input
        with torch.no_grad():
            prepared_model(example_input)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model)
        
        # Convert to TorchScript
        scripted_quantized = torch.jit.script(quantized_model)
        
        return scripted_quantized
    
    @staticmethod
    def create_custom_op_model():
        """Create model with custom operations"""
        
        class CustomOpModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
                self.custom_module = CustomModule()
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.conv(x)
                x = self.custom_module(x)
                return x
        
        return CustomOpModel()
    
    @staticmethod
    def create_dynamic_shape_model():
        """Create model that handles dynamic input shapes"""
        
        class DynamicShapeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
                self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(64, 10)
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                batch_size = x.size(0)
                x = self.conv(x)
                x = self.adaptive_pool(x)
                x = x.view(batch_size, -1)
                x = self.fc(x)
                return x
        
        return DynamicShapeModel()

if __name__ == "__main__":
    print("TorchScript Model Compilation")
    print("=" * 35)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create example models
    simple_model = ScriptableCNN(num_classes=10)
    conditional_model = ConditionalModel(num_classes=10)
    control_flow_model = ControlFlowModel()
    
    # Example input
    example_input = torch.randn(4, 3, 32, 32)
    
    print("\n1. Basic Model Tracing")
    print("-" * 25)
    
    compiler = TorchScriptCompiler()
    
    # Trace simple model
    traced_simple = compiler.trace_model(simple_model, example_input)
    
    # Test traced model
    with torch.no_grad():
        original_output = simple_model(example_input)
        traced_output = traced_simple(example_input)
        
        max_diff = torch.max(torch.abs(original_output - traced_output)).item()
        print(f"Max difference between original and traced: {max_diff:.2e}")
    
    print("\n2. Model Scripting")
    print("-" * 20)
    
    # Script models
    scripted_simple = compiler.script_model(simple_model)
    scripted_conditional = compiler.script_model(conditional_model)
    
    # Test scripted models
    with torch.no_grad():
        scripted_output = scripted_simple(example_input)
        conditional_output = scripted_conditional(example_input, False)
        
        print(f"Scripted simple output shape: {scripted_output.shape}")
        print(f"Scripted conditional output shape: {conditional_output.shape}")
    
    print("\n3. Hybrid Compilation")
    print("-" * 25)
    
    # Try hybrid compilation on control flow model
    try:
        hybrid_model = compiler.hybrid_compilation(control_flow_model, example_input)
        print("✓ Hybrid compilation successful")
        
        with torch.no_grad():
            hybrid_output = hybrid_model(example_input)
            print(f"Hybrid model output shape: {hybrid_output.shape}")
    
    except Exception as e:
        print(f"✗ Hybrid compilation failed: {e}")
    
    print("\n4. Performance Comparison")
    print("-" * 30)
    
    comparer = PerformanceComparer(device='cpu')  # Use CPU for consistent comparison
    
    # Compare eager vs scripted performance
    comparison = comparer.compare_eager_vs_script(simple_model, example_input)
    
    if comparison['success']:
        print("Performance Results:")
        print(f"  Eager mode: {comparison['eager']['mean_time_ms']:.2f} ± {comparison['eager']['std_time_ms']:.2f} ms")
        print(f"  Scripted:   {comparison['scripted']['mean_time_ms']:.2f} ± {comparison['scripted']['std_time_ms']:.2f} ms")
        print(f"  Speedup:    {comparison['speedup']:.2f}x")
    else:
        print(f"Performance comparison failed: {comparison['error']}")
    
    print("\n5. Model Optimization")
    print("-" * 24)
    
    # Apply different optimization levels
    optimizations = ['default', 'aggressive']
    
    for opt_level in optimizations:
        try:
            optimized_model = compiler.optimize_script(traced_simple, opt_level)
            
            # Test optimized model
            with torch.no_grad():
                opt_output = optimized_model(example_input)
                print(f"✓ {opt_level.capitalize()} optimization successful")
        
        except Exception as e:
            print(f"✗ {opt_level.capitalize()} optimization failed: {e}")
    
    print("\n6. Model Serialization")
    print("-" * 25)
    
    manager = ScriptModelManager("demo_scripted_models")
    
    # Convert and save models
    models_to_save = [
        (simple_model, "simple_cnn.pt"),
        (conditional_model, "conditional_model.pt")
    ]
    
    for model, filename in models_to_save:
        success = manager.convert_and_save(model, filename, example_input)
        if success:
            print(f"✓ {filename} saved successfully")
    
    # Load and test saved models
    for _, filename in models_to_save:
        try:
            loaded_model = manager.load_scripted_model(filename)
            metadata = manager.load_model_metadata(filename)
            
            print(f"✓ {filename} loaded successfully")
            if metadata:
                print(f"  Metadata: {metadata['model_class']}, {metadata['timestamp']}")
        
        except Exception as e:
            print(f"✗ Failed to load {filename}: {e}")
    
    print("\n7. Advanced Features")
    print("-" * 24)
    
    advanced = AdvancedScriptFeatures()
    
    # Custom operations model
    custom_op_model = advanced.create_custom_op_model()
    try:
        scripted_custom = compiler.script_model(custom_op_model)
        print("✓ Custom operations model scripted successfully")
    except Exception as e:
        print(f"✗ Custom operations scripting failed: {e}")
    
    # Dynamic shape model
    dynamic_model = advanced.create_dynamic_shape_model()
    try:
        scripted_dynamic = compiler.script_model(dynamic_model)
        
        # Test with different input sizes
        test_sizes = [(1, 3, 32, 32), (2, 3, 64, 64), (3, 3, 128, 128)]
        
        for size in test_sizes:
            test_input = torch.randn(size)
            with torch.no_grad():
                output = scripted_dynamic(test_input)
                print(f"  Input {size} -> Output {tuple(output.shape)}")
        
        print("✓ Dynamic shape model works with variable inputs")
    
    except Exception as e:
        print(f"✗ Dynamic shape scripting failed: {e}")
    
    print("\n8. TorchScript Best Practices")
    print("-" * 35)
    
    best_practices = [
        "Use type annotations for better scripting compatibility",
        "Avoid Python built-ins that aren't supported in TorchScript",
        "Use torch.jit.trace for models without control flow",
        "Use torch.jit.script for models with control flow",
        "Test scripted models thoroughly before deployment",
        "Apply appropriate optimizations for target platform",
        "Handle dynamic input shapes with adaptive layers",
        "Use @torch.jit.script decorator for custom functions",
        "Profile performance gains from TorchScript conversion",
        "Keep metadata with saved TorchScript models"
    ]
    
    print("TorchScript Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\n9. Debugging TorchScript Issues")
    print("-" * 35)
    
    debugging_tips = [
        "Use torch.jit.trace with strict=False for debugging",
        "Check model.graph for understanding traced operations",
        "Use model.code to see generated TorchScript code",
        "Test with different input shapes and data types",
        "Use torch.jit.script_if_tracing for conditional behavior",
        "Avoid mutable Python containers in forward method",
        "Use torch.jit.annotate for complex type hints",
        "Test both eager and script modes for consistency"
    ]
    
    print("Debugging Tips:")
    for i, tip in enumerate(debugging_tips, 1):
        print(f"{i:2d}. {tip}")
    
    # Example debugging - show graph
    print(f"\nExample - Simple model graph:")
    try:
        print(traced_simple.graph)
    except:
        print("Graph visualization not available")
    
    print("\nTorchScript compilation demonstration completed!")
    print("Generated directory: demo_scripted_models/")