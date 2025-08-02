import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as quantization
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any

# Sample Models for Quantization Demo
class QuantizableCNN(nn.Module):
    """CNN designed for quantization"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # Use quantization-friendly layers
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()
        
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
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

class QuantizableResNetBlock(nn.Module):
    """Quantization-friendly ResNet block"""
    
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
        
        # For quantization-aware training
        self.add_relu = nn.quantized.FloatFunctional()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Use quantized add for proper quantization
        out = self.add_relu.add_relu(out, self.shortcut(x))
        return out

# Quantization Utilities
class ModelQuantizer:
    """Utilities for model quantization"""
    
    def __init__(self):
        self.supported_backends = ['fbgemm', 'qnnpack']
    
    def dynamic_quantization(self, model: nn.Module, 
                           qconfig_spec: Optional[Dict] = None) -> nn.Module:
        """Apply dynamic quantization to model"""
        
        model.eval()
        
        if qconfig_spec is None:
            # Default: quantize Linear and Conv2d layers
            qconfig_spec = {nn.Linear, nn.Conv2d}
        
        quantized_model = quantization.quantize_dynamic(
            model, qconfig_spec, dtype=torch.qint8
        )
        
        print("✓ Dynamic quantization applied")
        return quantized_model
    
    def static_quantization(self, model: nn.Module, 
                           calibration_data: torch.utils.data.DataLoader,
                           backend: str = 'fbgemm') -> nn.Module:
        """Apply static quantization with calibration"""
        
        if backend not in self.supported_backends:
            raise ValueError(f"Backend {backend} not supported. Use one of {self.supported_backends}")
        
        # Set backend
        torch.backends.quantized.engine = backend
        
        # Set model to eval mode
        model.eval()
        
        # Set quantization config
        model.qconfig = quantization.get_default_qconfig(backend)
        
        # Prepare model for quantization
        prepared_model = quantization.prepare(model)
        
        # Calibrate model
        print("Calibrating model...")
        self._calibrate_model(prepared_model, calibration_data)
        
        # Convert to quantized model
        quantized_model = quantization.convert(prepared_model)
        
        print("✓ Static quantization applied")
        return quantized_model
    
    def quantization_aware_training_prepare(self, model: nn.Module,
                                          backend: str = 'fbgemm') -> nn.Module:
        """Prepare model for quantization-aware training"""
        
        # Set backend
        torch.backends.quantized.engine = backend
        
        # Set model to train mode
        model.train()
        
        # Set quantization config for QAT
        model.qconfig = quantization.get_default_qat_qconfig(backend)
        
        # Prepare model for QAT
        prepared_model = quantization.prepare_qat(model)
        
        print("✓ Model prepared for quantization-aware training")
        return prepared_model
    
    def quantization_aware_training_convert(self, model: nn.Module) -> nn.Module:
        """Convert QAT model to quantized model"""
        
        model.eval()
        quantized_model = quantization.convert(model)
        
        print("✓ QAT model converted to quantized model")
        return quantized_model
    
    def _calibrate_model(self, model: nn.Module, 
                        calibration_data: torch.utils.data.DataLoader):
        """Calibrate model with representative data"""
        
        model.eval()
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(calibration_data):
                if batch_idx >= 10:  # Use only a few batches for calibration
                    break
                _ = model(data)
    
    def fuse_modules(self, model: nn.Module) -> nn.Module:
        """Fuse conv-bn-relu modules for better quantization"""
        
        # Find modules to fuse
        modules_to_fuse = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential):
                # Look for conv-bn-relu patterns
                for i in range(len(module) - 2):
                    if (isinstance(module[i], nn.Conv2d) and
                        isinstance(module[i + 1], nn.BatchNorm2d) and
                        isinstance(module[i + 2], nn.ReLU)):
                        modules_to_fuse.append([f"{name}.{i}", f"{name}.{i+1}", f"{name}.{i+2}"])
        
        if modules_to_fuse:
            fused_model = quantization.fuse_modules(model, modules_to_fuse)
            print(f"✓ Fused {len(modules_to_fuse)} module groups")
            return fused_model
        else:
            print("No fuseable modules found")
            return model

class QuantizationAnalyzer:
    """Analyze quantization effects on model performance"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def compare_models(self, original_model: nn.Module, 
                      quantized_model: nn.Module,
                      test_data: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Compare original and quantized model performance"""
        
        results = {}
        
        # Accuracy comparison
        original_accuracy = self._calculate_accuracy(original_model, test_data)
        quantized_accuracy = self._calculate_accuracy(quantized_model, test_data)
        
        results['accuracy'] = {
            'original': original_accuracy,
            'quantized': quantized_accuracy,
            'difference': original_accuracy - quantized_accuracy
        }
        
        # Speed comparison
        test_input = torch.randn(1, 3, 32, 32)
        original_speed = self._benchmark_speed(original_model, test_input)
        quantized_speed = self._benchmark_speed(quantized_model, test_input)
        
        results['speed'] = {
            'original_ms': original_speed,
            'quantized_ms': quantized_speed,
            'speedup': original_speed / quantized_speed
        }
        
        # Size comparison
        original_size = self._calculate_model_size(original_model)
        quantized_size = self._calculate_model_size(quantized_model)
        
        results['size'] = {
            'original_mb': original_size,
            'quantized_mb': quantized_size,
            'compression': original_size / quantized_size
        }
        
        return results
    
    def _calculate_accuracy(self, model: nn.Module, 
                           test_data: torch.utils.data.DataLoader) -> float:
        """Calculate model accuracy on test data"""
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in test_data:
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return correct / total
    
    def _benchmark_speed(self, model: nn.Module, test_input: torch.Tensor,
                        num_runs: int = 100) -> float:
        """Benchmark model inference speed"""
        
        model.eval()
        test_input = test_input.to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = model(test_input)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return np.mean(times)
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB"""
        
        total_size = 0
        
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        
        for buffer in model.buffers():
            total_size += buffer.numel() * buffer.element_size()
        
        return total_size / (1024 * 1024)  # Convert to MB

# Quantization-Aware Training
class QATTrainer:
    """Trainer for quantization-aware training"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.device = device
        self.original_model = model.to(device)
    
    def train_qat(self, train_loader: torch.utils.data.DataLoader,
                  val_loader: torch.utils.data.DataLoader,
                  num_epochs: int = 10,
                  lr: float = 0.001) -> nn.Module:
        """Perform quantization-aware training"""
        
        # Prepare model for QAT
        quantizer = ModelQuantizer()
        qat_model = quantizer.quantization_aware_training_prepare(self.original_model)
        qat_model = qat_model.to(self.device)
        
        # Setup training
        optimizer = torch.optim.Adam(qat_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            qat_model.train()
            train_loss = 0.0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = qat_model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx >= 10:  # Limit batches for demo
                    break
            
            # Validation phase
            val_accuracy = self._validate(qat_model, val_loader)
            
            scheduler.step()
            
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss = {train_loss/(batch_idx+1):.4f}, "
                  f"Val Acc = {val_accuracy:.4f}")
        
        # Convert to quantized model
        quantized_model = quantizer.quantization_aware_training_convert(qat_model)
        
        return quantized_model
    
    def _validate(self, model: nn.Module, val_loader: torch.utils.data.DataLoader) -> float:
        """Validate model performance"""
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(val_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                if batch_idx >= 5:  # Limit batches for demo
                    break
        
        return correct / total if total > 0 else 0.0

# Mobile Quantization
class MobileQuantizer:
    """Specialized quantization for mobile deployment"""
    
    @staticmethod
    def quantize_for_mobile(model: nn.Module, 
                           calibration_data: torch.utils.data.DataLoader) -> nn.Module:
        """Quantize model specifically for mobile deployment"""
        
        # Set qnnpack backend for mobile
        torch.backends.quantized.engine = 'qnnpack'
        
        # Prepare model
        model.eval()
        model.qconfig = quantization.get_default_qconfig('qnnpack')
        
        # Fuse modules
        quantizer = ModelQuantizer()
        fused_model = quantizer.fuse_modules(model)
        
        # Prepare and calibrate
        prepared_model = quantization.prepare(fused_model)
        
        # Calibrate
        quantizer._calibrate_model(prepared_model, calibration_data)
        
        # Convert to quantized
        quantized_model = quantization.convert(prepared_model)
        
        # Convert to TorchScript for mobile
        scripted_model = torch.jit.script(quantized_model)
        
        # Optimize for mobile
        mobile_model = torch.utils.mobile_optimizer.optimize_for_mobile(scripted_model)
        
        return mobile_model

if __name__ == "__main__":
    print("Model Quantization")
    print("=" * 20)
    
    device = torch.device('cpu')  # Quantization works best on CPU
    print(f"Using device: {device}")
    
    # Create sample model and data
    model = QuantizableCNN(num_classes=10)
    
    # Create sample dataset
    sample_data = torch.randn(100, 3, 32, 32)
    sample_labels = torch.randint(0, 10, (100,))
    dataset = torch.utils.data.TensorDataset(sample_data, sample_labels)
    
    # Split into train/test
    train_size = 80
    test_size = 20
    
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"Dataset: {len(train_dataset)} train, {len(test_dataset)} test samples")
    
    print("\n1. Dynamic Quantization")
    print("-" * 27)
    
    quantizer = ModelQuantizer()
    
    # Apply dynamic quantization
    dynamic_quantized = quantizer.dynamic_quantization(model)
    
    # Test dynamic quantized model
    test_input = torch.randn(1, 3, 32, 32)
    
    with torch.no_grad():
        original_output = model(test_input)
        dynamic_output = dynamic_quantized(test_input)
        
        output_diff = torch.max(torch.abs(original_output - dynamic_output)).item()
        print(f"Max output difference: {output_diff:.6f}")
    
    print("\n2. Static Quantization")
    print("-" * 25)
    
    # Apply static quantization
    static_quantized = quantizer.static_quantization(model, train_loader)
    
    # Test static quantized model
    with torch.no_grad():
        static_output = static_quantized(test_input)
        static_diff = torch.max(torch.abs(original_output - static_output)).item()
        print(f"Max output difference: {static_diff:.6f}")
    
    print("\n3. Module Fusion")
    print("-" * 20)
    
    # Apply module fusion
    fused_model = quantizer.fuse_modules(model)
    
    # Test fused model
    with torch.no_grad():
        fused_output = fused_model(test_input)
        fused_diff = torch.max(torch.abs(original_output - fused_output)).item()
        print(f"Max output difference after fusion: {fused_diff:.6f}")
    
    print("\n4. Quantization-Aware Training")
    print("-" * 37)
    
    # Perform QAT
    qat_trainer = QATTrainer(model, device)
    qat_model = qat_trainer.train_qat(train_loader, test_loader, num_epochs=3)
    
    # Test QAT model
    with torch.no_grad():
        qat_output = qat_model(test_input)
        qat_diff = torch.max(torch.abs(original_output - qat_output)).item()
        print(f"Max output difference after QAT: {qat_diff:.6f}")
    
    print("\n5. Performance Analysis")
    print("-" * 28)
    
    analyzer = QuantizationAnalyzer(device)
    
    # Compare different quantization methods
    quantized_models = {
        'Dynamic': dynamic_quantized,
        'Static': static_quantized,
        'QAT': qat_model
    }
    
    print("Performance Comparison:")
    print("-" * 25)
    print(f"{'Method':<12} {'Speed (ms)':<12} {'Size (MB)':<12} {'Speedup':<10} {'Compression':<12}")
    print("-" * 70)
    
    # Benchmark original model
    original_speed = analyzer._benchmark_speed(model, test_input)
    original_size = analyzer._calculate_model_size(model)
    
    print(f"{'Original':<12} {original_speed:<12.2f} {original_size:<12.2f} {'1.00x':<10} {'1.00x':<12}")
    
    # Benchmark quantized models
    for name, quantized_model in quantized_models.items():
        try:
            speed = analyzer._benchmark_speed(quantized_model, test_input)
            size = analyzer._calculate_model_size(quantized_model)
            speedup = original_speed / speed
            compression = original_size / size
            
            print(f"{name:<12} {speed:<12.2f} {size:<12.2f} {speedup:<10.2f}x {compression:<12.2f}x")
        
        except Exception as e:
            print(f"{name:<12} Error: {str(e)[:30]}")
    
    print("\n6. Mobile Quantization")
    print("-" * 25)
    
    # Quantize for mobile
    try:
        mobile_quantizer = MobileQuantizer()
        mobile_model = mobile_quantizer.quantize_for_mobile(model, train_loader)
        
        print("✓ Mobile quantization successful")
        
        # Test mobile model
        with torch.no_grad():
            mobile_output = mobile_model(test_input)
            mobile_diff = torch.max(torch.abs(original_output - mobile_output)).item()
            print(f"Max output difference (mobile): {mobile_diff:.6f}")
    
    except Exception as e:
        print(f"✗ Mobile quantization failed: {e}")
    
    print("\n7. Quantization Best Practices")
    print("-" * 35)
    
    best_practices = [
        "Use calibration data representative of production data",
        "Apply module fusion before quantization for better results",
        "Use QAT when accuracy loss is significant with post-training quantization",
        "Choose appropriate backend (fbgemm for x86, qnnpack for ARM)",
        "Test quantized models thoroughly before deployment",
        "Use QuantStub and DeQuantStub for proper quantization boundaries",
        "Avoid quantizing batch normalization layers directly",
        "Monitor accuracy degradation and adjust strategy accordingly",
        "Use INT8 quantization for optimal speed/accuracy trade-off",
        "Profile quantized models on target hardware"
    ]
    
    print("Quantization Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\n8. Troubleshooting Common Issues")
    print("-" * 38)
    
    troubleshooting = [
        "Accuracy drop: Try QAT or different calibration data",
        "Slow inference: Ensure proper backend selection",
        "Runtime errors: Check for unsupported operations",
        "Memory issues: Use dynamic quantization for large models",
        "Platform compatibility: Use appropriate quantization backend",
        "Numerical instability: Add QuantStub/DeQuantStub properly"
    ]
    
    print("Common Issues and Solutions:")
    for i, issue in enumerate(troubleshooting, 1):
        print(f"{i}. {issue}")
    
    print("\nModel quantization demonstration completed!")
    print("Key takeaways:")
    print("- Dynamic quantization: Easy to apply, good for memory reduction")
    print("- Static quantization: Better accuracy, requires calibration data")
    print("- QAT: Best accuracy, requires retraining")
    print("- Mobile quantization: Optimized for mobile deployment")