import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as quant
from torch.nn.utils import prune
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import copy
import time
import numpy as np

# Model Quantization
class QuantizationOptimizer:
    """Model quantization for inference speedup"""
    
    def __init__(self, model):
        self.model = model
        self.quantized_model = None
    
    def post_training_quantization(self, calibration_loader=None):
        """Post-training quantization (PTQ)"""
        # Prepare model for quantization
        self.model.eval()
        
        # Set quantization config
        self.model.qconfig = quant.get_default_qconfig('fbgemm')
        
        # Prepare model
        prepared_model = quant.prepare(self.model, inplace=False)
        
        # Calibration (if data provided)
        if calibration_loader:
            with torch.no_grad():
                for data, _ in calibration_loader:
                    prepared_model(data)
        
        # Convert to quantized model
        self.quantized_model = quant.convert(prepared_model, inplace=False)
        return self.quantized_model
    
    def dynamic_quantization(self):
        """Dynamic quantization (weights only)"""
        self.quantized_model = quant.quantize_dynamic(
            self.model, 
            {nn.Linear, nn.Conv2d}, 
            dtype=torch.qint8
        )
        return self.quantized_model
    
    def compare_models(self, test_input):
        """Compare original and quantized model performance"""
        if self.quantized_model is None:
            print("No quantized model available")
            return
        
        # Timing comparison
        with torch.no_grad():
            # Original model
            start_time = time.time()
            for _ in range(100):
                _ = self.model(test_input)
            original_time = time.time() - start_time
            
            # Quantized model
            start_time = time.time()
            for _ in range(100):
                _ = self.quantized_model(test_input)
            quantized_time = time.time() - start_time
        
        # Model size comparison
        original_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        quantized_size = sum(p.numel() * p.element_size() for p in self.quantized_model.parameters())
        
        print(f"Original model time: {original_time:.4f}s")
        print(f"Quantized model time: {quantized_time:.4f}s")
        print(f"Speedup: {original_time/quantized_time:.2f}x")
        print(f"Original model size: {original_size/1024**2:.2f} MB")
        print(f"Quantized model size: {quantized_size/1024**2:.2f} MB")
        print(f"Size reduction: {original_size/quantized_size:.2f}x")

# Model Pruning
class PruningOptimizer:
    """Model pruning for efficiency"""
    
    def __init__(self, model):
        self.model = model
        self.original_model = copy.deepcopy(model)
    
    def magnitude_pruning(self, pruning_ratio=0.2):
        """Magnitude-based unstructured pruning"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
        
        return self.model
    
    def structured_pruning(self, pruning_ratio=0.2):
        """Structured pruning (remove entire channels/filters)"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, name='weight', amount=pruning_ratio, n=2, dim=0)
            elif isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
        
        return self.model
    
    def gradual_pruning(self, initial_sparsity=0.0, final_sparsity=0.8, num_steps=10):
        """Gradual magnitude pruning"""
        sparsity_schedule = torch.linspace(initial_sparsity, final_sparsity, num_steps)
        
        for step, sparsity in enumerate(sparsity_schedule):
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    prune.l1_unstructured(module, name='weight', amount=sparsity.item())
            
            print(f"Pruning step {step + 1}/{num_steps}, Sparsity: {sparsity:.2f}")
        
        return self.model
    
    def remove_pruning_masks(self):
        """Permanently remove pruned weights"""
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                try:
                    prune.remove(module, 'weight')
                except ValueError:
                    pass  # No pruning mask found
        
        return self.model
    
    def calculate_sparsity(self):
        """Calculate overall model sparsity"""
        total_params = 0
        zero_params = 0
        
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                total_params += module.weight.numel()
                zero_params += (module.weight == 0).sum().item()
        
        sparsity = zero_params / total_params
        print(f"Model sparsity: {sparsity:.2%}")
        return sparsity

# Knowledge Distillation
class KnowledgeDistillationOptimizer:
    """Knowledge distillation for model compression"""
    
    def __init__(self, teacher_model, student_model, temperature=4.0, alpha=0.7):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha  # Balance between hard and soft targets
        
        # Freeze teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
    
    def distillation_loss(self, student_outputs, teacher_outputs, targets):
        """Calculate knowledge distillation loss"""
        # Soft target loss (teacher-student)
        soft_targets = F.softmax(teacher_outputs / self.temperature, dim=1)
        soft_prob = F.log_softmax(student_outputs / self.temperature, dim=1)
        soft_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean')
        soft_loss *= (self.temperature ** 2)
        
        # Hard target loss (ground truth)
        hard_loss = F.cross_entropy(student_outputs, targets)
        
        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        return total_loss, soft_loss, hard_loss
    
    def train_student(self, train_loader, num_epochs=10, lr=1e-3):
        """Train student model with knowledge distillation"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.teacher.to(device)
        self.student.to(device)
        
        optimizer = torch.optim.Adam(self.student.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            self.student.train()
            total_loss = 0.0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(device), targets.to(device)
                
                # Teacher predictions
                with torch.no_grad():
                    teacher_outputs = self.teacher(data)
                
                # Student predictions
                student_outputs = self.student(data)
                
                # Calculate loss
                loss, soft_loss, hard_loss = self.distillation_loss(
                    student_outputs, teacher_outputs, targets
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 50 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, '
                          f'Loss: {loss.item():.4f}, '
                          f'Soft: {soft_loss.item():.4f}, '
                          f'Hard: {hard_loss.item():.4f}')
            
            print(f'Epoch {epoch}: Average Loss: {total_loss/len(train_loader):.4f}')

# Neural Architecture Search (NAS) Components
class DifferentiableCell(nn.Module):
    """Differentiable cell for NAS"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Candidate operations
        self.ops = nn.ModuleList([
            nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.Conv2d(in_channels, out_channels, 5, padding=2),
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.AvgPool2d(3, stride=1, padding=1),
            SeparableConv2d(in_channels, out_channels, 3),
            SeparableConv2d(in_channels, out_channels, 5),
        ])
        
        # Architecture parameters (learnable)
        self.arch_params = nn.Parameter(torch.randn(len(self.ops)))
    
    def forward(self, x):
        # Weighted combination of all operations
        weights = F.softmax(self.arch_params, dim=0)
        output = sum(w * op(x) for w, op in zip(weights, self.ops))
        return output

class SeparableConv2d(nn.Module):
    """Separable convolution (depthwise + pointwise)"""
    
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                  padding=kernel_size//2, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# TensorRT Integration (Mock implementation)
class TensorRTOptimizer:
    """TensorRT optimization for deployment"""
    
    def __init__(self, model):
        self.model = model
        self.trt_model = None
    
    def convert_to_tensorrt(self, example_input, precision='fp16'):
        """Convert PyTorch model to TensorRT (mock implementation)"""
        print(f"Converting model to TensorRT with {precision} precision...")
        
        # In real implementation, you would use:
        # import torch_tensorrt
        # self.trt_model = torch_tensorrt.compile(self.model, ...)
        
        # Mock conversion
        self.trt_model = self.model  # Placeholder
        print("Model converted to TensorRT format")
        return self.trt_model
    
    def benchmark_inference(self, test_input, num_runs=100):
        """Benchmark inference speed"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_input = test_input.to(device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(test_input)
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(test_input)
        
        avg_time = (time.time() - start_time) / num_runs
        throughput = 1.0 / avg_time
        
        print(f"Average inference time: {avg_time*1000:.2f} ms")
        print(f"Throughput: {throughput:.2f} FPS")
        return avg_time, throughput

# Mixed Precision Training
class MixedPrecisionOptimizer:
    """Mixed precision training for memory and speed optimization"""
    
    def __init__(self, model):
        self.model = model
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train_with_amp(self, train_loader, num_epochs=5):
        """Train with Automatic Mixed Precision"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            self.model.train()
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(device), targets.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass with autocast
                with torch.cuda.amp.autocast():
                    outputs = self.model(data)
                    loss = criterion(outputs, targets)
                
                # Backward pass with scaled gradients
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                if batch_idx % 50 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

# Comprehensive Optimization Pipeline
class VisionModelOptimizer:
    """Complete optimization pipeline for vision models"""
    
    def __init__(self, model):
        self.original_model = copy.deepcopy(model)
        self.model = model
        self.optimization_history = {}
    
    def full_optimization_pipeline(self, train_loader, val_loader, target_accuracy=0.9):
        """Apply multiple optimization techniques"""
        print("Starting comprehensive optimization pipeline...")
        
        # 1. Knowledge Distillation (if teacher model available)
        print("\n1. Knowledge Distillation...")
        teacher = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        kd_optimizer = KnowledgeDistillationOptimizer(teacher, self.model)
        kd_optimizer.train_student(train_loader, num_epochs=3)
        
        # 2. Magnitude Pruning
        print("\n2. Magnitude Pruning...")
        prune_optimizer = PruningOptimizer(self.model)
        prune_optimizer.magnitude_pruning(pruning_ratio=0.3)
        sparsity = prune_optimizer.calculate_sparsity()
        self.optimization_history['sparsity'] = sparsity
        
        # 3. Fine-tuning after pruning
        print("\n3. Fine-tuning after pruning...")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self._fine_tune(train_loader, optimizer, num_epochs=2)
        
        # 4. Quantization
        print("\n4. Post-training Quantization...")
        quant_optimizer = QuantizationOptimizer(self.model)
        quantized_model = quant_optimizer.post_training_quantization(val_loader)
        
        # 5. Performance evaluation
        print("\n5. Performance Evaluation...")
        test_input = torch.randn(1, 3, 224, 224)
        quant_optimizer.compare_models(test_input)
        
        return quantized_model
    
    def _fine_tune(self, train_loader, optimizer, num_epochs=2):
        """Fine-tune model after optimization"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            self.model.train()
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                if batch_idx % 100 == 0:
                    print(f'Fine-tune Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

# Benchmarking utilities
def benchmark_model_efficiency(model, input_size=(1, 3, 224, 224), num_runs=100):
    """Comprehensive model efficiency benchmark"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    test_input = torch.randn(*input_size).to(device)
    
    # Memory usage
    torch.cuda.empty_cache()
    memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    # Inference timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(test_input)
    
    total_time = time.time() - start_time
    memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    # Results
    avg_time = total_time / num_runs
    throughput = 1.0 / avg_time
    memory_usage = (memory_after - memory_before) / (1024**2)  # MB
    
    # Model size
    param_count = sum(p.numel() for p in model.parameters())
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)  # MB
    
    print(f"Model Efficiency Benchmark:")
    print(f"Parameters: {param_count:,}")
    print(f"Model size: {model_size:.2f} MB")
    print(f"Memory usage: {memory_usage:.2f} MB")
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} FPS")
    
    return {
        'parameters': param_count,
        'model_size_mb': model_size,
        'memory_usage_mb': memory_usage,
        'inference_time_ms': avg_time * 1000,
        'throughput_fps': throughput
    }

if __name__ == "__main__":
    # Create test model
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    print("Testing vision model optimization techniques...")
    
    # Test quantization
    print("\n1. Testing Quantization:")
    quant_opt = QuantizationOptimizer(copy.deepcopy(model))
    quantized_model = quant_opt.dynamic_quantization()
    test_input = torch.randn(1, 3, 224, 224)
    quant_opt.compare_models(test_input)
    
    # Test pruning
    print("\n2. Testing Pruning:")
    prune_opt = PruningOptimizer(copy.deepcopy(model))
    pruned_model = prune_opt.magnitude_pruning(0.3)
    sparsity = prune_opt.calculate_sparsity()
    
    # Test knowledge distillation setup
    print("\n3. Testing Knowledge Distillation setup:")
    teacher = models.resnet34(weights=None)
    teacher.fc = nn.Linear(teacher.fc.in_features, 10)
    student = models.resnet18(weights=None)
    student.fc = nn.Linear(student.fc.in_features, 10)
    
    kd_opt = KnowledgeDistillationOptimizer(teacher, student)
    # Note: Would need actual data for training
    
    # Test mixed precision
    print("\n4. Testing Mixed Precision setup:")
    mp_opt = MixedPrecisionOptimizer(copy.deepcopy(model))
    
    # Test TensorRT optimizer
    print("\n5. Testing TensorRT optimizer:")
    trt_opt = TensorRTOptimizer(copy.deepcopy(model))
    trt_model = trt_opt.convert_to_tensorrt(test_input)
    timing_results = trt_opt.benchmark_inference(test_input, num_runs=50)
    
    # Comprehensive benchmark
    print("\n6. Comprehensive Efficiency Benchmark:")
    benchmark_results = benchmark_model_efficiency(model)
    
    # Test differentiable cell
    print("\n7. Testing Differentiable NAS Cell:")
    nas_cell = DifferentiableCell(64, 128)
    x = torch.randn(1, 64, 32, 32)
    nas_output = nas_cell(x)
    print(f"NAS cell output shape: {nas_output.shape}")
    print(f"Architecture parameters: {nas_cell.arch_params.data}")
    
    print("\nVision model optimization testing completed!")