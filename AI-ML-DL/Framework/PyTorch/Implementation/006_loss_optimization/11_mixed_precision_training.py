#!/usr/bin/env python3
"""PyTorch Mixed Precision Training - AMP, FP16 training, GradScaler"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import time

print("=== Mixed Precision Training Overview ===")

print("Mixed precision training concepts:")
print("1. Automatic Mixed Precision (AMP)")
print("2. FP16 vs FP32 precision")
print("3. GradScaler for gradient scaling")
print("4. Loss scaling techniques")
print("5. Memory and speed optimization")
print("6. Numerical stability considerations")
print("7. Model compatibility")
print("8. Performance benchmarking")

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"CUDA capability: {torch.cuda.get_device_capability()}")

print("\n=== Model Setup for Mixed Precision ===")

class MixedPrecisionModel(nn.Module):
    def __init__(self, input_size=128, hidden_sizes=[256, 512, 256], num_classes=10):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Create model and move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MixedPrecisionModel().to(device)

# Create sample data
batch_size = 64
input_size = 128
num_classes = 10

sample_input = torch.randn(batch_size, input_size).to(device)
sample_target = torch.randint(0, num_classes, (batch_size,)).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Device: {device}")
print(f"Input shape: {sample_input.shape}")
print(f"Target shape: {sample_target.shape}")

print("\n=== Basic Automatic Mixed Precision (AMP) ===")

def train_step_fp32(model, optimizer, input_data, targets, loss_fn):
    """Standard FP32 training step"""
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(input_data)
    loss = loss_fn(outputs, targets)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train_step_amp(model, optimizer, scaler, input_data, targets, loss_fn):
    """AMP training step with autocast and GradScaler"""
    optimizer.zero_grad()
    
    # Forward pass with autocast
    with autocast():
        outputs = model(input_data)
        loss = loss_fn(outputs, targets)
    
    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item()

# Initialize optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Initialize GradScaler for AMP
scaler = GradScaler()

print("Comparing FP32 vs AMP training:")

# FP32 training
start_time = time.time()
for epoch in range(3):
    loss_fp32 = train_step_fp32(model, optimizer, sample_input, sample_target, loss_fn)
    if epoch == 0:
        print(f"FP32 - Epoch {epoch}: Loss = {loss_fp32:.6f}")
fp32_time = time.time() - start_time

# Reset model for fair comparison
model = MixedPrecisionModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = GradScaler()

# AMP training
start_time = time.time()
for epoch in range(3):
    loss_amp = train_step_amp(model, optimizer, scaler, sample_input, sample_target, loss_fn)
    if epoch == 0:
        print(f"AMP - Epoch {epoch}: Loss = {loss_amp:.6f}")
amp_time = time.time() - start_time

print(f"\nTiming comparison (3 steps):")
print(f"FP32 time: {fp32_time:.4f}s")
print(f"AMP time: {amp_time:.4f}s")
if fp32_time > 0:
    speedup = fp32_time / amp_time
    print(f"Speedup: {speedup:.2f}x")

print("\n=== GradScaler Configuration ===")

# Different GradScaler configurations
scalers_config = {
    'Default': GradScaler(),
    'Conservative': GradScaler(init_scale=2**12, growth_factor=1.5, backoff_factor=0.8),
    'Aggressive': GradScaler(init_scale=2**16, growth_factor=2.0, backoff_factor=0.5),
    'Custom': GradScaler(init_scale=2**10, growth_factor=1.2, backoff_factor=0.9, 
                        growth_interval=1000, enabled=True)
}

print("GradScaler configurations:")
for name, scaler_config in scalers_config.items():
    print(f"\n{name} GradScaler:")
    print(f"  Initial scale: {scaler_config.get_scale():.0f}")
    print(f"  Growth factor: {scaler_config.get_growth_factor()}")
    print(f"  Backoff factor: {scaler_config.get_backoff_factor()}")
    print(f"  Growth interval: {scaler_config.get_growth_interval()}")

# Test gradient scaling behavior
print(f"\nTesting gradient scaling with different configurations:")

def test_gradient_scaling(scaler, name, num_steps=10):
    """Test gradient scaling behavior"""
    model_test = MixedPrecisionModel().to(device)
    optimizer_test = optim.Adam(model_test.parameters(), lr=0.001)
    
    scales = []
    for step in range(num_steps):
        optimizer_test.zero_grad()
        
        with autocast():
            outputs = model_test(sample_input)
            loss = loss_fn(outputs, sample_target)
        
        # Record scale before update
        scales.append(scaler.get_scale())
        
        scaler.scale(loss).backward()
        scaler.step(optimizer_test)
        scaler.update()
    
    print(f"{name:12}: Scale progression: {scales[0]:.0f} → {scales[-1]:.0f}")
    return scales

# Test different scalers
for name, scaler_test in scalers_config.items():
    test_gradient_scaling(scaler_test, name)

print("\n=== Custom autocast Contexts ===")

# Custom autocast with different dtypes
def demonstrate_autocast_contexts():
    """Demonstrate different autocast contexts"""
    
    input_tensor = torch.randn(10, 10).to(device)
    
    print("Tensor dtypes in different contexts:")
    
    # Default context (FP32)
    result_fp32 = torch.mm(input_tensor, input_tensor)
    print(f"Default context: {result_fp32.dtype}")
    
    # Autocast context (automatic mixed precision)
    with autocast():
        result_amp = torch.mm(input_tensor, input_tensor)
        print(f"Autocast context: {result_amp.dtype}")
    
    # Autocast with specific dtype
    with autocast(dtype=torch.float16):
        result_fp16 = torch.mm(input_tensor, input_tensor)
        print(f"Autocast FP16: {result_fp16.dtype}")
    
    # Autocast disabled
    with autocast(enabled=False):
        result_disabled = torch.mm(input_tensor, input_tensor)
        print(f"Autocast disabled: {result_disabled.dtype}")

demonstrate_autocast_contexts()

print("\n=== Memory Usage Comparison ===")

def measure_memory_usage(model, input_data, use_amp=False):
    """Measure GPU memory usage"""
    if not torch.cuda.is_available():
        return None
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    if use_amp:
        with autocast():
            output = model(input_data)
    else:
        output = model(input_data)
    
    # Force computation
    loss = output.sum()
    loss.backward()
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    return peak_memory

# Create larger model for memory comparison
large_model = MixedPrecisionModel(input_size=512, hidden_sizes=[1024, 2048, 1024]).to(device)
large_input = torch.randn(128, 512).to(device)

if torch.cuda.is_available():
    fp32_memory = measure_memory_usage(large_model, large_input, use_amp=False)
    amp_memory = measure_memory_usage(large_model, large_input, use_amp=True)
    
    print(f"Memory usage comparison:")
    print(f"FP32 peak memory: {fp32_memory:.1f} MB")
    print(f"AMP peak memory: {amp_memory:.1f} MB")
    if fp32_memory and amp_memory:
        memory_savings = (fp32_memory - amp_memory) / fp32_memory * 100
        print(f"Memory savings: {memory_savings:.1f}%")
else:
    print("CUDA not available - skipping memory comparison")

print("\n=== Loss Scaling Strategies ===")

class AdaptiveLossScaling:
    """Custom adaptive loss scaling implementation"""
    
    def __init__(self, init_scale=2**15, min_scale=1, max_scale=2**24):
        self.scale = init_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.growth_tracker = 0
        self.growth_interval = 2000
    
    def scale_loss(self, loss):
        """Scale the loss"""
        return loss * self.scale
    
    def unscale_gradients(self, optimizer):
        """Unscale gradients"""
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.div_(self.scale)
    
    def update_scale(self, found_inf):
        """Update the scale based on gradient overflow"""
        if found_inf:
            # Reduce scale if overflow detected
            self.scale = max(self.scale / 2, self.min_scale)
            self.growth_tracker = 0
        else:
            # Increase scale if no overflow for growth_interval steps
            self.growth_tracker += 1
            if self.growth_tracker >= self.growth_interval:
                self.scale = min(self.scale * 2, self.max_scale)
                self.growth_tracker = 0
        
        return self.scale
    
    def check_overflow(self, optimizer):
        """Check for gradient overflow"""
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                        return True
        return False

# Test custom loss scaling
print("Custom adaptive loss scaling:")
adaptive_scaler = AdaptiveLossScaling(init_scale=2**10)

model_custom = MixedPrecisionModel().to(device)
optimizer_custom = optim.Adam(model_custom.parameters(), lr=0.001)

for step in range(10):
    optimizer_custom.zero_grad()
    
    with autocast():
        outputs = model_custom(sample_input)
        loss = loss_fn(outputs, sample_target)
    
    # Scale loss
    scaled_loss = adaptive_scaler.scale_loss(loss)
    scaled_loss.backward()
    
    # Check for overflow
    adaptive_scaler.unscale_gradients(optimizer_custom)
    found_inf = adaptive_scaler.check_overflow(optimizer_custom)
    
    if not found_inf:
        optimizer_custom.step()
    
    # Update scale
    new_scale = adaptive_scaler.update_scale(found_inf)
    
    if step % 3 == 0:
        status = "OVERFLOW" if found_inf else "OK"
        print(f"  Step {step}: Scale = {new_scale:.0f}, Loss = {loss.item():.6f} [{status}]")

print("\n=== Model Compatibility Testing ===")

def test_amp_compatibility(model, input_data):
    """Test if model is compatible with AMP"""
    try:
        with autocast():
            output = model(input_data)
        
        # Check output dtype
        if hasattr(output, 'dtype'):
            compatible = True
            output_dtype = output.dtype
        else:
            compatible = False
            output_dtype = "N/A"
        
        return compatible, output_dtype
    
    except Exception as e:
        return False, str(e)

# Test different model types
test_models = {
    'Standard CNN': nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 10)
    ).to(device),
    
    'Batch Norm Model': nn.Sequential(
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, 10)
    ).to(device),
    
    'Layer Norm Model': nn.Sequential(
        nn.Linear(128, 64),
        nn.LayerNorm(64),
        nn.ReLU(),
        nn.Linear(64, 10)
    ).to(device)
}

print("AMP compatibility testing:")
for name, test_model in test_models.items():
    if 'CNN' in name:
        test_input = torch.randn(4, 3, 32, 32).to(device)
    else:
        test_input = sample_input
    
    compatible, result = test_amp_compatibility(test_model, test_input)
    print(f"  {name:18}: {'✓' if compatible else '✗'} ({result})")

print("\n=== Advanced AMP Techniques ===")

class AMPTrainer:
    """Advanced AMP trainer with monitoring"""
    
    def __init__(self, model, optimizer, scaler=None, clip_grad_norm=None):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler or GradScaler()
        self.clip_grad_norm = clip_grad_norm
        
        # Monitoring
        self.scale_history = []
        self.overflow_count = 0
        self.step_count = 0
    
    def train_step(self, input_data, targets, loss_fn):
        """Advanced training step with monitoring"""
        self.step_count += 1
        
        # Record scale
        self.scale_history.append(self.scaler.get_scale())
        
        self.optimizer.zero_grad()
        
        # Forward pass with autocast
        with autocast():
            outputs = self.model(input_data)
            loss = loss_fn(outputs, targets)
        
        # Backward pass
        self.scaler.scale(loss).backward()
        
        # Optional gradient clipping
        if self.clip_grad_norm:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        
        # Check for overflow before stepping
        scale_before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        scale_after = self.scaler.get_scale()
        
        # Detect overflow
        if scale_after < scale_before:
            self.overflow_count += 1
        
        return loss.item()
    
    def get_statistics(self):
        """Get training statistics"""
        return {
            'total_steps': self.step_count,
            'overflow_count': self.overflow_count,
            'overflow_rate': self.overflow_count / max(self.step_count, 1),
            'current_scale': self.scaler.get_scale(),
            'scale_range': (min(self.scale_history), max(self.scale_history)) if self.scale_history else (0, 0)
        }

# Test advanced AMP trainer
print("Advanced AMP training with monitoring:")

model_advanced = MixedPrecisionModel().to(device)
optimizer_advanced = optim.Adam(model_advanced.parameters(), lr=0.001)
trainer = AMPTrainer(model_advanced, optimizer_advanced, clip_grad_norm=1.0)

# Simulate training with varying difficulty
for epoch in range(15):
    # Occasionally introduce numerical instability
    if epoch % 7 == 0 and epoch > 0:
        # Large input to potentially cause overflow
        noisy_input = sample_input * (10 if epoch == 7 else 1)
    else:
        noisy_input = sample_input
    
    loss = trainer.train_step(noisy_input, sample_target, loss_fn)
    
    if epoch % 5 == 0:
        stats = trainer.get_statistics()
        print(f"  Epoch {epoch:2d}: Loss = {loss:.6f}, Scale = {stats['current_scale']:.0f}, "
              f"Overflow rate = {stats['overflow_rate']:.3f}")

# Final statistics
final_stats = trainer.get_statistics()
print(f"\nFinal training statistics:")
print(f"  Total steps: {final_stats['total_steps']}")
print(f"  Overflow count: {final_stats['overflow_count']}")
print(f"  Overflow rate: {final_stats['overflow_rate']:.3f}")
print(f"  Scale range: {final_stats['scale_range'][0]:.0f} - {final_stats['scale_range'][1]:.0f}")

print("\n=== Performance Benchmarking ===")

def benchmark_training(model, optimizer, input_data, targets, loss_fn, use_amp=False, num_iterations=50):
    """Benchmark training performance"""
    if use_amp:
        scaler = GradScaler()
    
    # Warm-up
    for _ in range(5):
        if use_amp:
            train_step_amp(model, optimizer, scaler, input_data, targets, loss_fn)
        else:
            train_step_fp32(model, optimizer, input_data, targets, loss_fn)
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    for _ in range(num_iterations):
        if use_amp:
            train_step_amp(model, optimizer, scaler, input_data, targets, loss_fn)
        else:
            train_step_fp32(model, optimizer, input_data, targets, loss_fn)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    return avg_time

# Benchmark comparison
if torch.cuda.is_available():
    print("Performance benchmarking (50 iterations):")
    
    # Create fresh models for fair comparison
    model_fp32 = MixedPrecisionModel().to(device)
    model_amp = MixedPrecisionModel().to(device)
    model_amp.load_state_dict(model_fp32.state_dict())
    
    optimizer_fp32 = optim.Adam(model_fp32.parameters(), lr=0.001)
    optimizer_amp = optim.Adam(model_amp.parameters(), lr=0.001)
    
    fp32_time = benchmark_training(model_fp32, optimizer_fp32, sample_input, sample_target, loss_fn, use_amp=False)
    amp_time = benchmark_training(model_amp, optimizer_amp, sample_input, sample_target, loss_fn, use_amp=True)
    
    print(f"FP32 avg time per step: {fp32_time*1000:.2f} ms")
    print(f"AMP avg time per step: {amp_time*1000:.2f} ms")
    print(f"Speedup: {fp32_time/amp_time:.2f}x")
    
    # Throughput comparison
    batch_per_sec_fp32 = 1.0 / fp32_time
    batch_per_sec_amp = 1.0 / amp_time
    
    print(f"FP32 throughput: {batch_per_sec_fp32:.1f} batches/sec")
    print(f"AMP throughput: {batch_per_sec_amp:.1f} batches/sec")
else:
    print("CUDA not available - skipping performance benchmarking")

print("\n=== Mixed Precision Best Practices ===")

print("When to Use Mixed Precision:")
print("1. Large models with memory constraints")
print("2. Training on modern GPUs (V100, A100, RTX series)")
print("3. Computer vision and NLP tasks")
print("4. When training speed is critical")
print("5. Models with many matrix multiplications")

print("\nAMP Implementation Guidelines:")
print("1. Wrap forward pass with autocast()")
print("2. Use GradScaler for gradient scaling")
print("3. Monitor gradient overflow rates")
print("4. Test model compatibility first")
print("5. Adjust scaler parameters if needed")

print("\nTroubleshooting Common Issues:")
print("1. Gradient overflow: Reduce learning rate or adjust scaler")
print("2. NaN losses: Check for numerical instabilities")
print("3. Poor convergence: Verify gradient scaling is working")
print("4. Memory not reduced: Ensure autocast covers main computations")
print("5. No speedup: Check if operations are AMP-compatible")

print("\nOptimization Tips:")
print("1. Use autocast only around compute-intensive operations")
print("2. Keep loss computation inside autocast")
print("3. Monitor scale factor throughout training")
print("4. Use gradient clipping with AMP carefully")
print("5. Profile to identify actual bottlenecks")

print("\nCompatibility Notes:")
print("1. Most PyTorch operations support AMP")
print("2. Some operations always run in FP32 (e.g., softmax)")
print("3. Custom CUDA kernels may need special handling")
print("4. Batch/Layer normalization work well with AMP")
print("5. RNNs may have limited AMP benefits")

print("\n=== Mixed Precision Training Complete ===")

# Memory cleanup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
del model, sample_input, sample_target