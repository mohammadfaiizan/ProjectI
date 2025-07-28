#!/usr/bin/env python3
"""PyTorch Mixed Precision Training - AMP operations and syntax"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== Mixed Precision Training Overview ===")

# Check AMP availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"AMP available: {hasattr(torch.cuda, 'amp')}")

# Mixed precision benefits
print("\nMixed Precision Benefits:")
print("1. ~2x faster training on modern GPUs (V100, A100, RTX series)")
print("2. ~50% memory reduction")
print("3. Maintains model accuracy with proper loss scaling")
print("4. Automatic handling of overflow/underflow")

if torch.cuda.is_available():
    print(f"GPU compute capability: {torch.cuda.get_device_capability()}")
    print(f"Supports Tensor Cores: {torch.cuda.get_device_capability()[0] >= 7}")

print("\n=== Basic AMP Usage ===")

# Import AMP components
try:
    from torch.cuda.amp import autocast, GradScaler
    amp_available = True
except ImportError:
    print("AMP not available in this PyTorch version")
    amp_available = False

if amp_available:
    # Simple model for demonstration
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 10)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x):
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.dropout(self.relu(self.fc2(x)))
            x = self.fc3(x)
            return x
    
    # Create model and move to GPU
    model = SimpleModel()
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Create optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize the scaler for gradient scaling
    scaler = GradScaler()
    
    print("AMP components initialized")

print("\n=== autocast Context Manager ===")

if amp_available and torch.cuda.is_available():
    # Sample data
    input_data = torch.randn(32, 784, device='cuda')
    target_data = torch.randint(0, 10, (32,), device='cuda')
    
    # Standard forward pass (FP32)
    model.train()
    output_fp32 = model(input_data)
    loss_fp32 = criterion(output_fp32, target_data)
    
    print(f"FP32 forward pass:")
    print(f"  Input dtype: {input_data.dtype}")
    print(f"  Output dtype: {output_fp32.dtype}")
    print(f"  Loss dtype: {loss_fp32.dtype}")
    print(f"  Loss value: {loss_fp32.item():.6f}")
    
    # Mixed precision forward pass
    with autocast():
        output_amp = model(input_data)
        loss_amp = criterion(output_amp, target_data)
    
    print(f"\nMixed precision forward pass:")
    print(f"  Input dtype: {input_data.dtype}")
    print(f"  Output dtype: {output_amp.dtype}")
    print(f"  Loss dtype: {loss_amp.dtype}")
    print(f"  Loss value: {loss_amp.item():.6f}")
    
    # Check if outputs are close
    outputs_close = torch.allclose(output_fp32, output_amp.float(), atol=1e-3)
    print(f"  Outputs are close: {outputs_close}")

print("\n=== GradScaler Usage ===")

if amp_available and torch.cuda.is_available():
    # Training step with AMP
    def amp_training_step(model, data, target, optimizer, criterion, scaler):
        """Single training step with AMP"""
        optimizer.zero_grad()
        
        # Forward pass with autocast
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Optimizer step with scaler
        scaler.step(optimizer)
        
        # Update scaler
        scaler.update()
        
        return loss.item()
    
    # Test AMP training step
    loss_value = amp_training_step(model, input_data, target_data, 
                                  optimizer, criterion, scaler)
    print(f"AMP training step completed, loss: {loss_value:.6f}")
    
    # Check scaler state
    print(f"Scaler scale factor: {scaler.get_scale()}")
    print(f"Scaler growth interval: {scaler.get_growth_interval()}")

print("\n=== Gradient Clipping with AMP ===")

if amp_available and torch.cuda.is_available():
    def amp_training_step_with_clipping(model, data, target, optimizer, 
                                       criterion, scaler, max_norm=1.0):
        """AMP training step with gradient clipping"""
        optimizer.zero_grad()
        
        # Forward pass
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Unscale gradients for clipping
        scaler.unscale_(optimizer)
        
        # Clip gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        return loss.item(), grad_norm.item()
    
    # Test with gradient clipping
    loss_val, grad_norm = amp_training_step_with_clipping(
        model, input_data, target_data, optimizer, criterion, scaler
    )
    print(f"AMP with clipping - Loss: {loss_val:.6f}, Grad norm: {grad_norm:.6f}")

print("\n=== Custom autocast Behavior ===")

if amp_available:
    # Operations that support autocast
    print("Operations that benefit from autocast:")
    print("- nn.Linear, nn.Conv2d, nn.ConvTranspose2d")
    print("- nn.RNN, nn.LSTM, nn.GRU")
    print("- Activations: ReLU, GELU, etc.")
    print("- Loss functions: CrossEntropyLoss, MSELoss, etc.")
    
    # Operations that should stay in FP32
    print("\nOperations that should remain in FP32:")
    print("- BatchNorm (handled automatically)")
    print("- Softmax with large tensors")
    print("- Loss computation (some cases)")
    
    # Custom autocast usage
    if torch.cuda.is_available():
        # Enable/disable autocast for specific operations
        x = torch.randn(32, 512, device='cuda')
        
        with autocast():
            # This will use FP16
            y1 = torch.matmul(x, x.transpose(-2, -1))
            
            # Disable autocast for specific operation
            with autocast(enabled=False):
                y2 = torch.matmul(x, x.transpose(-2, -1))  # Stays FP32
            
            # Continue with autocast
            y3 = torch.softmax(y1, dim=-1)
        
        print(f"Autocast enabled - y1 dtype: {y1.dtype}")
        print(f"Autocast disabled - y2 dtype: {y2.dtype}")
        print(f"Autocast re-enabled - y3 dtype: {y3.dtype}")

print("\n=== AMP with Different Model Types ===")

if amp_available and torch.cuda.is_available():
    # Convolutional model
    class ConvModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc = nn.Linear(64 * 8 * 8, 10)
        
        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    conv_model = ConvModel().cuda()
    conv_optimizer = torch.optim.Adam(conv_model.parameters())
    conv_scaler = GradScaler()
    
    # Test with image data
    image_data = torch.randn(16, 3, 32, 32, device='cuda')
    image_targets = torch.randint(0, 10, (16,), device='cuda')
    
    # AMP training with CNN
    conv_optimizer.zero_grad()
    with autocast():
        conv_output = conv_model(image_data)
        conv_loss = criterion(conv_output, image_targets)
    
    conv_scaler.scale(conv_loss).backward()
    conv_scaler.step(conv_optimizer)
    conv_scaler.update()
    
    print(f"CNN with AMP - Loss: {conv_loss.item():.6f}")

print("\n=== AMP Performance Monitoring ===")

if amp_available and torch.cuda.is_available():
    import time
    
    def benchmark_amp_vs_fp32(model, data, target, iterations=100):
        """Benchmark AMP vs FP32 training"""
        criterion_bench = nn.CrossEntropyLoss()
        
        # FP32 benchmark
        model.train()
        optimizer_fp32 = torch.optim.Adam(model.parameters())
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(iterations):
            optimizer_fp32.zero_grad()
            output = model(data)
            loss = criterion_bench(output, target)
            loss.backward()
            optimizer_fp32.step()
        
        torch.cuda.synchronize()
        fp32_time = time.time() - start_time
        
        # AMP benchmark
        optimizer_amp = torch.optim.Adam(model.parameters())
        scaler_bench = GradScaler()
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(iterations):
            optimizer_amp.zero_grad()
            with autocast():
                output = model(data)
                loss = criterion_bench(output, target)
            scaler_bench.scale(loss).backward()
            scaler_bench.step(optimizer_amp)
            scaler_bench.update()
        
        torch.cuda.synchronize()
        amp_time = time.time() - start_time
        
        return fp32_time, amp_time
    
    # Run benchmark
    fp32_time, amp_time = benchmark_amp_vs_fp32(model, input_data, target_data, 
                                               iterations=50)
    
    speedup = fp32_time / amp_time
    print(f"Performance benchmark (50 iterations):")
    print(f"  FP32 time: {fp32_time:.4f}s")
    print(f"  AMP time: {amp_time:.4f}s")
    print(f"  Speedup: {speedup:.2f}x")

print("\n=== Memory Usage Comparison ===")

if amp_available and torch.cuda.is_available():
    def measure_memory_usage(model, data, target, use_amp=False):
        """Measure memory usage with and without AMP"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        optimizer_mem = torch.optim.Adam(model.parameters())
        criterion_mem = nn.CrossEntropyLoss()
        
        if use_amp:
            scaler_mem = GradScaler()
        
        # Training step
        optimizer_mem.zero_grad()
        
        if use_amp:
            with autocast():
                output = model(data)
                loss = criterion_mem(output, target)
            scaler_mem.scale(loss).backward()
            scaler_mem.step(optimizer_mem)
            scaler_mem.update()
        else:
            output = model(data)
            loss = criterion_mem(output, target)
            loss.backward()
            optimizer_mem.step()
        
        memory_used = torch.cuda.max_memory_allocated() / 1e6  # MB
        return memory_used
    
    # Measure memory usage
    memory_fp32 = measure_memory_usage(model, input_data, target_data, use_amp=False)
    memory_amp = measure_memory_usage(model, input_data, target_data, use_amp=True)
    
    memory_savings = (memory_fp32 - memory_amp) / memory_fp32 * 100
    
    print(f"Memory usage comparison:")
    print(f"  FP32 memory: {memory_fp32:.2f} MB")
    print(f"  AMP memory: {memory_amp:.2f} MB")
    print(f"  Memory savings: {memory_savings:.1f}%")

print("\n=== AMP with Loss Scaling Edge Cases ===")

if amp_available and torch.cuda.is_available():
    # Monitor scaler behavior
    def monitor_scaler_behavior(scaler, num_steps=10):
        """Monitor how scaler behaves over training steps"""
        print("Scaler behavior monitoring:")
        
        for step in range(num_steps):
            # Simulate training step
            optimizer.zero_grad()
            
            # Simulate different loss magnitudes
            if step < 3:
                # Normal loss
                with autocast():
                    fake_loss = torch.randn(1, device='cuda', requires_grad=True) * 0.1
            elif step < 6:
                # Very small loss (potential underflow)
                with autocast():
                    fake_loss = torch.randn(1, device='cuda', requires_grad=True) * 1e-7
            else:
                # Large loss (potential overflow)
                with autocast():
                    fake_loss = torch.randn(1, device='cuda', requires_grad=True) * 1000
            
            # Backward pass
            scaler.scale(fake_loss).backward()
            
            # Check if step was skipped
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            
            step_skipped = scale_after < scale_before
            
            print(f"  Step {step}: scale_before={scale_before:.1e}, "
                  f"scale_after={scale_after:.1e}, skipped={step_skipped}")
    
    # Create fresh scaler for monitoring
    monitor_scaler = GradScaler()
    monitor_scaler_behavior(monitor_scaler)

print("\n=== AMP Best Practices ===")

print("Mixed Precision Training Guidelines:")
print("1. Always use autocast() for forward pass")
print("2. Use GradScaler for backward pass and optimizer steps")
print("3. Unscale gradients before clipping")
print("4. Monitor scaler behavior for overflow/underflow")
print("5. Test model accuracy with AMP vs FP32")
print("6. Use proper initialization to avoid early overflows")
print("7. Consider model architecture impacts on FP16")

print("\nPerformance Optimization:")
print("- Ensure GPU supports Tensor Cores (compute capability >= 7.0)")
print("- Use tensor sizes that are multiples of 8 for optimal performance")
print("- Avoid frequent autocast enable/disable switching")
print("- Use larger batch sizes to fully utilize Tensor Cores")
print("- Profile memory usage to maximize batch size")

print("\nCommon Issues:")
print("- Loss scaling underflow/overflow")
print("- Model accuracy degradation with FP16")
print("- Gradient clipping with scaled gradients")
print("- Mixed FP16/FP32 operations causing slowdowns")
print("- Memory savings not as expected")

print("\nDebugging Tips:")
print("- Monitor loss scale factor changes")
print("- Compare FP32 vs AMP accuracy regularly")
print("- Use loss_scale=2**20 for manual scaling if needed")
print("- Check for inf/nan in gradients")
print("- Profile both time and memory usage")

print("\n=== AMP Code Template ===")

amp_template = '''
# Complete AMP training template
from torch.cuda.amp import autocast, GradScaler

# Initialize
model = YourModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad()
        
        # Forward pass with autocast
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        
        # Backward pass with scaling
        scaler.scale(loss).backward()
        
        # Optional: gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
'''

print("AMP training template:")
print(amp_template)

print("\n=== Mixed Precision Complete ===") 