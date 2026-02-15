#!/usr/bin/env python3
"""PyTorch Device Movement - Moving models and data between devices"""

import torch
import torch.nn as nn

print("=== Device Detection and Management ===")

# Check available devices
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")

# Check for MPS (Apple Silicon)
mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
print(f"MPS available: {mps_available}")

# Set default device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif mps_available:
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")

print("\n=== Basic Tensor Device Operations ===")

# Create tensors on different devices
cpu_tensor = torch.randn(3, 4)
print(f"CPU tensor device: {cpu_tensor.device}")

# Move tensor to device
if torch.cuda.is_available():
    gpu_tensor = cpu_tensor.cuda()
    print(f"GPU tensor device: {gpu_tensor.device}")
    
    # Alternative method
    gpu_tensor2 = cpu_tensor.to('cuda')
    print(f"GPU tensor2 device: {gpu_tensor2.device}")
    
    # Move back to CPU
    cpu_tensor2 = gpu_tensor.cpu()
    print(f"Back to CPU device: {cpu_tensor2.device}")

# Direct creation on device
if torch.cuda.is_available():
    direct_gpu = torch.randn(3, 4, device='cuda')
    print(f"Direct GPU creation: {direct_gpu.device}")

# Create on specific device
device_tensor = torch.randn(3, 4, device=device)
print(f"Device tensor: {device_tensor.device}")

print("\n=== Model Device Movement ===")

# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

model = SimpleModel()
print(f"Model parameters on CPU:")
for name, param in list(model.named_parameters())[:2]:
    print(f"  {name}: {param.device}")

# Move model to device
model = model.to(device)
print(f"Model parameters after .to({device}):")
for name, param in list(model.named_parameters())[:2]:
    print(f"  {name}: {param.device}")

print("\n=== Data and Model Compatibility ===")

# Input data must be on same device as model
input_data = torch.randn(32, 784, device=device)
print(f"Input data device: {input_data.device}")

# Forward pass
with torch.no_grad():
    output = model(input_data)
    print(f"Output device: {output.device}")

# Error demonstration (if CUDA available)
if torch.cuda.is_available():
    try:
        cpu_input = torch.randn(32, 784)  # CPU tensor
        # This would cause an error
        # output_error = model(cpu_input)
        print("Would cause device mismatch error if uncommented")
    except RuntimeError as e:
        print(f"Device mismatch error: {e}")

print("\n=== Multi-GPU Operations ===")

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"Multiple GPUs available: {torch.cuda.device_count()}")
    
    # Specify different GPU
    gpu0_tensor = torch.randn(3, 4, device='cuda:0')
    
    if torch.cuda.device_count() > 1:
        gpu1_tensor = torch.randn(3, 4, device='cuda:1')
        print(f"GPU 0 tensor: {gpu0_tensor.device}")
        print(f"GPU 1 tensor: {gpu1_tensor.device}")
        
        # Move tensor between GPUs
        gpu0_to_gpu1 = gpu0_tensor.to('cuda:1')
        print(f"Moved to GPU 1: {gpu0_to_gpu1.device}")
    
    # DataParallel for multi-GPU training
    if torch.cuda.device_count() > 1:
        parallel_model = nn.DataParallel(model)
        print(f"DataParallel model device: {next(parallel_model.parameters()).device}")
        
        # Multi-GPU forward pass
        multi_gpu_output = parallel_model(input_data)
        print(f"Multi-GPU output device: {multi_gpu_output.device}")

else:
    print("Single GPU or CPU only")

print("\n=== Device Context Management ===")

if torch.cuda.is_available():
    # Set current device
    original_device = torch.cuda.current_device()
    
    # Context manager for device
    with torch.cuda.device(0):
        context_tensor = torch.randn(3, 4, device='cuda')
        print(f"Context tensor device: {context_tensor.device}")
    
    print(f"Current device after context: {torch.cuda.current_device()}")

print("\n=== Memory Management ===")

if torch.cuda.is_available():
    # Check GPU memory
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
    
    # Create large tensor
    large_tensor = torch.randn(1000, 1000, device='cuda')
    print(f"After large tensor - allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    
    # Clear memory
    del large_tensor
    torch.cuda.empty_cache()
    print(f"After cleanup - allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")

print("\n=== Efficient Device Movement ===")

# Pinned memory for faster transfers
if torch.cuda.is_available():
    # Regular tensor transfer
    regular_tensor = torch.randn(1000, 1000)
    
    import time
    start_time = time.time()
    gpu_regular = regular_tensor.cuda()
    regular_time = time.time() - start_time
    
    # Pinned memory transfer
    pinned_tensor = torch.randn(1000, 1000).pin_memory()
    
    start_time = time.time()
    gpu_pinned = pinned_tensor.cuda(non_blocking=True)
    torch.cuda.synchronize()  # Wait for transfer to complete
    pinned_time = time.time() - start_time
    
    print(f"Regular transfer time: {regular_time:.6f}s")
    print(f"Pinned transfer time: {pinned_time:.6f}s")
    print(f"Speedup: {regular_time / pinned_time:.2f}x")

print("\n=== Device-Agnostic Code ===")

def create_model_and_data(device):
    """Create model and data on specified device"""
    model = SimpleModel().to(device)
    data = torch.randn(32, 784, device=device)
    target = torch.randint(0, 10, (32,), device=device)
    return model, data, target

def train_step(model, data, target, optimizer, criterion):
    """Device-agnostic training step"""
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()

# Test device-agnostic code
test_model, test_data, test_target = create_model_and_data(device)
test_optimizer = torch.optim.Adam(test_model.parameters())
test_criterion = nn.CrossEntropyLoss()

loss = train_step(test_model, test_data, test_target, test_optimizer, test_criterion)
print(f"Training step completed on {device}, loss: {loss:.4f}")

print("\n=== Automatic Mixed Precision (AMP) ===")

if torch.cuda.is_available():
    from torch.cuda.amp import autocast, GradScaler
    
    # AMP training setup
    amp_model = SimpleModel().cuda()
    amp_optimizer = torch.optim.Adam(amp_model.parameters())
    scaler = GradScaler()
    
    # AMP training step
    def amp_train_step(model, data, target, optimizer, criterion, scaler):
        optimizer.zero_grad()
        
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        return loss.item()
    
    # Test AMP training
    amp_data = torch.randn(32, 784, device='cuda')
    amp_target = torch.randint(0, 10, (32,), device='cuda')
    
    amp_loss = amp_train_step(amp_model, amp_data, amp_target, 
                             amp_optimizer, test_criterion, scaler)
    print(f"AMP training step completed, loss: {amp_loss:.4f}")

print("\n=== Device Movement Utilities ===")

def move_to_device(obj, device):
    """Recursively move object to device"""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
    else:
        return obj

# Test utility function
mixed_data = {
    'input': torch.randn(10, 5),
    'target': torch.randint(0, 2, (10,)),
    'metadata': ['sample1', 'sample2'],
    'nested': {
        'features': torch.randn(10, 3),
        'labels': torch.tensor([1, 0, 1])
    }
}

moved_data = move_to_device(mixed_data, device)
print(f"Original input device: {mixed_data['input'].device}")
print(f"Moved input device: {moved_data['input'].device}")
print(f"Metadata preserved: {moved_data['metadata']}")

print("\n=== Model Deployment Considerations ===")

def prepare_model_for_inference(model, device='cpu'):
    """Prepare model for inference deployment"""
    # Move to target device
    model = model.to(device)
    
    # Set to evaluation mode
    model.eval()
    
    # Compile model for inference (if supported)
    try:
        if hasattr(torch, 'compile'):
            model = torch.compile(model)
            print("Model compiled for optimization")
    except:
        print("Model compilation not available")
    
    return model

# Prepare for CPU inference
inference_model = prepare_model_for_inference(test_model, device='cpu')
print(f"Inference model device: {next(inference_model.parameters()).device}")

# Test inference
with torch.no_grad():
    cpu_input = torch.randn(1, 784)
    inference_output = inference_model(cpu_input)
    print(f"Inference output shape: {inference_output.shape}")

print("\n=== Error Handling and Debugging ===")

def safe_to_device(tensor, device, fallback_device='cpu'):
    """Safely move tensor to device with fallback"""
    try:
        return tensor.to(device)
    except RuntimeError as e:
        print(f"Failed to move to {device}: {e}")
        print(f"Falling back to {fallback_device}")
        return tensor.to(fallback_device)

# Test safe device movement
test_tensor = torch.randn(3, 3)
safe_tensor = safe_to_device(test_tensor, device)
print(f"Safe tensor device: {safe_tensor.device}")

def debug_device_state(model, data_batch=None):
    """Debug device state of model and data"""
    print("=== Device Debug Information ===")
    
    # Model device info
    model_devices = set()
    for name, param in model.named_parameters():
        model_devices.add(str(param.device))
    
    print(f"Model devices: {model_devices}")
    
    if len(model_devices) > 1:
        print("WARNING: Model parameters on multiple devices!")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.device}")
    
    # Data device info
    if data_batch is not None:
        if isinstance(data_batch, torch.Tensor):
            print(f"Data device: {data_batch.device}")
        elif isinstance(data_batch, (list, tuple)):
            for i, item in enumerate(data_batch):
                if isinstance(item, torch.Tensor):
                    print(f"Data[{i}] device: {item.device}")

# Debug current state
debug_device_state(test_model, test_data)

print("\n=== Performance Optimization ===")

if torch.cuda.is_available():
    # Benchmark device transfer speeds
    def benchmark_transfer(tensor_size, iterations=10):
        cpu_tensor = torch.randn(*tensor_size)
        
        # Warm up
        for _ in range(5):
            gpu_tensor = cpu_tensor.cuda()
            cpu_back = gpu_tensor.cpu()
        
        # Benchmark CPU to GPU
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(iterations):
            gpu_tensor = cpu_tensor.cuda()
            torch.cuda.synchronize()
        cpu_to_gpu_time = (time.time() - start_time) / iterations
        
        # Benchmark GPU to CPU
        start_time = time.time()
        for _ in range(iterations):
            cpu_back = gpu_tensor.cpu()
            torch.cuda.synchronize()
        gpu_to_cpu_time = (time.time() - start_time) / iterations
        
        tensor_mb = (cpu_tensor.numel() * cpu_tensor.element_size()) / 1e6
        
        print(f"Tensor size: {tensor_size} ({tensor_mb:.2f} MB)")
        print(f"CPU->GPU: {cpu_to_gpu_time:.6f}s ({tensor_mb/cpu_to_gpu_time:.1f} MB/s)")
        print(f"GPU->CPU: {gpu_to_cpu_time:.6f}s ({tensor_mb/gpu_to_cpu_time:.1f} MB/s)")
    
    # Benchmark different sizes
    sizes = [(1000, 1000), (2000, 2000)]
    for size in sizes:
        benchmark_transfer(size)

print("\n=== Device Movement Best Practices ===")

print("Device Management Guidelines:")
print("1. Always check device availability before use")
print("2. Keep model and data on the same device")
print("3. Use .to(device) for explicit device movement")
print("4. Use pinned memory for faster GPU transfers")
print("5. Clear GPU memory with del and torch.cuda.empty_cache()")
print("6. Use device-agnostic code for portability")
print("7. Monitor GPU memory usage during training")

print("\nPerformance Tips:")
print("- Create tensors directly on target device when possible")
print("- Use non_blocking=True with pinned memory")
print("- Batch device transfers instead of moving individual tensors")
print("- Consider mixed precision training for memory efficiency")
print("- Use torch.cuda.device() context for multi-GPU setups")

print("\nCommon Pitfalls:")
print("- Forgetting to move model to GPU")
print("- Moving data to wrong device")
print("- Not synchronizing CUDA operations when timing")
print("- Memory leaks from not clearing GPU memory")
print("- Using .cuda() instead of .to(device) for portability")

print("\n=== Device Movement Complete ===") 