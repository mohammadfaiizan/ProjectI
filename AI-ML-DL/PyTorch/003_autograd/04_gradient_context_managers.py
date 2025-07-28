#!/usr/bin/env python3
"""PyTorch Gradient Context Managers - torch.no_grad(), torch.enable_grad()"""

import torch
import torch.nn as nn

print("=== Gradient Context Managers Overview ===")

print("PyTorch context managers for gradient control:")
print("1. torch.no_grad() - Disable gradient computation")
print("2. torch.enable_grad() - Enable gradient computation")
print("3. torch.set_grad_enabled() - Conditionally enable/disable")
print("4. torch.inference_mode() - More aggressive inference optimization")

print("\n=== torch.no_grad() ===")

# Basic no_grad usage
x = torch.tensor([1.0, 2.0], requires_grad=True)

# Normal operation (gradients tracked)
y_normal = x**2
print(f"Normal: y.requires_grad = {y_normal.requires_grad}")

# With no_grad context
with torch.no_grad():
    y_no_grad = x**2
    print(f"no_grad: y.requires_grad = {y_no_grad.requires_grad}")

# Outside context, gradients tracked again
y_after = x**2
print(f"After no_grad: y.requires_grad = {y_after.requires_grad}")

print("\n=== Inference with no_grad ===")

# Model inference example
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleModel()
input_data = torch.randn(100, 10)

# Training mode (gradients tracked)
model.train()
output_train = model(input_data)
print(f"Training output requires_grad: {output_train.requires_grad}")

# Inference mode with no_grad
model.eval()
with torch.no_grad():
    output_inference = model(input_data)
    print(f"Inference output requires_grad: {output_inference.requires_grad}")

print("\n=== Memory Savings with no_grad ===")

# Memory comparison
if torch.cuda.is_available():
    model_cuda = model.cuda()
    input_cuda = input_data.cuda()
    
    # With gradients
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    output_with_grad = model_cuda(input_cuda)
    memory_with_grad = torch.cuda.max_memory_allocated() / 1e6
    
    # Without gradients
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        output_no_grad = model_cuda(input_cuda)
    memory_no_grad = torch.cuda.max_memory_allocated() / 1e6
    
    print(f"Memory with gradients: {memory_with_grad:.2f} MB")
    print(f"Memory without gradients: {memory_no_grad:.2f} MB")
    print(f"Memory savings: {((memory_with_grad - memory_no_grad) / memory_with_grad * 100):.1f}%")

print("\n=== torch.enable_grad() ===")

# enable_grad context
with torch.no_grad():
    print("Inside no_grad context")
    
    # Normally gradients would be disabled
    x_inner = torch.tensor([1.0, 2.0], requires_grad=True)
    y_inner = x_inner**2
    print(f"  y_inner.requires_grad: {y_inner.requires_grad}")
    
    # Re-enable gradients within no_grad
    with torch.enable_grad():
        print("  Inside enable_grad (nested in no_grad)")
        z_inner = x_inner**3
        print(f"    z_inner.requires_grad: {z_inner.requires_grad}")
        
        # Can compute gradients
        z_inner.sum().backward()
        print(f"    x_inner.grad: {x_inner.grad}")

print("\n=== torch.set_grad_enabled() ===")

# Conditional gradient enabling
def conditional_computation(x, training=True):
    with torch.set_grad_enabled(training):
        y = x**2 + 2*x + 1
        return y

x_cond = torch.tensor([1.0, 2.0], requires_grad=True)

# Training mode
y_training = conditional_computation(x_cond, training=True)
print(f"Training mode: y.requires_grad = {y_training.requires_grad}")

# Inference mode
y_inference = conditional_computation(x_cond, training=False)
print(f"Inference mode: y.requires_grad = {y_inference.requires_grad}")

print("\n=== torch.inference_mode() ===")

# More aggressive inference optimization (PyTorch 1.9+)
if hasattr(torch, 'inference_mode'):
    x_inference = torch.tensor([1.0, 2.0], requires_grad=True)
    
    with torch.inference_mode():
        y_inference_mode = x_inference**2
        print(f"inference_mode: y.requires_grad = {y_inference_mode.requires_grad}")
    
    # Can't enable gradients inside inference_mode
    with torch.inference_mode():
        try:
            with torch.enable_grad():
                z_fail = x_inference**3
        except RuntimeError as e:
            print(f"Error enabling grad in inference_mode: {str(e)[:50]}...")

print("\n=== Context Manager Nesting ===")

# Complex nesting scenarios
x_nest = torch.tensor([1.0, 2.0], requires_grad=True)

# Default state
y1 = x_nest**2
print(f"Default: y1.requires_grad = {y1.requires_grad}")

with torch.no_grad():
    # Disabled
    y2 = x_nest**2
    print(f"no_grad: y2.requires_grad = {y2.requires_grad}")
    
    with torch.enable_grad():
        # Re-enabled
        y3 = x_nest**2
        print(f"enable_grad in no_grad: y3.requires_grad = {y3.requires_grad}")
        
        with torch.no_grad():
            # Disabled again
            y4 = x_nest**2
            print(f"no_grad in enable_grad: y4.requires_grad = {y4.requires_grad}")

print("\n=== Performance Impact ===")

# Benchmark context manager overhead
import time

def benchmark_context_managers(iterations=10000):
    x = torch.randn(100, 100, requires_grad=True)
    
    # No context manager
    start = time.time()
    for _ in range(iterations):
        y = x.sum()
    time_normal = time.time() - start
    
    # With no_grad
    start = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            y = x.sum()
    time_no_grad = time.time() - start
    
    # With set_grad_enabled
    start = time.time()
    for _ in range(iterations):
        with torch.set_grad_enabled(False):
            y = x.sum()
    time_set_grad = time.time() - start
    
    return time_normal, time_no_grad, time_set_grad

times = benchmark_context_managers(1000)
print(f"Normal: {times[0]:.4f}s")
print(f"no_grad: {times[1]:.4f}s")
print(f"set_grad_enabled: {times[2]:.4f}s")

print("\n=== Training Loop Integration ===")

# Typical training/validation pattern
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()  # Set training mode
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Forward pass (gradients enabled by default)
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion):
    model.eval()  # Set evaluation mode
    total_loss = 0
    correct = 0
    
    # Disable gradients for validation
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    accuracy = 100.0 * correct / len(dataloader.dataset)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy

# Simulate training/validation
class FakeDataLoader:
    def __init__(self, batch_size, num_batches, input_size, num_classes):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.input_size = input_size
        self.num_classes = num_classes
        self.dataset = type('Dataset', (), {'__len__': lambda: num_batches * batch_size})()
    
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        for _ in range(self.num_batches):
            data = torch.randn(self.batch_size, self.input_size)
            target = torch.randint(0, self.num_classes, (self.batch_size,))
            yield data, target

# Create fake data loaders
train_loader = FakeDataLoader(32, 10, 10, 5)
val_loader = FakeDataLoader(32, 5, 10, 5)

# Create model and training components
model_train = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 5)
)
optimizer = torch.optim.Adam(model_train.parameters())
criterion = nn.CrossEntropyLoss()

# Training and validation
train_loss = train_epoch(model_train, train_loader, optimizer, criterion)
val_loss, val_acc = validate_epoch(model_train, val_loader, criterion)

print(f"Training completed:")
print(f"  Train loss: {train_loss:.4f}")
print(f"  Val loss: {val_loss:.4f}")
print(f"  Val accuracy: {val_acc:.2f}%")

print("\n=== Gradient Accumulation with Context ===")

# Gradient accumulation with context managers
def train_with_accumulation(model, data_loader, optimizer, criterion, 
                          accumulation_steps=4):
    model.train()
    optimizer.zero_grad()
    
    for batch_idx, (data, target) in enumerate(data_loader):
        # Forward pass (always with gradients in training)
        output = model(data)
        loss = criterion(output, target) / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

print("Gradient accumulation with context managers demonstrated")

print("\n=== Context Manager Best Practices ===")

print("Context Manager Guidelines:")
print("1. Use torch.no_grad() for all inference/validation")
print("2. Use torch.inference_mode() for pure inference (no backprop)")
print("3. Use torch.set_grad_enabled() for conditional logic")
print("4. Nest context managers carefully")
print("5. Remember context managers don't change requires_grad permanently")

print("\nPerformance Tips:")
print("- torch.no_grad() saves memory and computation")
print("- torch.inference_mode() is faster than no_grad")
print("- Context managers have minimal overhead")
print("- Use in validation loops and inference")

print("\nCommon Patterns:")
print("- Training: no context manager needed")
print("- Validation: with torch.no_grad()")
print("- Inference: with torch.inference_mode()")
print("- Conditional: with torch.set_grad_enabled(training)")

print("\n=== Context Managers Complete ===") 