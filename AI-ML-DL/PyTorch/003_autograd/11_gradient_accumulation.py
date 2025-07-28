#!/usr/bin/env python3
"""PyTorch Gradient Accumulation - Accumulating gradients across batches"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== Gradient Accumulation Overview ===")

print("Gradient accumulation enables:")
print("1. Larger effective batch sizes")
print("2. Memory-efficient training")
print("3. Stable training on limited hardware")
print("4. Consistent gradient statistics")
print("5. Better convergence for some models")

print("\n=== Basic Gradient Accumulation ===")

# Simple model for demonstration
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = SimpleModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Without gradient accumulation - single large batch
large_batch_data = torch.randn(32, 10)
large_batch_target = torch.randint(0, 5, (32,))

optimizer.zero_grad()
output_large = model(large_batch_data)
loss_large = criterion(output_large, large_batch_target)
loss_large.backward()
grad_large = {name: param.grad.clone() for name, param in model.named_parameters()}
optimizer.step()

print(f"Large batch loss: {loss_large.item():.4f}")

print("\n=== Manual Gradient Accumulation ===")

# Reset model state
model = SimpleModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Gradient accumulation - multiple small batches
accumulation_steps = 4
batch_size = 8  # 4 * 8 = 32 total (same as large batch)

optimizer.zero_grad()
accumulated_loss = 0

for step in range(accumulation_steps):
    # Get small batch
    start_idx = step * batch_size
    end_idx = start_idx + batch_size
    small_batch_data = large_batch_data[start_idx:end_idx]
    small_batch_target = large_batch_target[start_idx:end_idx]
    
    # Forward pass
    output_small = model(small_batch_data)
    loss_small = criterion(output_small, small_batch_target)
    
    # Scale loss by accumulation steps
    scaled_loss = loss_small / accumulation_steps
    
    # Backward pass (gradients accumulate)
    scaled_loss.backward()
    
    accumulated_loss += loss_small.item()
    print(f"  Step {step}: loss = {loss_small.item():.4f}")

# Update weights after accumulation
optimizer.step()

print(f"Accumulated average loss: {accumulated_loss / accumulation_steps:.4f}")

# Compare accumulated gradients with large batch gradients
grad_accumulated = {name: param.grad.clone() for name, param in model.named_parameters()}

# Check if gradients are similar
for name in grad_large.keys():
    grad_diff = torch.abs(grad_large[name] - grad_accumulated[name]).max()
    print(f"Max gradient difference for {name}: {grad_diff:.6f}")

print("\n=== Gradient Accumulation Function ===")

def train_with_accumulation(model, data_loader, optimizer, criterion, 
                          accumulation_steps=4, max_grad_norm=None):
    """Training function with gradient accumulation"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (data, target) in enumerate(data_loader):
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Scale loss by accumulation steps
        scaled_loss = loss / accumulation_steps
        
        # Backward pass
        scaled_loss.backward()
        
        total_loss += loss.item()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            # Optional gradient clipping
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            optimizer.zero_grad()
            num_batches += 1
    
    # Handle remaining gradients if batch count not divisible by accumulation_steps
    if (batch_idx + 1) % accumulation_steps != 0:
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        num_batches += 1
    
    return total_loss / len(data_loader)

# Test accumulation function
class FakeDataLoader:
    def __init__(self, batch_size, num_batches):
        self.batch_size = batch_size
        self.num_batches = num_batches
    
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        for _ in range(self.num_batches):
            data = torch.randn(self.batch_size, 10)
            target = torch.randint(0, 5, (self.batch_size,))
            yield data, target

fake_loader = FakeDataLoader(batch_size=8, num_batches=12)
model_test = SimpleModel()
optimizer_test = torch.optim.Adam(model_test.parameters(), lr=0.001)

avg_loss = train_with_accumulation(
    model_test, fake_loader, optimizer_test, criterion, 
    accumulation_steps=3, max_grad_norm=1.0
)

print(f"Training with accumulation completed, avg loss: {avg_loss:.4f}")

print("\n=== Memory Comparison ===")

# Compare memory usage with and without accumulation
if torch.cuda.is_available():
    def measure_memory_usage(model, batch_size, use_accumulation=False):
        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        if use_accumulation:
            # Multiple small batches
            accumulation_steps = 4
            small_batch_size = batch_size // accumulation_steps
            
            optimizer.zero_grad()
            for _ in range(accumulation_steps):
                data = torch.randn(small_batch_size, 10, device='cuda')
                target = torch.randint(0, 5, (small_batch_size,), device='cuda')
                
                output = model(data)
                loss = criterion(output, target) / accumulation_steps
                loss.backward()
            
            optimizer.step()
        else:
            # Single large batch
            data = torch.randn(batch_size, 10, device='cuda')
            target = torch.randint(0, 5, (batch_size,), device='cuda')
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        return torch.cuda.max_memory_allocated() / 1e6  # MB
    
    model_memory = SimpleModel()
    
    memory_large_batch = measure_memory_usage(model_memory, 64, use_accumulation=False)
    memory_accumulated = measure_memory_usage(model_memory, 64, use_accumulation=True)
    
    print(f"Memory usage - Large batch: {memory_large_batch:.2f} MB")
    print(f"Memory usage - Accumulated: {memory_accumulated:.2f} MB")
    print(f"Memory savings: {((memory_large_batch - memory_accumulated) / memory_large_batch * 100):.1f}%")

print("\n=== Dynamic Accumulation ===")

class DynamicAccumulator:
    """Dynamic gradient accumulation based on available memory"""
    
    def __init__(self, model, optimizer, target_batch_size=32, min_batch_size=4):
        self.model = model
        self.optimizer = optimizer
        self.target_batch_size = target_batch_size
        self.min_batch_size = min_batch_size
        self.accumulated_samples = 0
        self.accumulated_loss = 0
        
    def step(self, data, target, criterion):
        """Process a batch with dynamic accumulation"""
        current_batch_size = data.size(0)
        
        # Forward pass
        output = self.model(data)
        loss = criterion(output, target)
        
        # Scale loss by target batch size
        scaled_loss = loss * (current_batch_size / self.target_batch_size)
        scaled_loss.backward()
        
        self.accumulated_samples += current_batch_size
        self.accumulated_loss += loss.item()
        
        # Update when we've accumulated enough samples
        if self.accumulated_samples >= self.target_batch_size:
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            avg_loss = self.accumulated_loss * current_batch_size / self.accumulated_samples
            self.accumulated_samples = 0
            self.accumulated_loss = 0
            
            return avg_loss
        
        return None
    
    def finalize(self):
        """Final update for remaining accumulated gradients"""
        if self.accumulated_samples > 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.accumulated_samples = 0
            self.accumulated_loss = 0

# Test dynamic accumulator
model_dynamic = SimpleModel()
optimizer_dynamic = torch.optim.Adam(model_dynamic.parameters())
accumulator = DynamicAccumulator(model_dynamic, optimizer_dynamic, target_batch_size=24)

# Simulate varying batch sizes
batch_sizes = [8, 6, 10, 4, 8]
for i, bs in enumerate(batch_sizes):
    data = torch.randn(bs, 10)
    target = torch.randint(0, 5, (bs,))
    
    loss = accumulator.step(data, target, criterion)
    if loss is not None:
        print(f"Update triggered at step {i}, loss: {loss:.4f}")

accumulator.finalize()
print("Dynamic accumulation completed")

print("\n=== Gradient Accumulation with Mixed Precision ===")

# Gradient accumulation with AMP
if hasattr(torch.cuda, 'amp'):
    from torch.cuda.amp import autocast, GradScaler
    
    def train_with_amp_accumulation(model, data_loader, optimizer, criterion, 
                                  accumulation_steps=4):
        """Training with AMP and gradient accumulation"""
        model.train()
        scaler = GradScaler()
        total_loss = 0
        
        optimizer.zero_grad()
        
        for batch_idx, (data, target) in enumerate(data_loader):
            # Forward pass with autocast
            with autocast():
                output = model(data)
                loss = criterion(output, target) / accumulation_steps
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
            
            total_loss += loss.item() * accumulation_steps
            
            # Update every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
        return total_loss / len(data_loader)
    
    print("AMP + gradient accumulation function defined")

print("\n=== Monitoring Gradient Accumulation ===")

def monitor_gradient_accumulation(model, accumulation_steps=4):
    """Monitor gradient statistics during accumulation"""
    
    class GradientMonitor:
        def __init__(self):
            self.step_gradients = []
        
        def record_gradients(self, step):
            step_grads = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    step_grads[name] = {
                        'norm': param.grad.norm().item(),
                        'mean': param.grad.mean().item(),
                        'std': param.grad.std().item()
                    }
            self.step_gradients.append(step_grads)
        
        def analyze(self):
            if not self.step_gradients:
                return
            
            print("Gradient accumulation analysis:")
            for param_name in self.step_gradients[0].keys():
                norms = [step[param_name]['norm'] for step in self.step_gradients]
                print(f"  {param_name}:")
                print(f"    Gradient norms: {norms}")
                print(f"    Final norm: {norms[-1]:.6f}")
    
    monitor = GradientMonitor()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    optimizer.zero_grad()
    
    for step in range(accumulation_steps):
        # Forward pass
        data = torch.randn(8, 10)
        target = torch.randint(0, 5, (8,))
        
        output = model(data)
        loss = criterion(output, target) / accumulation_steps
        loss.backward()
        
        # Record gradients after each accumulation step
        monitor.record_gradients(step)
    
    monitor.analyze()
    optimizer.step()

# Test gradient monitoring
model_monitor = SimpleModel()
monitor_gradient_accumulation(model_monitor, accumulation_steps=3)

print("\n=== Gradient Accumulation Best Practices ===")

print("Gradient Accumulation Guidelines:")
print("1. Scale loss by accumulation_steps to maintain gradient magnitude")
print("2. Clear gradients only after optimizer.step()")
print("3. Apply gradient clipping after accumulation, before step")
print("4. Handle remaining batches when total isn't divisible")
print("5. Use with mixed precision for additional memory savings")
print("6. Monitor gradient statistics during accumulation")
print("7. Consider learning rate scaling with effective batch size")

print("\nMemory Optimization:")
print("- Accumulation reduces peak memory usage")
print("- Enables training larger models on limited hardware")
print("- Allows consistent batch statistics across different hardware")
print("- Works well with gradient checkpointing")

print("\nCommon Pitfalls:")
print("- Forgetting to scale loss by accumulation steps")
print("- Calling optimizer.step() too frequently")
print("- Not handling partial batches at the end")
print("- Inconsistent gradient statistics")
print("- Memory leaks from retained computation graphs")

print("\nUse Cases:")
print("- Large language models")
print("- High-resolution image processing")
print("- Limited GPU memory scenarios")
print("- Distributed training with small local batches")
print("- Consistent training across different hardware")

print("\n=== Gradient Accumulation Complete ===") 