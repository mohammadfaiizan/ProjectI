import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Optional, Dict, List
import time

# Basic Gradient Accumulation Trainer
class GradientAccumulationTrainer:
    """Trainer with gradient accumulation support"""
    
    def __init__(self, model, optimizer, criterion, device='cuda', accumulation_steps=4):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.accumulation_steps = accumulation_steps
        
        # Gradient scaler for mixed precision
        self.scaler = GradScaler()
        
        # Metrics tracking
        self.step_count = 0
        self.accumulated_loss = 0.0
        
    def train_step(self, batch, use_amp=False):
        """Single training step with gradient accumulation"""
        data, targets = batch
        data, targets = data.to(self.device), targets.to(self.device)
        
        # Scale loss by accumulation steps to maintain same effective learning rate
        loss_scale = 1.0 / self.accumulation_steps
        
        if use_amp:
            # Mixed precision forward pass
            with autocast():
                outputs = self.model(data)
                loss = self.criterion(outputs, targets) * loss_scale
            
            # Scaled backward pass
            self.scaler.scale(loss).backward()
        else:
            # Standard forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, targets) * loss_scale
            
            # Backward pass
            loss.backward()
        
        # Accumulate loss for logging
        self.accumulated_loss += loss.item() / loss_scale
        self.step_count += 1
        
        # Update parameters every accumulation_steps
        if self.step_count % self.accumulation_steps == 0:
            if use_amp:
                # Unscale gradients and step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            # Return averaged loss
            avg_loss = self.accumulated_loss / self.accumulation_steps
            self.accumulated_loss = 0.0
            
            return avg_loss, True  # True indicates parameter update occurred
        
        return self.accumulated_loss / (self.step_count % self.accumulation_steps), False
    
    def train_epoch(self, dataloader, use_amp=False, log_interval=100):
        """Train for one epoch with gradient accumulation"""
        self.model.train()
        total_loss = 0.0
        num_updates = 0
        
        for batch_idx, batch in enumerate(dataloader):
            loss, updated = self.train_step(batch, use_amp)
            
            if updated:
                total_loss += loss
                num_updates += 1
                
                if num_updates % log_interval == 0:
                    print(f'Batch {batch_idx}, Update {num_updates}, Loss: {loss:.4f}')
        
        # Handle remaining accumulated gradients
        if self.step_count % self.accumulation_steps != 0:
            if use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            final_loss = self.accumulated_loss / (self.step_count % self.accumulation_steps)
            total_loss += final_loss
            num_updates += 1
            self.accumulated_loss = 0.0
        
        return total_loss / max(num_updates, 1)

# Dynamic Gradient Accumulation
class DynamicGradientAccumulator:
    """Gradient accumulation with dynamic accumulation steps"""
    
    def __init__(self, model, optimizer, criterion, device='cuda', 
                 target_batch_size=512, base_batch_size=32):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Calculate dynamic accumulation steps
        self.target_batch_size = target_batch_size
        self.base_batch_size = base_batch_size
        self.accumulation_steps = target_batch_size // base_batch_size
        
        print(f"Dynamic accumulation steps: {self.accumulation_steps}")
        print(f"Effective batch size: {self.accumulation_steps * base_batch_size}")
        
        self.scaler = GradScaler()
        self.reset_accumulation()
    
    def reset_accumulation(self):
        """Reset accumulation counters"""
        self.accumulated_samples = 0
        self.accumulated_loss = 0.0
        self.step_count = 0
    
    def accumulate_gradients(self, batch, use_amp=False):
        """Accumulate gradients from a batch"""
        data, targets = batch
        data, targets = data.to(self.device), targets.to(self.device)
        
        batch_size = data.size(0)
        loss_scale = 1.0 / self.accumulation_steps
        
        if use_amp:
            with autocast():
                outputs = self.model(data)
                loss = self.criterion(outputs, targets) * loss_scale
            
            self.scaler.scale(loss).backward()
        else:
            outputs = self.model(data)
            loss = self.criterion(outputs, targets) * loss_scale
            loss.backward()
        
        self.accumulated_samples += batch_size
        self.accumulated_loss += loss.item() / loss_scale
        self.step_count += 1
        
        return loss.item() / loss_scale
    
    def should_update(self):
        """Check if we should update parameters"""
        return self.step_count >= self.accumulation_steps
    
    def update_parameters(self, use_amp=False):
        """Update model parameters"""
        if use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        # Calculate average loss
        avg_loss = self.accumulated_loss / self.step_count
        
        # Reset for next accumulation cycle
        self.reset_accumulation()
        
        return avg_loss

# Memory-Efficient Gradient Accumulation
class MemoryEfficientAccumulator:
    """Memory-efficient gradient accumulation with gradient checkpointing"""
    
    def __init__(self, model, optimizer, criterion, device='cuda', 
                 accumulation_steps=4, use_checkpointing=True):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.use_checkpointing = use_checkpointing
        
        # Enable gradient checkpointing if requested
        if use_checkpointing:
            self.enable_gradient_checkpointing()
        
        self.scaler = GradScaler()
        self.accumulated_gradients = None
        
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        def checkpoint_forward(module):
            if hasattr(module, 'forward'):
                original_forward = module.forward
                
                def checkpointed_forward(*args, **kwargs):
                    if self.model.training:
                        return torch.utils.checkpoint.checkpoint(original_forward, *args, **kwargs)
                    else:
                        return original_forward(*args, **kwargs)
                
                module.forward = checkpointed_forward
        
        # Apply checkpointing to all modules (simplified example)
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.LSTM, nn.GRU)):
                checkpoint_forward(module)
    
    def accumulate_step(self, batch, use_amp=False):
        """Accumulate gradients from one batch"""
        data, targets = batch
        data, targets = data.to(self.device), targets.to(self.device)
        
        # Scale loss to maintain effective learning rate
        loss_scale = 1.0 / self.accumulation_steps
        
        if use_amp:
            with autocast():
                outputs = self.model(data)
                loss = self.criterion(outputs, targets) * loss_scale
            
            self.scaler.scale(loss).backward()
        else:
            outputs = self.model(data)
            loss = self.criterion(outputs, targets) * loss_scale
            loss.backward()
        
        return loss.item() / loss_scale
    
    def sync_gradients(self):
        """Synchronize accumulated gradients (useful for distributed training)"""
        if torch.distributed.is_initialized():
            for param in self.model.parameters():
                if param.grad is not None:
                    torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
                    param.grad.data /= torch.distributed.get_world_size()
    
    def update_model(self, use_amp=False, clip_grad_norm=None):
        """Update model parameters with optional gradient clipping"""
        
        # Gradient clipping
        if clip_grad_norm is not None:
            if use_amp:
                self.scaler.unscale_(self.optimizer)
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
        
        # Parameter update
        if use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()

# Advanced Gradient Accumulation with Multiple Optimizers
class MultiOptimizerAccumulator:
    """Gradient accumulation with multiple optimizers for different model parts"""
    
    def __init__(self, model, optimizers_config, criterion, device='cuda', accumulation_steps=4):
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
        self.accumulation_steps = accumulation_steps
        
        # Setup multiple optimizers
        self.optimizers = {}
        self.scalers = {}
        
        for name, config in optimizers_config.items():
            params = config['params']
            optimizer_class = config['optimizer']
            optimizer_kwargs = config.get('kwargs', {})
            
            self.optimizers[name] = optimizer_class(params, **optimizer_kwargs)
            self.scalers[name] = GradScaler()
        
        self.step_count = 0
        
    def accumulate_and_update(self, batch, use_amp=False):
        """Accumulate gradients and update if needed"""
        data, targets = batch
        data, targets = data.to(self.device), targets.to(self.device)
        
        loss_scale = 1.0 / self.accumulation_steps
        
        if use_amp:
            with autocast():
                outputs = self.model(data)
                loss = self.criterion(outputs, targets) * loss_scale
            
            # Scale and backward
            loss_sum = sum(scaler.scale(loss) for scaler in self.scalers.values()) / len(self.scalers)
            loss_sum.backward()
        else:
            outputs = self.model(data)
            loss = self.criterion(outputs, targets) * loss_scale
            loss.backward()
        
        self.step_count += 1
        
        # Update parameters if accumulation is complete
        if self.step_count % self.accumulation_steps == 0:
            for name, optimizer in self.optimizers.items():
                if use_amp:
                    self.scalers[name].step(optimizer)
                    self.scalers[name].update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
            
            return loss.item() / loss_scale, True
        
        return loss.item() / loss_scale, False

# Adaptive Gradient Accumulation
class AdaptiveGradientAccumulator:
    """Gradient accumulation with adaptive accumulation steps based on memory usage"""
    
    def __init__(self, model, optimizer, criterion, device='cuda', 
                 max_memory_gb=8, min_accumulation_steps=2, max_accumulation_steps=16):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        self.max_memory_gb = max_memory_gb
        self.min_accumulation_steps = min_accumulation_steps
        self.max_accumulation_steps = max_accumulation_steps
        
        # Current accumulation steps (starts at minimum)
        self.accumulation_steps = min_accumulation_steps
        
        self.scaler = GradScaler()
        self.step_count = 0
        self.accumulated_loss = 0.0
        
    def get_memory_usage_gb(self):
        """Get current GPU memory usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(self.device) / 1024**3
        return 0
    
    def adapt_accumulation_steps(self):
        """Adapt accumulation steps based on memory usage"""
        current_memory = self.get_memory_usage_gb()
        memory_ratio = current_memory / self.max_memory_gb
        
        if memory_ratio > 0.8 and self.accumulation_steps < self.max_accumulation_steps:
            # Increase accumulation steps to reduce memory usage
            self.accumulation_steps = min(self.accumulation_steps * 2, self.max_accumulation_steps)
            print(f"Increased accumulation steps to {self.accumulation_steps} (Memory: {current_memory:.2f}GB)")
        elif memory_ratio < 0.4 and self.accumulation_steps > self.min_accumulation_steps:
            # Decrease accumulation steps to use more memory efficiently
            self.accumulation_steps = max(self.accumulation_steps // 2, self.min_accumulation_steps)
            print(f"Decreased accumulation steps to {self.accumulation_steps} (Memory: {current_memory:.2f}GB)")
    
    def train_step(self, batch, use_amp=False):
        """Training step with adaptive accumulation"""
        data, targets = batch
        data, targets = data.to(self.device), targets.to(self.device)
        
        loss_scale = 1.0 / self.accumulation_steps
        
        if use_amp:
            with autocast():
                outputs = self.model(data)
                loss = self.criterion(outputs, targets) * loss_scale
            
            self.scaler.scale(loss).backward()
        else:
            outputs = self.model(data)
            loss = self.criterion(outputs, targets) * loss_scale
            loss.backward()
        
        self.accumulated_loss += loss.item() / loss_scale
        self.step_count += 1
        
        # Update parameters if needed
        if self.step_count % self.accumulation_steps == 0:
            if use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            avg_loss = self.accumulated_loss / self.accumulation_steps
            self.accumulated_loss = 0.0
            
            # Adapt accumulation steps periodically
            if self.step_count % (self.accumulation_steps * 10) == 0:
                self.adapt_accumulation_steps()
            
            return avg_loss, True
        
        return self.accumulated_loss / (self.step_count % self.accumulation_steps), False

# Sample Model and Dataset
class SampleCNN(nn.Module):
    """Sample CNN for testing gradient accumulation"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class SampleDataset(Dataset):
    """Sample dataset for testing"""
    
    def __init__(self, size=1000, input_shape=(3, 32, 32), num_classes=10):
        self.data = torch.randn(size, *input_shape)
        self.targets = torch.randint(0, num_classes, (size,))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Performance Comparison
def compare_training_methods(model, dataloader, device, num_epochs=2):
    """Compare different training methods"""
    
    print("Comparing Training Methods")
    print("=" * 40)
    
    results = {}
    
    # 1. Standard training
    print("1. Standard Training")
    standard_model = SampleCNN().to(device)
    standard_optimizer = torch.optim.Adam(standard_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    standard_model.train()
    
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(dataloader):
            if batch_idx >= 10:  # Limit for demo
                break
                
            data, targets = data.to(device), targets.to(device)
            
            standard_optimizer.zero_grad()
            outputs = standard_model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            standard_optimizer.step()
    
    standard_time = time.time() - start_time
    results['standard'] = standard_time
    print(f"Standard training time: {standard_time:.2f}s")
    
    # 2. Gradient accumulation training
    print("\n2. Gradient Accumulation Training")
    accum_model = SampleCNN().to(device)
    accum_optimizer = torch.optim.Adam(accum_model.parameters(), lr=0.001)
    trainer = GradientAccumulationTrainer(accum_model, accum_optimizer, criterion, device, accumulation_steps=4)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        batch_count = 0
        for batch in dataloader:
            if batch_count >= 10:  # Limit for demo
                break
            
            trainer.train_step(batch, use_amp=False)
            batch_count += 1
    
    accum_time = time.time() - start_time
    results['accumulation'] = accum_time
    print(f"Gradient accumulation time: {accum_time:.2f}s")
    
    # 3. Mixed precision with accumulation
    print("\n3. Mixed Precision + Accumulation Training")
    mixed_model = SampleCNN().to(device)
    mixed_optimizer = torch.optim.Adam(mixed_model.parameters(), lr=0.001)
    mixed_trainer = GradientAccumulationTrainer(mixed_model, mixed_optimizer, criterion, device, accumulation_steps=4)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        batch_count = 0
        for batch in dataloader:
            if batch_count >= 10:  # Limit for demo
                break
            
            mixed_trainer.train_step(batch, use_amp=True)
            batch_count += 1
    
    mixed_time = time.time() - start_time
    results['mixed_precision'] = mixed_time
    print(f"Mixed precision + accumulation time: {mixed_time:.2f}s")
    
    return results

if __name__ == "__main__":
    print("Gradient Accumulation Training Techniques")
    print("=" * 50)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model and data
    model = SampleCNN(num_classes=10)
    dataset = SampleDataset(size=200)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Test basic gradient accumulation
    print("\n1. Basic Gradient Accumulation")
    print("-" * 30)
    
    trainer = GradientAccumulationTrainer(model, optimizer, criterion, device, accumulation_steps=4)
    
    # Train for a few batches
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 5:
            break
        
        loss, updated = trainer.train_step(batch, use_amp=False)
        print(f"Batch {batch_idx + 1}: Loss = {loss:.4f}, Updated = {updated}")
    
    # Test dynamic accumulation
    print("\n2. Dynamic Gradient Accumulation")
    print("-" * 30)
    
    dynamic_model = SampleCNN(num_classes=10)
    dynamic_optimizer = torch.optim.Adam(dynamic_model.parameters(), lr=0.001)
    
    dynamic_accumulator = DynamicGradientAccumulator(
        dynamic_model, dynamic_optimizer, criterion, device,
        target_batch_size=64, base_batch_size=8
    )
    
    # Simulate training
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 8:
            break
        
        loss = dynamic_accumulator.accumulate_gradients(batch, use_amp=False)
        
        if dynamic_accumulator.should_update():
            avg_loss = dynamic_accumulator.update_parameters(use_amp=False)
            print(f"Updated parameters: Average loss = {avg_loss:.4f}")
        else:
            print(f"Batch {batch_idx + 1}: Accumulated loss = {loss:.4f}")
    
    # Test adaptive accumulation
    print("\n3. Adaptive Gradient Accumulation")
    print("-" * 30)
    
    adaptive_model = SampleCNN(num_classes=10)
    adaptive_optimizer = torch.optim.Adam(adaptive_model.parameters(), lr=0.001)
    
    adaptive_accumulator = AdaptiveGradientAccumulator(
        adaptive_model, adaptive_optimizer, criterion, device,
        max_memory_gb=2, min_accumulation_steps=2, max_accumulation_steps=8
    )
    
    # Simulate training with adaptive accumulation
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 10:
            break
        
        loss, updated = adaptive_accumulator.train_step(batch, use_amp=False)
        
        if updated:
            memory_usage = adaptive_accumulator.get_memory_usage_gb()
            print(f"Batch {batch_idx + 1}: Loss = {loss:.4f}, Memory = {memory_usage:.3f}GB")
    
    # Performance comparison
    print("\n4. Performance Comparison")
    print("-" * 30)
    
    if torch.cuda.is_available():
        comparison_results = compare_training_methods(model, dataloader, device, num_epochs=1)
        
        print("\nTraining Time Comparison:")
        for method, time_taken in comparison_results.items():
            print(f"{method.replace('_', ' ').title()}: {time_taken:.2f}s")
    else:
        print("GPU not available, skipping performance comparison")
    
    print("\nGradient accumulation demonstrations completed!") 