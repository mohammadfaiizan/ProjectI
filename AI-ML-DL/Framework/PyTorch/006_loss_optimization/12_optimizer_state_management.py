#!/usr/bin/env python3
"""PyTorch Optimizer State Management - Optimizer state saving/loading"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import json
import os
from collections import defaultdict

print("=== Optimizer State Management Overview ===")

print("Optimizer state management topics:")
print("1. Saving and loading optimizer states")
print("2. State dictionary structure")
print("3. Transfer learning with optimizers")
print("4. Optimizer state inspection")
print("5. State manipulation and surgery")
print("6. Checkpoint management")
print("7. Optimizer state reset")
print("8. Multi-optimizer state handling")

print("\n=== Model and Optimizer Setup ===")

class StateTestModel(nn.Module):
    def __init__(self, input_size=20, hidden_size=64, num_classes=10):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size // 2)
        self.linear3 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
    
    def forward(self, x):
        x = self.relu(self.batch_norm(self.linear1(x)))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# Initialize model and optimizers
model = StateTestModel()
sample_input = torch.randn(32, 20)
sample_target = torch.randint(0, 10, (32,))
loss_fn = nn.CrossEntropyLoss()

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# Different optimizers for testing
optimizers = {
    'SGD': optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4),
    'Adam': optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8),
    'AdamW': optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01),
    'RMSprop': optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, momentum=0.9)
}

print("\n=== Basic State Saving and Loading ===")

def training_step(model, optimizer, input_data, targets, loss_fn):
    """Perform a single training step"""
    optimizer.zero_grad()
    outputs = model(input_data)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save a complete checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'optimizer_type': type(optimizer).__name__
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(model, optimizer, filepath):
    """Load a complete checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']

# Train for a few steps and save checkpoint
print("Training and saving checkpoint:")
optimizer_test = optimizers['Adam']

for epoch in range(5):
    loss = training_step(model, optimizer_test, sample_input, sample_target, loss_fn)
    print(f"  Epoch {epoch}: Loss = {loss:.6f}")

# Save checkpoint
checkpoint_path = 'optimizer_checkpoint.pth'
save_checkpoint(model, optimizer_test, epoch, loss, checkpoint_path)

# Create new model and optimizer to test loading
print("\nLoading checkpoint into new model and optimizer:")
new_model = StateTestModel()
new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)

loaded_epoch, loaded_loss = load_checkpoint(new_model, new_optimizer, checkpoint_path)
print(f"Loaded checkpoint from epoch {loaded_epoch} with loss {loaded_loss:.6f}")

# Verify that optimizer states match
print(f"Original optimizer state keys: {list(optimizer_test.state_dict().keys())}")
print(f"Loaded optimizer state keys: {list(new_optimizer.state_dict().keys())}")

# Clean up
if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)

print("\n=== Optimizer State Dictionary Structure ===")

def analyze_optimizer_state(optimizer, name):
    """Analyze optimizer state dictionary structure"""
    state_dict = optimizer.state_dict()
    
    print(f"\n{name} Optimizer State Analysis:")
    print(f"  State dict keys: {list(state_dict.keys())}")
    
    # Analyze parameter groups
    param_groups = state_dict['param_groups']
    print(f"  Number of parameter groups: {len(param_groups)}")
    
    for i, group in enumerate(param_groups):
        print(f"  Group {i} keys: {list(group.keys())}")
        print(f"  Group {i} params count: {len(group['params'])}")
    
    # Analyze state
    state = state_dict['state']
    print(f"  State entries: {len(state)}")
    
    if state:
        # Get first state entry to show structure
        first_param_id = next(iter(state.keys()))
        first_state = state[first_param_id]
        print(f"  First state keys: {list(first_state.keys())}")
        
        # Show state details
        for key, value in first_state.items():
            if isinstance(value, torch.Tensor):
                print(f"    {key}: Tensor{tuple(value.shape)} ({value.dtype})")
            else:
                print(f"    {key}: {value}")

# Train optimizers to populate their states
print("Training optimizers to populate states:")
for name, opt in optimizers.items():
    # Reset model for fair comparison
    model = StateTestModel()
    if name == 'SGD':
        opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    elif name == 'Adam':
        opt = optim.Adam(model.parameters(), lr=0.001)
    elif name == 'AdamW':
        opt = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    elif name == 'RMSprop':
        opt = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, momentum=0.9)
    
    # Train for a few steps
    for _ in range(3):
        training_step(model, opt, sample_input, sample_target, loss_fn)
    
    analyze_optimizer_state(opt, name)

print("\n=== State Inspection and Monitoring ===")

class OptimizerStateMonitor:
    """Monitor optimizer state throughout training"""
    
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.history = defaultdict(list)
        self.step_count = 0
    
    def record_state(self):
        """Record current optimizer state"""
        state_dict = self.optimizer.state_dict()
        self.step_count += 1
        
        # Record parameter group statistics
        for i, group in enumerate(state_dict['param_groups']):
            self.history[f'group_{i}_lr'].append(group['lr'])
        
        # Record state statistics
        state = state_dict['state']
        if state:
            # Statistics for momentum-based optimizers
            momentum_norms = []
            gradient_norms = []
            
            for param_state in state.values():
                if 'momentum_buffer' in param_state:
                    momentum_norms.append(param_state['momentum_buffer'].norm().item())
                
                if 'exp_avg' in param_state:  # Adam-style
                    gradient_norms.append(param_state['exp_avg'].norm().item())
            
            if momentum_norms:
                self.history['avg_momentum_norm'].append(sum(momentum_norms) / len(momentum_norms))
            
            if gradient_norms:
                self.history['avg_gradient_norm'].append(sum(gradient_norms) / len(gradient_norms))
    
    def get_summary(self):
        """Get summary of recorded statistics"""
        summary = {}
        for key, values in self.history.items():
            if values:
                summary[key] = {
                    'current': values[-1],
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values)
                }
        return summary

# Test state monitoring
print("Optimizer state monitoring during training:")
model_monitor = StateTestModel()
optimizer_monitor = optim.Adam(model_monitor.parameters(), lr=0.001)
monitor = OptimizerStateMonitor(optimizer_monitor)

for epoch in range(10):
    loss = training_step(model_monitor, optimizer_monitor, sample_input, sample_target, loss_fn)
    monitor.record_state()
    
    if epoch % 3 == 0:
        summary = monitor.get_summary()
        print(f"  Epoch {epoch}: Loss = {loss:.6f}")
        for key, stats in summary.items():
            print(f"    {key}: current={stats['current']:.6f}, mean={stats['mean']:.6f}")

print("\n=== Optimizer State Surgery ===")

def modify_optimizer_learning_rate(optimizer, new_lr):
    """Modify learning rate in optimizer state"""
    state_dict = optimizer.state_dict()
    
    print(f"Modifying learning rate to {new_lr}")
    for group in state_dict['param_groups']:
        old_lr = group['lr']
        group['lr'] = new_lr
        print(f"  Changed LR from {old_lr} to {new_lr}")
    
    optimizer.load_state_dict(state_dict)

def scale_optimizer_momentum(optimizer, scale_factor):
    """Scale momentum buffers in optimizer state"""
    state_dict = optimizer.state_dict()
    modified_count = 0
    
    for param_state in state_dict['state'].values():
        if 'momentum_buffer' in param_state:
            param_state['momentum_buffer'] *= scale_factor
            modified_count += 1
        
        if 'exp_avg' in param_state:  # Adam-style momentum
            param_state['exp_avg'] *= scale_factor
            modified_count += 1
    
    optimizer.load_state_dict(state_dict)
    print(f"Scaled {modified_count} momentum buffers by factor {scale_factor}")

def reset_optimizer_state(optimizer, keep_param_groups=True):
    """Reset optimizer state while optionally keeping parameter groups"""
    if keep_param_groups:
        # Keep parameter groups but reset state
        state_dict = optimizer.state_dict()
        state_dict['state'] = {}
        optimizer.load_state_dict(state_dict)
    else:
        # Complete reset
        for group in optimizer.param_groups:
            for param in group['params']:
                param_state = optimizer.state[param]
                param_state.clear()

# Test optimizer state surgery
print("Testing optimizer state surgery:")
model_surgery = StateTestModel()
optimizer_surgery = optim.SGD(model_surgery.parameters(), lr=0.01, momentum=0.9)

# Train to populate state
for _ in range(3):
    training_step(model_surgery, optimizer_surgery, sample_input, sample_target, loss_fn)

print("Original optimizer state:")
original_lr = optimizer_surgery.param_groups[0]['lr']
print(f"  Learning rate: {original_lr}")

# Modify learning rate
modify_optimizer_learning_rate(optimizer_surgery, 0.001)
new_lr = optimizer_surgery.param_groups[0]['lr']
print(f"  New learning rate: {new_lr}")

# Scale momentum
scale_optimizer_momentum(optimizer_surgery, 0.5)

# Reset state
print("Resetting optimizer state...")
reset_optimizer_state(optimizer_surgery, keep_param_groups=True)

print("\n=== Transfer Learning with Optimizer States ===")

def transfer_optimizer_state(source_optimizer, target_optimizer, layer_mapping=None):
    """Transfer optimizer state between models with potentially different architectures"""
    
    source_state = source_optimizer.state_dict()
    target_state = target_optimizer.state_dict()
    
    print("Transferring optimizer state...")
    
    # Transfer parameter groups (learning rates, etc.)
    for i, (src_group, tgt_group) in enumerate(zip(source_state['param_groups'], target_state['param_groups'])):
        for key in ['lr', 'momentum', 'weight_decay', 'eps', 'betas']:
            if key in src_group and key in tgt_group:
                old_value = tgt_group[key]
                tgt_group[key] = src_group[key]
                print(f"  Group {i} {key}: {old_value} → {src_group[key]}")
    
    # Transfer compatible state entries
    transferred_count = 0
    source_state_items = list(source_state['state'].items())
    target_state_items = list(target_state['state'].items())
    
    # Simple transfer for same-sized models
    min_length = min(len(source_state_items), len(target_state_items))
    for i in range(min_length):
        src_id, src_state_dict = source_state_items[i]
        tgt_id, tgt_state_dict = target_state_items[i]
        
        # Transfer compatible state
        for key in src_state_dict:
            if key in tgt_state_dict:
                src_tensor = src_state_dict[key]
                tgt_tensor = tgt_state_dict[key]
                
                # Only transfer if shapes match
                if src_tensor.shape == tgt_tensor.shape:
                    target_state['state'][tgt_id][key] = src_tensor.clone()
                    transferred_count += 1
    
    target_optimizer.load_state_dict(target_state)
    print(f"  Transferred {transferred_count} state tensors")

# Test transfer learning
print("Testing optimizer state transfer:")

# Create source model and train it
source_model = StateTestModel()
source_optimizer = optim.Adam(source_model.parameters(), lr=0.001)

print("Training source model...")
for _ in range(5):
    training_step(source_model, source_optimizer, sample_input, sample_target, loss_fn)

# Create target model with same architecture
target_model = StateTestModel()
target_optimizer = optim.Adam(target_model.parameters(), lr=0.01)  # Different LR

print("Before transfer:")
print(f"  Source optimizer LR: {source_optimizer.param_groups[0]['lr']}")
print(f"  Target optimizer LR: {target_optimizer.param_groups[0]['lr']}")

# Transfer optimizer state
transfer_optimizer_state(source_optimizer, target_optimizer)

print("After transfer:")
print(f"  Target optimizer LR: {target_optimizer.param_groups[0]['lr']}")

print("\n=== Multi-Optimizer State Management ===")

class MultiOptimizerManager:
    """Manage multiple optimizers for different parts of a model"""
    
    def __init__(self):
        self.optimizers = {}
        self.schedulers = {}
    
    def add_optimizer(self, name, optimizer, scheduler=None):
        """Add an optimizer (and optional scheduler)"""
        self.optimizers[name] = optimizer
        if scheduler:
            self.schedulers[name] = scheduler
    
    def zero_grad(self):
        """Zero gradients for all optimizers"""
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
    
    def step(self):
        """Step all optimizers"""
        for optimizer in self.optimizers.values():
            optimizer.step()
    
    def scheduler_step(self, **kwargs):
        """Step all schedulers"""
        for scheduler in self.schedulers.values():
            scheduler.step(**kwargs)
    
    def save_state(self, filepath):
        """Save all optimizer and scheduler states"""
        state = {
            'optimizers': {name: opt.state_dict() for name, opt in self.optimizers.items()},
            'schedulers': {name: sch.state_dict() for name, sch in self.schedulers.items()}
        }
        torch.save(state, filepath)
    
    def load_state(self, filepath):
        """Load all optimizer and scheduler states"""
        state = torch.load(filepath, map_location='cpu')
        
        for name, opt_state in state['optimizers'].items():
            if name in self.optimizers:
                self.optimizers[name].load_state_dict(opt_state)
        
        for name, sch_state in state['schedulers'].items():
            if name in self.schedulers:
                self.schedulers[name].load_state_dict(sch_state)
    
    def get_learning_rates(self):
        """Get current learning rates for all optimizers"""
        lrs = {}
        for name, optimizer in self.optimizers.items():
            lrs[name] = [group['lr'] for group in optimizer.param_groups]
        return lrs

# Test multi-optimizer management
print("Testing multi-optimizer management:")

# Create model with different parameter groups
model_multi = StateTestModel()

# Different optimizers for different parts
backbone_params = list(model_multi.linear1.parameters()) + list(model_multi.batch_norm.parameters())
head_params = list(model_multi.linear2.parameters()) + list(model_multi.linear3.parameters())

backbone_optimizer = optim.Adam(backbone_params, lr=0.0001)
head_optimizer = optim.SGD(head_params, lr=0.01, momentum=0.9)

# Create schedulers
import torch.optim.lr_scheduler as lr_scheduler
backbone_scheduler = lr_scheduler.ExponentialLR(backbone_optimizer, gamma=0.95)
head_scheduler = lr_scheduler.StepLR(head_optimizer, step_size=3, gamma=0.5)

# Setup multi-optimizer manager
manager = MultiOptimizerManager()
manager.add_optimizer('backbone', backbone_optimizer, backbone_scheduler)
manager.add_optimizer('head', head_optimizer, head_scheduler)

print("Training with multi-optimizer setup:")
for epoch in range(8):
    manager.zero_grad()
    
    # Forward pass
    outputs = model_multi(sample_input)
    loss = loss_fn(outputs, sample_target)
    
    # Backward pass
    loss.backward()
    
    # Step optimizers and schedulers
    manager.step()
    manager.scheduler_step()
    
    if epoch % 2 == 0:
        lrs = manager.get_learning_rates()
        print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")
        print(f"    Backbone LR: {lrs['backbone'][0]:.6f}")
        print(f"    Head LR: {lrs['head'][0]:.6f}")

# Save and load multi-optimizer state
multi_checkpoint_path = 'multi_optimizer_checkpoint.pth'
manager.save_state(multi_checkpoint_path)
print(f"Multi-optimizer state saved to {multi_checkpoint_path}")

# Clean up
if os.path.exists(multi_checkpoint_path):
    os.remove(multi_checkpoint_path)

print("\n=== State Management Best Practices ===")

print("Checkpoint Saving Guidelines:")
print("1. Save optimizer state along with model state")
print("2. Include epoch, loss, and other training metadata")
print("3. Save scheduler states for reproducible training")
print("4. Use descriptive checkpoint filenames")
print("5. Implement checkpoint versioning for long training")

print("\nState Transfer Recommendations:")
print("1. Verify architecture compatibility before transfer")
print("2. Handle different learning rates appropriately")
print("3. Consider freezing transferred layers initially")
print("4. Monitor convergence after state transfer")
print("5. Test transfer on validation data first")

print("\nOptimizer State Debugging:")
print("1. Inspect state dict structure before/after loading")
print("2. Verify parameter group configurations")
print("3. Check momentum buffer sizes and values")
print("4. Monitor state statistics during training")
print("5. Validate learning rate changes take effect")

print("\nPerformance Considerations:")
print("1. Optimizer states can be large (especially Adam)")
print("2. Use appropriate devices for state tensors")
print("3. Consider state compression for large models")
print("4. Implement incremental checkpointing for long runs")
print("5. Clean up old checkpoints to save disk space")

print("\nCommon Issues and Solutions:")
print("1. Version mismatch: Use map_location and strict=False")
print("2. Device mismatch: Load to CPU then move to target device")
print("3. Architecture changes: Implement flexible state transfer")
print("4. Memory issues: Load state dicts in chunks")
print("5. Corruption: Implement checkpoint validation")

print("\n=== Optimizer State Management Complete ===")

# Memory cleanup
del model, sample_input, sample_target