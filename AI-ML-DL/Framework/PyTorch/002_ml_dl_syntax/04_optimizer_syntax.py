#!/usr/bin/env python3
"""PyTorch Optimizers - All optimizers and their parameters"""

import torch
import torch.nn as nn
import torch.optim as optim

print("=== Basic Optimizers ===")

# Create a simple model for testing
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Stochastic Gradient Descent (SGD)
sgd_optimizer = optim.SGD(model.parameters(), lr=0.01)
sgd_momentum = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
sgd_nesterov = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
sgd_weight_decay = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

print(f"SGD optimizer: {type(sgd_optimizer).__name__}")
print(f"SGD with momentum: momentum={sgd_momentum.param_groups[0]['momentum']}")
print(f"SGD Nesterov: nesterov={sgd_nesterov.param_groups[0]['nesterov']}")
print(f"SGD weight decay: weight_decay={sgd_weight_decay.param_groups[0]['weight_decay']}")

# Adam Optimizer
adam_optimizer = optim.Adam(model.parameters(), lr=0.001)
adam_custom = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)

print(f"Adam optimizer: {type(adam_optimizer).__name__}")
print(f"Adam betas: {adam_custom.param_groups[0]['betas']}")
print(f"Adam eps: {adam_custom.param_groups[0]['eps']}")

print("\n=== Adam Variants ===")

# AdamW (Adam with decoupled weight decay)
adamw_optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
print(f"AdamW weight decay: {adamw_optimizer.param_groups[0]['weight_decay']}")

# Adamax
adamax_optimizer = optim.Adamax(model.parameters(), lr=0.002, betas=(0.9, 0.999))
print(f"Adamax optimizer: {type(adamax_optimizer).__name__}")

# NAdam (Nesterov Adam)
nadam_optimizer = optim.NAdam(model.parameters(), lr=0.002, betas=(0.9, 0.999))
print(f"NAdam optimizer: {type(nadam_optimizer).__name__}")

# RAdam (Rectified Adam)
radam_optimizer = optim.RAdam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
print(f"RAdam optimizer: {type(radam_optimizer).__name__}")

print("\n=== Other Popular Optimizers ===")

# RMSprop
rmsprop_optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-8)
rmsprop_momentum = optim.RMSprop(model.parameters(), lr=0.01, momentum=0.9, centered=True)

print(f"RMSprop alpha: {rmsprop_optimizer.param_groups[0]['alpha']}")
print(f"RMSprop centered: {rmsprop_momentum.param_groups[0]['centered']}")

# Adagrad
adagrad_optimizer = optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0)
print(f"Adagrad lr_decay: {adagrad_optimizer.param_groups[0]['lr_decay']}")

# Adadelta
adadelta_optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-6)
print(f"Adadelta rho: {adadelta_optimizer.param_groups[0]['rho']}")

print("\n=== Optimizer State and Parameter Groups ===")

# Multiple parameter groups with different settings
conv_params = []
fc_params = []

for name, param in model.named_parameters():
    if 'weight' in name:
        conv_params.append(param)
    else:
        fc_params.append(param)

# Optimizer with different parameter groups
optimizer_groups = optim.Adam([
    {'params': conv_params, 'lr': 0.001, 'weight_decay': 1e-4},
    {'params': fc_params, 'lr': 0.01, 'weight_decay': 1e-3}
])

print(f"Number of parameter groups: {len(optimizer_groups.param_groups)}")
print(f"Group 0 lr: {optimizer_groups.param_groups[0]['lr']}")
print(f"Group 1 lr: {optimizer_groups.param_groups[1]['lr']}")

# Optimizer state
dummy_input = torch.randn(32, 784)
dummy_target = torch.randint(0, 10, (32,))
criterion = nn.CrossEntropyLoss()

# Forward and backward to create optimizer state
output = model(dummy_input)
loss = criterion(output, dummy_target)
loss.backward()
optimizer_groups.step()

print(f"Optimizer state keys: {list(optimizer_groups.state.keys())[:3]}")  # Show first 3

print("\n=== Learning Rate Scheduling ===")

# Learning rate schedulers
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau

base_optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step LR - decay by factor every step_size epochs
step_scheduler = StepLR(base_optimizer, step_size=30, gamma=0.1)

# Exponential LR - decay by factor every epoch
exp_scheduler = ExponentialLR(base_optimizer, gamma=0.95)

# Cosine Annealing - cosine decay
cosine_scheduler = CosineAnnealingLR(base_optimizer, T_max=100, eta_min=0)

# Reduce on Plateau - reduce when metric stops improving
plateau_scheduler = ReduceLROnPlateau(base_optimizer, mode='min', factor=0.1, patience=10)

print(f"Initial LR: {base_optimizer.param_groups[0]['lr']}")

# Simulate training steps
for epoch in range(5):
    # Simulate training
    output = model(dummy_input)
    loss = criterion(output, dummy_target)
    
    base_optimizer.zero_grad()
    loss.backward()
    base_optimizer.step()
    
    # Step schedulers
    step_scheduler.step()
    exp_scheduler.step()
    cosine_scheduler.step()
    plateau_scheduler.step(loss)  # Pass metric for plateau scheduler
    
    current_lr = base_optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}, LR: {current_lr:.6f}, Loss: {loss.item():.4f}")

print("\n=== Advanced Optimizer Techniques ===")

# Gradient clipping
optimizer_clip = optim.Adam(model.parameters(), lr=0.001)

def train_step_with_clipping(model, optimizer, data, target, max_norm=1.0):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    optimizer.step()
    return loss

# Test gradient clipping
loss_clipped = train_step_with_clipping(model, optimizer_clip, dummy_input, dummy_target)
print(f"Loss with gradient clipping: {loss_clipped.item():.4f}")

# Gradient accumulation
def train_with_accumulation(model, optimizer, data_list, target_list, accumulation_steps=4):
    optimizer.zero_grad()
    total_loss = 0
    
    for i, (data, target) in enumerate(zip(data_list, target_list)):
        output = model(data)
        loss = criterion(output, target) / accumulation_steps  # Scale loss
        loss.backward()
        total_loss += loss.item()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    return total_loss

# Simulate gradient accumulation
data_batches = [torch.randn(8, 784) for _ in range(4)]
target_batches = [torch.randint(0, 10, (8,)) for _ in range(4)]

accum_loss = train_with_accumulation(model, optimizer_clip, data_batches, target_batches)
print(f"Accumulated gradient loss: {accum_loss:.4f}")

print("\n=== Custom Optimizer Implementation ===")

# Simple custom optimizer (simplified SGD)
class SimpleSGD:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
    
    def step(self):
        for param in self.params:
            if param.grad is not None:
                param.data.add_(param.grad.data, alpha=-self.lr)

# Test custom optimizer
custom_optimizer = SimpleSGD(model.parameters(), lr=0.01)

output = model(dummy_input)
loss = criterion(output, dummy_target)
custom_optimizer.zero_grad()
loss.backward()
custom_optimizer.step()

print(f"Custom optimizer step completed, loss: {loss.item():.4f}")

print("\n=== Optimizer State Manipulation ===")

# Save and load optimizer state
optimizer_state = optim.Adam(model.parameters(), lr=0.001)

# Train for a few steps
for _ in range(3):
    output = model(dummy_input)
    loss = criterion(output, dummy_target)
    optimizer_state.zero_grad()
    loss.backward()
    optimizer_state.step()

# Save state
saved_state = optimizer_state.state_dict()
print(f"Saved optimizer state keys: {list(saved_state.keys())}")

# Create new optimizer and load state
new_optimizer = optim.Adam(model.parameters(), lr=0.001)
new_optimizer.load_state_dict(saved_state)

print("Optimizer state loaded successfully")

# Compare states
original_param_group = optimizer_state.param_groups[0]
loaded_param_group = new_optimizer.param_groups[0]
print(f"States match: {original_param_group['lr'] == loaded_param_group['lr']}")

print("\n=== Optimizer Comparison ===")

# Compare different optimizers on same task
optimizers_to_test = {
    'SGD': optim.SGD(model.parameters(), lr=0.01),
    'Adam': optim.Adam(model.parameters(), lr=0.001),
    'AdamW': optim.AdamW(model.parameters(), lr=0.001),
    'RMSprop': optim.RMSprop(model.parameters(), lr=0.01),
    'Adagrad': optim.Adagrad(model.parameters(), lr=0.01)
}

# Reset model parameters
def reset_model_parameters(model):
    for layer in model:
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

print("Optimizer comparison (5 steps):")
for name, optimizer in optimizers_to_test.items():
    # Reset model
    reset_model_parameters(model)
    
    losses = []
    for step in range(5):
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    final_loss = losses[-1]
    print(f"{name:>10}: Final loss = {final_loss:.4f}")

print("\n=== Learning Rate Finding ===")

# Learning rate range test (simplified)
def lr_range_test(model, data, target, start_lr=1e-7, end_lr=10, num_steps=100):
    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    lr_mult = (end_lr / start_lr) ** (1 / num_steps)
    
    lrs = []
    losses = []
    
    for step in range(num_steps):
        output = model(data)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())
        
        # Update learning rate
        optimizer.param_groups[0]['lr'] *= lr_mult
        
        if step % 20 == 0:
            print(f"Step {step}, LR: {optimizer.param_groups[0]['lr']:.2e}, Loss: {loss.item():.4f}")
    
    return lrs, losses

# Run LR range test
reset_model_parameters(model)
lrs, losses = lr_range_test(model, dummy_input, dummy_target)
print(f"LR range test completed: {len(lrs)} points")

print("\n=== Optimizer Best Practices ===")

print("Optimizer Selection Guidelines:")
print("1. Adam/AdamW: Good default choice, works well out of the box")
print("2. SGD with momentum: Often better final performance, needs tuning")
print("3. AdamW: Better than Adam for weight decay")
print("4. RMSprop: Good for RNNs")
print("5. RAdam: More stable than Adam in early training")

print("\nHyperparameter Tips:")
print("- Adam: lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4")
print("- SGD: lr=0.1, momentum=0.9, weight_decay=1e-4")
print("- Use learning rate schedules for better convergence")
print("- Consider gradient clipping for RNNs")
print("- Different LRs for different parameter groups")

print("\nCommon Mistakes:")
print("- Not resetting gradients (zero_grad())")
print("- Wrong learning rate scale")
print("- Not using weight decay appropriately")
print("- Forgetting to call scheduler.step()")
print("- Using Adam with weight decay instead of AdamW")

print("\n=== Optimizers Complete ===") 