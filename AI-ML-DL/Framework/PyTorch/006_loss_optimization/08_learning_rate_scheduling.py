#!/usr/bin/env python3
"""PyTorch Learning Rate Scheduling - All LR schedulers and syntax"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import math

print("=== Learning Rate Scheduling Overview ===")

print("PyTorch LR schedulers covered:")
print("1. StepLR - Step decay")
print("2. MultiStepLR - Multi-step decay")
print("3. ExponentialLR - Exponential decay")
print("4. CosineAnnealingLR - Cosine annealing")
print("5. ReduceLROnPlateau - Adaptive based on metric")
print("6. CyclicLR - Cyclical learning rates")
print("7. OneCycleLR - One cycle policy")
print("8. CosineAnnealingWarmRestarts - SGDR")
print("9. LambdaLR - Custom function")
print("10. MultiplicativeLR - Multiplicative decay")
print("11. PolynomialLR - Polynomial decay")
print("12. LinearLR - Linear decay")

print("\n=== Basic Model Setup ===")

class SchedulerTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 32)
        self.linear2 = nn.Linear(32, 16)
        self.linear3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# Create model and data
model = SchedulerTestModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)
sample_input = torch.randn(32, 10)
sample_target = torch.randn(32, 1)
loss_fn = nn.MSELoss()

print(f"Initial learning rate: {optimizer.param_groups[0]['lr']}")

print("\n=== StepLR Scheduler ===")

# StepLR - reduces LR by gamma every step_size epochs
step_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

print(f"StepLR parameters:")
print(f"  Step size: {step_scheduler.step_size}")
print(f"  Gamma: {step_scheduler.gamma}")
print(f"  Last epoch: {step_scheduler.last_epoch}")

# Simulate training with StepLR
optimizer = optim.Adam(model.parameters(), lr=0.1)
step_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

print(f"\nStepLR learning rate progression:")
for epoch in range(20):
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:2d}: LR = {current_lr:.6f}")
    
    # Simulate training step
    optimizer.zero_grad()
    output = model(sample_input)
    loss = loss_fn(output, sample_target)
    loss.backward()
    optimizer.step()
    
    # Step the scheduler
    step_scheduler.step()
    
    if epoch == 4 or epoch == 9 or epoch == 14:
        print(f"           ↓ LR reduced by gamma={step_scheduler.gamma}")

print("\n=== MultiStepLR Scheduler ===")

# MultiStepLR - reduces LR at specific milestones
optimizer = optim.Adam(model.parameters(), lr=0.1)
multistep_scheduler = lr_scheduler.MultiStepLR(
    optimizer, 
    milestones=[5, 10, 15], 
    gamma=0.2
)

print(f"MultiStepLR parameters:")
print(f"  Milestones: {multistep_scheduler.milestones}")
print(f"  Gamma: {multistep_scheduler.gamma}")

print(f"\nMultiStepLR learning rate progression:")
for epoch in range(20):
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:2d}: LR = {current_lr:.6f}")
    
    if epoch in [5, 10, 15]:
        print(f"           ↓ Milestone reached, LR reduced")
    
    multistep_scheduler.step()

print("\n=== ExponentialLR Scheduler ===")

# ExponentialLR - exponential decay
optimizer = optim.Adam(model.parameters(), lr=0.1)
exp_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

print(f"ExponentialLR parameters:")
print(f"  Gamma: {exp_scheduler.gamma}")

print(f"\nExponentialLR learning rate progression:")
for epoch in range(15):
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:2d}: LR = {current_lr:.6f}")
    exp_scheduler.step()

print("\n=== CosineAnnealingLR Scheduler ===")

# CosineAnnealingLR - cosine annealing
optimizer = optim.Adam(model.parameters(), lr=0.1)
cosine_scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=10, 
    eta_min=0.001
)

print(f"CosineAnnealingLR parameters:")
print(f"  T_max: {cosine_scheduler.T_max}")
print(f"  eta_min: {cosine_scheduler.eta_min}")

print(f"\nCosineAnnealingLR learning rate progression:")
for epoch in range(25):
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:2d}: LR = {current_lr:.6f}")
    cosine_scheduler.step()

print("\n=== ReduceLROnPlateau Scheduler ===")

# ReduceLROnPlateau - reduces LR when metric plateaus
optimizer = optim.Adam(model.parameters(), lr=0.1)
plateau_scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
    threshold=0.01,
    threshold_mode='rel',
    cooldown=0,
    min_lr=0.001,
    eps=1e-8
)

print(f"ReduceLROnPlateau parameters:")
print(f"  Mode: {plateau_scheduler.mode}")
print(f"  Factor: {plateau_scheduler.factor}")
print(f"  Patience: {plateau_scheduler.patience}")
print(f"  Threshold: {plateau_scheduler.threshold}")
print(f"  Min LR: {plateau_scheduler.min_lrs}")

# Simulate losses that plateau
simulated_losses = [
    1.0, 0.8, 0.6, 0.4, 0.35, 0.34, 0.33, 0.33, 0.33, 0.33,  # Plateau here
    0.32, 0.31, 0.30, 0.29, 0.29, 0.29, 0.29,  # Another plateau
    0.28, 0.27, 0.26
]

print(f"\nReduceLROnPlateau learning rate progression:")
for epoch, loss in enumerate(simulated_losses):
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:2d}: Loss = {loss:.3f}, LR = {current_lr:.6f}")
    
    # Step with loss
    plateau_scheduler.step(loss)

print("\n=== CyclicLR Scheduler ===")

# CyclicLR - cyclical learning rates
optimizer = optim.Adam(model.parameters(), lr=0.001)
cyclic_scheduler = lr_scheduler.CyclicLR(
    optimizer,
    base_lr=0.001,
    max_lr=0.1,
    step_size_up=5,
    step_size_down=None,
    mode='triangular',
    gamma=1.0,
    scale_fn=None,
    scale_mode='cycle',
    cycle_momentum=False
)

print(f"CyclicLR parameters:")
print(f"  Base LR: {cyclic_scheduler.base_lrs}")
print(f"  Max LR: {cyclic_scheduler.max_lrs}")
print(f"  Step size up: {cyclic_scheduler.step_size_up}")
print(f"  Mode: {cyclic_scheduler.mode}")

print(f"\nCyclicLR learning rate progression:")
for epoch in range(30):
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:2d}: LR = {current_lr:.6f}")
    
    # For CyclicLR, step after each batch (here we simulate batch steps)
    for batch in range(1):  # Simulating 1 batch per epoch
        cyclic_scheduler.step()

print("\n=== OneCycleLR Scheduler ===")

# OneCycleLR - one cycle policy
optimizer = optim.Adam(model.parameters(), lr=0.01)
one_cycle_scheduler = lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,
    steps_per_epoch=10,
    epochs=5,
    pct_start=0.3,
    anneal_strategy='cos',
    cycle_momentum=False,
    div_factor=25.0,
    final_div_factor=10000.0
)

print(f"OneCycleLR parameters:")
print(f"  Max LR: {one_cycle_scheduler.max_lrs}")
print(f"  Steps per epoch: 10")
print(f"  Total epochs: 5")
print(f"  Pct start: {one_cycle_scheduler.pct_start}")

print(f"\nOneCycleLR learning rate progression:")
total_steps = 5 * 10  # epochs * steps_per_epoch
for step in range(total_steps):
    current_lr = optimizer.param_groups[0]['lr']
    epoch = step // 10
    batch = step % 10
    print(f"Epoch {epoch}, Batch {batch}: LR = {current_lr:.6f}")
    one_cycle_scheduler.step()

print("\n=== CosineAnnealingWarmRestarts Scheduler ===")

# CosineAnnealingWarmRestarts - SGDR (Stochastic Gradient Descent with Warm Restarts)
optimizer = optim.Adam(model.parameters(), lr=0.1)
sgdr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=5,
    T_mult=2,
    eta_min=0.001
)

print(f"CosineAnnealingWarmRestarts parameters:")
print(f"  T_0: {sgdr_scheduler.T_0}")
print(f"  T_mult: {sgdr_scheduler.T_mult}")
print(f"  eta_min: {sgdr_scheduler.eta_min}")

print(f"\nSGDR learning rate progression:")
for epoch in range(25):
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:2d}: LR = {current_lr:.6f}")
    sgdr_scheduler.step()

print("\n=== LambdaLR Scheduler ===")

# LambdaLR - custom function-based scheduling
def lr_lambda(epoch):
    """Custom learning rate function"""
    if epoch < 10:
        return 1.0
    elif epoch < 20:
        return 0.1
    else:
        return 0.01

optimizer = optim.Adam(model.parameters(), lr=0.1)
lambda_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

print(f"LambdaLR with custom function:")
print(f"  Function: returns 1.0 for epoch<10, 0.1 for epoch<20, 0.01 otherwise")

print(f"\nLambdaLR learning rate progression:")
for epoch in range(25):
    current_lr = optimizer.param_groups[0]['lr']
    lambda_factor = lr_lambda(epoch)
    print(f"Epoch {epoch:2d}: Lambda = {lambda_factor:.3f}, LR = {current_lr:.6f}")
    lambda_scheduler.step()

print("\n=== MultiplicativeLR Scheduler ===")

# MultiplicativeLR - multiplicative decay
def lr_multiplier(epoch):
    """Multiplicative factor function"""
    return 0.98

optimizer = optim.Adam(model.parameters(), lr=0.1)
multiplicative_scheduler = lr_scheduler.MultiplicativeLR(
    optimizer, 
    lr_lambda=lr_multiplier
)

print(f"MultiplicativeLR learning rate progression:")
for epoch in range(15):
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:2d}: LR = {current_lr:.6f}")
    multiplicative_scheduler.step()

print("\n=== PolynomialLR Scheduler ===")

# PolynomialLR - polynomial decay
optimizer = optim.Adam(model.parameters(), lr=0.1)
polynomial_scheduler = lr_scheduler.PolynomialLR(
    optimizer,
    total_iters=20,
    power=2.0
)

print(f"PolynomialLR parameters:")
print(f"  Total iters: {polynomial_scheduler.total_iters}")
print(f"  Power: {polynomial_scheduler.power}")

print(f"\nPolynomialLR learning rate progression:")
for epoch in range(25):
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:2d}: LR = {current_lr:.6f}")
    polynomial_scheduler.step()

print("\n=== LinearLR Scheduler ===")

# LinearLR - linear decay
optimizer = optim.Adam(model.parameters(), lr=0.1)
linear_scheduler = lr_scheduler.LinearLR(
    optimizer,
    start_factor=1.0,
    end_factor=0.1,
    total_iters=10
)

print(f"LinearLR parameters:")
print(f"  Start factor: {linear_scheduler.start_factor}")
print(f"  End factor: {linear_scheduler.end_factor}")
print(f"  Total iters: {linear_scheduler.total_iters}")

print(f"\nLinearLR learning rate progression:")
for epoch in range(15):
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:2d}: LR = {current_lr:.6f}")
    linear_scheduler.step()

print("\n=== Chained Schedulers ===")

# ChainedScheduler - combine multiple schedulers
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Create multiple schedulers
linear_warmup = lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
cosine_annealing = lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=0.001)

# Chain them together
chained_scheduler = lr_scheduler.ChainedScheduler([linear_warmup, cosine_annealing])

print(f"ChainedScheduler: LinearLR (warmup) → CosineAnnealingLR")

print(f"\nChained scheduler learning rate progression:")
for epoch in range(20):
    current_lr = optimizer.param_groups[0]['lr']
    phase = "Warmup" if epoch < 5 else "Cosine Annealing"
    print(f"Epoch {epoch:2d} ({phase:15}): LR = {current_lr:.6f}")
    chained_scheduler.step()

print("\n=== Sequential Schedulers ===")

# SequentialLR - switch between schedulers at milestones
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Define schedulers
constant_lr = lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=5)
exp_lr = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
step_lr = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# Sequential scheduler
sequential_scheduler = lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[constant_lr, exp_lr, step_lr],
    milestones=[5, 15]
)

print(f"SequentialLR: Constant (0-5) → Exponential (5-15) → Step (15+)")

print(f"\nSequential scheduler learning rate progression:")
for epoch in range(25):
    current_lr = optimizer.param_groups[0]['lr']
    
    if epoch < 5:
        phase = "Constant"
    elif epoch < 15:
        phase = "Exponential"
    else:
        phase = "Step"
    
    print(f"Epoch {epoch:2d} ({phase:11}): LR = {current_lr:.6f}")
    sequential_scheduler.step()

print("\n=== Custom Scheduler Implementation ===")

class WarmupCosineScheduler:
    """Custom scheduler: Linear warmup + Cosine annealing"""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=0.0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
    
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

# Test custom scheduler
optimizer = optim.Adam(model.parameters(), lr=0.1)
custom_scheduler = WarmupCosineScheduler(
    optimizer, 
    warmup_epochs=5, 
    total_epochs=20, 
    min_lr=0.001
)

print(f"Custom Warmup + Cosine scheduler:")
for epoch in range(25):
    current_lr = custom_scheduler.get_lr()[0]
    phase = "Warmup" if epoch < 5 else "Cosine Annealing"
    print(f"Epoch {epoch:2d} ({phase:15}): LR = {current_lr:.6f}")
    custom_scheduler.step()

print("\n=== Scheduler with Parameter Groups ===")

# Different schedules for different parameter groups
def create_param_groups_with_schedulers():
    model = SchedulerTestModel()
    
    # Create parameter groups
    param_groups = [
        {
            'params': model.linear1.parameters(),
            'lr': 0.01,
            'name': 'backbone'
        },
        {
            'params': model.linear2.parameters(),
            'lr': 0.001,
            'name': 'middle'
        },
        {
            'params': model.linear3.parameters(),
            'lr': 0.1,
            'name': 'head'
        }
    ]
    
    optimizer = optim.Adam(param_groups)
    
    # Different schedulers for different groups
    def lambda_backbone(epoch):
        return 0.95 ** epoch
    
    def lambda_middle(epoch):
        return 1.0 if epoch < 10 else 0.1
    
    def lambda_head(epoch):
        return 0.5 ** (epoch // 5)
    
    scheduler = lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=[lambda_backbone, lambda_middle, lambda_head]
    )
    
    return optimizer, scheduler

optimizer_groups, scheduler_groups = create_param_groups_with_schedulers()

print(f"Parameter groups with different schedules:")
for epoch in range(15):
    lrs = [group['lr'] for group in optimizer_groups.param_groups]
    names = [group['name'] for group in optimizer_groups.param_groups]
    
    print(f"Epoch {epoch:2d}:")
    for name, lr in zip(names, lrs):
        print(f"  {name:8}: LR = {lr:.6f}")
    
    scheduler_groups.step()

print("\n=== Scheduler Best Practices ===")

print("Scheduler Selection Guidelines:")
print("1. StepLR: Simple decay at fixed intervals")
print("2. MultiStepLR: Decay at specific milestones")
print("3. ExponentialLR: Smooth exponential decay")
print("4. CosineAnnealingLR: Smooth annealing with restarts")
print("5. ReduceLROnPlateau: Adaptive based on validation metrics")
print("6. CyclicLR: Cyclical rates for finding optimal LR")
print("7. OneCycleLR: Super-convergence with single cycle")
print("8. SGDR: Warm restarts for avoiding local minima")

print("\nCommon Patterns:")
print("Training Phase Scheduling:")
print("  1. Warmup (LinearLR): 5-10% of training")
print("  2. Main training (CosineAnnealingLR or StepLR)")
print("  3. Fine-tuning (ReduceLROnPlateau)")

print("\nParameter Recommendations:")
print("StepLR:")
print("  - step_size: 1/3 to 1/2 of total epochs")
print("  - gamma: 0.1 to 0.5")

print("\nCosineAnnealingLR:")
print("  - T_max: Total epochs or cycle length")
print("  - eta_min: 1/100 to 1/1000 of initial LR")

print("\nReduceLROnPlateau:")
print("  - patience: 5-10 epochs for small datasets, 2-5 for large")
print("  - factor: 0.1 to 0.5")
print("  - threshold: 0.01 to 0.001")

print("\nOneCycleLR:")
print("  - max_lr: 10x initial LR (find with LR range test)")
print("  - pct_start: 0.3 (30% of training for warmup)")
print("  - div_factor: 25 (initial_lr = max_lr / div_factor)")

print("\nCommon Mistakes:")
print("1. Not stepping scheduler at right time (epoch vs batch)")
print("2. Using scheduler.step() before optimizer.step()")
print("3. Forgetting to step scheduler in validation phase")
print("4. Wrong scheduler for the optimization landscape")
print("5. Not saving/loading scheduler state")

print("\nDebugging Tips:")
print("1. Plot learning rate vs epoch/step")
print("2. Monitor loss curves with LR changes")
print("3. Use scheduler.get_lr() to check current rates")
print("4. Save scheduler state with model checkpoints")
print("5. Validate scheduler behavior with toy examples")

print("\nAdvanced Techniques:")
print("1. LR range test to find optimal max_lr")
print("2. Different schedules for different layer groups")
print("3. Combining multiple schedulers (Sequential, Chained)")
print("4. Custom schedulers for specific architectures")
print("5. Gradient-based LR adaptation")

print("\n=== Learning Rate Scheduling Complete ===")

# Cleanup
del model, optimizer