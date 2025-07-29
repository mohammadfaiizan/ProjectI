#!/usr/bin/env python3
"""PyTorch Adam Family Optimizers - Adam, AdamW, AdaMax implementations"""

import torch
import torch.nn as nn
import torch.optim as optim
import math

print("=== Adam Family Optimizers Overview ===")

print("Adam family optimizers:")
print("1. Adam (Adaptive Moment Estimation)")
print("2. AdamW (Adam with Weight Decay)")
print("3. AdaMax (Adam based on infinity norm)")
print("4. AdamWR (AdamW with Restarts)")
print("5. NAdam (Nesterov-accelerated Adam)")
print("6. RAdam (Rectified Adam)")
print("7. AdaBound (Adaptive gradient clipping)")
print("8. Custom Adam implementations")

print("\n=== Basic Adam Optimizer ===")

# Create model for testing
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(20, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# Initialize model and data
model = TestModel()
sample_input = torch.randn(64, 20)
sample_target = torch.randn(64, 1)
loss_fn = nn.MSELoss()

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# Basic Adam optimizer
adam_optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0
)

print(f"Adam optimizer settings:")
print(f"  Learning rate: {adam_optimizer.param_groups[0]['lr']}")
print(f"  Betas: {adam_optimizer.param_groups[0]['betas']}")
print(f"  Epsilon: {adam_optimizer.param_groups[0]['eps']}")
print(f"  Weight decay: {adam_optimizer.param_groups[0]['weight_decay']}")

# Training step function
def training_step(model, optimizer, input_data, target, loss_fn):
    optimizer.zero_grad()
    output = model(input_data)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()

# Test basic Adam
adam_loss = training_step(model, adam_optimizer, sample_input, sample_target, loss_fn)
print(f"Adam loss after one step: {adam_loss:.6f}")

print("\n=== Adam Parameter Analysis ===")

# Test different learning rates
learning_rates = [0.1, 0.01, 0.001, 0.0001]
lr_losses = []

for lr in learning_rates:
    test_model = TestModel()
    test_optimizer = optim.Adam(test_model.parameters(), lr=lr)
    
    loss = training_step(test_model, test_optimizer, sample_input, sample_target, loss_fn)
    lr_losses.append(loss)
    print(f"Learning rate {lr}: Loss = {loss:.6f}")

# Test different beta values
beta_configs = [(0.9, 0.999), (0.95, 0.999), (0.9, 0.99), (0.8, 0.9)]
beta_losses = []

print(f"\nBeta parameter analysis:")
for betas in beta_configs:
    test_model = TestModel()
    test_optimizer = optim.Adam(test_model.parameters(), lr=0.001, betas=betas)
    
    loss = training_step(test_model, test_optimizer, sample_input, sample_target, loss_fn)
    beta_losses.append(loss)
    print(f"Betas {betas}: Loss = {loss:.6f}")

# Test different epsilon values
epsilon_values = [1e-8, 1e-7, 1e-6, 1e-4]
eps_losses = []

print(f"\nEpsilon parameter analysis:")
for eps in epsilon_values:
    test_model = TestModel()
    test_optimizer = optim.Adam(test_model.parameters(), lr=0.001, eps=eps)
    
    loss = training_step(test_model, test_optimizer, sample_input, sample_target, loss_fn)
    eps_losses.append(loss)
    print(f"Epsilon {eps}: Loss = {loss:.6f}")

print("\n=== AdamW Optimizer ===")

# AdamW - Adam with decoupled weight decay
model_adamw = TestModel()
adamw_optimizer = optim.AdamW(
    model_adamw.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    amsgrad=False
)

print(f"AdamW optimizer settings:")
print(f"  Learning rate: {adamw_optimizer.param_groups[0]['lr']}")
print(f"  Betas: {adamw_optimizer.param_groups[0]['betas']}")
print(f"  Weight decay: {adamw_optimizer.param_groups[0]['weight_decay']}")
print(f"  AMSGrad: {adamw_optimizer.param_groups[0]['amsgrad']}")

# Compare Adam vs AdamW
model_adam = TestModel()
model_adamw_test = TestModel()

# Ensure same initial parameters
model_adamw_test.load_state_dict(model_adam.state_dict())

adam_comp = optim.Adam(model_adam.parameters(), lr=0.001, weight_decay=0.01)
adamw_comp = optim.AdamW(model_adamw_test.parameters(), lr=0.001, weight_decay=0.01)

adam_loss_comp = training_step(model_adam, adam_comp, sample_input, sample_target, loss_fn)
adamw_loss_comp = training_step(model_adamw_test, adamw_comp, sample_input, sample_target, loss_fn)

print(f"Adam with weight_decay=0.01: {adam_loss_comp:.6f}")
print(f"AdamW with weight_decay=0.01: {adamw_loss_comp:.6f}")

# Test different weight decay values for AdamW
weight_decays = [0.0, 0.001, 0.01, 0.1]
wd_losses = []

print(f"\nAdamW weight decay analysis:")
for wd in weight_decays:
    test_model = TestModel()
    test_optimizer = optim.AdamW(test_model.parameters(), lr=0.001, weight_decay=wd)
    
    loss = training_step(test_model, test_optimizer, sample_input, sample_target, loss_fn)
    wd_losses.append(loss)
    print(f"Weight decay {wd}: Loss = {loss:.6f}")

print("\n=== AdaMax Optimizer ===")

# AdaMax - Adam variant based on infinity norm
model_adamax = TestModel()
adamax_optimizer = optim.Adamax(
    model_adamax.parameters(),
    lr=0.002,  # AdaMax typically uses higher learning rate
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0
)

print(f"AdaMax optimizer settings:")
print(f"  Learning rate: {adamax_optimizer.param_groups[0]['lr']}")
print(f"  Betas: {adamax_optimizer.param_groups[0]['betas']}")
print(f"  Epsilon: {adamax_optimizer.param_groups[0]['eps']}")

adamax_loss = training_step(model_adamax, adamax_optimizer, sample_input, sample_target, loss_fn)
print(f"AdaMax loss: {adamax_loss:.6f}")

# Compare Adam vs AdaMax
model_adam_vs = TestModel()
model_adamax_vs = TestModel()
model_adamax_vs.load_state_dict(model_adam_vs.state_dict())

adam_vs = optim.Adam(model_adam_vs.parameters(), lr=0.001)
adamax_vs = optim.Adamax(model_adamax_vs.parameters(), lr=0.002)

print(f"\nAdam vs AdaMax comparison (different LRs):")
for epoch in range(10):
    adam_loss_vs = training_step(model_adam_vs, adam_vs, sample_input, sample_target, loss_fn)
    adamax_loss_vs = training_step(model_adamax_vs, adamax_vs, sample_input, sample_target, loss_fn)
    
    if epoch % 3 == 0:
        print(f"  Epoch {epoch}: Adam={adam_loss_vs:.6f}, AdaMax={adamax_loss_vs:.6f}")

print("\n=== Custom Adam Implementation ===")

class CustomAdam:
    """Custom Adam implementation for educational purposes"""
    
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize moment estimates
        self.m = [torch.zeros_like(p) for p in self.parameters]  # First moment
        self.v = [torch.zeros_like(p) for p in self.parameters]  # Second moment
        self.step_count = 0
    
    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()
    
    def step(self):
        self.step_count += 1
        
        with torch.no_grad():
            for i, param in enumerate(self.parameters):
                if param.grad is None:
                    continue
                
                grad = param.grad
                
                # Add weight decay
                if self.weight_decay != 0:
                    grad = grad.add(param, alpha=self.weight_decay)
                
                # Update biased first moment estimate
                self.m[i].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
                
                # Update biased second moment estimate
                self.v[i].mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
                
                # Bias correction
                bias_correction1 = 1 - self.beta1 ** self.step_count
                bias_correction2 = 1 - self.beta2 ** self.step_count
                
                # Apply bias correction
                corrected_m = self.m[i] / bias_correction1
                corrected_v = self.v[i] / bias_correction2
                
                # Update parameters
                param.addcdiv_(corrected_m, corrected_v.sqrt().add_(self.eps), value=-self.lr)

# Test custom Adam
model_custom = TestModel()
custom_adam = CustomAdam(
    model_custom.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0
)

custom_loss = training_step(model_custom, custom_adam, sample_input, sample_target, loss_fn)
print(f"Custom Adam loss: {custom_loss:.6f}")

print("\n=== Advanced Adam Variants ===")

class NAdam:
    """Nesterov-accelerated Adam"""
    
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.m = [torch.zeros_like(p) for p in self.parameters]
        self.v = [torch.zeros_like(p) for p in self.parameters]
        self.step_count = 0
    
    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()
    
    def step(self):
        self.step_count += 1
        
        with torch.no_grad():
            for i, param in enumerate(self.parameters):
                if param.grad is None:
                    continue
                
                grad = param.grad
                
                if self.weight_decay != 0:
                    grad = grad.add(param, alpha=self.weight_decay)
                
                # Update moments
                self.m[i].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
                self.v[i].mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
                
                # Bias correction
                bias_correction1 = 1 - self.beta1 ** self.step_count
                bias_correction2 = 1 - self.beta2 ** self.step_count
                
                # Nesterov-style update
                corrected_m = (self.beta1 * self.m[i] / bias_correction1 + 
                              (1 - self.beta1) * grad / bias_correction1)
                corrected_v = self.v[i] / bias_correction2
                
                # Update parameters
                param.addcdiv_(corrected_m, corrected_v.sqrt().add_(self.eps), value=-self.lr)

class RAdam:
    """Rectified Adam"""
    
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.m = [torch.zeros_like(p) for p in self.parameters]
        self.v = [torch.zeros_like(p) for p in self.parameters]
        self.step_count = 0
        
        # RAdam specific
        self.rho_inf = 2 / (1 - self.beta2) - 1
    
    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()
    
    def step(self):
        self.step_count += 1
        
        with torch.no_grad():
            for i, param in enumerate(self.parameters):
                if param.grad is None:
                    continue
                
                grad = param.grad
                
                if self.weight_decay != 0:
                    grad = grad.add(param, alpha=self.weight_decay)
                
                # Update moments
                self.m[i].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
                self.v[i].mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
                
                # Bias correction for first moment
                bias_correction1 = 1 - self.beta1 ** self.step_count
                corrected_m = self.m[i] / bias_correction1
                
                # Compute length of approximated SMA
                rho_t = self.rho_inf - 2 * self.step_count * (self.beta2 ** self.step_count) / (1 - self.beta2 ** self.step_count)
                
                if rho_t > 4:  # Use adaptive learning rate
                    # Bias correction for second moment
                    bias_correction2 = 1 - self.beta2 ** self.step_count
                    corrected_v = self.v[i] / bias_correction2
                    
                    # Compute variance rectification term
                    r_t = math.sqrt(((rho_t - 4) * (rho_t - 2) * self.rho_inf) / 
                                   ((self.rho_inf - 4) * (self.rho_inf - 2) * rho_t))
                    
                    # Update parameters with rectification
                    param.addcdiv_(corrected_m, corrected_v.sqrt().add_(self.eps), value=-self.lr * r_t)
                else:
                    # Use non-adaptive update
                    param.add_(corrected_m, alpha=-self.lr)

class AdaBound:
    """AdaBound - Adaptive gradient clipping"""
    
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0, final_lr=0.1, gamma=1e-3):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.final_lr = final_lr
        self.gamma = gamma
        
        self.m = [torch.zeros_like(p) for p in self.parameters]
        self.v = [torch.zeros_like(p) for p in self.parameters]
        self.step_count = 0
    
    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()
    
    def step(self):
        self.step_count += 1
        
        with torch.no_grad():
            for i, param in enumerate(self.parameters):
                if param.grad is None:
                    continue
                
                grad = param.grad
                
                if self.weight_decay != 0:
                    grad = grad.add(param, alpha=self.weight_decay)
                
                # Update moments
                self.m[i].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
                self.v[i].mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
                
                # Bias correction
                bias_correction1 = 1 - self.beta1 ** self.step_count
                bias_correction2 = 1 - self.beta2 ** self.step_count
                
                corrected_m = self.m[i] / bias_correction1
                corrected_v = self.v[i] / bias_correction2
                
                # Compute bounds
                final_lr = self.final_lr * self.lr / self.lr  # Normalize
                lower_bound = final_lr * (1 - 1 / (self.gamma * self.step_count + 1))
                upper_bound = final_lr * (1 + 1 / (self.gamma * self.step_count))
                
                # Clip step size
                step_size = self.lr / corrected_v.sqrt().add_(self.eps)
                step_size = torch.clamp(step_size, lower_bound, upper_bound)
                
                # Update parameters
                param.add_(corrected_m * step_size, alpha=-1)

# Test advanced variants
models_advanced = [TestModel() for _ in range(4)]
names_advanced = ["Standard Adam", "NAdam", "RAdam", "AdaBound"]

# Initialize with same weights
for model in models_advanced[1:]:
    model.load_state_dict(models_advanced[0].state_dict())

optimizers_advanced = [
    optim.Adam(models_advanced[0].parameters(), lr=0.001),
    NAdam(models_advanced[1].parameters(), lr=0.001),
    RAdam(models_advanced[2].parameters(), lr=0.001),
    AdaBound(models_advanced[3].parameters(), lr=0.001)
]

print(f"Advanced Adam variants comparison:")
for epoch in range(10):
    losses_advanced = []
    
    for model, opt in zip(models_advanced, optimizers_advanced):
        opt.zero_grad()
        output = model(sample_input)
        loss = loss_fn(output, sample_target)
        loss.backward()
        opt.step()
        losses_advanced.append(loss.item())
    
    if epoch % 3 == 0:
        print(f"  Epoch {epoch}:")
        for name, loss in zip(names_advanced, losses_advanced):
            print(f"    {name}: {loss:.6f}")

print("\n=== Adam with Learning Rate Scheduling ===")

# Adam with different schedulers
model_sched = TestModel()
adam_sched = optim.Adam(model_sched.parameters(), lr=0.01)

# Cosine Annealing
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(adam_sched, T_max=20)

# Reduce on Plateau
model_plateau = TestModel()
adam_plateau = optim.Adam(model_plateau.parameters(), lr=0.01)
plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(adam_plateau, patience=3, factor=0.5)

# OneCycleLR
model_onecycle = TestModel()
adam_onecycle = optim.Adam(model_onecycle.parameters(), lr=0.01)
onecycle_scheduler = optim.lr_scheduler.OneCycleLR(adam_onecycle, max_lr=0.1, steps_per_epoch=1, epochs=20)

print(f"Adam with schedulers:")
for epoch in range(20):
    # Cosine annealing
    cosine_lr = adam_sched.param_groups[0]['lr']
    cosine_scheduler.step()
    
    # One cycle
    onecycle_lr = adam_onecycle.param_groups[0]['lr']
    onecycle_scheduler.step()
    
    # Plateau (needs loss value)
    plateau_lr = adam_plateau.param_groups[0]['lr']
    if epoch > 0:  # Need at least one loss value
        plateau_scheduler.step(loss.item())
    
    if epoch % 5 == 0:
        print(f"  Epoch {epoch}: Cosine={cosine_lr:.6f}, OneCycle={onecycle_lr:.6f}, Plateau={plateau_lr:.6f}")

print("\n=== Adam Parameter Groups ===")

# Different learning rates for different layers
def create_adam_param_groups(model):
    """Create parameter groups for Adam optimizer"""
    param_groups = [
        {
            'params': [p for name, p in model.named_parameters() if 'linear1' in name],
            'lr': 0.001,
            'weight_decay': 0.01
        },
        {
            'params': [p for name, p in model.named_parameters() if 'linear2' in name],
            'lr': 0.0005,
            'weight_decay': 0.005
        },
        {
            'params': [p for name, p in model.named_parameters() if 'linear3' in name],
            'lr': 0.0001,
            'weight_decay': 0.001
        }
    ]
    return param_groups

model_groups = TestModel()
param_groups = create_adam_param_groups(model_groups)
adam_groups = optim.Adam(param_groups)

print(f"Adam with parameter groups:")
for i, group in enumerate(adam_groups.param_groups):
    print(f"  Group {i}:")
    print(f"    Learning rate: {group['lr']}")
    print(f"    Weight decay: {group['weight_decay']}")
    print(f"    Parameters: {sum(p.numel() for p in group['params'])}")

print("\n=== Adam Best Practices ===")

print("Learning Rate Guidelines:")
print("1. Start with lr=0.001 for Adam (default)")
print("2. Use lr=0.0001 for fine-tuning")
print("3. AdaMax can use higher lr (0.002-0.01)")
print("4. Scale learning rate with effective batch size")
print("5. Use learning rate scheduling for better convergence")

print("\nBeta Parameters:")
print("1. β₁=0.9: Good default for most problems")
print("2. β₁=0.95: For sparse gradients or RNNs")
print("3. β₂=0.999: Standard second moment decay")
print("4. β₂=0.99: For faster adaptation")
print("5. Lower β₁ for noisy gradients")

print("\nWeight Decay:")
print("1. Use AdamW for proper weight decay")
print("2. Start with weight_decay=0.01")
print("3. Increase for overfitting (0.1)")
print("4. Decrease for underfitting (0.001)")
print("5. Don't apply to bias and normalization parameters")

print("\nCommon Issues:")
print("1. Learning rate too high: Loss explodes early")
print("2. Learning rate too low: Very slow convergence")
print("3. β₂ too low: Noisy updates")
print("4. Weight decay too high: Poor performance")
print("5. No learning rate decay: Suboptimal final performance")

print("\nOptimizer Selection:")
print("1. Adam: General purpose, good default")
print("2. AdamW: When weight decay is important")
print("3. AdaMax: For sparse embeddings or very high dimensional problems")
print("4. RAdam: For better warm-up behavior")
print("5. NAdam: For problems requiring Nesterov acceleration")

print("\n=== Adam Family Optimizers Complete ===")

# Memory cleanup
del models_advanced, model, sample_input, sample_target