#!/usr/bin/env python3
"""PyTorch SGD Optimizer Syntax - SGD variations and parameters"""

import torch
import torch.nn as nn
import torch.optim as optim
import math

print("=== SGD Optimizer Overview ===")

print("SGD variants covered:")
print("1. Basic SGD")
print("2. SGD with Momentum")
print("3. Nesterov Accelerated Gradient (NAG)")
print("4. SGD with Weight Decay")
print("5. SGD parameter groups")
print("6. Custom SGD implementations")
print("7. SGD with different learning rates")
print("8. Momentum variations")

print("\n=== Basic SGD ===")

# Create a simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 50)
        self.linear2 = nn.Linear(50, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# Initialize model and data
model = SimpleModel()
sample_input = torch.randn(32, 10)
sample_target = torch.randn(32, 1)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# Basic SGD optimizer
basic_sgd = optim.SGD(model.parameters(), lr=0.01)

print(f"Basic SGD learning rate: {basic_sgd.param_groups[0]['lr']}")
print(f"Basic SGD momentum: {basic_sgd.param_groups[0]['momentum']}")
print(f"Basic SGD weight decay: {basic_sgd.param_groups[0]['weight_decay']}")

# Training step with basic SGD
def training_step(model, optimizer, input_data, target, loss_fn):
    optimizer.zero_grad()
    output = model(input_data)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()

# Test basic SGD
loss_fn = nn.MSELoss()
basic_loss = training_step(model, basic_sgd, sample_input, sample_target, loss_fn)
print(f"Basic SGD loss: {basic_loss:.6f}")

print("\n=== SGD with Momentum ===")

# Reset model
model = SimpleModel()

# SGD with momentum
momentum_sgd = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)

print(f"Momentum SGD:")
print(f"  Learning rate: {momentum_sgd.param_groups[0]['lr']}")
print(f"  Momentum: {momentum_sgd.param_groups[0]['momentum']}")

# Test different momentum values
momentum_values = [0.0, 0.5, 0.9, 0.99]
momentum_losses = []

for momentum in momentum_values:
    test_model = SimpleModel()
    test_optimizer = optim.SGD(test_model.parameters(), lr=0.01, momentum=momentum)
    
    loss = training_step(test_model, test_optimizer, sample_input, sample_target, loss_fn)
    momentum_losses.append(loss)
    print(f"Momentum {momentum}: Loss = {loss:.6f}")

print("\n=== Nesterov Accelerated Gradient (NAG) ===")

# SGD with Nesterov momentum
nesterov_sgd = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    nesterov=True
)

print(f"Nesterov SGD:")
print(f"  Learning rate: {nesterov_sgd.param_groups[0]['lr']}")
print(f"  Momentum: {nesterov_sgd.param_groups[0]['momentum']}")
print(f"  Nesterov: {nesterov_sgd.param_groups[0]['nesterov']}")

# Compare momentum vs Nesterov
test_model_momentum = SimpleModel()
test_model_nesterov = SimpleModel()

# Make sure both models start with same parameters
test_model_nesterov.load_state_dict(test_model_momentum.state_dict())

momentum_opt = optim.SGD(test_model_momentum.parameters(), lr=0.01, momentum=0.9, nesterov=False)
nesterov_opt = optim.SGD(test_model_nesterov.parameters(), lr=0.01, momentum=0.9, nesterov=True)

momentum_loss = training_step(test_model_momentum, momentum_opt, sample_input, sample_target, loss_fn)
nesterov_loss = training_step(test_model_nesterov, nesterov_opt, sample_input, sample_target, loss_fn)

print(f"Momentum SGD loss: {momentum_loss:.6f}")
print(f"Nesterov SGD loss: {nesterov_loss:.6f}")

print("\n=== SGD with Weight Decay ===")

# SGD with weight decay (L2 regularization)
weight_decay_sgd = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4
)

print(f"Weight Decay SGD:")
print(f"  Learning rate: {weight_decay_sgd.param_groups[0]['lr']}")
print(f"  Momentum: {weight_decay_sgd.param_groups[0]['momentum']}")
print(f"  Weight decay: {weight_decay_sgd.param_groups[0]['weight_decay']}")

# Test different weight decay values
weight_decay_values = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
weight_decay_losses = []

for wd in weight_decay_values:
    test_model = SimpleModel()
    test_optimizer = optim.SGD(test_model.parameters(), lr=0.01, momentum=0.9, weight_decay=wd)
    
    loss = training_step(test_model, test_optimizer, sample_input, sample_target, loss_fn)
    weight_decay_losses.append(loss)
    print(f"Weight decay {wd}: Loss = {loss:.6f}")

print("\n=== SGD Parameter Groups ===")

# Different learning rates for different layers
def create_param_groups(model):
    """Create parameter groups with different learning rates"""
    param_groups = [
        {
            'params': model.linear1.parameters(),
            'lr': 0.01,
            'momentum': 0.9,
            'weight_decay': 1e-4
        },
        {
            'params': model.linear2.parameters(),
            'lr': 0.001,  # Lower learning rate for final layer
            'momentum': 0.95,
            'weight_decay': 1e-3
        }
    ]
    return param_groups

# SGD with parameter groups
model = SimpleModel()
param_groups = create_param_groups(model)

grouped_sgd = optim.SGD(param_groups)

print(f"Parameter Groups SGD:")
for i, group in enumerate(grouped_sgd.param_groups):
    print(f"  Group {i}:")
    print(f"    Learning rate: {group['lr']}")
    print(f"    Momentum: {group['momentum']}")
    print(f"    Weight decay: {group['weight_decay']}")
    print(f"    Parameters: {sum(p.numel() for p in group['params'])}")

# Train with grouped parameters
grouped_loss = training_step(model, grouped_sgd, sample_input, sample_target, loss_fn)
print(f"Grouped SGD loss: {grouped_loss:.6f}")

print("\n=== Custom SGD Implementation ===")

class CustomSGD:
    """Custom SGD implementation for educational purposes"""
    
    def __init__(self, parameters, lr=1e-3, momentum=0, weight_decay=0, nesterov=False):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        
        # Initialize momentum buffers
        self.momentum_buffers = [torch.zeros_like(p) for p in self.parameters]
    
    def zero_grad(self):
        """Zero gradients of all parameters"""
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()
    
    def step(self):
        """Perform a single optimization step"""
        with torch.no_grad():
            for i, param in enumerate(self.parameters):
                if param.grad is None:
                    continue
                
                grad = param.grad
                
                # Apply weight decay
                if self.weight_decay != 0:
                    grad = grad.add(param, alpha=self.weight_decay)
                
                # Apply momentum
                if self.momentum != 0:
                    buf = self.momentum_buffers[i]
                    buf.mul_(self.momentum).add_(grad)
                    
                    if self.nesterov:
                        grad = grad.add(buf, alpha=self.momentum)
                    else:
                        grad = buf
                
                # Update parameters
                param.add_(grad, alpha=-self.lr)

# Test custom SGD
model = SimpleModel()
custom_optimizer = CustomSGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True
)

custom_loss = training_step(model, custom_optimizer, sample_input, sample_target, loss_fn)
print(f"Custom SGD loss: {custom_loss:.6f}")

print("\n=== Advanced SGD Variants ===")

class SGDWithWarmup:
    """SGD with learning rate warmup"""
    
    def __init__(self, optimizer, warmup_steps=1000, warmup_factor=0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        self.step_count = 0
        
        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        """Step with warmup"""
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Warmup phase
            warmup_ratio = self.step_count / self.warmup_steps
            current_factor = self.warmup_factor + (1 - self.warmup_factor) * warmup_ratio
            
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group['lr'] = base_lr * current_factor
        
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

class SGDWithGradientClipping:
    """SGD with gradient clipping"""
    
    def __init__(self, optimizer, max_norm=1.0, norm_type=2):
        self.optimizer = optimizer
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def step(self):
        """Step with gradient clipping"""
        # Clip gradients
        parameters = []
        for group in self.optimizer.param_groups:
            parameters.extend(group['params'])
        
        torch.nn.utils.clip_grad_norm_(parameters, self.max_norm, self.norm_type)
        
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()

# Test advanced variants
model = SimpleModel()
base_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# SGD with warmup
warmup_sgd = SGDWithWarmup(base_optimizer, warmup_steps=10, warmup_factor=0.1)

print(f"SGD with Warmup:")
for step in range(15):
    warmup_sgd.zero_grad()
    output = model(sample_input)
    loss = loss_fn(output, sample_target)
    loss.backward()
    
    lr_before = warmup_sgd.get_lr()[0]
    warmup_sgd.step()
    
    if step % 5 == 0:
        print(f"  Step {step}: LR = {lr_before:.6f}, Loss = {loss.item():.6f}")

# SGD with gradient clipping
model = SimpleModel()
base_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
clipping_sgd = SGDWithGradientClipping(base_optimizer, max_norm=1.0)

clipping_sgd.zero_grad()
output = model(sample_input)
loss = loss_fn(output, sample_target)
loss.backward()

# Check gradient norms before clipping
total_norm_before = 0
for param in model.parameters():
    if param.grad is not None:
        param_norm = param.grad.data.norm(2)
        total_norm_before += param_norm.item() ** 2
total_norm_before = total_norm_before ** (1. / 2)

clipping_sgd.step()

print(f"Gradient clipping: Norm before = {total_norm_before:.6f}")

print("\n=== SGD Learning Rate Schedules ===")

# SGD with different learning rate schedules
model = SimpleModel()
sgd_scheduler = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Step LR scheduler
step_scheduler = optim.lr_scheduler.StepLR(sgd_scheduler, step_size=10, gamma=0.1)

# Exponential LR scheduler
model2 = SimpleModel()
sgd_exp = optim.SGD(model2.parameters(), lr=0.1, momentum=0.9)
exp_scheduler = optim.lr_scheduler.ExponentialLR(sgd_exp, gamma=0.95)

# Cosine Annealing scheduler
model3 = SimpleModel()
sgd_cosine = optim.SGD(model3.parameters(), lr=0.1, momentum=0.9)
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(sgd_cosine, T_max=20)

print(f"Learning Rate Schedules:")
for epoch in range(25):
    # Step LR
    step_lr = sgd_scheduler.param_groups[0]['lr']
    step_scheduler.step()
    
    # Exponential LR
    exp_lr = sgd_exp.param_groups[0]['lr']
    exp_scheduler.step()
    
    # Cosine LR
    cosine_lr = sgd_cosine.param_groups[0]['lr']
    cosine_scheduler.step()
    
    if epoch % 5 == 0:
        print(f"  Epoch {epoch}: Step={step_lr:.6f}, Exp={exp_lr:.6f}, Cosine={cosine_lr:.6f}")

print("\n=== SGD Momentum Variants ===")

class HeavyBallSGD:
    """Heavy Ball SGD implementation"""
    
    def __init__(self, parameters, lr=1e-3, momentum=0.9):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [torch.zeros_like(p) for p in self.parameters]
    
    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()
    
    def step(self):
        with torch.no_grad():
            for i, param in enumerate(self.parameters):
                if param.grad is None:
                    continue
                
                grad = param.grad
                velocity = self.velocities[i]
                
                # Heavy ball update
                velocity.mul_(self.momentum).sub_(grad, alpha=self.lr)
                param.add_(velocity)

class AdaptiveMomentumSGD:
    """SGD with adaptive momentum"""
    
    def __init__(self, parameters, lr=1e-3, initial_momentum=0.9, adaptation_rate=0.01):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = initial_momentum
        self.adaptation_rate = adaptation_rate
        self.velocities = [torch.zeros_like(p) for p in self.parameters]
        self.prev_loss = None
    
    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()
    
    def step(self, current_loss=None):
        # Adapt momentum based on loss change
        if current_loss is not None and self.prev_loss is not None:
            loss_change = current_loss - self.prev_loss
            if loss_change > 0:  # Loss increased
                self.momentum = max(0.1, self.momentum - self.adaptation_rate)
            else:  # Loss decreased
                self.momentum = min(0.99, self.momentum + self.adaptation_rate)
            
            self.prev_loss = current_loss
        elif current_loss is not None:
            self.prev_loss = current_loss
        
        with torch.no_grad():
            for i, param in enumerate(self.parameters):
                if param.grad is None:
                    continue
                
                grad = param.grad
                velocity = self.velocities[i]
                
                velocity.mul_(self.momentum).add_(grad, alpha=-self.lr)
                param.add_(velocity)

# Test momentum variants
models = [SimpleModel() for _ in range(3)]

# Standard SGD with momentum
standard_sgd = optim.SGD(models[0].parameters(), lr=0.01, momentum=0.9)

# Heavy Ball SGD
heavy_ball = HeavyBallSGD(models[1].parameters(), lr=0.01, momentum=0.9)

# Adaptive Momentum SGD
adaptive_momentum = AdaptiveMomentumSGD(models[2].parameters(), lr=0.01, initial_momentum=0.5)

optimizers = [standard_sgd, heavy_ball, adaptive_momentum]
names = ["Standard SGD", "Heavy Ball", "Adaptive Momentum"]

print(f"Momentum Variants Comparison:")
for epoch in range(10):
    losses = []
    
    for i, (model, opt, name) in enumerate(zip(models, optimizers, names)):
        opt.zero_grad()
        output = model(sample_input)
        loss = loss_fn(output, sample_target)
        loss.backward()
        
        if name == "Adaptive Momentum":
            opt.step(loss.item())
        else:
            opt.step()
        
        losses.append(loss.item())
    
    if epoch % 3 == 0:
        print(f"  Epoch {epoch}: {names[0]}={losses[0]:.6f}, {names[1]}={losses[1]:.6f}, {names[2]}={losses[2]:.6f}")

if hasattr(adaptive_momentum, 'momentum'):
    print(f"Final adaptive momentum: {adaptive_momentum.momentum:.6f}")

print("\n=== SGD Best Practices ===")

print("Learning Rate Selection:")
print("1. Start with lr=0.1 for most problems")
print("2. Use lr=0.01 for fine-tuning or sensitive models")
print("3. Scale learning rate with batch size (linear scaling rule)")
print("4. Use learning rate schedules for better convergence")
print("5. Consider warmup for large learning rates")

print("\nMomentum Guidelines:")
print("1. Standard momentum: 0.9 works well for most cases")
print("2. Use Nesterov momentum for faster convergence")
print("3. Lower momentum (0.5-0.7) for noisy gradients")
print("4. Higher momentum (0.95-0.99) for smooth loss landscapes")

print("\nWeight Decay Tips:")
print("1. Start with weight_decay=1e-4")
print("2. Increase for overfitting (1e-3 to 1e-2)")
print("3. Decrease for underfitting (1e-5 to 1e-6)")
print("4. Different values for different layers")
print("5. Don't apply to bias terms typically")

print("\nParameter Group Strategies:")
print("1. Lower learning rates for pre-trained layers")
print("2. Higher learning rates for randomly initialized layers")
print("3. Different weight decay for different layer types")
print("4. Separate momentum for different components")

print("\nCommon Issues:")
print("1. Learning rate too high: Loss explodes or oscillates")
print("2. Learning rate too low: Slow convergence")
print("3. High momentum with high LR: Overshooting")
print("4. No momentum: Slow convergence in narrow valleys")
print("5. High weight decay: Underfitting")

print("\n=== SGD Optimizer Complete ===")

# Memory cleanup
del model, models, sample_input, sample_target