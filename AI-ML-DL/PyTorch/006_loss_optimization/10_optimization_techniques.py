#!/usr/bin/env python3
"""PyTorch Optimization Techniques - Warm-up, restarts, cyclical learning"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import math
import numpy as np

print("=== Optimization Techniques Overview ===")

print("Advanced optimization techniques:")
print("1. Learning Rate Warm-up")
print("2. Cosine Annealing with Warm Restarts (SGDR)")
print("3. Cyclical Learning Rates")
print("4. One Cycle Learning")
print("5. Progressive Resizing")
print("6. Stochastic Weight Averaging (SWA)")
print("7. Lookahead Optimizer")
print("8. Gradient Accumulation")
print("9. Learning Rate Range Test")
print("10. Advanced Regularization Techniques")

print("\n=== Model Setup ===")

class OptimizationTestModel(nn.Module):
    def __init__(self, input_size=20, hidden_size=64):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size // 2)
        self.linear3 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
    
    def forward(self, x):
        x = self.relu(self.batch_norm(self.linear1(x)))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# Create model and data
model = OptimizationTestModel()
sample_input = torch.randn(64, 20)
sample_target = torch.randn(64, 1)
loss_fn = nn.MSELoss()

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

print("\n=== Learning Rate Warm-up ===")

class LinearWarmupScheduler:
    """Linear learning rate warm-up scheduler"""
    
    def __init__(self, optimizer, warmup_steps, target_lr=None):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.target_lrs = target_lr if target_lr else self.base_lrs
        
        # Set initial learning rate to 0
        for group in optimizer.param_groups:
            group['lr'] = 0.0
    
    def step(self):
        if self.current_step < self.warmup_steps:
            # Linear warm-up
            for i, group in enumerate(self.optimizer.param_groups):
                lr = self.target_lrs[i] * (self.current_step + 1) / self.warmup_steps
                group['lr'] = lr
        
        self.current_step += 1
    
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

class CosineWarmupScheduler:
    """Cosine learning rate warm-up scheduler"""
    
    def __init__(self, optimizer, warmup_steps, max_lr, min_lr=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0
    
    def step(self):
        if self.current_step < self.warmup_steps:
            # Cosine warm-up
            progress = self.current_step / self.warmup_steps
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 - math.cos(math.pi * progress))
            
            for group in self.optimizer.param_groups:
                group['lr'] = lr
        
        self.current_step += 1
    
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

# Test different warm-up strategies
print("Testing warm-up strategies:")

# Linear warm-up
optimizer_linear = optim.Adam(model.parameters(), lr=0.001)
linear_warmup = LinearWarmupScheduler(optimizer_linear, warmup_steps=10, target_lr=[0.01])

print("Linear warm-up progression:")
for step in range(15):
    current_lr = linear_warmup.get_lr()[0]
    print(f"  Step {step:2d}: LR = {current_lr:.6f}")
    linear_warmup.step()

# Cosine warm-up
optimizer_cosine = optim.Adam(model.parameters(), lr=0.001)
cosine_warmup = CosineWarmupScheduler(optimizer_cosine, warmup_steps=10, max_lr=0.01)

print("\nCosine warm-up progression:")
for step in range(15):
    current_lr = cosine_warmup.get_lr()[0]
    print(f"  Step {step:2d}: LR = {current_lr:.6f}")
    cosine_warmup.step()

print("\n=== Stochastic Gradient Descent with Warm Restarts (SGDR) ===")

# SGDR implementation
optimizer_sgdr = optim.Adam(model.parameters(), lr=0.1)
sgdr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer_sgdr,
    T_0=10,      # Initial restart period
    T_mult=2,    # Factor to increase T_0 after each restart
    eta_min=0.001 # Minimum learning rate
)

print("SGDR learning rate progression:")
print("T_0=10, T_mult=2 (restart periods: 10, 20, 40, ...)")

restart_points = []
for epoch in range(50):
    current_lr = optimizer_sgdr.param_groups[0]['lr']
    
    # Detect restarts (when LR jumps back up)
    if epoch > 0 and current_lr > prev_lr * 2:
        restart_points.append(epoch)
        print(f"  Epoch {epoch:2d}: LR = {current_lr:.6f} [RESTART]")
    elif epoch % 5 == 0:
        print(f"  Epoch {epoch:2d}: LR = {current_lr:.6f}")
    
    prev_lr = current_lr
    sgdr_scheduler.step()

print(f"Restart points detected at epochs: {restart_points}")

print("\n=== Cyclical Learning Rates ===")

# Cyclical LR with different policies
optimizer_cyclic = optim.Adam(model.parameters(), lr=0.001)

# Triangular policy
triangular_scheduler = lr_scheduler.CyclicLR(
    optimizer_cyclic,
    base_lr=0.001,
    max_lr=0.01,
    step_size_up=5,
    mode='triangular',
    cycle_momentum=False
)

print("Cyclical LR (Triangular policy):")
for epoch in range(20):
    current_lr = optimizer_cyclic.param_groups[0]['lr']
    print(f"  Epoch {epoch:2d}: LR = {current_lr:.6f}")
    
    # Simulate batches per epoch
    for batch in range(2):
        triangular_scheduler.step()

# Triangular2 policy (amplitude decreases by half each cycle)
optimizer_cyclic2 = optim.Adam(model.parameters(), lr=0.001)
triangular2_scheduler = lr_scheduler.CyclicLR(
    optimizer_cyclic2,
    base_lr=0.001,
    max_lr=0.01,
    step_size_up=5,
    mode='triangular2',
    cycle_momentum=False
)

print("\nCyclical LR (Triangular2 policy - decreasing amplitude):")
for epoch in range(20):
    current_lr = optimizer_cyclic2.param_groups[0]['lr']
    print(f"  Epoch {epoch:2d}: LR = {current_lr:.6f}")
    
    for batch in range(2):
        triangular2_scheduler.step()

print("\n=== One Cycle Learning ===")

# One Cycle Policy implementation
optimizer_onecycle = optim.Adam(model.parameters(), lr=0.001)
onecycle_scheduler = lr_scheduler.OneCycleLR(
    optimizer_onecycle,
    max_lr=0.1,
    steps_per_epoch=10,
    epochs=10,
    pct_start=0.3,      # 30% of training for warm-up
    anneal_strategy='cos',
    div_factor=25.0,    # initial_lr = max_lr / div_factor
    final_div_factor=10000.0  # final_lr = initial_lr / final_div_factor
)

print("One Cycle Learning Rate Policy:")
print("30% warm-up, 70% annealing with cosine strategy")

total_steps = 10 * 10  # epochs * steps_per_epoch
for step in range(total_steps):
    epoch = step // 10
    batch = step % 10
    current_lr = optimizer_onecycle.param_groups[0]['lr']
    
    if batch == 0:  # Print at start of each epoch
        phase = "Warm-up" if step < total_steps * 0.3 else "Annealing"
        print(f"  Epoch {epoch:2d}: LR = {current_lr:.6f} [{phase}]")
    
    onecycle_scheduler.step()

print("\n=== Stochastic Weight Averaging (SWA) ===")

class SWAOptimizer:
    """Stochastic Weight Averaging implementation"""
    
    def __init__(self, optimizer, swa_start=5, swa_freq=1, swa_lr=0.01):
        self.optimizer = optimizer
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_lr = swa_lr
        self.step_count = 0
        self.swa_state = {}
        self.n_averaged = 0
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def step(self):
        self.optimizer.step()
        self.step_count += 1
        
        # Start SWA averaging
        if self.step_count >= self.swa_start and (self.step_count - self.swa_start) % self.swa_freq == 0:
            self.update_swa()
    
    def update_swa(self):
        """Update SWA moving average"""
        if not self.swa_state:
            # Initialize SWA state
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    self.swa_state[p] = p.data.clone()
            self.n_averaged = 1
        else:
            # Update moving average
            self.n_averaged += 1
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    self.swa_state[p] = (self.swa_state[p] * (self.n_averaged - 1) + p.data) / self.n_averaged
    
    def swap_swa_sgd(self):
        """Swap current parameters with SWA parameters"""
        if self.swa_state:
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    tmp = p.data.clone()
                    p.data = self.swa_state[p]
                    self.swa_state[p] = tmp
    
    def set_swa_lr(self):
        """Set learning rate for SWA phase"""
        for group in self.optimizer.param_groups:
            group['lr'] = self.swa_lr

# Test SWA
print("Stochastic Weight Averaging simulation:")
model_swa = OptimizationTestModel()
base_optimizer = optim.SGD(model_swa.parameters(), lr=0.1, momentum=0.9)
swa_optimizer = SWAOptimizer(base_optimizer, swa_start=5, swa_freq=1, swa_lr=0.01)

for epoch in range(15):
    # Simulate training
    swa_optimizer.zero_grad()
    output = model_swa(sample_input)
    loss = loss_fn(output, sample_target)
    loss.backward()
    swa_optimizer.step()
    
    if epoch >= swa_optimizer.swa_start:
        status = f"SWA active (n_averaged={swa_optimizer.n_averaged})"
        if epoch == swa_optimizer.swa_start:
            swa_optimizer.set_swa_lr()
    else:
        status = "Regular training"
    
    print(f"  Epoch {epoch:2d}: Loss = {loss.item():.6f} [{status}]")

print("\n=== Lookahead Optimizer ===")

class LookaheadOptimizer:
    """Lookahead optimizer wrapper"""
    
    def __init__(self, base_optimizer, k=5, alpha=0.5):
        self.base_optimizer = base_optimizer
        self.k = k  # Update frequency
        self.alpha = alpha  # Step size
        self.step_count = 0
        
        # Store slow weights
        self.slow_weights = {}
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                self.slow_weights[p] = p.data.clone()
    
    def zero_grad(self):
        self.base_optimizer.zero_grad()
    
    def step(self):
        self.base_optimizer.step()
        self.step_count += 1
        
        if self.step_count % self.k == 0:
            self.update_slow_weights()
    
    def update_slow_weights(self):
        """Update slow weights using lookahead rule"""
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                # Lookahead update: slow_weight = slow_weight + alpha * (fast_weight - slow_weight)
                self.slow_weights[p] = self.slow_weights[p] + self.alpha * (p.data - self.slow_weights[p])
                p.data = self.slow_weights[p]

# Test Lookahead
print("Lookahead optimizer simulation:")
model_lookahead = OptimizationTestModel()
base_opt = optim.Adam(model_lookahead.parameters(), lr=0.01)
lookahead_opt = LookaheadOptimizer(base_opt, k=5, alpha=0.5)

for step in range(15):
    lookahead_opt.zero_grad()
    output = model_lookahead(sample_input)
    loss = loss_fn(output, sample_target)
    loss.backward()
    lookahead_opt.step()
    
    status = "Lookahead update" if step % lookahead_opt.k == 0 and step > 0 else "Fast update"
    print(f"  Step {step:2d}: Loss = {loss.item():.6f} [{status}]")

print("\n=== Gradient Accumulation ===")

class GradientAccumulator:
    """Gradient accumulation for effective larger batch sizes"""
    
    def __init__(self, model, optimizer, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def step(self, input_data, targets, loss_fn):
        # Forward pass
        outputs = self.model(input_data)
        loss = loss_fn(outputs, targets)
        
        # Scale loss by accumulation steps
        loss = loss / self.accumulation_steps
        
        # Backward pass
        loss.backward()
        
        self.current_step += 1
        
        # Accumulate gradients and step optimizer
        if self.current_step % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            return True, loss.item() * self.accumulation_steps
        
        return False, loss.item() * self.accumulation_steps

# Test gradient accumulation
print("Gradient accumulation simulation:")
print("Effective batch size = batch_size * accumulation_steps")

model_accum = OptimizationTestModel()
optimizer_accum = optim.Adam(model_accum.parameters(), lr=0.01)
accumulator = GradientAccumulator(model_accum, optimizer_accum, accumulation_steps=4)

for step in range(12):
    # Use smaller batches
    mini_batch_input = sample_input[:16]  # Smaller batch
    mini_batch_target = sample_target[:16]
    
    stepped, loss = accumulator.step(mini_batch_input, mini_batch_target, loss_fn)
    
    status = "OPTIMIZER STEP" if stepped else "accumulating"
    effective_batch = 16 * (step % 4 + 1) if not stepped else 64
    print(f"  Step {step:2d}: Loss = {loss:.6f}, Effective batch = {effective_batch:2d} [{status}]")

print("\n=== Learning Rate Range Test ===")

class LRRangeFinder:
    """Learning rate range test to find optimal learning rates"""
    
    def __init__(self, model, optimizer, loss_fn, min_lr=1e-7, max_lr=10, num_iter=100):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_iter = num_iter
        
        # Save initial state
        self.initial_state = model.state_dict()
        
        # Learning rates to test
        self.lrs = torch.logspace(math.log10(min_lr), math.log10(max_lr), num_iter)
        self.losses = []
    
    def range_test(self, train_loader_fn, reset_model=True):
        """Perform learning rate range test"""
        if reset_model:
            self.model.load_state_dict(self.initial_state)
        
        # Test each learning rate
        for i, lr in enumerate(self.lrs):
            # Set learning rate
            for group in self.optimizer.param_groups:
                group['lr'] = lr.item()
            
            # Get training data
            input_data, targets = train_loader_fn()
            
            # Training step
            self.optimizer.zero_grad()
            outputs = self.model(input_data)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            self.losses.append(loss.item())
            
            # Stop if loss explodes
            if i > 10 and loss.item() > 4 * min(self.losses):
                break
        
        return self.lrs[:len(self.losses)], self.losses
    
    def plot_results(self):
        """Find suggested learning rates"""
        if not self.losses:
            return None
        
        # Find minimum loss
        min_loss_idx = torch.tensor(self.losses).argmin()
        min_lr = self.lrs[min_loss_idx]
        
        # Find learning rate with steepest gradient (fastest decrease)
        if len(self.losses) > 10:
            gradients = []
            for i in range(1, len(self.losses) - 1):
                grad = (self.losses[i+1] - self.losses[i-1]) / 2
                gradients.append(grad)
            
            steepest_idx = torch.tensor(gradients).argmin() + 1
            steepest_lr = self.lrs[steepest_idx]
        else:
            steepest_lr = min_lr
        
        return {
            'min_loss_lr': min_lr.item(),
            'steepest_grad_lr': steepest_lr.item(),
            'suggested_lr': steepest_lr.item() / 10  # Conservative suggestion
        }

# Test LR range finder
print("Learning Rate Range Test:")

def sample_data_loader():
    """Simulate data loader"""
    return torch.randn(32, 20), torch.randn(32, 1)

model_lr_test = OptimizationTestModel()
optimizer_lr_test = optim.Adam(model_lr_test.parameters(), lr=0.001)
lr_finder = LRRangeFinder(model_lr_test, optimizer_lr_test, loss_fn, 
                         min_lr=1e-5, max_lr=1, num_iter=50)

lrs_tested, losses = lr_finder.range_test(sample_data_loader)
results = lr_finder.plot_results()

print(f"Tested {len(lrs_tested)} learning rates from {lrs_tested[0]:.2e} to {lrs_tested[-1]:.2e}")
print(f"Loss range: {min(losses):.6f} to {max(losses):.6f}")

if results:
    print("Suggested learning rates:")
    print(f"  Minimum loss LR: {results['min_loss_lr']:.2e}")
    print(f"  Steepest gradient LR: {results['steepest_grad_lr']:.2e}")
    print(f"  Conservative suggestion: {results['suggested_lr']:.2e}")

print("\n=== Progressive Learning Techniques ===")

class ProgressiveResizing:
    """Progressive resizing for computer vision tasks"""
    
    def __init__(self, initial_size=64, final_size=224, resize_epochs=[10, 20, 30]):
        self.initial_size = initial_size
        self.final_size = final_size
        self.resize_epochs = resize_epochs
        self.current_size = initial_size
        self.sizes = [initial_size] + [final_size // (2 ** (len(resize_epochs) - i - 1)) 
                                      for i in range(len(resize_epochs))]
    
    def get_current_size(self, epoch):
        """Get current image size for the epoch"""
        for i, resize_epoch in enumerate(self.resize_epochs):
            if epoch < resize_epoch:
                return self.sizes[i]
        return self.final_size
    
    def should_resize(self, epoch):
        """Check if should resize at this epoch"""
        return epoch in self.resize_epochs

# Progressive learning rate decay
class ProgressiveLearningRate:
    """Progressive learning rate adjustment"""
    
    def __init__(self, optimizer, milestones, factors):
        self.optimizer = optimizer
        self.milestones = milestones
        self.factors = factors
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch):
        """Adjust learning rate based on epoch"""
        factor = 1.0
        for milestone, f in zip(self.milestones, self.factors):
            if epoch >= milestone:
                factor *= f
        
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.base_lrs[i] * factor

# Test progressive techniques
print("Progressive Learning Techniques:")

# Progressive resizing
progressive_resize = ProgressiveResizing(
    initial_size=32, 
    final_size=128, 
    resize_epochs=[5, 10, 15]
)

print("Progressive Resizing Schedule:")
for epoch in range(20):
    size = progressive_resize.get_current_size(epoch)
    should_resize = progressive_resize.should_resize(epoch)
    status = " [RESIZE]" if should_resize else ""
    print(f"  Epoch {epoch:2d}: Image size = {size:3d}x{size:3d}{status}")

# Progressive learning rate
optimizer_prog = optim.Adam(model.parameters(), lr=0.01)
progressive_lr = ProgressiveLearningRate(
    optimizer_prog, 
    milestones=[5, 10, 15], 
    factors=[0.5, 0.5, 0.1]
)

print("\nProgressive Learning Rate Schedule:")
for epoch in range(20):
    progressive_lr.step(epoch)
    current_lr = optimizer_prog.param_groups[0]['lr']
    print(f"  Epoch {epoch:2d}: LR = {current_lr:.6f}")

print("\n=== Advanced Regularization Techniques ===")

class DropConnect:
    """DropConnect regularization"""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def apply(self, weight):
        """Apply DropConnect to weight matrix"""
        if self.training:
            mask = torch.bernoulli(torch.full_like(weight, 1 - self.p))
            return weight * mask / (1 - self.p)
        return weight

class LayerWiseAdaptiveRateScaling:
    """LARS optimizer component"""
    
    def __init__(self, trust_coeff=0.02, eps=1e-8):
        self.trust_coeff = trust_coeff
        self.eps = eps
    
    def get_lars_lr(self, param, grad):
        """Compute LARS learning rate for parameter"""
        param_norm = param.norm()
        grad_norm = grad.norm()
        
        if param_norm > 0 and grad_norm > 0:
            lars_lr = self.trust_coeff * param_norm / (grad_norm + self.eps)
        else:
            lars_lr = 1.0
        
        return lars_lr

# Label smoothing
class LabelSmoothingLoss(nn.Module):
    """Label smoothing regularization"""
    
    def __init__(self, smoothing=0.1, num_classes=10):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
    
    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_target = torch.full_like(pred, self.smoothing / (self.num_classes - 1))
        smooth_target.scatter_(1, target.unsqueeze(1), confidence)
        
        log_probs = torch.log_softmax(pred, dim=1)
        loss = -(smooth_target * log_probs).sum(dim=1).mean()
        return loss

print("Advanced Regularization Examples:")

# DropConnect example
dropconnect = DropConnect(p=0.3)
weight_example = torch.randn(10, 5)
print(f"Original weight norm: {weight_example.norm():.6f}")

# Simulate training mode
dropconnect.training = True
dropped_weight = dropconnect.apply(weight_example)
print(f"DropConnect weight norm: {dropped_weight.norm():.6f}")

# LARS example
lars = LayerWiseAdaptiveRateScaling(trust_coeff=0.02)
param_example = torch.randn(5, 5)
grad_example = torch.randn(5, 5) * 0.1
lars_lr = lars.get_lars_lr(param_example, grad_example)
print(f"LARS learning rate multiplier: {lars_lr:.6f}")

print("\n=== Optimization Techniques Best Practices ===")

print("Warm-up Guidelines:")
print("1. Use 5-10% of total training for warm-up")
print("2. Linear warm-up for most cases, cosine for special cases")
print("3. Essential for large batch training")
print("4. Helps stabilize training with high learning rates")
print("5. Particularly important for transformer models")

print("\nCyclical and Restart Techniques:")
print("1. SGDR: Good for finding better minima")
print("2. Cyclical LR: Helps escape local minima")
print("3. One Cycle: Fastest convergence for many tasks")
print("4. Use LR range test to find optimal bounds")

print("\nAdvanced Techniques Selection:")
print("1. SWA: Use in final epochs for better generalization")
print("2. Lookahead: Combine with any base optimizer")
print("3. Gradient accumulation: When GPU memory is limited")
print("4. Progressive techniques: For computer vision tasks")

print("\nCommon Combinations:")
print("1. Warm-up + Cosine Annealing + SWA")
print("2. One Cycle + Lookahead")
print("3. Progressive Resizing + Cyclical LR")
print("4. SGDR + Label Smoothing")

print("\nImplementation Tips:")
print("1. Start simple, add complexity gradually")
print("2. Always validate on held-out data")
print("3. Monitor learning rate and loss curves")
print("4. Use learning rate range test before training")
print("5. Save checkpoints at different training phases")

print("\n=== Optimization Techniques Complete ===")

# Memory cleanup
del model, sample_input, sample_target