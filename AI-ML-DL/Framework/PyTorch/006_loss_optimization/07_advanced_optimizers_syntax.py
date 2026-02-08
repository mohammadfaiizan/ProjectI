#!/usr/bin/env python3
"""PyTorch Advanced Optimizers - RMSprop, Adagrad, LBFGS syntax"""

import torch
import torch.nn as nn
import torch.optim as optim
import math

print("=== Advanced Optimizers Overview ===")

print("Advanced optimizers covered:")
print("1. RMSprop (Root Mean Square Propagation)")
print("2. Adagrad (Adaptive Gradient)")
print("3. Adadelta (Adaptive Delta)")
print("4. LBFGS (Limited-memory BFGS)")
print("5. ASGD (Averaged Stochastic Gradient Descent)")
print("6. SparseAdam (Sparse Adam)")
print("7. Rprop (Resilient Propagation)")
print("8. Custom implementations")

print("\n=== RMSprop Optimizer ===")

# Create test model
class AdvancedTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(15, 32)
        self.linear2 = nn.Linear(32, 16)
        self.linear3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# Sample data
model = AdvancedTestModel()
sample_input = torch.randn(48, 15)
sample_target = torch.randn(48, 1)
loss_fn = nn.MSELoss()

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# Basic RMSprop
rmsprop_optimizer = optim.RMSprop(
    model.parameters(),
    lr=0.01,
    alpha=0.99,
    eps=1e-8,
    weight_decay=0,
    momentum=0,
    centered=False
)

print(f"RMSprop optimizer settings:")
print(f"  Learning rate: {rmsprop_optimizer.param_groups[0]['lr']}")
print(f"  Alpha (decay factor): {rmsprop_optimizer.param_groups[0]['alpha']}")
print(f"  Epsilon: {rmsprop_optimizer.param_groups[0]['eps']}")
print(f"  Weight decay: {rmsprop_optimizer.param_groups[0]['weight_decay']}")
print(f"  Momentum: {rmsprop_optimizer.param_groups[0]['momentum']}")
print(f"  Centered: {rmsprop_optimizer.param_groups[0]['centered']}")

def training_step(model, optimizer, input_data, target, loss_fn):
    optimizer.zero_grad()
    output = model(input_data)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()

rmsprop_loss = training_step(model, rmsprop_optimizer, sample_input, sample_target, loss_fn)
print(f"RMSprop loss: {rmsprop_loss:.6f}")

# Test different alpha values
alpha_values = [0.9, 0.95, 0.99, 0.999]
alpha_losses = []

print(f"\nRMSprop alpha parameter analysis:")
for alpha in alpha_values:
    test_model = AdvancedTestModel()
    test_optimizer = optim.RMSprop(test_model.parameters(), lr=0.01, alpha=alpha)
    
    loss = training_step(test_model, test_optimizer, sample_input, sample_target, loss_fn)
    alpha_losses.append(loss)
    print(f"Alpha {alpha}: Loss = {loss:.6f}")

# RMSprop with momentum
rmsprop_momentum = optim.RMSprop(
    model.parameters(),
    lr=0.01,
    alpha=0.99,
    momentum=0.9
)

# RMSprop centered
rmsprop_centered = optim.RMSprop(
    model.parameters(),
    lr=0.01,
    alpha=0.99,
    centered=True
)

# Test variations
momentum_loss = training_step(AdvancedTestModel(), rmsprop_momentum, sample_input, sample_target, loss_fn)
centered_loss = training_step(AdvancedTestModel(), rmsprop_centered, sample_input, sample_target, loss_fn)

print(f"\nRMSprop variations:")
print(f"Basic RMSprop: {rmsprop_loss:.6f}")
print(f"RMSprop + Momentum: {momentum_loss:.6f}")
print(f"RMSprop Centered: {centered_loss:.6f}")

print("\n=== Adagrad Optimizer ===")

# Adagrad optimizer
adagrad_optimizer = optim.Adagrad(
    model.parameters(),
    lr=0.01,
    lr_decay=0,
    weight_decay=0,
    initial_accumulator_value=0,
    eps=1e-10
)

print(f"Adagrad optimizer settings:")
print(f"  Learning rate: {adagrad_optimizer.param_groups[0]['lr']}")
print(f"  LR decay: {adagrad_optimizer.param_groups[0]['lr_decay']}")
print(f"  Weight decay: {adagrad_optimizer.param_groups[0]['weight_decay']}")
print(f"  Initial accumulator: {adagrad_optimizer.param_groups[0]['initial_accumulator_value']}")
print(f"  Epsilon: {adagrad_optimizer.param_groups[0]['eps']}")

model_adagrad = AdvancedTestModel()
adagrad_loss = training_step(model_adagrad, adagrad_optimizer, sample_input, sample_target, loss_fn)
print(f"Adagrad loss: {adagrad_loss:.6f}")

# Test Adagrad learning rate decay
print(f"\nAdagrad with learning rate decay:")
for lr_decay in [0.0, 1e-6, 1e-4, 1e-2]:
    test_model = AdvancedTestModel()
    test_optimizer = optim.Adagrad(test_model.parameters(), lr=0.01, lr_decay=lr_decay)
    
    loss = training_step(test_model, test_optimizer, sample_input, sample_target, loss_fn)
    print(f"LR decay {lr_decay}: Loss = {loss:.6f}")

print("\n=== Adadelta Optimizer ===")

# Adadelta optimizer
adadelta_optimizer = optim.Adadelta(
    model.parameters(),
    lr=1.0,  # Default for Adadelta
    rho=0.9,
    eps=1e-6,
    weight_decay=0
)

print(f"Adadelta optimizer settings:")
print(f"  Learning rate: {adadelta_optimizer.param_groups[0]['lr']}")
print(f"  Rho (decay factor): {adadelta_optimizer.param_groups[0]['rho']}")
print(f"  Epsilon: {adadelta_optimizer.param_groups[0]['eps']}")
print(f"  Weight decay: {adadelta_optimizer.param_groups[0]['weight_decay']}")

model_adadelta = AdvancedTestModel()
adadelta_loss = training_step(model_adadelta, adadelta_optimizer, sample_input, sample_target, loss_fn)
print(f"Adadelta loss: {adadelta_loss:.6f}")

# Test different rho values
print(f"\nAdadelta rho parameter analysis:")
for rho in [0.8, 0.9, 0.95, 0.99]:
    test_model = AdvancedTestModel()
    test_optimizer = optim.Adadelta(test_model.parameters(), rho=rho)
    
    loss = training_step(test_model, test_optimizer, sample_input, sample_target, loss_fn)
    print(f"Rho {rho}: Loss = {loss:.6f}")

print("\n=== LBFGS Optimizer ===")

# LBFGS optimizer
model_lbfgs = AdvancedTestModel()
lbfgs_optimizer = optim.LBFGS(
    model_lbfgs.parameters(),
    lr=1.0,
    max_iter=20,
    max_eval=None,
    tolerance_grad=1e-7,
    tolerance_change=1e-9,
    history_size=100,
    line_search_fn=None
)

print(f"LBFGS optimizer settings:")
print(f"  Learning rate: {lbfgs_optimizer.param_groups[0]['lr']}")
print(f"  Max iterations: {lbfgs_optimizer.param_groups[0]['max_iter']}")
print(f"  Max evaluations: {lbfgs_optimizer.param_groups[0]['max_eval']}")
print(f"  Tolerance grad: {lbfgs_optimizer.param_groups[0]['tolerance_grad']}")
print(f"  Tolerance change: {lbfgs_optimizer.param_groups[0]['tolerance_change']}")
print(f"  History size: {lbfgs_optimizer.param_groups[0]['history_size']}")

# LBFGS requires a closure function
def closure():
    lbfgs_optimizer.zero_grad()
    output = model_lbfgs(sample_input)
    loss = loss_fn(output, sample_target)
    loss.backward()
    return loss

# Perform LBFGS step
initial_loss = closure().item()
lbfgs_optimizer.step(closure)
final_loss = closure().item()

print(f"LBFGS optimization:")
print(f"  Initial loss: {initial_loss:.6f}")
print(f"  Final loss: {final_loss:.6f}")
print(f"  Loss reduction: {initial_loss - final_loss:.6f}")

# LBFGS with line search
model_lbfgs_ls = AdvancedTestModel()
lbfgs_line_search = optim.LBFGS(
    model_lbfgs_ls.parameters(),
    lr=1.0,
    line_search_fn='strong_wolfe'
)

def closure_ls():
    lbfgs_line_search.zero_grad()
    output = model_lbfgs_ls(sample_input)
    loss = loss_fn(output, sample_target)
    loss.backward()
    return loss

initial_loss_ls = closure_ls().item()
lbfgs_line_search.step(closure_ls)
final_loss_ls = closure_ls().item()

print(f"\nLBFGS with line search:")
print(f"  Initial loss: {initial_loss_ls:.6f}")
print(f"  Final loss: {final_loss_ls:.6f}")
print(f"  Loss reduction: {initial_loss_ls - final_loss_ls:.6f}")

print("\n=== ASGD Optimizer ===")

# ASGD (Averaged Stochastic Gradient Descent)
asgd_optimizer = optim.ASGD(
    model.parameters(),
    lr=0.01,
    lambd=1e-4,
    alpha=0.75,
    t0=1e6,
    weight_decay=0
)

print(f"ASGD optimizer settings:")
print(f"  Learning rate: {asgd_optimizer.param_groups[0]['lr']}")
print(f"  Lambda: {asgd_optimizer.param_groups[0]['lambd']}")
print(f"  Alpha: {asgd_optimizer.param_groups[0]['alpha']}")
print(f"  T0: {asgd_optimizer.param_groups[0]['t0']}")
print(f"  Weight decay: {asgd_optimizer.param_groups[0]['weight_decay']}")

model_asgd = AdvancedTestModel()
asgd_loss = training_step(model_asgd, asgd_optimizer, sample_input, sample_target, loss_fn)
print(f"ASGD loss: {asgd_loss:.6f}")

print("\n=== SparseAdam Optimizer ===")

# Create sparse model for SparseAdam
class SparseModel(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.linear = nn.Linear(embed_dim, 1)
    
    def forward(self, x):
        # x should be indices
        x = self.embedding(x)
        x = x.mean(dim=1)  # Simple averaging
        x = self.linear(x)
        return x

sparse_model = SparseModel()
sparse_input = torch.randint(0, 1000, (32, 10))  # Batch of sequences
sparse_target = torch.randn(32, 1)

# SparseAdam optimizer (for sparse gradients)
sparse_adam = optim.SparseAdam(
    sparse_model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8
)

print(f"SparseAdam optimizer settings:")
print(f"  Learning rate: {sparse_adam.param_groups[0]['lr']}")
print(f"  Betas: {sparse_adam.param_groups[0]['betas']}")
print(f"  Epsilon: {sparse_adam.param_groups[0]['eps']}")

sparse_loss = training_step(sparse_model, sparse_adam, sparse_input, sparse_target, loss_fn)
print(f"SparseAdam loss: {sparse_loss:.6f}")

print("\n=== Custom Advanced Optimizers ===")

class CustomRMSprop:
    """Custom RMSprop implementation"""
    
    def __init__(self, parameters, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0):
        self.parameters = list(parameters)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        
        # State variables
        self.square_avg = [torch.zeros_like(p) for p in self.parameters]
        self.momentum_buffer = [torch.zeros_like(p) for p in self.parameters] if momentum > 0 else None
    
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
                
                # Apply weight decay
                if self.weight_decay != 0:
                    grad = grad.add(param, alpha=self.weight_decay)
                
                # Update moving average of squared gradients
                self.square_avg[i].mul_(self.alpha).addcmul_(grad, grad, value=1 - self.alpha)
                
                # Compute step
                avg = self.square_avg[i].sqrt().add_(self.eps)
                
                if self.momentum > 0:
                    # Apply momentum
                    self.momentum_buffer[i].mul_(self.momentum).addcdiv_(grad, avg)
                    param.add_(self.momentum_buffer[i], alpha=-self.lr)
                else:
                    # Direct update
                    param.addcdiv_(grad, avg, value=-self.lr)

class CustomAdagrad:
    """Custom Adagrad implementation"""
    
    def __init__(self, parameters, lr=1e-2, lr_decay=0, weight_decay=0, eps=1e-10):
        self.parameters = list(parameters)
        self.lr = lr
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.eps = eps
        
        # State variables
        self.square_avg = [torch.zeros_like(p) for p in self.parameters]
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
                
                # Apply weight decay
                if self.weight_decay != 0:
                    grad = grad.add(param, alpha=self.weight_decay)
                
                # Accumulate squared gradients
                self.square_avg[i].addcmul_(grad, grad)
                
                # Compute effective learning rate
                clr = self.lr / (1 + (self.step_count - 1) * self.lr_decay)
                
                # Update parameters
                std = self.square_avg[i].sqrt().add_(self.eps)
                param.addcdiv_(grad, std, value=-clr)

class AdamaxOptimizer:
    """Custom AdaMax implementation"""
    
    def __init__(self, parameters, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # State variables
        self.exp_avg = [torch.zeros_like(p) for p in self.parameters]
        self.exp_inf = [torch.zeros_like(p) for p in self.parameters]
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
                
                # Apply weight decay
                if self.weight_decay != 0:
                    grad = grad.add(param, alpha=self.weight_decay)
                
                # Update exponential moving averages
                self.exp_avg[i].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
                
                # Update infinity norm
                norm_buf = torch.max(
                    self.exp_inf[i].mul_(self.beta2),
                    grad.abs().add_(self.eps)
                )
                self.exp_inf[i].copy_(norm_buf)
                
                # Bias correction
                bias_correction = 1 - self.beta1 ** self.step_count
                
                # Update parameters
                param.addcdiv_(self.exp_avg[i], norm_buf, value=-self.lr / bias_correction)

# Test custom optimizers
models_custom = [AdvancedTestModel() for _ in range(4)]
names_custom = ["Built-in RMSprop", "Custom RMSprop", "Custom Adagrad", "Custom AdaMax"]

# Initialize with same weights
for model in models_custom[1:]:
    model.load_state_dict(models_custom[0].state_dict())

optimizers_custom = [
    optim.RMSprop(models_custom[0].parameters(), lr=0.01),
    CustomRMSprop(models_custom[1].parameters(), lr=0.01),
    CustomAdagrad(models_custom[2].parameters(), lr=0.01),
    AdamaxOptimizer(models_custom[3].parameters(), lr=0.01)
]

print(f"Custom optimizer comparison:")
for epoch in range(8):
    losses_custom = []
    
    for model, opt in zip(models_custom, optimizers_custom):
        opt.zero_grad()
        output = model(sample_input)
        loss = loss_fn(output, sample_target)
        loss.backward()
        opt.step()
        losses_custom.append(loss.item())
    
    if epoch % 2 == 0:
        print(f"  Epoch {epoch}:")
        for name, loss in zip(names_custom, losses_custom):
            print(f"    {name}: {loss:.6f}")

print("\n=== Optimizer Comparison ===")

# Compare all optimizers
optimizers_comparison = {
    'SGD': optim.SGD,
    'Adam': optim.Adam,
    'AdamW': optim.AdamW,
    'RMSprop': optim.RMSprop,
    'Adagrad': optim.Adagrad,
    'Adadelta': optim.Adadelta,
    'ASGD': optim.ASGD
}

print(f"Optimizer performance comparison (single step):")
base_model = AdvancedTestModel()

for name, optimizer_class in optimizers_comparison.items():
    test_model = AdvancedTestModel()
    test_model.load_state_dict(base_model.state_dict())
    
    # Use appropriate parameters for each optimizer
    if name == 'SGD':
        optimizer = optimizer_class(test_model.parameters(), lr=0.01, momentum=0.9)
    elif name in ['Adam', 'AdamW']:
        optimizer = optimizer_class(test_model.parameters(), lr=0.001)
    elif name == 'RMSprop':
        optimizer = optimizer_class(test_model.parameters(), lr=0.01)
    elif name == 'Adagrad':
        optimizer = optimizer_class(test_model.parameters(), lr=0.01)
    elif name == 'Adadelta':
        optimizer = optimizer_class(test_model.parameters())
    elif name == 'ASGD':
        optimizer = optimizer_class(test_model.parameters(), lr=0.01)
    
    loss = training_step(test_model, optimizer, sample_input, sample_target, loss_fn)
    print(f"  {name}: {loss:.6f}")

print("\n=== Second-Order Optimizers ===")

class NewtonCG:
    """Newton's method with conjugate gradient"""
    
    def __init__(self, parameters, lr=1.0, damping=1e-3):
        self.parameters = list(parameters)
        self.lr = lr
        self.damping = damping
    
    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()
    
    def step(self, closure):
        # This is a simplified version - full implementation would be more complex
        loss = closure()
        
        # Compute gradients
        grads = []
        for param in self.parameters:
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        
        if not grads:
            return loss
        
        grad_vec = torch.cat(grads)
        
        # Simple approximation: use identity + damping as Hessian approximation
        hessian_approx = torch.eye(len(grad_vec)) + self.damping
        
        # Solve H * step = grad
        try:
            step_vec = torch.linalg.solve(hessian_approx, grad_vec)
        except:
            # Fallback to gradient descent
            step_vec = grad_vec
        
        # Apply update
        idx = 0
        with torch.no_grad():
            for param in self.parameters:
                if param.grad is not None:
                    param_size = param.numel()
                    param_step = step_vec[idx:idx + param_size].view(param.shape)
                    param.add_(param_step, alpha=-self.lr)
                    idx += param_size
        
        return loss

# Test Newton-CG (simplified)
model_newton = AdvancedTestModel()
newton_optimizer = NewtonCG(model_newton.parameters(), lr=0.1)

def newton_closure():
    newton_optimizer.zero_grad()
    output = model_newton(sample_input)
    loss = loss_fn(output, sample_target)
    loss.backward()
    return loss

newton_loss = newton_optimizer.step(newton_closure).item()
print(f"Newton-CG loss: {newton_loss:.6f}")

print("\n=== Advanced Optimizer Best Practices ===")

print("Optimizer Selection Guidelines:")
print("1. RMSprop: Good for RNNs and non-stationary objectives")
print("2. Adagrad: Sparse data and features, but can be too aggressive")
print("3. Adadelta: Adagrad without learning rate decay")
print("4. LBFGS: Small datasets, smooth objectives, batch optimization")
print("5. ASGD: Simple problems, when you want parameter averaging")
print("6. SparseAdam: Sparse embeddings and high-dimensional sparse problems")

print("\nParameter Guidelines:")
print("RMSprop:")
print("  - lr: 0.01 (higher than Adam)")
print("  - alpha: 0.99 (decay factor for squared gradients)")
print("  - Use momentum for faster convergence")
print("  - centered=True for variance reduction")

print("\nAdagrad:")
print("  - lr: 0.01 (can be higher initially)")
print("  - lr_decay: Usually 0, built-in decay from accumulation")
print("  - Good for sparse features")
print("  - May stop learning too early")

print("\nLBFGS:")
print("  - lr: 1.0 (line search handles step size)")
print("  - max_iter: 20 (per optimization step)")
print("  - Use strong_wolfe line search for robustness")
print("  - Best for small to medium datasets")

print("\nCommon Issues:")
print("1. RMSprop: Can be unstable with large learning rates")
print("2. Adagrad: Learning rate decay can be too aggressive")
print("3. LBFGS: Memory intensive, requires closure function")
print("4. Adadelta: Can be slow to converge")
print("5. All adaptive methods: May generalize worse than SGD")

print("\nWhen to Use Each:")
print("- RMSprop: RNNs, non-convex optimization")
print("- Adagrad: NLP with sparse features")
print("- Adadelta: When you don't want to tune learning rate")
print("- LBFGS: Small datasets, when you can afford second-order methods")
print("- ASGD: Simple baseline, parameter averaging")

print("\n=== Advanced Optimizers Complete ===")

# Memory cleanup
del models_custom, model, sparse_model, sample_input, sparse_input