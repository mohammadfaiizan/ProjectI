#!/usr/bin/env python3
"""PyTorch Custom Optimizers - Implementing custom optimizers"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import math
from collections import defaultdict

print("=== Custom Optimizers Overview ===")

print("Custom optimizer topics:")
print("1. PyTorch Optimizer base class")
print("2. Implementing basic optimizers from scratch")
print("3. Advanced optimizer features")
print("4. State management in custom optimizers")
print("5. Custom learning rate scheduling")
print("6. Optimizer hooks and callbacks")
print("7. Distributed optimizer considerations")
print("8. Performance optimization techniques")

print("\n=== Understanding Optimizer Base Class ===")

class BasicOptimizer(Optimizer):
    """Basic custom optimizer template"""
    
    def __init__(self, params, lr=1e-3, **kwargs):
        # Validate learning rate
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        # Set default values
        defaults = dict(lr=lr, **kwargs)
        super(BasicOptimizer, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Perform a single optimization step"""
        loss = None
        if closure is not None:
            loss = closure()
        
        # Iterate through parameter groups
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get parameter and gradient
                grad = p.grad.data
                
                # Basic gradient descent update
                p.data.add_(grad, alpha=-group['lr'])
        
        return loss

# Test basic optimizer
print("Testing basic custom optimizer:")
test_model = nn.Linear(10, 1)
test_optimizer = BasicOptimizer(test_model.parameters(), lr=0.01)

sample_input = torch.randn(5, 10)
sample_target = torch.randn(5, 1)
loss_fn = nn.MSELoss()

for step in range(3):
    test_optimizer.zero_grad()
    output = test_model(sample_input)
    loss = loss_fn(output, sample_target)
    loss.backward()
    test_optimizer.step()
    print(f"  Step {step}: Loss = {loss.item():.6f}")

print("\n=== Custom SGD with Momentum ===")

class CustomSGDMomentum(Optimizer):
    """Custom SGD with momentum implementation"""
    
    def __init__(self, params, lr=1e-3, momentum=0, weight_decay=0, dampening=0, nesterov=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                       dampening=dampening, nesterov=nesterov)
        
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        
        super(CustomSGDMomentum, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Perform optimization step"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                # Apply momentum
                if momentum != 0:
                    param_state = self.state[p]
                    
                    # Initialize momentum buffer
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(grad)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                    
                    # Apply Nesterov momentum if specified
                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        grad = buf
                
                # Update parameters
                p.data.add_(grad, alpha=-group['lr'])
        
        return loss

# Test custom SGD
print("Testing custom SGD with momentum:")
test_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
custom_sgd = CustomSGDMomentum(test_model.parameters(), lr=0.01, momentum=0.9, nesterov=True)

for step in range(5):
    custom_sgd.zero_grad()
    output = test_model(sample_input)
    loss = loss_fn(output, sample_target)
    loss.backward()
    custom_sgd.step()
    
    if step % 2 == 0:
        print(f"  Step {step}: Loss = {loss.item():.6f}")

print("\n=== Custom Adam Optimizer ===")

class CustomAdam(Optimizer):
    """Custom Adam optimizer implementation"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(CustomAdam, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Perform optimization step"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Custom Adam does not support sparse gradients')
                
                amsgrad = group['amsgrad']
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maximum of exponential moving averages of squared gradients
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Update exponential moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Apply AMSGrad variant if specified
                if amsgrad:
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                # Compute step size
                step_size = group['lr'] / bias_correction1
                
                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss

# Test custom Adam
print("Testing custom Adam optimizer:")
test_model = nn.Sequential(nn.Linear(10, 8), nn.ReLU(), nn.Linear(8, 1))
custom_adam = CustomAdam(test_model.parameters(), lr=0.001, amsgrad=True)

for step in range(5):
    custom_adam.zero_grad()
    output = test_model(sample_input)
    loss = loss_fn(output, sample_target)
    loss.backward()
    custom_adam.step()
    
    if step % 2 == 0:
        print(f"  Step {step}: Loss = {loss.item():.6f}")

print("\n=== Advanced Custom Optimizer: AdaBound ===")

class AdaBound(Optimizer):
    """AdaBound optimizer - combines Adam and SGD"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,
                 eps=1e-8, weight_decay=0, amsbound=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= final_lr:
            raise ValueError(f"Invalid final learning rate: {final_lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                       weight_decay=weight_decay, amsbound=amsbound)
        super(AdaBound, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Perform optimization step"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaBound does not support sparse gradients')
                
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if group['amsbound']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsbound']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Update exponential moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                if group['amsbound']:
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt()
                else:
                    denom = exp_avg_sq.sqrt()
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute bounds
                final_lr = group['final_lr'] * group['lr'] / group['lr']
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                
                # Compute step size with bounds
                step_size = group['lr'] / bias_correction1
                denom = denom / math.sqrt(bias_correction2)
                denom = denom.add_(group['eps'])
                
                # Apply bounds to step size
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)
                
                # Update parameters
                p.data.add_(step_size, alpha=-1)
        
        return loss

# Test AdaBound
print("Testing AdaBound optimizer:")
test_model = nn.Sequential(nn.Linear(10, 6), nn.ReLU(), nn.Linear(6, 1))
adabound = AdaBound(test_model.parameters(), lr=0.001, final_lr=0.1, amsbound=True)

for step in range(5):
    adabound.zero_grad()
    output = test_model(sample_input)
    loss = loss_fn(output, sample_target)
    loss.backward()
    adabound.step()
    
    if step % 2 == 0:
        print(f"  Step {step}: Loss = {loss.item():.6f}")

print("\n=== Custom Optimizer with Learning Rate Scheduling ===")

class CustomAdaptiveLR(Optimizer):
    """Custom optimizer with built-in adaptive learning rate"""
    
    def __init__(self, params, lr=1e-3, lr_decay=0.99, min_lr=1e-6, patience=5):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= lr_decay <= 1.0:
            raise ValueError(f"Invalid lr_decay: {lr_decay}")
        if not 0.0 <= min_lr:
            raise ValueError(f"Invalid min_lr: {min_lr}")
        
        defaults = dict(lr=lr, lr_decay=lr_decay, min_lr=min_lr, patience=patience)
        super(CustomAdaptiveLR, self).__init__(params, defaults)
        
        # Global state for loss tracking
        self.loss_history = []
        self.no_improvement_count = 0
    
    def step(self, closure=None, current_loss=None):
        """Perform optimization step with adaptive learning rate"""
        loss = None
        if closure is not None:
            loss = closure()
        
        # Update loss history and adjust learning rate
        if current_loss is not None:
            self.loss_history.append(current_loss)
            self._adjust_learning_rate()
        
        # Perform optimization
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                # Update exponential moving averages (Adam-like)
                exp_avg.mul_(0.9).add_(grad, alpha=0.1)
                exp_avg_sq.mul_(0.999).addcmul_(grad, grad, value=0.001)
                
                # Compute denominator
                denom = exp_avg_sq.sqrt().add_(1e-8)
                
                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-group['lr'])
        
        return loss
    
    def _adjust_learning_rate(self):
        """Adjust learning rate based on loss history"""
        if len(self.loss_history) < 2:
            return
        
        # Check if loss improved
        if self.loss_history[-1] >= self.loss_history[-2]:
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0
        
        # Decay learning rate if no improvement for patience steps
        if self.no_improvement_count >= self.param_groups[0]['patience']:
            for group in self.param_groups:
                old_lr = group['lr']
                new_lr = max(old_lr * group['lr_decay'], group['min_lr'])
                group['lr'] = new_lr
                
                if new_lr < old_lr:
                    print(f"    Reducing LR from {old_lr:.6f} to {new_lr:.6f}")
            
            self.no_improvement_count = 0

# Test adaptive LR optimizer
print("Testing custom adaptive LR optimizer:")
test_model = nn.Sequential(nn.Linear(10, 8), nn.ReLU(), nn.Linear(8, 1))
adaptive_opt = CustomAdaptiveLR(test_model.parameters(), lr=0.01, patience=3)

for step in range(15):
    adaptive_opt.zero_grad()
    output = test_model(sample_input)
    loss = loss_fn(output, sample_target)
    loss.backward()
    adaptive_opt.step(current_loss=loss.item())
    
    if step % 3 == 0:
        current_lr = adaptive_opt.param_groups[0]['lr']
        print(f"  Step {step:2d}: Loss = {loss.item():.6f}, LR = {current_lr:.6f}")

print("\n=== Custom Optimizer with Hooks ===")

class OptimzerWithHooks(Optimizer):
    """Custom optimizer with pre/post step hooks"""
    
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super(OptimzerWithHooks, self).__init__(params, defaults)
        
        self.pre_step_hooks = []
        self.post_step_hooks = []
        self.step_count = 0
    
    def add_pre_step_hook(self, hook):
        """Add hook to be called before each step"""
        self.pre_step_hooks.append(hook)
    
    def add_post_step_hook(self, hook):
        """Add hook to be called after each step"""
        self.post_step_hooks.append(hook)
    
    def step(self, closure=None):
        """Perform optimization step with hooks"""
        # Call pre-step hooks
        for hook in self.pre_step_hooks:
            hook(self)
        
        loss = None
        if closure is not None:
            loss = closure()
        
        # Perform optimization
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Simple gradient descent
                p.data.add_(p.grad.data, alpha=-group['lr'])
        
        self.step_count += 1
        
        # Call post-step hooks
        for hook in self.post_step_hooks:
            hook(self)
        
        return loss

# Define hook functions
def gradient_norm_hook(optimizer):
    """Hook to monitor gradient norms"""
    total_norm = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    
    if optimizer.step_count % 3 == 0:
        print(f"    Gradient norm: {total_norm:.6f}")

def parameter_norm_hook(optimizer):
    """Hook to monitor parameter norms"""
    total_norm = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            total_norm += p.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    
    if optimizer.step_count % 3 == 0:
        print(f"    Parameter norm: {total_norm:.6f}")

# Test optimizer with hooks
print("Testing optimizer with hooks:")
test_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
hook_optimizer = OptimzerWithHooks(test_model.parameters(), lr=0.01)

# Add hooks
hook_optimizer.add_pre_step_hook(gradient_norm_hook)
hook_optimizer.add_post_step_hook(parameter_norm_hook)

for step in range(10):
    hook_optimizer.zero_grad()
    output = test_model(sample_input)
    loss = loss_fn(output, sample_target)
    loss.backward()
    hook_optimizer.step()
    
    if step % 3 == 0:
        print(f"  Step {step:2d}: Loss = {loss.item():.6f}")

print("\n=== Custom Optimizer for Sparse Gradients ===")

class SparseSGD(Optimizer):
    """Custom SGD for sparse gradients"""
    
    def __init__(self, params, lr=1e-3, momentum=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        
        defaults = dict(lr=lr, momentum=momentum)
        super(SparseSGD, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Perform optimization step for sparse gradients"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            momentum = group['momentum']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # Handle sparse gradients
                if grad.is_sparse:
                    if momentum != 0:
                        if 'momentum_buffer' not in state:
                            state['momentum_buffer'] = torch.zeros_like(p.data)
                        
                        buf = state['momentum_buffer']
                        
                        # Convert sparse gradient to dense for momentum update
                        grad_indices = grad._indices()
                        grad_values = grad._values()
                        
                        # Update momentum buffer only for non-zero gradients
                        if grad_indices.numel() > 0:
                            # Create dense gradient
                            dense_grad = torch.zeros_like(p.data)
                            dense_grad.index_add_(0, grad_indices.view(-1), 
                                                grad_values.view(grad_indices.size(0), -1))
                            
                            buf.mul_(momentum).add_(dense_grad)
                            grad = buf
                        else:
                            grad = buf.mul(momentum)
                    else:
                        # Convert sparse to dense
                        grad = grad.to_dense()
                else:
                    # Dense gradient handling
                    if momentum != 0:
                        if 'momentum_buffer' not in state:
                            state['momentum_buffer'] = torch.zeros_like(p.data)
                        
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(grad)
                        grad = buf
                
                # Update parameters
                p.data.add_(grad, alpha=-group['lr'])
        
        return loss

print("Testing sparse gradient optimizer:")
# Create embedding layer (produces sparse gradients)
embedding = nn.Embedding(100, 10, sparse=True)
sparse_opt = SparseSGD(embedding.parameters(), lr=0.1, momentum=0.9)

# Create sparse input
sparse_input = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
sparse_target = torch.randn(8, 10)

for step in range(5):
    sparse_opt.zero_grad()
    output = embedding(sparse_input)
    loss = nn.MSELoss()(output, sparse_target)
    loss.backward()
    sparse_opt.step()
    
    if step % 2 == 0:
        print(f"  Step {step}: Loss = {loss.item():.6f}")

print("\n=== Custom Optimizer Best Practices ===")

print("Implementation Guidelines:")
print("1. Inherit from torch.optim.Optimizer")
print("2. Validate all hyperparameters in __init__")
print("3. Use defaults dict for parameter groups")
print("4. Handle closure parameter in step()")
print("5. Properly manage optimizer state")

print("\nState Management:")
print("1. Initialize state lazily in first step")
print("2. Use self.state[param] for per-parameter state")
print("3. Store tensors with same device/dtype as parameters")
print("4. Consider memory efficiency for large models")
print("5. Implement proper state_dict() if needed")

print("\nError Handling:")
print("1. Validate learning rates and other hyperparameters")
print("2. Handle sparse gradients appropriately")
print("3. Check for None gradients")
print("4. Provide meaningful error messages")
print("5. Consider numerical stability")

print("\nPerformance Considerations:")
print("1. Minimize tensor allocations in step()")
print("2. Use in-place operations where possible")
print("3. Avoid unnecessary tensor copies")
print("4. Consider vectorized operations")
print("5. Profile custom optimizers vs built-in ones")

print("\nTesting Custom Optimizers:")
print("1. Test against known optimizers on simple problems")
print("2. Verify convergence properties")
print("3. Check gradient flow and update magnitudes")
print("4. Test state saving and loading")
print("5. Validate with different model architectures")

print("\nCommon Pitfalls:")
print("1. Forgetting bias correction in Adam-style optimizers")
print("2. Incorrect momentum buffer initialization")
print("3. Not handling parameter groups properly")
print("4. Memory leaks from accumulating state")
print("5. Device/dtype mismatches in state tensors")

print("\n=== Custom Optimizers Complete ===")

# Memory cleanup
del test_model, sample_input, sample_target