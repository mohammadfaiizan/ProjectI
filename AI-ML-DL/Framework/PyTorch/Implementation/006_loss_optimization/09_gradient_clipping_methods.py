#!/usr/bin/env python3
"""PyTorch Gradient Clipping Methods - Gradient clipping techniques"""

import torch
import torch.nn as nn
import torch.nn.utils as utils
import math

print("=== Gradient Clipping Overview ===")

print("Gradient clipping techniques:")
print("1. Gradient Norm Clipping (clip_grad_norm_)")
print("2. Gradient Value Clipping (clip_grad_value_)")
print("3. Per-parameter Clipping")
print("4. Adaptive Gradient Clipping")
print("5. Layer-wise Clipping")
print("6. Percentile-based Clipping")
print("7. Custom Clipping Functions")
print("8. Gradient Scaling and Clipping")

print("\n=== Model Setup for Testing ===")

class GradientTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 64)
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

# Create model and data
model = GradientTestModel()
sample_input = torch.randn(32, 10)
sample_target = torch.randn(32, 1)
loss_fn = nn.MSELoss()

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

def compute_gradients(model, input_data, target, loss_fn):
    """Compute gradients without stepping optimizer"""
    model.zero_grad()
    output = model(input_data)
    loss = loss_fn(output, target)
    loss.backward()
    return loss.item()

def get_gradient_norm(model, norm_type=2):
    """Compute gradient norm for all parameters"""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

# Compute initial gradients
initial_loss = compute_gradients(model, sample_input, sample_target, loss_fn)
initial_grad_norm = get_gradient_norm(model)

print(f"Initial loss: {initial_loss:.6f}")
print(f"Initial gradient norm: {initial_grad_norm:.6f}")

print("\n=== Gradient Norm Clipping ===")

# Test gradient norm clipping with different max_norm values
max_norms = [0.1, 0.5, 1.0, 2.0, 10.0]

print(f"Gradient norm clipping with different max_norm values:")
for max_norm in max_norms:
    # Reset gradients
    model.zero_grad()
    output = model(sample_input)
    loss = loss_fn(output, sample_target)
    loss.backward()
    
    # Get norm before clipping
    norm_before = get_gradient_norm(model)
    
    # Apply gradient clipping
    actual_norm = utils.clip_grad_norm_(model.parameters(), max_norm, norm_type=2)
    
    # Get norm after clipping
    norm_after = get_gradient_norm(model)
    
    print(f"  max_norm={max_norm:4.1f}: before={norm_before:.6f}, after={norm_after:.6f}, actual={actual_norm:.6f}")

print("\n=== Gradient Value Clipping ===")

# Test gradient value clipping
clip_values = [0.01, 0.1, 0.5, 1.0, 5.0]

print(f"Gradient value clipping with different clip_value:")
for clip_value in clip_values:
    # Reset gradients
    model.zero_grad()
    output = model(sample_input)
    loss = loss_fn(output, sample_target)
    loss.backward()
    
    # Count parameters with large gradients before clipping
    large_grads_before = 0
    for param in model.parameters():
        if param.grad is not None:
            large_grads_before += (torch.abs(param.grad) > clip_value).sum().item()
    
    # Apply gradient value clipping
    utils.clip_grad_value_(model.parameters(), clip_value)
    
    # Count parameters with large gradients after clipping
    large_grads_after = 0
    max_grad_after = 0
    for param in model.parameters():
        if param.grad is not None:
            large_grads_after += (torch.abs(param.grad) > clip_value).sum().item()
            max_grad_after = max(max_grad_after, torch.abs(param.grad).max().item())
    
    print(f"  clip_value={clip_value:4.2f}: large_grads before={large_grads_before:4d}, after={large_grads_after:4d}, max_grad={max_grad_after:.6f}")

print("\n=== Different Norm Types ===")

# Test different norm types for gradient clipping
norm_types = [1, 2, float('inf')]
max_norm = 1.0

print(f"Gradient norm clipping with different norm types (max_norm={max_norm}):")
for norm_type in norm_types:
    # Reset gradients
    model.zero_grad()
    output = model(sample_input)
    loss = loss_fn(output, sample_target)
    loss.backward()
    
    # Get norm before clipping
    norm_before = get_gradient_norm(model, norm_type)
    
    # Apply gradient clipping
    actual_norm = utils.clip_grad_norm_(model.parameters(), max_norm, norm_type=norm_type)
    
    # Get norm after clipping
    norm_after = get_gradient_norm(model, norm_type)
    
    norm_name = "L1" if norm_type == 1 else "L2" if norm_type == 2 else "L∞"
    print(f"  {norm_name} norm: before={norm_before:.6f}, after={norm_after:.6f}, actual={actual_norm:.6f}")

print("\n=== Per-Parameter Clipping ===")

def clip_grad_per_param(model, max_norm_per_param=1.0):
    """Clip gradients per parameter instead of globally"""
    clipped_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_params += 1
            param_norm = param.grad.data.norm(2)
            
            if param_norm > max_norm_per_param:
                param.grad.data.mul_(max_norm_per_param / param_norm)
                clipped_params += 1
                print(f"    Clipped {name}: norm {param_norm:.6f} → {max_norm_per_param:.6f}")
    
    return clipped_params, total_params

# Test per-parameter clipping
model.zero_grad()
output = model(sample_input)
loss = loss_fn(output, sample_target)
loss.backward()

print(f"Per-parameter gradient clipping (max_norm_per_param=1.0):")
clipped, total = clip_grad_per_param(model, max_norm_per_param=1.0)
print(f"  Clipped {clipped}/{total} parameters")

print("\n=== Adaptive Gradient Clipping ===")

class AdaptiveGradientClipper:
    """Adaptive gradient clipping based on gradient norm history"""
    
    def __init__(self, percentile=95, history_size=100):
        self.percentile = percentile
        self.history_size = history_size
        self.gradient_norms = []
    
    def clip_gradients(self, model):
        # Compute current gradient norm
        current_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                current_norm += param_norm.item() ** 2
        current_norm = current_norm ** 0.5
        
        # Update history
        self.gradient_norms.append(current_norm)
        if len(self.gradient_norms) > self.history_size:
            self.gradient_norms.pop(0)
        
        # Compute adaptive threshold
        if len(self.gradient_norms) >= 10:  # Need some history
            threshold = torch.tensor(self.gradient_norms).quantile(self.percentile / 100.0)
            
            if current_norm > threshold:
                # Clip gradients
                clip_factor = threshold / current_norm
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data.mul_(clip_factor)
                return True, current_norm, threshold.item()
        
        return False, current_norm, None

# Test adaptive clipping over multiple iterations
adaptive_clipper = AdaptiveGradientClipper(percentile=90, history_size=20)

print(f"Adaptive gradient clipping simulation:")
for iteration in range(25):
    # Add some noise to create varying gradient norms
    noisy_input = sample_input + 0.1 * torch.randn_like(sample_input) * (iteration % 5)
    
    model.zero_grad()
    output = model(noisy_input)
    loss = loss_fn(output, sample_target)
    loss.backward()
    
    was_clipped, norm, threshold = adaptive_clipper.clip_gradients(model)
    
    if iteration % 5 == 0:
        status = "CLIPPED" if was_clipped else "ok"
        thresh_str = f", threshold={threshold:.4f}" if threshold is not None else ""
        print(f"  Iter {iteration:2d}: norm={norm:.4f} {thresh_str} [{status}]")

print("\n=== Layer-wise Gradient Clipping ===")

def layerwise_gradient_clipping(model, max_norm_per_layer=1.0):
    """Clip gradients layer by layer"""
    layer_info = []
    
    # Group parameters by layer
    layer_params = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            layer_name = name.split('.')[0]  # Get layer name (e.g., 'linear1')
            if layer_name not in layer_params:
                layer_params[layer_name] = []
            layer_params[layer_name].append((name, param))
    
    # Clip each layer separately
    for layer_name, params in layer_params.items():
        # Compute layer gradient norm
        layer_norm = 0.0
        for _, param in params:
            param_norm = param.grad.data.norm(2)
            layer_norm += param_norm.item() ** 2
        layer_norm = layer_norm ** 0.5
        
        # Clip if necessary
        if layer_norm > max_norm_per_layer:
            clip_factor = max_norm_per_layer / layer_norm
            for _, param in params:
                param.grad.data.mul_(clip_factor)
            status = "CLIPPED"
        else:
            status = "ok"
        
        layer_info.append((layer_name, layer_norm, status))
    
    return layer_info

# Test layer-wise clipping
model.zero_grad()
output = model(sample_input)
loss = loss_fn(output, sample_target)
loss.backward()

print(f"Layer-wise gradient clipping (max_norm_per_layer=1.0):")
layer_results = layerwise_gradient_clipping(model, max_norm_per_layer=1.0)
for layer_name, norm, status in layer_results:
    print(f"  {layer_name:8}: norm={norm:.6f} [{status}]")

print("\n=== Percentile-based Clipping ===")

def percentile_gradient_clipping(model, percentile=95):
    """Clip gradients based on percentile of gradient magnitudes"""
    # Collect all gradient magnitudes
    all_grads = []
    for param in model.parameters():
        if param.grad is not None:
            all_grads.extend(torch.abs(param.grad).flatten().tolist())
    
    if not all_grads:
        return 0, 0, 0
    
    # Compute percentile threshold
    all_grads_tensor = torch.tensor(all_grads)
    threshold = torch.quantile(all_grads_tensor, percentile / 100.0)
    
    # Count and clip
    clipped_count = 0
    total_count = 0
    
    for param in model.parameters():
        if param.grad is not None:
            mask = torch.abs(param.grad) > threshold
            clipped_count += mask.sum().item()
            total_count += param.grad.numel()
            
            # Apply clipping
            param.grad.data = torch.clamp(param.grad.data, -threshold, threshold)
    
    return clipped_count, total_count, threshold.item()

# Test percentile-based clipping
model.zero_grad()
output = model(sample_input)
loss = loss_fn(output, sample_target)
loss.backward()

print(f"Percentile-based gradient clipping:")
for percentile in [90, 95, 99]:
    # Reset gradients
    model.zero_grad()
    output = model(sample_input)
    loss = loss_fn(output, sample_target)
    loss.backward()
    
    clipped, total, threshold = percentile_gradient_clipping(model, percentile)
    percentage_clipped = 100.0 * clipped / total
    
    print(f"  {percentile}th percentile: threshold={threshold:.6f}, clipped {clipped}/{total} ({percentage_clipped:.1f}%)")

print("\n=== Custom Clipping Functions ===")

class CustomGradientClipper:
    """Custom gradient clipping with various strategies"""
    
    def __init__(self, strategy='norm', **kwargs):
        self.strategy = strategy
        self.kwargs = kwargs
        self.history = []
    
    def clip(self, model):
        if self.strategy == 'norm':
            return self._clip_by_norm(model)
        elif self.strategy == 'value':
            return self._clip_by_value(model)
        elif self.strategy == 'adaptive_norm':
            return self._clip_adaptive_norm(model)
        elif self.strategy == 'outlier':
            return self._clip_outliers(model)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _clip_by_norm(self, model):
        max_norm = self.kwargs.get('max_norm', 1.0)
        return utils.clip_grad_norm_(model.parameters(), max_norm)
    
    def _clip_by_value(self, model):
        clip_value = self.kwargs.get('clip_value', 1.0)
        utils.clip_grad_value_(model.parameters(), clip_value)
        return clip_value
    
    def _clip_adaptive_norm(self, model):
        # Compute current norm
        current_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                current_norm += param.grad.data.norm(2).item() ** 2
        current_norm = current_norm ** 0.5
        
        # Update history
        self.history.append(current_norm)
        if len(self.history) > 50:
            self.history.pop(0)
        
        # Adaptive threshold
        if len(self.history) > 10:
            mean_norm = sum(self.history) / len(self.history)
            std_norm = (sum((x - mean_norm) ** 2 for x in self.history) / len(self.history)) ** 0.5
            threshold = mean_norm + 2 * std_norm  # 2-sigma rule
            
            if current_norm > threshold:
                utils.clip_grad_norm_(model.parameters(), threshold)
                return threshold
        
        return current_norm
    
    def _clip_outliers(self, model):
        # Collect all gradients
        all_grads = []
        for param in model.parameters():
            if param.grad is not None:
                all_grads.extend(param.grad.flatten().tolist())
        
        if not all_grads:
            return 0
        
        # Compute IQR-based outlier threshold
        all_grads_tensor = torch.tensor(all_grads)
        q25 = torch.quantile(all_grads_tensor, 0.25)
        q75 = torch.quantile(all_grads_tensor, 0.75)
        iqr = q75 - q25
        
        # Outlier thresholds
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        # Clip outliers
        clipped_count = 0
        for param in model.parameters():
            if param.grad is not None:
                original_grad = param.grad.data.clone()
                param.grad.data = torch.clamp(param.grad.data, lower_bound, upper_bound)
                clipped_count += (original_grad != param.grad.data).sum().item()
        
        return clipped_count

# Test custom clippers
clippers = {
    'Norm Clipper': CustomGradientClipper('norm', max_norm=1.0),
    'Value Clipper': CustomGradientClipper('value', clip_value=0.5),
    'Adaptive Clipper': CustomGradientClipper('adaptive_norm'),
    'Outlier Clipper': CustomGradientClipper('outlier')
}

print(f"Custom gradient clippers comparison:")
for name, clipper in clippers.items():
    # Reset gradients
    model.zero_grad()
    output = model(sample_input)
    loss = loss_fn(output, sample_target)
    loss.backward()
    
    # Get norm before clipping
    norm_before = get_gradient_norm(model)
    
    # Apply clipping
    result = clipper.clip(model)
    
    # Get norm after clipping
    norm_after = get_gradient_norm(model)
    
    print(f"  {name:16}: before={norm_before:.6f}, after={norm_after:.6f}, result={result:.6f}")

print("\n=== Gradient Clipping in Training Loop ===")

class TrainingWithClipping:
    """Training loop with different clipping strategies"""
    
    def __init__(self, model, optimizer, clip_strategy='norm', **clip_kwargs):
        self.model = model
        self.optimizer = optimizer
        self.clip_strategy = clip_strategy
        self.clip_kwargs = clip_kwargs
        self.gradient_norms = []
    
    def train_step(self, input_data, targets, loss_fn):
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(input_data)
        loss = loss_fn(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Record gradient norm before clipping
        grad_norm_before = self._compute_grad_norm()
        
        # Apply gradient clipping
        if self.clip_strategy == 'norm':
            clipped_norm = utils.clip_grad_norm_(
                self.model.parameters(), 
                self.clip_kwargs.get('max_norm', 1.0)
            )
        elif self.clip_strategy == 'value':
            utils.clip_grad_value_(
                self.model.parameters(), 
                self.clip_kwargs.get('clip_value', 1.0)
            )
            clipped_norm = self._compute_grad_norm()
        else:
            clipped_norm = grad_norm_before
        
        # Record gradient norm after clipping
        grad_norm_after = self._compute_grad_norm()
        self.gradient_norms.append((grad_norm_before, grad_norm_after))
        
        # Optimizer step
        self.optimizer.step()
        
        return loss.item(), grad_norm_before, grad_norm_after
    
    def _compute_grad_norm(self):
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

# Test training with different clipping strategies
strategies = [
    ('No Clipping', None, {}),
    ('Norm Clipping', 'norm', {'max_norm': 1.0}),
    ('Value Clipping', 'value', {'clip_value': 0.5}),
]

print(f"Training simulation with different clipping strategies:")
for strategy_name, clip_strategy, clip_kwargs in strategies:
    # Reset model
    model = GradientTestModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    trainer = TrainingWithClipping(model, optimizer, clip_strategy, **clip_kwargs)
    
    print(f"\n{strategy_name}:")
    for epoch in range(5):
        loss, norm_before, norm_after = trainer.train_step(sample_input, sample_target, loss_fn)
        
        if clip_strategy:
            clip_ratio = norm_after / norm_before if norm_before > 0 else 1.0
            print(f"  Epoch {epoch}: loss={loss:.6f}, grad_norm: {norm_before:.4f}→{norm_after:.4f} ({clip_ratio:.3f})")
        else:
            print(f"  Epoch {epoch}: loss={loss:.6f}, grad_norm: {norm_before:.4f}")

print("\n=== Gradient Clipping Best Practices ===")

print("When to Use Gradient Clipping:")
print("1. RNNs and LSTMs (common to use norm clipping)")
print("2. Training instability with exploding gradients")
print("3. Very deep networks")
print("4. High learning rates")
print("5. Reinforcement learning (policy gradients)")
print("6. GANs (especially discriminator)")

print("\nClipping Method Selection:")
print("1. Norm clipping: Most common, preserves gradient direction")
print("2. Value clipping: Simple, but can change gradient direction")
print("3. Per-parameter clipping: When different layers need different limits")
print("4. Adaptive clipping: When gradient norms vary significantly")
print("5. Layer-wise clipping: For very deep networks")

print("\nRecommended Values:")
print("Norm Clipping:")
print("  - RNNs/LSTMs: max_norm = 1.0 to 5.0")
print("  - Transformers: max_norm = 1.0 to 2.0")
print("  - CNNs: max_norm = 1.0 to 10.0")
print("  - GANs: max_norm = 0.1 to 1.0")

print("\nValue Clipping:")
print("  - Conservative: clip_value = 0.1 to 0.5")
print("  - Moderate: clip_value = 0.5 to 1.0")
print("  - Aggressive: clip_value = 1.0 to 5.0")

print("\nImplementation Tips:")
print("1. Apply clipping after loss.backward() but before optimizer.step()")
print("2. Monitor gradient norms to choose appropriate thresholds")
print("3. Use different clipping for different parameter groups")
print("4. Log gradient norms for debugging")
print("5. Consider adaptive clipping for varying gradient scales")

print("\nCommon Mistakes:")
print("1. Clipping before computing gradients")
print("2. Too aggressive clipping (hampers learning)")
print("3. No clipping when needed (training instability)")
print("4. Not monitoring gradient statistics")
print("5. Same clipping for all layers/parameters")

print("\nDebugging Gradient Issues:")
print("1. Plot gradient norms over training")
print("2. Check individual layer gradient norms")
print("3. Monitor the ratio of clipped to unclipped steps")
print("4. Validate clipping threshold with gradient norm distribution")
print("5. Compare training with/without clipping")

print("\n=== Gradient Clipping Complete ===")

# Memory cleanup
del model