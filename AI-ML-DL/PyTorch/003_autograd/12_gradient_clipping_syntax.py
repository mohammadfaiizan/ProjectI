#!/usr/bin/env python3
"""PyTorch Gradient Clipping - Gradient clipping implementations"""

import torch
import torch.nn as nn

print("=== Gradient Clipping Overview ===")

print("Gradient clipping prevents:")
print("1. Exploding gradients in deep networks")
print("2. Training instability")
print("3. NaN/Inf gradients")
print("4. Overshooting optimal solutions")
print("5. Poor convergence in RNNs")

print("\n=== Basic Gradient Norm Clipping ===")

# Create model with potential for exploding gradients
class DeepModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(50, 50) for _ in range(10)
        ])
        self.output = nn.Linear(50, 1)
        
        # Initialize with large weights to cause gradient explosion
        for layer in self.layers:
            nn.init.normal_(layer.weight, std=2.0)
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)

model = DeepModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

# Training step without clipping
input_data = torch.randn(32, 50)
target = torch.randn(32, 1)

optimizer.zero_grad()
output = model(input_data)
loss = criterion(output, target)
loss.backward()

# Check gradient norms before clipping
grad_norms_before = []
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_norms_before.append((name, grad_norm))

print("Gradient norms before clipping:")
for name, norm in grad_norms_before[:3]:  # Show first 3
    print(f"  {name}: {norm:.6f}")

# Apply gradient clipping
max_norm = 1.0
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

print(f"Total gradient norm before clipping: {total_norm:.6f}")
print(f"Clipping threshold: {max_norm}")

# Check gradient norms after clipping
grad_norms_after = []
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_norms_after.append((name, grad_norm))

print("Gradient norms after clipping:")
for name, norm in grad_norms_after[:3]:  # Show first 3
    print(f"  {name}: {norm:.6f}")

optimizer.step()

print("\n=== Gradient Value Clipping ===")

# Clip gradient values directly
model_value_clip = DeepModel()
optimizer_value = torch.optim.SGD(model_value_clip.parameters(), lr=0.1)

optimizer_value.zero_grad()
output_value = model_value_clip(input_data)
loss_value = criterion(output_value, target)
loss_value.backward()

# Value clipping
clip_value = 0.5
torch.nn.utils.clip_grad_value_(model_value_clip.parameters(), clip_value)

print(f"Applied gradient value clipping with threshold: {clip_value}")

# Check clipped values
max_grad_values = []
for name, param in model_value_clip.named_parameters():
    if param.grad is not None:
        max_val = param.grad.abs().max().item()
        max_grad_values.append((name, max_val))
        
print("Max gradient values after clipping:")
for name, max_val in max_grad_values[:3]:
    print(f"  {name}: {max_val:.6f}")

optimizer_value.step()

print("\n=== Custom Gradient Clipping ===")

def custom_clip_grad_norm(parameters, max_norm, norm_type=2):
    """Custom implementation of gradient norm clipping"""
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    
    if len(parameters) == 0:
        return torch.tensor(0.)
    
    device = parameters[0].grad.device
    
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) 
                                           for p in parameters]), norm_type)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)
    
    return total_norm

# Test custom clipping
model_custom = DeepModel()
optimizer_custom = torch.optim.SGD(model_custom.parameters(), lr=0.1)

optimizer_custom.zero_grad()
output_custom = model_custom(input_data)
loss_custom = criterion(output_custom, target)
loss_custom.backward()

# Apply custom clipping
total_norm_custom = custom_clip_grad_norm(model_custom.parameters(), max_norm=1.0)
print(f"Custom clipping applied, total norm: {total_norm_custom:.6f}")

optimizer_custom.step()

print("\n=== Adaptive Gradient Clipping ===")

class AdaptiveGradientClipper:
    """Adaptive gradient clipping based on gradient history"""
    
    def __init__(self, parameters, percentile=95, window_size=100):
        self.parameters = list(parameters)
        self.percentile = percentile
        self.window_size = window_size
        self.grad_norms = []
    
    def clip(self):
        # Calculate current gradient norm
        total_norm = torch.norm(torch.stack([
            torch.norm(p.grad.detach()) for p in self.parameters if p.grad is not None
        ]))
        
        self.grad_norms.append(total_norm.item())
        
        # Keep only recent history
        if len(self.grad_norms) > self.window_size:
            self.grad_norms.pop(0)
        
        # Adaptive threshold based on percentile
        if len(self.grad_norms) >= 10:  # Need some history
            threshold = torch.tensor(self.grad_norms).quantile(self.percentile / 100.0)
            
            # Apply clipping if needed
            if total_norm > threshold:
                clip_coef = threshold / total_norm
                for p in self.parameters:
                    if p.grad is not None:
                        p.grad.mul_(clip_coef)
                
                print(f"Adaptive clipping applied: {total_norm:.4f} -> {threshold:.4f}")
        
        return total_norm.item()

# Test adaptive clipping
model_adaptive = DeepModel()
optimizer_adaptive = torch.optim.SGD(model_adaptive.parameters(), lr=0.1)
adaptive_clipper = AdaptiveGradientClipper(model_adaptive.parameters())

# Simulate training steps
for step in range(5):
    optimizer_adaptive.zero_grad()
    output_adap = model_adaptive(input_data)
    loss_adap = criterion(output_adap, target)
    loss_adap.backward()
    
    norm = adaptive_clipper.clip()
    optimizer_adaptive.step()
    
    print(f"Step {step}: gradient norm = {norm:.4f}")

print("\n=== Layer-wise Gradient Clipping ===")

def clip_grad_norm_layerwise(model, max_norm_per_layer=1.0):
    """Apply gradient clipping to each layer separately"""
    layer_norms = {}
    
    for name, module in model.named_modules():
        if len(list(module.parameters())) > 0:
            # Get parameters for this layer only
            layer_params = list(module.parameters())
            layer_params = [p for p in layer_params if p.grad is not None]
            
            if layer_params:
                # Calculate layer norm
                layer_norm = torch.norm(torch.stack([
                    torch.norm(p.grad.detach()) for p in layer_params
                ]))
                
                # Apply clipping to this layer
                if layer_norm > max_norm_per_layer:
                    clip_coef = max_norm_per_layer / layer_norm
                    for p in layer_params:
                        p.grad.mul_(clip_coef)
                
                layer_norms[name] = layer_norm.item()
    
    return layer_norms

# Test layer-wise clipping
model_layerwise = DeepModel()
optimizer_layerwise = torch.optim.SGD(model_layerwise.parameters(), lr=0.1)

optimizer_layerwise.zero_grad()
output_layer = model_layerwise(input_data)
loss_layer = criterion(output_layer, target)
loss_layer.backward()

layer_norms = clip_grad_norm_layerwise(model_layerwise, max_norm_per_layer=0.5)

print("Layer-wise gradient norms:")
for name, norm in list(layer_norms.items())[:3]:
    print(f"  {name}: {norm:.6f}")

optimizer_layerwise.step()

print("\n=== Gradient Clipping in Training Loop ===")

def train_with_clipping(model, data_loader, optimizer, criterion, 
                       clip_type='norm', clip_value=1.0, epochs=1):
    """Training loop with gradient clipping"""
    
    model.train()
    clip_stats = {'applied': 0, 'total_steps': 0, 'max_norm': 0}
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(data_loader):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if clip_type == 'norm':
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                clip_stats['max_norm'] = max(clip_stats['max_norm'], total_norm.item())
                if total_norm > clip_value:
                    clip_stats['applied'] += 1
            
            elif clip_type == 'value':
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
                clip_stats['applied'] += 1  # Always applied
            
            # Optimizer step
            optimizer.step()
            clip_stats['total_steps'] += 1
    
    return clip_stats

# Test training with clipping
class FakeDataLoader:
    def __init__(self, batch_size, num_batches):
        self.batch_size = batch_size
        self.num_batches = num_batches
    
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        for _ in range(self.num_batches):
            data = torch.randn(self.batch_size, 50)
            target = torch.randn(self.batch_size, 1)
            yield data, target

fake_loader = FakeDataLoader(16, 10)
model_train = DeepModel()
optimizer_train = torch.optim.Adam(model_train.parameters(), lr=0.01)

clip_stats = train_with_clipping(
    model_train, fake_loader, optimizer_train, criterion,
    clip_type='norm', clip_value=1.0
)

print(f"Clipping statistics:")
print(f"  Total steps: {clip_stats['total_steps']}")
print(f"  Clipping applied: {clip_stats['applied']}")
print(f"  Clipping rate: {clip_stats['applied']/clip_stats['total_steps']*100:.1f}%")
print(f"  Max gradient norm: {clip_stats['max_norm']:.4f}")

print("\n=== RNN Gradient Clipping ===")

# RNN with gradient clipping (common use case)
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take last time step
        return out

# Train RNN with clipping
rnn_model = SimpleRNN(input_size=20, hidden_size=50, num_layers=3, output_size=1)
rnn_optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.001)

# Generate sequence data
seq_length = 30
batch_size = 16
sequence_data = torch.randn(batch_size, seq_length, 20)
sequence_target = torch.randn(batch_size, 1)

rnn_optimizer.zero_grad()
rnn_output = rnn_model(sequence_data)
rnn_loss = criterion(rnn_output, sequence_target)
rnn_loss.backward()

# RNNs often need gradient clipping
rnn_grad_norm = torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), max_norm=5.0)
print(f"RNN gradient norm before clipping: {rnn_grad_norm:.4f}")

rnn_optimizer.step()

print("\n=== Monitoring Gradient Clipping ===")

class GradientClippingMonitor:
    """Monitor gradient clipping behavior"""
    
    def __init__(self):
        self.history = []
    
    def record(self, pre_clip_norm, post_clip_norm, threshold):
        self.history.append({
            'pre_clip_norm': pre_clip_norm,
            'post_clip_norm': post_clip_norm,
            'threshold': threshold,
            'was_clipped': pre_clip_norm > threshold
        })
    
    def get_stats(self):
        if not self.history:
            return {}
        
        total_steps = len(self.history)
        clipped_steps = sum(1 for h in self.history if h['was_clipped'])
        
        pre_norms = [h['pre_clip_norm'] for h in self.history]
        post_norms = [h['post_clip_norm'] for h in self.history]
        
        return {
            'total_steps': total_steps,
            'clipped_steps': clipped_steps,
            'clipping_rate': clipped_steps / total_steps * 100,
            'avg_pre_norm': sum(pre_norms) / len(pre_norms),
            'avg_post_norm': sum(post_norms) / len(post_norms),
            'max_pre_norm': max(pre_norms),
            'min_pre_norm': min(pre_norms)
        }

def clip_and_monitor(model, max_norm, monitor):
    """Clip gradients and record statistics"""
    # Calculate pre-clipping norm
    pre_clip_norm = torch.norm(torch.stack([
        torch.norm(p.grad.detach()) for p in model.parameters() if p.grad is not None
    ])).item()
    
    # Apply clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    # Calculate post-clipping norm
    post_clip_norm = torch.norm(torch.stack([
        torch.norm(p.grad.detach()) for p in model.parameters() if p.grad is not None
    ])).item()
    
    # Record in monitor
    monitor.record(pre_clip_norm, post_clip_norm, max_norm)
    
    return pre_clip_norm, post_clip_norm

# Test monitoring
monitor = GradientClippingMonitor()
model_monitor = DeepModel()
optimizer_monitor = torch.optim.SGD(model_monitor.parameters(), lr=0.1)

for step in range(10):
    optimizer_monitor.zero_grad()
    output_mon = model_monitor(input_data)
    loss_mon = criterion(output_mon, target)
    loss_mon.backward()
    
    pre_norm, post_norm = clip_and_monitor(model_monitor, max_norm=1.0, monitor=monitor)
    optimizer_monitor.step()

# Print monitoring statistics
stats = monitor.get_stats()
print("Gradient clipping monitoring results:")
for key, value in stats.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

print("\n=== Gradient Clipping Best Practices ===")

print("Gradient Clipping Guidelines:")
print("1. Use norm clipping (clip_grad_norm_) for most cases")
print("2. Start with max_norm=1.0 and adjust based on training")
print("3. RNNs typically need clipping (max_norm=5.0 to 10.0)")
print("4. Monitor clipping frequency - shouldn't be too high")
print("5. Consider adaptive clipping for automatic tuning")
print("6. Apply clipping before optimizer.step()")
print("7. Use layer-wise clipping for very deep networks")

print("\nCommon Thresholds:")
print("- Transformers: 1.0 - 5.0")
print("- RNNs/LSTMs: 5.0 - 10.0")
print("- CNNs: 1.0 - 2.0")
print("- GANs: 0.01 - 0.1 (discriminator)")

print("\nWarning Signs:")
print("- Clipping applied every step (threshold too low)")
print("- Never clipping (threshold too high or no explosion)")
print("- Training instability despite clipping")
print("- Gradients still exploding after clipping")

print("\n=== Gradient Clipping Complete ===") 