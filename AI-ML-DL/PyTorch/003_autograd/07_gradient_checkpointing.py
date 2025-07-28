#!/usr/bin/env python3
"""PyTorch Gradient Checkpointing - Memory-efficient training"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import time

print("=== Gradient Checkpointing Overview ===")

print("Gradient checkpointing enables:")
print("1. Memory-efficient training of deep networks")
print("2. Trading computation for memory")
print("3. Training larger models with limited GPU memory")
print("4. Selective activation saving")
print("5. Fine-grained memory control")

print("\n=== Basic Gradient Checkpointing ===")

# Simple function to checkpoint
def basic_function(x):
    """Simple function for basic checkpointing"""
    h1 = torch.relu(x)
    h2 = torch.tanh(h1)
    h3 = torch.sigmoid(h2)
    return h3

# Without checkpointing
x_basic = torch.randn(1000, 500, requires_grad=True)

# Memory usage without checkpointing
if torch.cuda.is_available():
    device = torch.device('cuda')
    x_basic = x_basic.to(device)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    y_no_checkpoint = basic_function(x_basic)
    loss_no_checkpoint = y_no_checkpoint.sum()
    loss_no_checkpoint.backward()
    
    memory_no_checkpoint = torch.cuda.max_memory_allocated() / 1e6
    print(f"Memory without checkpointing: {memory_no_checkpoint:.2f} MB")
    
    # With checkpointing
    x_basic.grad = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    y_checkpoint = checkpoint.checkpoint(basic_function, x_basic)
    loss_checkpoint = y_checkpoint.sum()
    loss_checkpoint.backward()
    
    memory_checkpoint = torch.cuda.max_memory_allocated() / 1e6
    print(f"Memory with checkpointing: {memory_checkpoint:.2f} MB")
    print(f"Memory savings: {((memory_no_checkpoint - memory_checkpoint) / memory_no_checkpoint * 100):.1f}%")
else:
    print("CUDA not available, using CPU for demonstration")
    y_checkpoint = checkpoint.checkpoint(basic_function, x_basic)
    loss_checkpoint = y_checkpoint.sum()
    loss_checkpoint.backward()
    print("Gradient checkpointing completed on CPU")

print("\n=== Checkpointing in Neural Networks ===")

class CheckpointedBlock(nn.Module):
    """Neural network block with gradient checkpointing"""
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        # Use checkpointing for this block
        return checkpoint.checkpoint(self.layers, x)

class CheckpointedNetwork(nn.Module):
    """Network with multiple checkpointed blocks"""
    def __init__(self, input_dim, hidden_dim, num_blocks):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            CheckpointedBlock(hidden_dim) for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.use_checkpointing = True
    
    def forward(self, x):
        x = self.input_layer(x)
        
        for block in self.blocks:
            if self.use_checkpointing:
                x = checkpoint.checkpoint(block.layers, x)
            else:
                x = block.layers(x)
        
        x = self.output_layer(x)
        return x

# Test checkpointed network
net = CheckpointedNetwork(input_dim=128, hidden_dim=256, num_blocks=10)
input_data = torch.randn(32, 128, requires_grad=True)

print(f"Network has {sum(p.numel() for p in net.parameters())} parameters")

# With checkpointing
net.use_checkpointing = True
output_checkpointed = net(input_data)
loss_net = output_checkpointed.sum()

print(f"Forward pass with checkpointing completed")
print(f"Output shape: {output_checkpointed.shape}")

loss_net.backward()
print(f"Backward pass with checkpointing completed")

print("\n=== Custom Checkpointing Function ===")

def custom_checkpoint_function(func, *args, preserve_rng_state=True):
    """Custom implementation of gradient checkpointing"""
    
    class CheckpointFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, func, preserve_rng_state, *args):
            # Save function and RNG state
            ctx.func = func
            ctx.preserve_rng_state = preserve_rng_state
            ctx.args = args
            
            # Save RNG state if required
            if preserve_rng_state:
                ctx.fwd_cpu_state = torch.get_rng_state()
                if torch.cuda.is_available():
                    ctx.fwd_cuda_state = torch.cuda.get_rng_state()
            
            # Run forward pass without saving activations
            with torch.no_grad():
                outputs = func(*args)
            
            return outputs
        
        @staticmethod
        def backward(ctx, *grad_outputs):
            # Restore RNG state
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if torch.cuda.is_available():
                    torch.cuda.set_rng_state(ctx.fwd_cuda_state)
            
            # Re-run forward pass with gradients
            args = ctx.args
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    arg.requires_grad_(True)
            
            with torch.enable_grad():
                outputs = ctx.func(*args)
            
            # Compute gradients
            if isinstance(outputs, torch.Tensor):
                outputs = (outputs,)
            
            grad_inputs = torch.autograd.grad(
                outputs, args, grad_outputs=grad_outputs,
                only_inputs=True, allow_unused=True
            )
            
            return (None, None) + grad_inputs
    
    return CheckpointFunction.apply(func, preserve_rng_state, *args)

# Test custom checkpointing
def test_function(x, y):
    """Function to test custom checkpointing"""
    z = torch.matmul(x, y.t())
    return torch.relu(z)

x_custom = torch.randn(100, 50, requires_grad=True)
y_custom = torch.randn(80, 50, requires_grad=True)

# Use custom checkpointing
result_custom = custom_checkpoint_function(test_function, x_custom, y_custom)
loss_custom = result_custom.sum()
loss_custom.backward()

print(f"Custom checkpointing result shape: {result_custom.shape}")
print(f"Gradients computed successfully")

print("\n=== Selective Checkpointing ===")

class SelectiveCheckpointNetwork(nn.Module):
    """Network with selective checkpointing based on layer type"""
    def __init__(self, layers_config):
        super().__init__()
        self.layers = nn.ModuleList()
        self.checkpoint_flags = []
        
        for layer_type, dim, checkpoint in layers_config:
            if layer_type == 'linear':
                self.layers.append(nn.Linear(dim[0], dim[1]))
            elif layer_type == 'conv':
                self.layers.append(nn.Conv2d(dim[0], dim[1], dim[2]))
            elif layer_type == 'relu':
                self.layers.append(nn.ReLU())
            elif layer_type == 'norm':
                self.layers.append(nn.BatchNorm1d(dim))
            
            self.checkpoint_flags.append(checkpoint)
    
    def forward(self, x):
        for layer, use_checkpoint in zip(self.layers, self.checkpoint_flags):
            if use_checkpoint:
                x = checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
        return x

# Configure selective checkpointing
config = [
    ('linear', (64, 128), False),    # Don't checkpoint first layer
    ('relu', None, False),
    ('linear', (128, 256), True),    # Checkpoint expensive layers
    ('relu', None, True),
    ('linear', (256, 512), True),
    ('relu', None, True),
    ('linear', (512, 256), True),
    ('relu', None, False),
    ('linear', (256, 1), False),     # Don't checkpoint output layer
]

selective_net = SelectiveCheckpointNetwork(config)
selective_input = torch.randn(16, 64)

selective_output = selective_net(selective_input)
selective_loss = selective_output.sum()
selective_loss.backward()

print(f"Selective checkpointing network output: {selective_output.shape}")

print("\n=== Checkpointing with Transformer Blocks ===")

class CheckpointedTransformerBlock(nn.Module):
    """Transformer block with gradient checkpointing"""
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def _attention_block(self, x):
        """Attention sub-block"""
        attn_output, _ = self.self_attn(x, x, x)
        return self.norm1(x + self.dropout(attn_output))
    
    def _feedforward_block(self, x):
        """Feedforward sub-block"""
        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        return self.norm2(x + self.dropout(ff_output))
    
    def forward(self, x):
        # Checkpoint both attention and feedforward blocks
        x = checkpoint.checkpoint(self._attention_block, x)
        x = checkpoint.checkpoint(self._feedforward_block, x)
        return x

class CheckpointedTransformer(nn.Module):
    """Multi-layer transformer with checkpointing"""
    def __init__(self, num_layers, d_model, nhead, dim_feedforward):
        super().__init__()
        self.layers = nn.ModuleList([
            CheckpointedTransformerBlock(d_model, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Test checkpointed transformer
transformer = CheckpointedTransformer(
    num_layers=6, d_model=512, nhead=8, dim_feedforward=2048
)

transformer_input = torch.randn(8, 100, 512)  # (batch, seq_len, d_model)
transformer_output = transformer(transformer_input)
transformer_loss = transformer_output.sum()

print(f"Checkpointed transformer output shape: {transformer_output.shape}")

transformer_loss.backward()
print("Transformer checkpointing completed")

print("\n=== Activation Checkpointing Strategies ===")

class AdaptiveCheckpointNetwork(nn.Module):
    """Network with adaptive checkpointing based on memory usage"""
    def __init__(self, layers, memory_threshold_mb=100):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.memory_threshold = memory_threshold_mb * 1e6  # Convert to bytes
        self.checkpoint_decisions = []
    
    def _should_checkpoint(self, layer_idx, input_tensor):
        """Decide whether to checkpoint based on memory usage"""
        if not torch.cuda.is_available():
            return layer_idx % 2 == 0  # Fallback: checkpoint every other layer
        
        current_memory = torch.cuda.memory_allocated()
        
        # Estimate memory needed for this layer
        estimated_activation_memory = input_tensor.numel() * input_tensor.element_size()
        
        return (current_memory + estimated_activation_memory) > self.memory_threshold
    
    def forward(self, x):
        self.checkpoint_decisions = []
        
        for i, layer in enumerate(self.layers):
            should_checkpoint = self._should_checkpoint(i, x)
            self.checkpoint_decisions.append(should_checkpoint)
            
            if should_checkpoint:
                x = checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
        
        return x

# Test adaptive checkpointing
adaptive_layers = [
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 1)
]

adaptive_net = AdaptiveCheckpointNetwork(adaptive_layers, memory_threshold_mb=50)
adaptive_input = torch.randn(64, 256)

adaptive_output = adaptive_net(adaptive_input)
print(f"Adaptive checkpointing decisions: {adaptive_net.checkpoint_decisions}")

print("\n=== Performance Analysis ===")

def benchmark_checkpointing(model_fn, input_data, use_checkpoint=True, iterations=10):
    """Benchmark checkpointing vs no checkpointing"""
    
    # Time measurement
    start_time = time.time()
    
    for _ in range(iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        if use_checkpoint:
            output = checkpoint.checkpoint(model_fn, input_data)
        else:
            output = model_fn(input_data)
        
        loss = output.sum()
        loss.backward()
        
        # Clear gradients for next iteration
        if hasattr(input_data, 'grad') and input_data.grad is not None:
            input_data.grad.zero_()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    return (end_time - start_time) / iterations

# Benchmark function
def benchmark_function(x):
    """Function for benchmarking"""
    for _ in range(5):  # Multiple operations
        x = torch.relu(x)
        x = torch.tanh(x)
    return x

benchmark_input = torch.randn(100, 100, requires_grad=True)

# Benchmark with and without checkpointing
time_no_checkpoint = benchmark_checkpointing(benchmark_function, benchmark_input, False, 5)
time_checkpoint = benchmark_checkpointing(benchmark_function, benchmark_input, True, 5)

print(f"Time without checkpointing: {time_no_checkpoint:.4f}s")
print(f"Time with checkpointing: {time_checkpoint:.4f}s")
print(f"Time overhead: {((time_checkpoint - time_no_checkpoint) / time_no_checkpoint * 100):.1f}%")

print("\n=== Debugging Checkpointed Models ===")

class DebuggableCheckpointBlock(nn.Module):
    """Checkpointed block with debugging capabilities"""
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.debug_mode = False
    
    def _forward_with_debug(self, x):
        """Forward pass with debug information"""
        if self.debug_mode:
            print(f"Checkpoint block input norm: {x.norm():.6f}")
        
        output = self.layers(x)
        
        if self.debug_mode:
            print(f"Checkpoint block output norm: {output.norm():.6f}")
        
        return output
    
    def forward(self, x):
        if self.training:
            return checkpoint.checkpoint(self._forward_with_debug, x)
        else:
            return self._forward_with_debug(x)

# Test debugging
debug_block = DebuggableCheckpointBlock(64)
debug_input = torch.randn(16, 64)

# Enable debug mode
debug_block.debug_mode = True
debug_block.train()

debug_output = debug_block(debug_input)
debug_loss = debug_output.sum()
debug_loss.backward()

print("Debugging checkpointed block completed")

print("\n=== Integration with Mixed Precision ===")

if torch.cuda.is_available():
    from torch.cuda.amp import autocast, GradScaler
    
    class AMPCheckpointedModel(nn.Module):
        """Model with both AMP and gradient checkpointing"""
        def __init__(self, layers):
            super().__init__()
            self.layers = nn.ModuleList(layers)
        
        def forward(self, x):
            for i, layer in enumerate(self.layers):
                if i % 2 == 0:  # Checkpoint every other layer
                    # AMP autocast is preserved through checkpointing
                    x = checkpoint.checkpoint(layer, x)
                else:
                    x = layer(x)
            return x
    
    # Test AMP + checkpointing
    amp_layers = [
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    ]
    
    amp_model = AMPCheckpointedModel(amp_layers)
    if torch.cuda.is_available():
        amp_model = amp_model.cuda()
        amp_input = torch.randn(32, 128, device='cuda')
        
        scaler = GradScaler()
        
        with autocast():
            amp_output = amp_model(amp_input)
            amp_loss = amp_output.sum()
        
        scaler.scale(amp_loss).backward()
        print("AMP + checkpointing integration successful")

print("\n=== Best Practices for Gradient Checkpointing ===")

print("Gradient Checkpointing Guidelines:")
print("1. Use for memory-bound training scenarios")
print("2. Checkpoint expensive computational blocks")
print("3. Avoid checkpointing very cheap operations")
print("4. Consider selective checkpointing strategies")
print("5. Profile memory usage to find optimal checkpointing points")
print("6. Be aware of 2x computation overhead")
print("7. Test checkpointing with your specific architecture")

print("\nWhen to Use Checkpointing:")
print("- Training very deep networks")
print("- Limited GPU memory")
print("- Large batch sizes")
print("- Complex architectures (Transformers, ResNets)")
print("- When activations dominate memory usage")

print("\nWhen NOT to Use Checkpointing:")
print("- Compute-bound scenarios")
print("- Small models with ample memory")
print("- When training speed is critical")
print("- Very cheap forward operations")
print("- Models with many branches/skip connections")

print("\nPerformance Tips:")
print("- Checkpoint entire sub-networks, not individual operations")
print("- Use with mixed precision training")
print("- Consider adaptive checkpointing strategies")
print("- Profile before and after implementation")
print("- Balance memory savings vs computation cost")

print("\n=== Gradient Checkpointing Complete ===") 