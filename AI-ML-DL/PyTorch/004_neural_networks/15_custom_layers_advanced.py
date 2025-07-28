#!/usr/bin/env python3
"""PyTorch Custom Layers Advanced - Advanced custom layer implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=== Advanced Custom Layers Overview ===")

print("Advanced custom layers provide:")
print("1. Complex functionality beyond built-in layers")
print("2. Specialized operations for specific domains")
print("3. Performance-optimized implementations")
print("4. Learnable and adaptive components")
print("5. Research and experimental features")

print("\n=== Advanced Attention Layers ===")

class MultiHeadSelfAttention(nn.Module):
    """Advanced multi-head self-attention with relative position encoding"""
    def __init__(self, d_model, num_heads, max_relative_position=10, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_relative_position = max_relative_position
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Relative position embeddings
        vocab_size = 2 * max_relative_position + 1
        self.relative_position_k = nn.Embedding(vocab_size, self.d_k)
        self.relative_position_v = nn.Embedding(vocab_size, self.d_k)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Relative position indices
        range_vec = torch.arange(seq_len, device=x.device)
        relative_positions = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)
        relative_positions = torch.clamp(
            relative_positions, -self.max_relative_position, self.max_relative_position
        ) + self.max_relative_position
        
        # Attention with relative positions
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add relative position bias
        rel_pos_k = self.relative_position_k(relative_positions)
        rel_pos_scores = torch.einsum('bhld,lrd->bhlr', Q, rel_pos_k)
        attention_scores += rel_pos_scores
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Add relative position bias to values
        rel_pos_v = self.relative_position_v(relative_positions)
        rel_pos_context = torch.einsum('bhlr,lrd->bhld', attention_weights, rel_pos_v)
        context += rel_pos_context
        
        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(context)
        
        return output, attention_weights

# Test advanced attention
adv_attention = MultiHeadSelfAttention(d_model=256, num_heads=8)
attn_input = torch.randn(2, 20, 256)
attn_output, attn_weights = adv_attention(attn_input)

print(f"Advanced attention output: {attn_output.shape}")
print(f"Attention weights: {attn_weights.shape}")

print("\n=== Adaptive Layers ===")

class AdaptiveConv2d(nn.Module):
    """Convolutional layer with adaptive kernel size"""
    def __init__(self, in_channels, out_channels, max_kernel_size=7, stride=1, padding_mode='same'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_kernel_size = max_kernel_size
        self.stride = stride
        self.padding_mode = padding_mode
        
        # Create convolutions for different kernel sizes
        self.convs = nn.ModuleList()
        for k in range(1, max_kernel_size + 1, 2):  # 1, 3, 5, 7
            padding = k // 2 if padding_mode == 'same' else 0
            self.convs.append(nn.Conv2d(in_channels, out_channels, k, stride, padding))
        
        # Learnable weights for kernel selection
        self.kernel_weights = nn.Parameter(torch.ones(len(self.convs)))
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, len(self.convs)),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # Compute adaptive weights
        gate_weights = self.gate(x)  # [batch_size, num_kernels]
        
        # Apply all convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(x))
        
        # Weighted combination
        output = 0
        for i, conv_out in enumerate(conv_outputs):
            weight = gate_weights[:, i:i+1, None, None]  # Broadcast to spatial dims
            output += weight * conv_out
        
        return output

# Test adaptive convolution
adaptive_conv = AdaptiveConv2d(32, 64, max_kernel_size=7)
conv_input = torch.randn(4, 32, 16, 16)
conv_output = adaptive_conv(conv_input)

print(f"Adaptive conv output: {conv_output.shape}")

class AdaptiveActivation(nn.Module):
    """Activation function that adapts based on input statistics"""
    def __init__(self, num_features, num_activations=3):
        super().__init__()
        self.num_features = num_features
        self.num_activations = num_activations
        
        # Learnable mixing weights
        self.mix_weights = nn.Parameter(torch.ones(num_features, num_activations))
        
        # Statistics computation
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.momentum = 0.1
    
    def forward(self, x):
        # Update running statistics (during training)
        if self.training:
            if len(x.shape) == 4:  # Conv feature maps
                mean = x.mean(dim=[0, 2, 3])
                var = x.var(dim=[0, 2, 3], unbiased=False)
            else:  # Fully connected
                mean = x.mean(dim=0)
                var = x.var(dim=0, unbiased=False)
            
            self.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(var, alpha=self.momentum)
        
        # Compute adaptive weights based on current statistics
        if len(x.shape) == 4:
            current_mean = x.mean(dim=[0, 2, 3])
            current_var = x.var(dim=[0, 2, 3], unbiased=False)
        else:
            current_mean = x.mean(dim=0)
            current_var = x.var(dim=0, unbiased=False)
        
        # Adaptation signal
        mean_diff = torch.abs(current_mean - self.running_mean)
        var_diff = torch.abs(current_var - self.running_var)
        adaptation = torch.sigmoid(mean_diff + var_diff)
        
        # Different activation functions
        activations = [
            torch.relu(x),
            torch.tanh(x),
            torch.sigmoid(x)
        ]
        
        # Compute mixing weights
        mix_weights = F.softmax(self.mix_weights, dim=1)
        
        # Apply adaptive activation
        result = 0
        for i, activation in enumerate(activations):
            weight = mix_weights[:, i]
            if len(x.shape) == 4:
                weight = weight.view(1, -1, 1, 1)
            else:
                weight = weight.view(1, -1)
            result += weight * activation
        
        return result

print("\n=== Specialized Domain Layers ===")

class SpectralConv1d(nn.Module):
    """1D Fourier Neural Operator layer"""
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Fourier weights
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, self.modes, 2) * 0.02
        )
    
    def compl_mul1d(self, a, b):
        """Complex multiplication for 1D"""
        return torch.einsum("bix,iox->box", a, torch.view_as_complex(b))
    
    def forward(self, x):
        batch_size, channels, length = x.shape
        
        # Forward FFT
        x_ft = torch.fft.rfft(x)
        
        # Truncate to lower modes
        x_ft = x_ft[:, :, :self.modes]
        
        # Multiply with Fourier weights
        out_ft = self.compl_mul1d(x_ft, self.weights)
        
        # Inverse FFT
        x = torch.fft.irfft(out_ft, n=length)
        
        return x

class GraphConv(nn.Module):
    """Graph convolution layer"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        x: node features [num_nodes, in_features]
        adj: adjacency matrix [num_nodes, num_nodes]
        """
        # Linear transformation
        support = torch.mm(x, self.weight)
        
        # Graph convolution
        output = torch.mm(adj, support)
        
        return output + self.bias

class CapsuleLayer(nn.Module):
    """Capsule layer implementation"""
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, 
                 kernel_size=None, stride=None, num_iterations=3):
        super().__init__()
        
        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules
        
        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(
                torch.randn(num_route_nodes, num_capsules, in_channels, out_channels)
            )
        else:
            self.capsules = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                         stride=stride, padding=0) for _ in range(num_capsules)
            ])
    
    def squash(self, tensor, dim=-1):
        """Squashing function"""
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm + 1e-8)
    
    def forward(self, x):
        if self.num_route_nodes != -1:
            # Routing by agreement
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
            
            logits = torch.zeros_like(priors)
            for i in range(self.num_iterations):
                probs = F.softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=1, keepdim=True))
                
                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
            
            return outputs.squeeze(1)
        else:
            # Primary capsules
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            return self.squash(outputs)

print("\n=== Performance-Optimized Layers ===")

class DepthwiseSeparableConv(nn.Module):
    """Optimized depthwise separable convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 bias=False, channel_multiplier=1):
        super().__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels * channel_multiplier, 
            kernel_size=kernel_size, stride=stride, padding=padding, 
            groups=in_channels, bias=bias
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(
            in_channels * channel_multiplier, out_channels, 
            kernel_size=1, bias=bias
        )
        
        # Batch normalization and activation
        self.bn1 = nn.BatchNorm2d(in_channels * channel_multiplier)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x

class ShuffleBlock(nn.Module):
    """Channel shuffle block for efficient group convolutions"""
    def __init__(self, groups):
        super().__init__()
        self.groups = groups
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // self.groups
        
        # Reshape and transpose
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, channels, height, width)
        
        return x

class MobileInvertedResidual(nn.Module):
    """Mobile inverted residual block"""
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = int(round(in_channels * expand_ratio))
        
        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Pointwise projection
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

print("\n=== Regularization and Normalization Layers ===")

class SpectralNorm(nn.Module):
    """Spectral normalization wrapper"""
    def __init__(self, module, power_iterations=1):
        super().__init__()
        self.module = module
        self.power_iterations = power_iterations
        
        if hasattr(module, 'weight'):
            w = module.weight.data
            height = w.size(0)
            width = w.view(height, -1).size(1)
            
            u = nn.Parameter(w.new_empty(height).normal_(0, 1), requires_grad=False)
            v = nn.Parameter(w.new_empty(width).normal_(0, 1), requires_grad=False)
            
            self.register_parameter('weight_u', u)
            self.register_parameter('weight_v', v)
            
            self.weight_bar = nn.Parameter(w.data)
            del self.module._parameters['weight']
    
    def _update_u_v(self):
        """Power iteration to compute spectral norm"""
        w = self.weight_bar.view(self.weight_bar.size(0), -1)
        
        for _ in range(self.power_iterations):
            v = F.normalize(torch.mv(w.t(), self.weight_u), dim=0, eps=1e-12)
            u = F.normalize(torch.mv(w, v), dim=0, eps=1e-12)
        
        self.weight_u.data.copy_(u)
        self.weight_v.data.copy_(v)
        
        sigma = torch.dot(u, torch.mv(w, v))
        return sigma
    
    def forward(self, *args, **kwargs):
        if self.training:
            sigma = self._update_u_v()
            self.module.weight = self.weight_bar / sigma
        else:
            sigma = torch.dot(self.weight_u, torch.mv(self.weight_bar.view(self.weight_bar.size(0), -1), self.weight_v))
            self.module.weight = self.weight_bar / sigma
        
        return self.module(*args, **kwargs)

class DropBlock2D(nn.Module):
    """DropBlock regularization for 2D feature maps"""
    def __init__(self, drop_rate, block_size):
        super().__init__()
        self.drop_rate = drop_rate
        self.block_size = block_size
    
    def forward(self, x):
        if not self.training or self.drop_rate == 0:
            return x
        
        gamma = self._compute_gamma(x)
        
        # Sample mask
        mask = torch.bernoulli(torch.ones_like(x) * gamma)
        
        # Apply block structure
        mask = F.max_pool2d(
            mask, kernel_size=self.block_size, stride=1, 
            padding=self.block_size // 2
        )
        
        # Invert mask and normalize
        mask = 1 - mask
        normalize_scale = mask.numel() / mask.sum()
        
        return x * mask * normalize_scale
    
    def _compute_gamma(self, x):
        """Compute gamma parameter"""
        return self.drop_rate / (self.block_size ** 2)

print("\n=== Custom Layer Utilities ===")

class LayerFactory:
    """Factory for creating custom layers"""
    
    @staticmethod
    def create_residual_block(channels, kernel_size=3, dropout=0.0):
        """Create a residual block"""
        class ResidualBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
                self.bn1 = nn.BatchNorm2d(channels)
                self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
                self.bn2 = nn.BatchNorm2d(channels)
                self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
                self.relu = nn.ReLU(inplace=True)
            
            def forward(self, x):
                identity = x
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.dropout(out)
                out = self.bn2(self.conv2(out))
                out += identity
                return self.relu(out)
        
        return ResidualBlock()
    
    @staticmethod
    def create_attention_block(d_model, num_heads):
        """Create an attention block"""
        return MultiHeadSelfAttention(d_model, num_heads)
    
    @staticmethod
    def create_mobile_block(in_channels, out_channels, stride=1, expand_ratio=6):
        """Create a mobile inverted residual block"""
        return MobileInvertedResidual(in_channels, out_channels, stride, expand_ratio)

class LayerRegistry:
    """Registry for custom layer types"""
    _layers = {}
    
    @classmethod
    def register(cls, name, layer_class):
        """Register a custom layer"""
        cls._layers[name] = layer_class
    
    @classmethod
    def create(cls, name, *args, **kwargs):
        """Create a layer by name"""
        if name not in cls._layers:
            raise ValueError(f"Layer {name} not registered")
        return cls._layers[name](*args, **kwargs)
    
    @classmethod
    def list_layers(cls):
        """List all registered layers"""
        return list(cls._layers.keys())

# Register custom layers
LayerRegistry.register('adaptive_conv', AdaptiveConv2d)
LayerRegistry.register('spectral_conv', SpectralConv1d)
LayerRegistry.register('depthwise_sep', DepthwiseSeparableConv)

print("Registered layers:", LayerRegistry.list_layers())

print("\n=== Advanced Custom Layer Testing ===")

# Test multiple custom layers
test_layers = [
    ('MultiHeadSelfAttention', MultiHeadSelfAttention(128, 8)),
    ('AdaptiveConv2d', AdaptiveConv2d(16, 32)),
    ('DepthwiseSeparableConv', DepthwiseSeparableConv(16, 32, 3, padding=1)),
    ('MobileInvertedResidual', MobileInvertedResidual(32, 64, stride=2, expand_ratio=6))
]

for name, layer in test_layers:
    print(f"\nTesting {name}:")
    
    if 'Attention' in name:
        test_input = torch.randn(2, 10, 128)
        output, _ = layer(test_input)
    elif 'Conv' in name or 'Mobile' in name:
        if 'Adaptive' in name:
            test_input = torch.randn(2, 16, 32, 32)
        else:
            test_input = torch.randn(2, 16, 32, 32)
        output = layer(test_input)
    
    print(f"  Input: {test_input.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in layer.parameters())}")

print("\n=== Custom Layer Best Practices ===")

print("Advanced Custom Layer Guidelines:")
print("1. Inherit from nn.Module for proper integration")
print("2. Implement __init__ and forward methods")
print("3. Use nn.Parameter for learnable parameters")
print("4. Use register_buffer for non-learnable state")
print("5. Handle different input shapes gracefully")
print("6. Implement proper parameter initialization")
print("7. Consider memory and computational efficiency")

print("\nPerformance Optimization:")
print("- Use in-place operations when safe")
print("- Minimize memory allocations")
print("- Consider fused operations")
print("- Use efficient tensor operations")
print("- Profile custom implementations")
print("- Compare against built-in alternatives")

print("\nTesting and Validation:")
print("- Test with different input shapes")
print("- Verify gradient flow")
print("- Check parameter updates")
print("- Test training vs evaluation modes")
print("- Validate mathematical correctness")
print("- Benchmark performance")

print("\nDocumentation and Maintenance:")
print("- Document layer purpose and usage")
print("- Provide clear parameter descriptions")
print("- Include usage examples")
print("- Consider backwards compatibility")
print("- Plan for different PyTorch versions")

print("\n=== Advanced Custom Layers Complete ===") 