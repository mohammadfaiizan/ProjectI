import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple

# Basic Attention Mechanisms
class AdditiveAttention(nn.Module):
    """Additive (Bahdanau) attention mechanism"""
    
    def __init__(self, query_dim, key_dim, hidden_dim):
        super().__init__()
        self.query_projection = nn.Linear(query_dim, hidden_dim, bias=False)
        self.key_projection = nn.Linear(key_dim, hidden_dim, bias=False)
        self.energy_projection = nn.Linear(hidden_dim, 1, bias=False)
        self.hidden_dim = hidden_dim
    
    def forward(self, query, keys, values, mask=None):
        # query: [batch_size, query_dim]
        # keys: [batch_size, seq_len, key_dim]
        # values: [batch_size, seq_len, value_dim]
        
        batch_size, seq_len, _ = keys.size()
        
        # Project query and keys
        query_proj = self.query_projection(query).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        keys_proj = self.key_projection(keys)  # [batch_size, seq_len, hidden_dim]
        
        # Compute energy
        energy = torch.tanh(query_proj + keys_proj)  # [batch_size, seq_len, hidden_dim]
        attention_scores = self.energy_projection(energy).squeeze(2)  # [batch_size, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
        
        # Compute context
        context = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)  # [batch_size, value_dim]
        
        return context, attention_weights

class MultiplicativeAttention(nn.Module):
    """Multiplicative (Luong) attention mechanism"""
    
    def __init__(self, query_dim, key_dim, attention_type='general'):
        super().__init__()
        self.attention_type = attention_type
        
        if attention_type == 'general':
            self.linear = nn.Linear(query_dim, key_dim, bias=False)
        elif attention_type == 'concat':
            self.linear = nn.Linear(query_dim + key_dim, key_dim, bias=False)
            self.vector = nn.Parameter(torch.randn(key_dim))
    
    def forward(self, query, keys, values, mask=None):
        # query: [batch_size, query_dim]
        # keys: [batch_size, seq_len, key_dim]
        # values: [batch_size, seq_len, value_dim]
        
        batch_size, seq_len, key_dim = keys.size()
        
        if self.attention_type == 'dot':
            # Dot product attention (requires same dimensions)
            attention_scores = torch.bmm(keys, query.unsqueeze(2)).squeeze(2)
            
        elif self.attention_type == 'general':
            # General multiplicative attention
            query_proj = self.linear(query)  # [batch_size, key_dim]
            attention_scores = torch.bmm(keys, query_proj.unsqueeze(2)).squeeze(2)
            
        elif self.attention_type == 'concat':
            # Concatenation-based attention
            query_expanded = query.unsqueeze(1).expand(batch_size, seq_len, -1)
            concat_features = torch.cat([keys, query_expanded], dim=2)
            attention_scores = torch.bmm(
                torch.tanh(self.linear(concat_features)),
                self.vector.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, 1)
            ).squeeze(2)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Compute context
        context = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)
        
        return context, attention_weights

# Self-Attention
class SelfAttention(nn.Module):
    """Self-attention mechanism"""
    
    def __init__(self, embed_dim, num_heads=1, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, embed_dim]
        batch_size, seq_len, embed_dim = x.size()
        
        # Project to Q, K, V
        Q = self.query_projection(x)
        K = self.key_projection(x)
        V = self.value_projection(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Reshape and project
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        output = self.output_projection(attention_output)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q, K, V: [batch_size, num_heads, seq_len, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 3:  # [batch_size, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None, need_weights=False):
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        # Linear transformations and split into heads
        Q = self.q_linear(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and put through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.embed_dim
        )
        output = self.out_linear(attention_output)
        
        if need_weights:
            return output, attention_weights
        return output
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply to values
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights

# Cross-Attention
class CrossAttention(nn.Module):
    """Cross-attention mechanism between two sequences"""
    
    def __init__(self, query_dim, key_dim, value_dim, embed_dim, num_heads=1, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(query_dim, embed_dim)
        self.k_linear = nn.Linear(key_dim, embed_dim)
        self.v_linear = nn.Linear(value_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_kv = key.size(1)
        
        # Project and reshape
        Q = self.q_linear(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Reshape and project
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.embed_dim
        )
        output = self.out_linear(attention_output)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights

# Position-wise Attention
class PositionalAttention(nn.Module):
    """Attention mechanism with positional encoding"""
    
    def __init__(self, embed_dim, max_len=5000, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
        # Attention layers
        self.attention = MultiHeadAttention(embed_dim, num_heads=8, dropout=dropout)
    
    def forward(self, x, mask=None):
        # Add positional encoding
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        x = self.dropout(x)
        
        # Apply attention
        output = self.attention(x, x, x, mask)
        
        return output

# Local Attention
class LocalAttention(nn.Module):
    """Local attention mechanism (attends to a local window)"""
    
    def __init__(self, embed_dim, window_size=10, num_heads=1, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        
        # Project to Q, K, V
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Create local attention mask
        local_mask = self.create_local_mask(seq_len, self.window_size, x.device)
        if mask is not None:
            local_mask = local_mask & mask
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, local_mask)
        
        # Reshape and project
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        output = self.out_linear(attention_output)
        
        return output, attention_weights
    
    def create_local_mask(self, seq_len, window_size, device):
        """Create local attention mask"""
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = True
        
        return mask
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights

# Sparse Attention
class SparseAttention(nn.Module):
    """Sparse attention mechanism"""
    
    def __init__(self, embed_dim, num_heads=1, block_size=64, num_random_blocks=2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.block_size = block_size
        self.num_random_blocks = num_random_blocks
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        
        # Project to Q, K, V
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Create sparse attention pattern
        sparse_mask = self.create_sparse_mask(seq_len, x.device)
        if mask is not None:
            sparse_mask = sparse_mask & mask
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, sparse_mask)
        
        # Reshape and project
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        output = self.out_linear(attention_output)
        
        return output, attention_weights
    
    def create_sparse_mask(self, seq_len, device):
        """Create sparse attention mask with block and random patterns"""
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        
        # Block-local pattern
        num_blocks = seq_len // self.block_size
        for i in range(num_blocks):
            start = i * self.block_size
            end = min((i + 1) * self.block_size, seq_len)
            mask[start:end, start:end] = True
        
        # Random connections
        for _ in range(self.num_random_blocks):
            i = torch.randint(0, seq_len, (1,)).item()
            j = torch.randint(0, seq_len, (1,)).item()
            mask[i, j] = True
            mask[j, i] = True  # Make symmetric
        
        return mask
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights

# Attention with Relative Position Encoding
class RelativePositionAttention(nn.Module):
    """Attention with relative position encoding (Transformer-XL style)"""
    
    def __init__(self, embed_dim, num_heads=1, max_relative_position=32, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_relative_position = max_relative_position
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        # Relative position embeddings
        self.relative_position_k = nn.Embedding(2 * max_relative_position + 1, self.head_dim)
        self.relative_position_v = nn.Embedding(2 * max_relative_position + 1, self.head_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        
        # Project to Q, K, V
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply attention with relative positions
        attention_output, attention_weights = self.attention_with_relative_position(Q, K, V, mask)
        
        # Reshape and project
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        output = self.out_linear(attention_output)
        
        return output, attention_weights
    
    def attention_with_relative_position(self, Q, K, V, mask=None):
        batch_size, num_heads, seq_len, head_dim = Q.size()
        
        # Standard attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Relative position scores
        relative_positions = self.get_relative_positions(seq_len, Q.device)
        relative_position_embeddings_k = self.relative_position_k(relative_positions)
        
        # Add relative position scores
        relative_scores = torch.matmul(Q.unsqueeze(-2), relative_position_embeddings_k.transpose(-2, -1)).squeeze(-2)
        scores = scores + relative_scores
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply to values with relative positions
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights
    
    def get_relative_positions(self, seq_len, device):
        """Get relative position indices"""
        range_vec = torch.arange(seq_len, device=device)
        range_mat = range_vec.unsqueeze(0).expand(seq_len, seq_len)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # Clip to max relative position
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position
        )
        
        # Shift to positive indices
        final_mat = distance_mat_clipped + self.max_relative_position
        
        return final_mat

# Attention Visualization
def visualize_attention_weights(attention_weights, tokens=None, save_path=None):
    """Visualize attention weights as a heatmap"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Convert to numpy if tensor
    if torch.is_tensor(attention_weights):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # If multi-head, take the first head
    if attention_weights.ndim == 4:
        attention_weights = attention_weights[0, 0]  # First batch, first head
    elif attention_weights.ndim == 3:
        attention_weights = attention_weights[0]  # First batch
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, cmap='Blues', cbar=True, square=True)
    
    if tokens:
        plt.xticks(range(len(tokens)), tokens, rotation=45)
        plt.yticks(range(len(tokens)), tokens, rotation=0)
    
    plt.title('Attention Weights')
    plt.xlabel('Keys')
    plt.ylabel('Queries')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

if __name__ == "__main__":
    print("Testing attention mechanisms...")
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    embed_dim = 64
    
    # Create test input
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Test Self-Attention
    print("Testing Self-Attention...")
    self_attn = SelfAttention(embed_dim, num_heads=4)
    output, weights = self_attn(x)
    print(f"Self-attention output shape: {output.shape}")
    print(f"Self-attention weights shape: {weights.shape}")
    
    # Test Multi-Head Attention
    print("\nTesting Multi-Head Attention...")
    mha = MultiHeadAttention(embed_dim, num_heads=8)
    output = mha(x, x, x)
    print(f"Multi-head attention output shape: {output.shape}")
    
    # Test Cross-Attention
    print("\nTesting Cross-Attention...")
    y = torch.randn(batch_size, seq_len + 5, embed_dim)  # Different sequence length
    cross_attn = CrossAttention(embed_dim, embed_dim, embed_dim, embed_dim, num_heads=4)
    output, weights = cross_attn(x, y, y)
    print(f"Cross-attention output shape: {output.shape}")
    print(f"Cross-attention weights shape: {weights.shape}")
    
    # Test Local Attention
    print("\nTesting Local Attention...")
    local_attn = LocalAttention(embed_dim, window_size=5, num_heads=4)
    output, weights = local_attn(x)
    print(f"Local attention output shape: {output.shape}")
    
    # Test Sparse Attention
    print("\nTesting Sparse Attention...")
    sparse_attn = SparseAttention(embed_dim, num_heads=4, block_size=4)
    output, weights = sparse_attn(x)
    print(f"Sparse attention output shape: {output.shape}")
    
    # Test Additive Attention
    print("\nTesting Additive Attention...")
    additive_attn = AdditiveAttention(embed_dim, embed_dim, 32)
    query = torch.randn(batch_size, embed_dim)
    context, weights = additive_attn(query, x, x)
    print(f"Additive attention context shape: {context.shape}")
    print(f"Additive attention weights shape: {weights.shape}")
    
    # Test Multiplicative Attention
    print("\nTesting Multiplicative Attention...")
    mult_attn = MultiplicativeAttention(embed_dim, embed_dim, 'general')
    context, weights = mult_attn(query, x, x)
    print(f"Multiplicative attention context shape: {context.shape}")
    
    # Test with mask
    print("\nTesting with attention mask...")
    mask = torch.ones(batch_size, seq_len, seq_len)
    mask[:, :, seq_len//2:] = 0  # Mask out second half
    
    masked_output, masked_weights = self_attn(x, mask)
    print(f"Masked attention output shape: {masked_output.shape}")
    print(f"Attention weights sum (should be 1.0): {masked_weights[0, 0, 0, :seq_len//2].sum().item():.4f}")
    
    print("Attention mechanisms testing completed!")