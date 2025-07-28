#!/usr/bin/env python3
"""PyTorch Transformer Components - Attention, transformer blocks"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=== Transformer Components Overview ===")

print("Transformer components include:")
print("1. Self-attention mechanisms")
print("2. Multi-head attention")
print("3. Positional encoding")
print("4. Feed-forward networks")
print("5. Layer normalization")
print("6. Residual connections")

print("\n=== Basic Attention Mechanism ===")

class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights

# Test basic attention
attention = ScaledDotProductAttention()
seq_len, d_model = 10, 64
batch_size = 2

query = torch.randn(batch_size, seq_len, d_model)
key = torch.randn(batch_size, seq_len, d_model)
value = torch.randn(batch_size, seq_len, d_model)

output, weights = attention(query, key, value)
print(f"Attention output: {output.shape}")
print(f"Attention weights: {weights.shape}")

print("\n=== Multi-Head Attention ===")

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Adjust mask for multi-head
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        
        # Apply attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear projection
        output = self.w_o(attn_output)
        
        return output, attn_weights

# Test multi-head attention
mha = MultiHeadAttention(d_model=512, num_heads=8)
mha_input = torch.randn(4, 20, 512)

mha_output, mha_weights = mha(mha_input, mha_input, mha_input)
print(f"Multi-head attention output: {mha_output.shape}")
print(f"Multi-head attention weights: {mha_weights.shape}")

print("\n=== Positional Encoding ===")

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding"""
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(max_len, d_model))
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)

# Test positional encoding
pos_enc = PositionalEncoding(d_model=256, max_len=100)
learnable_pos_enc = LearnablePositionalEncoding(d_model=256, max_len=100)

pe_input = torch.randn(32, 50, 256)  # (batch, seq_len, d_model)
pe_output = pos_enc(pe_input.transpose(0, 1)).transpose(0, 1)  # PE expects (seq, batch, d_model)
learnable_pe_output = learnable_pos_enc(pe_input)

print(f"Positional encoding output: {pe_output.shape}")
print(f"Learnable PE output: {learnable_pe_output.shape}")

print("\n=== Feed Forward Network ===")

class PositionwiseFeedForward(nn.Module):
    """Position-wise feed forward network"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class GLUFeedForward(nn.Module):
    """GLU-based feed forward network"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff * 2)  # Double for GLU
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.w_1(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * torch.sigmoid(gate)  # GLU activation
        x = self.dropout(x)
        return self.w_2(x)

# Test feed forward networks
ff = PositionwiseFeedForward(d_model=512, d_ff=2048)
glu_ff = GLUFeedForward(d_model=512, d_ff=2048)

ff_input = torch.randn(8, 25, 512)
ff_output = ff(ff_input)
glu_ff_output = glu_ff(ff_input)

print(f"Feed forward output: {ff_output.shape}")
print(f"GLU feed forward output: {glu_ff_output.shape}")

print("\n=== Transformer Block ===")

class TransformerBlock(nn.Module):
    """Single transformer encoder block"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x, attn_weights

# Test transformer block
transformer_block = TransformerBlock(d_model=512, num_heads=8, d_ff=2048)
block_input = torch.randn(4, 30, 512)

block_output, block_weights = transformer_block(block_input)
print(f"Transformer block output: {block_output.shape}")
print(f"Transformer block attention weights: {block_weights.shape}")

print("\n=== Complete Transformer Encoder ===")

class TransformerEncoder(nn.Module):
    """Complete transformer encoder"""
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=5000, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # Pass through transformer layers
        attention_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
        
        x = self.norm(x)
        
        return x, attention_weights

# Test complete transformer encoder
vocab_size = 10000
encoder = TransformerEncoder(
    vocab_size=vocab_size, 
    d_model=512, 
    num_heads=8, 
    num_layers=6, 
    d_ff=2048
)

# Random token sequence
token_input = torch.randint(0, vocab_size, (4, 25))
encoder_output, encoder_weights = encoder(token_input)

print(f"Transformer encoder output: {encoder_output.shape}")
print(f"Number of attention weight tensors: {len(encoder_weights)}")
print(f"Each attention weight shape: {encoder_weights[0].shape}")

print("\n=== Transformer Decoder Block ===")

class TransformerDecoderBlock(nn.Module):
    """Single transformer decoder block"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        # Self-attention
        self_attn_output, self_attn_weights = self.self_attention(x, x, x, self_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))
        
        # Cross-attention
        cross_attn_output, cross_attn_weights = self.cross_attention(x, encoder_output, encoder_output, cross_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x, self_attn_weights, cross_attn_weights

print("\n=== Attention Variants ===")

class RelativePositionAttention(nn.Module):
    """Attention with relative position encoding"""
    def __init__(self, d_model, num_heads, max_relative_position=10):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Relative position embeddings
        self.max_relative_position = max_relative_position
        vocab_size = 2 * max_relative_position + 1
        self.relative_position_k = nn.Embedding(vocab_size, self.d_k)
        self.relative_position_v = nn.Embedding(vocab_size, self.d_k)
    
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
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
        
        # Relative position embeddings
        rel_pos_k = self.relative_position_k(relative_positions)
        rel_pos_v = self.relative_position_v(relative_positions)
        
        # Attention with relative positions (simplified)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(output)

print("\n=== Cross-Attention for Sequence-to-Sequence ===")

class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for encoder-decoder attention"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, decoder_input, encoder_output, mask=None):
        # Query from decoder, Key and Value from encoder
        attn_output, attn_weights = self.cross_attention(
            decoder_input, encoder_output, encoder_output, mask
        )
        
        # Residual connection and layer norm
        output = self.norm(decoder_input + self.dropout(attn_output))
        
        return output, attn_weights

# Test cross-attention
cross_attn = CrossAttentionLayer(d_model=256, num_heads=4)
decoder_seq = torch.randn(2, 15, 256)
encoder_seq = torch.randn(2, 20, 256)

cross_output, cross_weights = cross_attn(decoder_seq, encoder_seq)
print(f"Cross-attention output: {cross_output.shape}")
print(f"Cross-attention weights: {cross_weights.shape}")

print("\n=== Transformer Utilities ===")

def create_padding_mask(seq, pad_token=0):
    """Create padding mask for sequences"""
    return (seq != pad_token).unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(size):
    """Create look-ahead mask for decoder self-attention"""
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0

def create_combined_mask(seq, pad_token=0):
    """Create combined padding and look-ahead mask"""
    seq_len = seq.size(1)
    
    # Padding mask
    padding_mask = create_padding_mask(seq, pad_token)
    
    # Look-ahead mask
    look_ahead_mask = create_look_ahead_mask(seq_len)
    look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0)
    
    # Combine masks
    combined_mask = torch.minimum(padding_mask, look_ahead_mask)
    
    return combined_mask

# Test mask creation
sample_seq = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
padding_mask = create_padding_mask(sample_seq)
look_ahead_mask = create_look_ahead_mask(5)
combined_mask = create_combined_mask(sample_seq)

print(f"Sample sequence: {sample_seq}")
print(f"Padding mask shape: {padding_mask.shape}")
print(f"Look-ahead mask shape: {look_ahead_mask.shape}")
print(f"Combined mask shape: {combined_mask.shape}")

print("\n=== Transformer Best Practices ===")

print("Transformer Guidelines:")
print("1. Use layer normalization before attention and FFN")
print("2. Apply residual connections around each sub-layer")
print("3. Use dropout for regularization")
print("4. Scale embeddings by sqrt(d_model)")
print("5. Use appropriate positional encoding")
print("6. Consider relative position encoding for long sequences")
print("7. Use proper masking for different tasks")

print("\nArchitecture Variants:")
print("- Pre-norm vs Post-norm layer normalization")
print("- Different positional encoding schemes")
print("- GLU vs ReLU in feed-forward networks")
print("- Different attention variants (sparse, local, etc.)")
print("- Encoder-only vs Decoder-only vs Encoder-Decoder")

print("\nOptimization Tips:")
print("- Use mixed precision training")
print("- Gradient clipping for stability")
print("- Learning rate warmup")
print("- Label smoothing for classification")
print("- Proper weight initialization")
print("- Consider attention dropout")

print("\nCommon Applications:")
print("- Language modeling (decoder-only)")
print("- Machine translation (encoder-decoder)")
print("- Text classification (encoder-only)")
print("- Question answering (encoder-only)")
print("- Image processing (Vision Transformer)")
print("- Multimodal tasks (cross-attention)")

print("\n=== Transformer Components Complete ===") 