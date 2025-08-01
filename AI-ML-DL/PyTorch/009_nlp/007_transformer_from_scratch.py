import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple

# Positional Encoding
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create division term for sinusoidal pattern
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Compute scaled dot-product attention"""
        # Q, K, V: [batch_size, num_heads, seq_len, d_k]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear projection
        output = self.w_o(attention_output)
        
        return output, attention_weights

# Feed-Forward Network
class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output, attn_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights

# Transformer Decoder Layer
class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Self-attention on target sequence
        self_attn_output, self_attn_weights = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Cross-attention with encoder output
        cross_attn_output, cross_attn_weights = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x, self_attn_weights, cross_attn_weights

# Transformer Encoder
class TransformerEncoder(nn.Module):
    """Complete transformer encoder"""
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # x: [batch_size, seq_len]
        seq_len = x.size(1)
        
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)  # Scale embeddings
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Apply encoder layers
        attention_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
        
        return x, attention_weights

# Transformer Decoder
class TransformerDecoder(nn.Module):
    """Complete transformer decoder"""
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # x: [batch_size, seq_len]
        seq_len = x.size(1)
        
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Apply decoder layers
        self_attention_weights = []
        cross_attention_weights = []
        
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, encoder_output, src_mask, tgt_mask)
            self_attention_weights.append(self_attn)
            cross_attention_weights.append(cross_attn)
        
        return x, self_attention_weights, cross_attention_weights

# Complete Transformer Model
class Transformer(nn.Module):
    """Complete transformer model for sequence-to-sequence tasks"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, 
                 max_len=5000, dropout=0.1):
        super().__init__()
        
        self.encoder = TransformerEncoder(src_vocab_size, d_model, num_heads, 
                                        num_encoder_layers, d_ff, max_len, dropout)
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, num_heads,
                                        num_decoder_layers, d_ff, max_len, dropout)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, x, pad_token=0):
        """Create padding mask"""
        return (x != pad_token).unsqueeze(1).unsqueeze(2)
    
    def create_causal_mask(self, size):
        """Create causal mask for decoder"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 0
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Create masks if not provided
        if src_mask is None:
            src_mask = self.create_padding_mask(src)
        if tgt_mask is None:
            # Combine padding mask and causal mask for target
            tgt_padding_mask = self.create_padding_mask(tgt)
            tgt_causal_mask = self.create_causal_mask(tgt.size(1)).to(tgt.device)
            tgt_mask = tgt_padding_mask & tgt_causal_mask
        
        # Encoder
        encoder_output, encoder_attn = self.encoder(src, src_mask)
        
        # Decoder
        decoder_output, decoder_self_attn, decoder_cross_attn = self.decoder(
            tgt, encoder_output, src_mask, tgt_mask
        )
        
        # Output projection
        output = self.output_projection(decoder_output)
        
        return output, {
            'encoder_attention': encoder_attn,
            'decoder_self_attention': decoder_self_attn,
            'decoder_cross_attention': decoder_cross_attn
        }
    
    def generate(self, src, max_length=50, start_token=1, end_token=2, temperature=1.0):
        """Generate sequences using the transformer"""
        self.eval()
        device = src.device
        batch_size = src.size(0)
        
        with torch.no_grad():
            # Encode source
            src_mask = self.create_padding_mask(src)
            encoder_output, _ = self.encoder(src, src_mask)
            
            # Initialize target with start token
            tgt = torch.full((batch_size, 1), start_token, device=device)
            
            for _ in range(max_length):
                # Create target mask
                tgt_mask = self.create_causal_mask(tgt.size(1)).to(device)
                
                # Decode
                decoder_output, _, _ = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
                
                # Get next token logits
                next_token_logits = self.output_projection(decoder_output[:, -1, :])
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Sample next token
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1)
                
                # Append to target sequence
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # Check if all sequences have generated end token
                if (next_token == end_token).all():
                    break
            
            return tgt

# Training utilities
class TransformerTrainer:
    """Trainer for transformer models"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def train_epoch(self, dataloader, optimizer, criterion, clip_grad=1.0):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            src = batch['source'].to(self.device)
            tgt_input = batch['target'].to(self.device)
            tgt_output = batch['target_output'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output, attention_weights = self.model(src, tgt_input)
            
            # Reshape for loss calculation
            output = output.view(-1, output.size(-1))
            tgt_output = tgt_output.view(-1)
            
            # Calculate loss
            loss = criterion(output, tgt_output)
            loss.backward()
            
            # Gradient clipping
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
            
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                src = batch['source'].to(self.device)
                tgt_input = batch['target'].to(self.device)
                tgt_output = batch['target_output'].to(self.device)
                
                # Forward pass
                output, _ = self.model(src, tgt_input)
                
                # Reshape for loss calculation
                output = output.view(-1, output.size(-1))
                tgt_output = tgt_output.view(-1)
                
                loss = criterion(output, tgt_output)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, train_loader, val_loader, num_epochs=10, lr=1e-4, warmup_steps=4000):
        """Train with learning rate scheduling"""
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        
        # Learning rate scheduler (Transformer paper schedule)
        def lr_lambda(step):
            if step == 0:
                step = 1
            return min(step ** (-0.5), step * warmup_steps ** (-1.5))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss = self.evaluate(val_loader, criterion)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
            print('-' * 50)
            
            scheduler.step()

# Utility functions
def create_transformer_masks(src, tgt, pad_token=0):
    """Create all necessary masks for transformer"""
    # Source padding mask
    src_mask = (src != pad_token).unsqueeze(1).unsqueeze(2)
    
    # Target padding mask
    tgt_padding_mask = (tgt != pad_token).unsqueeze(1).unsqueeze(2)
    
    # Target causal mask
    tgt_len = tgt.size(1)
    tgt_causal_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1) == 0
    tgt_causal_mask = tgt_causal_mask.to(tgt.device)
    
    # Combine target masks
    tgt_mask = tgt_padding_mask & tgt_causal_mask
    
    return src_mask, tgt_mask

def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    print("Testing Transformer implementation...")
    
    # Model parameters
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 256
    num_heads = 8
    num_layers = 4
    d_ff = 1024
    max_len = 100
    
    # Create model
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=d_ff,
        max_len=max_len
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test input
    batch_size = 2
    src_len = 20
    tgt_len = 15
    
    src = torch.randint(1, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))
    
    # Test forward pass
    output, attention_weights = model(src, tgt)
    print(f"Output shape: {output.shape}")
    print(f"Encoder attention layers: {len(attention_weights['encoder_attention'])}")
    print(f"Decoder self-attention layers: {len(attention_weights['decoder_self_attention'])}")
    print(f"Decoder cross-attention layers: {len(attention_weights['decoder_cross_attention'])}")
    
    # Test generation
    generated = model.generate(src, max_length=20, start_token=1, end_token=2)
    print(f"Generated sequence shape: {generated.shape}")
    
    # Test individual components
    print("\nTesting individual components...")
    
    # Test MultiHeadAttention
    attention = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch_size, src_len, d_model)
    attn_out, attn_weights = attention(x, x, x)
    print(f"Attention output shape: {attn_out.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Test PositionalEncoding
    pos_enc = PositionalEncoding(d_model, max_len)
    x_with_pos = pos_enc(x.transpose(0, 1)).transpose(0, 1)
    print(f"Positional encoding output shape: {x_with_pos.shape}")
    
    print("Transformer implementation testing completed!")