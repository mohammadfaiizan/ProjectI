import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple

# Basic Word Embeddings
class WordEmbedding(nn.Module):
    """Standard word embedding layer with optional padding index"""
    
    def __init__(self, vocab_size, embed_dim, padding_idx=0, max_norm=None, 
                 norm_type=2.0, scale_grad_by_freq=False, sparse=False):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx,
            max_norm=max_norm, norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq, sparse=sparse
        )
        self.embed_dim = embed_dim
        
    def forward(self, x):
        # x: [batch_size, seq_len]
        return self.embedding(x)  # [batch_size, seq_len, embed_dim]

# Positional Encoding
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models"""
    
    def __init__(self, embed_dim, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create division term for sinusoidal pattern
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, embed_dim]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [seq_len, batch_size, embed_dim]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding"""
    
    def __init__(self, embed_dim, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Parameter(torch.randn(max_len, 1, embed_dim))
        
    def forward(self, x):
        # x: [seq_len, batch_size, embed_dim]
        x = x + self.pos_embedding[:x.size(0), :]
        return self.dropout(x)

# Token Type Embeddings (BERT-style)
class TokenTypeEmbedding(nn.Module):
    """Token type embeddings for distinguishing different segments"""
    
    def __init__(self, type_vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(type_vocab_size, embed_dim)
        
    def forward(self, token_type_ids):
        # token_type_ids: [batch_size, seq_len]
        return self.embedding(token_type_ids)  # [batch_size, seq_len, embed_dim]

# Combined Embeddings (BERT-style)
class BERTEmbeddings(nn.Module):
    """BERT-style embeddings combining word, position, and token type embeddings"""
    
    def __init__(self, vocab_size, embed_dim, max_position_embeddings=512,
                 type_vocab_size=2, layer_norm_eps=1e-12, dropout=0.1):
        super().__init__()
        
        self.word_embeddings = WordEmbedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, embed_dim)
        self.token_type_embeddings = TokenTypeEmbedding(type_vocab_size, embed_dim)
        
        self.layer_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        
        # Register position ids as buffer
        self.register_buffer("position_ids", 
                           torch.arange(max_position_embeddings).expand((1, -1)))
        
    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
            
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            
        # Get embeddings
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        
        # Combine embeddings
        embeddings = word_embeds + position_embeds + token_type_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

# Relative Position Encoding (Transformer-XL style)
class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for improved position modeling"""
    
    def __init__(self, embed_dim, max_len=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        
        # Create relative position embeddings
        inv_freq = 1 / (10000 ** (torch.arange(0.0, embed_dim, 2.0) / embed_dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, seq_len):
        # Create relative positions
        pos_seq = torch.arange(seq_len - 1, -seq_len, -1.0, 
                              device=self.inv_freq.device)
        
        # Compute sinusoidal encodings
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        
        return pos_emb

# Character-level Embeddings
class CharacterEmbedding(nn.Module):
    """Character-level embedding with CNN"""
    
    def __init__(self, char_vocab_size, char_embed_dim, word_embed_dim,
                 kernel_sizes=[3, 4, 5], num_filters=50, max_word_len=20):
        super().__init__()
        
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(char_embed_dim, num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(0.2)
        self.projection = nn.Linear(len(kernel_sizes) * num_filters, word_embed_dim)
        
    def forward(self, char_ids):
        # char_ids: [batch_size, num_words, max_word_len]
        batch_size, num_words, max_word_len = char_ids.size()
        
        # Reshape for processing
        char_ids = char_ids.view(-1, max_word_len)  # [batch_size * num_words, max_word_len]
        
        # Get character embeddings
        char_embeds = self.char_embedding(char_ids)  # [batch_size * num_words, max_word_len, char_embed_dim]
        char_embeds = char_embeds.transpose(1, 2)  # [batch_size * num_words, char_embed_dim, max_word_len]
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(char_embeds))  # [batch_size * num_words, num_filters, new_len]
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # [batch_size * num_words, num_filters]
            conv_outputs.append(pooled)
        
        # Concatenate features
        char_features = torch.cat(conv_outputs, dim=1)  # [batch_size * num_words, total_filters]
        char_features = self.dropout(char_features)
        
        # Project to word embedding dimension
        word_embeds = self.projection(char_features)  # [batch_size * num_words, word_embed_dim]
        
        # Reshape back
        word_embeds = word_embeds.view(batch_size, num_words, -1)
        
        return word_embeds

# Subword Embeddings (BPE-style)
class SubwordEmbedding(nn.Module):
    """Subword embedding layer for handling OOV words"""
    
    def __init__(self, subword_vocab_size, embed_dim, max_subwords_per_word=5):
        super().__init__()
        self.embedding = nn.Embedding(subword_vocab_size, embed_dim, padding_idx=0)
        self.max_subwords = max_subwords_per_word
        
    def forward(self, subword_ids, subword_lengths):
        # subword_ids: [batch_size, num_words, max_subwords_per_word]
        # subword_lengths: [batch_size, num_words]
        
        batch_size, num_words, max_subwords = subword_ids.size()
        
        # Get subword embeddings
        subword_embeds = self.embedding(subword_ids)  # [batch_size, num_words, max_subwords, embed_dim]
        
        # Create mask for valid subwords
        mask = torch.arange(max_subwords, device=subword_ids.device).unsqueeze(0).unsqueeze(0)
        mask = mask < subword_lengths.unsqueeze(2)  # [batch_size, num_words, max_subwords]
        mask = mask.unsqueeze(3).float()  # [batch_size, num_words, max_subwords, 1]
        
        # Apply mask and average
        masked_embeds = subword_embeds * mask
        word_embeds = masked_embeds.sum(dim=2) / (subword_lengths.unsqueeze(2).float() + 1e-10)
        
        return word_embeds

# Highway Network for Embeddings
class HighwayNetwork(nn.Module):
    """Highway network for embedding combination"""
    
    def __init__(self, embed_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.linear_layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)])
        self.gate_layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)])
        
    def forward(self, x):
        for linear, gate in zip(self.linear_layers, self.gate_layers):
            gate_values = torch.sigmoid(gate(x))
            nonlinear = F.relu(linear(x))
            x = gate_values * nonlinear + (1 - gate_values) * x
        return x

# Combined Embedding Layer
class CombinedEmbedding(nn.Module):
    """Combines multiple embedding types (word, char, subword)"""
    
    def __init__(self, vocab_size, embed_dim, char_vocab_size=None, 
                 subword_vocab_size=None, use_highway=True):
        super().__init__()
        
        # Word embeddings
        self.word_embedding = WordEmbedding(vocab_size, embed_dim)
        
        # Optional character embeddings
        if char_vocab_size:
            self.char_embedding = CharacterEmbedding(
                char_vocab_size, 50, embed_dim // 2
            )
        else:
            self.char_embedding = None
            
        # Optional subword embeddings
        if subword_vocab_size:
            self.subword_embedding = SubwordEmbedding(
                subword_vocab_size, embed_dim // 2
            )
        else:
            self.subword_embedding = None
            
        # Projection layer if combining embeddings
        total_dim = embed_dim
        if self.char_embedding:
            total_dim += embed_dim // 2
        if self.subword_embedding:
            total_dim += embed_dim // 2
            
        if total_dim > embed_dim:
            self.projection = nn.Linear(total_dim, embed_dim)
        else:
            self.projection = None
            
        # Highway network
        if use_highway:
            self.highway = HighwayNetwork(embed_dim)
        else:
            self.highway = None
            
    def forward(self, word_ids, char_ids=None, subword_ids=None, subword_lengths=None):
        # Get word embeddings
        embeddings = [self.word_embedding(word_ids)]
        
        # Add character embeddings if available
        if self.char_embedding and char_ids is not None:
            char_embeds = self.char_embedding(char_ids)
            embeddings.append(char_embeds)
            
        # Add subword embeddings if available
        if self.subword_embedding and subword_ids is not None:
            subword_embeds = self.subword_embedding(subword_ids, subword_lengths)
            embeddings.append(subword_embeds)
            
        # Combine embeddings
        combined = torch.cat(embeddings, dim=-1)
        
        # Project if necessary
        if self.projection:
            combined = self.projection(combined)
            
        # Apply highway network
        if self.highway:
            combined = self.highway(combined)
            
        return combined

# Embedding utilities
def create_padding_mask(seq_lens, max_len=None):
    """Create padding mask for sequences"""
    if max_len is None:
        max_len = seq_lens.max().item()
    
    batch_size = seq_lens.size(0)
    mask = torch.arange(max_len, device=seq_lens.device).expand(batch_size, max_len)
    mask = mask < seq_lens.unsqueeze(1)
    
    return mask

def get_sinusoidal_embeddings(seq_len, embed_dim, device='cpu'):
    """Generate sinusoidal positional embeddings"""
    embeddings = torch.zeros(seq_len, embed_dim, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    
    div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float, device=device) * 
                        (-math.log(10000.0) / embed_dim))
    
    embeddings[:, 0::2] = torch.sin(position * div_term)
    embeddings[:, 1::2] = torch.cos(position * div_term)
    
    return embeddings

if __name__ == "__main__":
    # Test basic word embedding
    vocab_size, embed_dim = 1000, 128
    word_emb = WordEmbedding(vocab_size, embed_dim)
    
    # Test input
    input_ids = torch.randint(0, vocab_size, (2, 10))  # batch_size=2, seq_len=10
    word_embeddings = word_emb(input_ids)
    print(f"Word embeddings shape: {word_embeddings.shape}")
    
    # Test positional encoding
    pos_enc = PositionalEncoding(embed_dim, max_len=100)
    seq_embeddings = word_embeddings.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
    pos_embeddings = pos_enc(seq_embeddings)
    print(f"With positional encoding shape: {pos_embeddings.shape}")
    
    # Test BERT embeddings
    bert_emb = BERTEmbeddings(vocab_size, embed_dim)
    bert_output = bert_emb(input_ids)
    print(f"BERT embeddings shape: {bert_output.shape}")
    
    # Test character embeddings
    char_vocab_size = 100
    char_emb = CharacterEmbedding(char_vocab_size, 25, embed_dim)
    char_ids = torch.randint(0, char_vocab_size, (2, 10, 15))  # batch_size=2, num_words=10, max_word_len=15
    char_output = char_emb(char_ids)
    print(f"Character embeddings shape: {char_output.shape}")
    
    # Test combined embeddings
    combined_emb = CombinedEmbedding(vocab_size, embed_dim, char_vocab_size=char_vocab_size)
    combined_output = combined_emb(input_ids, char_ids)
    print(f"Combined embeddings shape: {combined_output.shape}")
    
    # Test positional encoding utilities
    pe_matrix = get_sinusoidal_embeddings(20, embed_dim)
    print(f"Sinusoidal embeddings shape: {pe_matrix.shape}")
    
    print("All embedding tests completed!")