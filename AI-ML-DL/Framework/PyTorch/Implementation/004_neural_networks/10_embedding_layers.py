#!/usr/bin/env python3
"""PyTorch Embedding Layers - Embedding layers, pretrained embeddings"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=== Embedding Layers Overview ===")

print("Embedding layers provide:")
print("1. Dense vector representations for discrete tokens")
print("2. Learnable lookup tables")
print("3. Efficient sparse-to-dense conversion")
print("4. Pretrained embedding integration")
print("5. Various embedding techniques")

print("\n=== Basic Embedding Layer ===")

# Simple embedding layer
vocab_size = 10000
embed_size = 300
embedding = nn.Embedding(vocab_size, embed_size)

# Input: token indices
input_tokens = torch.tensor([[1, 5, 3, 8], [2, 7, 4, 0]])  # (batch_size, seq_len)
embedded = embedding(input_tokens)

print(f"Input tokens: {input_tokens.shape}")
print(f"Embedded output: {embedded.shape}")
print(f"Embedding weight shape: {embedding.weight.shape}")
print(f"Number of parameters: {embedding.weight.numel()}")

# Access embedding vectors
token_id = 5
token_embedding = embedding(torch.tensor([token_id]))
print(f"Token {token_id} embedding: {token_embedding.shape}")

print("\n=== Embedding Parameters and Options ===")

# Embedding with padding
embedding_pad = nn.Embedding(vocab_size, embed_size, padding_idx=0)
embedded_pad = embedding_pad(input_tokens)

print(f"Embedding with padding_idx=0:")
print(f"Token 0 embedding norm: {embedding_pad(torch.tensor([0])).norm():.6f}")
print(f"Token 1 embedding norm: {embedding_pad(torch.tensor([1])).norm():.6f}")

# Max norm constraint
embedding_max_norm = nn.Embedding(vocab_size, embed_size, max_norm=1.0)
embedded_max_norm = embedding_max_norm(input_tokens)

print(f"Max norm constraint:")
print(f"Embedding norms (should be ≤ 1.0): {embedding_max_norm.weight.norm(dim=1)[:5]}")

# Sparse embeddings
embedding_sparse = nn.Embedding(vocab_size, embed_size, sparse=True)
print(f"Sparse embedding gradient: {embedding_sparse.weight.grad_fn}")

# Scale gradient by frequency
embedding_scale = nn.Embedding(vocab_size, embed_size, scale_grad_by_freq=True)

print("\n=== Embedding Initialization ===")

def init_embeddings(embedding_layer, init_type='normal'):
    """Initialize embedding weights"""
    if init_type == 'normal':
        nn.init.normal_(embedding_layer.weight, mean=0, std=0.1)
    elif init_type == 'uniform':
        nn.init.uniform_(embedding_layer.weight, -0.1, 0.1)
    elif init_type == 'xavier':
        nn.init.xavier_uniform_(embedding_layer.weight)
    elif init_type == 'kaiming':
        nn.init.kaiming_uniform_(embedding_layer.weight)
    elif init_type == 'zeros':
        nn.init.zeros_(embedding_layer.weight)
        
    # Zero out padding token if specified
    if hasattr(embedding_layer, 'padding_idx') and embedding_layer.padding_idx is not None:
        embedding_layer.weight.data[embedding_layer.padding_idx].fill_(0)

# Test different initializations
init_types = ['normal', 'uniform', 'xavier', 'kaiming']
for init_type in init_types:
    embed_test = nn.Embedding(1000, 100, padding_idx=0)
    init_embeddings(embed_test, init_type)
    
    weight_stats = {
        'mean': embed_test.weight.mean().item(),
        'std': embed_test.weight.std().item(),
        'min': embed_test.weight.min().item(),
        'max': embed_test.weight.max().item()
    }
    print(f"{init_type:8} init: mean={weight_stats['mean']:.4f}, std={weight_stats['std']:.4f}")

print("\n=== Pretrained Embeddings ===")

def load_pretrained_embeddings(embedding_layer, pretrained_weights):
    """Load pretrained embedding weights"""
    with torch.no_grad():
        embedding_layer.weight.copy_(pretrained_weights)
    
    # Optionally freeze pretrained embeddings
    embedding_layer.weight.requires_grad = False

# Simulate pretrained embeddings (normally loaded from file)
pretrained_vocab_size = 5000
pretrained_embed_size = 200
pretrained_weights = torch.randn(pretrained_vocab_size, pretrained_embed_size)

# Create embedding layer and load pretrained weights
pretrained_embedding = nn.Embedding(pretrained_vocab_size, pretrained_embed_size)
load_pretrained_embeddings(pretrained_embedding, pretrained_weights)

print(f"Pretrained embedding frozen: {not pretrained_embedding.weight.requires_grad}")

# Fine-tuning pretrained embeddings
def setup_finetuning(embedding_layer, freeze_portion=0.8):
    """Setup embedding for fine-tuning"""
    vocab_size = embedding_layer.weight.size(0)
    freeze_count = int(vocab_size * freeze_portion)
    
    # Create a custom parameter for fine-tunable portion
    embedding_layer.weight.requires_grad = False
    
    # Create trainable embeddings for a subset
    trainable_embeddings = nn.Parameter(
        embedding_layer.weight[freeze_count:].clone()
    )
    
    return trainable_embeddings, freeze_count

print("\n=== Embedding Bags ===")

# EmbeddingBag for efficient averaging/summing
embedding_bag = nn.EmbeddingBag(vocab_size, embed_size, mode='mean')

# Input: flattened indices and offsets
input_indices = torch.tensor([1, 2, 3, 5, 6, 8, 9])  # Flattened sequence
offsets = torch.tensor([0, 3, 5])  # Start indices for each bag

bag_output = embedding_bag(input_indices, offsets)
print(f"EmbeddingBag input indices: {input_indices}")
print(f"EmbeddingBag offsets: {offsets}")
print(f"EmbeddingBag output: {bag_output.shape}")

# Different aggregation modes
modes = ['sum', 'mean', 'max']
for mode in modes:
    embed_bag = nn.EmbeddingBag(100, 50, mode=mode)
    test_indices = torch.tensor([1, 2, 3, 4, 5])
    test_offsets = torch.tensor([0, 2])
    output = embed_bag(test_indices, test_offsets)
    print(f"EmbeddingBag {mode:4} mode: {output.shape}")

print("\n=== Positional Embeddings ===")

class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embeddings"""
    def __init__(self, max_len, embed_size):
        super().__init__()
        self.positional_embedding = nn.Embedding(max_len, embed_size)
        self.max_len = max_len
    
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        pos_embeddings = self.positional_embedding(positions)
        return x + pos_embeddings

class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings (Transformer-style)"""
    def __init__(self, embed_size, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * 
                           (-math.log(10000.0) / embed_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Test positional embeddings
learned_pos = LearnedPositionalEmbedding(max_len=100, embed_size=256)
sinusoidal_pos = SinusoidalPositionalEmbedding(embed_size=256)

pos_input = torch.randn(4, 20, 256)
learned_output = learned_pos(pos_input)
sinusoidal_output = sinusoidal_pos(pos_input)

print(f"Learned positional embedding: {pos_input.shape} -> {learned_output.shape}")
print(f"Sinusoidal positional embedding: {pos_input.shape} -> {sinusoidal_output.shape}")

print("\n=== Word and Subword Embeddings ===")

class WordEmbedding(nn.Module):
    """Word-level embedding layer"""
    def __init__(self, vocab_size, embed_size, padding_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.embed_size = embed_size
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embed_size)

class SubwordEmbedding(nn.Module):
    """Subword (BPE/SentencePiece) embedding layer"""
    def __init__(self, vocab_size, embed_size, max_subwords=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.max_subwords = max_subwords
    
    def forward(self, subword_ids, lengths):
        # subword_ids: (batch_size, max_subwords)
        # lengths: (batch_size,) - actual number of subwords per word
        
        embedded = self.embedding(subword_ids)
        
        # Mask out padding subwords
        mask = torch.arange(self.max_subwords).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()
        
        # Average over subwords
        masked_embedded = embedded * mask
        word_embeddings = masked_embedded.sum(dim=1) / lengths.unsqueeze(1).float()
        
        return word_embeddings

# Test word and subword embeddings
word_embed = WordEmbedding(vocab_size=5000, embed_size=300)
subword_embed = SubwordEmbedding(vocab_size=8000, embed_size=300, max_subwords=5)

word_input = torch.randint(0, 5000, (8, 15))
word_output = word_embed(word_input)

subword_input = torch.randint(0, 8000, (8, 5))  # Max 5 subwords per word
subword_lengths = torch.randint(1, 6, (8,))  # Random lengths 1-5
subword_output = subword_embed(subword_input, subword_lengths)

print(f"Word embedding: {word_input.shape} -> {word_output.shape}")
print(f"Subword embedding: {subword_input.shape} -> {subword_output.shape}")

print("\n=== Character-level Embeddings ===")

class CharCNN(nn.Module):
    """Character-level CNN embeddings"""
    def __init__(self, char_vocab_size, char_embed_size, word_embed_size, 
                 kernel_sizes=[3, 4, 5], num_filters=100, max_word_len=20):
        super().__init__()
        
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_size)
        self.max_word_len = max_word_len
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(char_embed_size, num_filters, kernel_size)
            for kernel_size in kernel_sizes
        ])
        
        # Output projection
        total_filters = len(kernel_sizes) * num_filters
        self.projection = nn.Linear(total_filters, word_embed_size)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, char_ids):
        # char_ids: (batch_size, num_words, max_word_len)
        batch_size, num_words, max_word_len = char_ids.size()
        
        # Reshape for character embedding
        char_ids = char_ids.view(-1, max_word_len)
        char_embeds = self.char_embedding(char_ids)  # (batch*words, max_word_len, char_embed)
        
        # Transpose for convolution
        char_embeds = char_embeds.transpose(1, 2)  # (batch*words, char_embed, max_word_len)
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(char_embeds))  # (batch*words, num_filters, conv_len)
            pooled = F.max_pool1d(conv_out, conv_out.size(2))  # (batch*words, num_filters, 1)
            conv_outputs.append(pooled.squeeze(2))
        
        # Concatenate all conv outputs
        concatenated = torch.cat(conv_outputs, dim=1)  # (batch*words, total_filters)
        
        # Project to word embedding size
        word_embeds = self.projection(self.dropout(concatenated))
        
        # Reshape back to (batch_size, num_words, word_embed_size)
        word_embeds = word_embeds.view(batch_size, num_words, -1)
        
        return word_embeds

# Test character CNN
char_cnn = CharCNN(
    char_vocab_size=100, 
    char_embed_size=50, 
    word_embed_size=200,
    max_word_len=15
)

char_input = torch.randint(0, 100, (4, 10, 15))  # 4 sentences, 10 words, 15 chars max
char_output = char_cnn(char_input)

print(f"Character CNN: {char_input.shape} -> {char_output.shape}")

print("\n=== Multi-level Embeddings ===")

class HierarchicalEmbedding(nn.Module):
    """Hierarchical embeddings combining multiple levels"""
    def __init__(self, word_vocab_size, char_vocab_size, pos_vocab_size,
                 word_embed_size=300, char_embed_size=50, pos_embed_size=50,
                 final_embed_size=400):
        super().__init__()
        
        # Different embedding types
        self.word_embedding = nn.Embedding(word_vocab_size, word_embed_size)
        self.char_embedding = CharCNN(char_vocab_size, char_embed_size, word_embed_size//2)
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_embed_size)
        
        # Combination layer
        input_size = word_embed_size + word_embed_size//2 + pos_embed_size
        self.combiner = nn.Linear(input_size, final_embed_size)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, word_ids, char_ids, pos_ids):
        # Get embeddings
        word_embeds = self.word_embedding(word_ids)
        char_embeds = self.char_embedding(char_ids)
        pos_embeds = self.pos_embedding(pos_ids)
        
        # Concatenate all embeddings
        combined = torch.cat([word_embeds, char_embeds, pos_embeds], dim=-1)
        
        # Final projection
        final_embeds = self.combiner(self.dropout(combined))
        
        return final_embeds

print("\n=== Embedding Sharing and Tying ===")

class EmbeddingTiedModel(nn.Module):
    """Model with tied input/output embeddings"""
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        
        # Shared embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Hidden layers
        self.hidden = nn.Linear(embed_size, hidden_size)
        self.output_hidden = nn.Linear(hidden_size, embed_size)
        
        # Tied output layer (shares weights with embedding)
        self.output_projection = nn.Linear(embed_size, vocab_size, bias=False)
        self.output_projection.weight = self.embedding.weight  # Weight tying
    
    def forward(self, x):
        embedded = self.embedding(x)
        hidden = F.relu(self.hidden(embedded))
        output_hidden = self.output_hidden(hidden)
        logits = self.output_projection(output_hidden)
        return logits

tied_model = EmbeddingTiedModel(vocab_size=1000, embed_size=256, hidden_size=512)
tied_input = torch.randint(0, 1000, (4, 10))
tied_output = tied_model(tied_input)

print(f"Tied embedding model: {tied_input.shape} -> {tied_output.shape}")
print(f"Embedding and output weights are same: {tied_model.embedding.weight is tied_model.output_projection.weight}")

print("\n=== Embedding Analysis and Visualization ===")

def analyze_embeddings(embedding_layer, token_ids=None):
    """Analyze embedding properties"""
    weights = embedding_layer.weight.data
    
    stats = {
        'vocab_size': weights.size(0),
        'embed_size': weights.size(1),
        'mean': weights.mean().item(),
        'std': weights.std().item(),
        'norm_mean': weights.norm(dim=1).mean().item(),
        'norm_std': weights.norm(dim=1).std().item(),
    }
    
    if token_ids is not None:
        # Analyze specific tokens
        token_embeddings = embedding_layer(torch.tensor(token_ids))
        similarities = F.cosine_similarity(
            token_embeddings.unsqueeze(1), 
            token_embeddings.unsqueeze(0), 
            dim=2
        )
        stats['token_similarities'] = similarities
    
    return stats

# Analyze embedding layer
embedding_stats = analyze_embeddings(embedding, token_ids=[1, 2, 3, 4, 5])

print("Embedding analysis:")
print(f"  Vocab size: {embedding_stats['vocab_size']}")
print(f"  Embed size: {embedding_stats['embed_size']}")
print(f"  Weight mean: {embedding_stats['mean']:.4f}")
print(f"  Weight std: {embedding_stats['std']:.4f}")
print(f"  Norm mean: {embedding_stats['norm_mean']:.4f}")

print("\n=== Embedding Best Practices ===")

print("Embedding Layer Guidelines:")
print("1. Choose appropriate embedding dimensions (100-1000)")
print("2. Use padding_idx for padding tokens")
print("3. Initialize embeddings properly")
print("4. Consider pretrained embeddings for better initialization")
print("5. Use dropout on embeddings for regularization")
print("6. Scale embeddings in transformers (sqrt(d_model))")
print("7. Consider weight tying for language models")

print("\nDimension Selection:")
print("- Small vocabularies (< 10K): 100-300 dimensions")
print("- Medium vocabularies (10K-100K): 300-600 dimensions")
print("- Large vocabularies (> 100K): 500-1000 dimensions")
print("- Character-level: 50-100 dimensions")
print("- Subword-level: 200-500 dimensions")

print("\nOptimization Tips:")
print("- Use sparse gradients for large vocabularies")
print("- Consider hierarchical softmax for large outputs")
print("- Freeze pretrained embeddings initially")
print("- Use learning rate scheduling for embeddings")
print("- Monitor embedding norms during training")

print("\nCommon Applications:")
print("- Word embeddings for NLP tasks")
print("- User/item embeddings for recommendations")
print("- Category embeddings for tabular data")
print("- Positional embeddings for sequences")
print("- Entity embeddings for knowledge graphs")

print("\n=== Embedding Layers Complete ===") 