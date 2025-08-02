"""
ERA 5: TRANSFORMER-XL (2019)
============================

Year: 2019
Paper: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
Innovation: Segment-level recurrence and relative positional encoding
Previous Limitation: Original Transformer limited to fixed-length contexts
Performance Gain: 80% longer context, better long-term dependencies
Impact: Enabled modeling of much longer sequences, influenced GPT and other LLMs

This file implements Transformer-XL that introduced segment-level recurrence
mechanism and relative positional encoding to overcome the context length
limitations of the original Transformer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
import math
from collections import defaultdict, Counter
from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
import random
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# HISTORICAL CONTEXT & MOTIVATION
# ============================================================================

YEAR = "2019"
INNOVATION = "Transformer-XL: Segment-level Recurrence + Relative Positional Encoding"
PREVIOUS_LIMITATION = "Transformer limited to fixed context length, poor long-term dependencies"
IMPACT = "80% longer effective context, foundation for long-context language models"

print(f"=== {INNOVATION} ({YEAR}) ===")
print(f"Innovation: {INNOVATION}")
print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
print(f"Impact: {IMPACT}")
print("="*60)

# ============================================================================
# STANDARDIZED DATASET LOADING
# ============================================================================

def load_wikitext2_dataset():
    """Load WikiText-2 dataset with consistent preprocessing"""
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset('wikitext', 'wikitext-2-v1')
    
    def preprocess_text(texts):
        sentences = []
        for text in texts:
            if text.strip():
                text = text.lower().replace('\n', ' ')
                text_sentences = text.split('.')
                for sentence in text_sentences:
                    tokens = word_tokenize(sentence.strip())
                    if 8 <= len(tokens) <= 40:  # Longer sequences for Transformer-XL
                        sentences.append(tokens)
        return sentences
    
    train_sentences = preprocess_text(dataset['train']['text'])
    val_sentences = preprocess_text(dataset['validation']['text'])
    test_sentences = preprocess_text(dataset['test']['text'])
    
    print(f"Train sentences: {len(train_sentences):,}")
    print(f"Validation sentences: {len(val_sentences):,}")
    print(f"Test sentences: {len(test_sentences):,}")
    
    return train_sentences, val_sentences, test_sentences

def build_vocabulary(sentences, vocab_size=8000):
    """Build vocabulary with special tokens"""
    word_counts = Counter()
    for sentence in sentences:
        word_counts.update(sentence)
    
    most_common = word_counts.most_common(vocab_size - 4)
    
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1, 
        '<SOS>': 2,
        '<EOS>': 3
    }
    
    for word, _ in most_common:
        vocab[word] = len(vocab)
    
    idx_to_word = {idx: word for word, idx in vocab.items()}
    
    print(f"Vocabulary size: {len(vocab):,}")
    return vocab, idx_to_word

# ============================================================================
# DATASET CLASS FOR TRANSFORMER-XL
# ============================================================================

class TransformerXLDataset(Dataset):
    """
    Dataset for Transformer-XL that creates longer sequences 
    and supports segment-level processing
    """
    
    def __init__(self, sentences, vocab, segment_length=32, task='language_modeling'):
        self.vocab = vocab
        self.segment_length = segment_length
        self.task = task
        self.sequences = []
        
        # Concatenate sentences to create longer sequences
        all_tokens = []
        for sentence in sentences:
            all_tokens.extend(sentence)
            all_tokens.append('<EOS>')  # Sentence boundary
        
        # Create overlapping segments for language modeling
        if task == 'language_modeling':
            for i in range(0, len(all_tokens) - segment_length - 1, segment_length // 2):
                segment = all_tokens[i:i + segment_length]
                target = all_tokens[i + 1:i + segment_length + 1]
                
                if len(segment) == segment_length and len(target) == segment_length:
                    self.sequences.append((segment, target))
        
        print(f"Created {len(self.sequences)} segments of length {segment_length}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.sequences[idx]
        
        # Convert to indices
        input_indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in input_seq]
        target_indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in target_seq]
        
        return torch.tensor(input_indices, dtype=torch.long), torch.tensor(target_indices, dtype=torch.long)

def collate_xl_fn(batch):
    """Collate function for Transformer-XL"""
    input_sequences, target_sequences = zip(*batch)
    
    # All sequences should be same length (segment_length)
    inputs = torch.stack(input_sequences)
    targets = torch.stack(target_sequences)
    
    return inputs, targets

# ============================================================================
# RELATIVE POSITIONAL ENCODING
# ============================================================================

class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding for Transformer-XL
    Uses learnable relative position embeddings instead of absolute positions
    """
    
    def __init__(self, d_model, max_relative_position=512):
        super(RelativePositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # Learnable relative position embeddings
        vocab_size = 2 * max_relative_position + 1
        self.relative_position_embeddings = nn.Embedding(vocab_size, d_model)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.relative_position_embeddings.weight)
    
    def forward(self, length):
        """
        Generate relative position embeddings for a sequence of given length
        
        Args:
            length: int, sequence length
            
        Returns:
            relative_embeddings: (length, length, d_model)
        """
        # Create relative position matrix
        range_vec = torch.arange(length)
        distance_mat = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)
        
        # Clamp to maximum relative position
        distance_mat_clipped = torch.clamp(
            distance_mat, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        
        # Shift to make all values positive
        final_mat = distance_mat_clipped + self.max_relative_position
        
        # Get embeddings
        relative_embeddings = self.relative_position_embeddings(final_mat)
        
        return relative_embeddings

# ============================================================================
# RELATIVE MULTI-HEAD ATTENTION
# ============================================================================

class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding
    Core innovation of Transformer-XL
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        # Relative position projections
        self.w_r = nn.Linear(d_model, d_model, bias=False)
        
        # Learnable biases for relative attention
        self.u = nn.Parameter(torch.randn(num_heads, self.d_k))  # Content bias
        self.v = nn.Parameter(torch.randn(num_heads, self.d_k))  # Position bias
        
        self.dropout = nn.Dropout(dropout)
        
    def relative_attention_score(self, query, key_content, key_position, mask=None):
        """
        Compute relative attention scores using both content and position
        
        Args:
            query: (batch_size, num_heads, q_len, d_k)
            key_content: (batch_size, num_heads, k_len, d_k) 
            key_position: (k_len, q_len, d_k) relative position embeddings
            mask: optional attention mask
            
        Returns:
            attention_scores: (batch_size, num_heads, q_len, k_len)
        """
        batch_size, num_heads, q_len, d_k = query.size()
        k_len = key_content.size(2)
        
        # Content-based attention
        content_score = torch.matmul(query + self.u.unsqueeze(0).unsqueeze(2), 
                                   key_content.transpose(-2, -1))
        
        # Position-based attention  
        key_position_heads = key_position.view(k_len, q_len, num_heads, d_k).permute(2, 1, 0, 3)
        position_score = torch.matmul(query + self.v.unsqueeze(0).unsqueeze(2), 
                                    key_position_heads.transpose(-2, -1))
        
        # Combine scores
        attention_scores = (content_score + position_score) / math.sqrt(d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        return attention_scores
    
    def forward(self, query, key, value, relative_embeddings=None, 
                memory=None, mask=None):
        """
        Forward pass with relative attention and optional memory
        
        Args:
            query: (batch_size, q_len, d_model)
            key: (batch_size, k_len, d_model)
            value: (batch_size, k_len, d_model)
            relative_embeddings: (k_len, q_len, d_model) relative position embeddings
            memory: (batch_size, mem_len, d_model) optional memory from previous segment
            mask: attention mask
            
        Returns:
            output: (batch_size, q_len, d_model)
            attention_weights: (batch_size, num_heads, q_len, k_len)
        """
        batch_size, q_len, d_model = query.size()
        k_len = key.size(1)
        
        # Concatenate memory if provided
        if memory is not None:
            mem_len = memory.size(1)
            key = torch.cat([memory, key], dim=1)
            value = torch.cat([memory, value], dim=1)
            k_len += mem_len
        
        # Linear projections and reshape for multi-head
        Q = self.w_q(query).view(batch_size, q_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, k_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, k_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Process relative embeddings
        if relative_embeddings is not None:
            R = self.w_r(relative_embeddings)  # (k_len, q_len, d_model)
        else:
            R = None
        
        # Compute attention scores with relative positions
        if R is not None:
            attention_scores = self.relative_attention_score(Q, K, R, mask)
        else:
            # Fallback to standard attention
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, q_len, d_model)
        output = self.w_o(output)
        
        return output, attention_weights

# ============================================================================
# TRANSFORMER-XL LAYER
# ============================================================================

class TransformerXLLayer(nn.Module):
    """
    Single layer of Transformer-XL with relative attention and memory
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerXLLayer, self).__init__()
        
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, relative_embeddings=None, memory=None, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            relative_embeddings: relative position embeddings
            memory: memory from previous segment
            mask: attention mask
            
        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: attention weights
        """
        # Self-attention with memory and relative positions
        attn_output, attention_weights = self.attention(
            x, x, x, relative_embeddings, memory, mask
        )
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        output = self.norm2(x + self.dropout(ff_output))
        
        return output, attention_weights

# ============================================================================
# TRANSFORMER-XL MODEL
# ============================================================================

class TransformerXL(nn.Module):
    """
    Complete Transformer-XL model with segment-level recurrence
    """
    
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6,
                 d_ff=2048, segment_length=32, memory_length=32, dropout=0.1):
        super(TransformerXL, self).__init__()
        
        self.d_model = d_model
        self.segment_length = segment_length
        self.memory_length = memory_length
        self.num_layers = num_layers
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Relative positional encoding
        self.relative_encoding = RelativePositionalEncoding(d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerXLLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize memories
        self.register_buffer('memories', None)
        
    def init_memory(self, batch_size, device):
        """Initialize empty memory for each layer"""
        memories = []
        for _ in range(self.num_layers):
            memory = torch.zeros(batch_size, self.memory_length, self.d_model, 
                               device=device, dtype=torch.float)
            memories.append(memory)
        return memories
    
    def update_memory(self, hidden_states, memories):
        """Update memory with current hidden states"""
        if memories is None:
            return hidden_states
        
        new_memories = []
        for layer_idx, (hidden, memory) in enumerate(zip(hidden_states, memories)):
            # Concatenate new hidden state with memory
            extended = torch.cat([memory, hidden], dim=1)
            # Keep only the most recent memory_length tokens
            new_memory = extended[:, -self.memory_length:].detach()
            new_memories.append(new_memory)
        
        return new_memories
    
    def create_causal_mask(self, q_len, k_len, device):
        """Create causal mask for autoregressive modeling"""
        mask = torch.tril(torch.ones(q_len, k_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, q_len, k_len)
    
    def forward(self, input_ids, memories=None, return_memories=False):
        """
        Forward pass with segment-level recurrence
        
        Args:
            input_ids: (batch_size, seq_len) input token ids
            memories: list of memory tensors from previous segments
            return_memories: whether to return updated memories
            
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            new_memories: updated memories (if return_memories=True)
            attention_weights: list of attention weights from each layer
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Initialize memories if not provided
        if memories is None:
            memories = self.init_memory(batch_size, device)
        
        # Token embeddings
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.dropout(x)
        
        # Get relative position embeddings
        total_len = seq_len + (memories[0].size(1) if memories else 0)
        relative_embeddings = self.relative_encoding(total_len)
        
        # Create causal mask
        mask = self.create_causal_mask(seq_len, total_len, device)
        
        # Apply Transformer-XL layers
        hidden_states = []
        all_attention_weights = []
        
        for layer_idx, layer in enumerate(self.layers):
            memory = memories[layer_idx] if memories else None
            x, attention_weights = layer(x, relative_embeddings, memory, mask)
            hidden_states.append(x)
            all_attention_weights.append(attention_weights)
        
        # Output projection
        logits = self.output_projection(x)
        
        # Update memories
        new_memories = self.update_memory(hidden_states, memories)
        
        if return_memories:
            return logits, new_memories, all_attention_weights
        else:
            return logits, all_attention_weights
    
    def generate(self, input_ids, vocab, idx_to_word, max_length=100, 
                 temperature=1.0, memories=None):
        """
        Generate sequences using Transformer-XL with memory
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # Initialize memories if not provided
        if memories is None:
            memories = self.init_memory(batch_size, device)
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits, new_memories, _ = self.forward(
                    generated[:, -self.segment_length:], memories, return_memories=True
                )
                
                # Get next token
                next_token_logits = logits[:, -1, :] / temperature
                next_tokens = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1)
                
                # Check for EOS
                if next_tokens[0].item() == vocab['<EOS>']:
                    break
                
                # Append to generated sequence
                generated = torch.cat([generated, next_tokens], dim=1)
                
                # Update memories
                memories = new_memories
            
            # Convert to words
            generated_words = []
            for token_idx in generated[0].tolist():
                if token_idx in idx_to_word and token_idx not in [vocab['<PAD>'], vocab['<SOS>']]:
                    if token_idx == vocab['<EOS>']:
                        break
                    generated_words.append(idx_to_word[token_idx])
        
        return generated_words

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_transformer_xl(model, train_loader, val_loader, epochs=10, learning_rate=0.00025):
    """Train Transformer-XL model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    train_losses = []
    val_perplexities = []
    
    print(f"Training Transformer-XL on device: {device}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        # Initialize memories for each epoch
        memories = None
        
        for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with memory
            logits, new_memories, _ = model(input_ids, memories, return_memories=True)
            
            # Compute loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update memories (detach to prevent backprop through time)
            memories = [mem.detach() for mem in new_memories] if new_memories else None
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_perplexity = evaluate_transformer_xl(model, val_loader, criterion, device)
        val_perplexities.append(val_perplexity)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Perplexity: {val_perplexity:.2f}")
    
    return train_losses, val_perplexities

def evaluate_transformer_xl(model, data_loader, criterion, device):
    """Evaluate Transformer-XL model perplexity"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        memories = None
        
        for input_ids, target_ids in data_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            logits, new_memories, _ = model(input_ids, memories, return_memories=True)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
            
            total_loss += loss.item() * target_ids.numel()
            total_tokens += target_ids.numel()
            
            # Update memories
            memories = [mem.detach() for mem in new_memories] if new_memories else None
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    return perplexity.item()

# ============================================================================
# PERFORMANCE TRACKING
# ============================================================================

def track_computational_metrics(model_name, train_function, *args):
    """Track computational metrics during training"""
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024
    
    start_time = time.time()
    result = train_function(*args)
    training_time = time.time() - start_time
    
    memory_after = process.memory_info().rss / 1024 / 1024
    memory_used = memory_after - memory_before
    
    return {
        'model_name': model_name,
        'training_time_minutes': training_time / 60,
        'memory_usage_mb': memory_used,
        'result': result
    }

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================================================
# MEMORY VISUALIZATION
# ============================================================================

def visualize_memory_usage(model, sample_input, vocab, idx_to_word):
    """Visualize how memory is used across segments"""
    device = next(model.parameters()).device
    model.eval()
    
    segment_length = model.segment_length
    memory_length = model.memory_length
    
    # Process multiple segments
    memories = None
    attention_patterns = []
    
    with torch.no_grad():
        for i in range(0, len(sample_input) - segment_length, segment_length):
            segment = sample_input[i:i + segment_length]
            segment_tensor = torch.tensor([segment], device=device)
            
            logits, new_memories, attention_weights = model(
                segment_tensor, memories, return_memories=True
            )
            
            # Store attention pattern
            if attention_weights:
                attention_patterns.append(attention_weights[0][0, 0].cpu().numpy())
            
            memories = new_memories
    
    # Visualize attention across segments
    if attention_patterns:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, attn in enumerate(attention_patterns[:4]):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            sns.heatmap(attn, ax=ax, cmap='Blues', cbar=True)
            ax.set_title(f'Segment {i+1} Attention Pattern')
            ax.set_xlabel('Key Positions')
            ax.set_ylabel('Query Positions')
        
        plt.tight_layout()
        plt.savefig('AI-ML-DL/Models/NLP/014_transformer_xl_memory_visualization.png', 
                   dpi=300, bbox_inches='tight')
        print("Memory visualization saved: 014_transformer_xl_memory_visualization.png")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"=== Transformer-XL ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load dataset
    train_sentences, val_sentences, test_sentences = load_wikitext2_dataset()
    
    # Use subset for demonstration
    train_subset = train_sentences[:800]
    val_subset = val_sentences[:160]
    test_subset = test_sentences[:160]
    
    # Build vocabulary
    vocab, idx_to_word = build_vocabulary(train_subset, vocab_size=4000)
    
    results = []
    training_histories = {}
    
    # Test Transformer-XL
    print("\n" + "="*50)
    print("Training Transformer-XL")
    
    # Create dataset with longer segments
    segment_length = 32
    train_dataset = TransformerXLDataset(train_subset, vocab, segment_length=segment_length)
    val_dataset = TransformerXLDataset(val_subset, vocab, segment_length=segment_length)
    
    # Create data loaders
    batch_size = 8  # Smaller batch size due to memory requirements
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            collate_fn=collate_xl_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          collate_fn=collate_xl_fn)
    
    # Initialize Transformer-XL
    transformer_xl = TransformerXL(
        vocab_size=len(vocab),
        d_model=256,           # Reduced for demo
        num_heads=8,
        num_layers=4,          # Reduced for demo
        d_ff=512,              # Reduced for demo
        segment_length=segment_length,
        memory_length=32,
        dropout=0.1
    )
    
    print(f"Transformer-XL parameters: {count_parameters(transformer_xl):,}")
    
    # Train model
    metrics = track_computational_metrics(
        'Transformer-XL',
        train_transformer_xl,
        transformer_xl, train_loader, val_loader, 6, 0.00025
    )
    
    train_losses, val_perplexities = metrics['result']
    training_histories['Transformer-XL'] = (train_losses, val_perplexities)
    
    result = {
        'model': 'Transformer-XL',
        'year': '2019',
        'final_perplexity': val_perplexities[-1] if val_perplexities else 0,
        'parameters': count_parameters(transformer_xl),
        'training_time': metrics['training_time_minutes'],
        'memory_usage': metrics['memory_usage_mb'],
        'innovation': 'Segment-level recurrence + relative positional encoding'
    }
    results.append(result)
    
    # Demonstrate generation with memory
    print("\nGENERATION WITH MEMORY:")
    print("="*30)
    
    transformer_xl.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformer_xl.to(device)
    
    # Create seed sequence
    seed_words = ['the', 'quick', 'brown', 'fox']
    seed_indices = [vocab.get(word, vocab['<UNK>']) for word in seed_words]
    seed_tensor = torch.tensor([seed_indices], device=device)
    
    # Generate with memory
    generated = transformer_xl.generate(
        seed_tensor, vocab, idx_to_word, max_length=20, temperature=0.8
    )
    
    print(f"Seed: {' '.join(seed_words)}")
    print(f"Generated: {' '.join(generated)}")
    
    # Visualize memory usage
    if len(test_subset) > 0:
        print("\nAnalyzing Memory Usage...")
        # Create longer sample for memory analysis
        sample_tokens = []
        for sentence in test_subset[:3]:
            sample_tokens.extend(sentence)
        
        if len(sample_tokens) > segment_length * 2:
            sample_indices = [vocab.get(token, vocab['<UNK>']) for token in sample_tokens]
            visualize_memory_usage(transformer_xl, sample_indices, vocab, idx_to_word)
    
    # Create comprehensive visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training curves
    ax = axes[0, 0]
    if train_losses:
        ax.plot(train_losses, label='Training Loss', linewidth=2, color='#E74C3C')
    ax.set_title('Transformer-XL Training Loss', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation perplexity
    ax = axes[0, 1]
    if val_perplexities:
        ax.plot(val_perplexities, label='Validation Perplexity', linewidth=2, color='#3498DB')
    ax.set_title('Transformer-XL Validation Perplexity', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Perplexity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Architecture improvements
    ax = axes[1, 0]
    improvements = ['Segment\nRecurrence', 'Relative\nPositioning', 'Longer\nContext', 'Memory\nMechanism']
    impact_scores = [9, 10, 8, 7]
    bars = ax.bar(improvements, impact_scores, 
                  color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax.set_title('Transformer-XL Innovations', fontsize=14)
    ax.set_ylabel('Impact Score')
    
    # Context length comparison
    ax = axes[1, 1]
    models = ['Original\nTransformer', 'Transformer-XL']
    context_lengths = [512, 900]  # Approximate effective context lengths
    bars = ax.bar(models, context_lengths, color=['#95A5A6', '#E67E22'])
    ax.set_title('Effective Context Length', fontsize=14)
    ax.set_ylabel('Context Length (tokens)')
    
    # Add value labels on bars
    for bar, value in zip(bars, context_lengths):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/NLP/014_transformer_xl_results.png', dpi=300, bbox_inches='tight')
    print("\nComprehensive visualization saved: 014_transformer_xl_results.png")
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("🧠 TRANSFORMER-XL BREAKTHROUGH SUMMARY 🧠")
    print("="*70)
    
    for result in results:
        print(f"\n{result['model']} ({result['year']}):")
        print(f"  📊 Final Perplexity: {result['final_perplexity']:.2f}")
        print(f"  🔢 Parameters: {result['parameters']:,}")
        print(f"  ⏱️  Training Time: {result['training_time']:.2f} minutes")
        print(f"  💾 Memory Usage: {result['memory_usage']:.2f} MB")
        print(f"  💡 Innovation: {result['innovation']}")
    
    print("\n🚀 KEY INNOVATIONS:")
    print("="*50)
    print("1. 🔄 SEGMENT-LEVEL RECURRENCE:")
    print("   • Reuses hidden states from previous segments")
    print("   • Enables modeling of longer dependencies")
    print("   • Maintains gradient flow across segments")
    
    print("\n2. 📏 RELATIVE POSITIONAL ENCODING:")
    print("   • Replaces absolute positions with relative distances")
    print("   • Enables better generalization to longer sequences")
    print("   • Learnable position-dependent biases (u, v)")
    
    print("\n3. 🧮 MEMORY MECHANISM:")
    print("   • Caches representations from previous segments")
    print("   • Extends effective context length significantly")
    print("   • Enables truly long-range modeling")
    
    print("\n4. ⚡ IMPROVED ATTENTION:")
    print("   • Content-based + position-based attention")
    print("   • Better handling of relative positions")
    print("   • More stable training on long sequences")
    
    print("\n📈 PERFORMANCE IMPROVEMENTS:")
    print("="*50)
    print("• 📏 Context Length: 80% longer effective context")
    print("• 🎯 Perplexity: 18% improvement on enwik8")
    print("• 🚀 Speed: Better parallelization than RNNs")
    print("• 📚 Language Modeling: State-of-the-art results")
    print("• 🧠 Long-term Dependencies: Much better modeling")
    
    print("\n🔬 TECHNICAL DETAILS:")
    print("="*50)
    print("• Relative attention: Attention(Q,K,V,R) with position bias")
    print("• Memory update: Hidden states cached across segments")
    print("• Position encoding: R_ij represents relative distance i-j")
    print("• Gradient flow: Truncated backprop with memory reuse")
    print("• Training: Segment-by-segment with memory persistence")
    
    print("\n💡 WHY IT WORKED:")
    print("="*50)
    print("1. 🔗 MEMORY REUSE: Previous computations not wasted")
    print("2. 📏 RELATIVE POSITIONS: More natural position representation")
    print("3. 🎯 FOCUSED ATTENTION: Better attention distribution")
    print("4. 📈 SCALABILITY: Enables much longer sequences")
    print("5. 🧮 EFFICIENCY: Better computational utilization")
    
    print("\n🌟 HISTORICAL IMPACT:")
    print("="*50)
    print("• 📖 First major improvement to original Transformer")
    print("• 🔬 Introduced relative positional encoding concept")
    print("• 🧠 Enabled long-context language modeling")
    print("• 🚀 Influenced GPT-2, GPT-3 design decisions")
    print("• 📏 Showed importance of context length scaling")
    print("• 🔄 Memory mechanism inspired later architectures")
    
    print("\n⚖️ ORIGINAL TRANSFORMER vs TRANSFORMER-XL:")
    print("="*50)
    print("ORIGINAL TRANSFORMER:")
    print("  ❌ Fixed context window (e.g., 512 tokens)")
    print("  ❌ Absolute positional encoding")
    print("  ❌ No memory between segments")
    print("  ❌ Context fragmentation problem")
    
    print("\nTRANSFORMER-XL:")
    print("  ✅ Variable context length with memory")
    print("  ✅ Relative positional encoding")
    print("  ✅ Segment-level recurrence")
    print("  ✅ Continuous context modeling")
    
    print("\n🎓 KEY LESSONS:")
    print("="*50)
    print("• Memory mechanisms can dramatically improve sequence modeling")
    print("• Relative positions are more natural than absolute positions")
    print("• Context length is crucial for language understanding")
    print("• Recurrence can be beneficial even in attention-based models")
    print("• Simple architectural changes can have profound impact")
    
    print(f"\n{'='*70}")
    print("🎯 TRANSFORMER-XL: BREAKING THE CONTEXT BARRIER 🎯")
    print(f"{'='*70}")
    
    return results

if __name__ == "__main__":
    results = main()