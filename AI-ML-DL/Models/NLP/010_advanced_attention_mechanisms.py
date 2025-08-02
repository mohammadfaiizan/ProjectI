"""
ERA 4: ADVANCED ATTENTION MECHANISMS (2016-2017)
================================================

Year: 2016-2017
Innovation: Self-attention, Multi-head attention, and Scaled Dot-Product attention
Previous Limitation: Attention only between encoder-decoder, single attention head
Performance Gain: Self-attention within sequences, multiple attention heads
Impact: Direct foundation for Transformer architecture, parallelizable attention

This file demonstrates the evolution of attention mechanisms that directly led to
the Transformer revolution, introducing self-attention and multi-head concepts.
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

YEAR = "2016-2017"
INNOVATION = "Self-Attention and Multi-Head Attention"
PREVIOUS_LIMITATION = "Attention only encoder-decoder, single attention representation"
IMPACT = "Self-attention within sequences, multiple attention types, Transformer foundation"

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
                    if 8 <= len(tokens) <= 30:  # Longer sequences for self-attention
                        sentences.append(tokens)
        return sentences
    
    train_sentences = preprocess_text(dataset['train']['text'])
    val_sentences = preprocess_text(dataset['validation']['text'])
    test_sentences = preprocess_text(dataset['test']['text'])
    
    print(f"Train sentences: {len(train_sentences):,}")
    print(f"Validation sentences: {len(val_sentences):,}")
    print(f"Test sentences: {len(test_sentences):,}")
    
    return train_sentences, val_sentences, test_sentences

def build_vocabulary(sentences, vocab_size=5000):
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
# DATASET CLASS FOR SELF-ATTENTION EXPERIMENTS
# ============================================================================

class SelfAttentionDataset(Dataset):
    """Dataset for self-attention and advanced attention experiments"""
    
    def __init__(self, sentences, vocab, task='language_modeling', max_length=25):
        self.vocab = vocab
        self.task = task
        self.max_length = max_length
        self.pairs = []
        
        if task == 'language_modeling':
            # Next word prediction task
            for sentence in sentences:
                if len(sentence) <= max_length:
                    input_seq = sentence[:-1] if len(sentence) > 1 else sentence
                    target_seq = sentence[1:] if len(sentence) > 1 else sentence
                    self.pairs.append((input_seq, target_seq))
        
        elif task == 'masked_lm':
            # Masked language modeling (BERT-style)
            for sentence in sentences:
                if len(sentence) <= max_length:
                    # Create masked version
                    masked_sentence = sentence[:]
                    targets = []
                    
                    # Mask 15% of tokens
                    num_to_mask = max(1, int(0.15 * len(sentence)))
                    mask_positions = random.sample(range(len(sentence)), num_to_mask)
                    
                    for pos in mask_positions:
                        targets.append((pos, sentence[pos]))
                        masked_sentence[pos] = '<UNK>'  # Use UNK as mask token
                    
                    self.pairs.append((masked_sentence, targets))
        
        elif task == 'sequence_classification':
            # Sentence classification task
            for sentence in sentences:
                if len(sentence) <= max_length:
                    # Simple heuristic: classify by length
                    label = 1 if len(sentence) > 15 else 0
                    self.pairs.append((sentence, label))
        
        print(f"Created {len(self.pairs)} {task} pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        if self.task == 'language_modeling':
            input_seq, target_seq = self.pairs[idx]
            input_indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in input_seq]
            target_indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in target_seq]
            return torch.tensor(input_indices, dtype=torch.long), torch.tensor(target_indices, dtype=torch.long)
        
        elif self.task == 'sequence_classification':
            input_seq, label = self.pairs[idx]
            input_indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in input_seq]
            return torch.tensor(input_indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)
        
        else:  # masked_lm
            masked_seq, targets = self.pairs[idx]
            input_indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in masked_seq]
            return torch.tensor(input_indices, dtype=torch.long), targets

def collate_fn_advanced(batch):
    """Enhanced collate function for advanced attention tasks"""
    if len(batch[0]) == 2 and isinstance(batch[0][1], torch.Tensor):
        # Standard sequence tasks
        input_sequences, targets = zip(*batch)
        
        if targets[0].dim() == 0:  # Classification
            input_lengths = [len(seq) for seq in input_sequences]
            max_len = max(input_lengths)
            
            padded_inputs = []
            for seq in input_sequences:
                pad_length = max_len - len(seq)
                padded_seq = torch.cat([seq, torch.zeros(pad_length, dtype=torch.long)])
                padded_inputs.append(padded_seq)
            
            return torch.stack(padded_inputs), torch.tensor(input_lengths), torch.stack(list(targets))
        
        else:  # Language modeling
            input_lengths = [len(seq) for seq in input_sequences]
            target_lengths = [len(seq) for seq in targets]
            max_input_len = max(input_lengths)
            max_target_len = max(target_lengths)
            
            padded_inputs = []
            padded_targets = []
            
            for input_seq, target_seq in zip(input_sequences, targets):
                # Pad inputs
                pad_length = max_input_len - len(input_seq)
                padded_input = torch.cat([input_seq, torch.zeros(pad_length, dtype=torch.long)])
                padded_inputs.append(padded_input)
                
                # Pad targets
                pad_length = max_target_len - len(target_seq)
                padded_target = torch.cat([target_seq, torch.zeros(pad_length, dtype=torch.long)])
                padded_targets.append(padded_target)
            
            return (torch.stack(padded_inputs), torch.tensor(input_lengths),
                    torch.stack(padded_targets), torch.tensor(target_lengths))
    
    else:
        # Handle other formats (like masked LM)
        return batch

# ============================================================================
# POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Positional encoding for self-attention models
    Essential since self-attention has no inherent notion of position
    """
    
    def __init__(self, d_model, max_length=512):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_length, batch_size, d_model)
        Returns:
            x with positional encoding added
        """
        return x + self.pe[:x.size(0), :]

# ============================================================================
# SCALED DOT-PRODUCT ATTENTION
# ============================================================================

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism
    Foundation of the Transformer architecture
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    
    def __init__(self, d_k, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch_size, seq_length, d_k)
            key: (batch_size, seq_length, d_k)
            value: (batch_size, seq_length, d_v)
            mask: (batch_size, seq_length, seq_length) optional
            
        Returns:
            output: (batch_size, seq_length, d_v)
            attention_weights: (batch_size, seq_length, seq_length)
        """
        batch_size, seq_length, d_k = query.size()
        
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

# ============================================================================
# MULTI-HEAD ATTENTION
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism
    Allows model to jointly attend to information from different representation subspaces
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: (batch_size, seq_length, d_model)
            mask: (batch_size, seq_length, seq_length) optional
            
        Returns:
            output: (batch_size, seq_length, d_model)
            attention_weights: (batch_size, num_heads, seq_length, seq_length)
        """
        batch_size, seq_length, d_model = query.size()
        
        # Linear projections and reshape for multi-head
        Q = self.w_q(query).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
        # Expand mask for multiple heads
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        
        # Apply attention to each head
        attention_output, attention_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_length, d_model
        )
        
        # Final linear projection
        output = self.w_o(attention_output)
        
        return output, attention_weights

# ============================================================================
# SELF-ATTENTION ENCODER LAYER
# ============================================================================

class SelfAttentionEncoderLayer(nn.Module):
    """
    Self-Attention Encoder Layer with feed-forward network
    Core building block of Transformer encoder
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(SelfAttentionEncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_length, d_model)
            mask: (batch_size, seq_length, seq_length) optional
            
        Returns:
            output: (batch_size, seq_length, d_model)
            attention_weights: attention weights from self-attention
        """
        # Self-attention with residual connection and layer norm
        attn_output, attention_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        output = self.norm2(x + self.dropout(ff_output))
        
        return output, attention_weights

# ============================================================================
# SELF-ATTENTION MODEL FOR CLASSIFICATION
# ============================================================================

class SelfAttentionClassifier(nn.Module):
    """
    Self-attention model for sequence classification
    Demonstrates self-attention without recurrence
    """
    
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=3, 
                 d_ff=512, max_length=512, num_classes=2, dropout=0.1):
        super(SelfAttentionClassifier, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_length)
        
        # Stack of self-attention layers
        self.layers = nn.ModuleList([
            SelfAttentionEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, input_ids, input_lengths=None):
        """
        Args:
            input_ids: (batch_size, seq_length)
            input_lengths: (batch_size,) actual sequence lengths
            
        Returns:
            logits: (batch_size, num_classes)
            attention_weights: List of attention weights from each layer
        """
        batch_size, seq_length = input_ids.size()
        
        # Create padding mask
        if input_lengths is not None:
            mask = torch.zeros(batch_size, seq_length, seq_length, device=input_ids.device)
            for i, length in enumerate(input_lengths):
                mask[i, :length, :length] = 1
        else:
            mask = None
        
        # Embedding + positional encoding
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)  # (seq_length, batch_size, d_model)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_length, d_model)
        x = self.dropout(x)
        
        # Apply self-attention layers
        all_attention_weights = []
        for layer in self.layers:
            x, attention_weights = layer(x, mask)
            all_attention_weights.append(attention_weights)
        
        # Global average pooling for classification
        if input_lengths is not None:
            # Masked average pooling
            pooled = torch.zeros(batch_size, self.d_model, device=input_ids.device)
            for i, length in enumerate(input_lengths):
                pooled[i] = x[i, :length].mean(dim=0)
        else:
            pooled = x.mean(dim=1)  # (batch_size, d_model)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits, all_attention_weights

# ============================================================================
# SELF-ATTENTION LANGUAGE MODEL
# ============================================================================

class SelfAttentionLanguageModel(nn.Module):
    """
    Self-attention based language model
    Similar to Transformer decoder but for language modeling
    """
    
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=3, 
                 d_ff=512, max_length=512, dropout=0.1):
        super(SelfAttentionLanguageModel, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_length)
        
        # Stack of self-attention layers
        self.layers = nn.ModuleList([
            SelfAttentionEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def create_causal_mask(self, seq_length, device):
        """Create causal mask for autoregressive generation"""
        mask = torch.tril(torch.ones(seq_length, seq_length, device=device))
        return mask.unsqueeze(0)  # (1, seq_length, seq_length)
    
    def forward(self, input_ids, causal=True):
        """
        Args:
            input_ids: (batch_size, seq_length)
            causal: Whether to use causal masking for autoregressive modeling
            
        Returns:
            logits: (batch_size, seq_length, vocab_size)
            attention_weights: List of attention weights
        """
        batch_size, seq_length = input_ids.size()
        
        # Create causal mask if needed
        if causal:
            mask = self.create_causal_mask(seq_length, input_ids.device)
            mask = mask.expand(batch_size, -1, -1)
        else:
            mask = None
        
        # Embedding + positional encoding
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)
        x = self.dropout(x)
        
        # Apply self-attention layers
        all_attention_weights = []
        for layer in self.layers:
            x, attention_weights = layer(x, mask)
            all_attention_weights.append(attention_weights)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits, all_attention_weights

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_classification_model(model, train_loader, val_loader, epochs=10, learning_rate=0.001):
    """Train self-attention classification model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_accuracies = []
    
    print(f"Training on device: {device}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        
        for batch_idx, (input_ids, input_lengths, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            input_lengths = input_lengths.to(device) 
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            logits, attention_weights = model(input_ids, input_lengths)
            loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_accuracy = evaluate_classification_model(model, val_loader, device)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    return train_losses, val_accuracies

def evaluate_classification_model(model, data_loader, device):
    """Evaluate classification model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for input_ids, input_lengths, labels in data_loader:
            input_ids = input_ids.to(device)
            input_lengths = input_lengths.to(device)
            labels = labels.to(device)
            
            logits, _ = model(input_ids, input_lengths)
            predictions = logits.argmax(dim=1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    return correct / total

def train_language_model(model, train_loader, val_loader, epochs=10, learning_rate=0.001):
    """Train self-attention language model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    train_losses = []
    val_perplexities = []
    
    print(f"Training on device: {device}")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (input_ids, input_lengths, target_ids, target_lengths) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            optimizer.zero_grad()
            
            logits, attention_weights = model(input_ids, causal=True)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        val_perplexity = evaluate_language_model(model, val_loader, criterion, device)
        val_perplexities.append(val_perplexity)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Perplexity: {val_perplexity:.2f}")
    
    return train_losses, val_perplexities

def evaluate_language_model(model, data_loader, criterion, device):
    """Evaluate language model perplexity"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for input_ids, input_lengths, target_ids, target_lengths in data_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            logits, _ = model(input_ids, causal=True)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
            
            total_loss += loss.item() * target_ids.numel()
            total_tokens += target_ids.numel()
    
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
# ATTENTION VISUALIZATION
# ============================================================================

def visualize_self_attention(tokens, attention_weights, layer_idx=0, head_idx=0, save_path=None):
    """Visualize self-attention patterns"""
    # Extract attention for specific layer and head
    if attention_weights[layer_idx].dim() == 4:  # (batch, heads, seq, seq)
        attn_matrix = attention_weights[layer_idx][0, head_idx].cpu().numpy()
    else:
        attn_matrix = attention_weights[layer_idx][0].cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn_matrix,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='Blues',
        cbar=True,
        square=True
    )
    
    plt.title(f'Self-Attention Patterns (Layer {layer_idx}, Head {head_idx})')
    plt.xlabel('Key Positions')
    plt.ylabel('Query Positions')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return plt.gcf()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"=== Advanced Attention Mechanisms ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load dataset
    train_sentences, val_sentences, test_sentences = load_wikitext2_dataset()
    
    # Use subset for demonstration
    train_subset = train_sentences[:1000]
    val_subset = val_sentences[:200]
    test_subset = test_sentences[:200]
    
    # Build vocabulary
    vocab, idx_to_word = build_vocabulary(train_subset, vocab_size=3000)
    
    results = []
    training_histories = {}
    
    # Test Self-Attention Classification
    print("\n" + "="*50)
    print("Training Self-Attention Classifier")
    
    # Create classification dataset
    train_dataset = SelfAttentionDataset(train_subset, vocab, task='sequence_classification')
    val_dataset = SelfAttentionDataset(val_subset, vocab, task='sequence_classification')
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_advanced)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_advanced)
    
    # Initialize classification model
    classifier = SelfAttentionClassifier(
        vocab_size=len(vocab),
        d_model=256,
        num_heads=8,
        num_layers=3,
        d_ff=512,
        num_classes=2
    )
    
    # Train classifier
    metrics = track_computational_metrics(
        'Self-Attention-Classifier',
        train_classification_model,
        classifier, train_loader, val_loader, 8, 0.001
    )
    
    train_losses, val_accuracies = metrics['result']
    training_histories['Self-Attention-Classifier'] = (train_losses, val_accuracies)
    
    result1 = {
        'model': 'Self-Attention-Classifier',
        'task': 'sequence_classification',
        'year': '2016',
        'final_accuracy': val_accuracies[-1] if val_accuracies else 0,
        'parameters': count_parameters(classifier),
        'training_time': metrics['training_time_minutes'],
        'memory_usage': metrics['memory_usage_mb'],
        'innovation': 'Self-attention for sequence classification'
    }
    results.append(result1)
    
    # Test Self-Attention Language Model
    print("\n" + "="*50)
    print("Training Self-Attention Language Model")
    
    # Create language modeling dataset
    train_dataset_lm = SelfAttentionDataset(train_subset, vocab, task='language_modeling')
    val_dataset_lm = SelfAttentionDataset(val_subset, vocab, task='language_modeling')
    
    train_loader_lm = DataLoader(train_dataset_lm, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_advanced)
    val_loader_lm = DataLoader(val_dataset_lm, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_advanced)
    
    # Initialize language model
    language_model = SelfAttentionLanguageModel(
        vocab_size=len(vocab),
        d_model=256,
        num_heads=8,
        num_layers=3,
        d_ff=512
    )
    
    # Train language model
    metrics = track_computational_metrics(
        'Self-Attention-LM',
        train_language_model,
        language_model, train_loader_lm, val_loader_lm, 8, 0.001
    )
    
    train_losses, val_perplexities = metrics['result']
    training_histories['Self-Attention-LM'] = (train_losses, val_perplexities)
    
    result2 = {
        'model': 'Self-Attention-LM',
        'task': 'language_modeling',
        'year': '2017',
        'final_perplexity': val_perplexities[-1] if val_perplexities else 0,
        'parameters': count_parameters(language_model),
        'training_time': metrics['training_time_minutes'],
        'memory_usage': metrics['memory_usage_mb'],
        'innovation': 'Self-attention for language modeling'
    }
    results.append(result2)
    
    # Demonstrate self-attention visualization
    print("\n" + "="*50)
    print("SELF-ATTENTION VISUALIZATION")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier.to(device)
    classifier.eval()
    
    # Get example for visualization
    if len(test_subset) > 0:
        example_sentence = test_subset[0][:10]  # Take first 10 words
        example_tokens = [vocab.get(word, vocab['<UNK>']) for word in example_sentence]
        
        input_tensor = torch.tensor([example_tokens], device=device)
        input_lengths = torch.tensor([len(example_tokens)], device=device)
        
        with torch.no_grad():
            logits, attention_weights = classifier(input_tensor, input_lengths)
        
        # Visualize attention for each layer
        for layer_idx in range(min(2, len(attention_weights))):
            for head_idx in range(min(2, attention_weights[layer_idx].size(1))):
                save_path = f'AI-ML-DL/Models/NLP/010_self_attention_layer_{layer_idx}_head_{head_idx}.png'
                fig = visualize_self_attention(
                    example_sentence,
                    attention_weights,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    save_path=save_path
                )
                print(f"Self-attention visualization saved: {save_path}")
    
    # Create training results visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Classification training
    ax = axes[0, 0]
    ax.plot(train_losses, label='Training Loss')
    ax.set_title('Classification Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Classification accuracy
    ax = axes[0, 1]
    ax.plot(val_accuracies, label='Validation Accuracy')
    ax.set_title('Classification Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
    
    # Language model training
    ax = axes[1, 0]
    if 'Self-Attention-LM' in training_histories:
        lm_train_losses, _ = training_histories['Self-Attention-LM']
        ax.plot(lm_train_losses, label='LM Training Loss')
    ax.set_title('Language Model Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Model comparison
    ax = axes[1, 1]
    models = [r['model'] for r in results]
    parameters = [r['parameters'] for r in results]
    ax.bar(models, parameters)
    ax.set_title('Model Size Comparison')
    ax.set_ylabel('Number of Parameters')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/NLP/010_advanced_attention_results.png', dpi=300, bbox_inches='tight')
    print("\nTraining results saved: 010_advanced_attention_results.png")
    
    # Print summary
    print("\n" + "="*60)
    print("ADVANCED ATTENTION MECHANISMS SUMMARY")
    print("="*60)
    
    for result in results:
        print(f"\n{result['model']} ({result['year']}):")
        print(f"  Task: {result['task']}")
        if 'final_accuracy' in result:
            print(f"  Final Accuracy: {result['final_accuracy']:.4f}")
        if 'final_perplexity' in result:
            print(f"  Final Perplexity: {result['final_perplexity']:.2f}")
        print(f"  Parameters: {result['parameters']:,}")
        print(f"  Training Time: {result['training_time']:.2f} minutes")
        print(f"  Innovation: {result['innovation']}")
    
    print("\nKey Innovations Demonstrated:")
    print("- Self-Attention: Attention within single sequences")
    print("- Multi-Head Attention: Multiple attention representations")
    print("- Scaled Dot-Product: Efficient attention computation")
    print("- Positional Encoding: Position information for attention")
    print("- Layer Normalization: Stable training for deep networks")
    
    print("\nSelf-Attention Advantages:")
    print("- No recurrence: Fully parallelizable training")
    print("- Long-range dependencies: Direct connections between distant positions")
    print("- Multiple attention heads: Different types of relationships")
    print("- Interpretability: Attention weights show model focus")
    print("- Efficiency: O(n²) vs O(n³) for some operations")
    
    print("\nFoundation for Transformers:")
    print("- Core attention mechanism established")
    print("- Multi-head attention concept proven")
    print("- Position encoding necessity demonstrated")
    print("- Layer normalization and residual connections")
    print("- Path to 'Attention Is All You Need' (2017)")
    
    return results

if __name__ == "__main__":
    results = main()