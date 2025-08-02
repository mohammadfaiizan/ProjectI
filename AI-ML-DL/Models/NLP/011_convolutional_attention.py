"""
ERA 4: CONVOLUTIONAL ATTENTION (2017)
=====================================

Year: 2017
Innovation: ConvS2S - Convolutional Sequence-to-Sequence with attention
Previous Limitation: RNN-based seq2seq sequential processing, limited parallelization
Performance Gain: Fully parallel training, hierarchical feature extraction
Impact: Alternative to RNN approach, competitive with early Transformers

This file demonstrates Facebook's ConvS2S architecture that combined convolutional
networks with attention mechanisms, showing an alternative path to the Transformer.
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

YEAR = "2017"
INNOVATION = "Convolutional Sequence-to-Sequence (ConvS2S)"
PREVIOUS_LIMITATION = "RNN sequential processing limits parallelization and training speed"
IMPACT = "Fully parallel training, hierarchical features, competitive alternative to Transformers"

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
                    if 6 <= len(tokens) <= 25:  # Good length for ConvS2S
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
# DATASET CLASS FOR CONVS2S EXPERIMENTS
# ============================================================================

class ConvS2SDataset(Dataset):
    """Dataset for ConvS2S experiments"""
    
    def __init__(self, sentences, vocab, task='translation', max_length=20):
        self.vocab = vocab
        self.task = task
        self.max_length = max_length
        self.pairs = []
        
        if task == 'translation':
            # Simulate translation by word substitution and reordering
            substitutions = {
                'the': 'le', 'a': 'un', 'is': 'est', 'and': 'et',
                'of': 'de', 'to': 'à', 'in': 'dans', 'for': 'pour',
                'with': 'avec', 'on': 'sur', 'at': 'à', 'by': 'par'
            }
            
            for sentence in sentences:
                if len(sentence) <= max_length:
                    input_seq = sentence
                    # Apply substitutions and slight reordering for "translation"
                    target_seq = []
                    for word in sentence:
                        if word in substitutions:
                            target_seq.append(substitutions[word])
                        else:
                            target_seq.append(word)
                    
                    # Occasionally reverse order for more complex transformation
                    if len(target_seq) > 3 and random.random() < 0.2:
                        mid = len(target_seq) // 2
                        target_seq = target_seq[mid:] + target_seq[:mid]
                    
                    self.pairs.append((input_seq, target_seq))
        
        elif task == 'summarization':
            # Extractive summarization
            for sentence in sentences:
                if len(sentence) >= 6:
                    input_seq = sentence
                    # Extract key words (first, last, longest)
                    summary = [sentence[0], sentence[-1]]
                    if len(sentence) > 2:
                        # Add longest word not already included
                        longest = max(sentence[1:-1], key=len, default='')
                        if longest and longest not in summary:
                            summary.append(longest)
                    target_seq = summary[:3]  # Max 3 words
                    self.pairs.append((input_seq, target_seq))
        
        elif task == 'reverse':
            # Sequence reversal task
            for sentence in sentences:
                if len(sentence) <= max_length:
                    input_seq = sentence
                    target_seq = sentence[::-1]
                    self.pairs.append((input_seq, target_seq))
        
        print(f"Created {len(self.pairs)} {task} pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.pairs[idx]
        
        input_indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in input_seq]
        target_indices = [self.vocab['<SOS>']] + [self.vocab.get(word, self.vocab['<UNK>']) for word in target_seq] + [self.vocab['<EOS>']]
        
        return torch.tensor(input_indices, dtype=torch.long), torch.tensor(target_indices, dtype=torch.long)

def collate_fn(batch):
    """Custom collate function for ConvS2S"""
    input_sequences, target_sequences = zip(*batch)
    
    input_lengths = [len(seq) for seq in input_sequences]
    target_lengths = [len(seq) for seq in target_sequences]
    
    max_input_len = max(input_lengths)
    max_target_len = max(target_lengths)
    
    padded_inputs = []
    padded_targets = []
    
    for input_seq, target_seq in zip(input_sequences, target_sequences):
        pad_input_length = max_input_len - len(input_seq)
        pad_target_length = max_target_len - len(target_seq)
        
        padded_input = torch.cat([input_seq, torch.zeros(pad_input_length, dtype=torch.long)])
        padded_target = torch.cat([target_seq, torch.zeros(pad_target_length, dtype=torch.long)])
        
        padded_inputs.append(padded_input)
        padded_targets.append(padded_target)
    
    return (torch.stack(padded_inputs), torch.tensor(input_lengths), 
            torch.stack(padded_targets), torch.tensor(target_lengths))

# ============================================================================
# GATED LINEAR UNIT (GLU)
# ============================================================================

class GLU(nn.Module):
    """
    Gated Linear Unit activation function
    GLU(X) = X ⊗ σ(X) where ⊗ is element-wise product
    Used in ConvS2S for better gradient flow
    """
    
    def __init__(self, input_dim):
        super(GLU, self).__init__()
        self.input_dim = input_dim
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, channels, seq_length) where channels = 2 * output_channels
        Returns:
            output: (batch_size, output_channels, seq_length)
        """
        assert x.size(1) % 2 == 0, "Input channels must be even for GLU"
        
        # Split into two halves
        a, b = x.chunk(2, dim=1)
        
        # Apply gating: a ⊗ σ(b)
        return a * torch.sigmoid(b)

# ============================================================================
# CONVOLUTIONAL ENCODER
# ============================================================================

class ConvEncoder(nn.Module):
    """
    Convolutional encoder for ConvS2S
    Uses stacked convolutions with GLU activations and residual connections
    """
    
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=6, 
                 kernel_size=3, dropout=0.1):
        super(ConvEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(512, embed_dim)  # Max sequence length 512
        
        # Initial projection to hidden dimension
        self.input_projection = nn.Linear(embed_dim, hidden_dim)
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            # Each conv layer outputs 2 * hidden_dim for GLU
            conv = nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=2 * hidden_dim,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2  # Same padding
            )
            self.conv_layers.append(conv)
        
        self.glu = GLU(2 * hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization for each layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
    def forward(self, input_ids, input_lengths):
        """
        Args:
            input_ids: (batch_size, seq_length)
            input_lengths: (batch_size,) actual sequence lengths
            
        Returns:
            encoder_outputs: (batch_size, seq_length, hidden_dim)
            encoder_embeddings: (batch_size, seq_length, embed_dim) for attention
        """
        batch_size, seq_length = input_ids.size()
        
        # Token embeddings
        token_embeddings = self.embedding(input_ids)  # (batch_size, seq_length, embed_dim)
        
        # Position embeddings
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(positions)
        
        # Combine embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        
        # Store original embeddings for attention
        encoder_embeddings = embeddings
        
        # Project to hidden dimension
        x = self.input_projection(embeddings)  # (batch_size, seq_length, hidden_dim)
        
        # Transpose for conv1d: (batch_size, hidden_dim, seq_length)
        x = x.transpose(1, 2)
        
        # Apply convolutional layers with residual connections
        for i, (conv, layer_norm) in enumerate(zip(self.conv_layers, self.layer_norms)):
            residual = x
            
            # Convolutional layer with GLU
            conv_output = conv(x)  # (batch_size, 2*hidden_dim, seq_length)
            x = self.glu(conv_output)  # (batch_size, hidden_dim, seq_length)
            
            # Residual connection
            x = x + residual
            
            # Layer normalization (transpose for layer norm)
            x = x.transpose(1, 2)  # (batch_size, seq_length, hidden_dim)
            x = layer_norm(x)
            x = x.transpose(1, 2)  # (batch_size, hidden_dim, seq_length)
            
            x = self.dropout(x)
        
        # Transpose back: (batch_size, seq_length, hidden_dim)
        encoder_outputs = x.transpose(1, 2)
        
        return encoder_outputs, encoder_embeddings

# ============================================================================
# CONVOLUTIONAL ATTENTION
# ============================================================================

class ConvAttention(nn.Module):
    """
    Attention mechanism for ConvS2S
    Computes attention between decoder state and encoder outputs
    """
    
    def __init__(self, decoder_embed_dim, encoder_embed_dim, decoder_hidden_dim):
        super(ConvAttention, self).__init__()
        
        self.decoder_embed_dim = decoder_embed_dim
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        
        # Attention projections
        self.decoder_projection = nn.Linear(decoder_hidden_dim, decoder_embed_dim)
        self.encoder_projection = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.output_projection = nn.Linear(encoder_embed_dim + decoder_hidden_dim, decoder_hidden_dim)
        
    def forward(self, decoder_state, encoder_outputs, encoder_embeddings, input_lengths):
        """
        Compute ConvS2S attention
        
        Args:
            decoder_state: (batch_size, decoder_hidden_dim) current decoder state
            encoder_outputs: (batch_size, seq_length, encoder_hidden_dim) 
            encoder_embeddings: (batch_size, seq_length, encoder_embed_dim)
            input_lengths: (batch_size,) actual input lengths
            
        Returns:
            context_vector: (batch_size, decoder_hidden_dim)
            attention_weights: (batch_size, seq_length)
        """
        batch_size, seq_length, _ = encoder_embeddings.size()
        
        # Project decoder state to embedding dimension
        decoder_proj = self.decoder_projection(decoder_state)  # (batch_size, decoder_embed_dim)
        decoder_proj = decoder_proj.unsqueeze(1)  # (batch_size, 1, decoder_embed_dim)
        
        # Project encoder embeddings
        encoder_proj = self.encoder_projection(encoder_embeddings)  # (batch_size, seq_length, decoder_embed_dim)
        
        # Compute attention scores
        attention_scores = torch.bmm(decoder_proj, encoder_proj.transpose(1, 2))  # (batch_size, 1, seq_length)
        attention_scores = attention_scores.squeeze(1)  # (batch_size, seq_length)
        
        # Create padding mask
        mask = torch.zeros(batch_size, seq_length, device=decoder_state.device)
        for i, length in enumerate(input_lengths):
            mask[i, :length] = 1
        
        # Apply mask
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_length)
        
        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_embeddings)  # (batch_size, 1, encoder_embed_dim)
        context = context.squeeze(1)  # (batch_size, encoder_embed_dim)
        
        # Combine context with decoder state
        combined = torch.cat([context, decoder_state], dim=1)  # (batch_size, encoder_embed_dim + decoder_hidden_dim)
        context_vector = self.output_projection(combined)  # (batch_size, decoder_hidden_dim)
        
        return context_vector, attention_weights

# ============================================================================
# CONVOLUTIONAL DECODER
# ============================================================================

class ConvDecoder(nn.Module):
    """
    Convolutional decoder for ConvS2S with attention
    """
    
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=6,
                 kernel_size=3, dropout=0.1):
        super(ConvDecoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(512, embed_dim)
        
        # Initial projection
        self.input_projection = nn.Linear(embed_dim, hidden_dim)
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=2 * hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size - 1  # Causal padding
            )
            self.conv_layers.append(conv)
        
        self.glu = GLU(2 * hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Attention layers (one per conv layer)
        self.attention_layers = nn.ModuleList([
            ConvAttention(embed_dim, embed_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, target_ids, encoder_outputs, encoder_embeddings, input_lengths):
        """
        Args:
            target_ids: (batch_size, target_length)
            encoder_outputs: (batch_size, input_length, hidden_dim)
            encoder_embeddings: (batch_size, input_length, embed_dim)
            input_lengths: (batch_size,)
            
        Returns:
            logits: (batch_size, target_length, vocab_size)
            attention_weights: List of attention weights from each layer
        """
        batch_size, target_length = target_ids.size()
        
        # Token embeddings
        token_embeddings = self.embedding(target_ids)
        
        # Position embeddings
        positions = torch.arange(target_length, device=target_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(positions)
        
        # Combine embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        
        # Project to hidden dimension
        x = self.input_projection(embeddings)  # (batch_size, target_length, hidden_dim)
        
        # Transpose for conv1d
        x = x.transpose(1, 2)  # (batch_size, hidden_dim, target_length)
        
        all_attention_weights = []
        
        # Apply convolutional layers with attention
        for i, (conv, layer_norm, attention) in enumerate(zip(self.conv_layers, self.layer_norms, self.attention_layers)):
            residual = x
            
            # Convolutional layer with causal masking
            conv_output = conv(x)  # (batch_size, 2*hidden_dim, target_length + kernel_size - 1)
            
            # Remove future information (causal)
            conv_output = conv_output[:, :, :target_length]  # (batch_size, 2*hidden_dim, target_length)
            
            # GLU activation
            x = self.glu(conv_output)  # (batch_size, hidden_dim, target_length)
            
            # Residual connection
            x = x + residual
            
            # Transpose for attention and layer norm
            x = x.transpose(1, 2)  # (batch_size, target_length, hidden_dim)
            
            # Apply attention at each position
            attended_outputs = []
            position_attention_weights = []
            
            for t in range(target_length):
                decoder_state = x[:, t, :]  # (batch_size, hidden_dim)
                context_vector, attention_weights = attention(
                    decoder_state, encoder_outputs, encoder_embeddings, input_lengths
                )
                attended_outputs.append(context_vector)
                position_attention_weights.append(attention_weights)
            
            # Stack attended outputs
            attended = torch.stack(attended_outputs, dim=1)  # (batch_size, target_length, hidden_dim)
            
            # Add attended context to layer output
            x = x + attended
            
            # Layer normalization
            x = layer_norm(x)
            x = x.transpose(1, 2)  # (batch_size, hidden_dim, target_length)
            
            x = self.dropout(x)
            
            # Store attention weights
            all_attention_weights.append(torch.stack(position_attention_weights, dim=1))
        
        # Transpose back and project to vocabulary
        x = x.transpose(1, 2)  # (batch_size, target_length, hidden_dim)
        logits = self.output_projection(x)  # (batch_size, target_length, vocab_size)
        
        return logits, all_attention_weights

# ============================================================================
# CONVS2S MODEL
# ============================================================================

class ConvS2S(nn.Module):
    """
    Complete Convolutional Sequence-to-Sequence model
    Demonstrates fully convolutional approach with attention
    """
    
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, 
                 num_encoder_layers=6, num_decoder_layers=6, dropout=0.1):
        super(ConvS2S, self).__init__()
        
        self.encoder = ConvEncoder(
            vocab_size, embed_dim, hidden_dim, num_encoder_layers, dropout=dropout
        )
        self.decoder = ConvDecoder(
            vocab_size, embed_dim, hidden_dim, num_decoder_layers, dropout=dropout
        )
        self.vocab_size = vocab_size
        
    def forward(self, input_ids, input_lengths, target_ids, teacher_forcing_ratio=1.0):
        """
        Training forward pass
        
        Args:
            input_ids: (batch_size, input_length)
            input_lengths: (batch_size,)
            target_ids: (batch_size, target_length)
            teacher_forcing_ratio: Always use teacher forcing for ConvS2S training
            
        Returns:
            logits: (batch_size, target_length-1, vocab_size)
            attention_weights: Attention weights from decoder layers
        """
        # Encode
        encoder_outputs, encoder_embeddings = self.encoder(input_ids, input_lengths)
        
        # Decode (exclude last token for input, exclude first for target)
        decoder_input = target_ids[:, :-1]
        decoder_target = target_ids[:, 1:]
        
        logits, attention_weights = self.decoder(
            decoder_input, encoder_outputs, encoder_embeddings, input_lengths
        )
        
        return logits, attention_weights
    
    def generate(self, input_ids, input_lengths, max_length=20, vocab=None, idx_to_word=None):
        """
        Generate sequence using the trained model
        Note: ConvS2S generation is more complex due to fully parallel training
        This is a simplified version for demonstration
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        with torch.no_grad():
            # Encode
            encoder_outputs, encoder_embeddings = self.encoder(input_ids, input_lengths)
            
            # Initialize with SOS token
            generated = [vocab['<SOS>']]
            
            for _ in range(max_length):
                # Prepare decoder input
                decoder_input = torch.tensor([generated], device=device)
                
                # Decode
                logits, _ = self.decoder(
                    decoder_input, encoder_outputs, encoder_embeddings, input_lengths
                )
                
                # Get next token
                next_token_logits = logits[0, -1, :]  # Last position
                next_token_idx = next_token_logits.argmax().item()
                
                if next_token_idx == vocab['<EOS>']:
                    break
                
                generated.append(next_token_idx)
            
            # Convert to words
            generated_words = []
            for idx in generated[1:]:  # Skip SOS
                if idx in idx_to_word:
                    generated_words.append(idx_to_word[idx])
        
        return generated_words

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_model(model, train_loader, val_loader, epochs=10, learning_rate=0.001):
    """Train ConvS2S model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    train_losses = []
    val_losses = []
    
    print(f"Training on device: {device}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        
        for batch_idx, (input_ids, input_lengths, target_ids, target_lengths) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, attention_weights = model(input_ids, input_lengths, target_ids)
            
            # Compute loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids[:, 1:].reshape(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_loss = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return train_losses, val_losses

def evaluate_model(model, data_loader, criterion, device):
    """Evaluate ConvS2S model"""
    model.eval()
    total_loss = 0
    total_batches = 0
    
    with torch.no_grad():
        for input_ids, input_lengths, target_ids, target_lengths in data_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            logits, _ = model(input_ids, input_lengths, target_ids)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids[:, 1:].reshape(-1))
            
            total_loss += loss.item()
            total_batches += 1
    
    return total_loss / total_batches

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
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"=== Convolutional Attention ({YEAR}) ===")
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
    
    # Test ConvS2S on different tasks
    tasks = ['reverse', 'translation']
    
    for task in tasks[:1]:  # Test one task for demo
        print(f"\n" + "="*50)
        print(f"Training ConvS2S for {task.upper()} task")
        
        # Create datasets
        train_dataset = ConvS2SDataset(train_subset, vocab, task=task)
        val_dataset = ConvS2SDataset(val_subset, vocab, task=task)
        test_dataset = ConvS2SDataset(test_subset, vocab, task=task)
        
        # Create data loaders
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        # Initialize model
        model = ConvS2S(
            vocab_size=len(vocab),
            embed_dim=256,
            hidden_dim=512,
            num_encoder_layers=4,  # Reduced for demo
            num_decoder_layers=4,
            dropout=0.1
        )
        
        # Train model
        model_name = f'ConvS2S-{task}'
        metrics = track_computational_metrics(
            model_name,
            train_model,
            model, train_loader, val_loader, 8, 0.001
        )
        
        train_losses, val_losses = metrics['result']
        training_histories[model_name] = (train_losses, val_losses)
        
        result = {
            'model': model_name,
            'task': task,
            'year': '2017',
            'final_loss': val_losses[-1] if val_losses else 0,
            'parameters': count_parameters(model),
            'training_time': metrics['training_time_minutes'],
            'memory_usage': metrics['memory_usage_mb'],
            'innovation': 'Fully convolutional seq2seq with attention'
        }
        results.append(result)
        
        # Demonstrate generation
        print(f"\n{task.upper()} TASK EXAMPLES:")
        print("="*30)
        
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        for i in range(3):
            if i < len(test_dataset):
                input_seq, target_seq = test_dataset[i]
                
                # Convert to words
                input_words = [idx_to_word[idx.item()] for idx in input_seq if idx.item() in idx_to_word and idx.item() != 0]
                target_words = [idx_to_word[idx.item()] for idx in target_seq[1:-1] if idx.item() in idx_to_word and idx.item() != 0]
                
                # Generate
                input_tensor = input_seq.unsqueeze(0).to(device)
                input_lengths = torch.tensor([len(input_seq)]).to(device)
                
                generated = model.generate(input_tensor, input_lengths, vocab=vocab, idx_to_word=idx_to_word)
                
                print(f"  Input:     {' '.join(input_words)}")
                print(f"  Target:    {' '.join(target_words)}")
                print(f"  Generated: {' '.join(generated)}")
                print()
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss curves
    ax = axes[0, 0]
    for model_name, (train_losses, _) in training_histories.items():
        ax.plot(train_losses, label=model_name)
    ax.set_title('Training Loss Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Validation loss curves
    ax = axes[0, 1]
    for model_name, (_, val_losses) in training_histories.items():
        ax.plot(val_losses, label=model_name)
    ax.set_title('Validation Loss Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Model comparison
    ax = axes[1, 0]
    models = [r['model'] for r in results]
    parameters = [r['parameters'] for r in results]
    ax.bar(models, parameters)
    ax.set_title('Model Size (Parameters)')
    ax.set_ylabel('Number of Parameters')
    ax.tick_params(axis='x', rotation=45)
    
    # Training time comparison
    ax = axes[1, 1]
    training_times = [r['training_time'] for r in results]
    ax.bar(models, training_times)
    ax.set_title('Training Time')
    ax.set_ylabel('Time (minutes)')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/NLP/011_convolutional_attention_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved: 011_convolutional_attention_results.png")
    
    # Print summary
    print("\n" + "="*60)
    print("CONVOLUTIONAL ATTENTION SUMMARY")
    print("="*60)
    
    for result in results:
        print(f"\n{result['model']} ({result['year']}):")
        print(f"  Task: {result['task']}")
        print(f"  Final Loss: {result['final_loss']:.4f}")
        print(f"  Parameters: {result['parameters']:,}")
        print(f"  Training Time: {result['training_time']:.2f} minutes")
        print(f"  Memory Usage: {result['memory_usage']:.2f} MB")
        print(f"  Innovation: {result['innovation']}")
    
    print("\nConvS2S Key Components:")
    print("- Convolutional Encoders: Stacked 1D convolutions with GLU")
    print("- Convolutional Decoders: Causal convolutions for generation")
    print("- GLU Activation: Gated Linear Units for better gradient flow")
    print("- Position Embeddings: Learned positional representations")
    print("- Multi-layer Attention: Attention at each decoder layer")
    
    print("\nConvS2S Advantages:")
    print("- Fully parallel training (no sequential dependencies)")
    print("- Hierarchical feature extraction through stacked convolutions")
    print("- Competitive performance with early Transformers")
    print("- Efficient for shorter sequences")
    print("- GLU activation prevents vanishing gradients")
    
    print("\nHistorical Context:")
    print("- Alternative to RNN-based seq2seq models")
    print("- Competitive with Transformer (released same year)")
    print("- Showed CNNs could work for sequence modeling")
    print("- Influenced later convolutional language models")
    print("- Bridge between RNN and Transformer eras")
    
    print("\nComparison with Transformers:")
    print("- ConvS2S: Fixed receptive field, local connectivity")
    print("- Transformer: Global attention, all-to-all connections")
    print("- ConvS2S: More efficient for short sequences")
    print("- Transformer: Better for long-range dependencies")
    print("- Both: Fully parallelizable training")
    
    return results

if __name__ == "__main__":
    results = main()