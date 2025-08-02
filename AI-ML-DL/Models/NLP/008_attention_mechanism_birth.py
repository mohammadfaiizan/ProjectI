"""
ERA 3: ATTENTION MECHANISM BIRTH (2015)
=======================================

Year: 2015
Innovation: Attention mechanism for sequence-to-sequence models (Bahdanau et al.)
Previous Limitation: Seq2seq information bottleneck - entire input compressed to fixed vector
Performance Gain: Dynamic context vectors, better handling of long sequences
Impact: Revolutionary breakthrough enabling modern NLP, foundation for Transformers

This file demonstrates the birth of attention mechanisms that solved the information
bottleneck problem and revolutionized sequence-to-sequence modeling.
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

YEAR = "2015"
INNOVATION = "Attention Mechanism (Bahdanau et al.)"
PREVIOUS_LIMITATION = "Seq2seq bottleneck: entire input sequence compressed to fixed vector"
IMPACT = "Dynamic attention to input positions, foundation for Transformer revolution"

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
                    if 6 <= len(tokens) <= 25:  # Longer sequences to show attention benefit
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
# DATASET CLASS FOR ATTENTION TRAINING
# ============================================================================

class AttentionDataset(Dataset):
    """Dataset for training attention-based seq2seq models"""
    
    def __init__(self, sentences, vocab, task='reverse', max_length=20):
        self.vocab = vocab
        self.task = task
        self.max_length = max_length
        self.pairs = []
        
        if task == 'reverse':
            # Reverse sequence task - benefits greatly from attention
            for sentence in sentences:
                if len(sentence) <= max_length:
                    input_seq = sentence
                    target_seq = sentence[::-1]
                    self.pairs.append((input_seq, target_seq))
        
        elif task == 'sort':
            # Sort words alphabetically - requires attention to all positions
            for sentence in sentences:
                if len(sentence) <= max_length:
                    input_seq = sentence
                    target_seq = sorted(sentence)
                    self.pairs.append((input_seq, target_seq))
        
        elif task == 'summarize':
            # Extractive summarization - attention to important words
            for sentence in sentences:
                if len(sentence) >= 8:
                    input_seq = sentence
                    # Simple heuristic: keep first, last, and longest words
                    important_words = [sentence[0], sentence[-1]]
                    longest_word = max(sentence, key=len)
                    if longest_word not in important_words:
                        important_words.append(longest_word)
                    target_seq = important_words[:3]  # Max 3 words summary
                    self.pairs.append((input_seq, target_seq))
        
        print(f"Created {len(self.pairs)} {task} pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.pairs[idx]
        
        # Convert to indices
        input_indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in input_seq]
        target_indices = [self.vocab['<SOS>']] + [self.vocab.get(word, self.vocab['<UNK>']) for word in target_seq] + [self.vocab['<EOS>']]
        
        return torch.tensor(input_indices, dtype=torch.long), torch.tensor(target_indices, dtype=torch.long)

def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    input_sequences, target_sequences = zip(*batch)
    
    input_lengths = [len(seq) for seq in input_sequences]
    target_lengths = [len(seq) for seq in target_sequences]
    
    max_input_len = max(input_lengths)
    max_target_len = max(target_lengths)
    
    padded_inputs = []
    padded_targets = []
    
    for input_seq, target_seq in zip(input_sequences, target_sequences):
        # Pad sequences
        pad_input_length = max_input_len - len(input_seq)
        pad_target_length = max_target_len - len(target_seq)
        
        padded_input = torch.cat([input_seq, torch.zeros(pad_input_length, dtype=torch.long)])
        padded_target = torch.cat([target_seq, torch.zeros(pad_target_length, dtype=torch.long)])
        
        padded_inputs.append(padded_input)
        padded_targets.append(padded_target)
    
    return (torch.stack(padded_inputs), torch.tensor(input_lengths), 
            torch.stack(padded_targets), torch.tensor(target_lengths))

# ============================================================================
# BAHDANAU ATTENTION MECHANISM
# ============================================================================

class BahdanauAttention(nn.Module):
    """
    Bahdanau Attention (Additive Attention) - The first attention mechanism
    
    This mechanism computes attention scores using a feedforward network
    and enables the decoder to focus on different parts of the input sequence.
    """
    
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, attention_dim=128):
        super(BahdanauAttention, self).__init__()
        
        self.attention_dim = attention_dim
        
        # Linear transformations for attention computation
        self.encoder_projection = nn.Linear(encoder_hidden_dim, attention_dim, bias=False)
        self.decoder_projection = nn.Linear(decoder_hidden_dim, attention_dim, bias=False)
        self.attention_vector = nn.Linear(attention_dim, 1, bias=False)
        
    def forward(self, encoder_outputs, decoder_hidden, input_lengths):
        """
        Compute attention weights and context vector
        
        Args:
            encoder_outputs: (batch_size, max_seq_length, encoder_hidden_dim)
            decoder_hidden: (batch_size, decoder_hidden_dim)
            input_lengths: (batch_size,) actual lengths of input sequences
            
        Returns:
            context_vector: (batch_size, encoder_hidden_dim)
            attention_weights: (batch_size, max_seq_length)
        """
        batch_size, max_seq_length, encoder_hidden_dim = encoder_outputs.size()
        
        # Project encoder outputs
        encoder_proj = self.encoder_projection(encoder_outputs)  # (batch_size, max_seq_length, attention_dim)
        
        # Project decoder hidden state and expand
        decoder_proj = self.decoder_projection(decoder_hidden).unsqueeze(1)  # (batch_size, 1, attention_dim)
        decoder_proj = decoder_proj.expand(batch_size, max_seq_length, self.attention_dim)
        
        # Compute attention scores (additive attention)
        attention_scores = self.attention_vector(torch.tanh(encoder_proj + decoder_proj))  # (batch_size, max_seq_length, 1)
        attention_scores = attention_scores.squeeze(-1)  # (batch_size, max_seq_length)
        
        # Create mask for padding positions
        mask = torch.zeros(batch_size, max_seq_length, device=encoder_outputs.device)
        for i, length in enumerate(input_lengths):
            mask[i, :length] = 1
        
        # Apply mask (set padded positions to large negative value)
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, max_seq_length)
        
        # Compute context vector
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # (batch_size, 1, encoder_hidden_dim)
        context_vector = context_vector.squeeze(1)  # (batch_size, encoder_hidden_dim)
        
        return context_vector, attention_weights

# ============================================================================
# ATTENTION-BASED ENCODER
# ============================================================================

class AttentionEncoder(nn.Module):
    """
    Encoder for attention-based seq2seq model
    Outputs all hidden states for attention computation
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=1, dropout=0.2):
        super(AttentionEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Bidirectional for better representations
        )
        self.dropout = nn.Dropout(dropout)
        
        # Output dimension is 2 * hidden_dim due to bidirectionality
        self.output_dim = 2 * hidden_dim
    
    def forward(self, input_sequence, input_lengths):
        """
        Encode input sequence
        
        Returns:
            outputs: All hidden states (batch_size, max_seq_length, 2*hidden_dim)
            hidden: Final hidden state
        """
        embedded = self.embedding(input_sequence)
        embedded = self.dropout(embedded)
        
        # Pack for efficiency
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, input_lengths, batch_first=True, enforce_sorted=False
        )
        
        packed_outputs, hidden = self.lstm(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        
        return outputs, hidden

# ============================================================================
# ATTENTION-BASED DECODER
# ============================================================================

class AttentionDecoder(nn.Module):
    """
    Decoder with Bahdanau attention mechanism
    Uses attention to dynamically focus on relevant parts of input
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, encoder_output_dim=512, num_layers=1, dropout=0.2):
        super(AttentionDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Attention mechanism
        self.attention = BahdanauAttention(encoder_output_dim, hidden_dim)
        
        # LSTM takes concatenated input and context
        self.lstm = nn.LSTM(
            embedding_dim + encoder_output_dim,  # Input + context
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim + encoder_output_dim, vocab_size)
    
    def forward(self, input_token, hidden, encoder_outputs, input_lengths):
        """
        Single step decoding with attention
        
        Args:
            input_token: (batch_size, 1) current input token
            hidden: Previous hidden state
            encoder_outputs: (batch_size, max_seq_length, encoder_output_dim)
            input_lengths: (batch_size,) input sequence lengths
            
        Returns:
            output: (batch_size, 1, vocab_size) output probabilities
            hidden: Updated hidden state
            attention_weights: (batch_size, max_seq_length) attention weights
        """
        embedded = self.embedding(input_token)
        embedded = self.dropout(embedded)
        
        # Get current decoder hidden state for attention
        if isinstance(hidden, tuple):
            decoder_hidden = hidden[0][-1]  # Last layer hidden state
        else:
            decoder_hidden = hidden[-1]
        
        # Compute attention
        context_vector, attention_weights = self.attention(encoder_outputs, decoder_hidden, input_lengths)
        
        # Concatenate input embedding with context
        lstm_input = torch.cat([embedded, context_vector.unsqueeze(1)], dim=2)
        
        # LSTM forward pass
        lstm_output, hidden = self.lstm(lstm_input, hidden)
        lstm_output = self.dropout(lstm_output)
        
        # Concatenate LSTM output with context for final prediction
        output_input = torch.cat([lstm_output, context_vector.unsqueeze(1)], dim=2)
        output = self.output_projection(output_input)
        
        return output, hidden, attention_weights

# ============================================================================
# ATTENTION-BASED SEQ2SEQ MODEL
# ============================================================================

class AttentionSeq2Seq(nn.Module):
    """
    Complete attention-based sequence-to-sequence model
    Demonstrates the revolutionary impact of attention mechanisms
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=1, dropout=0.2):
        super(AttentionSeq2Seq, self).__init__()
        
        self.encoder = AttentionEncoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.decoder = AttentionDecoder(vocab_size, embedding_dim, hidden_dim, self.encoder.output_dim, num_layers, dropout)
        self.vocab_size = vocab_size
    
    def forward(self, input_sequence, input_lengths, target_sequence, teacher_forcing_ratio=0.5):
        """
        Training forward pass with attention
        
        Returns:
            outputs: Decoder outputs
            attention_weights: All attention weights for visualization
        """
        batch_size = input_sequence.size(0)
        target_length = target_sequence.size(1)
        
        # Encode
        encoder_outputs, encoder_hidden = self.encoder(input_sequence, input_lengths)
        
        # Initialize decoder
        # Convert bidirectional encoder hidden to unidirectional decoder hidden
        if isinstance(encoder_hidden, tuple):
            h, c = encoder_hidden
            # Take the sum of forward and backward hidden states
            decoder_h = (h[0::2] + h[1::2])  # Sum forward and backward
            decoder_c = (c[0::2] + c[1::2])
            decoder_hidden = (decoder_h, decoder_c)
        else:
            decoder_hidden = encoder_hidden[0::2] + encoder_hidden[1::2]
        
        decoder_input = target_sequence[:, 0:1]  # <SOS> token
        
        outputs = []
        attention_weights_list = []
        
        # Decode sequence
        for t in range(1, target_length):
            decoder_output, decoder_hidden, attention_weights = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs, input_lengths
            )
            
            outputs.append(decoder_output)
            attention_weights_list.append(attention_weights)
            
            # Teacher forcing decision
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            
            if use_teacher_forcing:
                decoder_input = target_sequence[:, t:t+1]
            else:
                decoder_input = decoder_output.argmax(dim=-1)
        
        outputs = torch.cat(outputs, dim=1)
        attention_weights = torch.stack(attention_weights_list, dim=1)  # (batch_size, target_length-1, input_length)
        
        return outputs, attention_weights
    
    def generate_with_attention(self, input_sequence, input_lengths, max_length=20, vocab=None, idx_to_word=None):
        """
        Generate output sequence with attention tracking
        
        Returns:
            generated_sequence: List of generated words
            attention_weights: Attention weights for visualization
        """
        self.eval()
        device = input_sequence.device
        
        with torch.no_grad():
            # Encode
            encoder_outputs, encoder_hidden = self.encoder(input_sequence, input_lengths)
            
            # Initialize decoder
            if isinstance(encoder_hidden, tuple):
                h, c = encoder_hidden
                decoder_h = (h[0::2] + h[1::2])
                decoder_c = (c[0::2] + c[1::2])
                decoder_hidden = (decoder_h, decoder_c)
            else:
                decoder_hidden = encoder_hidden[0::2] + encoder_hidden[1::2]
            
            decoder_input = torch.tensor([[vocab['<SOS>']]], device=device)
            
            generated = []
            attention_weights_list = []
            
            for _ in range(max_length):
                decoder_output, decoder_hidden, attention_weights = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs, input_lengths
                )
                
                attention_weights_list.append(attention_weights.cpu().numpy())
                
                next_token_idx = decoder_output.argmax(dim=-1).item()
                
                if next_token_idx == vocab['<EOS>']:
                    break
                
                if next_token_idx in idx_to_word:
                    generated.append(idx_to_word[next_token_idx])
                
                decoder_input = torch.tensor([[next_token_idx]], device=device)
        
        return generated, np.array(attention_weights_list)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_model(model, train_loader, val_loader, epochs=15, learning_rate=0.001):
    """Train attention-based seq2seq model"""
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
        
        for batch_idx, (input_seq, input_lengths, target_seq, target_lengths) in enumerate(train_loader):
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs, attention_weights = model(input_seq, input_lengths, target_seq, teacher_forcing_ratio=0.5)
            
            # Compute loss
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), target_seq[:, 1:].reshape(-1))
            
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
    """Evaluate attention-based model"""
    model.eval()
    total_loss = 0
    total_batches = 0
    
    with torch.no_grad():
        for input_seq, input_lengths, target_seq, target_lengths in data_loader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            outputs, _ = model(input_seq, input_lengths, target_seq, teacher_forcing_ratio=0.0)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), target_seq[:, 1:].reshape(-1))
            
            total_loss += loss.item()
            total_batches += 1
    
    return total_loss / total_batches

# ============================================================================
# ATTENTION VISUALIZATION
# ============================================================================

def visualize_attention(input_words, output_words, attention_weights, save_path=None):
    """
    Visualize attention weights as a heatmap
    
    Args:
        input_words: List of input words
        output_words: List of output words  
        attention_weights: (output_length, input_length) attention matrix
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        attention_weights,
        xticklabels=input_words,
        yticklabels=output_words,
        cmap='Blues',
        cbar=True,
        square=True,
        linewidths=0.5
    )
    
    plt.title('Attention Weights Visualization')
    plt.xlabel('Input Sequence')
    plt.ylabel('Output Sequence')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    return plt.gcf()

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
    print(f"=== Attention Mechanism Birth ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load dataset
    train_sentences, val_sentences, test_sentences = load_wikitext2_dataset()
    
    # Use subset for demonstration
    train_subset = train_sentences[:1500]
    val_subset = val_sentences[:300]
    test_subset = test_sentences[:300]
    
    # Build vocabulary
    vocab, idx_to_word = build_vocabulary(train_subset, vocab_size=3000)
    
    results = []
    training_histories = {}
    
    # Test attention on reverse task (shows clear benefit)
    task = 'reverse'
    
    print(f"\n" + "="*50)
    print(f"Training Attention-based Seq2Seq for {task.upper()} task")
    
    # Create datasets
    train_dataset = AttentionDataset(train_subset, vocab, task=task)
    val_dataset = AttentionDataset(val_subset, vocab, task=task)
    test_dataset = AttentionDataset(test_subset, vocab, task=task)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Initialize attention model
    model = AttentionSeq2Seq(len(vocab), embedding_dim=128, hidden_dim=256, num_layers=1)
    
    # Train model
    metrics = track_computational_metrics(
        f'Attention-Seq2Seq',
        train_model,
        model, train_loader, val_loader, 12, 0.001
    )
    
    train_losses, val_losses = metrics['result']
    training_histories['Attention-Seq2Seq'] = (train_losses, val_losses)
    
    result = {
        'model': 'Attention-Seq2Seq',
        'year': '2015',
        'final_loss': val_losses[-1] if val_losses else 0,
        'parameters': count_parameters(model),
        'training_time': metrics['training_time_minutes'],
        'memory_usage': metrics['memory_usage_mb'],
        'innovation': 'Dynamic attention to input positions'
    }
    results.append(result)
    
    # Demonstrate attention visualization
    print(f"\n" + "="*50)
    print("ATTENTION VISUALIZATION EXAMPLES")
    print("="*50)
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Show attention for a few examples
    for i in range(3):
        if i < len(test_dataset):
            input_seq, target_seq = test_dataset[i]
            
            # Convert to words
            input_words = [idx_to_word[idx.item()] for idx in input_seq if idx.item() in idx_to_word and idx.item() != 0]
            target_words = [idx_to_word[idx.item()] for idx in target_seq[1:-1] if idx.item() in idx_to_word and idx.item() != 0]
            
            # Generate with attention
            input_tensor = input_seq.unsqueeze(0).to(device)
            input_lengths = torch.tensor([len(input_seq)]).to(device)
            
            generated, attention_weights = model.generate_with_attention(
                input_tensor, input_lengths, vocab=vocab, idx_to_word=idx_to_word
            )
            
            print(f"\nExample {i+1}:")
            print(f"  Input:     {' '.join(input_words)}")
            print(f"  Target:    {' '.join(target_words)}")
            print(f"  Generated: {' '.join(generated)}")
            
            # Show attention patterns
            if len(attention_weights) > 0 and len(generated) > 0:
                # Take first example from batch and first few timesteps
                attention_matrix = attention_weights[:min(len(generated), 5), 0, :len(input_words)]
                print(f"  Attention shape: {attention_matrix.shape}")
                
                # Create attention visualization
                fig = visualize_attention(
                    input_words, 
                    generated[:min(len(generated), 5)], 
                    attention_matrix,
                    f'AI-ML-DL/Models/NLP/008_attention_example_{i+1}.png'
                )
                plt.close(fig)
                print(f"  Attention visualization saved: 008_attention_example_{i+1}.png")
    
    # Create training visualizations
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Training loss
    ax = axes[0]
    ax.plot(train_losses, label='Attention-Seq2Seq')
    ax.set_title('Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Validation loss
    ax = axes[1]
    ax.plot(val_losses, label='Attention-Seq2Seq')
    ax.set_title('Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Model complexity
    ax = axes[2]
    models = [r['model'] for r in results]
    parameters = [r['parameters'] for r in results]
    ax.bar(models, parameters)
    ax.set_title('Model Parameters')
    ax.set_ylabel('Number of Parameters')
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/NLP/008_attention_mechanism_results.png', dpi=300, bbox_inches='tight')
    print("\nTraining visualization saved: 008_attention_mechanism_results.png")
    
    # Print summary
    print("\n" + "="*60)
    print("ATTENTION MECHANISM BIRTH SUMMARY")
    print("="*60)
    
    for result in results:
        print(f"\n{result['model']} ({result['year']}):")
        print(f"  Final Loss: {result['final_loss']:.4f}")
        print(f"  Parameters: {result['parameters']:,}")
        print(f"  Training Time: {result['training_time']:.2f} minutes")
        print(f"  Memory Usage: {result['memory_usage']:.2f} MB")
        print(f"  Innovation: {result['innovation']}")
    
    print("\nBahdanau Attention Mechanism:")
    print("- Additive attention: tanh(Wa*encoder + Wd*decoder)")
    print("- Alignment scores computed for each encoder position")
    print("- Softmax normalization creates attention weights")
    print("- Weighted sum of encoder states creates context vector")
    print("- Context vector changes dynamically for each decoder step")
    
    print("\nKey Insights:")
    print("- Solves information bottleneck of vanilla seq2seq")
    print("- Enables handling of longer input sequences")
    print("- Provides interpretability through attention weights")
    print("- Revolutionary breakthrough enabling modern NLP")
    print("- Foundation for Transformer architecture")
    print("- Attention weights show model 'focus' on relevant input positions")
    print("- Enables alignment between input and output sequences")
    
    return results

if __name__ == "__main__":
    results = main()