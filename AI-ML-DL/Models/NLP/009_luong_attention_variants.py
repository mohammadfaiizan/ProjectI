"""
ERA 3: LUONG ATTENTION VARIANTS (2015)
======================================

Year: 2015
Innovation: Luong attention mechanisms with multiple scoring functions
Previous Limitation: Bahdanau attention only used additive scoring function
Performance Gain: More efficient scoring functions, global vs local attention
Impact: Multiple attention mechanisms, input-feeding, foundation for modern attention

This file demonstrates Luong attention variants that improved upon Bahdanau attention
and introduced key concepts used in modern attention mechanisms.
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
INNOVATION = "Luong Attention Mechanisms (Multiple Variants)"
PREVIOUS_LIMITATION = "Bahdanau attention used only additive scoring, limited efficiency"
IMPACT = "Multiple scoring functions, global/local attention, input-feeding approach"

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
                    if 6 <= len(tokens) <= 25:
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
# DATASET CLASS
# ============================================================================

class LuongDataset(Dataset):
    """Dataset for Luong attention experiments"""
    
    def __init__(self, sentences, vocab, task='reverse', max_length=20):
        self.vocab = vocab
        self.task = task
        self.max_length = max_length
        self.pairs = []
        
        if task == 'reverse':
            for sentence in sentences:
                if len(sentence) <= max_length:
                    input_seq = sentence
                    target_seq = sentence[::-1]
                    self.pairs.append((input_seq, target_seq))
        
        elif task == 'translation':
            # Simulate translation by word replacement
            replacements = {
                'the': 'le', 'a': 'un', 'is': 'est', 'and': 'et',
                'of': 'de', 'to': 'à', 'in': 'dans', 'for': 'pour'
            }
            for sentence in sentences:
                if len(sentence) <= max_length:
                    input_seq = sentence
                    target_seq = [replacements.get(word, word) for word in sentence]
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
    """Custom collate function"""
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
# LUONG ATTENTION MECHANISMS
# ============================================================================

class LuongAttention(nn.Module):
    """
    Luong Attention with multiple scoring functions
    
    Three variants:
    1. Dot: score(h_t, h_s) = h_t^T * h_s
    2. General: score(h_t, h_s) = h_t^T * W_a * h_s  
    3. Concat: score(h_t, h_s) = v_a^T * tanh(W_a * [h_t; h_s])
    """
    
    def __init__(self, hidden_dim, attention_type='general'):
        super(LuongAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.attention_type = attention_type
        
        if attention_type == 'general':
            self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif attention_type == 'concat':
            self.W_a = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
            self.v_a = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, decoder_hidden, encoder_outputs, input_lengths):
        """
        Compute Luong attention
        
        Args:
            decoder_hidden: (batch_size, hidden_dim) current decoder hidden state
            encoder_outputs: (batch_size, max_seq_length, hidden_dim)
            input_lengths: (batch_size,) actual sequence lengths
            
        Returns:
            context_vector: (batch_size, hidden_dim)
            attention_weights: (batch_size, max_seq_length)
        """
        batch_size, max_seq_length, hidden_dim = encoder_outputs.size()
        
        if self.attention_type == 'dot':
            # Dot product attention: h_t^T * h_s
            attention_scores = torch.bmm(
                decoder_hidden.unsqueeze(1),  # (batch_size, 1, hidden_dim)
                encoder_outputs.transpose(1, 2)  # (batch_size, hidden_dim, max_seq_length)
            ).squeeze(1)  # (batch_size, max_seq_length)
        
        elif self.attention_type == 'general':
            # General attention: h_t^T * W_a * h_s
            projected_encoder = self.W_a(encoder_outputs)  # (batch_size, max_seq_length, hidden_dim)
            attention_scores = torch.bmm(
                decoder_hidden.unsqueeze(1),  # (batch_size, 1, hidden_dim)
                projected_encoder.transpose(1, 2)  # (batch_size, hidden_dim, max_seq_length)
            ).squeeze(1)  # (batch_size, max_seq_length)
        
        elif self.attention_type == 'concat':
            # Concatenation attention: v_a^T * tanh(W_a * [h_t; h_s])
            decoder_expanded = decoder_hidden.unsqueeze(1).expand(batch_size, max_seq_length, hidden_dim)
            concatenated = torch.cat([decoder_expanded, encoder_outputs], dim=2)  # (batch_size, max_seq_length, 2*hidden_dim)
            projected = torch.tanh(self.W_a(concatenated))  # (batch_size, max_seq_length, hidden_dim)
            attention_scores = self.v_a(projected).squeeze(2)  # (batch_size, max_seq_length)
        
        # Create mask for padding
        mask = torch.zeros(batch_size, max_seq_length, device=encoder_outputs.device)
        for i, length in enumerate(input_lengths):
            mask[i, :length] = 1
        
        # Apply mask
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Compute context vector
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context_vector, attention_weights

class LocalAttention(nn.Module):
    """
    Local Attention mechanism (Luong et al.)
    Focuses on a small window of encoder positions rather than all positions
    """
    
    def __init__(self, hidden_dim, window_size=5, attention_type='general'):
        super(LocalAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.attention_type = attention_type
        
        # Position prediction network
        self.position_network = nn.Linear(hidden_dim, 1)
        
        # Attention scoring (same as global)
        if attention_type == 'general':
            self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif attention_type == 'concat':
            self.W_a = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
            self.v_a = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, decoder_hidden, encoder_outputs, input_lengths):
        """
        Compute local attention with predicted position
        """
        batch_size, max_seq_length, hidden_dim = encoder_outputs.size()
        
        # Predict position to focus on
        position_scores = self.position_network(decoder_hidden)  # (batch_size, 1)
        positions = torch.sigmoid(position_scores) * max_seq_length  # (batch_size, 1)
        
        # Create attention scores only for local window
        attention_scores = torch.zeros(batch_size, max_seq_length, device=encoder_outputs.device)
        
        for i in range(batch_size):
            center = int(positions[i].item())
            left = max(0, center - self.window_size // 2)
            right = min(max_seq_length, center + self.window_size // 2 + 1)
            
            # Compute attention scores for window
            window_encoder = encoder_outputs[i, left:right, :]  # (window_size, hidden_dim)
            
            if self.attention_type == 'general':
                projected_window = self.W_a(window_encoder)
                window_scores = torch.matmul(decoder_hidden[i], projected_window.t())
            else:  # dot
                window_scores = torch.matmul(decoder_hidden[i], window_encoder.t())
            
            attention_scores[i, left:right] = window_scores
        
        # Apply sequence length mask
        mask = torch.zeros(batch_size, max_seq_length, device=encoder_outputs.device)
        for i, length in enumerate(input_lengths):
            mask[i, :length] = 1
        
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Compute context vector
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context_vector, attention_weights

# ============================================================================
# LUONG ENCODER
# ============================================================================

class LuongEncoder(nn.Module):
    """Encoder for Luong attention experiments"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=1, dropout=0.2):
        super(LuongEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # Unidirectional for simplicity
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_sequence, input_lengths):
        embedded = self.embedding(input_sequence)
        embedded = self.dropout(embedded)
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, input_lengths, batch_first=True, enforce_sorted=False
        )
        
        packed_outputs, hidden = self.lstm(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        
        return outputs, hidden

# ============================================================================
# LUONG DECODER WITH INPUT FEEDING
# ============================================================================

class LuongDecoder(nn.Module):
    """
    Luong Decoder with input-feeding approach
    Concatenates previous attention context with current input
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=1, 
                 attention_type='general', use_local_attention=False, dropout=0.2):
        super(LuongDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_local_attention = use_local_attention
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM with input feeding (embedding + previous context)
        self.lstm = nn.LSTM(
            embedding_dim + hidden_dim,  # Input + previous context
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        if use_local_attention:
            self.attention = LocalAttention(hidden_dim, attention_type=attention_type)
        else:
            self.attention = LuongAttention(hidden_dim, attention_type=attention_type)
        
        self.dropout = nn.Dropout(dropout)
        
        # Output projection (h_t and c_t)
        self.output_projection = nn.Linear(2 * hidden_dim, vocab_size)
    
    def forward(self, input_token, hidden, encoder_outputs, input_lengths, prev_context=None):
        """
        Single step decoding with input feeding
        
        Args:
            input_token: (batch_size, 1)
            hidden: Previous hidden state
            encoder_outputs: (batch_size, max_seq_length, hidden_dim)
            input_lengths: (batch_size,)
            prev_context: (batch_size, hidden_dim) previous context vector
            
        Returns:
            output: (batch_size, 1, vocab_size)
            hidden: Updated hidden state
            context_vector: (batch_size, hidden_dim) current context
            attention_weights: (batch_size, max_seq_length)
        """
        embedded = self.embedding(input_token)
        embedded = self.dropout(embedded)
        
        # Input feeding: concatenate with previous context
        if prev_context is not None:
            lstm_input = torch.cat([embedded, prev_context.unsqueeze(1)], dim=2)
        else:
            # First step: use zeros for context
            batch_size = embedded.size(0)
            zero_context = torch.zeros(batch_size, 1, self.hidden_dim, device=embedded.device)
            lstm_input = torch.cat([embedded, zero_context], dim=2)
        
        # LSTM forward
        lstm_output, hidden = self.lstm(lstm_input, hidden)
        lstm_output = self.dropout(lstm_output)
        
        # Get current decoder hidden state for attention
        current_hidden = lstm_output.squeeze(1)  # (batch_size, hidden_dim)
        
        # Compute attention
        context_vector, attention_weights = self.attention(current_hidden, encoder_outputs, input_lengths)
        
        # Concatenate hidden state and context for output
        output_input = torch.cat([current_hidden, context_vector], dim=1)  # (batch_size, 2*hidden_dim)
        output = self.output_projection(output_input).unsqueeze(1)  # (batch_size, 1, vocab_size)
        
        return output, hidden, context_vector, attention_weights

# ============================================================================
# LUONG SEQ2SEQ MODEL
# ============================================================================

class LuongSeq2Seq(nn.Module):
    """
    Complete Luong Seq2Seq model with different attention variants
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=1, 
                 attention_type='general', use_local_attention=False, dropout=0.2):
        super(LuongSeq2Seq, self).__init__()
        
        self.encoder = LuongEncoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.decoder = LuongDecoder(vocab_size, embedding_dim, hidden_dim, num_layers, 
                                  attention_type, use_local_attention, dropout)
        self.vocab_size = vocab_size
        self.attention_type = attention_type
        self.use_local_attention = use_local_attention
    
    def forward(self, input_sequence, input_lengths, target_sequence, teacher_forcing_ratio=0.5):
        """Training forward pass with input feeding"""
        batch_size = input_sequence.size(0)
        target_length = target_sequence.size(1)
        
        # Encode
        encoder_outputs, encoder_hidden = self.encoder(input_sequence, input_lengths)
        
        # Initialize decoder
        decoder_hidden = encoder_hidden
        decoder_input = target_sequence[:, 0:1]  # <SOS>
        prev_context = None
        
        outputs = []
        attention_weights_list = []
        
        # Decode with input feeding
        for t in range(1, target_length):
            decoder_output, decoder_hidden, context_vector, attention_weights = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs, input_lengths, prev_context
            )
            
            outputs.append(decoder_output)
            attention_weights_list.append(attention_weights)
            
            # Input feeding: use current context for next step
            prev_context = context_vector
            
            # Teacher forcing
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing:
                decoder_input = target_sequence[:, t:t+1]
            else:
                decoder_input = decoder_output.argmax(dim=-1)
        
        outputs = torch.cat(outputs, dim=1)
        attention_weights = torch.stack(attention_weights_list, dim=1)
        
        return outputs, attention_weights
    
    def generate(self, input_sequence, input_lengths, max_length=20, vocab=None, idx_to_word=None):
        """Generate with attention tracking"""
        self.eval()
        device = input_sequence.device
        
        with torch.no_grad():
            # Encode
            encoder_outputs, encoder_hidden = self.encoder(input_sequence, input_lengths)
            
            # Initialize decoder
            decoder_hidden = encoder_hidden
            decoder_input = torch.tensor([[vocab['<SOS>']]], device=device)
            prev_context = None
            
            generated = []
            attention_weights_list = []
            
            for _ in range(max_length):
                decoder_output, decoder_hidden, context_vector, attention_weights = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs, input_lengths, prev_context
                )
                
                attention_weights_list.append(attention_weights.cpu().numpy())
                prev_context = context_vector
                
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

def train_model(model, train_loader, val_loader, epochs=12, learning_rate=0.001):
    """Train Luong attention model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    train_losses = []
    val_losses = []
    
    print(f"Training on device: {device}")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (input_seq, input_lengths, target_seq, target_lengths) in enumerate(train_loader):
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            optimizer.zero_grad()
            
            outputs, attention_weights = model(input_seq, input_lengths, target_seq, teacher_forcing_ratio=0.5)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), target_seq[:, 1:].reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        val_loss = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return train_losses, val_losses

def evaluate_model(model, data_loader, criterion, device):
    """Evaluate Luong model"""
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
# ATTENTION COMPARISON
# ============================================================================

def compare_attention_mechanisms(input_words, attention_weights_dict, save_path=None):
    """Compare different attention mechanisms side by side"""
    num_mechanisms = len(attention_weights_dict)
    fig, axes = plt.subplots(1, num_mechanisms, figsize=(5*num_mechanisms, 6))
    
    if num_mechanisms == 1:
        axes = [axes]
    
    for i, (mechanism_name, attention_weights) in enumerate(attention_weights_dict.items()):
        sns.heatmap(
            attention_weights,
            xticklabels=input_words,
            yticklabels=[f'step_{j+1}' for j in range(attention_weights.shape[0])],
            cmap='Blues',
            cbar=True,
            ax=axes[i]
        )
        axes[i].set_title(f'{mechanism_name} Attention')
        axes[i].set_xlabel('Input Sequence')
        axes[i].set_ylabel('Output Steps')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"=== Luong Attention Variants ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load dataset
    train_sentences, val_sentences, test_sentences = load_wikitext2_dataset()
    
    # Use subset for demonstration
    train_subset = train_sentences[:1200]
    val_subset = val_sentences[:240]
    test_subset = test_sentences[:240]
    
    # Build vocabulary
    vocab, idx_to_word = build_vocabulary(train_subset, vocab_size=3000)
    
    results = []
    training_histories = {}
    attention_comparisons = {}
    
    # Test different Luong attention variants
    attention_variants = [
        ('dot', False, 'Dot Product'),
        ('general', False, 'General (Multiplicative)'),
        ('concat', False, 'Concat (Additive-like)'),
        ('general', True, 'Local Attention')
    ]
    
    task = 'reverse'
    
    # Create datasets once
    train_dataset = LuongDataset(train_subset, vocab, task=task)
    val_dataset = LuongDataset(val_subset, vocab, task=task)
    test_dataset = LuongDataset(test_subset, vocab, task=task)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    for attention_type, use_local, description in attention_variants[:2]:  # Test first 2 for demo
        print(f"\n" + "="*50)
        print(f"Training Luong Seq2Seq with {description} Attention")
        
        # Initialize model
        model = LuongSeq2Seq(
            len(vocab), 
            embedding_dim=128, 
            hidden_dim=256, 
            num_layers=1,
            attention_type=attention_type,
            use_local_attention=use_local
        )
        
        # Train model
        model_name = f'Luong-{description.replace(" ", "")}'
        metrics = track_computational_metrics(
            model_name,
            train_model,
            model, train_loader, val_loader, 10, 0.001
        )
        
        train_losses, val_losses = metrics['result']
        training_histories[model_name] = (train_losses, val_losses)
        
        result = {
            'model': model_name,
            'attention_type': attention_type,
            'year': '2015',
            'final_loss': val_losses[-1] if val_losses else 0,
            'parameters': count_parameters(model),
            'training_time': metrics['training_time_minutes'],
            'memory_usage': metrics['memory_usage_mb'],
            'innovation': f'{description} scoring function'
        }
        results.append(result)
        
        # Test generation and attention
        print(f"\n{description} ATTENTION EXAMPLES:")
        print("="*40)
        
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Get one example for attention comparison
        if len(test_dataset) > 0:
            input_seq, target_seq = test_dataset[0]
            input_words = [idx_to_word[idx.item()] for idx in input_seq if idx.item() in idx_to_word and idx.item() != 0]
            
            input_tensor = input_seq.unsqueeze(0).to(device)
            input_lengths = torch.tensor([len(input_seq)]).to(device)
            
            generated, attention_weights = model.generate(
                input_tensor, input_lengths, vocab=vocab, idx_to_word=idx_to_word
            )
            
            print(f"  Input:     {' '.join(input_words)}")
            print(f"  Generated: {' '.join(generated)}")
            
            # Store attention for comparison
            if len(attention_weights) > 0:
                attention_matrix = attention_weights[:min(len(generated), 4), 0, :len(input_words)]
                attention_comparisons[description] = attention_matrix
    
    # Create attention comparison visualization
    if attention_comparisons:
        print(f"\nCreating attention mechanism comparison...")
        input_words_example = [idx_to_word[idx.item()] for idx in test_dataset[0][0] if idx.item() in idx_to_word and idx.item() != 0]
        
        compare_fig = compare_attention_mechanisms(
            input_words_example,
            attention_comparisons,
            'AI-ML-DL/Models/NLP/009_luong_attention_comparison.png'
        )
        plt.close(compare_fig)
        print("Attention comparison saved: 009_luong_attention_comparison.png")
    
    # Create training results visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training losses
    ax = axes[0, 0]
    for model_name, (train_losses, _) in training_histories.items():
        ax.plot(train_losses, label=model_name)
    ax.set_title('Training Loss Comparison')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Validation losses
    ax = axes[0, 1]
    for model_name, (_, val_losses) in training_histories.items():
        ax.plot(val_losses, label=model_name)
    ax.set_title('Validation Loss Comparison')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Final performance
    ax = axes[1, 0]
    models = [r['model'] for r in results]
    final_losses = [r['final_loss'] for r in results]
    ax.bar(models, final_losses)
    ax.set_title('Final Validation Loss')
    ax.set_ylabel('Loss')
    ax.tick_params(axis='x', rotation=45)
    
    # Training times
    ax = axes[1, 1]
    training_times = [r['training_time'] for r in results]
    ax.bar(models, training_times)
    ax.set_title('Training Time Comparison')
    ax.set_ylabel('Time (minutes)')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/NLP/009_luong_attention_results.png', dpi=300, bbox_inches='tight')
    print("\nTraining results saved: 009_luong_attention_results.png")
    
    # Print summary
    print("\n" + "="*60)
    print("LUONG ATTENTION VARIANTS SUMMARY")
    print("="*60)
    
    for result in results:
        print(f"\n{result['model']} ({result['year']}):")
        print(f"  Attention Type: {result['attention_type']}")
        print(f"  Final Loss: {result['final_loss']:.4f}")
        print(f"  Parameters: {result['parameters']:,}")
        print(f"  Training Time: {result['training_time']:.2f} minutes")
        print(f"  Innovation: {result['innovation']}")
    
    print("\nLuong Attention Scoring Functions:")
    print("- Dot: score(h_t, h_s) = h_t^T * h_s (simplest, requires same dimensions)")
    print("- General: score(h_t, h_s) = h_t^T * W_a * h_s (learnable transformation)")
    print("- Concat: score(h_t, h_s) = v_a^T * tanh(W_a * [h_t; h_s]) (like Bahdanau)")
    
    print("\nKey Luong Innovations:")
    print("- Multiple attention scoring functions")
    print("- Global vs Local attention mechanisms")
    print("- Input-feeding approach (use previous context)")
    print("- More computationally efficient than Bahdanau")
    print("- Simpler architecture with comparable performance")
    
    print("\nKey Insights:")
    print("- Different scoring functions have different computational costs")
    print("- General attention often performs best (learnable transformation)")
    print("- Local attention reduces computational complexity")
    print("- Input-feeding improves information flow")
    print("- Foundation for modern scaled dot-product attention")
    print("- Influenced Transformer attention mechanisms")
    
    return results

if __name__ == "__main__":
    results = main()