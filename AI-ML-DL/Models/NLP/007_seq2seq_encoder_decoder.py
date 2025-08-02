"""
ERA 3: SEQUENCE-TO-SEQUENCE ENCODER-DECODER (2014)
==================================================

Year: 2014
Innovation: Encoder-Decoder architecture for sequence-to-sequence tasks
Previous Limitation: RNN/LSTM could only handle fixed input-output relationships
Performance Gain: Enabled variable-length input to variable-length output mapping
Impact: Breakthrough for machine translation, summarization, and generation tasks

This file demonstrates the seq2seq architecture that revolutionized NLP by enabling
tasks like translation where input and output sequences have different lengths.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
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

YEAR = "2014"
INNOVATION = "Sequence-to-Sequence (Seq2Seq) Models"
PREVIOUS_LIMITATION = "RNN/LSTM required fixed input-output relationships"
IMPACT = "Enabled variable-length sequence transformations (translation, summarization)"

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
                # Split into sentences for seq2seq tasks
                text_sentences = text.split('.')
                for sentence in text_sentences:
                    tokens = word_tokenize(sentence.strip())
                    if 5 <= len(tokens) <= 20:  # Good length for seq2seq
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
    """Build vocabulary with special tokens for seq2seq"""
    word_counts = Counter()
    for sentence in sentences:
        word_counts.update(sentence)
    
    most_common = word_counts.most_common(vocab_size - 4)  # Reserve space for special tokens
    
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1, 
        '<SOS>': 2,  # Start of sequence
        '<EOS>': 3   # End of sequence
    }
    
    for word, _ in most_common:
        vocab[word] = len(vocab)
    
    idx_to_word = {idx: word for word, idx in vocab.items()}
    
    print(f"Vocabulary size: {len(vocab):,}")
    return vocab, idx_to_word

# ============================================================================
# DATASET CLASS FOR SEQ2SEQ TRAINING
# ============================================================================

class Seq2SeqDataset(Dataset):
    """
    Dataset for sequence-to-sequence training
    Creates various seq2seq tasks from the same corpus
    """
    
    def __init__(self, sentences, vocab, task='reverse', max_length=15):
        self.vocab = vocab
        self.task = task
        self.max_length = max_length
        
        self.pairs = []
        
        if task == 'reverse':
            # Task: Reverse the input sequence
            for sentence in sentences:
                if len(sentence) <= max_length:
                    input_seq = sentence
                    target_seq = sentence[::-1]  # Reverse
                    self.pairs.append((input_seq, target_seq))
        
        elif task == 'copy':
            # Task: Copy the input sequence (identity function)
            for sentence in sentences:
                if len(sentence) <= max_length:
                    input_seq = sentence
                    target_seq = sentence[:]  # Copy
                    self.pairs.append((input_seq, target_seq))
        
        elif task == 'summarize':
            # Task: Generate first half as summary of whole sentence
            for sentence in sentences:
                if len(sentence) >= 8:
                    input_seq = sentence
                    target_seq = sentence[:len(sentence)//2]  # First half as summary
                    self.pairs.append((input_seq, target_seq))
        
        elif task == 'language_modeling':
            # Task: Predict next words (seq2seq style)
            for sentence in sentences:
                if len(sentence) >= 4:
                    split_point = len(sentence) // 2
                    input_seq = sentence[:split_point]
                    target_seq = sentence[split_point:]
                    self.pairs.append((input_seq, target_seq))
        
        print(f"Created {len(self.pairs)} {task} pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.pairs[idx]
        
        # Convert to indices and add special tokens
        input_indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in input_seq]
        target_indices = [self.vocab['<SOS>']] + [self.vocab.get(word, self.vocab['<UNK>']) for word in target_seq] + [self.vocab['<EOS>']]
        
        return torch.tensor(input_indices, dtype=torch.long), torch.tensor(target_indices, dtype=torch.long)

def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    input_sequences, target_sequences = zip(*batch)
    
    # Pad sequences
    input_lengths = [len(seq) for seq in input_sequences]
    target_lengths = [len(seq) for seq in target_sequences]
    
    max_input_len = max(input_lengths)
    max_target_len = max(target_lengths)
    
    padded_inputs = []
    padded_targets = []
    
    for input_seq, target_seq in zip(input_sequences, target_sequences):
        # Pad input
        pad_length = max_input_len - len(input_seq)
        padded_input = torch.cat([input_seq, torch.zeros(pad_length, dtype=torch.long)])
        padded_inputs.append(padded_input)
        
        # Pad target
        pad_length = max_target_len - len(target_seq)
        padded_target = torch.cat([target_seq, torch.zeros(pad_length, dtype=torch.long)])
        padded_targets.append(padded_target)
    
    return (torch.stack(padded_inputs), torch.tensor(input_lengths), 
            torch.stack(padded_targets), torch.tensor(target_lengths))

# ============================================================================
# ENCODER IMPLEMENTATION
# ============================================================================

class Encoder(nn.Module):
    """
    Encoder part of the seq2seq model
    Encodes variable-length input sequences into fixed-size context vector
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super(Encoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_sequence, input_lengths):
        """
        Encode input sequence
        
        Args:
            input_sequence: (batch_size, max_seq_length)
            input_lengths: (batch_size,) actual lengths
            
        Returns:
            outputs: All hidden states (batch_size, max_seq_length, hidden_dim)
            hidden: Final hidden state tuple (h_n, c_n)
        """
        embedded = self.embedding(input_sequence)
        embedded = self.dropout(embedded)
        
        # Pack padded sequence for efficiency
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, input_lengths, batch_first=True, enforce_sorted=False
        )
        
        # LSTM forward pass
        packed_outputs, hidden = self.lstm(packed_embedded)
        
        # Unpack sequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        
        return outputs, hidden

# ============================================================================
# DECODER IMPLEMENTATION
# ============================================================================

class Decoder(nn.Module):
    """
    Decoder part of the seq2seq model
    Generates variable-length output sequences from encoder's context vector
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super(Decoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input_token, hidden):
        """
        Single step decoding
        
        Args:
            input_token: (batch_size, 1) current input token
            hidden: Hidden state from previous step
            
        Returns:
            output: (batch_size, 1, vocab_size) output probabilities
            hidden: Updated hidden state
        """
        embedded = self.embedding(input_token)
        embedded = self.dropout(embedded)
        
        lstm_output, hidden = self.lstm(embedded, hidden)
        lstm_output = self.dropout(lstm_output)
        
        output = self.output_projection(lstm_output)
        
        return output, hidden
    
    def forward_sequence(self, target_sequence, hidden):
        """
        Teacher forcing: decode entire sequence at once
        
        Args:
            target_sequence: (batch_size, max_target_length)
            hidden: Initial hidden state from encoder
            
        Returns:
            outputs: (batch_size, max_target_length, vocab_size)
        """
        embedded = self.embedding(target_sequence)
        embedded = self.dropout(embedded)
        
        lstm_output, _ = self.lstm(embedded, hidden)
        lstm_output = self.dropout(lstm_output)
        
        outputs = self.output_projection(lstm_output)
        
        return outputs

# ============================================================================
# SEQ2SEQ MODEL
# ============================================================================

class Seq2SeqModel(nn.Module):
    """
    Complete Sequence-to-Sequence model
    Combines encoder and decoder for variable-length sequence transformations
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super(Seq2SeqModel, self).__init__()
        
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.vocab_size = vocab_size
    
    def forward(self, input_sequence, input_lengths, target_sequence, teacher_forcing_ratio=0.5):
        """
        Training forward pass with teacher forcing
        
        Args:
            input_sequence: (batch_size, input_length)
            input_lengths: (batch_size,)
            target_sequence: (batch_size, target_length)
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            outputs: (batch_size, target_length-1, vocab_size)
        """
        batch_size = input_sequence.size(0)
        target_length = target_sequence.size(1)
        
        # Encode input sequence
        encoder_outputs, encoder_hidden = self.encoder(input_sequence, input_lengths)
        
        # Initialize decoder
        decoder_hidden = encoder_hidden
        decoder_input = target_sequence[:, 0:1]  # <SOS> token
        
        outputs = []
        
        # Decode sequence
        for t in range(1, target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs.append(decoder_output)
            
            # Teacher forcing decision
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            
            if use_teacher_forcing:
                decoder_input = target_sequence[:, t:t+1]  # Use ground truth
            else:
                decoder_input = decoder_output.argmax(dim=-1)  # Use prediction
        
        outputs = torch.cat(outputs, dim=1)
        return outputs
    
    def encode(self, input_sequence, input_lengths):
        """Encode input sequence and return context"""
        encoder_outputs, encoder_hidden = self.encoder(input_sequence, input_lengths)
        return encoder_outputs, encoder_hidden
    
    def decode_step(self, input_token, hidden):
        """Single decoding step"""
        return self.decoder(input_token, hidden)
    
    def generate(self, input_sequence, input_lengths, max_length=20, vocab=None, idx_to_word=None):
        """
        Generate output sequence given input
        
        Args:
            input_sequence: (1, input_length) single input sequence
            input_lengths: (1,) length of input
            max_length: Maximum generation length
            
        Returns:
            generated_sequence: List of generated tokens
        """
        self.eval()
        device = input_sequence.device
        
        with torch.no_grad():
            # Encode input
            encoder_outputs, encoder_hidden = self.encoder(input_sequence, input_lengths)
            
            # Initialize decoder
            decoder_hidden = encoder_hidden
            decoder_input = torch.tensor([[vocab['<SOS>']]], device=device)
            
            generated = []
            
            for _ in range(max_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                
                # Get most likely next token
                next_token_idx = decoder_output.argmax(dim=-1).item()
                
                if next_token_idx == vocab['<EOS>']:
                    break
                
                if next_token_idx in idx_to_word:
                    generated.append(idx_to_word[next_token_idx])
                
                decoder_input = torch.tensor([[next_token_idx]], device=device)
        
        return generated

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_model(model, train_loader, val_loader, epochs=15, learning_rate=0.001):
    """Train seq2seq model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
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
            outputs = model(input_seq, input_lengths, target_seq, teacher_forcing_ratio=0.5)
            
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
    """Evaluate seq2seq model"""
    model.eval()
    total_loss = 0
    total_batches = 0
    
    with torch.no_grad():
        for input_seq, input_lengths, target_seq, target_lengths in data_loader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            outputs = model(input_seq, input_lengths, target_seq, teacher_forcing_ratio=0.0)
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
# SEQUENCE EVALUATION METRICS
# ============================================================================

def calculate_bleu_score(reference, candidate):
    """Simple BLEU score calculation"""
    if not candidate or not reference:
        return 0.0
    
    # 1-gram precision
    ref_words = set(reference)
    cand_words = set(candidate)
    
    if len(cand_words) == 0:
        return 0.0
    
    precision = len(ref_words.intersection(cand_words)) / len(cand_words)
    
    # Length penalty
    length_penalty = min(1.0, len(candidate) / len(reference)) if reference else 0.0
    
    return precision * length_penalty

def evaluate_generation_quality(model, test_dataset, vocab, idx_to_word, device, num_samples=20):
    """Evaluate generation quality using BLEU scores"""
    model.eval()
    total_bleu = 0
    
    with torch.no_grad():
        for i in range(min(num_samples, len(test_dataset))):
            input_seq, target_seq = test_dataset[i]
            
            # Prepare input
            input_tensor = input_seq.unsqueeze(0).to(device)
            input_lengths = torch.tensor([len(input_seq)]).to(device)
            
            # Generate
            generated = model.generate(input_tensor, input_lengths, vocab=vocab, idx_to_word=idx_to_word)
            
            # Convert target to words
            target_words = []
            for idx in target_seq[1:-1]:  # Skip SOS and EOS
                if idx.item() in idx_to_word and idx.item() != 0:  # Skip padding
                    target_words.append(idx_to_word[idx.item()])
            
            # Calculate BLEU
            bleu = calculate_bleu_score(target_words, generated)
            total_bleu += bleu
    
    return total_bleu / num_samples if num_samples > 0 else 0.0

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(results, training_histories):
    """Create comprehensive visualizations for seq2seq analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
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
    
    # Final loss comparison
    ax = axes[0, 2]
    models = [r['model'] for r in results]
    final_losses = [r.get('final_loss', 0) for r in results]
    ax.bar(models, final_losses)
    ax.set_title('Final Validation Loss')
    ax.set_ylabel('Loss')
    ax.tick_params(axis='x', rotation=45)
    
    # Parameter count comparison
    ax = axes[1, 0]
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
    
    # BLEU score comparison
    ax = axes[1, 2]
    bleu_scores = [r.get('bleu_score', 0) for r in results]
    ax.bar(models, bleu_scores)
    ax.set_title('Generation Quality (BLEU Score)')
    ax.set_ylabel('BLEU Score')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/NLP/007_seq2seq_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved: 007_seq2seq_results.png")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"=== Seq2Seq Encoder-Decoder ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load dataset
    train_sentences, val_sentences, test_sentences = load_wikitext2_dataset()
    
    # Use subset for demonstration
    train_subset = train_sentences[:2000]
    val_subset = val_sentences[:400]
    test_subset = test_sentences[:400]
    
    # Build vocabulary
    vocab, idx_to_word = build_vocabulary(train_subset, vocab_size=3000)
    
    results = []
    training_histories = {}
    
    # Test different seq2seq tasks
    tasks = ['reverse', 'copy']  # Focus on clear tasks for demonstration
    
    for task in tasks:
        print(f"\n" + "="*50)
        print(f"Training Seq2Seq for {task.upper()} task")
        
        # Create datasets for this task
        train_dataset = Seq2SeqDataset(train_subset, vocab, task=task)
        val_dataset = Seq2SeqDataset(val_subset, vocab, task=task)
        test_dataset = Seq2SeqDataset(test_subset, vocab, task=task)
        
        # Create data loaders
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        # Initialize model
        model = Seq2SeqModel(len(vocab), embedding_dim=128, hidden_dim=256, num_layers=2)
        
        # Train model
        metrics = track_computational_metrics(
            f'Seq2Seq-{task}',
            train_model,
            model, train_loader, val_loader, 10, 0.001  # Reduced epochs for demo
        )
        
        train_losses, val_losses = metrics['result']
        training_histories[f'Seq2Seq-{task}'] = (train_losses, val_losses)
        
        # Evaluate generation quality
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bleu_score = evaluate_generation_quality(model, test_dataset, vocab, idx_to_word, device)
        
        result = {
            'model': f'Seq2Seq-{task}',
            'task': task,
            'year': '2014',
            'final_loss': val_losses[-1] if val_losses else 0,
            'bleu_score': bleu_score,
            'parameters': count_parameters(model),
            'training_time': metrics['training_time_minutes'],
            'memory_usage': metrics['memory_usage_mb'],
            'innovation': f'Encoder-decoder for {task} transformation'
        }
        results.append(result)
        
        # Demonstrate generation
        print(f"\n{task.upper()} TASK EXAMPLES:")
        print("="*30)
        
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        for i in range(3):  # Show 3 examples
            if i < len(test_dataset):
                input_seq, target_seq = test_dataset[i]
                
                # Convert input to words
                input_words = [idx_to_word[idx.item()] for idx in input_seq if idx.item() in idx_to_word and idx.item() != 0]
                
                # Convert target to words
                target_words = [idx_to_word[idx.item()] for idx in target_seq[1:-1] if idx.item() in idx_to_word and idx.item() != 0]
                
                # Generate prediction
                input_tensor = input_seq.unsqueeze(0).to(device)
                input_lengths = torch.tensor([len(input_seq)]).to(device)
                generated = model.generate(input_tensor, input_lengths, vocab=vocab, idx_to_word=idx_to_word)
                
                print(f"  Input:  {' '.join(input_words)}")
                print(f"  Target: {' '.join(target_words)}")
                print(f"  Output: {' '.join(generated)}")
                print()
    
    # Create visualizations
    create_visualizations(results, training_histories)
    
    # Print summary
    print("\n" + "="*60)
    print("SEQ2SEQ ENCODER-DECODER SUMMARY")
    print("="*60)
    
    for result in results:
        print(f"\n{result['model']} ({result['year']}):")
        print(f"  Task: {result['task']}")
        print(f"  Final Loss: {result['final_loss']:.4f}")
        print(f"  BLEU Score: {result['bleu_score']:.4f}")
        print(f"  Parameters: {result['parameters']:,}")
        print(f"  Training Time: {result['training_time']:.2f} minutes")
        print(f"  Memory Usage: {result['memory_usage']:.2f} MB")
        print(f"  Innovation: {result['innovation']}")
    
    print("\nSeq2Seq Architecture Components:")
    print("- Encoder: RNN/LSTM that processes input sequence")
    print("- Context Vector: Fixed-size representation of input")
    print("- Decoder: RNN/LSTM that generates output sequence")
    print("- Teacher Forcing: Training technique using ground truth")
    
    print("\nKey Insights:")
    print("- Enables variable input → variable output mapping")
    print("- Breakthrough for machine translation and generation")
    print("- Information bottleneck: entire input compressed to fixed vector")
    print("- Teacher forcing improves training stability")
    print("- Foundation for attention mechanisms (next breakthrough)")
    print("- Encoder-decoder paradigm still used in modern transformers")
    
    return results

if __name__ == "__main__":
    results = main()