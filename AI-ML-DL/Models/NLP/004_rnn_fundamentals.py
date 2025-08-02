"""
ERA 2: RNN FUNDAMENTALS (2010-2015)
===================================

Year: 2010-2015
Innovation: Recurrent Neural Networks for sequence modeling
Previous Limitation: Feedforward networks couldn't handle variable-length sequences
Performance Gain: Can model sequences of arbitrary length
Impact: Enabled sequential processing but revealed vanishing gradient problem

This file demonstrates vanilla RNN implementation and the vanishing gradient problem
that motivated the development of LSTM and GRU architectures.
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
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# HISTORICAL CONTEXT & MOTIVATION
# ============================================================================

YEAR = "2010-2015"
INNOVATION = "Recurrent Neural Networks"
PREVIOUS_LIMITATION = "Feedforward networks required fixed-size inputs, couldn't model sequences"
IMPACT = "Enabled sequence modeling but revealed vanishing gradient problem"

print(f"=== {INNOVATION} ({YEAR}) ===")
print(f"Innovation: {INNOVATION}")
print(f"Previous Limitation: {PREVIOUS_LIMITATION}")
print(f"Impact: {IMPACT}")
print("="*60)

# ============================================================================
# STANDARDIZED DATASET LOADING
# ============================================================================

def load_wikitext2_dataset():
    """Load WikiText-2 dataset with consistent preprocessing for RNN training"""
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset('wikitext', 'wikitext-2-v1')
    
    def preprocess_text(texts):
        all_tokens = []
        for text in texts:
            if text.strip():
                text = text.lower().replace('\n', ' ')
                tokens = word_tokenize(text)
                if len(tokens) > 5:
                    all_tokens.extend(tokens)
        return all_tokens
    
    train_tokens = preprocess_text(dataset['train']['text'])
    val_tokens = preprocess_text(dataset['validation']['text'])
    test_tokens = preprocess_text(dataset['test']['text'])
    
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Validation tokens: {len(val_tokens):,}")
    print(f"Test tokens: {len(test_tokens):,}")
    
    return train_tokens, val_tokens, test_tokens

def build_vocabulary(tokens, vocab_size=5000):
    """Build vocabulary with most frequent words"""
    word_counts = Counter(tokens)
    most_common = word_counts.most_common(vocab_size - 2)
    
    vocab = {'<UNK>': 0, '<PAD>': 1}
    for word, _ in most_common:
        vocab[word] = len(vocab)
    
    idx_to_word = {idx: word for word, idx in vocab.items()}
    
    print(f"Vocabulary size: {len(vocab):,}")
    return vocab, idx_to_word

# ============================================================================
# DATASET CLASS FOR RNN TRAINING
# ============================================================================

class RNNLanguageModelDataset(Dataset):
    """Dataset for RNN language modeling with variable-length sequences"""
    
    def __init__(self, tokens, vocab, seq_length=50):
        self.seq_length = seq_length
        self.vocab = vocab
        
        # Convert tokens to indices
        self.data = []
        for token in tokens:
            self.data.append(vocab.get(token, vocab['<UNK>']))
        
        # Create sequences
        self.sequences = []
        for i in range(0, len(self.data) - seq_length, seq_length // 2):  # Overlapping sequences
            if i + seq_length + 1 < len(self.data):
                input_seq = self.data[i:i + seq_length]
                target_seq = self.data[i + 1:i + seq_length + 1]
                self.sequences.append((input_seq, target_seq))
        
        print(f"Created {len(self.sequences)} sequences of length {seq_length}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.sequences[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

# ============================================================================
# VANILLA RNN IMPLEMENTATION
# ============================================================================

class VanillaRNN(nn.Module):
    """
    Vanilla RNN implementation demonstrating the basic recurrent architecture
    and the vanishing gradient problem that motivated LSTM/GRU development
    """
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_layers=1, dropout=0.2):
        super(VanillaRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layer
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            nonlinearity='tanh'  # Classic RNN uses tanh
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability"""
        # Embedding initialization
        nn.init.uniform_(self.embeddings.weight, -0.1, 0.1)
        
        # RNN weight initialization (Xavier)
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        # Output layer initialization
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(self, input_sequence, hidden=None):
        """
        Forward pass of the RNN
        
        Args:
            input_sequence: Tensor of shape (batch_size, seq_length)
            hidden: Initial hidden state
            
        Returns:
            output: Tensor of shape (batch_size, seq_length, vocab_size)
            hidden: Final hidden state
        """
        batch_size, seq_length = input_sequence.size()
        
        # Get embeddings
        embeddings = self.embeddings(input_sequence)  # (batch_size, seq_length, embedding_dim)
        embeddings = self.dropout(embeddings)
        
        # RNN forward pass
        rnn_output, hidden = self.rnn(embeddings, hidden)  # (batch_size, seq_length, hidden_dim)
        
        # Apply dropout
        rnn_output = self.dropout(rnn_output)
        
        # Project to vocabulary size
        output = self.output_projection(rnn_output)  # (batch_size, seq_length, vocab_size)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
    
    def generate_text(self, vocab, idx_to_word, prompt="the", max_length=20, temperature=1.0):
        """Generate text using the trained RNN"""
        self.eval()
        device = next(self.parameters()).device
        
        # Convert prompt to tokens
        prompt_tokens = prompt.lower().split()
        input_ids = [vocab.get(token, vocab['<UNK>']) for token in prompt_tokens]
        
        if not input_ids:
            input_ids = [vocab.get('the', vocab['<UNK>'])]
        
        generated = prompt_tokens[:]
        hidden = self.init_hidden(1, device)
        
        with torch.no_grad():
            # Process prompt
            if len(input_ids) > 1:
                prompt_tensor = torch.tensor([input_ids[:-1]], device=device)
                _, hidden = self.forward(prompt_tensor, hidden)
            
            # Generate new tokens
            current_input = torch.tensor([[input_ids[-1]]], device=device)
            
            for _ in range(max_length):
                output, hidden = self.forward(current_input, hidden)
                
                # Apply temperature and sample
                logits = output[0, -1] / temperature
                probabilities = F.softmax(logits, dim=0)
                next_token_idx = torch.multinomial(probabilities, 1).item()
                
                # Convert back to word
                next_word = idx_to_word.get(next_token_idx, '<UNK>')
                if next_word in ['<PAD>', '<UNK>']:
                    break
                
                generated.append(next_word)
                current_input = torch.tensor([[next_token_idx]], device=device)
        
        return ' '.join(generated)

# ============================================================================
# IMPROVED RNN WITH DEEPER ARCHITECTURE
# ============================================================================

class DeepRNN(nn.Module):
    """
    Deeper RNN to demonstrate scaling challenges and vanishing gradients
    """
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_layers=3, dropout=0.3):
        super(DeepRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Deep RNN
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            nonlinearity='tanh'
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Careful initialization for deeper networks"""
        nn.init.uniform_(self.embeddings.weight, -0.1, 0.1)
        
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # Use smaller initialization for deeper networks
                nn.init.orthogonal_(param.data, gain=0.5)
            elif 'bias' in name:
                param.data.fill_(0)
        
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(self, input_sequence, hidden=None):
        batch_size, seq_length = input_sequence.size()
        
        embeddings = self.embeddings(input_sequence)
        embeddings = self.dropout(embeddings)
        
        rnn_output, hidden = self.rnn(embeddings, hidden)
        rnn_output = self.dropout(rnn_output)
        
        output = self.output_projection(rnn_output)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)

# ============================================================================
# GRADIENT ANALYSIS TOOLS
# ============================================================================

def analyze_gradients(model, data_loader, criterion, device):
    """Analyze gradient magnitudes to demonstrate vanishing gradient problem"""
    model.train()
    gradient_norms = []
    
    # Sample a few batches for analysis
    for batch_idx, (input_seq, target_seq) in enumerate(data_loader):
        if batch_idx >= 5:  # Analyze first 5 batches
            break
        
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        batch_size = input_seq.size(0)
        
        # Initialize hidden state
        hidden = model.init_hidden(batch_size, device)
        
        # Forward pass
        output, _ = model(input_seq, hidden)
        
        # Compute loss
        loss = criterion(output.reshape(-1, output.size(-1)), target_seq.reshape(-1))
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Collect gradient norms for each layer
        batch_gradient_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None and 'rnn' in name:
                grad_norm = param.grad.data.norm().item()
                batch_gradient_norms.append(grad_norm)
        
        gradient_norms.append(batch_gradient_norms)
    
    return gradient_norms

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_model(model, train_loader, val_loader, epochs=10, learning_rate=0.001, clip_grad=1.0):
    """Train RNN with gradient clipping to handle exploding gradients"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_perplexities = []
    gradient_norms_history = []
    
    print(f"Training on device: {device}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        
        for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            batch_size = input_seq.size(0)
            
            # Initialize hidden state
            hidden = model.init_hidden(batch_size, device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output, _ = model(input_seq, hidden)
            
            # Compute loss
            loss = criterion(output.reshape(-1, output.size(-1)), target_seq.reshape(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_perplexity = evaluate_model(model, val_loader, criterion, device)
        val_perplexities.append(val_perplexity)
        
        # Analyze gradients every few epochs
        if epoch % 2 == 0:
            grad_norms = analyze_gradients(model, train_loader, criterion, device)
            gradient_norms_history.append(grad_norms)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Perplexity: {val_perplexity:.2f}")
    
    return train_losses, val_perplexities, gradient_norms_history

def evaluate_model(model, data_loader, criterion, device):
    """Evaluate model and return perplexity"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for input_seq, target_seq in data_loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            batch_size = input_seq.size(0)
            
            hidden = model.init_hidden(batch_size, device)
            output, _ = model(input_seq, hidden)
            
            loss = criterion(output.reshape(-1, output.size(-1)), target_seq.reshape(-1))
            
            total_loss += loss.item() * target_seq.numel()
            total_tokens += target_seq.numel()
    
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
# VISUALIZATION
# ============================================================================

def create_visualizations(results, training_histories, gradient_histories):
    """Create comprehensive visualizations including gradient analysis"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Training loss curves
    ax = axes[0, 0]
    for model_name, (train_losses, _, _) in training_histories.items():
        ax.plot(train_losses, label=model_name)
    ax.set_title('Training Loss Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Validation perplexity curves
    ax = axes[0, 1]
    for model_name, (_, val_perplexities, _) in training_histories.items():
        ax.plot(val_perplexities, label=model_name)
    ax.set_title('Validation Perplexity')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Perplexity')
    ax.legend()
    ax.grid(True)
    
    # Final perplexity comparison
    ax = axes[0, 2]
    models = [r['model'] for r in results]
    perplexities = [r['perplexity'] for r in results]
    ax.bar(models, perplexities)
    ax.set_title('Final Perplexity Comparison')
    ax.set_ylabel('Perplexity')
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
    
    # Memory usage comparison
    ax = axes[1, 2]
    memory_usage = [r['memory_usage'] for r in results]
    ax.bar(models, memory_usage)
    ax.set_title('Memory Usage')
    ax.set_ylabel('Memory (MB)')
    ax.tick_params(axis='x', rotation=45)
    
    # Gradient norm analysis (demonstrate vanishing gradients)
    ax = axes[2, 0]
    if gradient_histories:
        for model_name, grad_history in gradient_histories.items():
            if grad_history:
                # Average gradient norms across layers and batches
                avg_grad_norms = []
                for epoch_grads in grad_history:
                    if epoch_grads:
                        epoch_avg = np.mean([np.mean(batch_grads) for batch_grads in epoch_grads])
                        avg_grad_norms.append(epoch_avg)
                
                epochs = range(0, len(grad_history) * 2, 2)  # Every 2 epochs
                ax.plot(epochs[:len(avg_grad_norms)], avg_grad_norms, label=model_name, marker='o')
        
        ax.set_title('Gradient Magnitude Over Training')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Gradient Norm')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True)
    
    # Model architecture comparison
    ax = axes[2, 1]
    hidden_dims = [getattr(r, 'hidden_dim', 128) for r in [128, 128]]  # Default values
    num_layers = [getattr(r, 'num_layers', 1) for r in [1, 3]]  # Default values
    ax.scatter(num_layers, hidden_dims, s=[p/1000 for p in parameters], alpha=0.7)
    for i, model in enumerate(models):
        ax.annotate(model, (num_layers[i], hidden_dims[i]), fontsize=8)
    ax.set_title('Architecture Comparison')
    ax.set_xlabel('Number of Layers')
    ax.set_ylabel('Hidden Dimension')
    ax.grid(True)
    
    # Performance vs complexity
    ax = axes[2, 2]
    ax.scatter(parameters, perplexities, alpha=0.7)
    for i, model in enumerate(models):
        ax.annotate(model, (parameters[i], perplexities[i]), fontsize=8)
    ax.set_title('Performance vs Model Complexity')
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Perplexity')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/NLP/004_rnn_fundamentals_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved: 004_rnn_fundamentals_results.png")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"=== RNN Fundamentals ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load dataset
    train_tokens, val_tokens, test_tokens = load_wikitext2_dataset()
    
    # Use subset for demonstration
    train_subset = train_tokens[:15000]
    val_subset = val_tokens[:3000]
    test_subset = test_tokens[:3000]
    
    # Build vocabulary
    vocab, idx_to_word = build_vocabulary(train_subset, vocab_size=3000)
    
    # Create datasets
    seq_length = 50
    train_dataset = RNNLanguageModelDataset(train_subset, vocab, seq_length)
    val_dataset = RNNLanguageModelDataset(val_subset, vocab, seq_length)
    test_dataset = RNNLanguageModelDataset(test_subset, vocab, seq_length)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    results = []
    training_histories = {}
    gradient_histories = {}
    
    # Train Vanilla RNN
    print("\n" + "="*50)
    print("Training Vanilla RNN")
    
    model1 = VanillaRNN(len(vocab), embedding_dim=100, hidden_dim=128, num_layers=1)
    
    metrics = track_computational_metrics(
        'Vanilla-RNN',
        train_model,
        model1, train_loader, val_loader, 8, 0.001, 1.0
    )
    
    train_losses, val_perplexities, grad_norms = metrics['result']
    training_histories['Vanilla-RNN'] = (train_losses, val_perplexities, grad_norms)
    gradient_histories['Vanilla-RNN'] = grad_norms
    
    # Final evaluation
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final_perplexity = evaluate_model(model1, test_loader, criterion, device)
    
    result1 = {
        'model': 'Vanilla-RNN',
        'year': '2010',
        'perplexity': final_perplexity,
        'parameters': count_parameters(model1),
        'training_time': metrics['training_time_minutes'],
        'memory_usage': metrics['memory_usage_mb'],
        'innovation': 'First successful sequence modeling architecture'
    }
    results.append(result1)
    
    # Train Deep RNN
    print("\n" + "="*50)
    print("Training Deep RNN (3 layers)")
    
    model2 = DeepRNN(len(vocab), embedding_dim=100, hidden_dim=128, num_layers=3)
    
    metrics = track_computational_metrics(
        'Deep-RNN',
        train_model,
        model2, train_loader, val_loader, 8, 0.001, 0.5  # Lower gradient clipping
    )
    
    train_losses, val_perplexities, grad_norms = metrics['result']
    training_histories['Deep-RNN'] = (train_losses, val_perplexities, grad_norms)
    gradient_histories['Deep-RNN'] = grad_norms
    
    final_perplexity = evaluate_model(model2, test_loader, criterion, device)
    
    result2 = {
        'model': 'Deep-RNN',
        'year': '2015',
        'perplexity': final_perplexity,
        'parameters': count_parameters(model2),
        'training_time': metrics['training_time_minutes'],
        'memory_usage': metrics['memory_usage_mb'],
        'innovation': 'Deeper networks reveal vanishing gradient problem'
    }
    results.append(result2)
    
    # Generate sample text
    print("\n" + "="*50)
    print("TEXT GENERATION EXAMPLES")
    print("="*50)
    
    test_prompts = ["the quick brown", "in the beginning", "scientists have discovered"]
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        
        gen1 = model1.generate_text(vocab, idx_to_word, prompt, max_length=15)
        print(f"  Vanilla-RNN: {gen1}")
        
        gen2 = model2.generate_text(vocab, idx_to_word, prompt, max_length=15)
        print(f"  Deep-RNN: {gen2}")
    
    # Create visualizations
    create_visualizations(results, training_histories, gradient_histories)
    
    # Print summary
    print("\n" + "="*60)
    print("RNN FUNDAMENTALS SUMMARY")
    print("="*60)
    
    for result in results:
        print(f"\n{result['model']} ({result['year']}):")
        print(f"  Perplexity: {result['perplexity']:.2f}")
        print(f"  Parameters: {result['parameters']:,}")
        print(f"  Training Time: {result['training_time']:.2f} minutes")
        print(f"  Memory Usage: {result['memory_usage']:.2f} MB")
        print(f"  Innovation: {result['innovation']}")
    
    print("\nKey Insights:")
    print("- RNNs enable variable-length sequence modeling")
    print("- Deeper RNN networks suffer from vanishing gradients")
    print("- Gradient clipping helps with exploding gradients")
    print("- Tanh activation compounds gradient vanishing problem")
    print("- Need for better architectures (LSTM/GRU) becomes clear")
    print("- Sequential processing limits parallelization")
    
    return results

if __name__ == "__main__":
    results = main()