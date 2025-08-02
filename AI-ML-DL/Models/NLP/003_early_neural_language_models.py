"""
ERA 1: EARLY NEURAL LANGUAGE MODELS (2003-2010)
===============================================

Year: 2003-2010
Innovation: Neural networks for language modeling
Previous Limitation: Statistical models couldn't capture long-range dependencies
Performance Gain: Better perplexity through learned representations
Impact: Bridge between statistical and modern neural NLP

This file demonstrates the transition from statistical to neural language modeling,
implementing feedforward neural language models that preceded RNN-based approaches.
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

YEAR = "2003-2010"
INNOVATION = "Neural Language Models"
PREVIOUS_LIMITATION = "Statistical models couldn't learn complex patterns or long-range dependencies"
IMPACT = "First neural approach to language modeling, enabling learned representations"

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
        all_tokens = []
        for text in texts:
            if text.strip():
                text = text.lower().replace('\n', ' ')
                tokens = word_tokenize(text)
                if len(tokens) > 3:
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
    most_common = word_counts.most_common(vocab_size - 2)  # Reserve space for special tokens
    
    vocab = {'<UNK>': 0, '<PAD>': 1}
    for word, _ in most_common:
        vocab[word] = len(vocab)
    
    # Create reverse mapping
    idx_to_word = {idx: word for word, idx in vocab.items()}
    
    print(f"Vocabulary size: {len(vocab):,}")
    return vocab, idx_to_word

# ============================================================================
# DATASET CLASS FOR NEURAL TRAINING
# ============================================================================

class LanguageModelDataset(Dataset):
    """Dataset for neural language modeling with fixed context windows"""
    
    def __init__(self, tokens, vocab, context_size=5):
        self.context_size = context_size
        self.vocab = vocab
        
        # Convert tokens to indices
        self.data = []
        for token in tokens:
            self.data.append(vocab.get(token, vocab['<UNK>']))
        
        # Create training examples (context -> target)
        self.examples = []
        for i in range(len(self.data) - context_size):
            context = self.data[i:i + context_size]
            target = self.data[i + context_size]
            self.examples.append((context, target))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        context, target = self.examples[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

# ============================================================================
# FEEDFORWARD NEURAL LANGUAGE MODEL (BENGIO ET AL. 2003)
# ============================================================================

class FeedforwardNeuralLM(nn.Module):
    """
    Feedforward Neural Language Model (Bengio et al., 2003)
    
    This was the first successful neural language model that outperformed
    n-gram models by learning distributed representations of words.
    """
    
    def __init__(self, vocab_size, embedding_dim=50, hidden_dim=100, context_size=5):
        super(FeedforwardNeuralLM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_size = context_size
        
        # Word embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Neural network layers
        self.hidden = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following the original paper"""
        # Embedding initialization
        nn.init.uniform_(self.embeddings.weight, -0.1, 0.1)
        
        # Hidden layer initialization
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        
        # Output layer initialization
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)
    
    def forward(self, context):
        """
        Forward pass of the neural language model
        
        Args:
            context: Tensor of shape (batch_size, context_size)
            
        Returns:
            logits: Tensor of shape (batch_size, vocab_size)
        """
        batch_size = context.size(0)
        
        # Get embeddings for context words
        embeddings = self.embeddings(context)  # (batch_size, context_size, embedding_dim)
        
        # Concatenate embeddings
        concat_embeddings = embeddings.view(batch_size, -1)  # (batch_size, context_size * embedding_dim)
        
        # Pass through hidden layer with tanh activation (as in original paper)
        hidden_output = torch.tanh(self.hidden(concat_embeddings))
        
        # Output layer
        logits = self.output(hidden_output)
        
        return logits
    
    def generate_text(self, vocab, idx_to_word, prompt="the", max_length=20):
        """Generate text using the trained model"""
        self.eval()
        
        # Convert prompt to indices
        prompt_tokens = prompt.lower().split()
        context = [vocab.get(token, vocab['<UNK>']) for token in prompt_tokens]
        
        # Pad or truncate to context_size
        if len(context) < self.context_size:
            context = [vocab['<PAD>']] * (self.context_size - len(context)) + context
        else:
            context = context[-self.context_size:]
        
        generated = prompt_tokens[:]
        
        with torch.no_grad():
            for _ in range(max_length):
                # Convert context to tensor
                context_tensor = torch.tensor([context], dtype=torch.long)
                
                # Get predictions
                logits = self.forward(context_tensor)
                probabilities = F.softmax(logits, dim=1)
                
                # Sample next word (with temperature for diversity)
                temperature = 1.0
                scaled_logits = logits / temperature
                next_word_idx = torch.multinomial(F.softmax(scaled_logits, dim=1), 1).item()
                
                # Convert back to word
                next_word = idx_to_word.get(next_word_idx, '<UNK>')
                if next_word in ['<PAD>', '<UNK>']:
                    break
                
                generated.append(next_word)
                
                # Update context (sliding window)
                context = context[1:] + [next_word_idx]
        
        return ' '.join(generated)

# ============================================================================
# IMPROVED NEURAL LANGUAGE MODEL WITH REGULARIZATION
# ============================================================================

class ImprovedNeuralLM(nn.Module):
    """
    Improved neural language model with modern techniques
    Shows evolution from basic feedforward to more sophisticated architectures
    """
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=200, context_size=5, dropout=0.2):
        super(ImprovedNeuralLM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_size = context_size
        
        # Embeddings with larger dimension
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Multiple hidden layers with dropout
        self.hidden1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Modern weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.uniform_(module.weight, -0.1, 0.1)
    
    def forward(self, context):
        batch_size = context.size(0)
        
        # Embeddings
        embeddings = self.embeddings(context)
        concat_embeddings = embeddings.view(batch_size, -1)
        
        # First hidden layer
        hidden1_output = F.relu(self.hidden1(concat_embeddings))
        hidden1_output = self.dropout(hidden1_output)
        
        # Second hidden layer
        hidden2_output = F.relu(self.hidden2(hidden1_output))
        hidden2_output = self.dropout(hidden2_output)
        
        # Output
        logits = self.output(hidden2_output)
        
        return logits

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_model(model, train_loader, val_loader, epochs=10, learning_rate=0.001):
    """Train neural language model with consistent evaluation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_perplexities = []
    
    print(f"Training on device: {device}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        
        for batch_idx, (context, target) in enumerate(train_loader):
            context, target = context.to(device), target.to(device)
            
            optimizer.zero_grad()
            logits = model(context)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_perplexity = evaluate_model(model, val_loader, criterion, device)
        val_perplexities.append(val_perplexity)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Perplexity: {val_perplexity:.2f}")
    
    return train_losses, val_perplexities

def evaluate_model(model, data_loader, criterion, device):
    """Evaluate model and return perplexity"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for context, target in data_loader:
            context, target = context.to(device), target.to(device)
            logits = model(context)
            loss = criterion(logits, target)
            
            total_loss += loss.item() * target.size(0)
            total_tokens += target.size(0)
    
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

def create_visualizations(results, training_histories):
    """Create comprehensive visualizations"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Training loss curves
    ax = axes[0, 0]
    for model_name, (train_losses, _) in training_histories.items():
        ax.plot(train_losses, label=model_name)
    ax.set_title('Training Loss Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Validation perplexity curves
    ax = axes[0, 1]
    for model_name, (_, val_perplexities) in training_histories.items():
        ax.plot(val_perplexities, label=model_name)
    ax.set_title('Validation Perplexity')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Perplexity')
    ax.legend()
    ax.grid(True)
    
    # Perplexity comparison
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
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/NLP/003_early_neural_lm_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved: 003_early_neural_lm_results.png")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"=== Early Neural Language Models ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load dataset
    train_tokens, val_tokens, test_tokens = load_wikitext2_dataset()
    
    # Use subset for demonstration
    train_subset = train_tokens[:10000]
    val_subset = val_tokens[:2000]
    test_subset = test_tokens[:2000]
    
    # Build vocabulary
    vocab, idx_to_word = build_vocabulary(train_subset, vocab_size=2000)
    
    # Create datasets
    context_size = 5
    train_dataset = LanguageModelDataset(train_subset, vocab, context_size)
    val_dataset = LanguageModelDataset(val_subset, vocab, context_size)
    test_dataset = LanguageModelDataset(test_subset, vocab, context_size)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    results = []
    training_histories = {}
    
    # Train Feedforward Neural LM (2003)
    print("\n" + "="*50)
    print("Training Feedforward Neural Language Model (Bengio et al., 2003)")
    
    model1 = FeedforwardNeuralLM(len(vocab), embedding_dim=50, hidden_dim=100, context_size=context_size)
    
    metrics = track_computational_metrics(
        'Feedforward-NLM',
        train_model,
        model1, train_loader, val_loader, 5, 0.001  # Reduced epochs for demo
    )
    
    train_losses, val_perplexities = metrics['result']
    training_histories['Feedforward-NLM'] = (train_losses, val_perplexities)
    
    # Final evaluation
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final_perplexity = evaluate_model(model1, test_loader, criterion, device)
    
    result1 = {
        'model': 'Feedforward-NLM',
        'year': '2003',
        'perplexity': final_perplexity,
        'parameters': count_parameters(model1),
        'training_time': metrics['training_time_minutes'],
        'memory_usage': metrics['memory_usage_mb'],
        'innovation': 'First neural language model to beat n-grams'
    }
    results.append(result1)
    
    # Train Improved Neural LM
    print("\n" + "="*50)
    print("Training Improved Neural Language Model")
    
    model2 = ImprovedNeuralLM(len(vocab), embedding_dim=100, hidden_dim=200, context_size=context_size)
    
    metrics = track_computational_metrics(
        'Improved-NLM',
        train_model,
        model2, train_loader, val_loader, 5, 0.001
    )
    
    train_losses, val_perplexities = metrics['result']
    training_histories['Improved-NLM'] = (train_losses, val_perplexities)
    
    final_perplexity = evaluate_model(model2, test_loader, criterion, device)
    
    result2 = {
        'model': 'Improved-NLM',
        'year': '2010',
        'perplexity': final_perplexity,
        'parameters': count_parameters(model2),
        'training_time': metrics['training_time_minutes'],
        'memory_usage': metrics['memory_usage_mb'],
        'innovation': 'Multi-layer networks with regularization'
    }
    results.append(result2)
    
    # Generate sample text
    print("\n" + "="*50)
    print("TEXT GENERATION EXAMPLES")
    print("="*50)
    
    test_prompts = ["the quick brown", "in the beginning", "scientists have discovered"]
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        
        gen1 = model1.generate_text(vocab, idx_to_word, prompt, max_length=10)
        print(f"  Feedforward-NLM: {gen1}")
        
        gen2 = model2.generate_text(vocab, idx_to_word, prompt, max_length=10)
        print(f"  Improved-NLM: {gen2}")
    
    # Create visualizations
    create_visualizations(results, training_histories)
    
    # Print summary
    print("\n" + "="*60)
    print("EARLY NEURAL LANGUAGE MODELS SUMMARY")
    print("="*60)
    
    for result in results:
        print(f"\n{result['model']} ({result['year']}):")
        print(f"  Perplexity: {result['perplexity']:.2f}")
        print(f"  Parameters: {result['parameters']:,}")
        print(f"  Training Time: {result['training_time']:.2f} minutes")
        print(f"  Memory Usage: {result['memory_usage']:.2f} MB")
        print(f"  Innovation: {result['innovation']}")
    
    print("\nKey Insights:")
    print("- Neural networks outperform statistical n-gram models")
    print("- Learned embeddings capture semantic relationships")
    print("- Fixed context window still limits long-range dependencies")
    print("- Foundation for RNN-based language models")
    print("- Established neural approach to language modeling")
    
    return results

if __name__ == "__main__":
    results = main()