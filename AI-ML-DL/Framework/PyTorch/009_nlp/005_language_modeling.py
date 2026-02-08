import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import random
from typing import List, Tuple, Optional

# Character-level Language Model Dataset
class CharLMDataset(Dataset):
    """Dataset for character-level language modeling"""
    
    def __init__(self, text, seq_length=100):
        self.seq_length = seq_length
        
        # Create character mappings
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        # Convert text to indices
        self.data = [self.char_to_idx[ch] for ch in text]
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        # Input sequence and target (shifted by 1)
        input_seq = torch.tensor(self.data[idx:idx + self.seq_length], dtype=torch.long)
        target_seq = torch.tensor(self.data[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        return input_seq, target_seq

# Word-level Language Model Dataset
class WordLMDataset(Dataset):
    """Dataset for word-level language modeling"""
    
    def __init__(self, text, seq_length=50, min_freq=2):
        self.seq_length = seq_length
        
        # Tokenize and build vocabulary
        words = text.lower().split()
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Create vocabulary with special tokens
        self.word_to_idx = {'<UNK>': 0, '<BOS>': 1, '<EOS>': 2}
        self.idx_to_word = {0: '<UNK>', 1: '<BOS>', 2: '<EOS>'}
        
        for word, count in word_counts.items():
            if count >= min_freq:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
        
        self.vocab_size = len(self.word_to_idx)
        
        # Convert words to indices
        self.data = []
        for word in words:
            self.data.append(self.word_to_idx.get(word, self.word_to_idx['<UNK>']))
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        input_seq = torch.tensor(self.data[idx:idx + self.seq_length], dtype=torch.long)
        target_seq = torch.tensor(self.data[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        return input_seq, target_seq

# Character-level LSTM Language Model
class CharLSTMLM(nn.Module):
    """Character-level LSTM language model"""
    
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.5):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights"""
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-init_range, init_range)
    
    def forward(self, x, hidden=None):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        if hidden is None:
            lstm_out, hidden = self.lstm(embedded)
        else:
            lstm_out, hidden = self.lstm(embedded, hidden)
        
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)  # [batch_size, seq_len, vocab_size]
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

# Word-level GRU Language Model
class WordGRULM(nn.Module):
    """Word-level GRU language model"""
    
    def __init__(self, vocab_size, embed_dim=200, hidden_dim=512, num_layers=2, dropout=0.5):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers,
                         dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        self.init_weights()
    
    def init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-init_range, init_range)
    
    def forward(self, x, hidden=None):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)
        
        if hidden is None:
            gru_out, hidden = self.gru(embedded)
        else:
            gru_out, hidden = self.gru(embedded, hidden)
        
        gru_out = self.dropout(gru_out)
        output = self.fc(gru_out)  # [batch_size, seq_len, vocab_size]
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)

# Transformer Language Model
class TransformerLM(nn.Module):
    """Transformer-based language model"""
    
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_layers=6, 
                 ff_dim=2048, max_seq_len=1024, dropout=0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        self.init_weights()
    
    def init_weights(self):
        init_range = 0.1
        self.token_embedding.weight.data.uniform_(-init_range, init_range)
        self.pos_embedding.data.uniform_(-init_range, init_range)
        self.head.weight.data.uniform_(-init_range, init_range)
    
    def forward(self, x):
        # x: [batch_size, seq_len]
        seq_len = x.size(1)
        
        # Token embeddings + positional embeddings
        token_emb = self.token_embedding(x)  # [batch_size, seq_len, embed_dim]
        pos_emb = self.pos_embedding[:, :seq_len, :]
        
        x = self.dropout(token_emb + pos_emb)
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        
        # Transformer forward
        x = self.transformer(x, mask=mask)
        x = self.ln_f(x)
        
        # Output projection
        logits = self.head(x)  # [batch_size, seq_len, vocab_size]
        
        return logits

# Attention-based Language Model
class AttentionLM(nn.Module):
    """Language model with attention mechanism"""
    
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.5):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                           dropout=dropout, batch_first=True)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)
        
        if hidden is None:
            lstm_out, hidden = self.lstm(embedded)
        else:
            lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Apply attention
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(lstm_out * attention_weights, dim=1)  # [batch_size, hidden_dim]
        
        context = self.dropout(context)
        output = self.fc(context)  # [batch_size, vocab_size]
        
        return output, hidden, attention_weights

# Language Model Trainer
class LanguageModelTrainer:
    """Trainer for language models"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def train_epoch(self, dataloader, optimizer, criterion, clip_grad=1.0):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if isinstance(self.model, TransformerLM):
                outputs = self.model(inputs)
                # Reshape for loss calculation
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
            else:
                outputs, _ = self.model(inputs)
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
            
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                if isinstance(self.model, TransformerLM):
                    outputs = self.model(inputs)
                    outputs = outputs.view(-1, outputs.size(-1))
                    targets = targets.view(-1)
                else:
                    outputs, _ = self.model(inputs)
                    outputs = outputs.view(-1, outputs.size(-1))
                    targets = targets.view(-1)
                
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, train_loader, val_loader, num_epochs=10, lr=1e-3):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss = self.evaluate(val_loader, criterion)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'Train Perplexity: {math.exp(train_loss):.2f}, Val Perplexity: {math.exp(val_loss):.2f}')
            print('-' * 50)
            
            scheduler.step()

# Text Generation Functions
def generate_text_rnn(model, start_text, idx_to_token, token_to_idx, 
                     max_length=200, temperature=1.0, device='cuda'):
    """Generate text using RNN-based language model"""
    model.eval()
    
    # Convert start text to indices
    if isinstance(start_text, str):
        tokens = list(start_text)  # For character-level
    else:
        tokens = start_text.split()  # For word-level
    
    generated = tokens.copy()
    hidden = None
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get last token
            if len(generated) == 0:
                break
            
            last_token = generated[-1]
            if last_token not in token_to_idx:
                break
            
            # Convert to tensor
            input_idx = torch.tensor([[token_to_idx[last_token]]], dtype=torch.long).to(device)
            
            # Forward pass
            output, hidden = model(input_idx, hidden)
            
            # Apply temperature scaling
            logits = output[0, -1] / temperature
            probabilities = F.softmax(logits, dim=0)
            
            # Sample next token
            next_idx = torch.multinomial(probabilities, 1).item()
            next_token = idx_to_token[next_idx]
            
            generated.append(next_token)
    
    if isinstance(start_text, str):
        return ''.join(generated)
    else:
        return ' '.join(generated)

def generate_text_transformer(model, start_text, idx_to_token, token_to_idx, 
                             max_length=200, temperature=1.0, device='cuda'):
    """Generate text using Transformer language model"""
    model.eval()
    
    # Convert start text to indices
    if isinstance(start_text, str):
        tokens = list(start_text)  # For character-level
    else:
        tokens = start_text.split()  # For word-level
    
    generated_indices = [token_to_idx.get(token, 0) for token in tokens]
    
    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input (use sliding window if too long)
            input_indices = generated_indices[-model.max_seq_len:]
            input_tensor = torch.tensor([input_indices], dtype=torch.long).to(device)
            
            # Forward pass
            logits = model(input_tensor)
            
            # Get last token logits and apply temperature
            last_logits = logits[0, -1] / temperature
            probabilities = F.softmax(last_logits, dim=0)
            
            # Sample next token
            next_idx = torch.multinomial(probabilities, 1).item()
            generated_indices.append(next_idx)
    
    # Convert back to text
    generated_tokens = [idx_to_token.get(idx, '<UNK>') for idx in generated_indices]
    
    if isinstance(start_text, str):
        return ''.join(generated_tokens)
    else:
        return ' '.join(generated_tokens)

# Perplexity calculation
def calculate_perplexity(model, dataloader, device='cuda'):
    """Calculate perplexity of the model"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if isinstance(model, TransformerLM):
                outputs = model(inputs)
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
            else:
                outputs, _ = model(inputs)
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
            
            # Mask out padding tokens (assuming 0 is padding)
            mask = targets != 0
            if mask.sum() > 0:
                loss = criterion(outputs[mask], targets[mask])
                total_loss += loss.item() * mask.sum().item()
                total_tokens += mask.sum().item()
    
    average_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(average_loss)
    
    return perplexity

if __name__ == "__main__":
    print("Testing language models...")
    
    # Sample text data
    sample_text = """
    The quick brown fox jumps over the lazy dog. This is a sample text for testing
    language models. We will use this text to train character-level and word-level
    models to see how they perform in generating new text.
    """
    
    # Test character-level model
    print("Testing Character-level LSTM...")
    char_dataset = CharLMDataset(sample_text, seq_length=50)
    char_dataloader = DataLoader(char_dataset, batch_size=2, shuffle=True)
    
    char_model = CharLSTMLM(char_dataset.vocab_size, embed_dim=64, hidden_dim=128, num_layers=1)
    print(f"Character vocab size: {char_dataset.vocab_size}")
    print(f"Character model parameters: {sum(p.numel() for p in char_model.parameters())}")
    
    # Test forward pass
    for inputs, targets in char_dataloader:
        outputs, hidden = char_model(inputs)
        print(f"Char model input shape: {inputs.shape}")
        print(f"Char model output shape: {outputs.shape}")
        break
    
    # Test word-level model
    print("\nTesting Word-level GRU...")
    word_dataset = WordLMDataset(sample_text, seq_length=20)
    word_dataloader = DataLoader(word_dataset, batch_size=2, shuffle=True)
    
    word_model = WordGRULM(word_dataset.vocab_size, embed_dim=64, hidden_dim=128, num_layers=1)
    print(f"Word vocab size: {word_dataset.vocab_size}")
    print(f"Word model parameters: {sum(p.numel() for p in word_model.parameters())}")
    
    # Test forward pass
    for inputs, targets in word_dataloader:
        outputs, hidden = word_model(inputs)
        print(f"Word model input shape: {inputs.shape}")
        print(f"Word model output shape: {outputs.shape}")
        break
    
    # Test Transformer model
    print("\nTesting Transformer LM...")
    transformer_model = TransformerLM(char_dataset.vocab_size, embed_dim=128, 
                                    num_heads=4, num_layers=2, max_seq_len=100)
    print(f"Transformer model parameters: {sum(p.numel() for p in transformer_model.parameters())}")
    
    # Test forward pass
    for inputs, targets in char_dataloader:
        outputs = transformer_model(inputs)
        print(f"Transformer input shape: {inputs.shape}")
        print(f"Transformer output shape: {outputs.shape}")
        break
    
    # Test text generation
    print("\nTesting text generation...")
    device = 'cpu'
    
    # Character-level generation
    start_text = "The"
    generated = generate_text_rnn(char_model, start_text, 
                                char_dataset.idx_to_char, char_dataset.char_to_idx,
                                max_length=50, temperature=0.8, device=device)
    print(f"Generated text (char-level): {generated[:100]}...")
    
    # Calculate perplexity
    perplexity = calculate_perplexity(char_model, char_dataloader, device=device)
    print(f"Character model perplexity: {perplexity:.2f}")
    
    print("\nLanguage modeling tests completed!")