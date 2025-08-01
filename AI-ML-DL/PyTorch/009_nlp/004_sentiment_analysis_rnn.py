import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt

# Dataset for Sentiment Analysis
class SentimentDataset(Dataset):
    """Dataset class for sentiment analysis"""
    
    def __init__(self, texts, labels, vocab, max_length=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text to indices
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in text.split()]
        
        # Truncate if too long
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        
        return {
            'text': indices,
            'label': label,
            'length': len(indices)
        }

def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    texts = [torch.tensor(item['text']) for item in batch]
    labels = torch.tensor([item['label'] for item in batch])
    lengths = torch.tensor([item['length'] for item in batch])
    
    # Pad sequences
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    
    return {
        'texts': texts_padded,
        'labels': labels,
        'lengths': lengths
    }

# Basic LSTM Sentiment Classifier
class LSTMSentimentClassifier(nn.Module):
    """LSTM-based sentiment classifier"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, 
                 num_layers=1, dropout=0.5, bidirectional=False):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           dropout=dropout if num_layers > 1 else 0,
                           bidirectional=bidirectional, batch_first=True)
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, output_dim)
        
    def forward(self, x, lengths=None):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        if lengths is not None:
            # Pack sequences for efficiency
            packed = pack_padded_sequence(embedded, lengths.cpu(), 
                                        batch_first=True, enforce_sorted=False)
            lstm_out, (hidden, cell) = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state
        if self.lstm.bidirectional:
            # Concatenate forward and backward final hidden states
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        output = self.dropout(hidden)
        output = self.fc(output)
        
        return output

# GRU Sentiment Classifier
class GRUSentimentClassifier(nn.Module):
    """GRU-based sentiment classifier"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, 
                 num_layers=1, dropout=0.5, bidirectional=False):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers,
                         dropout=dropout if num_layers > 1 else 0,
                         bidirectional=bidirectional, batch_first=True)
        
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(gru_output_dim, output_dim)
        
    def forward(self, x, lengths=None):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        if lengths is not None:
            packed = pack_padded_sequence(embedded, lengths.cpu(),
                                        batch_first=True, enforce_sorted=False)
            gru_out, hidden = self.gru(packed)
            gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)
        else:
            gru_out, hidden = self.gru(embedded)
        
        # Use last hidden state
        if self.gru.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        output = self.dropout(hidden)
        output = self.fc(output)
        
        return output

# LSTM with Attention
class LSTMWithAttention(nn.Module):
    """LSTM with attention mechanism for sentiment analysis"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, 
                 num_layers=1, dropout=0.5, bidirectional=True):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                           dropout=dropout if num_layers > 1 else 0,
                           bidirectional=bidirectional, batch_first=True)
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, output_dim)
        
    def forward(self, x, lengths=None):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        if lengths is not None:
            packed = pack_padded_sequence(embedded, lengths.cpu(),
                                        batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(embedded)
        
        # Apply attention
        attention_weights = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention mask if lengths provided
        if lengths is not None:
            mask = torch.arange(x.size(1), device=x.device).expand(x.size(0), x.size(1))
            mask = mask < lengths.unsqueeze(1)
            attention_weights = attention_weights.masked_fill(~mask.unsqueeze(2), 0)
            # Renormalize
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-10)
        
        # Compute attended representation
        attended = torch.sum(lstm_out * attention_weights, dim=1)  # [batch_size, lstm_output_dim]
        
        output = self.dropout(attended)
        output = self.fc(output)
        
        return output, attention_weights

# Hierarchical LSTM for Document-level Sentiment
class HierarchicalLSTM(nn.Module):
    """Hierarchical LSTM for document-level sentiment analysis"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Word-level LSTM
        self.word_lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        
        # Sentence-level LSTM
        self.sentence_lstm = nn.LSTM(hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
        
        # Attention mechanisms
        self.word_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.sentence_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        # x: [batch_size, num_sentences, max_sentence_len]
        batch_size, num_sentences, max_sentence_len = x.size()
        
        # Reshape for word-level processing
        x = x.view(-1, max_sentence_len)  # [batch_size * num_sentences, max_sentence_len]
        
        # Word-level encoding
        embedded = self.embedding(x)  # [batch_size * num_sentences, max_sentence_len, embed_dim]
        word_lstm_out, _ = self.word_lstm(embedded)  # [batch_size * num_sentences, max_sentence_len, hidden_dim * 2]
        
        # Word-level attention
        word_attention_weights = F.softmax(self.word_attention(word_lstm_out), dim=1)
        sentence_vectors = torch.sum(word_lstm_out * word_attention_weights, dim=1)
        
        # Reshape for sentence-level processing
        sentence_vectors = sentence_vectors.view(batch_size, num_sentences, -1)
        
        # Sentence-level encoding
        sentence_lstm_out, _ = self.sentence_lstm(sentence_vectors)
        
        # Sentence-level attention
        sentence_attention_weights = F.softmax(self.sentence_attention(sentence_lstm_out), dim=1)
        document_vector = torch.sum(sentence_lstm_out * sentence_attention_weights, dim=1)
        
        # Classification
        output = self.dropout(document_vector)
        output = self.fc(output)
        
        return output

# Multi-aspect Sentiment Analysis
class MultiAspectSentiment(nn.Module):
    """Multi-aspect sentiment analysis model"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, aspects, dropout=0.5):
        super().__init__()
        
        self.aspects = aspects
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Shared LSTM encoder
        self.shared_lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        
        # Aspect-specific attention and classifiers
        self.aspect_attentions = nn.ModuleDict()
        self.aspect_classifiers = nn.ModuleDict()
        
        for aspect in aspects:
            self.aspect_attentions[aspect] = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
            self.aspect_classifiers[aspect] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, 3)  # Positive, Negative, Neutral
            )
    
    def forward(self, x, lengths=None):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)
        
        if lengths is not None:
            packed = pack_padded_sequence(embedded, lengths.cpu(),
                                        batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.shared_lstm(packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.shared_lstm(embedded)
        
        outputs = {}
        
        for aspect in self.aspects:
            # Aspect-specific attention
            attention_weights = F.softmax(self.aspect_attentions[aspect](lstm_out), dim=1)
            
            if lengths is not None:
                mask = torch.arange(x.size(1), device=x.device).expand(x.size(0), x.size(1))
                mask = mask < lengths.unsqueeze(1)
                attention_weights = attention_weights.masked_fill(~mask.unsqueeze(2), 0)
                attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-10)
            
            # Compute aspect-specific representation
            aspect_repr = torch.sum(lstm_out * attention_weights, dim=1)
            
            # Classify
            outputs[aspect] = self.aspect_classifiers[aspect](aspect_repr)
        
        return outputs

# Training utilities
class SentimentTrainer:
    """Trainer class for sentiment analysis models"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            texts = batch['texts'].to(self.device)
            labels = batch['labels'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if hasattr(self.model, 'forward') and 'lengths' in self.model.forward.__code__.co_varnames:
                outputs = self.model(texts, lengths)
            else:
                outputs = self.model(texts)
            
            # Handle attention models that return multiple outputs
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return total_loss / len(train_loader), 100.0 * correct / total
    
    def evaluate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                texts = batch['texts'].to(self.device)
                labels = batch['labels'].to(self.device)
                lengths = batch['lengths'].to(self.device)
                
                if hasattr(self.model, 'forward') and 'lengths' in self.model.forward.__code__.co_varnames:
                    outputs = self.model(texts, lengths)
                else:
                    outputs = self.model(texts)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return total_loss / len(val_loader), 100.0 * correct / total
    
    def train(self, train_loader, val_loader, num_epochs=10, lr=1e-3, weight_decay=1e-5):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print('-' * 50)
            
            scheduler.step()
        
        return train_losses, train_accs, val_losses, val_accs

# Utility functions
def build_vocab(texts, min_freq=2, max_vocab_size=10000):
    """Build vocabulary from texts"""
    word_counts = {}
    for text in texts:
        for word in text.split():
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Filter by frequency and limit size
    vocab = {'<PAD>': 0, '<UNK>': 1}
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    for word, count in sorted_words[:max_vocab_size-2]:
        if count >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab

def predict_sentiment(model, text, vocab, device='cuda', max_length=256):
    """Predict sentiment for a single text"""
    model.eval()
    
    # Tokenize and convert to indices
    tokens = text.split()
    indices = [vocab.get(word, vocab['<UNK>']) for word in tokens]
    
    if len(indices) > max_length:
        indices = indices[:max_length]
    
    # Convert to tensor
    text_tensor = torch.tensor([indices]).to(device)
    length_tensor = torch.tensor([len(indices)]).to(device)
    
    with torch.no_grad():
        if hasattr(model, 'forward') and 'lengths' in model.forward.__code__.co_varnames:
            outputs = model(text_tensor, length_tensor)
        else:
            outputs = model(text_tensor)
        
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities[0].cpu().numpy()

if __name__ == "__main__":
    print("Testing RNN-based sentiment analysis models...")
    
    # Create dummy data
    texts = [
        "This movie is amazing and wonderful",
        "I hate this film it's terrible",
        "The acting was good but story was bad",
        "Best movie ever made",
        "Worst film I've ever seen"
    ]
    labels = [1, 0, 0, 1, 0]  # 1: positive, 0: negative
    
    # Build vocabulary
    vocab = build_vocab(texts, min_freq=1, max_vocab_size=1000)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create dataset and dataloader
    dataset = SentimentDataset(texts, labels, vocab, max_length=20)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    # Model parameters
    vocab_size = len(vocab)
    embed_dim = 50
    hidden_dim = 32
    output_dim = 2
    
    # Test models
    models = {
        'LSTM': LSTMSentimentClassifier(vocab_size, embed_dim, hidden_dim, output_dim),
        'GRU': GRUSentimentClassifier(vocab_size, embed_dim, hidden_dim, output_dim),
        'BiLSTM': LSTMSentimentClassifier(vocab_size, embed_dim, hidden_dim, output_dim, bidirectional=True),
        'LSTM+Attention': LSTMWithAttention(vocab_size, embed_dim, hidden_dim, output_dim)
    }
    
    # Test forward pass
    for name, model in models.items():
        print(f"\nTesting {name}:")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")
        
        # Test with a batch
        for batch in dataloader:
            texts = batch['texts']
            lengths = batch['lengths']
            
            if name == 'LSTM+Attention':
                outputs, attention_weights = model(texts, lengths)
                print(f"Output shape: {outputs.shape}")
                print(f"Attention weights shape: {attention_weights.shape}")
            else:
                outputs = model(texts, lengths)
                print(f"Output shape: {outputs.shape}")
            break
    
    # Test prediction
    test_text = "This is a great movie"
    model = models['LSTM']
    prediction, confidence, probs = predict_sentiment(model, test_text, vocab, device='cpu')
    print(f"\nTest prediction for '{test_text}':")
    print(f"Predicted class: {prediction}, Confidence: {confidence:.4f}")
    print(f"Probabilities: {probs}")
    
    print("\nRNN sentiment analysis models testing completed!")