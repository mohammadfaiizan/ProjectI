import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional

# Dataset for Text Classification
class TextClassificationDataset(Dataset):
    """Dataset for text classification tasks"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize and encode
        encoding = self.tokenizer(text, max_length=self.max_length)
        
        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Simple Tokenizer
class SimpleTokenizer:
    """Simple word-based tokenizer"""
    
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_built = False
    
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        word_counts = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency and take top vocab_size - 2 (excluding PAD and UNK)
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        for i, (word, _) in enumerate(sorted_words[:self.vocab_size - 2]):
            idx = i + 2  # Start from 2 (after PAD and UNK)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
        
        self.vocab_built = True
    
    def __call__(self, text, max_length=512):
        """Tokenize and encode text"""
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab first.")
        
        words = text.lower().split()
        input_ids = []
        
        for word in words:
            if word in self.word_to_idx:
                input_ids.append(self.word_to_idx[word])
            else:
                input_ids.append(self.word_to_idx['<UNK>'])
        
        # Truncate or pad
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        
        attention_mask = [1] * len(input_ids)
        
        # Pad to max_length
        while len(input_ids) < max_length:
            input_ids.append(self.word_to_idx['<PAD>'])
            attention_mask.append(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

# CNN Text Classifier
class CNNTextClassifier(nn.Module):
    """CNN-based text classifier"""
    
    def __init__(self, vocab_size, embed_dim, num_classes, num_filters=100, 
                 filter_sizes=[3, 4, 5], dropout=0.5):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Multiple convolution layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k)
            for k in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        # input_ids: [batch_size, seq_len]
        x = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        x = x.transpose(1, 2)  # [batch_size, embed_dim, seq_len]
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # [batch_size, num_filters, new_seq_len]
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # [batch_size, num_filters]
            conv_outputs.append(pooled)
        
        # Concatenate all conv outputs
        concatenated = torch.cat(conv_outputs, dim=1)  # [batch_size, total_filters]
        concatenated = self.dropout(concatenated)
        
        logits = self.fc(concatenated)  # [batch_size, num_classes]
        return logits

# RNN Text Classifier
class RNNTextClassifier(nn.Module):
    """RNN-based text classifier (LSTM/GRU)"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, 
                 num_layers=2, rnn_type='LSTM', dropout=0.5, bidirectional=True):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                              dropout=dropout, bidirectional=bidirectional, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers, 
                             dropout=dropout, bidirectional=bidirectional, batch_first=True)
        else:
            raise ValueError("rnn_type must be 'LSTM' or 'GRU'")
        
        self.dropout = nn.Dropout(dropout)
        
        # Calculate output dimension
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(rnn_output_dim, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        # input_ids: [batch_size, seq_len]
        x = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        
        # RNN forward pass
        rnn_output, _ = self.rnn(x)  # [batch_size, seq_len, hidden_dim * num_directions]
        
        # Use last output or attention pooling
        if attention_mask is not None:
            # Masked average pooling
            mask = attention_mask.unsqueeze(2).float()  # [batch_size, seq_len, 1]
            pooled = (rnn_output * mask).sum(dim=1) / mask.sum(dim=1)  # [batch_size, hidden_dim * num_directions]
        else:
            # Use last output
            pooled = rnn_output[:, -1, :]  # [batch_size, hidden_dim * num_directions]
        
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)  # [batch_size, num_classes]
        
        return logits

# Attention-based Text Classifier
class AttentionTextClassifier(nn.Module):
    """Self-attention based text classifier"""
    
    def __init__(self, vocab_size, embed_dim, num_classes, num_heads=8, 
                 num_layers=2, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        # input_ids: [batch_size, seq_len]
        x = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        x = x.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, embed_dim]
        
        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to boolean mask (True for positions to attend to)
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None
        
        # Transformer encoding
        encoded = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Global average pooling
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(2).float()  # [batch_size, seq_len, 1]
            pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1)  # [batch_size, embed_dim]
        else:
            pooled = encoded.mean(dim=1)  # [batch_size, embed_dim]
        
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)  # [batch_size, num_classes]
        
        return logits

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, embed_dim, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Hierarchical Attention Network (HAN)
class HierarchicalAttentionNetwork(nn.Module):
    """Hierarchical Attention Network for document classification"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Word-level attention
        self.word_gru = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.word_attention = AttentionLayer(hidden_dim * 2)
        
        # Sentence-level attention
        self.sentence_gru = nn.GRU(hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
        self.sentence_attention = AttentionLayer(hidden_dim * 2)
        
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask=None):
        # Assume input_ids is [batch_size, num_sentences, max_sentence_len]
        batch_size, num_sentences, max_sentence_len = input_ids.size()
        
        # Reshape for word-level processing
        input_ids = input_ids.view(-1, max_sentence_len)  # [batch_size * num_sentences, max_sentence_len]
        
        # Word-level encoding
        word_embeds = self.embedding(input_ids)  # [batch_size * num_sentences, max_sentence_len, embed_dim]
        word_output, _ = self.word_gru(word_embeds)  # [batch_size * num_sentences, max_sentence_len, hidden_dim * 2]
        
        # Word-level attention
        sentence_representations = self.word_attention(word_output)  # [batch_size * num_sentences, hidden_dim * 2]
        sentence_representations = sentence_representations.view(batch_size, num_sentences, -1)
        
        # Sentence-level encoding
        sentence_output, _ = self.sentence_gru(sentence_representations)  # [batch_size, num_sentences, hidden_dim * 2]
        
        # Sentence-level attention
        document_representation = self.sentence_attention(sentence_output)  # [batch_size, hidden_dim * 2]
        
        document_representation = self.dropout(document_representation)
        logits = self.fc(document_representation)
        
        return logits

class AttentionLayer(nn.Module):
    """Attention layer for HAN"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim]
        attention_weights = torch.tanh(self.attention(x))  # [batch_size, seq_len, hidden_dim]
        attention_weights = self.context_vector(attention_weights).squeeze(2)  # [batch_size, seq_len]
        attention_weights = F.softmax(attention_weights, dim=1)  # [batch_size, seq_len]
        
        # Weighted sum
        attended = torch.sum(x * attention_weights.unsqueeze(2), dim=1)  # [batch_size, hidden_dim]
        
        return attended

# Multi-task Text Classifier
class MultiTaskTextClassifier(nn.Module):
    """Multi-task learning for text classification"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, task_configs, shared_layers=2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Shared encoder
        self.shared_encoder = nn.LSTM(embed_dim, hidden_dim, shared_layers, 
                                     bidirectional=True, batch_first=True, dropout=0.5)
        
        # Task-specific layers
        self.task_heads = nn.ModuleDict()
        for task_name, num_classes in task_configs.items():
            self.task_heads[task_name] = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, num_classes)
            )
    
    def forward(self, input_ids, attention_mask=None, task_name=None):
        # input_ids: [batch_size, seq_len]
        x = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        
        # Shared encoding
        encoded, _ = self.shared_encoder(x)  # [batch_size, seq_len, hidden_dim * 2]
        
        # Pooling
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(2).float()
            pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            pooled = encoded.mean(dim=1)
        
        # Task-specific prediction
        if task_name is not None:
            return self.task_heads[task_name](pooled)
        else:
            # Return predictions for all tasks
            outputs = {}
            for task, head in self.task_heads.items():
                outputs[task] = head(pooled)
            return outputs

# Training utilities
def train_classifier(model, train_loader, val_loader, num_epochs=10, lr=1e-3):
    """Train text classifier"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        scheduler.step()

if __name__ == "__main__":
    # Example usage
    print("Testing text classification models...")
    
    # Dummy data
    texts = ["This is a positive review", "This movie is terrible", "Great film!"]
    labels = [1, 0, 1]  # Binary classification
    
    # Build tokenizer
    tokenizer = SimpleTokenizer(vocab_size=1000)
    tokenizer.build_vocab(texts)
    
    # Create dataset
    dataset = TextClassificationDataset(texts, labels, tokenizer, max_length=50)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Test models
    vocab_size = len(tokenizer.word_to_idx)
    embed_dim = 100
    num_classes = 2
    
    # CNN model
    cnn_model = CNNTextClassifier(vocab_size, embed_dim, num_classes)
    print(f"CNN model parameters: {sum(p.numel() for p in cnn_model.parameters())}")
    
    # RNN model
    rnn_model = RNNTextClassifier(vocab_size, embed_dim, 64, num_classes)
    print(f"RNN model parameters: {sum(p.numel() for p in rnn_model.parameters())}")
    
    # Attention model
    attention_model = AttentionTextClassifier(vocab_size, embed_dim, num_classes)
    print(f"Attention model parameters: {sum(p.numel() for p in attention_model.parameters())}")
    
    # Test forward pass
    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        cnn_output = cnn_model(input_ids, attention_mask)
        rnn_output = rnn_model(input_ids, attention_mask)
        attention_output = attention_model(input_ids, attention_mask)
        
        print(f"CNN output shape: {cnn_output.shape}")
        print(f"RNN output shape: {rnn_output.shape}")
        print(f"Attention output shape: {attention_output.shape}")
        break
    
    print("Text classification models testing completed!")