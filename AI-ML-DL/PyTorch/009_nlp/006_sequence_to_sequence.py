import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
import numpy as np
from typing import Tuple, List, Optional

# Seq2Seq Dataset
class Seq2SeqDataset(Dataset):
    """Dataset for sequence-to-sequence tasks"""
    
    def __init__(self, source_texts, target_texts, source_vocab, target_vocab, max_length=50):
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.source_texts)
    
    def __getitem__(self, idx):
        source_text = self.source_texts[idx]
        target_text = self.target_texts[idx]
        
        # Convert to indices
        source_indices = [self.source_vocab.get(word, self.source_vocab['<UNK>']) 
                         for word in source_text.split()]
        
        # Add BOS and EOS to target
        target_words = ['<BOS>'] + target_text.split() + ['<EOS>']
        target_indices = [self.target_vocab.get(word, self.target_vocab['<UNK>']) 
                         for word in target_words]
        
        # Truncate if necessary
        if len(source_indices) > self.max_length:
            source_indices = source_indices[:self.max_length]
        if len(target_indices) > self.max_length + 1:  # +1 for BOS/EOS
            target_indices = target_indices[:self.max_length + 1]
        
        return {
            'source': source_indices,
            'target': target_indices[:-1],  # Input to decoder (without EOS)
            'target_output': target_indices[1:],  # Target output (without BOS)
            'source_length': len(source_indices),
            'target_length': len(target_indices) - 1
        }

def collate_fn_seq2seq(batch):
    """Custom collate function for seq2seq batches"""
    sources = [torch.tensor(item['source']) for item in batch]
    targets = [torch.tensor(item['target']) for item in batch]
    target_outputs = [torch.tensor(item['target_output']) for item in batch]
    source_lengths = torch.tensor([item['source_length'] for item in batch])
    target_lengths = torch.tensor([item['target_length'] for item in batch])
    
    # Pad sequences
    sources_padded = pad_sequence(sources, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    target_outputs_padded = pad_sequence(target_outputs, batch_first=True, padding_value=0)
    
    return {
        'source': sources_padded,
        'target': targets_padded,
        'target_output': target_outputs_padded,
        'source_lengths': source_lengths,
        'target_lengths': target_lengths
    }

# Basic Encoder
class Encoder(nn.Module):
    """Basic LSTM encoder"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.5):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           dropout=dropout if num_layers > 1 else 0,
                           batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lengths=None):
        # x: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(x))
        
        if lengths is not None:
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            packed = pack_padded_sequence(embedded, lengths.cpu(), 
                                        batch_first=True, enforce_sorted=False)
            outputs, (hidden, cell) = self.lstm(packed)
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, (hidden, cell) = self.lstm(embedded)
        
        # Combine bidirectional hidden states
        # hidden: [num_layers * 2, batch_size, hidden_dim]
        hidden = hidden.view(self.num_layers, 2, hidden.size(1), hidden.size(2))
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)  # Concatenate forward and backward
        
        cell = cell.view(self.num_layers, 2, cell.size(1), cell.size(2))
        cell = torch.cat([cell[:, 0], cell[:, 1]], dim=2)
        
        return outputs, (hidden, cell)

# Basic Decoder
class Decoder(nn.Module):
    """Basic LSTM decoder"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.5):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim * 2, num_layers,  # *2 for bidirectional encoder
                           dropout=dropout if num_layers > 1 else 0,
                           batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim * 2, vocab_size)
        
    def forward(self, x, hidden, cell):
        # x: [batch_size, 1] - single time step
        embedded = self.dropout(self.embedding(x))
        
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = self.output_projection(self.dropout(output))
        
        return output, (hidden, cell)

# Attention Mechanism
class BahdanauAttention(nn.Module):
    """Bahdanau (additive) attention mechanism"""
    
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, attention_dim):
        super().__init__()
        
        self.encoder_projection = nn.Linear(encoder_hidden_dim, attention_dim)
        self.decoder_projection = nn.Linear(decoder_hidden_dim, attention_dim)
        self.attention_vector = nn.Linear(attention_dim, 1)
        
    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: [batch_size, src_seq_len, encoder_hidden_dim]
        # decoder_hidden: [batch_size, decoder_hidden_dim]
        
        batch_size, src_seq_len, _ = encoder_outputs.size()
        
        # Project encoder outputs
        encoder_proj = self.encoder_projection(encoder_outputs)  # [batch_size, src_seq_len, attention_dim]
        
        # Project decoder hidden and expand
        decoder_proj = self.decoder_projection(decoder_hidden).unsqueeze(1)  # [batch_size, 1, attention_dim]
        decoder_proj = decoder_proj.expand(batch_size, src_seq_len, -1)  # [batch_size, src_seq_len, attention_dim]
        
        # Compute attention scores
        energy = torch.tanh(encoder_proj + decoder_proj)  # [batch_size, src_seq_len, attention_dim]
        attention_scores = self.attention_vector(energy).squeeze(2)  # [batch_size, src_seq_len]
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, src_seq_len]
        
        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [batch_size, encoder_hidden_dim]
        
        return context, attention_weights

class LuongAttention(nn.Module):
    """Luong (multiplicative) attention mechanism"""
    
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, attention_type='general'):
        super().__init__()
        
        self.attention_type = attention_type
        
        if attention_type == 'general':
            self.linear = nn.Linear(decoder_hidden_dim, encoder_hidden_dim)
        elif attention_type == 'concat':
            self.linear = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, encoder_hidden_dim)
            self.vector = nn.Parameter(torch.FloatTensor(encoder_hidden_dim))
        
    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: [batch_size, src_seq_len, encoder_hidden_dim]
        # decoder_hidden: [batch_size, decoder_hidden_dim]
        
        batch_size, src_seq_len, encoder_hidden_dim = encoder_outputs.size()
        
        if self.attention_type == 'dot':
            # Simple dot product (requires same dimensions)
            attention_scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)
            
        elif self.attention_type == 'general':
            # General: decoder_hidden * W * encoder_outputs
            projected_decoder = self.linear(decoder_hidden)  # [batch_size, encoder_hidden_dim]
            attention_scores = torch.bmm(encoder_outputs, projected_decoder.unsqueeze(2)).squeeze(2)
            
        elif self.attention_type == 'concat':
            # Concatenation-based attention
            decoder_expanded = decoder_hidden.unsqueeze(1).expand(batch_size, src_seq_len, -1)
            concat_features = torch.cat([encoder_outputs, decoder_expanded], dim=2)
            attention_scores = torch.bmm(
                torch.tanh(self.linear(concat_features)),
                self.vector.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, 1)
            ).squeeze(2)
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, src_seq_len]
        
        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attention_weights

# Attention Decoder
class AttentionDecoder(nn.Module):
    """LSTM decoder with attention mechanism"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, encoder_hidden_dim, 
                 attention_type='bahdanau', num_layers=1, dropout=0.5):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.attention_type = attention_type
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # LSTM input size includes context vector
        self.lstm = nn.LSTM(embed_dim + encoder_hidden_dim, hidden_dim, num_layers,
                           dropout=dropout if num_layers > 1 else 0,
                           batch_first=True)
        
        # Attention mechanism
        if attention_type == 'bahdanau':
            self.attention = BahdanauAttention(encoder_hidden_dim, hidden_dim, hidden_dim)
        else:  # luong
            self.attention = LuongAttention(encoder_hidden_dim, hidden_dim, attention_type)
        
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim + encoder_hidden_dim, vocab_size)
        
    def forward(self, x, hidden, cell, encoder_outputs, src_mask=None):
        # x: [batch_size, 1] - single time step
        batch_size = x.size(0)
        
        # Embedding
        embedded = self.dropout(self.embedding(x))  # [batch_size, 1, embed_dim]
        
        # Get current decoder hidden state for attention
        if self.attention_type == 'bahdanau':
            # Use previous hidden state
            decoder_hidden = hidden[-1]  # [batch_size, hidden_dim]
        else:
            # Use current hidden state (after LSTM)
            decoder_hidden = hidden[-1]
        
        # Compute attention
        context, attention_weights = self.attention(encoder_outputs, decoder_hidden)
        
        # Apply source mask if provided
        if src_mask is not None:
            attention_weights = attention_weights.masked_fill(~src_mask, 0)
            # Renormalize
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-10)
            # Recompute context with masked attention
            context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        # Concatenate embedding and context
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)  # [batch_size, 1, embed_dim + encoder_hidden_dim]
        
        # LSTM forward pass
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        # Concatenate LSTM output and context for final projection
        combined = torch.cat([output.squeeze(1), context], dim=1)  # [batch_size, hidden_dim + encoder_hidden_dim]
        output = self.output_projection(self.dropout(combined))  # [batch_size, vocab_size]
        
        return output, (hidden, cell), attention_weights

# Complete Seq2Seq Model
class Seq2SeqWithAttention(nn.Module):
    """Complete sequence-to-sequence model with attention"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, hidden_dim, 
                 attention_type='bahdanau', num_layers=1, dropout=0.5):
        super().__init__()
        
        self.encoder = Encoder(src_vocab_size, embed_dim, hidden_dim, num_layers, dropout)
        self.decoder = AttentionDecoder(tgt_vocab_size, embed_dim, hidden_dim * 2, 
                                      hidden_dim * 2, attention_type, num_layers, dropout)
        
        self.tgt_vocab_size = tgt_vocab_size
        
    def forward(self, src, tgt, src_lengths=None, teacher_forcing_ratio=1.0):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        
        # Encoder
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
        
        # Initialize decoder
        decoder_input = tgt[:, 0].unsqueeze(1)  # [batch_size, 1] - BOS token
        decoder_hidden = hidden
        decoder_cell = cell
        
        # Store outputs
        outputs = torch.zeros(batch_size, tgt_len, self.tgt_vocab_size).to(src.device)
        attentions = []
        
        # Create source mask for attention
        if src_lengths is not None:
            src_mask = torch.arange(src.size(1), device=src.device).expand(batch_size, src.size(1))
            src_mask = src_mask < src_lengths.unsqueeze(1)
        else:
            src_mask = None
        
        for t in range(tgt_len):
            # Decoder step
            output, (decoder_hidden, decoder_cell), attention_weights = self.decoder(
                decoder_input, decoder_hidden, decoder_cell, encoder_outputs, src_mask
            )
            
            outputs[:, t] = output
            attentions.append(attention_weights)
            
            # Teacher forcing
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing and t < tgt_len - 1:
                decoder_input = tgt[:, t + 1].unsqueeze(1)
            else:
                decoder_input = output.argmax(1).unsqueeze(1)
        
        return outputs, torch.stack(attentions, dim=1)
    
    def generate(self, src, src_lengths=None, max_length=50, start_token=1, end_token=2):
        """Generate sequence using greedy decoding"""
        self.eval()
        batch_size = src.size(0)
        
        with torch.no_grad():
            # Encoder
            encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
            
            # Initialize decoder
            decoder_input = torch.full((batch_size, 1), start_token, device=src.device)
            decoder_hidden = hidden
            decoder_cell = cell
            
            # Create source mask
            if src_lengths is not None:
                src_mask = torch.arange(src.size(1), device=src.device).expand(batch_size, src.size(1))
                src_mask = src_mask < src_lengths.unsqueeze(1)
            else:
                src_mask = None
            
            generated = []
            attentions = []
            
            for _ in range(max_length):
                # Decoder step
                output, (decoder_hidden, decoder_cell), attention_weights = self.decoder(
                    decoder_input, decoder_hidden, decoder_cell, encoder_outputs, src_mask
                )
                
                # Get next token
                next_token = output.argmax(1).unsqueeze(1)
                generated.append(next_token)
                attentions.append(attention_weights)
                
                # Check for end token
                if (next_token == end_token).all():
                    break
                
                decoder_input = next_token
            
            return torch.cat(generated, dim=1), torch.stack(attentions, dim=1)

# Training utilities
class Seq2SeqTrainer:
    """Trainer for sequence-to-sequence models"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def train_epoch(self, dataloader, optimizer, criterion, teacher_forcing_ratio=1.0, clip_grad=1.0):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            src = batch['source'].to(self.device)
            tgt = batch['target'].to(self.device)
            tgt_output = batch['target_output'].to(self.device)
            src_lengths = batch['source_lengths'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs, attentions = self.model(src, tgt, src_lengths, teacher_forcing_ratio)
            
            # Reshape for loss calculation
            outputs = outputs.view(-1, outputs.size(-1))
            tgt_output = tgt_output.view(-1)
            
            # Calculate loss (ignore padding tokens)
            loss = criterion(outputs, tgt_output)
            loss.backward()
            
            # Gradient clipping
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
            
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                src = batch['source'].to(self.device)
                tgt = batch['target'].to(self.device)
                tgt_output = batch['target_output'].to(self.device)
                src_lengths = batch['source_lengths'].to(self.device)
                
                # Forward pass (no teacher forcing during evaluation)
                outputs, attentions = self.model(src, tgt, src_lengths, teacher_forcing_ratio=0.0)
                
                # Reshape for loss calculation
                outputs = outputs.view(-1, outputs.size(-1))
                tgt_output = tgt_output.view(-1)
                
                loss = criterion(outputs, tgt_output)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, train_loader, val_loader, num_epochs=10, lr=1e-3, 
              teacher_forcing_schedule=None):
        """Train the model with optional teacher forcing schedule"""
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        for epoch in range(num_epochs):
            # Teacher forcing ratio scheduling
            if teacher_forcing_schedule:
                tf_ratio = teacher_forcing_schedule(epoch)
            else:
                tf_ratio = max(0.5, 1.0 - epoch * 0.1)  # Decay from 1.0 to 0.5
            
            train_loss = self.train_epoch(train_loader, optimizer, criterion, tf_ratio)
            val_loss = self.evaluate(val_loader, criterion)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'Teacher Forcing Ratio: {tf_ratio:.2f}')
            print('-' * 50)
            
            scheduler.step()

# Utility functions
def build_vocab_pair(source_texts, target_texts, min_freq=2, max_vocab_size=10000):
    """Build vocabularies for source and target languages"""
    def build_single_vocab(texts, prefix=""):
        word_counts = {}
        for text in texts:
            for word in text.split():
                word_counts[word] = word_counts.get(word, 0) + 1
        
        vocab = {'<PAD>': 0, '<UNK>': 1}
        if prefix:
            vocab[f'<{prefix}BOS>'] = 2
            vocab[f'<{prefix}EOS>'] = 3
        else:
            vocab['<BOS>'] = 2
            vocab['<EOS>'] = 3
        
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for word, count in sorted_words[:max_vocab_size-4]:
            if count >= min_freq:
                vocab[word] = len(vocab)
        
        return vocab
    
    source_vocab = build_single_vocab(source_texts)
    target_vocab = build_single_vocab(target_texts)
    
    return source_vocab, target_vocab

def translate_sentence(model, sentence, src_vocab, tgt_vocab, device='cuda', max_length=50):
    """Translate a single sentence"""
    model.eval()
    
    # Convert sentence to indices
    tokens = sentence.lower().split()
    indices = [src_vocab.get(word, src_vocab['<UNK>']) for word in tokens]
    
    # Convert to tensor
    src_tensor = torch.tensor([indices]).to(device)
    src_lengths = torch.tensor([len(indices)]).to(device)
    
    # Generate translation
    with torch.no_grad():
        generated_indices, attention_weights = model.generate(
            src_tensor, src_lengths, max_length=max_length,
            start_token=tgt_vocab['<BOS>'], end_token=tgt_vocab['<EOS>']
        )
    
    # Convert back to words
    tgt_idx_to_word = {idx: word for word, idx in tgt_vocab.items()}
    translated_words = []
    
    for idx in generated_indices[0]:
        word = tgt_idx_to_word.get(idx.item(), '<UNK>')
        if word == '<EOS>':
            break
        if word not in ['<PAD>', '<BOS>']:
            translated_words.append(word)
    
    return ' '.join(translated_words), attention_weights[0]

if __name__ == "__main__":
    print("Testing Sequence-to-Sequence models...")
    
    # Sample data (English to French translation)
    source_texts = ["hello world", "how are you", "good morning", "thank you very much"]
    target_texts = ["bonjour monde", "comment allez vous", "bon matin", "merci beaucoup"]
    
    # Build vocabularies
    src_vocab, tgt_vocab = build_vocab_pair(source_texts, target_texts, min_freq=1)
    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(tgt_vocab)}")
    
    # Create dataset and dataloader
    dataset = Seq2SeqDataset(source_texts, target_texts, src_vocab, tgt_vocab, max_length=20)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_seq2seq)
    
    # Test models
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    embed_dim = 64
    hidden_dim = 32
    
    # Test Seq2Seq with Attention
    model = Seq2SeqWithAttention(src_vocab_size, tgt_vocab_size, embed_dim, hidden_dim)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    for batch in dataloader:
        src = batch['source']
        tgt = batch['target']
        src_lengths = batch['source_lengths']
        
        outputs, attentions = model(src, tgt, src_lengths, teacher_forcing_ratio=1.0)
        print(f"Seq2Seq output shape: {outputs.shape}")
        print(f"Attention weights shape: {attentions.shape}")
        break
    
    # Test translation
    test_sentence = "hello world"
    translation, attention = translate_sentence(model, test_sentence, src_vocab, tgt_vocab, device='cpu')
    print(f"Translation of '{test_sentence}': {translation}")
    print(f"Attention shape: {attention.shape}")
    
    print("Sequence-to-sequence models testing completed!")