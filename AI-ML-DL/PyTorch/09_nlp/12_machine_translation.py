import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import math
import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import Counter
import re

# Machine Translation Dataset
class TranslationDataset(Dataset):
    """Dataset for machine translation tasks"""
    
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_length=50):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length
        
        # Special tokens
        self.src_pad_token = src_vocab.get('<PAD>', 0)
        self.src_unk_token = src_vocab.get('<UNK>', 1)
        self.src_bos_token = src_vocab.get('<BOS>', 2)
        self.src_eos_token = src_vocab.get('<EOS>', 3)
        
        self.tgt_pad_token = tgt_vocab.get('<PAD>', 0)
        self.tgt_unk_token = tgt_vocab.get('<UNK>', 1)
        self.tgt_bos_token = tgt_vocab.get('<BOS>', 2)
        self.tgt_eos_token = tgt_vocab.get('<EOS>', 3)
    
    def __len__(self):
        return len(self.src_sentences)
    
    def tokenize_sentence(self, sentence):
        """Simple tokenization"""
        return sentence.lower().split()
    
    def encode_sentence(self, sentence, vocab, add_special_tokens=True):
        """Encode sentence to indices"""
        tokens = self.tokenize_sentence(sentence)
        
        if add_special_tokens:
            if vocab == self.src_vocab:
                tokens = ['<BOS>'] + tokens + ['<EOS>']
            else:
                tokens = ['<BOS>'] + tokens + ['<EOS>']
        
        # Truncate if too long
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length-1] + ['<EOS>']
        
        # Convert to indices
        unk_token = vocab.get('<UNK>', 1)
        indices = [vocab.get(token, unk_token) for token in tokens]
        
        return indices, len(indices)
    
    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]
        
        # Encode sentences
        src_indices, src_length = self.encode_sentence(src_sentence, self.src_vocab)
        tgt_indices, tgt_length = self.encode_sentence(tgt_sentence, self.tgt_vocab)
        
        # Create decoder input (target without last token)
        tgt_input = tgt_indices[:-1]
        # Create decoder target (target without first token)
        tgt_output = tgt_indices[1:]
        
        return {
            'src': src_indices,
            'tgt_input': tgt_input,
            'tgt_output': tgt_output,
            'src_length': src_length,
            'tgt_length': tgt_length - 1  # Subtract 1 because we removed first token
        }

def collate_fn_translation(batch):
    """Collate function for translation batches"""
    src_sequences = [torch.tensor(item['src']) for item in batch]
    tgt_input_sequences = [torch.tensor(item['tgt_input']) for item in batch]
    tgt_output_sequences = [torch.tensor(item['tgt_output']) for item in batch]
    
    src_lengths = torch.tensor([item['src_length'] for item in batch])
    tgt_lengths = torch.tensor([item['tgt_length'] for item in batch])
    
    # Pad sequences
    src_padded = pad_sequence(src_sequences, batch_first=True, padding_value=0)
    tgt_input_padded = pad_sequence(tgt_input_sequences, batch_first=True, padding_value=0)
    tgt_output_padded = pad_sequence(tgt_output_sequences, batch_first=True, padding_value=0)
    
    return {
        'src': src_padded,
        'tgt_input': tgt_input_padded,
        'tgt_output': tgt_output_padded,
        'src_lengths': src_lengths,
        'tgt_lengths': tgt_lengths
    }

# Basic Encoder-Decoder Model
class Encoder(nn.Module):
    """LSTM-based encoder for machine translation"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, lengths=None):
        # x: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(x))
        
        if lengths is not None:
            packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            outputs, (hidden, cell) = self.lstm(packed)
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, (hidden, cell) = self.lstm(embedded)
        
        # Combine bidirectional hidden states
        # hidden: [num_layers * 2, batch_size, hidden_dim]
        hidden = hidden.view(self.num_layers, 2, hidden.size(1), -1)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)
        
        cell = cell.view(self.num_layers, 2, cell.size(1), -1)
        cell = torch.cat([cell[:, 0], cell[:, 1]], dim=2)
        
        return outputs, (hidden, cell)

class AttentionDecoder(nn.Module):
    """LSTM decoder with attention mechanism"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, encoder_hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_dim, encoder_hidden_dim)
        
        # LSTM input includes embedding + context vector
        self.lstm = nn.LSTM(embed_dim + encoder_hidden_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim + encoder_hidden_dim, vocab_size)
    
    def forward(self, input_token, hidden, cell, encoder_outputs, src_mask=None):
        # input_token: [batch_size, 1]
        # encoder_outputs: [batch_size, src_seq_len, encoder_hidden_dim]
        
        embedded = self.dropout(self.embedding(input_token))
        
        # Compute attention
        context, attention_weights = self.attention(hidden[-1], encoder_outputs, src_mask)
        
        # Concatenate embedding and context
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        
        # LSTM forward
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        # Concatenate LSTM output and context for final projection
        combined = torch.cat([output.squeeze(1), context], dim=1)
        output_logits = self.output_projection(combined)
        
        return output_logits, (hidden, cell), attention_weights

class BahdanauAttention(nn.Module):
    """Bahdanau attention mechanism"""
    
    def __init__(self, decoder_hidden_dim, encoder_hidden_dim, attention_dim=128):
        super().__init__()
        
        self.decoder_projection = nn.Linear(decoder_hidden_dim, attention_dim)
        self.encoder_projection = nn.Linear(encoder_hidden_dim, attention_dim)
        self.attention_vector = nn.Linear(attention_dim, 1)
    
    def forward(self, decoder_hidden, encoder_outputs, src_mask=None):
        # decoder_hidden: [batch_size, decoder_hidden_dim]
        # encoder_outputs: [batch_size, src_seq_len, encoder_hidden_dim]
        
        batch_size, src_seq_len, _ = encoder_outputs.size()
        
        # Project decoder hidden state
        decoder_proj = self.decoder_projection(decoder_hidden).unsqueeze(1)  # [batch_size, 1, attention_dim]
        decoder_proj = decoder_proj.expand(-1, src_seq_len, -1)  # [batch_size, src_seq_len, attention_dim]
        
        # Project encoder outputs
        encoder_proj = self.encoder_projection(encoder_outputs)  # [batch_size, src_seq_len, attention_dim]
        
        # Compute energy
        energy = torch.tanh(decoder_proj + encoder_proj)  # [batch_size, src_seq_len, attention_dim]
        attention_scores = self.attention_vector(energy).squeeze(2)  # [batch_size, src_seq_len]
        
        # Apply mask if provided
        if src_mask is not None:
            attention_scores = attention_scores.masked_fill(~src_mask, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, src_seq_len]
        
        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [batch_size, encoder_hidden_dim]
        
        return context, attention_weights

class Seq2SeqWithAttention(nn.Module):
    """Complete sequence-to-sequence model with attention"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=256, hidden_dim=512, num_layers=1, dropout=0.1):
        super().__init__()
        
        self.encoder = Encoder(src_vocab_size, embed_dim, hidden_dim, num_layers, dropout)
        self.decoder = AttentionDecoder(tgt_vocab_size, embed_dim, hidden_dim * 2, hidden_dim * 2, num_layers, dropout)
        
        self.tgt_vocab_size = tgt_vocab_size
        
    def forward(self, src, tgt_input, src_lengths=None, teacher_forcing_ratio=1.0):
        batch_size = src.size(0)
        tgt_seq_len = tgt_input.size(1)
        
        # Encode source
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
        
        # Create source mask
        if src_lengths is not None:
            src_mask = torch.arange(src.size(1), device=src.device).expand(batch_size, src.size(1))
            src_mask = src_mask < src_lengths.unsqueeze(1)
        else:
            src_mask = None
        
        # Initialize decoder
        decoder_hidden = hidden
        decoder_cell = cell
        
        outputs = []
        input_token = tgt_input[:, 0].unsqueeze(1)  # [batch_size, 1] - Start token
        
        for t in range(tgt_seq_len):
            # Decoder step
            output, (decoder_hidden, decoder_cell), attention_weights = self.decoder(
                input_token, decoder_hidden, decoder_cell, encoder_outputs, src_mask
            )
            outputs.append(output)
            
            # Teacher forcing
            if t < tgt_seq_len - 1:
                use_teacher_forcing = random.random() < teacher_forcing_ratio
                if use_teacher_forcing:
                    input_token = tgt_input[:, t + 1].unsqueeze(1)
                else:
                    input_token = output.argmax(dim=1).unsqueeze(1)
        
        outputs = torch.stack(outputs, dim=1)  # [batch_size, tgt_seq_len, vocab_size]
        return outputs
    
    def translate(self, src, src_lengths=None, max_length=50, start_token=2, end_token=3):
        """Translate source sequence using greedy decoding"""
        self.eval()
        batch_size = src.size(0)
        device = src.device
        
        with torch.no_grad():
            # Encode source
            encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
            
            # Create source mask
            if src_lengths is not None:
                src_mask = torch.arange(src.size(1), device=device).expand(batch_size, src.size(1))
                src_mask = src_mask < src_lengths.unsqueeze(1)
            else:
                src_mask = None
            
            # Initialize decoder
            decoder_hidden = hidden
            decoder_cell = cell
            
            # Start with start token
            input_token = torch.full((batch_size, 1), start_token, device=device)
            
            translations = []
            for _ in range(max_length):
                # Decoder step
                output, (decoder_hidden, decoder_cell), _ = self.decoder(
                    input_token, decoder_hidden, decoder_cell, encoder_outputs, src_mask
                )
                
                # Get next token
                next_token = output.argmax(dim=1)
                translations.append(next_token)
                
                # Check for end token
                if (next_token == end_token).all():
                    break
                
                input_token = next_token.unsqueeze(1)
            
            translations = torch.stack(translations, dim=1)  # [batch_size, generated_length]
            return translations

# Transformer-based Translation Model
class TransformerTranslator(nn.Module):
    """Transformer model for machine translation"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, 
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, 
                 dropout=0.1, max_seq_length=5000):
        super().__init__()
        
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        # src: [batch_size, src_seq_len]
        # tgt: [batch_size, tgt_seq_len]
        
        # Embeddings and positional encoding
        src_embedded = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_embedded = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        # Create target mask (causal mask)
        tgt_seq_len = tgt.size(1)
        tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=tgt.device), diagonal=1)
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float('-inf'))
        
        # Transformer forward
        output = self.transformer(
            src_embedded, tgt_embedded,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Output projection
        logits = self.output_projection(output)
        
        return logits

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# Beam Search for Translation
class BeamSearch:
    """Beam search for sequence generation"""
    
    def __init__(self, model, beam_size=5, max_length=50, start_token=2, end_token=3, device='cuda'):
        self.model = model
        self.beam_size = beam_size
        self.max_length = max_length
        self.start_token = start_token
        self.end_token = end_token
        self.device = device
    
    def search(self, src, src_lengths=None):
        """Perform beam search translation"""
        self.model.eval()
        batch_size = src.size(0)
        
        with torch.no_grad():
            if isinstance(self.model, Seq2SeqWithAttention):
                return self._beam_search_rnn(src, src_lengths)
            elif isinstance(self.model, TransformerTranslator):
                return self._beam_search_transformer(src, src_lengths)
    
    def _beam_search_rnn(self, src, src_lengths=None):
        """Beam search for RNN-based model"""
        batch_size = src.size(0)
        
        # Encode source
        encoder_outputs, (hidden, cell) = self.model.encoder(src, src_lengths)
        
        # Create source mask
        if src_lengths is not None:
            src_mask = torch.arange(src.size(1), device=self.device).expand(batch_size, src.size(1))
            src_mask = src_mask < src_lengths.unsqueeze(1)
        else:
            src_mask = None
        
        # Initialize beams
        beams = []
        for b in range(batch_size):
            # Each beam: (sequence, log_prob, hidden_state, cell_state)
            initial_beam = (
                [self.start_token], 
                0.0, 
                hidden[:, b:b+1], 
                cell[:, b:b+1]
            )
            beams.append([initial_beam])
        
        # Beam search
        for step in range(self.max_length):
            new_beams = []
            
            for b in range(batch_size):
                candidates = []
                
                for sequence, log_prob, h_state, c_state in beams[b]:
                    if sequence[-1] == self.end_token:
                        candidates.append((sequence, log_prob, h_state, c_state))
                        continue
                    
                    # Get next token probabilities
                    input_token = torch.tensor([[sequence[-1]]], device=self.device)
                    encoder_out = encoder_outputs[b:b+1]
                    mask = src_mask[b:b+1] if src_mask is not None else None
                    
                    output, (new_h, new_c), _ = self.model.decoder(
                        input_token, h_state, c_state, encoder_out, mask
                    )
                    
                    log_probs = F.log_softmax(output, dim=1)
                    top_log_probs, top_indices = torch.topk(log_probs, self.beam_size)
                    
                    for i in range(self.beam_size):
                        new_sequence = sequence + [top_indices[0, i].item()]
                        new_log_prob = log_prob + top_log_probs[0, i].item()
                        candidates.append((new_sequence, new_log_prob, new_h, new_c))
                
                # Keep top beam_size candidates
                candidates.sort(key=lambda x: x[1], reverse=True)
                new_beams.append(candidates[:self.beam_size])
            
            beams = new_beams
        
        # Return best sequence for each batch item
        results = []
        for b in range(batch_size):
            best_sequence = max(beams[b], key=lambda x: x[1])[0]
            results.append(best_sequence[1:])  # Remove start token
        
        return results
    
    def _beam_search_transformer(self, src, src_lengths=None):
        """Beam search for Transformer model"""
        batch_size = src.size(0)
        
        # Create padding mask
        if src_lengths is not None:
            src_key_padding_mask = torch.arange(src.size(1), device=self.device).expand(batch_size, src.size(1))
            src_key_padding_mask = src_key_padding_mask >= src_lengths.unsqueeze(1)
        else:
            src_key_padding_mask = None
        
        # Initialize beams for each batch item
        beams = []
        for b in range(batch_size):
            beams.append([([self.start_token], 0.0)])
        
        # Beam search
        for step in range(self.max_length):
            new_beams = []
            
            for b in range(batch_size):
                candidates = []
                
                for sequence, log_prob in beams[b]:
                    if sequence[-1] == self.end_token:
                        candidates.append((sequence, log_prob))
                        continue
                    
                    # Prepare input
                    tgt = torch.tensor([sequence], device=self.device)
                    src_batch = src[b:b+1]
                    src_mask_batch = src_key_padding_mask[b:b+1] if src_key_padding_mask is not None else None
                    
                    # Forward pass
                    output = self.model(src_batch, tgt, src_key_padding_mask=src_mask_batch)
                    
                    # Get probabilities for last token
                    last_token_logits = output[0, -1, :]
                    log_probs = F.log_softmax(last_token_logits, dim=0)
                    
                    # Get top candidates
                    top_log_probs, top_indices = torch.topk(log_probs, self.beam_size)
                    
                    for i in range(self.beam_size):
                        new_sequence = sequence + [top_indices[i].item()]
                        new_log_prob = log_prob + top_log_probs[i].item()
                        candidates.append((new_sequence, new_log_prob))
                
                # Keep top beam_size candidates
                candidates.sort(key=lambda x: x[1], reverse=True)
                new_beams.append(candidates[:self.beam_size])
            
            beams = new_beams
        
        # Return best sequence for each batch item
        results = []
        for b in range(batch_size):
            best_sequence = max(beams[b], key=lambda x: x[1])[0]
            results.append(best_sequence[1:])  # Remove start token
        
        return results

# Translation Trainer
class TranslationTrainer:
    """Trainer for machine translation models"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            src = batch['src'].to(self.device)
            tgt_input = batch['tgt_input'].to(self.device)
            tgt_output = batch['tgt_output'].to(self.device)
            src_lengths = batch['src_lengths'].to(self.device)
            
            optimizer.zero_grad()
            
            if isinstance(self.model, Seq2SeqWithAttention):
                # RNN-based model
                output = self.model(src, tgt_input, src_lengths, teacher_forcing_ratio=0.5)
            else:
                # Transformer model
                src_key_padding_mask = (src == 0)
                tgt_key_padding_mask = (tgt_input == 0)
                output = self.model(src, tgt_input, src_key_padding_mask, tgt_key_padding_mask)
            
            # Compute loss
            output = output.view(-1, output.size(-1))
            tgt_output = tgt_output.view(-1)
            
            loss = criterion(output, tgt_output)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                src = batch['src'].to(self.device)
                tgt_input = batch['tgt_input'].to(self.device)
                tgt_output = batch['tgt_output'].to(self.device)
                src_lengths = batch['src_lengths'].to(self.device)
                
                if isinstance(self.model, Seq2SeqWithAttention):
                    output = self.model(src, tgt_input, src_lengths, teacher_forcing_ratio=0.0)
                else:
                    src_key_padding_mask = (src == 0)
                    tgt_key_padding_mask = (tgt_input == 0)
                    output = self.model(src, tgt_input, src_key_padding_mask, tgt_key_padding_mask)
                
                output = output.view(-1, output.size(-1))
                tgt_output = tgt_output.view(-1)
                
                loss = criterion(output, tgt_output)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, train_loader, val_loader, num_epochs=10, lr=1e-3):
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss = self.evaluate(val_loader, criterion)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print('-' * 50)
            
            scheduler.step()

# BLEU Score Evaluation
def compute_bleu_score(predictions, references, max_n=4):
    """Compute BLEU score for machine translation evaluation"""
    
    def get_ngrams(tokens, n):
        """Get n-grams from token list"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return ngrams
    
    def compute_precision(pred_ngrams, ref_ngrams):
        """Compute precision for n-grams"""
        if len(pred_ngrams) == 0:
            return 0.0
        
        # Count matches
        pred_counter = Counter(pred_ngrams)
        ref_counter = Counter(ref_ngrams)
        
        matches = 0
        for ngram, count in pred_counter.items():
            matches += min(count, ref_counter.get(ngram, 0))
        
        return matches / len(pred_ngrams)
    
    def compute_brevity_penalty(pred_len, ref_len):
        """Compute brevity penalty"""
        if pred_len > ref_len:
            return 1.0
        else:
            return math.exp(1 - ref_len / pred_len)
    
    # Compute BLEU for all n-grams
    total_precision = 0.0
    total_pred_len = 0
    total_ref_len = 0
    
    precisions = []
    
    for n in range(1, max_n + 1):
        total_matches = 0
        total_pred_ngrams = 0
        
        for pred, ref in zip(predictions, references):
            pred_ngrams = get_ngrams(pred, n)
            ref_ngrams = get_ngrams(ref, n)
            
            pred_counter = Counter(pred_ngrams)
            ref_counter = Counter(ref_ngrams)
            
            for ngram, count in pred_counter.items():
                total_matches += min(count, ref_counter.get(ngram, 0))
            
            total_pred_ngrams += len(pred_ngrams)
        
        if total_pred_ngrams > 0:
            precision = total_matches / total_pred_ngrams
        else:
            precision = 0.0
        
        precisions.append(precision)
    
    # Compute lengths
    for pred, ref in zip(predictions, references):
        total_pred_len += len(pred)
        total_ref_len += len(ref)
    
    # Compute BLEU score
    if any(p == 0 for p in precisions):
        return 0.0
    
    log_precisions = [math.log(p) for p in precisions]
    avg_log_precision = sum(log_precisions) / len(log_precisions)
    
    brevity_penalty = compute_brevity_penalty(total_pred_len, total_ref_len)
    
    bleu_score = brevity_penalty * math.exp(avg_log_precision)
    
    return bleu_score

# Utility functions
def build_vocab_from_sentences(sentences, min_freq=2, max_vocab=10000):
    """Build vocabulary from sentences"""
    word_counts = Counter()
    
    for sentence in sentences:
        words = sentence.lower().split()
        word_counts.update(words)
    
    # Start with special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
    
    # Add words meeting frequency threshold
    for word, count in word_counts.most_common(max_vocab - 4):
        if count >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab

def translate_sentence(model, sentence, src_vocab, tgt_vocab, device='cuda', beam_search=None):
    """Translate a single sentence"""
    model.eval()
    
    # Tokenize and encode
    tokens = sentence.lower().split()
    src_indices = [src_vocab.get('<BOS>', 2)] + [src_vocab.get(token, src_vocab.get('<UNK>', 1)) for token in tokens] + [src_vocab.get('<EOS>', 3)]
    
    # Convert to tensor
    src_tensor = torch.tensor([src_indices]).to(device)
    src_lengths = torch.tensor([len(src_indices)]).to(device)
    
    # Translate
    with torch.no_grad():
        if beam_search:
            translations = beam_search.search(src_tensor, src_lengths)
            translation_indices = translations[0]
        else:
            if hasattr(model, 'translate'):
                translations = model.translate(src_tensor, src_lengths)
                translation_indices = translations[0].cpu().tolist()
            else:
                # For transformer models, implement simple greedy decoding
                translation_indices = []
    
    # Convert back to words
    idx_to_word = {idx: word for word, idx in tgt_vocab.items()}
    translation_words = []
    
    for idx in translation_indices:
        word = idx_to_word.get(idx, '<UNK>')
        if word == '<EOS>':
            break
        if word not in ['<PAD>', '<BOS>']:
            translation_words.append(word)
    
    return ' '.join(translation_words)

if __name__ == "__main__":
    print("Testing machine translation models...")
    
    # Sample data (English to French)
    src_sentences = [
        "hello world",
        "how are you",
        "good morning",
        "thank you very much",
        "what is your name",
        "i am fine"
    ]
    
    tgt_sentences = [
        "bonjour monde",
        "comment allez vous",
        "bon matin",
        "merci beaucoup",
        "quel est votre nom",
        "je vais bien"
    ]
    
    # Build vocabularies
    src_vocab = build_vocab_from_sentences(src_sentences, min_freq=1)
    tgt_vocab = build_vocab_from_sentences(tgt_sentences, min_freq=1)
    
    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(tgt_vocab)}")
    
    # Create dataset
    dataset = TranslationDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_translation)
    
    # Test Seq2Seq model
    print("\nTesting Seq2Seq with Attention...")
    seq2seq_model = Seq2SeqWithAttention(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_dim=64,
        hidden_dim=128
    )
    
    print(f"Seq2Seq parameters: {sum(p.numel() for p in seq2seq_model.parameters()):,}")
    
    # Test forward pass
    for batch in dataloader:
        src = batch['src']
        tgt_input = batch['tgt_input']
        src_lengths = batch['src_lengths']
        
        output = seq2seq_model(src, tgt_input, src_lengths)
        print(f"Seq2Seq output shape: {output.shape}")
        break
    
    # Test Transformer model
    print("\nTesting Transformer...")
    transformer_model = TransformerTranslator(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256
    )
    
    print(f"Transformer parameters: {sum(p.numel() for p in transformer_model.parameters()):,}")
    
    # Test translation
    test_sentence = "hello world"
    translation = translate_sentence(seq2seq_model, test_sentence, src_vocab, tgt_vocab, device='cpu')
    print(f"Translation of '{test_sentence}': {translation}")
    
    # Test BLEU score
    predictions = [['bonjour', 'monde'], ['comment', 'vous']]
    references = [['bonjour', 'monde'], ['comment', 'allez', 'vous']]
    bleu = compute_bleu_score(predictions, references)
    print(f"BLEU score: {bleu:.4f}")
    
    # Test beam search
    print("\nTesting Beam Search...")
    beam_search = BeamSearch(seq2seq_model, beam_size=3, device='cpu')
    src_tensor = torch.tensor([[2, 4, 5, 3]]).to('cpu')  # Sample encoded sentence
    beam_results = beam_search.search(src_tensor)
    print(f"Beam search results: {beam_results}")
    
    print("Machine translation testing completed!")