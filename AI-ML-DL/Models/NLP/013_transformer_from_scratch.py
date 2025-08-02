"""
ERA 5: THE TRANSFORMER REVOLUTION (2017)
=========================================

Year: 2017
Paper: "Attention Is All You Need" - Vaswani et al.
Innovation: Complete elimination of recurrence and convolution
Previous Limitation: Sequential processing prevents parallelization
Performance Gain: Fully parallel training, superior long-range dependencies
Impact: Revolutionary architecture that became foundation for modern NLP

This file implements the original Transformer architecture that completely
changed NLP by showing that attention mechanisms alone are sufficient for
state-of-the-art sequence transduction, without any recurrence or convolution.
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
import math
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

YEAR = "2017"
INNOVATION = "Transformer: Attention Is All You Need"
PREVIOUS_LIMITATION = "RNNs require sequential processing, CNNs have limited receptive fields"
IMPACT = "Revolutionary pure-attention architecture, foundation for GPT/BERT/T5"

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
                    if 5 <= len(tokens) <= 30:  # Good length for Transformer
                        sentences.append(tokens)
        return sentences
    
    train_sentences = preprocess_text(dataset['train']['text'])
    val_sentences = preprocess_text(dataset['validation']['text'])
    test_sentences = preprocess_text(dataset['test']['text'])
    
    print(f"Train sentences: {len(train_sentences):,}")
    print(f"Validation sentences: {len(val_sentences):,}")
    print(f"Test sentences: {len(test_sentences):,}")
    
    return train_sentences, val_sentences, test_sentences

def build_vocabulary(sentences, vocab_size=8000):
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
# DATASET CLASS FOR TRANSFORMER EXPERIMENTS
# ============================================================================

class TransformerDataset(Dataset):
    """Dataset for Transformer experiments"""
    
    def __init__(self, sentences, vocab, task='translation', max_length=25):
        self.vocab = vocab
        self.task = task
        self.max_length = max_length
        self.pairs = []
        
        if task == 'translation':
            # Create synthetic translation pairs
            self._create_translation_pairs(sentences)
        elif task == 'language_modeling':
            # Language modeling task
            self._create_lm_pairs(sentences)
        elif task == 'summarization':
            # Extractive summarization
            self._create_summarization_pairs(sentences)
        
        print(f"Created {len(self.pairs)} {task} pairs")
    
    def _create_translation_pairs(self, sentences):
        """Create synthetic translation pairs for demonstration"""
        # Simple transformations to simulate translation
        transformations = {
            'word_order': lambda s: s[::-1],  # Reverse word order
            'synonym_replace': lambda s: self._replace_with_synonyms(s),
            'prefix_add': lambda s: ['start'] + s,
            'suffix_add': lambda s: s + ['end'],
            'case_change': lambda s: [w.upper() if len(w) > 3 else w for w in s]
        }
        
        for sentence in sentences:
            if len(sentence) <= self.max_length:
                # Apply random transformation
                transform_name = random.choice(list(transformations.keys()))
                transformed = transformations[transform_name](sentence[:])
                
                if len(transformed) <= self.max_length:
                    self.pairs.append((sentence, transformed))
    
    def _replace_with_synonyms(self, sentence):
        """Simple synonym replacement"""
        synonyms = {
            'good': 'great', 'bad': 'poor', 'big': 'large', 'small': 'tiny',
            'fast': 'quick', 'slow': 'sluggish', 'happy': 'joyful', 'sad': 'gloomy'
        }
        return [synonyms.get(word, word) for word in sentence]
    
    def _create_lm_pairs(self, sentences):
        """Create language modeling pairs"""
        for sentence in sentences:
            if len(sentence) <= self.max_length:
                input_seq = sentence[:-1] if len(sentence) > 1 else sentence
                target_seq = sentence[1:] if len(sentence) > 1 else sentence
                self.pairs.append((input_seq, target_seq))
    
    def _create_summarization_pairs(self, sentences):
        """Create summarization pairs"""
        for sentence in sentences:
            if len(sentence) >= 6:
                # Extract key words (simplified extractive summarization)
                summary_length = min(3, len(sentence) // 2)
                # Take first word, longest word, and last word
                summary = [sentence[0]]
                if len(sentence) > 2:
                    longest = max(sentence[1:-1], key=len, default='')
                    if longest:
                        summary.append(longest)
                summary.append(sentence[-1])
                
                self.pairs.append((sentence, summary[:summary_length]))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.pairs[idx]
        
        # Convert to indices
        input_indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in input_seq]
        
        if self.task == 'language_modeling':
            target_indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in target_seq]
        else:
            # Add SOS and EOS for seq2seq tasks
            target_indices = ([self.vocab['<SOS>']] + 
                            [self.vocab.get(word, self.vocab['<UNK>']) for word in target_seq] + 
                            [self.vocab['<EOS>']])
        
        return torch.tensor(input_indices, dtype=torch.long), torch.tensor(target_indices, dtype=torch.long)

def collate_transformer_fn(batch):
    """Collate function for Transformer data"""
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
# POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Positional encoding from the original Transformer paper
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model, max_length=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        # Create the div_term for positional encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_length, batch_size, d_model)
        Returns:
            x with positional encoding added
        """
        return x + self.pe[:x.size(0), :]

# ============================================================================
# MULTI-HEAD ATTENTION
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism from the Transformer paper
    Allows the model to jointly attend to information from different 
    representation subspaces at different positions
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, query, key, value, mask=None):
        """
        Compute scaled dot-product attention
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, value)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: (batch_size, seq_length, d_model)
            mask: (batch_size, seq_length, seq_length) optional
            
        Returns:
            output: (batch_size, seq_length, d_model)
            attention_weights: (batch_size, num_heads, seq_length, seq_length)
        """
        batch_size, seq_length, d_model = query.size()
        
        # Linear projections and reshape for multi-head
        Q = self.w_q(query).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
        # Expand mask for multiple heads
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        
        # Apply scaled dot-product attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_length, d_model
        )
        
        # Final linear projection
        output = self.w_o(attention_output)
        
        return output, attention_weights

# ============================================================================
# POSITION-WISE FEED FORWARD NETWORK
# ============================================================================

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed Forward Network
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length, d_model)
        Returns:
            output: (batch_size, seq_length, d_model)
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

# ============================================================================
# TRANSFORMER ENCODER LAYER
# ============================================================================

class TransformerEncoderLayer(nn.Module):
    """
    Single layer of the Transformer encoder
    Contains multi-head self-attention and position-wise feed-forward
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_length, d_model)
            mask: (batch_size, seq_length, seq_length) optional
            
        Returns:
            output: (batch_size, seq_length, d_model)
            attention_weights: attention weights from self-attention
        """
        # Self-attention with residual connection and layer norm
        attn_output, attention_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        output = self.norm2(x + self.dropout(ff_output))
        
        return output, attention_weights

# ============================================================================
# TRANSFORMER DECODER LAYER
# ============================================================================

class TransformerDecoderLayer(nn.Module):
    """
    Single layer of the Transformer decoder
    Contains masked self-attention, encoder-decoder attention, and feed-forward
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.encoder_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, self_mask=None, encoder_mask=None):
        """
        Args:
            x: (batch_size, target_length, d_model) decoder input
            encoder_output: (batch_size, source_length, d_model) encoder output
            self_mask: (batch_size, target_length, target_length) causal mask
            encoder_mask: (batch_size, target_length, source_length) padding mask
            
        Returns:
            output: (batch_size, target_length, d_model)
            self_attention_weights: self-attention weights
            encoder_attention_weights: encoder-decoder attention weights
        """
        # Masked self-attention
        self_attn_output, self_attention_weights = self.self_attention(x, x, x, self_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Encoder-decoder attention
        enc_attn_output, encoder_attention_weights = self.encoder_attention(
            x, encoder_output, encoder_output, encoder_mask
        )
        x = self.norm2(x + self.dropout(enc_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        output = self.norm3(x + self.dropout(ff_output))
        
        return output, self_attention_weights, encoder_attention_weights

# ============================================================================
# TRANSFORMER ENCODER
# ============================================================================

class TransformerEncoder(nn.Module):
    """
    Complete Transformer encoder stack
    """
    
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, 
                 d_ff=2048, max_length=5000, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_length)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None):
        """
        Args:
            src: (batch_size, src_length) source token indices
            src_mask: (batch_size, src_length, src_length) optional mask
            
        Returns:
            output: (batch_size, src_length, d_model)
            attention_weights: List of attention weights from each layer
        """
        # Embedding + positional encoding
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)  # (src_length, batch_size, d_model)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, src_length, d_model)
        x = self.dropout(x)
        
        # Apply encoder layers
        all_attention_weights = []
        for layer in self.layers:
            x, attention_weights = layer(x, src_mask)
            all_attention_weights.append(attention_weights)
        
        return x, all_attention_weights

# ============================================================================
# TRANSFORMER DECODER
# ============================================================================

class TransformerDecoder(nn.Module):
    """
    Complete Transformer decoder stack
    """
    
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6,
                 d_ff=2048, max_length=5000, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_length)
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, tgt, encoder_output, tgt_mask=None, src_mask=None):
        """
        Args:
            tgt: (batch_size, tgt_length) target token indices
            encoder_output: (batch_size, src_length, d_model) encoder output
            tgt_mask: (batch_size, tgt_length, tgt_length) causal mask
            src_mask: (batch_size, tgt_length, src_length) padding mask
            
        Returns:
            logits: (batch_size, tgt_length, vocab_size)
            self_attention_weights: List of self-attention weights
            encoder_attention_weights: List of encoder-decoder attention weights
        """
        # Embedding + positional encoding
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)  # (tgt_length, batch_size, d_model)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, tgt_length, d_model)
        x = self.dropout(x)
        
        # Apply decoder layers
        all_self_attention_weights = []
        all_encoder_attention_weights = []
        
        for layer in self.layers:
            x, self_attn_weights, enc_attn_weights = layer(x, encoder_output, tgt_mask, src_mask)
            all_self_attention_weights.append(self_attn_weights)
            all_encoder_attention_weights.append(enc_attn_weights)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits, all_self_attention_weights, all_encoder_attention_weights

# ============================================================================
# COMPLETE TRANSFORMER MODEL
# ============================================================================

class Transformer(nn.Module):
    """
    Complete Transformer model from "Attention Is All You Need"
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, 
                 max_length=5000, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, num_heads, num_encoder_layers, d_ff, max_length, dropout
        )
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, num_heads, num_decoder_layers, d_ff, max_length, dropout
        )
        
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
    def create_padding_mask(self, seq, pad_token=0):
        """Create padding mask for sequences"""
        return (seq != pad_token).unsqueeze(1).unsqueeze(2)
    
    def create_causal_mask(self, size, device):
        """Create causal (look-ahead) mask for decoder"""
        mask = torch.tril(torch.ones(size, size, device=device))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self, src, tgt, src_lengths=None, tgt_lengths=None):
        """
        Args:
            src: (batch_size, src_length) source sequences
            tgt: (batch_size, tgt_length) target sequences
            src_lengths: (batch_size,) source sequence lengths
            tgt_lengths: (batch_size,) target sequence lengths
            
        Returns:
            logits: (batch_size, tgt_length, tgt_vocab_size)
            encoder_attention_weights: attention weights from encoder
            decoder_self_attention_weights: self-attention weights from decoder
            decoder_encoder_attention_weights: encoder-decoder attention weights
        """
        batch_size, src_length = src.size()
        tgt_length = tgt.size(1)
        device = src.device
        
        # Create masks
        src_padding_mask = self.create_padding_mask(src)  # (batch_size, 1, 1, src_length)
        tgt_padding_mask = self.create_padding_mask(tgt)  # (batch_size, 1, 1, tgt_length)
        tgt_causal_mask = self.create_causal_mask(tgt_length, device)  # (1, 1, tgt_length, tgt_length)
        
        # Combine causal and padding masks for decoder self-attention
        tgt_mask = tgt_padding_mask & tgt_causal_mask
        
        # Encoder-decoder attention mask (decoder can attend to all encoder positions)
        src_tgt_mask = src_padding_mask.expand(-1, -1, tgt_length, -1)
        
        # Encode
        encoder_output, encoder_attention_weights = self.encoder(src, src_padding_mask)
        
        # Decode
        logits, decoder_self_attention_weights, decoder_encoder_attention_weights = self.decoder(
            tgt, encoder_output, tgt_mask, src_tgt_mask
        )
        
        return logits, encoder_attention_weights, decoder_self_attention_weights, decoder_encoder_attention_weights
    
    def generate(self, src, src_lengths, vocab, idx_to_word, max_length=50, beam_size=1):
        """
        Generate sequences using the trained Transformer
        Implements greedy decoding (beam_size=1) or beam search
        """
        self.eval()
        device = src.device
        batch_size = src.size(0)
        
        with torch.no_grad():
            # Encode source
            src_padding_mask = self.create_padding_mask(src)
            encoder_output, _ = self.encoder(src, src_padding_mask)
            
            # Initialize decoder input with SOS token
            decoder_input = torch.tensor([[vocab['<SOS>']] for _ in range(batch_size)], device=device)
            
            # Generate tokens one by one
            for _ in range(max_length):
                # Create causal mask for current sequence
                tgt_length = decoder_input.size(1)
                tgt_causal_mask = self.create_causal_mask(tgt_length, device)
                tgt_padding_mask = self.create_padding_mask(decoder_input)
                tgt_mask = tgt_padding_mask & tgt_causal_mask
                
                # Encoder-decoder attention mask
                src_tgt_mask = src_padding_mask.expand(-1, -1, tgt_length, -1)
                
                # Decode
                logits, _, _ = self.decoder(decoder_input, encoder_output, tgt_mask, src_tgt_mask)
                
                # Get next token (greedy)
                next_token_logits = logits[:, -1, :]  # Last position
                next_tokens = next_token_logits.argmax(dim=-1).unsqueeze(1)
                
                # Check for EOS token
                if next_tokens[0].item() == vocab['<EOS>']:
                    break
                
                # Append to sequence
                decoder_input = torch.cat([decoder_input, next_tokens], dim=1)
            
            # Convert to words
            generated_sequences = []
            for i in range(batch_size):
                sequence = decoder_input[i].tolist()
                words = []
                for token_idx in sequence[1:]:  # Skip SOS
                    if token_idx == vocab['<EOS>']:
                        break
                    if token_idx in idx_to_word:
                        words.append(idx_to_word[token_idx])
                generated_sequences.append(words)
        
        return generated_sequences

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_transformer(model, train_loader, val_loader, epochs=10, learning_rate=0.0001):
    """Train Transformer model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Use Adam with learning rate schedule (simplified version of original)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    train_losses = []
    val_losses = []
    
    print(f"Training Transformer on device: {device}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (src, src_lengths, tgt, tgt_lengths) in enumerate(train_loader):
            src = src.to(device)
            tgt = tgt.to(device)
            
            # Prepare decoder input and target
            decoder_input = tgt[:, :-1]  # All but last token
            decoder_target = tgt[:, 1:]   # All but first token
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, _, _, _ = model(src, decoder_input, src_lengths, tgt_lengths)
            
            # Compute loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), decoder_target.reshape(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_loss = evaluate_transformer(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return train_losses, val_losses

def evaluate_transformer(model, data_loader, criterion, device):
    """Evaluate Transformer model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for src, src_lengths, tgt, tgt_lengths in data_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            decoder_input = tgt[:, :-1]
            decoder_target = tgt[:, 1:]
            
            logits, _, _, _ = model(src, decoder_input, src_lengths, tgt_lengths)
            loss = criterion(logits.reshape(-1, logits.size(-1)), decoder_target.reshape(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

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
# ATTENTION VISUALIZATION
# ============================================================================

def visualize_transformer_attention(tokens, attention_weights, layer_idx=0, head_idx=0, 
                                  attention_type='encoder', save_path=None):
    """Visualize Transformer attention patterns"""
    if attention_type == 'encoder':
        # Encoder self-attention
        attn_matrix = attention_weights[layer_idx][0, head_idx].cpu().numpy()
    else:
        # Decoder self-attention or encoder-decoder attention
        attn_matrix = attention_weights[layer_idx][0, head_idx].cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn_matrix,
        xticklabels=tokens[:attn_matrix.shape[1]],
        yticklabels=tokens[:attn_matrix.shape[0]],
        cmap='Blues',
        cbar=True,
        square=True
    )
    
    plt.title(f'Transformer {attention_type.title()} Attention (Layer {layer_idx}, Head {head_idx})')
    plt.xlabel('Key Positions')
    plt.ylabel('Query Positions')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return plt.gcf()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"=== The Transformer Revolution ({YEAR}) ===")
    print(f"Demonstrating: {INNOVATION}")
    print("="*60)
    
    # Load dataset
    train_sentences, val_sentences, test_sentences = load_wikitext2_dataset()
    
    # Use subset for demonstration
    train_subset = train_sentences[:1000]
    val_subset = val_sentences[:200]
    test_subset = test_sentences[:200]
    
    # Build vocabulary
    vocab, idx_to_word = build_vocabulary(train_subset, vocab_size=4000)
    
    results = []
    training_histories = {}
    
    # Test Transformer on different tasks
    tasks = ['translation', 'language_modeling']
    
    for task in tasks[:1]:  # Test one task for demo
        print(f"\n" + "="*50)
        print(f"Training Transformer for {task.upper()}")
        
        # Create datasets
        train_dataset = TransformerDataset(train_subset, vocab, task=task)
        val_dataset = TransformerDataset(val_subset, vocab, task=task)
        test_dataset = TransformerDataset(test_subset, vocab, task=task)
        
        # Create data loaders
        batch_size = 16  # Smaller batch size for Transformer
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                collate_fn=collate_transformer_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_transformer_fn)
        
        # Initialize Transformer
        transformer = Transformer(
            src_vocab_size=len(vocab),
            tgt_vocab_size=len(vocab),
            d_model=256,      # Reduced for demo
            num_heads=8,
            num_encoder_layers=3,  # Reduced for demo
            num_decoder_layers=3,
            d_ff=512,         # Reduced for demo
            dropout=0.1
        )
        
        print(f"Transformer parameters: {count_parameters(transformer):,}")
        
        # Train Transformer
        model_name = f'Transformer-{task}'
        metrics = track_computational_metrics(
            model_name,
            train_transformer,
            transformer, train_loader, val_loader, 6, 0.0001
        )
        
        train_losses, val_losses = metrics['result']
        training_histories[model_name] = (train_losses, val_losses)
        
        result = {
            'model': model_name,
            'task': task,
            'year': '2017',
            'final_loss': val_losses[-1] if val_losses else 0,
            'parameters': count_parameters(transformer),
            'training_time': metrics['training_time_minutes'],
            'memory_usage': metrics['memory_usage_mb'],
            'innovation': 'Pure attention-based architecture, no recurrence or convolution'
        }
        results.append(result)
        
        # Demonstrate generation
        print(f"\n{task.upper()} EXAMPLES:")
        print("="*30)
        
        transformer.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        transformer.to(device)
        
        for i in range(3):
            if i < len(test_dataset):
                input_seq, target_seq = test_dataset[i]
                
                # Convert to words
                input_words = [idx_to_word[idx.item()] for idx in input_seq 
                             if idx.item() in idx_to_word and idx.item() != 0]
                target_words = [idx_to_word[idx.item()] for idx in target_seq[1:-1] 
                              if idx.item() in idx_to_word and idx.item() not in [0, 2, 3]]
                
                # Generate
                input_tensor = input_seq.unsqueeze(0).to(device)
                input_lengths = torch.tensor([len(input_seq)]).to(device)
                
                generated = transformer.generate(input_tensor, input_lengths, vocab, idx_to_word)
                
                print(f"  Input:     {' '.join(input_words)}")
                print(f"  Target:    {' '.join(target_words)}")
                print(f"  Generated: {' '.join(generated[0] if generated else [])}")
                print()
        
        # Visualize attention (if we have examples)
        if len(test_dataset) > 0:
            print("\nGenerating Attention Visualizations...")
            input_seq, _ = test_dataset[0]
            input_tensor = input_seq.unsqueeze(0).to(device)
            input_lengths = torch.tensor([len(input_seq)]).to(device)
            
            # Create a dummy target for visualization
            dummy_target = torch.tensor([[vocab['<SOS>'], vocab.get('the', vocab['<UNK>'])]], device=device)
            
            with torch.no_grad():
                _, encoder_attn, decoder_self_attn, decoder_enc_attn = transformer(
                    input_tensor, dummy_target, input_lengths, torch.tensor([2])
                )
            
            # Get input words for visualization
            input_words = [idx_to_word[idx.item()] for idx in input_seq[:10] 
                         if idx.item() in idx_to_word and idx.item() != 0]
            
            if len(input_words) > 2 and len(encoder_attn) > 0:
                # Encoder self-attention
                for layer_idx in range(min(2, len(encoder_attn))):
                    for head_idx in range(min(2, encoder_attn[layer_idx].size(1))):
                        save_path = f'AI-ML-DL/Models/NLP/013_transformer_encoder_L{layer_idx}_H{head_idx}.png'
                        visualize_transformer_attention(
                            input_words, encoder_attn, layer_idx, head_idx, 'encoder', save_path
                        )
                        print(f"Encoder attention saved: {save_path}")
    
    # Create comprehensive visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss curves
    ax = axes[0, 0]
    for model_name, (train_losses, _) in training_histories.items():
        if train_losses:
            ax.plot(train_losses, label=model_name, linewidth=2)
    ax.set_title('Transformer Training Loss', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation loss curves
    ax = axes[0, 1]
    for model_name, (_, val_losses) in training_histories.items():
        if val_losses:
            ax.plot(val_losses, label=model_name, linewidth=2)
    ax.set_title('Transformer Validation Loss', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Model architecture comparison
    ax = axes[1, 0]
    transformer_components = ['Multi-Head\nAttention', 'Position-wise\nFFN', 'Positional\nEncoding', 
                            'Layer\nNormalization', 'Residual\nConnections']
    component_importance = [10, 8, 7, 9, 8]  # Relative importance scores
    bars = ax.bar(transformer_components, component_importance, 
                  color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
    ax.set_title('Transformer Component Importance', fontsize=14)
    ax.set_ylabel('Importance Score')
    ax.tick_params(axis='x', rotation=45)
    
    # Revolutionary impact timeline
    ax = axes[1, 1]
    years = ['Pre-2017', '2017', '2018', '2019', '2020+']
    impact_scores = [2, 10, 15, 20, 25]  # Cumulative impact
    ax.plot(years, impact_scores, marker='o', linewidth=3, markersize=8, color='#E74C3C')
    ax.fill_between(years, impact_scores, alpha=0.3, color='#E74C3C')
    ax.set_title('Transformer Revolution Impact', fontsize=14)
    ax.set_ylabel('Cumulative Impact Score')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('AI-ML-DL/Models/NLP/013_transformer_revolution_results.png', dpi=300, bbox_inches='tight')
    print("\nComprehensive visualization saved: 013_transformer_revolution_results.png")
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("🚀 THE TRANSFORMER REVOLUTION SUMMARY 🚀")
    print("="*70)
    
    for result in results:
        print(f"\n{result['model']} ({result['year']}):")
        print(f"  📊 Task: {result['task']}")
        print(f"  📉 Final Loss: {result['final_loss']:.4f}")
        print(f"  🔢 Parameters: {result['parameters']:,}")
        print(f"  ⏱️  Training Time: {result['training_time']:.2f} minutes")
        print(f"  💾 Memory Usage: {result['memory_usage']:.2f} MB")
        print(f"  💡 Innovation: {result['innovation']}")
    
    print("\n🧠 TRANSFORMER ARCHITECTURE COMPONENTS:")
    print("="*50)
    print("1. 🎯 Multi-Head Attention: Parallel attention mechanisms")
    print("2. 🔄 Positional Encoding: Sine/cosine position information")
    print("3. 🍰 Layer Normalization: Stable training for deep networks")
    print("4. 🔗 Residual Connections: Gradient flow and training stability")
    print("5. 🏗️  Position-wise FFN: Non-linear transformations")
    print("6. 📚 Encoder-Decoder: Separate encoding and generation")
    print("7. 🎭 Masked Attention: Causal masking for generation")
    
    print("\n⚡ REVOLUTIONARY BREAKTHROUGHS:")
    print("="*50)
    print("• 🚫 NO RECURRENCE: Eliminates sequential dependencies")
    print("• 🚫 NO CONVOLUTION: Pure attention-based processing")
    print("• ⚡ FULL PARALLELIZATION: Massive training speedup")
    print("• 🎯 GLOBAL ATTENTION: Direct connections between all positions")
    print("• 🧮 SCALABLE ARCHITECTURE: Efficient for long sequences")
    print("• 🎨 MULTI-HEAD DESIGN: Multiple attention representations")
    print("• 📏 POSITION ENCODING: Explicit position information")
    
    print("\n🌟 HISTORICAL IMPACT:")
    print("="*50)
    print("• 📖 PAPER: 'Attention Is All You Need' - Most cited NLP paper")
    print("• 🏗️  FOUNDATION: Basis for GPT, BERT, T5, and all modern LLMs")
    print("• 🔄 PARADIGM SHIFT: From RNN/CNN to pure attention")
    print("• 🚀 SCALABILITY: Enabled training of massive language models")
    print("• 🌐 UNIVERSALITY: Applicable to vision, audio, multimodal tasks")
    print("• 📈 PERFORMANCE: State-of-the-art on all NLP benchmarks")
    
    print("\n🔬 TECHNICAL INNOVATIONS:")
    print("="*50)
    print("• Scaled Dot-Product Attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V")
    print("• Multi-Head Attention: h heads attending to different subspaces")
    print("• Positional Encoding: PE(pos,2i) = sin(pos/10000^(2i/d_model))")
    print("• Layer Normalization: Stable training for deep architectures")
    print("• Residual Connections: Skip connections for gradient flow")
    print("• Teacher Forcing: Parallel training for sequence generation")
    
    print("\n🎯 WHY IT WORKED:")
    print("="*50)
    print("1. 🔗 DIRECT CONNECTIONS: Every position can attend to every other")
    print("2. ⚡ PARALLELIZATION: No sequential bottlenecks")
    print("3. 🎯 FOCUSED ATTENTION: Learns what to focus on")
    print("4. 📏 POSITION AWARENESS: Explicit position information")
    print("5. 🧮 COMPUTATIONAL EFFICIENCY: Efficient matrix operations")
    print("6. 🎨 REPRESENTATIONAL POWER: Multiple attention heads")
    print("7. 📈 SCALABILITY: Performance improves with scale")
    
    print("\n🌈 COMPARISON WITH PREVIOUS ERAS:")
    print("="*50)
    print("ERA 1 (N-grams):    🐌 Limited context, no learning")
    print("ERA 2 (RNNs):       🐢 Sequential processing, vanishing gradients")
    print("ERA 3 (Seq2Seq):    🚶 Attention helps, but still sequential")
    print("ERA 4 (Self-Attn):  🏃 Self-attention emerges, foundation set")
    print("ERA 5 (Transformer): 🚀 PURE ATTENTION, PARALLEL, REVOLUTIONARY!")
    
    print("\n📊 BEFORE vs AFTER TRANSFORMER:")
    print("="*50)
    print("BEFORE 2017:")
    print("  • RNN/LSTM dominated sequence modeling")
    print("  • Sequential processing limited parallelization")
    print("  • Attention was auxiliary to RNNs")
    print("  • Limited ability to handle long sequences")
    print("  • Complex architectures for different tasks")
    
    print("\nAFTER 2017:")
    print("  • Attention became the primary mechanism")
    print("  • Fully parallel training and inference")
    print("  • Universal architecture for multiple tasks")
    print("  • Superior long-range dependency modeling")
    print("  • Foundation for all modern language models")
    
    print("\n🎓 EDUCATIONAL INSIGHTS:")
    print("="*50)
    print("• The Transformer didn't just improve performance - it changed everything")
    print("• Pure attention proved more powerful than hybrid approaches")
    print("• Parallelization enabled the scale that drives modern AI")
    print("• Simple, elegant architecture with profound implications")
    print("• Demonstrates how fundamental breakthroughs reshape entire fields")
    
    print(f"\n{'='*70}")
    print("🎉 TRANSFORMER REVOLUTION: THE MOMENT THAT CHANGED AI 🎉")
    print(f"{'='*70}")
    
    return results

if __name__ == "__main__":
    results = main()