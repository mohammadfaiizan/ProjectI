#!/usr/bin/env python3
"""PyTorch Recurrent Layers Syntax - RNN, LSTM, GRU implementation and syntax"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== Recurrent Layers Overview ===")

print("RNN layers provide:")
print("1. Sequential data processing")
print("2. Memory through hidden states")
print("3. Variable-length sequence handling")
print("4. Temporal pattern recognition")
print("5. Different variants (RNN, LSTM, GRU)")

print("\n=== Basic RNN ===")

# Simple RNN
rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=1, batch_first=True)

# Input: (batch_size, seq_len, input_size)
input_rnn = torch.randn(5, 8, 10)
output, hidden = rnn(input_rnn)

print(f"RNN input: {input_rnn.shape}")
print(f"RNN output: {output.shape}")
print(f"RNN hidden: {hidden.shape}")

# With initial hidden state
h0 = torch.zeros(1, 5, 20)  # (num_layers, batch_size, hidden_size)
output_h0, hidden_h0 = rnn(input_rnn, h0)
print(f"RNN with h0 - output: {output_h0.shape}, hidden: {hidden_h0.shape}")

print("\n=== LSTM (Long Short-Term Memory) ===")

# Basic LSTM
lstm = nn.LSTM(input_size=15, hidden_size=25, num_layers=1, batch_first=True)

input_lstm = torch.randn(3, 12, 15)
output_lstm, (hidden_lstm, cell_lstm) = lstm(input_lstm)

print(f"LSTM input: {input_lstm.shape}")
print(f"LSTM output: {output_lstm.shape}")
print(f"LSTM hidden: {hidden_lstm.shape}")
print(f"LSTM cell: {cell_lstm.shape}")

# LSTM with initial states
h0_lstm = torch.zeros(1, 3, 25)
c0_lstm = torch.zeros(1, 3, 25)
output_init, (h_final, c_final) = lstm(input_lstm, (h0_lstm, c0_lstm))

print(f"LSTM with init states - output: {output_init.shape}")

print("\n=== GRU (Gated Recurrent Unit) ===")

# Basic GRU
gru = nn.GRU(input_size=12, hidden_size=30, num_layers=1, batch_first=True)

input_gru = torch.randn(4, 10, 12)
output_gru, hidden_gru = gru(input_gru)

print(f"GRU input: {input_gru.shape}")
print(f"GRU output: {output_gru.shape}")
print(f"GRU hidden: {hidden_gru.shape}")

print("\n=== Multi-layer RNNs ===")

# Multi-layer configurations
rnn_multi = nn.RNN(input_size=8, hidden_size=16, num_layers=3, batch_first=True)
lstm_multi = nn.LSTM(input_size=8, hidden_size=16, num_layers=2, batch_first=True)
gru_multi = nn.GRU(input_size=8, hidden_size=16, num_layers=4, batch_first=True)

input_multi = torch.randn(2, 6, 8)

# Multi-layer RNN
output_rnn_multi, hidden_rnn_multi = rnn_multi(input_multi)
print(f"Multi-layer RNN: layers={rnn_multi.num_layers}, hidden shape: {hidden_rnn_multi.shape}")

# Multi-layer LSTM
output_lstm_multi, (h_multi, c_multi) = lstm_multi(input_multi)
print(f"Multi-layer LSTM: layers={lstm_multi.num_layers}, hidden: {h_multi.shape}, cell: {c_multi.shape}")

# Multi-layer GRU
output_gru_multi, hidden_gru_multi = gru_multi(input_multi)
print(f"Multi-layer GRU: layers={gru_multi.num_layers}, hidden shape: {hidden_gru_multi.shape}")

print("\n=== Bidirectional RNNs ===")

# Bidirectional variants
rnn_bidir = nn.RNN(input_size=10, hidden_size=20, num_layers=1, bidirectional=True, batch_first=True)
lstm_bidir = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, bidirectional=True, batch_first=True)
gru_bidir = nn.GRU(input_size=10, hidden_size=20, num_layers=1, bidirectional=True, batch_first=True)

input_bidir = torch.randn(3, 7, 10)

# Bidirectional RNN
output_rnn_bidir, hidden_rnn_bidir = rnn_bidir(input_bidir)
print(f"Bidirectional RNN:")
print(f"  Output: {output_rnn_bidir.shape}")  # [batch, seq, hidden_size * 2]
print(f"  Hidden: {hidden_rnn_bidir.shape}")  # [num_layers * 2, batch, hidden_size]

# Bidirectional LSTM
output_lstm_bidir, (h_bidir, c_bidir) = lstm_bidir(input_bidir)
print(f"Bidirectional LSTM:")
print(f"  Output: {output_lstm_bidir.shape}")
print(f"  Hidden: {h_bidir.shape}, Cell: {c_bidir.shape}")

# Bidirectional GRU
output_gru_bidir, hidden_gru_bidir = gru_bidir(input_bidir)
print(f"Bidirectional GRU:")
print(f"  Output: {output_gru_bidir.shape}")
print(f"  Hidden: {hidden_gru_bidir.shape}")

print("\n=== RNN Parameters and Options ===")

# Different activation functions
rnn_tanh = nn.RNN(input_size=8, hidden_size=16, nonlinearity='tanh', batch_first=True)
rnn_relu = nn.RNN(input_size=8, hidden_size=16, nonlinearity='relu', batch_first=True)

# Dropout
rnn_dropout = nn.RNN(input_size=8, hidden_size=16, num_layers=3, dropout=0.2, batch_first=True)
lstm_dropout = nn.LSTM(input_size=8, hidden_size=16, num_layers=2, dropout=0.3, batch_first=True)

# Bias
rnn_no_bias = nn.RNN(input_size=8, hidden_size=16, bias=False, batch_first=True)

input_params = torch.randn(2, 5, 8)

print("RNN parameter variations:")
print(f"RNN tanh: {rnn_tanh(input_params)[0].shape}")
print(f"RNN relu: {rnn_relu(input_params)[0].shape}")
print(f"RNN dropout: {rnn_dropout(input_params)[0].shape}")
print(f"LSTM dropout: {lstm_dropout(input_params)[0].shape}")
print(f"RNN no bias: {rnn_no_bias(input_params)[0].shape}")

print("\n=== Custom RNN Implementations ===")

class SimpleRNNCell(nn.Module):
    """Custom RNN cell implementation"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.weight_ih = nn.Linear(input_size, hidden_size)
        self.weight_hh = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
    
    def forward(self, input, hidden):
        return self.activation(self.weight_ih(input) + self.weight_hh(hidden))

class CustomRNN(nn.Module):
    """Custom RNN using RNN cells"""
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            input_dim = input_size if i == 0 else hidden_size
            self.cells.append(SimpleRNNCell(input_dim, hidden_size))
    
    def forward(self, input, hidden=None):
        batch_size, seq_len, _ = input.size()
        
        if hidden is None:
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=input.device)
        
        outputs = []
        
        for t in range(seq_len):
            x = input[:, t, :]
            new_hidden = []
            
            for layer in range(self.num_layers):
                x = self.cells[layer](x, hidden[layer])
                new_hidden.append(x)
            
            hidden = torch.stack(new_hidden)
            outputs.append(x)
        
        output = torch.stack(outputs, dim=1)
        return output, hidden

# Test custom RNN
custom_rnn = CustomRNN(input_size=10, hidden_size=20, num_layers=2)
custom_input = torch.randn(3, 8, 10)
custom_output, custom_hidden = custom_rnn(custom_input)

print(f"Custom RNN output: {custom_output.shape}")
print(f"Custom RNN hidden: {custom_hidden.shape}")

print("\n=== LSTM Cell Implementation ===")

class CustomLSTMCell(nn.Module):
    """Custom LSTM cell implementation"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Gates: input, forget, cell, output
        self.W_ii = nn.Linear(input_size, hidden_size)
        self.W_if = nn.Linear(input_size, hidden_size)
        self.W_ig = nn.Linear(input_size, hidden_size)
        self.W_io = nn.Linear(input_size, hidden_size)
        
        self.W_hi = nn.Linear(hidden_size, hidden_size)
        self.W_hf = nn.Linear(hidden_size, hidden_size)
        self.W_hg = nn.Linear(hidden_size, hidden_size)
        self.W_ho = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, input, hidden_cell):
        h_prev, c_prev = hidden_cell
        
        # Gates
        i_gate = torch.sigmoid(self.W_ii(input) + self.W_hi(h_prev))
        f_gate = torch.sigmoid(self.W_if(input) + self.W_hf(h_prev))
        g_gate = torch.tanh(self.W_ig(input) + self.W_hg(h_prev))
        o_gate = torch.sigmoid(self.W_io(input) + self.W_ho(h_prev))
        
        # Cell and hidden states
        c_new = f_gate * c_prev + i_gate * g_gate
        h_new = o_gate * torch.tanh(c_new)
        
        return h_new, c_new

# Test custom LSTM cell
custom_lstm_cell = CustomLSTMCell(input_size=8, hidden_size=16)
cell_input = torch.randn(4, 8)
h_init = torch.zeros(4, 16)
c_init = torch.zeros(4, 16)

h_new, c_new = custom_lstm_cell(cell_input, (h_init, c_init))
print(f"Custom LSTM cell - h: {h_new.shape}, c: {c_new.shape}")

print("\n=== Sequence Processing Patterns ===")

class Seq2SeqEncoder(nn.Module):
    """Sequence-to-sequence encoder"""
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        return hidden, cell  # Return final states

class Seq2SeqDecoder(nn.Module):
    """Sequence-to-sequence decoder"""
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.output_projection = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden_state):
        output, (hidden, cell) = self.lstm(x, hidden_state)
        output = self.output_projection(output)
        return output, (hidden, cell)

class Seq2OneClassifier(nn.Module):
    """Sequence to single output classifier"""
    def __init__(self, input_size, hidden_size, num_classes, pooling='last'):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.pooling = pooling
        
        # Bidirectional doubles the hidden size
        classifier_input = hidden_size * 2
        self.classifier = nn.Linear(classifier_input, num_classes)
    
    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        
        if self.pooling == 'last':
            # Use last time step
            pooled = output[:, -1, :]
        elif self.pooling == 'max':
            # Max pooling over time
            pooled, _ = torch.max(output, dim=1)
        elif self.pooling == 'mean':
            # Average pooling over time
            pooled = torch.mean(output, dim=1)
        else:
            pooled = output[:, -1, :]
        
        return self.classifier(pooled)

# Test sequence processing patterns
encoder = Seq2SeqEncoder(input_size=20, hidden_size=32, num_layers=2)
decoder = Seq2SeqDecoder(input_size=25, hidden_size=32, output_size=30, num_layers=2)
classifier = Seq2OneClassifier(input_size=15, hidden_size=24, num_classes=5, pooling='mean')

# Encoder test
encoder_input = torch.randn(4, 10, 20)
enc_hidden, enc_cell = encoder(encoder_input)
print(f"Encoder states - hidden: {enc_hidden.shape}, cell: {enc_cell.shape}")

# Decoder test
decoder_input = torch.randn(4, 8, 25)
dec_output, (dec_hidden, dec_cell) = decoder(decoder_input, (enc_hidden, enc_cell))
print(f"Decoder output: {dec_output.shape}")

# Classifier test
classifier_input = torch.randn(6, 12, 15)
class_output = classifier(classifier_input)
print(f"Classifier output: {class_output.shape}")

print("\n=== Variable Length Sequences ===")

# Pack and unpack sequences for variable lengths
def pack_sequences_example():
    """Example of handling variable length sequences"""
    
    # Sample sequences of different lengths
    seq1 = torch.randn(10, 8)  # Length 10
    seq2 = torch.randn(7, 8)   # Length 7
    seq3 = torch.randn(12, 8)  # Length 12
    
    # Pad sequences to same length
    max_len = max(seq1.size(0), seq2.size(0), seq3.size(0))
    
    padded_seq1 = F.pad(seq1, (0, 0, 0, max_len - seq1.size(0)))
    padded_seq2 = F.pad(seq2, (0, 0, 0, max_len - seq2.size(0)))
    padded_seq3 = F.pad(seq3, (0, 0, 0, max_len - seq3.size(0)))
    
    # Stack into batch
    batch = torch.stack([padded_seq1, padded_seq2, padded_seq3])
    lengths = torch.tensor([seq1.size(0), seq2.size(0), seq3.size(0)])
    
    print(f"Padded batch shape: {batch.shape}")
    print(f"Sequence lengths: {lengths}")
    
    # Pack sequences
    packed = nn.utils.rnn.pack_padded_sequence(
        batch, lengths, batch_first=True, enforce_sorted=False
    )
    
    print(f"Packed data shape: {packed.data.shape}")
    print(f"Batch sizes: {packed.batch_sizes}")
    
    return packed

# Test packing
print("Variable length sequence handling:")
packed_sequences = pack_sequences_example()

# Use packed sequences with LSTM
lstm_packed = nn.LSTM(input_size=8, hidden_size=16, batch_first=True)
output_packed, (hidden_packed, cell_packed) = lstm_packed(packed_sequences)

print(f"LSTM output on packed: {output_packed.data.shape}")

# Unpack the output
output_unpacked, output_lengths = nn.utils.rnn.pad_packed_sequence(
    output_packed, batch_first=True
)
print(f"Unpacked output: {output_unpacked.shape}")
print(f"Output lengths: {output_lengths}")

print("\n=== RNN for Different Tasks ===")

class LanguageModel(nn.Module):
    """Simple language model using LSTM"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x, hidden=None):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.lstm(embedded, hidden)
        output = self.output_projection(output)
        return output, hidden

class SentimentClassifier(nn.Module):
    """Sentiment classification using bidirectional LSTM"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, _) = self.lstm(embedded)
        
        # Concatenate final hidden states from both directions
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        final_hidden = self.dropout(final_hidden)
        
        return self.classifier(final_hidden)

# Test different tasks
vocab_size, embed_size, hidden_size = 1000, 128, 256

# Language model
lm = LanguageModel(vocab_size, embed_size, hidden_size)
lm_input = torch.randint(0, vocab_size, (4, 20))  # (batch, seq_len)
lm_output, lm_hidden = lm(lm_input)
print(f"Language Model: {lm_input.shape} -> {lm_output.shape}")

# Sentiment classifier
sentiment_clf = SentimentClassifier(vocab_size, embed_size, hidden_size)
sentiment_input = torch.randint(0, vocab_size, (8, 15))
sentiment_output = sentiment_clf(sentiment_input)
print(f"Sentiment Classifier: {sentiment_input.shape} -> {sentiment_output.shape}")

print("\n=== RNN Best Practices ===")

print("RNN Guidelines:")
print("1. Use LSTM/GRU instead of vanilla RNN for longer sequences")
print("2. Consider bidirectional RNNs for non-causal tasks")
print("3. Use dropout between layers (but not within LSTM/GRU)")
print("4. Gradient clipping helps with exploding gradients")
print("5. Pack sequences for variable lengths")
print("6. Initialize hidden states carefully")
print("7. Consider attention mechanisms for long sequences")

print("\nArchitecture Choices:")
print("- LSTM: Complex gating, good for long sequences")
print("- GRU: Simpler than LSTM, fewer parameters")
print("- Bidirectional: Better for non-causal tasks")
print("- Multi-layer: Increased capacity but harder to train")
print("- Residual connections: Help with deep RNNs")

print("\nTraining Tips:")
print("- Use teacher forcing for sequence generation")
print("- Clip gradients (norm clipping ~1.0)")
print("- Use learning rate scheduling")
print("- Pack sequences for efficiency")
print("- Consider truncated backpropagation")
print("- Monitor gradient norms")

print("\nCommon Issues:")
print("- Vanishing gradients in long sequences")
print("- Exploding gradients (use clipping)")
print("- Slow training compared to transformers")
print("- Difficulty parallelizing training")
print("- Hidden state initialization")
print("- Variable sequence length handling")

print("\n=== Recurrent Layers Complete ===") 