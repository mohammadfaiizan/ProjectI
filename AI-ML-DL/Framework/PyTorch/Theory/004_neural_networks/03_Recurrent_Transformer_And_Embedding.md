# Recurrent, Transformer, and Embedding Layers

## Table of Contents

- [Recurrent Layers](#recurrent-layers)
- [Transformer Components](#transformer-components)
- [Embedding Layers](#embedding-layers)

---

## Recurrent Layers

Recurrent layers process **sequential data** with **hidden state** memory. PyTorch provides RNN, LSTM, and GRU. Use `batch_first=True` for input shape `(batch, seq_len, features)`.

### Input Format

| batch_first | Input Shape | Hidden Shape |
|-------------|-------------|--------------|
| False (default) | (seq_len, batch, input_size) | (num_layers, batch, hidden_size) |
| True | (batch, seq_len, input_size) | (num_layers, batch, hidden_size) |

### RNN

```python
import torch
import torch.nn as nn

rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=1, batch_first=True)
x = torch.randn(5, 8, 10)
output, hidden = rnn(x)

print(output.shape)
print(hidden.shape)
```

### LSTM

LSTM returns both hidden state and cell state. Better for long sequences than vanilla RNN.

```python
lstm = nn.LSTM(input_size=15, hidden_size=25, num_layers=1, batch_first=True)
x = torch.randn(3, 12, 15)
output, (hidden, cell) = lstm(x)

print(output.shape)
print(hidden.shape)
print(cell.shape)
```

### GRU

GRU has fewer parameters than LSTM and often performs comparably.

```python
gru = nn.GRU(input_size=12, hidden_size=30, num_layers=1, batch_first=True)
x = torch.randn(4, 10, 12)
output, hidden = gru(x)
```

### Bidirectional RNNs

Bidirectional layers process the sequence in both directions. Output size is `hidden_size * 2`.

```python
lstm_bidir = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, bidirectional=True, batch_first=True)
x = torch.randn(3, 7, 10)
output, (h, c) = lstm_bidir(x)

print(output.shape)
```

### Multi-layer and Dropout

```python
lstm_multi = nn.LSTM(input_size=8, hidden_size=16, num_layers=3, dropout=0.2, batch_first=True)
```

### Initial Hidden State

```python
h0 = torch.zeros(1, 5, 20)
output, hidden = rnn(x, h0)

h0_lstm = torch.zeros(1, 3, 25)
c0_lstm = torch.zeros(1, 3, 25)
output, (h, c) = lstm(x, (h0_lstm, c0_lstm))
```

### Packed Sequences for Variable Length

Use `pack_padded_sequence` and `pad_packed_sequence` for variable-length sequences to avoid computing on padding.

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

lengths = torch.tensor([10, 7, 12])
packed = pack_padded_sequence(batch, lengths, batch_first=True, enforce_sorted=False)
output_packed, (h, c) = lstm(packed)
output, output_lengths = pad_packed_sequence(output_packed, batch_first=True)
```

### Sequence-to-One Classifier

```python
class Seq2OneClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, pooling='last'):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.pooling = pooling
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        output, _ = self.lstm(x)
        if self.pooling == 'last':
            pooled = output[:, -1, :]
        elif self.pooling == 'mean':
            pooled = output.mean(dim=1)
        elif self.pooling == 'max':
            pooled, _ = output.max(dim=1)
        return self.classifier(pooled)
```

---

## Transformer Components

Transformers use **self-attention** instead of recurrence. Key components: MultiheadAttention, positional encoding, feed-forward networks, and layer normalization.

### nn.Transformer

PyTorch provides a complete `nn.Transformer` module. For custom architectures, use individual components.

```python
transformer = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6)
```

### MultiheadAttention

```python
attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
x = torch.randn(4, 20, 512)
attn_output, attn_weights = attn(x, x, x)
```

### Scaled Dot-Product Attention

```python
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return torch.matmul(weights, value), weights
```

### Positional Encoding

```python
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

### Transformer Encoder Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x
```

### Transformer Decoder Block

Decoder blocks include self-attention, cross-attention to encoder output, and feed-forward layers.

```python
class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, memory, self_mask=None, memory_mask=None):
        attn_out, _ = self.self_attn(x, x, x, attn_mask=self_mask)
        x = self.norm1(x + attn_out)
        cross_out, _ = self.cross_attn(x, memory, memory, attn_mask=memory_mask)
        x = self.norm2(x + cross_out)
        x = self.norm3(x + self.ff(x))
        return x
```

### Masking Utilities

```python
def create_padding_mask(seq, pad_token=0):
    return (seq != pad_token).unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(size):
    return torch.triu(torch.ones(size, size), diagonal=1) == 0
```

---

## Embedding Layers

**nn.Embedding** maps discrete token indices to dense vectors. **nn.EmbeddingBag** efficiently computes bag-of-embeddings with sum, mean, or max aggregation.

### nn.Embedding

```python
embedding = nn.Embedding(vocab_size=10000, embedding_dim=300)
x = torch.tensor([[1, 5, 3, 8], [2, 7, 4, 0]])
out = embedding(x)
print(out.shape)
```

### Embedding Parameters

```python
embedding_pad = nn.Embedding(10000, 300, padding_idx=0)
embedding_norm = nn.Embedding(10000, 300, max_norm=1.0)
embedding_sparse = nn.Embedding(10000, 300, sparse=True)
```

### nn.EmbeddingBag

Efficient for variable-length sequences. Input: flattened indices and offsets marking the start of each sequence.

```python
embedding_bag = nn.EmbeddingBag(vocab_size=10000, embedding_dim=300, mode='mean')
indices = torch.tensor([1, 2, 3, 5, 6, 8, 9])
offsets = torch.tensor([0, 3, 5])
out = embedding_bag(indices, offsets)
```

### EmbeddingBag Modes

```python
embed_sum = nn.EmbeddingBag(100, 50, mode='sum')
embed_mean = nn.EmbeddingBag(100, 50, mode='mean')
embed_max = nn.EmbeddingBag(100, 50, mode='max')
```

### Pretrained Embeddings

```python
def load_pretrained(embedding_layer, pretrained_weights):
    with torch.no_grad():
        embedding_layer.weight.copy_(pretrained_weights)
    embedding_layer.weight.requires_grad = False
```

### Scaling Embeddings (Transformer-style)

```python
import math

class ScaledEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.scale = math.sqrt(embed_size)

    def forward(self, x):
        return self.embedding(x) * self.scale
```

### Positional Embeddings

```python
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len, embed_size):
        super().__init__()
        self.pe = nn.Embedding(max_len, embed_size)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        return x + self.pe(positions)
```
