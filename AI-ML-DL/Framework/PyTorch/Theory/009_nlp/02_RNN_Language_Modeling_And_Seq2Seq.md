# RNN, Language Modeling, and Sequence-to-Sequence

## Table of Contents

- [RNN/LSTM/GRU for NLP](#rnnlstmgru-for-nlp)
- [Language Modeling](#language-modeling)
- [Sequence-to-Sequence Models](#sequence-to-sequence-models)
- [Bahdanau and Luong Attention for Seq2Seq](#bahdanau-and-luong-attention-for-seq2seq)
- [Beam Search Decoding](#beam-search-decoding)

---

## RNN/LSTM/GRU for NLP

### Sentiment Analysis with LSTM

**LSTM** (Long Short-Term Memory) and **GRU** (Gated Recurrent Unit) capture long-range dependencies in sequences. For **sentiment analysis**, the final hidden state (or a pooled representation) is passed to a classifier.

```python
class LSTMSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=1, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True)
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, output_dim)

    def forward(self, x, lengths=None):
        embedded = self.embedding(x)
        if lengths is not None:
            packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out, (hidden, cell) = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (hidden, cell) = self.lstm(embedded)
        if self.lstm.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        return self.fc(hidden)
```

### Bidirectional and Multi-Layer RNNs

**Bidirectional** RNNs process the sequence forward and backward, concatenating outputs. **Multi-layer** RNNs stack layers for higher-level representations.

```python
self.lstm = nn.LSTM(
    embed_dim, hidden_dim, num_layers,
    dropout=dropout if num_layers > 1 else 0,
    bidirectional=True,
    batch_first=True
)
```

### LSTM with Attention for Sentiment

**Attention** computes a weighted sum over the RNN outputs, focusing on relevant tokens (e.g., sentiment-bearing words).

```python
class LSTMWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=bidirectional, batch_first=True)
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.fc = nn.Linear(lstm_output_dim, output_dim)

    def forward(self, x, lengths=None):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(packed if lengths else embedded)
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        if lengths is not None:
            mask = torch.arange(x.size(1), device=x.device).expand(x.size(0), x.size(1)) < lengths.unsqueeze(1)
            attention_weights = attention_weights.masked_fill(~mask.unsqueeze(2), 0)
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-10)
        attended = torch.sum(lstm_out * attention_weights, dim=1)
        return self.fc(attended), attention_weights
```

---

## Language Modeling

### Next-Token Prediction

**Language modeling** predicts the next token given previous tokens. The objective is to maximize the probability of the target sequence: \( P(x_t | x_{<t}) \).

```python
class CharLSTMLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        output = self.fc(lstm_out)
        return output, hidden
```

### Perplexity

**Perplexity** measures how well the model predicts the data. Lower is better. \( \text{Perplexity} = \exp(\text{cross-entropy loss}) \).

```python
def calculate_perplexity(model, dataloader, device='cuda'):
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            mask = targets != 0
            if mask.sum() > 0:
                loss = criterion(outputs[mask], targets[mask])
                total_loss += loss.item() * mask.sum().item()
                total_tokens += mask.sum().item()
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
```

### Teacher Forcing

**Teacher forcing** uses ground-truth tokens as decoder input during training instead of model predictions. Speeds convergence but can cause exposure bias. Often decayed over training.

```python
use_teacher_forcing = random.random() < teacher_forcing_ratio
if use_teacher_forcing and t < tgt_len - 1:
    decoder_input = tgt[:, t + 1].unsqueeze(1)
else:
    decoder_input = output.argmax(1).unsqueeze(1)
```

---

## Sequence-to-Sequence Models

### Encoder-Decoder Architecture

**Sequence-to-sequence (Seq2Seq)** uses an **encoder** to compress the source into a fixed representation and a **decoder** to generate the target autoregressively. The encoder's final hidden state initializes the decoder.

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)

    def forward(self, x, lengths=None):
        embedded = self.embedding(x)
        if lengths is not None:
            packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            outputs, (hidden, cell) = self.lstm(packed)
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, (hidden, cell) = self.lstm(embedded)
        hidden = hidden.view(self.num_layers, 2, hidden.size(1), -1)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)
        cell = cell.view(self.num_layers, 2, cell.size(1), -1)
        cell = torch.cat([cell[:, 0], cell[:, 1]], dim=2)
        return outputs, (hidden, cell)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim * 2, num_layers, batch_first=True)
        self.output_projection = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        return self.output_projection(output), (hidden, cell)
```

### Hidden State Passing

The **encoder** produces `(encoder_outputs, (hidden, cell))`. The **decoder** receives `(hidden, cell)` as its initial state. For bidirectional encoders, forward and backward final states are concatenated before passing to the decoder.

---

## Bahdanau and Luong Attention for Seq2Seq

### Bahdanau (Additive) Attention

**Bahdanau attention** uses a small MLP to compute energy scores between decoder hidden state and encoder outputs. The context vector is a weighted sum of encoder outputs.

```python
class BahdanauAttention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, attention_dim):
        super().__init__()
        self.encoder_projection = nn.Linear(encoder_hidden_dim, attention_dim)
        self.decoder_projection = nn.Linear(decoder_hidden_dim, attention_dim)
        self.attention_vector = nn.Linear(attention_dim, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        encoder_proj = self.encoder_projection(encoder_outputs)
        decoder_proj = self.decoder_projection(decoder_hidden).unsqueeze(1).expand(-1, encoder_outputs.size(1), -1)
        energy = torch.tanh(encoder_proj + decoder_proj)
        attention_scores = self.attention_vector(energy).squeeze(2)
        attention_weights = F.softmax(attention_scores, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention_weights
```

### Luong (Multiplicative) Attention

**Luong attention** uses dot product or a learned projection. Variants: **dot** (same dims), **general** (decoder projected to key dim), **concat** (concatenate and score).

```python
class LuongAttention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, attention_type='general'):
        super().__init__()
        self.attention_type = attention_type
        if attention_type == 'general':
            self.linear = nn.Linear(decoder_hidden_dim, encoder_hidden_dim)

    def forward(self, encoder_outputs, decoder_hidden):
        if self.attention_type == 'dot':
            attention_scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)
        elif self.attention_type == 'general':
            projected = self.linear(decoder_hidden)
            attention_scores = torch.bmm(encoder_outputs, projected.unsqueeze(2)).squeeze(2)
        attention_weights = F.softmax(attention_scores, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention_weights
```

### Attention Decoder

The **attention decoder** concatenates the context vector with the embedding (or LSTM output) before predicting the next token.

```python
class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, encoder_hidden_dim, attention_type='bahdanau'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(encoder_hidden_dim, hidden_dim, hidden_dim)
        self.lstm = nn.LSTM(embed_dim + encoder_hidden_dim, hidden_dim, batch_first=True)
        self.output_projection = nn.Linear(hidden_dim + encoder_hidden_dim, vocab_size)

    def forward(self, x, hidden, cell, encoder_outputs, src_mask=None):
        embedded = self.embedding(x)
        context, attention_weights = self.attention(encoder_outputs, hidden[-1])
        if src_mask is not None:
            attention_weights = attention_weights.masked_fill(~src_mask, 0)
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-10)
            context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        combined = torch.cat([output.squeeze(1), context], dim=1)
        return self.output_projection(combined), (hidden, cell), attention_weights
```

---

## Beam Search Decoding

### Greedy vs Beam Search

**Greedy decoding** selects the single most likely token at each step. **Beam search** maintains the top-k candidate sequences and extends each, keeping the best k overall.

```python
def beam_search(model, src, src_lengths, beam_size=5, max_length=50, start_token=1, end_token=2):
    model.eval()
    batch_size = src.size(0)
    encoder_outputs, (hidden, cell) = model.encoder(src, src_lengths)
    beams = [([start_token], 0.0, hidden, cell) for _ in range(beam_size)]

    for step in range(max_length - 1):
        candidates = []
        for sequence, log_prob, h, c in beams:
            if sequence[-1] == end_token:
                candidates.append((sequence, log_prob, h, c))
                continue
            input_token = torch.tensor([[sequence[-1]]], device=src.device)
            output, (new_h, new_c), _ = model.decoder(input_token, h, c, encoder_outputs, src_mask)
            log_probs = F.log_softmax(output, dim=-1)
            top_log_probs, top_indices = torch.topk(log_probs, beam_size)
            for i in range(beam_size):
                new_seq = sequence + [top_indices[0, i].item()]
                new_log_prob = log_prob + top_log_probs[0, i].item()
                candidates.append((new_seq, new_log_prob, new_h, new_c))
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_size]
        if all(b[0][-1] == end_token for b in beams):
            break

    return beams[0][0]
```

### Length Penalty

A **length penalty** discourages overly short sequences. Common form: \( \text{score} = \log P / (\text{length})^\alpha \).

```python
length_normalized_score = log_prob / (len(sequence) ** length_penalty)
```

### Seq2Seq Training Loop

```python
def train_seq2seq(model, batch, optimizer, criterion, teacher_forcing_ratio=0.5):
    src = batch['source']
    tgt = batch['target']
    tgt_output = batch['target_output']
    src_lengths = batch['source_lengths']
    outputs, _ = model(src, tgt, src_lengths, teacher_forcing_ratio)
    outputs = outputs.view(-1, outputs.size(-1))
    tgt_output = tgt_output.view(-1)
    loss = criterion(outputs, tgt_output)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()
```
