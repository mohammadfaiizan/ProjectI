# Text Preprocessing, Embeddings, and Classification

## Table of Contents

- [Text Preprocessing Pipeline](#text-preprocessing-pipeline)
- [Tokenization Techniques](#tokenization-techniques)
- [Vocabulary Building and Numericalization](#vocabulary-building-and-numericalization)
- [nn.Embedding and nn.EmbeddingBag](#nnembedding-and-nnembeddingbag)
- [Pretrained Embeddings](#pretrained-embeddings)
- [Text Classification Architectures](#text-classification-architectures)
- [Padding, Packing, and Collate Functions](#padding-packing-and-collate-functions)

---

## Text Preprocessing Pipeline

The **text preprocessing pipeline** transforms raw text into model-ready tensors. The typical flow is: **cleaning** (lowercasing, removing HTML/URLs, punctuation), **tokenization** (splitting into units), **numericalization** (mapping tokens to indices), and **tensor conversion**.

### Cleaning and Normalization

Standard cleaning steps include converting to lowercase, removing HTML tags, URLs, and email addresses, stripping punctuation, and normalizing whitespace.

```python
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text
```

### Pipeline to Tensor Conversion

A complete pipeline combines cleaning, tokenization, vocabulary lookup, and padding/truncation to produce fixed-length tensors.

```python
import collections

class TextTokenizer:
    def __init__(self, vocab_size=10000, min_freq=1):
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>', 2: '<BOS>', 3: '<EOS>'}
        self.vocab_built = False

    def build_vocab(self, texts):
        word_counts = collections.Counter()
        for text in texts:
            cleaned = clean_text(text)
            word_counts.update(cleaned.split())
        for word, count in word_counts.most_common(self.vocab_size - 4):
            if count >= self.min_freq:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
        self.vocab_built = True

    def encode(self, text, max_length=None, add_special_tokens=True):
        tokens = clean_text(text).split()
        if add_special_tokens:
            tokens = ['<BOS>'] + tokens + ['<EOS>']
        indices = [self.word_to_idx.get(t, self.word_to_idx['<UNK>']) for t in tokens]
        if max_length:
            if len(indices) > max_length:
                indices = indices[:max_length]
                if add_special_tokens:
                    indices[-1] = self.word_to_idx['<EOS>']
            else:
                indices.extend([self.word_to_idx['<PAD>']] * (max_length - len(indices)))
        return indices
```

---

## Tokenization Techniques

### Word-Level Tokenization

**Word-level tokenization** splits text on whitespace. Simple and interpretable, but suffers from **out-of-vocabulary (OOV)** words and large vocabularies for morphologically rich languages.

```python
class WordTokenizer:
    def tokenize(self, text):
        text = text.lower().strip()
        return text.split()

    def encode(self, text, add_special_tokens=True):
        words = self.tokenize(text)
        if add_special_tokens:
            words = ['<BOS>'] + words + ['<EOS>']
        return [self.word_to_idx.get(w, self.word_to_idx['<UNK>']) for w in words]
```

### Character-Level Tokenization

**Character-level tokenization** treats each character as a token. Small vocabulary, handles OOV, but long sequences and weaker semantic units.

```python
class CharacterTokenizer:
    def encode(self, text, add_special_tokens=True):
        chars = list(text)
        if add_special_tokens:
            chars = ['<BOS>'] + chars + ['<EOS>']
        return [self.char_to_idx.get(c, self.char_to_idx['<UNK>']) for c in chars]
```

### Subword Tokenization: BPE

**Byte Pair Encoding (BPE)** iteratively merges the most frequent character pairs into subword units. Balances vocabulary size and sequence length; handles OOV via subword composition.

```python
def get_pairs(word_tokens):
    pairs = defaultdict(int)
    for word, freq in word_tokens.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

def merge_vocab(pair, word_tokens):
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\\S)' + bigram + r'(?!\\S)')
    return {p.sub(''.join(pair), w): f for w, f in word_tokens.items()}
```

### SentencePiece

**SentencePiece** is language-agnostic subword tokenization. Uses a unigram language model or BPE, treats space as a special character, and supports direct training from raw text.

```python
class SentencePieceTokenizer:
    def preprocess_text(self, text):
        return '▁' + text.replace(' ', '▁')

    def encode(self, text, add_special_tokens=True):
        processed = self.preprocess_text(text)
        tokens = processed.split('▁')[1:]
        if add_special_tokens:
            tokens = ['<s>'] + ['▁' + t for t in tokens if t] + ['</s>']
        return [self.token_to_idx.get(t, self.token_to_idx['<unk>']) for t in tokens]
```

### Tokenization Comparison

| Technique | Vocab Size | OOV Handling | Sequence Length | Use Case |
|----------|------------|--------------|----------------|----------|
| Word-level | Large | Poor | Short | Simple tasks |
| Character-level | Small | Excellent | Long | Morphology, OOV |
| BPE | Medium | Good | Medium | Translation, LM |
| SentencePiece | Medium | Good | Medium | Multilingual |
| WordPiece | Medium | Good | Medium | BERT, RoBERTa |

---

## Vocabulary Building and Numericalization

### Building a Vocabulary

**Vocabulary building** collects unique tokens, filters by frequency, and assigns indices. Special tokens (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`) are reserved.

```python
def build_vocab(texts, min_freq=2, max_vocab_size=10000):
    word_counts = {}
    for text in texts:
        for word in text.lower().split():
            word_counts[word] = word_counts.get(word, 0) + 1
    vocab = {'<PAD>': 0, '<UNK>': 1}
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    for word, count in sorted_words[:max_vocab_size - 2]:
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab
```

### Numericalization

**Numericalization** maps tokens to integer indices. Unknown tokens map to `<UNK>`. Truncation and padding ensure fixed-length sequences for batching.

```python
def numericalize(tokens, vocab, max_length=None, pad_token_id=0):
    indices = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    if max_length:
        if len(indices) > max_length:
            indices = indices[:max_length]
        else:
            indices.extend([pad_token_id] * (max_length - len(indices)))
    return indices
```

---

## nn.Embedding and nn.EmbeddingBag

### nn.Embedding

**nn.Embedding** maps discrete token indices to dense vectors. Key parameters: `vocab_size`, `embedding_dim`, `padding_idx` (zero gradients for padding), `max_norm` (optional L2 norm clipping).

```python
import torch.nn as nn

embedding = nn.Embedding(
    num_embeddings=10000,
    embedding_dim=128,
    padding_idx=0,
    max_norm=1.0,
    norm_type=2.0
)

input_ids = torch.randint(0, 10000, (32, 50))
embedded = embedding(input_ids)
print(embedded.shape)
```

| Parameter | Description |
|-----------|-------------|
| num_embeddings | Size of vocabulary |
| embedding_dim | Dimension of each embedding vector |
| padding_idx | Index to ignore in gradient updates |
| max_norm | Max L2 norm; None disables |
| scale_grad_by_freq | Scale gradients by inverse frequency |
| sparse | Use sparse gradient (for large vocab) |

### nn.EmbeddingBag

**nn.EmbeddingBag** computes embeddings for variable-length sequences without explicit padding. Supports `mean`, `sum`, or `max` reduction. Efficient for bag-of-words and sentence classification.

```python
embedding_bag = nn.EmbeddingBag(
    num_embeddings=10000,
    embedding_dim=128,
    mode='mean',
    padding_idx=0
)

offsets = torch.tensor([0, 10, 25])
input_ids = torch.randint(1, 10000, (35,))
output = embedding_bag(input_ids, offsets)
print(output.shape)
```

---

## Pretrained Embeddings

### Loading GloVe or Word2Vec

**Pretrained embeddings** (GoVe, Word2Vec, FastText) provide semantic representations. Load vectors and align with vocabulary indices.

```python
def load_pretrained_embeddings(path, vocab, embed_dim):
    word2vec = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = [float(x) for x in parts[1:]]
            if len(vec) == embed_dim:
                word2vec[word] = vec

    embedding_matrix = torch.randn(len(vocab), embed_dim) * 0.01
    for word, idx in vocab.items():
        if word in word2vec:
            embedding_matrix[idx] = torch.tensor(word2vec[word])
    return embedding_matrix
```

### Freezing vs Fine-Tuning

| Strategy | When to Use | Implementation |
|----------|-------------|----------------|
| Freeze | Limited data, preserve semantics | `embedding.weight.requires_grad = False` |
| Fine-tune | Task-specific semantics needed | Default `requires_grad=True` |
| Partial | Large pretrained, small task | Freeze first N layers, tune rest |

```python
embedding = nn.Embedding(vocab_size, embed_dim)
embedding.weight.data.copy_(pretrained_matrix)
embedding.weight.requires_grad = False
```

---

## Text Classification Architectures

### CNN for Text (TextCNN)

**TextCNN** applies multiple 1D convolutions with different kernel sizes over the embedding matrix, then max-pools and concatenates. Captures n-gram patterns efficiently.

```python
class CNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, num_filters=100, filter_sizes=[3, 4, 5], dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids).transpose(1, 2)
        conv_outputs = [F.relu(conv(x)).max(2)[0] for conv in self.convs]
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        return self.fc(x)
```

### Bag-of-Words

**Bag-of-words** ignores order; each document is a count or TF-IDF vector over the vocabulary. Simple baseline for classification.

```python
def create_ngram_tensor(texts, n=2, max_features=1000):
    all_ngrams = set()
    for text in texts:
        tokens = text.lower().split()
        for i in range(len(tokens) - n + 1):
            all_ngrams.add(' '.join(tokens[i:i+n]))
    ngram_vocab = {ng: i for i, ng in enumerate(list(all_ngrams)[:max_features])}
    features = torch.zeros(len(texts), len(ngram_vocab))
    for i, text in enumerate(texts):
        tokens = text.lower().split()
        for j in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[j:j+n])
            if ngram in ngram_vocab:
                features[i, ngram_vocab[ngram]] += 1
    return features, ngram_vocab
```

### RNN and Attention-Based Classifiers

**RNN** (LSTM/GRU) processes sequences; **bidirectional** captures context from both directions. **Attention** pools over the sequence with learned weights.

```python
class RNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=bidirectional, batch_first=True)
        rnn_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(rnn_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        rnn_out, _ = self.lstm(x)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(2).float()
            pooled = (rnn_out * mask).sum(1) / mask.sum(1)
        else:
            pooled = rnn_out[:, -1, :]
        return self.fc(pooled)
```

---

## Padding, Packing, and Collate Functions

### Padding

**Padding** appends a special token (usually index 0) to shorter sequences so all sequences in a batch have the same length.

```python
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    texts = [torch.tensor(item['text']) for item in batch]
    labels = torch.tensor([item['label'] for item in batch])
    lengths = torch.tensor([item['length'] for item in batch])
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    return {'texts': texts_padded, 'labels': labels, 'lengths': lengths}
```

### Packing for RNNs

**pack_padded_sequence** removes padding before the RNN forward pass for efficiency. **pad_packed_sequence** restores padding after the RNN for downstream layers.

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

embedded = self.embedding(x)
packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
lstm_out, (hidden, cell) = self.lstm(packed)
lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
```

### Collate Function for Text

A **collate function** batches samples, pads sequences, and creates attention masks.

```python
def collate_text_batch(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    attention_mask = (input_ids != 0).long()
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
```

### Mask Creation

| Mask Type | Purpose | Shape |
|-----------|---------|-------|
| Padding mask | Exclude padding from attention | `[batch, seq_len]` |
| Attention mask | Boolean for valid positions | `[batch, seq_len]` |
| Causal mask | Prevent attending to future | `[seq_len, seq_len]` |

```python
def create_padding_mask(sequences, pad_token_id=0):
    return (sequences != pad_token_id).float()

def create_causal_mask(seq_len, device='cpu'):
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask == 0
```
