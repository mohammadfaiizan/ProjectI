# Transformer, BERT, and GPT

## Table of Contents

- [Attention Mechanisms](#attention-mechanisms)
- [Building a Transformer from Scratch](#building-a-transformer-from-scratch)
- [BERT](#bert)
- [GPT](#gpt)

---

## Attention Mechanisms

### Scaled Dot-Product Attention

**Scaled dot-product attention** computes \( \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \). The \( \sqrt{d_k} \) scaling prevents softmax gradients from vanishing when \( d_k \) is large.

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights
```

### Multi-Head Attention

**Multi-head attention** runs scaled dot-product attention in parallel with different linear projections for Q, K, V. Outputs are concatenated and projected. Enables attending to different representation subspaces.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(attention_output), attention_weights
```

### Self-Attention vs Cross-Attention

| Type | Query | Key/Value | Use Case |
|------|-------|-----------|----------|
| Self-attention | Same sequence | Same sequence | Encode context within one sequence |
| Cross-attention | Decoder | Encoder | Attend from decoder to encoder (Seq2Seq) |

### Causal and Padding Masks

**Causal mask** prevents attending to future positions (lower triangular). **Padding mask** excludes padding tokens from attention.

```python
def create_causal_mask(size, device):
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
    return mask == 0

def create_padding_mask(sequences, pad_token_id=0):
    return (sequences != pad_token_id).unsqueeze(1).unsqueeze(2)
```

---

## Building a Transformer from Scratch

### Positional Encoding

**Positional encoding** injects position information. Sinusoidal: \( PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d}) \), \( PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d}) \). Learnable embeddings are an alternative.

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
```

### Encoder Block

Each **encoder block** has multi-head self-attention (with residual + LayerNorm) followed by a position-wise feed-forward network (with residual + LayerNorm).

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
```

### Decoder Block

The **decoder block** adds cross-attention: self-attention (causal) on target, then cross-attention to encoder outputs, then feed-forward.

```python
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        self_attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
```

### Full Transformer Model

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, max_len=5000, dropout=0.1):
        super().__init__()
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
```

---

## BERT

### Masked Language Modeling

**Masked Language Modeling (MLM)** randomly masks 15% of tokens. 80% replaced with [MASK], 10% random token, 10% unchanged. The model predicts the original token at masked positions.

```python
def create_masked_lm_predictions(tokens, vocab_size, mask_prob=0.15, mask_token_id=103):
    labels = tokens.clone()
    rand = torch.rand(tokens.shape)
    mask_arr = (rand < mask_prob) & (tokens != 0) & (tokens != 101) & (tokens != 102)
    for idx in mask_arr.nonzero():
        pos = (idx // tokens.size(1), idx % tokens.size(1))
        if torch.rand(1) < 0.8:
            tokens[pos] = mask_token_id
        elif torch.rand(1) < 0.5:
            tokens[pos] = torch.randint(0, vocab_size, (1,)).item()
    labels[~mask_arr] = -100
    return tokens, labels
```

### Next Sentence Prediction

**Next Sentence Prediction (NSP)** is a binary classification: given two sentences A and B, predict whether B follows A. Uses the [CLS] token representation.

### BERT Architecture

**BERT** is encoder-only. Combines **token embeddings**, **positional embeddings**, and **segment (token type) embeddings**. The [CLS] token's representation is used for classification tasks.

```python
class BERTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        position_ids = self.position_ids[:, :seq_length] if position_ids is None else position_ids
        token_type_ids = torch.zeros_like(input_ids) if token_type_ids is None else token_type_ids
        embeddings = self.word_embeddings(input_ids) + self.position_embeddings(position_ids) + self.token_type_embeddings(token_type_ids)
        return self.dropout(self.layer_norm(embeddings))
```

### Fine-Tuning for Downstream Tasks

**Fine-tuning** adds a task-specific head on top of BERT. For classification: pool [CLS], add dropout, linear layer to num_labels.

```python
class BERTForSequenceClassification(nn.Module):
    def __init__(self, bert_model, num_labels=2):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(bert_model.config.hidden_dropout_prob)
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits
```

### [CLS] Token

The **[CLS]** token is prepended to every input. Its final hidden state is used as the aggregate sequence representation for classification, NSP, and similar tasks.

---

## GPT

### Causal/Autoregressive Language Model

**GPT** is a **decoder-only** transformer. It predicts the next token given all previous tokens. Training uses a **causal mask** so each position attends only to earlier positions.

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
```

### Decoder-Only Architecture

**GPT** stacks transformer decoder layers (self-attention + feed-forward) with no encoder or cross-attention. Uses learned positional embeddings.

```python
class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        return logits, None
```

### Generation

**Autoregressive generation** samples one token at a time, appending it to the context. Supports temperature, top-k, and top-p (nucleus) sampling.

```python
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

### BERT vs GPT Summary

| Aspect | BERT | GPT |
|--------|------|-----|
| Architecture | Encoder-only | Decoder-only |
| Pre-training | MLM + NSP | Causal LM |
| Attention | Bidirectional | Causal (masked) |
| Primary use | Understanding, classification | Generation |
| [CLS] token | Yes | No (use last token) |
