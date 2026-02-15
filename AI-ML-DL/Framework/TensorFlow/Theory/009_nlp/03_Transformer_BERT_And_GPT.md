# Transformer, BERT, and GPT

## Table of Contents

1. [Attention Mechanisms](#1-attention-mechanisms)
2. [Transformer Architecture](#2-transformer-architecture)
3. [BERT](#3-bert)
4. [GPT](#4-gpt)

---

## 1. Attention Mechanisms

### Bahdanau (Additive) Attention

- **Query**: Decoder hidden state.
- **Key/Value**: Encoder hidden states.
- **Score**: `v^T tanh(W [query; key])` — learned compatibility.

```python
concat = tf.concat([query_tiled, values], axis=-1)
score = tf.keras.layers.Dense(1)(tf.keras.activations.tanh(tf.keras.layers.Dense(units)(concat)))
attn_weights = tf.nn.softmax(score, axis=1)
context = tf.reduce_sum(attn_weights * values, axis=1)
```

### Luong (Dot-Product) Attention

- **Score**: `query^T * key` (or scaled).
- Simpler, no learned projection for scoring.

```python
score = tf.matmul(query, values, transpose_b=True)
attn_weights = tf.nn.softmax(score, axis=-1)
context = tf.matmul(attn_weights, values)
```

### Scaled Dot-Product Attention

- **Score**: `(Q K^T) / sqrt(d_k)`.
- Used in Transformers; scaling prevents softmax saturation.

```python
scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(d_k, tf.float32))
attn_weights = tf.nn.softmax(scores, axis=-1)
output = tf.matmul(attn_weights, v)
```

### Multi-Head Attention

- Project Q, K, V into multiple subspaces (heads).
- Compute attention in parallel per head.
- Concatenate and project.

```python
mha = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)
out = mha(query, value, key=key)
```

| Mechanism | Complexity | Use Case |
|-----------|------------|----------|
| Bahdanau | O(n * d^2) | RNN seq2seq |
| Luong | O(n * d) | RNN seq2seq |
| Scaled dot-product | O(n^2 * d) | Transformer |
| Multi-head | O(n^2 * d) | Transformer |

---

## 2. Transformer Architecture

### Encoder Block

1. **Multi-head self-attention** with residual and LayerNorm.
2. **Feed-forward network** (two linear layers with activation) with residual and LayerNorm.

```python
attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
x = tf.keras.layers.LayerNormalization()(x + attn)
ffn = tf.keras.Sequential([
    tf.keras.layers.Dense(ff_dim, activation="relu"),
    tf.keras.layers.Dense(d_model)
])(x)
x = tf.keras.layers.LayerNormalization()(x + ffn)
```

### Decoder Block

1. **Masked multi-head self-attention** (causal mask).
2. **Cross-attention** (decoder attends to encoder output).
3. **Feed-forward network**.

```python
causal_mask = (1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)) * -1e9
self_attn = MultiHeadAttention(...)(dec, dec, attention_mask=causal_mask)
cross_attn = MultiHeadAttention(...)(dec, enc_out)
ffn = ...
```

### Positional Encoding

- Add sinusoidal or learnable positional embeddings to token embeddings.
- Enables the model to use token order.

### Full Transformer

- Stack N encoder and M decoder blocks.
- Encoder: input -> embedding + PE -> encoder blocks.
- Decoder: output -> embedding + PE -> decoder blocks -> linear -> softmax.

---

## 3. BERT

**BERT** (Bidirectional Encoder Representations from Transformers) is an encoder-only model pretrained with **Masked Language Modeling (MLM)** and **Next Sentence Prediction (NSP)**.

### Architecture

- Stack of Transformer encoder blocks.
- ** [CLS] token** at position 0 for sequence-level tasks.
- ** [SEP] token** to separate segments.

### Masked Language Modeling

- Randomly mask ~15% of tokens.
- Predict masked tokens from context (bidirectional).

```python
mlm_logits = tf.keras.layers.Dense(vocab_size)(encoder_output)
loss = sparse_categorical_crossentropy(masked_positions, mlm_logits_at_masked)
```

### Fine-Tuning Pattern

- **Classification**: Use [CLS] output, add a dense layer.
- **Token-level**: Use full sequence output (e.g., NER, QA).

```python
cls_output = encoder_output[:, 0, :]
logits = tf.keras.layers.Dense(num_classes, activation="softmax")(cls_output)
```

### Key Properties

| Property | BERT |
|----------|------|
| Direction | Bidirectional |
| Pretraining | MLM, NSP |
| Output | Encoder hidden states |
| Use case | Classification, NER, QA, etc. |

---

## 4. GPT

**GPT** (Generative Pre-trained Transformer) is a **decoder-only** model with **causal (autoregressive)** attention.

### Architecture

- Stack of Transformer decoder blocks (no cross-attention).
- **Causal mask**: Each position attends only to previous positions.

```python
causal_mask = (1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)) * -1e9
attn = MultiHeadAttention(...)(x, x, attention_mask=causal_mask)
```

### Pretraining

- **Next-token prediction**: Predict `t_{i+1}` given `t_1, ..., t_i`.
- Trained on large text corpora.

### Generation

- Autoregressive: sample next token, append, repeat.
- Use **temperature**, **top-k**, **top-p** for diverse sampling.

### Key Properties

| Property | GPT |
|----------|-----|
| Direction | Causal (left-to-right) |
| Pretraining | Next-token prediction |
| Output | Next-token logits |
| Use case | Generation, completion |

### Layer Normalization Placement

Transformers use **Pre-LN** (normalize before sublayer) or **Post-LN** (normalize after). Pre-LN often trains more stably. BERT and GPT use Post-LN; some modern variants use Pre-LN.
