# Recurrent Layers, Transformers, and Embeddings

## Table of Contents

1. [Recurrent Layers](#1-recurrent-layers)
2. [Attention and Transformers](#2-attention-and-transformers)
3. [Embedding Layers](#3-embedding-layers)

---

## 1. Recurrent Layers

Recurrent layers process **sequences** by maintaining a hidden state that carries information across time steps.

### SimpleRNN

The simplest RNN. Hidden state: `h_t = activation(W_h * h_{t-1} + W_x * x_t + b)`.

```python
x = tf.random.normal((2, 10, 32))
rnn = tf.keras.layers.SimpleRNN(64, return_sequences=False)
out = rnn(x)
print(out.shape)  # (2, 64)
```

### LSTM

Long Short-Term Memory. Addresses vanishing gradients with **gates** (forget, input, output) and a **cell state**.

```python
lstm = tf.keras.layers.LSTM(64, return_sequences=True, return_state=True)
out, h, c = lstm(x)
print(out.shape)  # (2, 10, 64) - full sequence
print(h.shape)   # (2, 64) - hidden state
print(c.shape)   # (2, 64) - cell state
```

### GRU

Gated Recurrent Unit. Simpler than LSTM (reset and update gates), often comparable performance with fewer parameters.

```python
gru = tf.keras.layers.GRU(64, return_sequences=False)
out = gru(x)
print(out.shape)  # (2, 64)
```

### return_sequences and return_state

| Parameter | Effect |
|-----------|--------|
| return_sequences=False | Return only last timestep output |
| return_sequences=True | Return output for every timestep |
| return_state=True | Also return hidden (and cell for LSTM) state |

Use `return_sequences=True` when stacking RNN layers or when you need per-timestep outputs.

### Bidirectional

Processes the sequence in both **forward** and **backward** directions, concatenating outputs. Doubles the output dimension.

```python
bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))
out = bi_lstm(x)
print(out.shape)  # (2, 64) - 32*2 from concatenation
```

### Stacked RNNs

```python
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(None, 32)),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

---

## 2. Attention and Transformers

### MultiHeadAttention

**Scaled dot-product attention**: `Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V`

**Multi-head** splits Q, K, V into multiple heads, applies attention in parallel, then concatenates.

```python
mha = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=8)
x = tf.random.normal((2, 10, 64))
attn_out = mha(query=x, value=x, key=x)
print(attn_out.shape)  # (2, 10, 64)
```

For **decoder** (causal) attention, use `use_causal_mask=True` to prevent attending to future positions.

```python
attn_causal = mha(query=x, value=x, key=x, use_causal_mask=True)
```

### Transformer Encoder Block

1. Multi-head self-attention + residual + LayerNorm
2. Feed-forward network + residual + LayerNorm

```python
def transformer_encoder_block(x, d_model, num_heads, ff_dim):
    mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
    ln1 = tf.keras.layers.LayerNormalization()
    ln2 = tf.keras.layers.LayerNormalization()
    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(ff_dim, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])
    attn_out = mha(query=x, value=x, key=x)
    x = ln1(x + attn_out)
    return ln2(x + ffn(x))
```

### Transformer Decoder Block

1. Masked self-attention (causal) + residual + LayerNorm
2. Cross-attention (Q from decoder, K/V from encoder) + residual + LayerNorm
3. Feed-forward + residual + LayerNorm

```python
def transformer_decoder_block(x, enc_out, d_model, num_heads, ff_dim):
    mha1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
    mha2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
    ln1 = tf.keras.layers.LayerNormalization()
    ln2 = tf.keras.layers.LayerNormalization()
    ln3 = tf.keras.layers.LayerNormalization()
    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(ff_dim, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])
    self_attn = mha1(query=x, value=x, key=x, use_causal_mask=True)
    x = ln1(x + self_attn)
    cross_attn = mha2(query=x, value=enc_out, key=enc_out)
    x = ln2(x + cross_attn)
    return ln3(x + ffn(x))
```

---

## 3. Embedding Layers

The **Embedding** layer maps integer indices (e.g., token IDs) to dense vectors. It is a lookup table: `output[i] = embeddings[input[i]]`.

### Basic Usage

```python
vocab_size, embed_dim = 10000, 64
emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
x = tf.random.uniform((4, 20), maxval=vocab_size, dtype=tf.int32)
out = emb(x)
print(out.shape)  # (4, 20, 64)
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| input_dim | Vocabulary size (max index + 1) |
| output_dim | Embedding dimension |
| embeddings_initializer | How to initialize (default: uniform) |
| mask_zero | If True, index 0 is padding and will be masked |
| input_length | Fixed sequence length (optional) |

### Pretrained Weights

```python
pretrained = tf.random.normal((vocab_size, embed_dim))
emb = tf.keras.layers.Embedding(
    input_dim=vocab_size, output_dim=embed_dim,
    embeddings_initializer=tf.keras.initializers.Constant(pretrained)
)
```

### mask_zero

When `mask_zero=True`, index 0 is treated as padding. Downstream layers (e.g., LSTM) will receive a mask and skip padded positions.

```python
emb = tf.keras.layers.Embedding(vocab_size, 64, mask_zero=True)
out = emb(x)
mask = emb.compute_mask(x)  # True for non-padding
```

### Full NLP Model

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 64, input_length=100),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```
