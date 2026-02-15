# RNN, Language Modeling, and Sequence-to-Sequence

## Table of Contents

1. [Sentiment Analysis with RNNs](#1-sentiment-analysis-with-rnns)
2. [Language Modeling](#2-language-modeling)
3. [Sequence-to-Sequence with Attention](#3-sequence-to-sequence-with-attention)

---

## 1. Sentiment Analysis with RNNs

**Sentiment analysis** classifies text (e.g., sentences, reviews) into sentiment categories (positive, negative, neutral).

### LSTM and GRU

- **LSTM**: Long Short-Term Memory; addresses vanishing gradients with gates (forget, input, output).
- **GRU**: Gated Recurrent Unit; simpler than LSTM with fewer parameters; often comparable performance.

```python
lstm = tf.keras.layers.LSTM(units, return_sequences=False)
gru = tf.keras.layers.GRU(units, return_sequences=False)
```

### Bidirectional RNNs

**Bidirectional** processes the sequence both forward and backward, concatenating outputs. Useful when context from both directions matters (e.g., sentiment).

```python
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units))(x)
```

### Architecture Pattern

1. Embedding with **mask_zero=True** for padding.
2. Bidirectional LSTM or GRU.
3. Dense layers and dropout.
4. Softmax output.

```python
inp = tf.keras.layers.Input(shape=(seq_len,))
x = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)(inp)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units))(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
```

### return_sequences

- **False**: Output last hidden state only; use for sequence-level classification.
- **True**: Output at every timestep; use for token-level tasks or stacked RNNs.

---

## 2. Language Modeling

**Language modeling** predicts the next token given previous tokens. It is the foundation for text generation and pretraining.

### Character vs Word Level

| Level | Vocab Size | Sequence Length | Use Case |
|-------|------------|-----------------|----------|
| Character | Small (~100) | Long | Morphology, small data |
| Word | Large (~50k) | Shorter | Standard LM, generation |

### Next-Token Prediction

- Input: tokens `[t_1, ..., t_n]`
- Target: `[t_2, ..., t_{n+1}]`
- Loss: **sparse_categorical_crossentropy** on shifted logits.

```python
logits = model(x)
targets = x[:, 1:]
logits_shifted = logits[:, :-1, :]
loss = tf.keras.losses.sparse_categorical_crossentropy(targets, logits_shifted, from_logits=True)
```

### Architecture

- Embedding -> RNN (LSTM/GRU) with **return_sequences=True** -> Dense(vocab_size).

```python
x = tf.keras.layers.Embedding(vocab_size, embed_dim)(inp)
x = tf.keras.layers.LSTM(rnn_units, return_sequences=True)(x)
logits = tf.keras.layers.Dense(vocab_size)(x)
```

### Sampling

- **Greedy**: `argmax(logits)`.
- **Random**: Sample from softmax(logits).
- **Temperature**: Scale logits by T before softmax; lower T = sharper, higher T = more random.

---

## 3. Sequence-to-Sequence with Attention

**Seq2seq** maps an input sequence to an output sequence (e.g., translation, summarization).

### Encoder-Decoder

- **Encoder**: Processes input sequence; outputs final hidden state(s) and optionally all hidden states.
- **Decoder**: Generates output sequence step-by-step, conditioned on encoder output.

```python
enc_out, enc_h, enc_c = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)(enc_emb)
dec_out = tf.keras.layers.LSTM(units, return_sequences=True)(dec_emb, initial_state=[enc_h, enc_c])
```

### Attention Mechanism

**Attention** lets the decoder focus on relevant parts of the encoder output at each step.

- **Query**: Decoder hidden state.
- **Key/Value**: Encoder hidden states.
- **Scores**: Compatibility between query and keys (e.g., dot product, additive).
- **Context**: Weighted sum of values.

```python
attention = tf.keras.layers.Attention()([dec_emb, enc_out])
dec_concat = tf.keras.layers.Concatenate()([dec_emb, attention])
dec_lstm = tf.keras.layers.LSTM(units, return_sequences=True)(dec_concat, initial_state=[enc_h, enc_c])
```

### Teacher Forcing

During training, the decoder receives the **ground-truth** previous token as input instead of its own prediction. At inference, use the model's own predictions (autoregressive).

### Loss

- **Sparse categorical crossentropy** over decoder output.
- Shift targets: `targets = dec_inp[:, 1:]`, `pred = output[:, :-1, :]`.

### Limitations of RNN Seq2seq

- Sequential processing limits parallelism.
- Long-range dependencies can be hard to capture.
- Transformers largely replace RNN seq2seq for many tasks.

### Stacked RNNs

Use multiple RNN layers for increased capacity. Pass **return_sequences=True** to intermediate layers so the next layer receives a sequence.

```python
x = tf.keras.layers.LSTM(units, return_sequences=True)(x)
x = tf.keras.layers.LSTM(units, return_sequences=False)(x)
```

### Dropout in RNNs

Apply **recurrent_dropout** and **dropout** to LSTM/GRU to reduce overfitting. recurrent_dropout affects the recurrent connections.

```python
lstm = tf.keras.layers.LSTM(units, dropout=0.2, recurrent_dropout=0.1)
```

### Gradient Clipping

RNNs can suffer from exploding gradients. Use **clipnorm** or **clipvalue** in the optimizer.

```python
optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
```

### Data Preparation for Seq2seq

- **Source**: Tokenize and pad; often reverse the sequence (Sutskever et al.) for better gradient flow.
- **Target**: Add [BOS] at start, [EOS] at end; shift for teacher forcing.

### Perplexity

**Perplexity** is the exponentiated average negative log-likelihood per token. Lower perplexity indicates a better language model.

```python
perplexity = tf.exp(tf.reduce_mean(loss))
```

### Truncated Backpropagation Through Time (BPTT)

For long sequences, process in chunks and truncate gradients across chunk boundaries to reduce memory and compute.

### Comparison: LSTM vs GRU

| Aspect | LSTM | GRU |
|--------|------|-----|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| Parameters | More | Fewer |
| Training speed | Slower | Faster |
| Long sequences | Strong | Comparable |

### Attention Alignment Visualization

The attention weights from encoder-decoder models can be visualized as a heatmap: rows = decoder steps, columns = encoder steps. This helps interpret which source tokens the model attends to when generating each target token.

### Bucketing and Padding

For variable-length sequences, **bucketing** groups similar lengths together to minimize padding waste. Use **tf.data.experimental.bucket_by_sequence_length** or pad to max length per batch.

### Copy Mechanism

In summarization and dialogue, a **copy mechanism** allows the decoder to copy tokens directly from the source. Useful when output should include exact phrases from the input.
