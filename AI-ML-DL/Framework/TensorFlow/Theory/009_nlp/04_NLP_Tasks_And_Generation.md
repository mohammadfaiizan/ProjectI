# NLP Tasks and Text Generation

## Table of Contents

1. [Text Generation Methods](#1-text-generation-methods)
2. [Named Entity Recognition](#2-named-entity-recognition)
3. [Machine Translation](#3-machine-translation)
4. [Question Answering](#4-question-answering)

---

## 1. Text Generation Methods

Autoregressive language models produce text by sampling the next token repeatedly. The **sampling strategy** controls diversity vs coherence.

### Greedy Decoding

- Always pick the token with highest probability.
- Deterministic; often repetitive.

```python
next_token = tf.argmax(logits, axis=-1)
```

### Temperature Sampling

- Scale logits by **temperature T** before softmax: `logits / T`.
- **T < 1**: Sharper distribution, less random.
- **T > 1**: Flatter distribution, more random.
- **T = 1**: Standard softmax.

```python
scaled = logits / temperature
probs = tf.nn.softmax(scaled)
next_token = tf.random.categorical(tf.math.log(probs), 1)[:, 0]
```

### Top-k Sampling

- Keep only the top **k** tokens by probability; renormalize and sample.
- Reduces chance of sampling low-probability tokens.

```python
top_k_logits, top_k_indices = tf.math.top_k(logits, k)
probs = tf.nn.softmax(top_k_logits)
idx = tf.random.categorical(tf.math.log(probs), 1)[:, 0]
next_token = tf.gather(top_k_indices, idx, batch_dims=1)
```

### Top-p (Nucleus) Sampling

- Keep the smallest set of tokens whose cumulative probability >= **p**.
- Adapts k dynamically based on distribution shape.

```python
sorted_probs = tf.sort(probs, direction="DESCENDING")
cumsum = tf.cumsum(sorted_probs, axis=-1)
mask = cumsum <= p
filtered_probs = tf.where(mask, sorted_probs, 0)
filtered_probs = filtered_probs / tf.reduce_sum(filtered_probs, axis=-1, keepdims=True)
```

### Comparison

| Method | Diversity | Coherence | Use Case |
|--------|-----------|-----------|----------|
| Greedy | Low | High | Decoding, beam search |
| Temperature | Tunable | Tunable | General generation |
| Top-k | Medium | Medium | Constrained sampling |
| Top-p | High | Medium | Creative generation |

---

## 2. Named Entity Recognition

**NER** identifies entities (person, location, organization, etc.) in text. It is a **token-level classification** task.

### BIO Tagging

- **B-X**: Beginning of entity type X.
- **I-X**: Inside entity type X.
- **O**: Outside any entity.

Example: "John Smith works at Google" -> B-PER I-PER O O B-ORG

### Architecture

- Embedding -> Bidirectional LSTM/GRU -> Dense -> Softmax per token.

```python
x = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)(inp)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))(x)
out = tf.keras.layers.Dense(num_tags, activation="softmax")(x)
```

### Loss

- **Sparse categorical crossentropy** per token.
- **Mask padding**: Do not include padding tokens in loss.

```python
per_token_loss = sparse_categorical_crossentropy(labels, logits)
masked_loss = tf.reduce_sum(per_token_loss * mask) / (tf.reduce_sum(mask) + 1e-8)
```

### Evaluation

- Token-level accuracy.
- Entity-level F1 (exact match or partial match).

---

## 3. Machine Translation

**Neural Machine Translation (NMT)** maps source language text to target language text using encoder-decoder models.

### Transformer for NMT

- **Encoder**: Source tokens -> Transformer encoder blocks.
- **Decoder**: Target tokens (shifted) -> Causal self-attention + cross-attention to encoder -> Transformer decoder blocks.
- **Output**: Logits over target vocabulary.

```python
enc_out = encoder(enc_inp)
dec_out = decoder(dec_inp, enc_out)
logits = tf.keras.layers.Dense(tgt_vocab)(dec_out)
```

### Training

- **Teacher forcing**: Decoder input = ground-truth target (shifted).
- **Loss**: Sparse categorical crossentropy over target tokens.

### Inference

- **Autoregressive decoding**: Start with [BOS], sample/greedy next token until [EOS] or max length.
- **Beam search**: Maintain top-k hypotheses; often better than greedy.

---

## 4. Question Answering

**Extractive QA** finds a span in a context passage that answers a question. The model predicts **start** and **end** indices.

### Input Format

- Concatenate: `[CLS] question [SEP] context [SEP]`.
- Or use separate encodings with cross-attention.

### Architecture

- Encode question+context (e.g., BERT-style).
- Two linear heads: one for start logits, one for end logits.

```python
encoded = encoder(inp)
start_logits = tf.keras.layers.Dense(1)(encoded)
end_logits = tf.keras.layers.Dense(1)(encoded)
start_logits = tf.keras.layers.Flatten()(start_logits)
end_logits = tf.keras.layers.Flatten()(end_logits)
```

### Loss

- **Crossentropy** for start and end positions.
- Total loss = start_loss + end_loss.

```python
loss_start = sparse_categorical_crossentropy(start_labels, start_logits, from_logits=True)
loss_end = sparse_categorical_crossentropy(end_labels, end_logits, from_logits=True)
loss = tf.reduce_mean(loss_start) + tf.reduce_mean(loss_end)
```

### Span Extraction

- **Greedy**: `start = argmax(start_logits)`, `end = argmax(end_logits)`.
- **Constraint**: Ensure start <= end; optionally enforce max span length.

### Evaluation

- **Exact Match (EM)**: Predicted span exactly matches ground truth.
- **F1**: Token overlap F1 between predicted and ground-truth spans.

### Beam Search for Generation

Instead of greedy decoding, maintain **k** hypotheses at each step. Extend each with top-k next tokens, keep best k by cumulative score.

```python
def beam_step(beam_logits, beam_scores, k=5):
    flat_logits = tf.reshape(beam_logits, (-1,))
    top_scores, top_indices = tf.math.top_k(flat_logits, k)
    return top_scores, top_indices
```

### Label Smoothing

For generation and classification, **label smoothing** prevents overconfident predictions and can improve generalization.

```python
loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
```

### BLEU for Translation

**BLEU** (Bilingual Evaluation Understudy) measures n-gram overlap between hypothesis and reference. Commonly used for machine translation evaluation.

### SQuAD and QA Datasets

- **SQuAD**: Stanford Question Answering Dataset; extractive QA with Wikipedia passages.
- **Format**: (context, question, answer_span_start, answer_span_end).
