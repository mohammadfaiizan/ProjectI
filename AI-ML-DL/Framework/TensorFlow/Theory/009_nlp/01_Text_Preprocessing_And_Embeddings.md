# Text Preprocessing and Embeddings

## Table of Contents

1. [Text Preprocessing with tf.strings](#1-text-preprocessing-with-tfstrings)
2. [Tokenization Techniques](#2-tokenization-techniques)
3. [Embedding Layers](#3-embedding-layers)
4. [Text Classification](#4-text-classification)

---

## 1. Text Preprocessing with tf.strings

**tf.strings** provides tensor operations for string manipulation, enabling efficient text cleaning and normalization directly on tensors.

### Lowercasing and Normalization

- **tf.strings.lower**: Converts text to lowercase for case-insensitive processing.
- **tf.strings.strip**: Removes leading and trailing whitespace.
- **tf.strings.regex_replace**: Applies regex patterns for cleaning (e.g., remove punctuation).

```python
texts = tf.constant(["  Hello, World!  ", "TensorFlow NLP"])
lower = tf.strings.lower(texts)
stripped = tf.strings.strip(texts)
clean = tf.strings.regex_replace(texts, "[^a-zA-Z ]", "")
```

### Splitting and Joining

- **tf.strings.split**: Splits strings by delimiter (e.g., whitespace for word tokenization).
- **tf.strings.join**: Joins string tensors with a separator.

```python
split = tf.strings.split(texts, sep=" ")
joined = tf.strings.join([["a", "b"], ["c", "d"]], separator="-")
```

### Unicode and Length

- **tf.strings.unicode_decode**: Decodes bytes to Unicode code points.
- **tf.strings.length**: Returns byte or character length per string.

| Operation | Purpose |
|-----------|---------|
| lower | Case normalization |
| strip | Whitespace removal |
| regex_replace | Pattern-based cleaning |
| split | Tokenization |
| length | Sequence length for padding |

---

## 2. Tokenization Techniques

### TextVectorization Layer

**tf.keras.layers.TextVectorization** is a preprocessing layer that adapts to a corpus and converts text to integer sequences.

```python
tv = tf.keras.layers.TextVectorization(
    max_tokens=100,
    output_mode="int",
    output_sequence_length=10,
    standardize="lower_and_strip_punctuation"
)
tv.adapt(corpus)
encoded = tv(texts)
```

### Output Modes

| Mode | Output | Use Case |
|------|--------|----------|
| int | Integer indices | Embedding input |
| binary | Multi-hot | Bag-of-words |
| count | Token counts | Bag-of-words |
| tf_idf | TF-IDF weights | Sparse features |

### Subword Tokenization Concepts

- **Word-level**: Each word is a token; large vocabularies, OOV issues.
- **Character-level**: Each character is a token; small vocab, long sequences.
- **Subword**: BPE, WordPiece, SentencePiece; balance between vocab size and sequence length.

```python
def char_ngrams(text, n=3):
    text = text.lower().replace(" ", "")
    return [text[i:i+n] for i in range(len(text)-n+1)]
```

### Vocabulary and Decoding

- **get_vocabulary()**: Returns the learned vocabulary list.
- **decode()**: Converts integer sequences back to strings.

---

## 3. Embedding Layers

### Basic Embedding

**tf.keras.layers.Embedding** maps integer indices to dense vectors. It is the primary way to convert discrete tokens to continuous representations.

```python
embedding = tf.keras.layers.Embedding(
    input_dim=vocab_size,
    output_dim=embed_dim,
    input_length=seq_len
)
x = embedding(token_ids)
```

- **input_dim**: Vocabulary size.
- **output_dim**: Embedding dimension (e.g., 64, 128, 300).
- **input_length**: Optional; used for fixed-length sequences.

### Masking

Use **mask_zero=True** when 0 is the padding token so downstream layers (e.g., LSTM) can ignore padding.

```python
embedding = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)
x = embedding(padded_ids)
mask = embedding.compute_mask(padded_ids)
```

### Pretrained Embeddings

Load pretrained vectors (e.g., GloVe, Word2Vec) and initialize the Embedding layer with them.

```python
pretrained = np.load("glove_embeddings.npy")
embedding = tf.keras.layers.Embedding(
    vocab_size, embed_dim,
    embeddings_initializer=tf.keras.initializers.Constant(pretrained)
)
```

### Positional Encoding

Transformers and some RNNs need positional information. **Sinusoidal encoding** is a common fixed scheme:

```python
def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    dim = np.arange(d_model)[np.newaxis, :]
    angle = pos / np.power(10000, 2 * (dim // 2) / d_model)
    angle[:, 0::2] = np.sin(angle[:, 0::2])
    angle[:, 1::2] = np.cos(angle[:, 1::2])
    return tf.constant(angle, dtype=tf.float32)
```

Add to embeddings: `x = embeddings + positional_encoding(seq_len, embed_dim)`.

---

## 4. Text Classification

### Dense (Bag-of-Words) Classifier

- Embed tokens, then **GlobalAveragePooling1D** to get a fixed-size vector.
- Pass through dense layers and softmax for classification.

```python
inp = tf.keras.layers.Input(shape=(seq_len,))
x = tf.keras.layers.Embedding(vocab_size, embed_dim)(inp)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
```

### CNN Classifier

- Use **Conv1D** with multiple kernel sizes (e.g., 3, 4, 5) to capture n-gram patterns.
- **GlobalMaxPooling1D** after each convolution, then concatenate.

```python
conv_outputs = []
for k in [3, 4, 5]:
    c = tf.keras.layers.Conv1D(num_filters, k, activation="relu")(x)
    c = tf.keras.layers.GlobalMaxPooling1D()(c)
    conv_outputs.append(c)
x = tf.keras.layers.Concatenate()(conv_outputs)
```

### Comparison

| Architecture | Pros | Cons |
|--------------|------|------|
| Dense + pooling | Fast, simple | Loses order |
| CNN | Captures local n-grams | Limited long-range |
| RNN | Sequential modeling | Slower, vanishing gradients |

### Training

- **Loss**: SparseCategoricalCrossentropy or CategoricalCrossentropy with one-hot labels.
- **Metrics**: Accuracy, F1 for imbalanced data.

### Vocabulary and OOV

Out-of-vocabulary (OOV) tokens can be handled via a reserved index (e.g., 1) or subword tokenization. TextVectorization uses an [UNK] token for unknown words by default.
