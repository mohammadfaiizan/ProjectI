# Domain Libraries: TF Hub, TFDS, TF Text, TF Addons

## Table of Contents

1. [TensorFlow Hub](#1-tensorflow-hub)
2. [TensorFlow Datasets](#2-tensorflow-datasets)
3. [TensorFlow Text](#3-tensorflow-text)
4. [TensorFlow Addons](#4-tensorflow-addons)

---

## 1. TensorFlow Hub

**TensorFlow Hub** is a repository of pre-trained models, embeddings, and reusable components. It enables transfer learning by loading pre-trained modules as **Keras layers** without retraining from scratch.

### hub.KerasLayer

The primary interface is `hub.KerasLayer`, which wraps a Hub module as a Keras-compatible layer:

```python
import tensorflow_hub as hub

feature_extractor = hub.KerasLayer(
    "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5",
    trainable=False,
    output_shape=[1280]
)
```

| Parameter | Purpose |
|-----------|---------|
| trainable | If False, weights are frozen (feature extraction) |
| output_shape | Optional; specifies output dimension |
| input_shape | Override input shape for compatibility |

### Feature Extraction vs Fine-Tuning

**Feature extraction** uses a frozen backbone. Only the new head is trained:

```python
model = tf.keras.Sequential([
    hub.KerasLayer(hub_url, trainable=False),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
```

**Fine-tuning** unfreezes some or all backbone layers for domain adaptation:

```python
hub_layer = hub.KerasLayer(hub_url, trainable=True)
# Optionally: hub_layer.trainable_variables for selective unfreezing
```

### Module Types

- **Image feature vectors**: Mobilenet, ResNet, EfficientNet
- **Text embeddings**: Universal Sentence Encoder, BERT
- **Audio**: YAMNet, VGGish

---

## 2. TensorFlow Datasets

**TensorFlow Datasets (TFDS)** provides a catalog of ready-to-use datasets with consistent APIs, automatic downloading, and built-in preprocessing.

### tfds.load

Load a dataset with a single call:

```python
import tensorflow_datasets as tfds

ds, info = tfds.load("mnist", split="train", as_supervised=True, with_info=True)
print(info.splits["train"].num_examples)
```

| Parameter | Purpose |
|-----------|---------|
| split | "train", "test", "train[:80%]", etc. |
| as_supervised | Returns (input, label) tuples |
| with_info | Returns DatasetInfo (metadata) |
| shuffle_files | Shuffle file order for distributed training |

### tfds.builder

For more control, use the builder API:

```python
builder = tfds.builder("imagenet2012")
print(builder.info.description)
print(builder.info.splits)
print(builder.info.features)
```

### Catalog and Splits

The **catalog** lists all available datasets. Each dataset defines **splits** (e.g., train, validation, test) with metadata:

```python
for split_name, split_info in builder.info.splits.items():
    print(f"{split_name}: {split_info.num_examples} examples")
```

### Data Format

TFDS returns `tf.data.Dataset` objects. With `as_supervised=True`, each element is `(features, label)`. Without it, elements are dictionaries matching the dataset's feature structure.

```python
for image, label in ds.take(1):
    print(image.shape, label.numpy())
```

---

## 3. TensorFlow Text

**TensorFlow Text** provides text processing ops that integrate with the TensorFlow graph. It supports tokenizers, normalizers, and n-gram operations used in NLP pipelines.

### Tokenizers

**WhitespaceTokenizer** splits on whitespace:

```python
import tensorflow_text as text

tokenizer = text.WhitespaceTokenizer()
tokens = tokenizer.tokenize(["Hello world", "TensorFlow text"])
```

**UnicodeScriptTokenizer** splits by Unicode script (useful for multilingual text):

```python
script_tokenizer = text.UnicodeScriptTokenizer()
tokens = script_tokenizer.tokenize(["Hello, 世界!"])
```

**WordpieceTokenizer** and **BertTokenizer** support subword tokenization with a vocabulary file, matching BERT-style preprocessing.

### Normalizers

**case_fold_utf8** performs Unicode case folding:

```python
normalized = text.case_fold_utf8(["Hello WORLD"])
```

**NormalizeUTF8** and **RegexReplace** support custom normalization pipelines.

### BertTokenizer Concepts

**BertTokenizer** uses a vocabulary file and applies:

- **WordPiece** tokenization for subword units
- **Lowercasing** (optional)
- **Basic tokenization** (whitespace, punctuation) before WordPiece

```python
tokenizer = text.BertTokenizer(vocab_file, lower_case=True)
tokens = tokenizer.tokenize(["hello world"])
```

### Integration with Models

TF Text ops are graph-compatible, so tokenization can run inside `tf.data` pipelines or `tf.function` for efficient batching.

---

## 4. TensorFlow Addons

**TensorFlow Addons (TFA)** provides extra layers, losses, optimizers, and metrics that extend core TensorFlow. Many components are candidates for promotion to core TF.

### tfa.layers

| Layer | Purpose |
|-------|---------|
| GroupNormalization | Normalize across groups of channels |
| InstanceNormalization | Per-instance normalization |
| SpectralNormalization | Constrain weight spectral norm |
| CRF | Conditional Random Field for sequence labeling |

```python
import tensorflow_addons as tfa

gn = tfa.layers.GroupNormalization(groups=8)
x = gn(input_tensor)
```

### tfa.losses

| Loss | Purpose |
|------|---------|
| TripletSemiHardLoss | Metric learning, face recognition |
| ContrastiveLoss | Siamese networks |
| FocalLoss | Class imbalance |

```python
loss_fn = tfa.losses.TripletSemiHardLoss()
loss = loss_fn(anchor, positive, negative)
```

### tfa.optimizers

**LAMB** (Layer-wise Adaptive Moments for BERT) is designed for large batch training and BERT-style models:

```python
optimizer = tfa.optimizers.LAMB(learning_rate=1e-3)
```

**SGDW** adds decoupled weight decay to SGD (similar to AdamW):

```python
optimizer = tfa.optimizers.SGDW(weight_decay=0.01, learning_rate=0.01)
```

**SWA** (Stochastic Weight Averaging) averages model weights over the last epochs for better generalization. Use `tfa.optimizers.SWA` or manually average checkpoints.

### Deprecation Note

Some TFA components have been moved to core Keras (e.g., GroupNormalization). Check the TFA documentation for current status.

---

## Summary Table

| Library | Primary Use |
|---------|-------------|
| TF Hub | Pre-trained models, transfer learning |
| TFDS | Dataset loading, catalog, splits |
| TF Text | Tokenizers, normalizers, BERT-style preprocessing |
| TF Addons | Extra layers, losses, optimizers (LAMB, SGDW, SWA) |
