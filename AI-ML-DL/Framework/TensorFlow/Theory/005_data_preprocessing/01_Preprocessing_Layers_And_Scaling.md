# Preprocessing Layers and Feature Scaling

## Table of Contents

1. [Numeric Preprocessing Layers](#1-numeric-preprocessing-layers)
2. [Categorical Preprocessing Layers](#2-categorical-preprocessing-layers)
3. [Text Preprocessing Layers](#3-text-preprocessing-layers)
4. [Image Preprocessing Layers](#4-image-preprocessing-layers)
5. [Feature Scaling Methods](#5-feature-scaling-methods)

---

## 1. Numeric Preprocessing Layers

Numeric preprocessing layers transform continuous numerical data for model consumption. TensorFlow provides built-in layers that integrate with the Keras API and support **adaptation** from training data.

### Normalization Layer

The **Normalization** layer standardizes inputs to zero mean and unit variance. It learns the mean and variance during `adapt()` and applies `(x - mean) / sqrt(variance + epsilon)` at inference.

```python
import tensorflow as tf

data = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
norm_layer = tf.keras.layers.Normalization(axis=-1)
norm_layer.adapt(data)
normalized = norm_layer(data)
```

### Discretization Layer

**Discretization** bins continuous values into discrete buckets. Useful for converting numeric features to categorical or reducing noise.

```python
disc_layer = tf.keras.layers.Discretization(bin_boundaries=[0.0, 2.5, 5.0, 7.5])
disc_out = disc_layer(tf.constant([1.0, 3.0, 6.0, 8.0]))
```

### Rescaling Layer

**Rescaling** applies a linear transform: `output = scale * input + offset`. Common for image data: `scale=1/255` to map [0,255] to [0,1].

```python
rescale_layer = tf.keras.layers.Rescaling(scale=1.0/255.0, offset=0)
scaled = rescale_layer(image_tensor)
```

### Key Parameters

| Layer | Key Parameters | Use Case |
|-------|----------------|----------|
| Normalization | axis | Z-score standardization |
| Discretization | bin_boundaries | Binning, histograms |
| Rescaling | scale, offset | Image normalization |

---

## 2. Categorical Preprocessing Layers

Categorical data requires encoding before neural network input. TensorFlow provides several layers for string and integer categorical handling.

### StringLookup and IntegerLookup

**StringLookup** maps strings to integer indices. **IntegerLookup** does the same for integers. Both support vocabulary from a list or via `adapt()`.

```python
str_lookup = tf.keras.layers.StringLookup(vocabulary=["red", "green", "blue"])
indices = str_lookup(tf.constant([["red"], ["blue"]]))
```

### CategoryEncoding

**CategoryEncoding** converts integer indices to one-hot or multi-hot representations. Use with **StringLookup** or **IntegerLookup** for full pipeline.

```python
enc = tf.keras.layers.CategoryEncoding(num_tokens=4, output_mode="one_hot")
lookup = tf.keras.layers.StringLookup(vocabulary=["a", "b", "c"])
encoded = enc(lookup(categorical_data))
```

### Hashing Layer

**Hashing** maps inputs to a fixed number of bins via a hash function. No vocabulary needed; useful for high-cardinality or streaming data.

```python
hash_layer = tf.keras.layers.Hashing(num_bins=32)
hashed = hash_layer(tf.constant([["cat"], ["dog"]]))
```

### Output Modes

| Mode | Description |
|------|-------------|
| one_hot | Single category per sample |
| multi_hot | Multiple categories per sample |

---

## 3. Text Preprocessing Layers

**TextVectorization** converts raw text to numeric representations suitable for embedding or dense layers.

### output_mode Options

| output_mode | Output | Use Case |
|-------------|--------|----------|
| int | Sequence of token indices | Embedding input |
| multi_hot | Bag-of-words vector | Classification |
| count | Token counts | TF-IDF style |
| tf_idf | TF-IDF weights | Sparse text features |

### max_tokens and output_sequence_length

- **max_tokens**: Vocabulary size. Tokens beyond this are dropped or mapped to OOV.
- **output_sequence_length**: Pads or truncates sequences to fixed length.

```python
text_vec = tf.keras.layers.TextVectorization(
    max_tokens=1000,
    output_mode="int",
    output_sequence_length=128
)
text_vec.adapt(text_dataset)
encoded = text_vec(tf.constant(["hello world"]))
```

### Adapting from Data

Call `adapt()` on a dataset or tensor of strings to build the vocabulary from training data.

---

## 4. Image Preprocessing Layers

Image preprocessing layers handle resizing, cropping, and pixel value scaling.

### Resizing

**Resizing** changes spatial dimensions. Supports `bilinear`, `nearest`, `bicubic`, `area`, `lanczos3`, `lanczos5`, `gaussian`, `mitchellcubic`.

```python
resize_layer = tf.keras.layers.Resizing(64, 64, interpolation="bilinear")
resized = resize_layer(image_batch)
```

### Rescaling

**Rescaling** for images typically uses `scale=1/255` to normalize pixel values to [0, 1].

```python
rescale = tf.keras.layers.Rescaling(1.0/255.0)
scaled = rescale(image_tensor)
```

### CenterCrop

**CenterCrop** extracts a central region of specified height and width. Useful for fixed-input models.

```python
center_crop = tf.keras.layers.CenterCrop(224, 224)
cropped = center_crop(image_tensor)
```

### Pipeline Example

```python
pipeline = tf.keras.Sequential([
    tf.keras.layers.Resizing(256, 256),
    tf.keras.layers.Rescaling(1.0/255.0),
    tf.keras.layers.CenterCrop(224, 224)
])
preprocessed = pipeline(images)
```

---

## 5. Feature Scaling Methods

Feature scaling ensures features are on comparable scales, improving optimization and model performance.

### Min-Max Scaling

Maps values to [0, 1]: `(x - min) / (max - min)`.

```python
min_val = tf.reduce_min(data, axis=0)
max_val = tf.reduce_max(data, axis=0)
minmax = (data - min_val) / (max_val - min_val + 1e-8)
```

### Z-Score (Standardization)

Zero mean, unit variance: `(x - mean) / std`.

```python
mean = tf.reduce_mean(data, axis=0)
std = tf.math.reduce_std(data, axis=0)
zscore = (data - mean) / (std + 1e-8)
```

### Robust Scaling

Uses median and IQR; resistant to outliers: `(x - median) / IQR`.

```python
sorted_data = tf.sort(data, axis=0)
q25 = tf.gather(sorted_data, n // 4, axis=0)
q75 = tf.gather(sorted_data, 3 * n // 4, axis=0)
iqr = q75 - q25
robust = (data - median) / (iqr + 1e-8)
```

### Keras Normalization Layer

The **Normalization** layer encapsulates standardization with `adapt()` for training data.

```python
norm_layer = tf.keras.layers.Normalization(axis=-1)
norm_layer.adapt(train_data)
scaled = norm_layer(data)
```

### Scaling Comparison

| Method | Formula | When to Use |
|--------|---------|-------------|
| Min-max | (x-min)/(max-min) | Bounded output, no outliers |
| Z-score | (x-mean)/std | Normal-ish data |
| Robust | (x-median)/IQR | Outlier-heavy data |
