# Convolutional Layers, Pooling, Normalization, and Dropout

## Table of Contents

1. [Convolutional Layers](#1-convolutional-layers)
2. [Pooling Layers](#2-pooling-layers)
3. [Normalization Layers](#3-normalization-layers)
4. [Dropout Layers](#4-dropout-layers)

---

## 1. Convolutional Layers

Convolutional layers apply **learnable filters** to extract spatial (or temporal) features. They are the backbone of CNNs for images and sequences.

### Conv1D

For sequences (e.g., time series, text). Input shape: `(batch, steps, channels)`.

```python
x = tf.random.normal((2, 100, 32))
conv1d = tf.keras.layers.Conv1D(64, kernel_size=3, padding='same', strides=1)
out = conv1d(x)
print(out.shape)  # (2, 100, 64)
```

### Conv2D

For images. Input shape: `(batch, height, width, channels)`.

```python
x = tf.random.normal((2, 28, 28, 3))
conv2d = tf.keras.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1))
out = conv2d(x)
print(out.shape)  # (2, 28, 28, 64)
```

### Conv3D

For volumetric data (e.g., video, 3D medical images). Input shape: `(batch, depth, height, width, channels)`.

```python
x = tf.random.normal((2, 10, 20, 20, 4))
conv3d = tf.keras.layers.Conv3D(16, (2, 3, 3), padding='valid')
out = conv3d(x)
```

### Key Parameters

| Parameter | Description | Common Values |
|-----------|-------------|---------------|
| filters | Number of output channels | 32, 64, 128 |
| kernel_size | Size of convolution window | 3, (3,3), (2,3,3) |
| padding | 'valid' (no pad) or 'same' (pad to preserve size) | 'same', 'valid' |
| strides | Step size | 1, (2,2) |
| dilation_rate | Spacing between kernel elements | 1, (2,2) |
| groups | Split input channels into groups (depthwise conv) | 1, 4, 8 |

### Dilation and Groups

**Dilation** increases receptive field without adding parameters. `dilation_rate=(2,2)` skips every other pixel.

**Groups** enable depthwise separable convolutions. `groups=4` splits input into 4 groups, each convolved separately.

```python
conv_dilated = tf.keras.layers.Conv2D(32, 3, padding='same', dilation_rate=(2, 2))
conv_groups = tf.keras.layers.Conv2D(64, 3, padding='same', groups=4)
```

---

## 2. Pooling Layers

Pooling reduces spatial dimensions, providing **translation invariance** and reducing computation.

### MaxPooling2D

Takes the maximum value in each window. Preserves strongest activations.

```python
x = tf.random.normal((2, 28, 28, 64))
max_pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
out = max_pool(x)
print(out.shape)  # (2, 14, 14, 64)
```

### AveragePooling2D

Takes the average in each window. Smoother, less sensitive to outliers.

```python
avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2)
out = avg_pool(x)
```

### Global Pooling

Reduces spatial dimensions to 1. Output shape: `(batch, channels)`.

- **GlobalMaxPooling2D**: Max over entire spatial extent.
- **GlobalAveragePooling2D**: Mean over entire spatial extent.

```python
global_max = tf.keras.layers.GlobalMaxPooling2D()
global_avg = tf.keras.layers.GlobalAveragePooling2D()
out_gmax = global_max(x)  # (2, 64)
out_gavg = global_avg(x)  # (2, 64)
```

### When to Use

| Pooling | Use Case |
|---------|----------|
| MaxPooling | Default for CNNs, preserves sharp features |
| AveragePooling | Smoother representations |
| GlobalAveragePooling | Common before classifier, reduces params |
| GlobalMaxPooling | When max activation matters (e.g., attention) |

---

## 3. Normalization Layers

Normalization stabilizes training by standardizing activations.

### BatchNormalization

Normalizes across the **batch** dimension. For each channel: `(x - mean) / sqrt(var + eps) * gamma + beta`.

- **Training**: Uses batch statistics.
- **Inference**: Uses moving average of batch statistics.

```python
bn = tf.keras.layers.BatchNormalization()
x = tf.random.normal((8, 32, 32, 64))
out = bn(x, training=True)
```

**Caveat**: Sensitive to batch size. Small batches give noisy statistics.

### LayerNormalization

Normalizes across **feature** dimension (last axis). Independent of batch size. Common in Transformers and RNNs.

```python
ln = tf.keras.layers.LayerNormalization(axis=-1)
out = ln(x)
```

### GroupNormalization

Divides channels into groups and normalizes within each group. Good for small batches and segmentation.

```python
gn = tf.keras.layers.GroupNormalization(groups=8)
out = gn(x)
```

### Comparison

| Method | Normalizes Over | Best For |
|--------|-----------------|----------|
| BatchNorm | Batch + spatial | Large batches, CNNs |
| LayerNorm | Features | RNNs, Transformers |
| GroupNorm | Groups of channels | Small batches, segmentation |

---

## 4. Dropout Layers

Dropout randomly sets a fraction of inputs to zero during training, preventing co-adaptation of neurons.

### Standard Dropout

```python
dropout = tf.keras.layers.Dropout(0.5)
x = tf.random.normal((4, 64))
out = dropout(x, training=True)
```

At inference, dropout is disabled (no masking). Scale is handled internally.

### SpatialDropout1D / SpatialDropout2D

Drops entire **feature maps** (channels) instead of individual elements. Better for convolutional layers where nearby pixels are correlated.

```python
x2d = tf.random.normal((4, 28, 28, 64))
spatial2d = tf.keras.layers.SpatialDropout2D(0.3)
out = spatial2d(x2d, training=True)
```

### AlphaDropout

Designed for **SELU** networks. Maintains self-normalizing properties (mean 0, variance 1) after dropout.

```python
alpha_dropout = tf.keras.layers.AlphaDropout(0.2)
x_selu = tf.keras.activations.selu(tf.random.normal((4, 64)))
out = alpha_dropout(x_selu, training=True)
```

### Typical Usage

```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])
```
