# Vision Transformers, Attention, and Optimization

## Table of Contents

1. [Vision Transformers (ViT)](#1-vision-transformers-vit)
2. [Attention Mechanisms for Vision](#2-attention-mechanisms-for-vision)
3. [Data Augmentation](#3-data-augmentation)
4. [Custom Vision Architectures](#4-custom-vision-architectures)
5. [Model Optimization](#5-model-optimization)

---

## 1. Vision Transformers (ViT)

**Vision Transformers** apply the Transformer architecture to images by splitting them into **patches** and treating each patch as a token.

### Patch Embedding

- Split image into non-overlapping patches (e.g., 16x16).
- Flatten each patch and project to embedding dimension via linear layer or Conv2D.
- Result: sequence of patch tokens, shape (num_patches, embed_dim).

```python
def patch_embed(x, patch_size=16, embed_dim=768):
    x = tf.keras.layers.Conv2D(embed_dim, patch_size, strides=patch_size)(x)
    B, H, W, C = tf.shape(x)[0], x.shape[1], x.shape[2], x.shape[3]
    return tf.reshape(x, (-1, (H // patch_size) * (W // patch_size), embed_dim))
```

### Positional Encoding

- Add learnable or fixed positional embeddings to patch tokens.
- Enables the model to use spatial structure.

```python
pos_embed = tf.keras.layers.Embedding(num_patches, embed_dim)(tf.range(num_patches))
x = x + pos_embed
```

### Transformer Encoder

- Standard Transformer: MultiHeadAttention + FFN, with LayerNorm and residual connections.
- No decoder; classification via [CLS] token or global average of patch tokens.

```python
for _ in range(num_layers):
    attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)(x, x)
    x = tf.keras.layers.LayerNormalization()(x + attn)
    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(ff_dim, activation='gelu'),
        tf.keras.layers.Dense(embed_dim)
    ])(x)
    x = tf.keras.layers.LayerNormalization()(x + ffn)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
```

### ViT vs CNN

| Aspect | ViT | CNN |
|--------|-----|-----|
| Inductive bias | Minimal (patch + position) | Strong (locality, translation equivariance) |
| Data needs | Large (e.g., JFT-300M) | Moderate |
| Global context | Native (self-attention) | Requires many layers |
| Efficiency | O(n^2) in sequence length | O(n) with convolutions |

---

## 2. Attention Mechanisms for Vision

### Squeeze-and-Excitation (SE) Block

- **Squeeze**: Global average pooling to get channel-wise statistics.
- **Excitation**: FC layers with sigmoid to produce channel weights.
- **Scale**: Multiply original features by weights.

```python
def se_block(x, ratio=16):
    channels = x.shape[-1]
    squeeze = tf.keras.layers.GlobalAveragePooling2D()(x)
    excite = tf.keras.layers.Dense(channels // ratio, activation='relu')(squeeze)
    excite = tf.keras.layers.Dense(channels, activation='sigmoid')(excite)
    return x * tf.reshape(excite, (-1, 1, 1, channels))
```

### CBAM (Convolutional Block Attention Module)

- **Channel attention**: Avg + Max pool -> shared FC -> sigmoid.
- **Spatial attention**: Concatenate channel-wise avg and max -> Conv -> sigmoid.
- Apply channel then spatial attention sequentially.

```python
def channel_attention(x, ratio=8):
    avg = tf.keras.layers.GlobalAveragePooling2D()(x)
    max_p = tf.keras.layers.GlobalMaxPooling2D()(x)
    shared = tf.keras.Sequential([
        tf.keras.layers.Dense(channels // ratio, activation='relu'),
        tf.keras.layers.Dense(channels)
    ])
    ca = tf.keras.activations.sigmoid(shared(avg) + shared(max_p))
    return x * tf.reshape(ca, (-1, 1, 1, channels))

def spatial_attention(x):
    avg = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_p = tf.reduce_max(x, axis=-1, keepdims=True)
    concat = tf.keras.layers.Concatenate()([avg, max_p])
    sa = tf.keras.layers.Conv2D(1, 7, padding='same', activation='sigmoid')(concat)
    return x * sa
```

### Self-Attention for Images

- Reshape (H, W, C) to (H*W, C).
- Apply MultiHeadAttention (query=key=value=patches).
- Reshape back to (H, W, C).

---

## 3. Data Augmentation

### Standard Augmentations

| Augmentation | Effect |
|--------------|--------|
| RandomFlip | Horizontal/vertical flip |
| RandomRotation | Rotation invariance |
| RandomZoom | Scale variation |
| RandomBrightness/Contrast | Lighting robustness |
| RandomCrop | Spatial variation |

### Mixup

- Blend two images and their labels: `x = lam * x1 + (1-lam) * x2`, `y = lam * y1 + (1-lam) * y2`.
- lam ~ Beta(alpha, alpha).

```python
def mixup(images, labels, alpha=0.2):
    lam = tf.random.uniform([], 0, alpha)
    lam = tf.maximum(lam, 1 - lam)
    indices = tf.random.shuffle(tf.range(tf.shape(images)[0]))
    mixed_images = lam * images + (1 - lam) * tf.gather(images, indices)
    mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)
    return mixed_images, mixed_labels
```

### CutMix

- Cut a region from one image and paste into another.
- Adjust labels by area proportion: `lam = 1 - (cut_h * cut_w) / (H * W)`.

```python
# Cut region from image A, paste into image B
# Labels: lam * label_B + (1-lam) * label_A
```

### RandAugment

- Automatically search over augmentation strength.
- Fewer hyperparameters than AutoAugment.
- Use `tf.keras.layers.RandomContrast`, `RandomBrightness`, etc. in sequence.

---

## 4. Custom Vision Architectures

### Residual Blocks

- Skip connection: `output = F(x) + x`.
- Enables training of very deep networks.

```python
def residual_block(x, filters):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1)(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
    return tf.keras.layers.Activation('relu')(x + shortcut)
```

### Feature Pyramid Network (FPN)

- Multi-scale feature maps for detection/segmentation.
- Top-down pathway with lateral connections.
- Outputs: P2, P3, P4, P5 at different scales.

### Multi-Scale Design

- Process image at multiple resolutions.
- Fuse features for robustness to scale variation.

---

## 5. Model Optimization

### Quantization

- **Post-training quantization**: Reduce precision (float32 -> float16 or int8) after training.
- **Quantization-aware training**: Simulate quantization during training for better accuracy.
- Reduces model size and inference time.

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
```

### Pruning

- Remove low-magnitude weights (structured or unstructured).
- **Magnitude-based pruning**: Zero out smallest weights.
- Requires fine-tuning after pruning.

```python
import tensorflow_model_optimization as tfmot
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
model = prune_low_magnitude(model, pruning_schedule=tfmot.sparsity.keras.ConstantSparsity(0.5, 0))
```

### TensorFlow Lite Conversion

- Convert Keras model to TFLite for mobile/edge deployment.
- Supports quantization, dynamic range, full integer.

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Optimization Trade-offs

| Method | Size Reduction | Speed | Accuracy Impact |
|--------|----------------|-------|-----------------|
| FP16 quantization | ~2x | 1.5-2x | Minimal |
| INT8 quantization | ~4x | 2-4x | May need QAT |
| Pruning 50% | ~2x | Variable | May need fine-tuning |
| Knowledge distillation | Variable | Similar | Teacher-dependent |
