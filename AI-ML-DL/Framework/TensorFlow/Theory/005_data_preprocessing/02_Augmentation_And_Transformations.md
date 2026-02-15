# Augmentation and Data Transformations

## Table of Contents

1. [Augmentation Layers](#1-augmentation-layers)
2. [Manual Tensor Preprocessing](#2-manual-tensor-preprocessing)
3. [Data Transformation Operations](#3-data-transformation-operations)

---

## 1. Augmentation Layers

Augmentation layers apply random transformations during training to improve generalization. They are typically used only during training, not inference.

### RandomFlip

**RandomFlip** flips images horizontally, vertically, or both. The `mode` parameter: `"horizontal"`, `"vertical"`, or `"horizontal_and_vertical"`.

```python
flip_layer = tf.keras.layers.RandomFlip(mode="horizontal")
flipped = flip_layer(image_batch)
```

### RandomRotation

**RandomRotation** rotates images by a random angle. `factor` is a fraction of 2*pi (e.g., 0.2 means +/- 20% of 360 degrees).

```python
rot_layer = tf.keras.layers.RandomRotation(0.2)
rotated = rot_layer(image_batch)
```

### RandomZoom

**RandomZoom** zooms in or out. Positive `height_factor`/`width_factor` zooms out (adds padding); negative zooms in (crops).

```python
zoom_layer = tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2)
zoomed = zoom_layer(image_batch)
```

### RandomContrast

**RandomContrast** randomly adjusts contrast. `factor` controls the range (e.g., 0.3 means contrast varies by +/- 30%).

```python
contrast_layer = tf.keras.layers.RandomContrast(0.3)
contrasted = contrast_layer(image_batch)
```

### RandomCrop

**RandomCrop** extracts a random crop of specified size. Input must be larger than crop dimensions.

```python
crop_layer = tf.keras.layers.RandomCrop(32, 32)
cropped = crop_layer(image_batch)
```

### Combined Augmentation Pipeline

```python
aug_pipeline = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.2)
])
augmented = aug_pipeline(images)
```

### Augmentation Best Practices

- Apply augmentation inside `tf.data` pipeline or as model layers.
- Use `tf.keras.layers` for graph-mode compatibility and export.
- Disable at inference by setting `training=False` or excluding layers.

---

## 2. Manual Tensor Preprocessing

When preprocessing layers are insufficient, use raw TensorFlow ops for custom transformations.

### L2 Normalization

**tf.math.l2_normalize** scales vectors to unit norm along specified axis.

```python
x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
normalized = tf.math.l2_normalize(x, axis=-1)
norms = tf.norm(normalized, axis=-1)
```

### Standardization (Z-Score)

Manual z-score: subtract mean, divide by std.

```python
mean = tf.reduce_mean(data, axis=0)
std = tf.math.reduce_std(data, axis=0)
standardized = (data - mean) / (std + 1e-8)
```

### Clipping

**tf.clip_by_value** constrains values to a range. **tf.clip_by_global_norm** clips gradients by global norm.

```python
clipped = tf.clip_by_value(vals, 0.0, 5.0)
clipped_grads, _ = tf.clip_by_global_norm(grads, max_norm=2.0)
```

### Min-Max Scaling

```python
min_val = tf.reduce_min(raw, axis=0)
max_val = tf.reduce_max(raw, axis=0)
scaled = (raw - min_val) / (max_val - min_val + 1e-8)
```

### When to Use Manual Ops

- Custom formulas not covered by layers.
- Preprocessing outside the model (e.g., in `tf.data.map`).
- Gradient flow through preprocessing is not needed.

---

## 3. Data Transformation Operations

Transformations alter the distribution or representation of data for better model fit.

### Log Transform

Reduces skew in right-tailed distributions. Use `log(x+1)` or `log1p(x)` to handle zeros.

```python
log_x = tf.math.log(x + 1.0)
log1p_x = tf.math.log1p(x)
```

### Power Transform

**tf.pow** applies power transformations: square root (0.5), square (2.0), etc.

```python
sqrt_x = tf.pow(data, 0.5)
sq_x = tf.pow(data, 2.0)
```

### Binning (Discretization)

Convert continuous to discrete bins via **Discretization** layer or manual bucketing.

```python
disc_layer = tf.keras.layers.Discretization(bin_boundaries=[1.0, 2.0, 3.0, 4.0])
bin_indices = disc_layer(continuous_values)
```

### One-Hot Encoding

**tf.one_hot** converts integer indices to one-hot vectors.

```python
onehot = tf.one_hot(labels, depth=num_classes)
```

### Binarization (Thresholding)

```python
binary = tf.cast(tensor > threshold, tf.float32)
```

### Reciprocal and Other Transforms

```python
recip = tf.math.reciprocal(x)
```

### Transform Selection Guide

| Transform | Use Case |
|-----------|----------|
| log / log1p | Right-skewed, count data |
| sqrt | Mild skew, variance stabilization |
| power | Box-Cox style normalization |
| binning | Non-linear relationships |
| one-hot | Categorical to numeric |
