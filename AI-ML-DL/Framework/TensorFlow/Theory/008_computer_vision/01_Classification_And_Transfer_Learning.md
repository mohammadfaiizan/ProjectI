# Classification and Transfer Learning

## Table of Contents

1. [CNN Architectures for Classification](#1-cnn-architectures-for-classification)
2. [Building CNNs in TensorFlow](#2-building-cnns-in-tensorflow)
3. [Transfer Learning Overview](#3-transfer-learning-overview)
4. [tf.keras.applications](#4-tfkerasapplications)
5. [Loading Pretrained Weights](#5-loading-pretrained-weights)
6. [Modifying Final Layers](#6-modifying-final-layers)
7. [Fine-Tuning Strategies](#7-fine-tuning-strategies)

---

## 1. CNN Architectures for Classification

Convolutional Neural Networks (CNNs) are the standard architecture for image classification. They combine **convolutional layers** (local feature extraction), **pooling layers** (spatial downsampling), and **dense layers** (classification head).

### Core Components

| Component | Purpose |
|-----------|---------|
| Conv2D | Learn spatial filters, extract features |
| MaxPooling2D | Reduce spatial dimensions, add invariance |
| BatchNormalization | Stabilize training, faster convergence |
| GlobalAveragePooling2D | Replace flatten, reduce parameters |
| Dense | Final classification |

### Evolution of Architectures

Classic architectures progressed from shallow (LeNet, AlexNet) to deep residual networks (ResNet, DenseNet) and efficient mobile models (MobileNet, EfficientNet).

---

## 2. Building CNNs in TensorFlow

### Basic CNN from Scratch

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_cnn(input_shape=(224, 224, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)
```

### With Batch Normalization

```python
x = layers.Conv2D(64, 3, padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D(2)(x)
```

---

## 3. Transfer Learning Overview

**Transfer learning** reuses a model pretrained on a large dataset (e.g., ImageNet) for a new task. Instead of training from scratch, we leverage learned features.

### Feature Extraction vs Fine-Tuning

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Feature extraction** | Freeze base, train only new head | Small dataset, similar domain |
| **Fine-tuning** | Unfreeze some layers, train end-to-end | More data, domain shift |

### Benefits

- Faster convergence
- Better performance with limited data
- Reduced compute requirements

---

## 4. tf.keras.applications

**tf.keras.applications** provides pretrained models with ImageNet weights. Each model has configurable parameters.

### ResNet50

Deep residual network with 50 layers. Strong baseline for many tasks.

```python
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
```

### MobileNetV2

Lightweight architecture for mobile and edge. Uses inverted residual blocks and depthwise separable convolutions.

```python
base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3),
    alpha=1.0  # Width multiplier (0.35 to 1.0)
)
```

### EfficientNet

Scaled architecture (B0–B7) balancing depth, width, and resolution. State-of-the-art efficiency.

```python
base_model = tf.keras.applications.EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
```

### VGG16

Classic architecture with simple stacked convolutions. Good for understanding and teaching.

```python
base_model = tf.keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
```

### Comparison Table

| Model | Parameters | Input Size | Speed | Accuracy |
|-------|------------|------------|-------|----------|
| VGG16 | 138M | 224x224 | Slow | Good |
| ResNet50 | 25.6M | 224x224 | Medium | High |
| MobileNetV2 | 3.5M | 224x224 | Fast | Good |
| EfficientNetB0 | 5.3M | 224x224 | Fast | High |

---

## 5. Loading Pretrained Weights

### weights Parameter

```python
# ImageNet pretrained
model = tf.keras.applications.ResNet50(weights='imagenet')

# Random initialization
model = tf.keras.applications.ResNet50(weights=None)

# Load from file
model = tf.keras.applications.ResNet50(weights='/path/to/weights.h5')
```

### include_top

- `include_top=True`: Full model with ImageNet classification head (1000 classes)
- `include_top=False`: Feature extractor only (no final dense layers)

```python
# For transfer learning: typically include_top=False
base = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
```

### input_shape

Must match the expected input. Common sizes: 224x224 (ResNet, VGG), 299x299 (Inception, Xception).

---

## 6. Modifying Final Layers

Replace the original classification head with one suited to your task.

### Adding a Custom Head

```python
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base
base_model.trainable = False

# Add new head
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
```

### GlobalAveragePooling2D vs Flatten

**GlobalAveragePooling2D** reduces each feature map to its mean. Fewer parameters, less overfitting, standard in modern architectures.

```python
# Flatten: many parameters
x = layers.Flatten()(base_output)

# GlobalAveragePooling2D: recommended
x = layers.GlobalAveragePooling2D()(base_output)
```

---

## 7. Fine-Tuning Strategies

### When to Fine-Tune

- Dataset is large enough (thousands of images)
- Domain differs from ImageNet
- Feature extraction plateau reached

### Strategy 1: Unfreeze Top Layers Only

Unfreeze the last few convolutional blocks; keep earlier layers frozen.

```python
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = True

# Freeze all except last block
for layer in base_model.layers[:-10]:
    layer.trainable = False
```

### Strategy 2: Gradual Unfreezing

Train in stages: first head only, then unfreeze progressively.

```python
# Stage 1: Train head only
base_model.trainable = False
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(train_ds, epochs=5)

# Stage 2: Unfreeze top, use lower LR
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy'
)
model.fit(train_ds, epochs=10)
```

### Strategy 3: Per-Layer Learning Rates

Use different learning rates for base vs head (e.g., via custom training loop or optimizers).

```python
# Lower LR for base layers
optimizer = tf.keras.optimizers.Adam(1e-4)
for layer in model.layers:
    if 'resnet50' in layer.name:
        layer.trainable = True
```

### Best Practices

| Practice | Reason |
|----------|--------|
| Train head first | Stabilize before fine-tuning |
| Use low LR for base | Preserve pretrained features |
| Use data augmentation | Reduce overfitting |
| Monitor validation | Detect overfitting early |

### Complete Transfer Learning Example

```python
# 1. Load base
base = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base.trainable = False

# 2. Build model
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(5, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

# 3. Train head
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_ds, validation_data=val_ds, epochs=5)

# 4. Fine-tune
base.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_ds, validation_data=val_ds, epochs=5)
```
