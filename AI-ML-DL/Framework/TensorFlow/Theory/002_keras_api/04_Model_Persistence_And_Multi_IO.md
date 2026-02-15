# Model Persistence and Multi-IO

## Table of Contents

1. [SavedModel Format](#1-savedmodel-format)
2. [HDF5 Format](#2-hdf5-format)
3. [Weights-Only Saving](#3-weights-only-saving)
4. [Loading Models](#4-loading-models)
5. [Multi-Input Models](#5-multi-input-models)
6. [Multi-Output Models](#6-multi-output-models)
7. [Shared Layers](#7-shared-layers)
8. [Model Composition Patterns](#8-model-composition-patterns)

---

## 1. SavedModel Format

**SavedModel** is TensorFlow's default serialization format. It stores the model architecture, weights, and computation graph in a directory structure. It is the recommended format for deployment and cross-platform use.

### tf.saved_model.save

Saves a model or any object with a trackable structure (e.g., modules, layers, variables).

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Save to directory
tf.saved_model.save(model, '/path/to/saved_model')
```

### tf.saved_model.load

Loads a SavedModel. Returns a Python object that can be used for inference.

```python
loaded = tf.saved_model.load('/path/to/saved_model')

# For Keras models saved with saved_model.save:
# Use tf.keras.models.load_model for full Keras functionality
```

### Signatures

**Signatures** define the inputs and outputs of a SavedModel. They enable tools like TensorFlow Serving to know how to call the model.

```python
# Save with explicit signature
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 784], dtype=tf.float32)])
def serve(x):
    return model(x)

tf.saved_model.save(
    model,
    '/path/to/saved_model',
    signatures={'serving_default': serve}
)

# Inspect signatures
loaded = tf.saved_model.load('/path/to/saved_model')
print(list(loaded.signatures.keys()))
```

### SavedModel Directory Structure

| Path | Content |
|------|---------|
| `saved_model.pb` | Graph definition |
| `variables/` | Variable values (checkpoints) |
| `assets/` | Additional files (e.g., vocab) |
| `fingerprint.pb` | Model fingerprint |

---

## 2. HDF5 Format

The **HDF5** format stores the model as a single `.h5` or `.keras` file. It is convenient for sharing and version control but is less portable than SavedModel for non-Keras consumers.

### model.save (HDF5)

Saves architecture, weights, optimizer state, and training config in one file.

```python
model.save('/path/to/model.h5')

# Or with .keras extension (Keras 3 style)
model.save('/path/to/model.keras')
```

### What Gets Saved

| Component | Saved in HDF5 |
|-----------|---------------|
| Architecture | Yes |
| Weights | Yes |
| Optimizer state | Yes (if compile was called) |
| Training config | Yes |
| Custom objects | Requires handling |

---

## 3. Weights-Only Saving

Saving only weights reduces file size and is useful when the architecture is defined in code.

### save_weights

```python
# HDF5 format
model.save_weights('/path/to/weights.h5')

# TensorFlow checkpoint format (default)
model.save_weights('/path/to/checkpoint')
```

### load_weights

```python
# Must have matching architecture
model.load_weights('/path/to/weights.h5')

# Load into different model (if compatible)
model2 = create_model()
model2.load_weights('/path/to/weights.h5')
```

### By Name vs By Position

Weights are matched by layer name. If architectures differ, use `by_name=True` to load only matching layers.

```python
model.load_weights('/path/to/weights.h5', by_name=True)
```

---

## 4. Loading Models

### tf.keras.models.load_model

Loads a full Keras model (architecture + weights). Works with both SavedModel and HDF5.

```python
# From SavedModel directory
model = tf.keras.models.load_model('/path/to/saved_model')

# From HDF5 file
model = tf.keras.models.load_model('/path/to/model.h5')
```

### Custom Objects

When the model uses custom layers, losses, or metrics, pass them to `load_model`.

```python
model = tf.keras.models.load_model(
    '/path/to/model.h5',
    custom_objects={'CustomLayer': CustomLayer, 'custom_loss': custom_loss}
)
```

### compile=False

Load without recompiling (e.g., when you will compile with different settings).

```python
model = tf.keras.models.load_model('/path/to/model.h5', compile=False)
model.compile(optimizer='adam', loss='mse')
```

---

## 5. Multi-Input Models

The **Functional API** supports models with multiple inputs. Each input is a separate tensor.

### Building a Multi-Input Model

```python
from tensorflow.keras import layers, Model

# Define inputs
text_input = layers.Input(shape=(100,), name='text')
image_input = layers.Input(shape=(28, 28, 1), name='image')

# Process each branch
text_features = layers.Dense(64, activation='relu')(text_input)
image_features = layers.Flatten()(image_input)
image_features = layers.Dense(64, activation='relu')(image_features)

# Concatenate and add output
combined = layers.concatenate([text_features, image_features])
output = layers.Dense(10, activation='softmax')(combined)

model = Model(inputs=[text_input, image_input], outputs=output)
```

### Training and Inference

```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Training: pass list or dict of arrays
model.fit(
    [x_text, x_image],
    y_labels,
    epochs=5
)

# Or with named inputs
model.fit(
    {'text': x_text, 'image': x_image},
    y_labels,
    epochs=5
)

# Inference
predictions = model.predict([x_text_test, x_image_test])
```

---

## 6. Multi-Output Models

Models can have multiple outputs, each with its own loss. Useful for multi-task learning.

### Building a Multi-Output Model

```python
inputs = layers.Input(shape=(784,))
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)

# Multiple outputs
class_output = layers.Dense(10, activation='softmax', name='class')(x)
reg_output = layers.Dense(1, name='regression')(x)

model = Model(inputs=inputs, outputs=[class_output, reg_output])
```

### Compiling with Multiple Losses

```python
model.compile(
    optimizer='adam',
    loss={
        'class': 'sparse_categorical_crossentropy',
        'regression': 'mse'
    },
    loss_weights={'class': 1.0, 'regression': 0.5},
    metrics={'class': ['accuracy']}
)
```

### Training

```python
model.fit(
    x_train,
    [y_class, y_reg],
    epochs=10
)
```

---

## 7. Shared Layers

**Shared layers** are reused across different parts of the model. The same layer instance is applied to multiple inputs, so weights are shared.

### Example: Siamese Network

```python
# Shared embedding layer
embedding = layers.Dense(64, activation='relu')

input_a = layers.Input(shape=(100,))
input_b = layers.Input(shape=(100,))

# Same layer applied to both inputs
encoded_a = embedding(input_a)
encoded_b = embedding(input_b)

# Merge (e.g., distance)
merged = layers.concatenate([encoded_a, encoded_b])
output = layers.Dense(1, activation='sigmoid')(merged)

model = Model(inputs=[input_a, input_b], outputs=output)
```

### Weight Sharing Benefits

| Benefit | Description |
|---------|-------------|
| Parameter efficiency | Fewer parameters to train |
| Consistency | Same transformation for similar inputs |
| Regularization | Shared weights act as regularizer |

---

## 8. Model Composition Patterns

### Encoder-Decoder with Shared Encoder

```python
encoder_input = layers.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation='relu')(encoder_input)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
encoder_output = layers.Dense(64, activation='relu')(x)

encoder = Model(encoder_input, encoder_output, name='encoder')

# Decoder 1
dec1_input = layers.Input(shape=(64,))
dec1_out = layers.Dense(10, activation='softmax')(dec1_input)
decoder1 = Model(dec1_input, dec1_out, name='decoder1')

# Decoder 2
dec2_input = layers.Input(shape=(64,))
dec2_out = layers.Dense(1)(dec2_input)
decoder2 = Model(dec2_input, dec2_out, name='decoder2')

# Composed model
input_img = layers.Input(shape=(28, 28, 1))
encoded = encoder(input_img)
out1 = decoder1(encoded)
out2 = decoder2(encoded)
composed = Model(input_img, [out1, out2])
```

### Submodel Extraction

```python
# Extract encoder from full model
full_model = load_full_model()
encoder = Model(
    full_model.input,
    full_model.get_layer('encoder').output
)
```

### Summary of Persistence Options

| Method | Format | Use Case |
|--------|--------|----------|
| `tf.saved_model.save` | SavedModel | Deployment, TF Serving |
| `model.save` | HDF5/.keras | Keras-only, sharing |
| `model.save_weights` | HDF5 or checkpoint | Weights only |
| `tf.keras.models.load_model` | Both | Loading full model |
