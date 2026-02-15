# Custom Layers, Loss Layers, and Container Layers

## Table of Contents

1. [Loss Functions as Layers](#1-loss-functions-as-layers)
2. [Custom Layers: Basic](#2-custom-layers-basic)
3. [Custom Layers: Advanced](#3-custom-layers-advanced)
4. [Container Layers](#4-container-layers)

---

## 1. Loss Functions as Layers

Loss functions can be used standalone or integrated into custom layers for advanced training patterns.

### Standalone Loss Functions

```python
import tensorflow as tf

y_true = tf.constant([[0, 1, 0], [1, 0, 0]], dtype=tf.float32)
y_pred = tf.constant([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05]], dtype=tf.float32)

mse = tf.keras.losses.MeanSquaredError()
cce = tf.keras.losses.CategoricalCrossentropy()
bce = tf.keras.losses.BinaryCrossentropy()
huber = tf.keras.losses.Huber(delta=1.0)
sparse_cce = tf.keras.losses.SparseCategoricalCrossentropy()

print(mse(y_true, y_pred))
print(cce(y_true, y_pred))
```

### Common Losses

| Loss | Use Case |
|------|----------|
| MeanSquaredError | Regression |
| MeanAbsoluteError | Regression, robust to outliers |
| CategoricalCrossentropy | Multi-class, one-hot labels |
| SparseCategoricalCrossentropy | Multi-class, integer labels |
| BinaryCrossentropy | Binary classification |
| Huber | Regression, robust |

### Loss as a Layer

Custom layers can add losses via `self.add_loss()`. These are collected in `model.losses` and added to the training loss.

```python
class LossLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()

    def call(self, inputs):
        y_true, y_pred = inputs
        loss = self.loss_fn(y_true, y_pred)
        self.add_loss(loss)
        return y_pred
```

---

## 2. Custom Layers: Basic

Custom layers extend `tf.keras.layers.Layer` and implement `__init__`, `build`, and `call`.

### Structure

1. **__init__**: Store config (units, activation, etc.). Do not create weights here.
2. **build**: Create weights with `add_weight()`. Called once when input shape is known.
3. **call**: Forward pass logic.

### Example: Custom Dense Layer

```python
class DenseCustom(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        x = tf.matmul(inputs, self.kernel) + self.bias
        return self.activation(x)
```

### add_weight

```python
self.kernel = self.add_weight(
    name='kernel',
    shape=(in_dim, out_dim),
    initializer='glorot_uniform',
    trainable=True,
    dtype=self.dtype
)
```

Weights added via `add_weight` are automatically tracked for training and serialization.

---

## 3. Custom Layers: Advanced

### get_config and from_config

For **serialization** (saving/loading models), implement `get_config` to return a dict of constructor arguments. Keras uses this to recreate the layer.

```python
def get_config(self):
    config = super().get_config()
    config.update({
        'units': self.units,
        'activation': self.activation
    })
    return config

@classmethod
def from_config(cls, config):
    return cls(**config)
```

### compute_output_shape

Returns the output shape given input shape. Used for model building and validation.

```python
def compute_output_shape(self, input_shape):
    return (input_shape[0], self.units)
```

### Full Serializable Layer

```python
class DenseSerializable(tf.keras.layers.Layer):
    def __init__(self, units, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.kernel = self.add_weight(...)
        self.bias = self.add_weight(...)
        super().build(input_shape)

    def call(self, inputs):
        return tf.keras.activations.get(self.activation)(tf.matmul(inputs, self.kernel) + self.bias)

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units, 'activation': self.activation})
        return config

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
```

### Loading Custom Layers

When loading a model with custom layers, pass `custom_objects`:

```python
model = tf.keras.models.load_model('model.h5', custom_objects={'DenseSerializable': DenseSerializable})
```

---

## 4. Container Layers

### TimeDistributed

Applies a layer to **every timestep** of a sequence. Input: `(batch, time, ...)`, output: `(batch, time, ...)`.

```python
x = tf.random.normal((2, 10, 32))
dense = tf.keras.layers.Dense(64, activation='relu')
td = tf.keras.layers.TimeDistributed(dense)
out = td(x)
print(out.shape)  # (2, 10, 64)
```

Useful for applying the same Dense/Conv to each frame of a sequence (e.g., video, per-token classification).

### Lambda

Wraps a function as a layer. No trainable weights. Use for simple operations.

```python
lambda_layer = tf.keras.layers.Lambda(lambda x: tf.square(x))
out = lambda_layer(tf.constant([1.0, 2.0, 3.0]))

lambda_norm = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=-1))
```

**Limitation**: Lambda layers are not easily serializable with custom logic. Prefer subclassing `Layer` for complex logic.

### Wrapper

Base class for wrapping another layer. Override `call` and optionally `compute_output_shape`.

```python
class CustomWrapper(tf.keras.layers.Wrapper):
    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)

    def call(self, inputs, **kwargs):
        return self.layer(inputs, **kwargs)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)
```

### Example: TimeDistributed + LSTM

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10, 32)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation='relu')),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

Each of the 10 timesteps gets a Dense(64) applied independently, then the LSTM processes the sequence.
