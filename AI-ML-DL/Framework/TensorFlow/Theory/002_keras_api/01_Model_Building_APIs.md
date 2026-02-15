# Model Building APIs

## Table of Contents

1. [Introduction](#introduction)
2. [Sequential API](#sequential-api)
3. [Functional API](#functional-api)
4. [Model Subclassing API](#model-subclassing-api)
5. [Model Composition Patterns](#model-composition-patterns)
6. [When to Use Each API](#when-to-use-each-api)

---

## Introduction

TensorFlow Keras provides three primary ways to build neural network models. Each API offers different levels of flexibility and control. Understanding when and how to use each approach is essential for effective model development.

**Key concepts:**
- **Sequential API**: Linear stack of layers, simplest approach
- **Functional API**: Directed acyclic graphs, supports branching and merging
- **Subclassing API**: Full programmatic control via Python classes

---

## Sequential API

The **Sequential** model is the simplest way to build a Keras model. It represents a linear stack of layers where each layer has exactly one input and one output tensor.

### Creating a Sequential Model

Two common patterns for creating Sequential models:

**Pattern 1: Add layers incrementally**

```python
import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
```

**Pattern 2: Pass layers as a list to the constructor**

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### Input Shape Specification

Only the first layer in a Sequential model needs an `input_shape` argument. Subsequent layers infer their input shape automatically. You can also use `tf.keras.layers.Input` explicitly:

```python
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(784,)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
```

### Limitations

- Cannot represent models with multiple inputs or outputs
- Cannot share layers
- Cannot have branching (e.g., skip connections)
- Single path from input to output only

---

## Functional API

The **Functional API** treats models as directed acyclic graphs (DAGs) of layers. It supports multiple inputs, multiple outputs, layer sharing, and non-linear topology.

### Basic Syntax

```python
inputs = tf.keras.Input(shape=(784,))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

**Key concept:** Each layer is callable and returns a tensor. You pass the output of one layer as the input to the next. The `Model` is created by specifying input and output tensors.

### Convolutional Example

```python
inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D(2)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

### Advantages Over Sequential

- Multiple inputs and outputs
- Shared layers
- Non-linear topology (branching, merging)
- Access to intermediate layer outputs
- Model introspection (graph structure is known)

---

## Model Subclassing API

**Model subclassing** provides maximum flexibility by defining models as Python classes that inherit from `tf.keras.Model`. You implement `__init__` to create layers and `call` to define the forward pass.

### Basic Structure

```python
class CustomModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.dense3 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)
```

### Building the Model

Subclassed models do not have a known graph until they are built or run. Call `build()` with an input shape before `summary()`:

```python
model = CustomModel(num_classes=10)
model.build(input_shape=(None, 784))
model.summary()
```

### Convolutional Subclass Example

```python
class CNNSubclass(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool(x)
        return self.dense(x)
```

### Trade-offs

**Pros:**
- Full Python control (loops, conditionals, custom logic)
- Dynamic architectures
- Easy to implement research ideas

**Cons:**
- Graph is not serializable (no `model.summary()` until built)
- Harder to debug
- No automatic shape inference
- May have limitations with `tf.keras.models.load_model` for complex custom code

---

## Model Composition Patterns

### Shared Layers

A single layer instance can be reused across multiple inputs. The layer's weights are shared:

```python
shared_dense = tf.keras.layers.Dense(64, activation='relu', name='shared_dense')

input_a = tf.keras.Input(shape=(32,))
input_b = tf.keras.Input(shape=(32,))

out_a = shared_dense(input_a)
out_b = shared_dense(input_b)

out_a = tf.keras.layers.Dense(10, activation='softmax')(out_a)
out_b = tf.keras.layers.Dense(10, activation='softmax')(out_b)

model = tf.keras.Model(inputs=[input_a, input_b], outputs=[out_a, out_b])
```

### Nesting Models

Models can be used as layers. This enables encoder-decoder patterns and reusable blocks:

```python
encoder = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
    tf.keras.layers.Dense(32, activation='relu')
], name='encoder')

decoder = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
    tf.keras.layers.Dense(32, activation='sigmoid')
], name='decoder')

autoencoder_input = tf.keras.Input(shape=(32,))
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)
autoencoder = tf.keras.Model(autoencoder_input, decoded, name='autoencoder')
```

### Reuse Patterns

- **Shared encoder**: One encoder feeds multiple heads (e.g., classification + regression)
- **Siamese networks**: Same model applied to different inputs
- **Transfer learning**: Use a pre-trained model as a feature extractor

---

## When to Use Each API

| Use Case | Recommended API |
|----------|-----------------|
| Simple linear stack (MLP, basic CNN) | Sequential |
| Multiple inputs/outputs | Functional |
| Shared layers | Functional |
| Skip connections, branching | Functional |
| Need to save/load with full graph | Sequential or Functional |
| Dynamic architectures (variable layers) | Subclassing |
| Custom training logic in forward pass | Subclassing |
| Research prototypes, experiments | Subclassing |
| Production, deployment | Sequential or Functional |

### Summary

- **Sequential**: Fastest to write, limited to single-path models
- **Functional**: Best balance of flexibility and debuggability; supports multi-IO and composition
- **Subclassing**: Maximum flexibility; use when the above are insufficient
