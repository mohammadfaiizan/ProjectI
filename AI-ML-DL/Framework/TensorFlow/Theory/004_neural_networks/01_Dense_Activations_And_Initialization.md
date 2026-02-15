# Dense Layers, Activations, Initialization, and Regularization

## Table of Contents

1. [Dense Layer Fundamentals](#1-dense-layer-fundamentals)
2. [Activation Functions](#2-activation-functions)
3. [Weight Initialization](#3-weight-initialization)
4. [Regularization](#4-regularization)

---

## 1. Dense Layer Fundamentals

The **Dense** layer is the most fundamental building block of neural networks in TensorFlow/Keras. It implements the operation `output = activation(dot(input, kernel) + bias)`.

### Key Concepts

- **units**: Number of output dimensions (neurons).
- **activation**: Nonlinearity applied to the output (e.g., ReLU, softmax).
- **use_bias**: Whether to include a bias vector.
- **kernel**: The weight matrix of shape `(input_dim, units)`.
- **bias**: The bias vector of shape `(units,)`.

### Dense Layer Syntax

```python
import tensorflow as tf

layer = tf.keras.layers.Dense(64, activation='relu', input_shape=(32,))
x = tf.random.normal((2, 32))
out = layer(x)
print(out.shape)  # (2, 64)

layer2 = tf.keras.layers.Dense(10, use_bias=True)
layer2.build((None, 64))
kernel = layer2.kernel  # shape (64, 10)
bias = layer2.bias      # shape (10,)
```

### Accessing Weights

After calling `build()` or a forward pass, you can access `layer.kernel` and `layer.bias`. The `layer.weights` list contains all trainable parameters.

---

## 2. Activation Functions

Activation functions introduce **nonlinearity** into the network, enabling it to learn complex patterns.

### ReLU Family

| Activation | Formula | Use Case |
|------------|---------|----------|
| ReLU | max(0, x) | Default for hidden layers |
| LeakyReLU | max(alpha*x, x) | Avoids dead neurons |
| PReLU | max(alpha*x, x), alpha learned | Per-channel learnable slope |
| ELU | x if x>0 else alpha*(exp(x)-1) | Smoother, negative saturation |
| SELU | scale * ELU(x) | Self-normalizing networks |

### Sigmoid and Tanh

- **Sigmoid**: `1 / (1 + exp(-x))`, output in (0, 1). Used for binary classification outputs.
- **Tanh**: `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`, output in (-1, 1). Zero-centered alternative to sigmoid.

### Softmax

Normalizes outputs to a probability distribution: `exp(x_i) / sum(exp(x_j))`. Used for multi-class classification.

### Modern Activations

- **GELU**: Gaussian Error Linear Unit. Smooth approximation of ReLU, used in Transformers.
- **Swish**: `x * sigmoid(x)`. Often outperforms ReLU in deep networks.

### Code Examples

```python
x = tf.constant([[-2.0, -1.0, 0.0, 1.0, 2.0]])

relu = tf.keras.layers.ReLU()
leaky = tf.keras.layers.LeakyReLU(alpha=0.1)
elu = tf.keras.layers.ELU(alpha=1.0)
gelu = tf.keras.layers.Activation('gelu')
swish = tf.keras.layers.Activation('swish')
sigmoid = tf.keras.layers.Activation('sigmoid')
tanh = tf.keras.layers.Activation('tanh')
softmax = tf.keras.layers.Softmax()

print(relu(x))      # [[0, 0, 0, 1, 2]]
print(leaky(x))     # [[-0.2, -0.1, 0, 1, 2]]
print(softmax(x))   # sums to 1 along last axis
```

---

## 3. Weight Initialization

Proper **initialization** prevents vanishing/exploding gradients and speeds convergence.

### Glorot (Xavier) Uniform/Normal

Designed for tanh/sigmoid. Variance = 2 / (fan_in + fan_out).

```python
glorot = tf.keras.initializers.GlorotUniform(seed=42)
layer = tf.keras.layers.Dense(64, kernel_initializer=glorot, input_shape=(32,))
```

### He (Kaiming) Normal/Uniform

Designed for ReLU. Variance = 2 / fan_in. Use with ReLU-based networks.

```python
he = tf.keras.initializers.HeNormal(seed=42)
layer = tf.keras.layers.Dense(64, kernel_initializer=he, input_shape=(32,))
```

### Orthogonal

Initializes weights as orthogonal matrices. Preserves gradient norm, useful for RNNs.

```python
orth = tf.keras.initializers.Orthogonal(seed=42)
layer = tf.keras.layers.Dense(32, kernel_initializer=orth, input_shape=(32,))
```

### Zeros and Ones

- **Zeros**: Default for bias. `tf.keras.initializers.Zeros()`
- **Ones**: Rare. `tf.keras.initializers.Ones()`

### Lecun Normal

Variance = 1 / fan_in. Used with SELU.

### Custom Initializer

```python
def custom_init(shape, dtype=None):
    return tf.random.normal(shape, mean=0.1, stddev=0.05, dtype=dtype)

layer = tf.keras.layers.Dense(64, kernel_initializer=custom_init, input_shape=(32,))
```

---

## 4. Regularization

Regularization reduces overfitting by penalizing large weights or activations.

### L1 Regularization

Penalizes absolute values: `loss += l1 * sum(|w|)`. Encourages sparsity.

### L2 Regularization

Penalizes squared values: `loss += l2 * sum(w^2)`. Also known as weight decay.

### L1L2 (Elastic Net)

Combines both: `loss += l1 * sum(|w|) + l2 * sum(w^2)`.

### Usage in Layers

```python
l1 = tf.keras.regularizers.L1(l1=0.01)
l2 = tf.keras.regularizers.L2(l2=0.01)
l1l2 = tf.keras.regularizers.L1L2(l1=0.01, l2=0.01)

layer = tf.keras.layers.Dense(64, kernel_regularizer=l2, input_shape=(32,))
layer = tf.keras.layers.Dense(64, bias_regularizer=l1, input_shape=(32,))
layer = tf.keras.layers.Dense(64, activity_regularizer=l2, input_shape=(32,))
```

### Regularizer Types

| Type | Applies To | When Computed |
|------|------------|---------------|
| kernel_regularizer | Weight matrix | After build() |
| bias_regularizer | Bias vector | After build() |
| activity_regularizer | Layer output | After call() |

### Full Model Example

```python
l2 = tf.keras.regularizers.L2(l2=0.01)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, kernel_regularizer=l2, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, kernel_regularizer=l2, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(x_train, y_train, epochs=5)
```

The regularization losses are automatically added to `model.losses` and included in the total loss during training.
