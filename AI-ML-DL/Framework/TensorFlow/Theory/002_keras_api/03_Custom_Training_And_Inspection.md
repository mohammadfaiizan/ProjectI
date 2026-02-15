# Custom Training and Inspection

## Table of Contents

1. [Model Summary and Inspection](#model-summary-and-inspection)
2. [Layer Access and Weight Manipulation](#layer-access-and-weight-manipulation)
3. [Custom train_step and test_step](#custom-train_step-and-test_step)
4. [Custom Training Loop with GradientTape](#custom-training-loop-with-gradienttape)

---

## Model Summary and Inspection

### model.summary()

The **summary** method prints a text representation of the model: layer types, output shapes, and parameter counts.

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.summary()
```

**Output:**
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 64)                50240
dense_1 (Dense)              (None, 32)                2080
dense_2 (Dense)              (None, 10)                330
=================================================================
Total params: 52,650
Trainable params: 52,650
Non-trainable params: 0
_________________________________________________________________
```

**Note:** For subclassed models, call `model.build(input_shape=(None, 784))` before `summary()` so the graph is known.

### model.layers

The **layers** attribute returns a list of all layers in the model. Use it to inspect or modify individual layers.

```python
for i, layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.name}, output shape: {layer.output_shape}")
```

### Accessing Specific Layers

```python
first_layer = model.layers[0]
last_layer = model.layers[-1]

# By name (if layers have names)
dense_layer = model.get_layer('dense')
```

### Layer Output Shape

Each layer has `input_shape` and `output_shape` attributes (after the model is built):

```python
for layer in model.layers:
    print(f"{layer.name}: {layer.input_shape} -> {layer.output_shape}")
```

---

## Layer Access and Weight Manipulation

### get_weights()

**get_weights** returns a list of NumPy arrays: all trainable and non-trainable weights of the model. For a Dense layer, each layer contributes two arrays: kernel and bias.

```python
weights = model.get_weights()
print(f"Number of weight arrays: {len(weights)}")
for i, w in enumerate(weights):
    print(f"Weight {i}: shape={w.shape} dtype={w.dtype}")
```

### set_weights()

**set_weights** takes a list of NumPy arrays and assigns them to the model's layers. The list must match the structure expected by `get_weights()`.

```python
# Copy weights from model to model2
weights = model.get_weights()
model2 = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model2.set_weights(weights)
```

**Use cases:**
- Transfer weights between identical architectures
- Initialize from pre-trained weights
- Implement weight averaging (e.g., EMA)

### Per-Layer Weights

```python
layer = model.layers[0]
kernel, bias = layer.kernel, layer.bias
print(kernel.shape, bias.shape)
```

---

## Custom train_step and test_step

When the default training logic is insufficient, override **train_step** and **test_step** in a subclassed model. This keeps the high-level `fit()` and `evaluate()` APIs while customizing the inner loop.

### Overriding train_step

```python
class CustomTrainStepModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(y=y, y_pred=y_pred)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}
```

**Key points:**
- `data` is typically `(x, y)` for supervised learning
- Use `tf.GradientTape` to compute gradients
- Call `self.compute_loss()` to use the compiled loss
- Return a dict of metrics to log

### Overriding test_step

```python
def test_step(self, data):
    x, y = data
    y_pred = self(x, training=False)
    loss = self.compute_loss(y=y, y_pred=y_pred)
    return {"loss": loss}
```

No gradient computation or weight updates in `test_step`.

### Adding Custom Metrics

```python
def train_step(self, data):
    x, y = data
    with tf.GradientTape() as tape:
        y_pred = self(x, training=True)
        loss = self.compute_loss(y=y, y_pred=y_pred)
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    self.compiled_metrics.update_state(y, y_pred)
    return {m.name: m.result() for m in self.metrics}
```

### When to Use

- Custom loss computations (e.g., auxiliary losses)
- Gradient clipping
- Mixed precision training
- Custom regularization inside the step
- Multi-task learning with custom weighting

---

## Custom Training Loop with GradientTape

For full control over the training loop (e.g., different batch logic, manual epoch structure), use **tf.GradientTape** and implement the loop yourself.

### Basic Structure

```python
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.CategoricalCrossentropy()

for epoch in range(epochs):
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)
            loss = loss_fn(y_batch, y_pred)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

### GradientTape

**tf.GradientTape** records operations for automatic differentiation. Operations inside the `with` block are recorded; gradients are computed with `tape.gradient(loss, variables)`.

```python
with tf.GradientTape() as tape:
    y_pred = model(x, training=True)
    loss = loss_fn(y, y_pred)

gradients = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

### Persistent Tape (Multiple gradient calls)

By default, a tape can only compute gradients once. Use `persistent=True` for multiple gradient computations:

```python
with tf.GradientTape(persistent=True) as tape:
    y_pred = model(x, training=True)
    loss1 = loss_fn(y, y_pred)
    loss2 = auxiliary_loss(y, y_pred)

grad1 = tape.gradient(loss1, model.trainable_variables)
grad2 = tape.gradient(loss2, model.trainable_variables)
del tape
```

### Gradient Clipping

```python
gradients = tape.gradient(loss, model.trainable_variables)
gradients, _ = tf.clip_by_global_norm(gradients, max_norm=1.0)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

### Metrics in Custom Loop

```python
epoch_loss = tf.keras.metrics.Mean()
epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

for epoch in range(epochs):
    epoch_loss.reset_states()
    epoch_accuracy.reset_states()

    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)
            loss = loss_fn(y_batch, y_pred)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        epoch_loss.update_state(loss)
        epoch_accuracy.update_state(y_batch, y_pred)

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss.result():.4f}, Acc: {epoch_accuracy.result():.4f}")
```

### fit() vs Custom Loop

| Aspect | model.fit() | Custom Loop |
|--------|-------------|-------------|
| Callbacks | Built-in | Manual implementation |
| Progress bar | Automatic | Manual |
| Metrics | Automatic | Manual |
| Flexibility | Limited | Full |
| Debugging | Easier | Harder |

Use `fit()` when possible; use a custom loop when you need behavior that callbacks and `train_step` cannot provide.
