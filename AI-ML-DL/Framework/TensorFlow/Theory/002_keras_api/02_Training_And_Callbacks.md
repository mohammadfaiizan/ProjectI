# Training and Callbacks

## Table of Contents

1. [Compile, Fit, and Evaluate](#compile-fit-and-evaluate)
2. [Built-in Callbacks](#built-in-callbacks)
3. [Custom Callbacks](#custom-callbacks)
4. [TensorBoard Callback](#tensorboard-callback)
5. [Training History and Visualization](#training-history-and-visualization)

---

## Compile, Fit, and Evaluate

### model.compile()

Before training, a model must be **compiled** with an optimizer, loss function, and optional metrics. This configures the training process.

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

**Key parameters:**
- **optimizer**: Algorithm for weight updates (e.g., `'adam'`, `'sgd'`, `tf.keras.optimizers.Adam(learning_rate=0.001)`)
- **loss**: Objective to minimize (e.g., `'mse'`, `'binary_crossentropy'`, `'sparse_categorical_crossentropy'`)
- **metrics**: List of metrics to monitor during training and evaluation

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)
```

### model.fit()

The **fit** method trains the model for a fixed number of epochs.

```python
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    validation_data=(x_val, y_val),
    shuffle=True,
    verbose=1
)
```

**Key parameters:**
- **epochs**: Number of passes over the training data
- **batch_size**: Number of samples per gradient update
- **validation_split**: Fraction of training data to use for validation (e.g., 0.2 = 20%)
- **validation_data**: Explicit validation set (overrides validation_split if both provided)
- **shuffle**: Whether to shuffle training data each epoch
- **callbacks**: List of callback instances
- **verbose**: 0 (silent), 1 (progress bar), 2 (one line per epoch)

**Return value:** A `History` object whose `history` attribute is a dictionary of metric values per epoch.

### model.evaluate()

**evaluate** computes the loss and metrics on test/validation data.

```python
results = model.evaluate(x_test, y_test, batch_size=32, verbose=1)
# results is a list: [loss, metric1, metric2, ...]
loss, accuracy = results[0], results[1]
```

### model.predict()

**predict** generates output predictions for input samples.

```python
predictions = model.predict(x_test, batch_size=32, verbose=1)
# For classification: use np.argmax(predictions, axis=1) for class indices
```

---

## Built-in Callbacks

Callbacks are objects passed to `fit()` that get called at various stages of training. They enable monitoring, checkpointing, and early stopping.

### EarlyStopping

Stops training when a monitored metric stops improving. Optionally restores the best weights.

```python
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    min_delta=0.001,
    mode='min'
)
```

**Key parameters:**
- **monitor**: Metric to watch (`'val_loss'`, `'val_accuracy'`, etc.)
- **patience**: Epochs to wait for improvement before stopping
- **restore_best_weights**: If True, restore weights from the epoch with the best monitored value
- **min_delta**: Minimum change to qualify as an improvement
- **mode**: `'min'` (for loss) or `'max'` (for accuracy)

### ModelCheckpoint

Saves the model (or weights) at specified intervals or when a metric improves.

```python
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)
```

**Key parameters:**
- **filepath**: Path to save the model
- **monitor**: Metric to determine when to save
- **save_best_only**: If True, save only when the monitored metric improves
- **save_weights_only**: If True, save only weights (not full model)

### ReduceLROnPlateau

Reduces the learning rate when a metric has stopped improving.

```python
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)
```

**Key parameters:**
- **factor**: New lr = lr * factor
- **patience**: Epochs to wait before reducing
- **min_lr**: Lower bound on learning rate

### CSVLogger

Logs epoch results to a CSV file.

```python
csv_logger = tf.keras.callbacks.CSVLogger('training_log.csv')
```

### Using Multiple Callbacks

```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best.keras', monitor='val_accuracy', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),
    tf.keras.callbacks.CSVLogger('history.csv')
]
history = model.fit(x_train, y_train, epochs=50, validation_split=0.2, callbacks=callbacks)
```

---

## Custom Callbacks

Create custom callbacks by subclassing `tf.keras.callbacks.Callback` and overriding the appropriate methods.

### Callback Lifecycle Methods

| Method | When Called |
|--------|-------------|
| `on_train_begin` | At the start of training |
| `on_train_end` | At the end of training |
| `on_epoch_begin` | At the start of each epoch |
| `on_epoch_end` | At the end of each epoch |
| `on_batch_begin` | At the start of each batch |
| `on_batch_end` | At the end of each batch |

### Example: Custom Callback

```python
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_times = []

    def on_train_begin(self, logs=None):
        print("Training started.")

    def on_train_end(self, logs=None):
        print("Training ended.")

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time()
        print(f"Epoch {epoch + 1} began.")

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self._epoch_start
        self.epoch_times.append(elapsed)
        if logs:
            print(f"Epoch {epoch + 1} end - loss: {logs.get('loss', 0):.4f}")
```

The `logs` dictionary contains metrics such as `loss`, `accuracy`, `val_loss`, `val_accuracy`.

### Stopping Training from a Callback

Set `self.model.stop_training = True` in any callback method to stop training:

```python
def on_epoch_end(self, epoch, logs=None):
    if logs and logs.get('loss', 0) < 0.01:
        self.model.stop_training = True
```

---

## TensorBoard Callback

**TensorBoard** provides visualization of training metrics, model graphs, and weight histograms.

### Basic Usage

```python
tensorboard_cb = tf.keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    write_graph=True,
    write_images=False
)
model.fit(x_train, y_train, epochs=10, callbacks=[tensorboard_cb])
```

**Key parameters:**
- **log_dir**: Directory for TensorBoard log files
- **histogram_freq**: Frequency (in epochs) at which to compute weight histograms (0 = disabled)
- **write_graph**: Whether to visualize the model graph
- **write_images**: Whether to write model weights as images

### Viewing in TensorBoard

```bash
tensorboard --logdir=./logs
```

Then open the URL (typically http://localhost:6006) in a browser.

### What TensorBoard Shows

- **Scalars**: Loss and metrics over time
- **Graphs**: Model architecture
- **Distributions**: Weight and activation distributions (when histogram_freq > 0)
- **Histograms**: Weight histograms per layer

---

## Training History and Visualization

### History Object

`model.fit()` returns a `History` object. Its `history` attribute is a dictionary mapping metric names to lists of values (one per epoch).

```python
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)

print(history.history.keys())
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

loss_per_epoch = history.history['loss']
val_accuracy_per_epoch = history.history['val_accuracy']
```

### Plotting Loss and Accuracy

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['loss'], label='train')
axes[0].plot(history.history['val_loss'], label='val')
axes[0].set_title('Loss')
axes[0].legend()

axes[1].plot(history.history['accuracy'], label='train')
axes[1].plot(history.history['val_accuracy'], label='val')
axes[1].set_title('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.savefig('training_history.png')
```

### Metric Tracking

Useful metrics to track:
- **loss** vs **val_loss**: Overfitting if val_loss increases while loss decreases
- **accuracy** vs **val_accuracy**: Generalization gap
- **Best epoch**: Epoch with best validation metric (e.g., for EarlyStopping)

```python
best_epoch = np.argmin(history.history['val_loss'])
best_val_loss = history.history['val_loss'][best_epoch]
print(f"Best validation loss at epoch {best_epoch + 1}: {best_val_loss:.4f}")
```
