# Optimizer Management and Debugging

## Table of Contents

1. [Optimizer State Management](#1-optimizer-state-management)
2. [Saving and Loading Optimizer State](#2-saving-and-loading-optimizer-state)
3. [Multi-Optimizer Training](#3-multi-optimizer-training)
4. [Custom Optimizers](#4-custom-optimizers)
5. [Optimization Debugging](#5-optimization-debugging)

---

## 1. Optimizer State Management

Optimizers maintain internal state (e.g., momentum buffers, Adam's m and v). For correct resume of training, this state must be saved and restored along with model weights.

### get_weights and set_weights

**get_weights()** returns a list of numpy arrays representing the optimizer state. **set_weights()** restores that state. The order and shape must match.

```python
optimizer = tf.keras.optimizers.Adam(0.001)
model.compile(optimizer=optimizer, loss='mse')
model.fit(x_train, y_train, epochs=5)

opt_state = optimizer.get_weights()
print(f"Optimizer has {len(opt_state)} state tensors")

# Restore into a new optimizer
optimizer_new = tf.keras.optimizers.Adam(0.001)
optimizer_new.set_weights(opt_state)
```

### State Contents by Optimizer

| Optimizer | State Variables |
|-----------|-----------------|
| SGD | None (or momentum buffer if momentum > 0) |
| SGD + momentum | One buffer per variable (velocity) |
| Adam | Two buffers per variable (m, v) |
| AdamW | Same as Adam |
| RMSprop | One buffer per variable (accumulator) |

---

## 2. Saving and Loading Optimizer State

### tf.train.Checkpoint

**Checkpoint** saves and restores optimizer and model together.

```python
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
checkpoint.save("/path/to/ckpt")
# Later:
checkpoint.restore(tf.train.latest_checkpoint("/path/to"))
```

### Model Checkpoint Callback

`ModelCheckpoint` with `save_optimizer_state=True` (default in some APIs) includes optimizer state. Verify your Keras version's behavior.

```python
ckpt = tf.keras.callbacks.ModelCheckpoint(
    "model.keras",
    save_best_only=True
)
model.fit(x_train, y_train, epochs=10, callbacks=[ckpt])
```

### Restoring and Resuming

For exact training continuation:

1. Restore model weights.
2. Restore optimizer state.
3. Restore or recompute the training step counter if using schedules.

```python
model = create_model()
optimizer = tf.keras.optimizers.Adam(0.001)
ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model, step=tf.Variable(0))
ckpt.restore("/path/to/ckpt-1")
model.compile(optimizer=optimizer, loss='mse')
model.fit(x_train, y_train, initial_epoch=5, epochs=10)  # Resume from epoch 5
```

---

## 3. Multi-Optimizer Training

Some architectures use different optimizers for different parts (e.g., GANs, encoder-decoder with different learning dynamics).

### GAN-Style Training

Generator and discriminator often use separate optimizers and update steps.

```python
opt_generator = tf.keras.optimizers.Adam(0.001)
opt_discriminator = tf.keras.optimizers.Adam(0.0002)

@tf.function
def train_step(real_samples, noise):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake = generator(noise, training=True)
        real_out = discriminator(real_samples, training=True)
        fake_out = discriminator(fake, training=True)
        loss_d = disc_loss(real_out, fake_out)
        loss_g = gen_loss(fake_out)

    grad_g = gen_tape.gradient(loss_g, generator.trainable_variables)
    grad_d = disc_tape.gradient(loss_d, discriminator.trainable_variables)
    opt_generator.apply_gradients(zip(grad_g, generator.trainable_variables))
    opt_discriminator.apply_gradients(zip(grad_d, discriminator.trainable_variables))
```

### Splitting Variables by Optimizer

```python
encoder_vars = model.encoder.trainable_variables
decoder_vars = model.decoder.trainable_variables

grads = tape.gradient(loss, model.trainable_variables)
enc_grads = [g for g, v in zip(grads, model.trainable_variables) if v in encoder_vars]
dec_grads = [g for g, v in zip(grads, model.trainable_variables) if v in decoder_vars]

opt_encoder.apply_gradients(zip(enc_grads, encoder_vars))
opt_decoder.apply_gradients(zip(dec_grads, decoder_vars))
```

---

## 4. Custom Optimizers

Subclass `tf.keras.optimizers.Optimizer` and override the appropriate methods.

### Required Methods

- **__init__**: Call `super().__init__()` and use `_set_hyper()` for hyperparameters.
- **_create_slots**: Create optimizer state (e.g., momentum) with `add_slot(var, "slot_name")`.
- **_resource_apply_dense** (or `_resource_apply_sparse`): Implement the update rule.
- **get_config**: Return a dict for serialization.

### Example: Simple SGD with Momentum

```python
class SimpleSGD(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.01, name="SimpleSGD", **kwargs):
        super().__init__(name=name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr = self._get_hyper("learning_rate", var_dtype)
        m = self.get_slot(var, "m")
        m.assign(0.9 * m + grad)
        var.assign_sub(lr * m)

    def get_config(self):
        config = super().get_config()
        config.update({"learning_rate": self._serialize_hyperparameter("learning_rate")})
        return config
```

### Using Custom Optimizer

```python
model.compile(optimizer=SimpleSGD(0.01), loss='mse')
model.fit(x_train, y_train, epochs=5)
```

---

## 5. Optimization Debugging

### Loss NaN or Inf

**Causes**: Exploding gradients, bad learning rate, numerical instability (e.g., log(0)), bad data.

**Checks**:

```python
# Check for NaN/Inf in tensors
has_nan = tf.reduce_any(tf.math.is_nan(tensor))
has_inf = tf.reduce_any(tf.math.is_inf(tensor))

# Check gradients
with tf.GradientTape() as tape:
    loss = model(x, training=True)
grads = tape.gradient(loss, model.trainable_variables)
for i, g in enumerate(grads):
    if g is not None and tf.reduce_any(tf.math.is_nan(g)):
        print(f"NaN in gradient for variable {i}")
```

**Fixes**: Lower learning rate, gradient clipping, check for division by zero, validate inputs.

### Learning Rate Issues

- **Loss not decreasing**: LR may be too small; try warmup or higher initial LR.
- **Loss oscillating**: LR may be too large; decay or reduce LR.
- **Loss spikes**: Sudden LR changes or bad batches; use gradient clipping, gradient accumulation, or smaller LR.

### Callback for NaN Detection

```python
import numpy as np

class NanCheckingCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if logs and np.isnan(logs.get('loss', 0)):
            print(f"NaN detected at batch {batch}")
            self.model.stop_training = True

model.fit(x_train, y_train, callbacks=[NanCheckingCallback()])
```

### Gradient Magnitude Monitoring

```python
# Log gradient norms
with tf.GradientTape() as tape:
    loss = model(x, training=True)
grads = tape.gradient(loss, model.trainable_variables)
total_norm = tf.sqrt(sum(tf.reduce_sum(tf.square(g)) for g in grads if g is not None))
print(f"Gradient norm: {total_norm.numpy()}")
```

### Safe Division

Avoid division by zero in custom losses or layers:

```python
def safe_divide(a, b, default=0.0):
    return tf.where(tf.equal(b, 0), default, a / b)
```

### Summary Checklist

| Issue | Check | Fix |
|-------|-------|-----|
| Loss NaN | Gradients, loss computation | Clip gradients, lower LR, fix math |
| Loss explodes | Gradient norm | global_clipnorm, smaller LR |
| No improvement | LR, data, model | Increase LR, verify data, check architecture |
| Slow convergence | LR schedule | Warmup, cosine decay, LR finder |
