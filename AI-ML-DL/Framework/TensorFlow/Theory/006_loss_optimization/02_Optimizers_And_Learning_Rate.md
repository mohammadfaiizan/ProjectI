# Optimizers and Learning Rate

## Table of Contents

1. [SGD Optimizer](#1-sgd-optimizer)
2. [Adam Family Optimizers](#2-adam-family-optimizers)
3. [Advanced Optimizers](#3-advanced-optimizers)
4. [Built-in Learning Rate Schedules](#4-built-in-learning-rate-schedules)
5. [Custom Learning Rate Schedules](#5-custom-learning-rate-schedules)

---

## 1. SGD Optimizer

**Stochastic Gradient Descent (SGD)** is the foundational optimizer. Updates: `w = w - lr * gradient`.

### Basic SGD

```python
sgd = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=sgd, loss='mse')
```

### Momentum

**Momentum** accumulates past gradients to smooth updates: `v = momentum * v + grad`, `w = w - lr * v`. Reduces oscillations and accelerates in consistent directions.

```python
sgd_momentum = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
```

### Nesterov Accelerated Gradient

**Nesterov** uses a "look-ahead" gradient: compute gradient at `w + momentum * v` instead of `w`. Often converges faster than standard momentum.

```python
sgd_nesterov = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
```

### SGD Parameters

| Parameter | Default | Description |
|-----------|---------|--------------|
| learning_rate | 0.01 | Step size |
| momentum | 0.0 | Momentum coefficient |
| nesterov | False | Use Nesterov momentum |

---

## 2. Adam Family Optimizers

### Adam

**Adam** (Adaptive Moment Estimation) combines momentum and RMSprop. Maintains first and second moment estimates; bias-corrected updates.

```python
adam = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,    # momentum decay
    beta_2=0.999,  # second moment decay
    epsilon=1e-7
)
```

### AdamW

**AdamW** decouples weight decay from gradient updates. Applies **weight_decay** directly to weights rather than through the loss. Better generalization in many settings.

```python
adamw = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01)
```

### Adamax

**Adamax** uses infinity norm for the second moment instead of L2. Can be more stable in some cases.

```python
adamax = tf.keras.optimizers.Adamax(learning_rate=0.002)
```

---

## 3. Advanced Optimizers

### RMSprop

**RMSprop** adapts learning rate per parameter using a moving average of squared gradients. Good for non-stationary objectives.

```python
rmsprop = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
```

### Adagrad

**Adagrad** accumulates squared gradients; learning rate decreases for frequently updated parameters. Can cause premature decay.

```python
adagrad = tf.keras.optimizers.Adagrad(learning_rate=0.01, initial_accumulator_value=0.1)
```

### Adadelta

**Adadelta** extends Adagrad with a window of gradients and removes the need for a learning rate. Uses **rho** for decay.

```python
adadelta = tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
```

### Ftrl

**Ftrl** (Follow-The-Regularized-Leader) is designed for sparse features. Supports L1 and L2 regularization.

```python
ftrl = tf.keras.optimizers.Ftrl(
    learning_rate=0.1,
    l1_regularization_strength=0.01,
    l2_regularization_strength=0.01
)
```

### Nadam

**Nadam** combines Adam with Nesterov momentum. Often faster convergence than Adam.

```python
nadam = tf.keras.optimizers.Nadam(learning_rate=0.001)
```

### Optimizer Comparison

| Optimizer | Memory | Use Case |
|-----------|--------|----------|
| SGD | Low | Fine-tuning, simple tasks |
| Adam | Medium | Default for most tasks |
| AdamW | Medium | Better generalization |
| RMSprop | Medium | RNNs, non-stationary |
| Adagrad | High | Sparse features |
| Ftrl | Medium | Sparse linear models |

---

## 4. Built-in Learning Rate Schedules

### ExponentialDecay

**ExponentialDecay** multiplies learning rate by **decay_rate** every **decay_steps**: `lr = initial_lr * decay_rate^(step/decay_steps)`.

```python
schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.1,
    decay_steps=1000,
    decay_rate=0.96
)
optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)
```

### CosineDecay

**CosineDecay** smoothly decreases learning rate following a cosine curve from initial to zero (or **alpha**).

```python
schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.1,
    decay_steps=1000
)
```

### PiecewiseConstantDecay

**PiecewiseConstantDecay** holds constant learning rates between **boundaries**, changing at specified steps.

```python
schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[100, 500, 1000],
    values=[0.1, 0.05, 0.01, 0.001]
)
```

### PolynomialDecay

**PolynomialDecay** decays with a polynomial of given **power** from initial to **end_learning_rate**.

```python
schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=0.1,
    decay_steps=1000,
    end_learning_rate=0.001,
    power=2.0
)
```

### CosineDecayRestarts

**CosineDecayRestarts** restarts the cosine decay periodically. **t_mul** controls how decay_steps grows each restart.

```python
schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=0.1,
    first_decay_steps=100,
    t_mul=2.0
)
```

---

## 5. Custom Learning Rate Schedules

Subclass `tf.keras.optimizers.schedules.LearningRateSchedule` and implement `__call__(self, step)`.

### Warmup + Cosine Decay

```python
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, total_steps, min_lr=1e-6, name=None):
        super().__init__(name=name)
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)
        warmup_lr = self.initial_lr * step / warmup_steps
        progress = (step - warmup_steps) / tf.maximum(total_steps - warmup_steps, 1.0)
        progress = tf.minimum(progress, 1.0)
        cosine_decay = 0.5 * (1 + tf.cos(3.14159 * progress))
        decayed_lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
        return tf.cond(step < warmup_steps, lambda: warmup_lr, lambda: decayed_lr)

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
        }
```

### Cyclic Learning Rate

```python
class CyclicLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, max_lr, step_size, name=None):
        super().__init__(name=name)
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        cycle = tf.floor(1 + step / (2 * self.step_size))
        x = tf.abs(step / self.step_size - 2 * cycle + 1)
        return self.base_lr + (self.max_lr - self.base_lr) * tf.maximum(0.0, 1 - x)
```

### Using Custom Schedules

```python
schedule = WarmupCosineDecay(initial_lr=0.01, warmup_steps=100, total_steps=1000)
optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)
model.compile(optimizer=optimizer, loss='mse')
```

The optimizer passes the current step (e.g., `optimizer.iterations`) to the schedule automatically during training.
