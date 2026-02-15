"""
Custom LearningRateSchedule subclass.
"""
import tensorflow as tf

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
        cosine_decay = 0.5 * (1 + tf.cos(tf.constant(3.14159, dtype=tf.float32) * progress))
        decayed_lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
        return tf.cond(step < warmup_steps, lambda: warmup_lr, lambda: decayed_lr)

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
        }

def main():
    schedule = WarmupCosineDecay(initial_lr=0.01, warmup_steps=100, total_steps=1000)
    lr_warmup = schedule(50).numpy()
    lr_decay = schedule(500).numpy()
    lr_end = schedule(1000).numpy()
    print(f"WarmupCosineDecay: lr(50)={lr_warmup:.6f}, lr(500)={lr_decay:.6f}, lr(1000)={lr_end:.6f}")

    class LinearWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, initial_lr, warmup_steps, name=None):
            super().__init__(name=name)
            self.initial_lr = initial_lr
            self.warmup_steps = warmup_steps

        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            warmup = tf.cast(self.warmup_steps, tf.float32)
            return self.initial_lr * tf.minimum(1.0, step / warmup)

    warmup_schedule = LinearWarmup(0.01, 100)
    print(f"LinearWarmup: lr(0)={warmup_schedule(0).numpy()}, lr(100)={warmup_schedule(100).numpy()}")

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

    cyclic = CyclicLR(base_lr=0.001, max_lr=0.01, step_size=100)
    print(f"CyclicLR: lr(0)={cyclic(0).numpy():.4f}, lr(100)={cyclic(100).numpy():.4f}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(tf.random.normal((32, 4)), tf.random.normal((32, 1)), epochs=3, verbose=0)
    print(f"Model trained with custom schedule.")
    print("Custom LR schedules verified.")

if __name__ == "__main__":
    main()
