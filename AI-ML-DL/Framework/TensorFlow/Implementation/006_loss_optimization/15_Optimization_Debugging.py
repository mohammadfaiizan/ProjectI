"""
Debugging optimization: loss NaN, learning rate issues.
"""
import tensorflow as tf
import numpy as np

def main():
    def check_nan_inf(tensor, name="tensor"):
        has_nan = tf.reduce_any(tf.math.is_nan(tensor))
        has_inf = tf.reduce_any(tf.math.is_inf(tensor))
        print(f"{name}: has_nan={has_nan.numpy()}, has_inf={has_inf.numpy()}")

    x = tf.constant([1.0, float('nan'), 3.0])
    check_nan_inf(x, "x")

    grad = tf.constant([1e10, -1e10, 0.0])
    grad_clipped = tf.clip_by_value(grad, -1.0, 1.0)
    print(f"Gradient clipping: before norm={tf.norm(grad).numpy():.2e}, after={tf.norm(grad_clipped).numpy():.2f}")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(8,)),
        tf.keras.layers.Dense(1)
    ])
    opt = tf.keras.optimizers.Adam(0.001)
    model.compile(optimizer=opt, loss='mse')

    class NanCheckingCallback(tf.keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            if logs and np.isnan(logs.get('loss', 0)):
                print(f"NaN detected at batch {batch}")
                self.model.stop_training = True

    x_train = tf.random.normal((32, 8))
    y_train = tf.random.normal((32, 1))
    model.fit(x_train, y_train, epochs=2, callbacks=[NanCheckingCallback()], verbose=0)
    print(f"Training with NaN callback completed.")

    lr = 0.001
    for step in [0, 100, 500]:
        schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.1, 100, 0.96)
        lr_val = schedule(step).numpy()
        print(f"LR at step {step}: {lr_val:.6f}")

    opt_sgd = tf.keras.optimizers.SGD(0.01, momentum=0.9)
    var = tf.Variable(1.0)
    grads_before = []
    for _ in range(3):
        with tf.GradientTape() as tape:
            loss = (var - 0.5) ** 2
        grad = tape.gradient(loss, var)
        grads_before.append(grad.numpy())
        opt_sgd.apply_gradients([(grad, var)])
    print(f"Gradient magnitude over steps: {[f'{g:.4f}' for g in grads_before]}")

    def safe_divide(a, b, default=0.0):
        return tf.where(tf.equal(b, 0), default, a / b)

    a = tf.constant([1.0, 2.0, 0.0])
    b = tf.constant([2.0, 0.0, 1.0])
    result = safe_divide(a, b)
    print(f"Safe divide result: {result.numpy()}")
    print("Optimization debugging verified.")

if __name__ == "__main__":
    main()
