"""
Custom gradients: @tf.custom_gradient decorator for forward and backward.
"""
import tensorflow as tf

@tf.custom_gradient
def custom_square(x):
    def grad(dy):
        return 2.0 * x * dy
    return x * x, grad

x = tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = custom_square(x)

grad = tape.gradient(y, x)
print("Custom gradient - square with explicit backward:")
print(f"y = x^2 = {y.numpy()}, dy/dx = {grad.numpy()}")

@tf.custom_gradient
def safe_log(x):
    def grad(dy):
        return dy / tf.maximum(x, 1e-7)
    return tf.math.log(tf.maximum(x, 1e-7)), grad

x = tf.constant(2.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = safe_log(x)

grad = tape.gradient(y, x)
print(f"\nSafe log: y = {y.numpy():.4f}, dy/dx = {grad.numpy():.4f}")

@tf.custom_gradient
def scaled_relu(x, scale=2.0):
    def grad(dy):
        return dy * scale * tf.cast(x > 0, x.dtype)
    return scale * tf.nn.relu(x), grad

x = tf.constant([-1.0, 2.0, 0.0])
with tf.GradientTape() as tape:
    tape.watch(x)
    y = scaled_relu(x)

grad = tape.gradient(y, x)
print(f"\nScaled ReLU: x = {x.numpy()}, y = {y.numpy()}")
print(f"grad = {grad.numpy()}")
