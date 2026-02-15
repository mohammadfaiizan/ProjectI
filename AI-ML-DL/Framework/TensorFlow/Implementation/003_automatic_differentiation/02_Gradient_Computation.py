"""
Gradient computation: tape.gradient(), multiple targets, sources, unconnected_gradients.
"""
import tensorflow as tf

x = tf.constant(2.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x ** 2
    z = x ** 3

grad_y = tape.gradient(y, x)
grad_z = tape.gradient(z, x)
print("Single target gradients:")
print(f"dy/dx = {grad_y.numpy()}, dz/dx = {grad_z.numpy()}")

x = tf.constant(2.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x ** 2
    z = x ** 3
    total = y + z

grad = tape.gradient(total, x)
print(f"\nMultiple outputs summed: d(y+z)/dx = {grad.numpy()}")

w = tf.constant([1.0, 2.0])
b = tf.constant(1.0)
with tf.GradientTape() as tape:
    tape.watch([w, b])
    loss = tf.reduce_sum(w) + b

grads = tape.gradient(loss, [w, b])
print(f"\nMultiple sources: d(loss)/dw = {grads[0].numpy()}, d(loss)/db = {grads[1].numpy()}")

x = tf.constant(1.0)
y = tf.constant(2.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    z = x * 2
    w = y * 3

grad_none = tape.gradient(z, y)
grad_zero = tape.gradient(z, y, unconnected_gradients=tf.UnconnectedGradients.ZERO)
print(f"\nUnconnected: grad_none = {grad_none}, grad_zero = {grad_zero.numpy()}")
