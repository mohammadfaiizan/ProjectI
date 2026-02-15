"""
GradientTape basics: watching tensors and computing gradients.
"""
import tensorflow as tf

x = tf.constant(3.0)
y = tf.constant(4.0)

with tf.GradientTape() as tape:
    tape.watch(x)
    tape.watch(y)
    z = x * x + 2 * y

grad_x = tape.gradient(z, x)
grad_y = tape.gradient(z, y)

print("GradientTape Basics")
print("=" * 40)
print(f"x = {x.numpy()}, y = {y.numpy()}")
print(f"z = x^2 + 2y = {z.numpy()}")
print(f"dz/dx = 2x = {grad_x.numpy()}")
print(f"dz/dy = 2 = {grad_y.numpy()}")

w = tf.constant([1.0, 2.0, 3.0])
with tf.GradientTape() as tape:
    tape.watch(w)
    loss = tf.reduce_sum(w * w)

grad_w = tape.gradient(loss, w)
print(f"\nw = {w.numpy()}")
print(f"loss = sum(w^2) = {loss.numpy()}")
print(f"d(loss)/dw = 2w = {grad_w.numpy()}")

a = tf.constant(2.0)
with tf.GradientTape() as tape:
    tape.watch(a)
    b = a ** 3
    c = tf.sin(b)

grad_a = tape.gradient(c, a)
print(f"\na = {a.numpy()}")
print(f"c = sin(a^3) = {c.numpy():.4f}")
print(f"dc/da = cos(a^3) * 3a^2 = {grad_a.numpy():.4f}")
