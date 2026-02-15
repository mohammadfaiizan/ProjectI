"""
Higher-order gradients: nested GradientTape for second-order derivatives.
"""
import tensorflow as tf

x = tf.constant(2.0)
with tf.GradientTape() as tape2:
    tape2.watch(x)
    with tf.GradientTape() as tape1:
        tape1.watch(x)
        y = x ** 3
    dy_dx = tape1.gradient(y, x)
d2y_dx2 = tape2.gradient(dy_dx, x)

print("Nested GradientTape - Second-order derivative:")
print(f"y = x^3, dy/dx = 3x^2 = {dy_dx.numpy()}")
print(f"d2y/dx2 = 6x = {d2y_dx2.numpy()}")

x = tf.constant(1.0)
with tf.GradientTape() as tape2:
    tape2.watch(x)
    with tf.GradientTape() as tape1:
        tape1.watch(x)
        y = tf.sin(x)
    dy_dx = tape1.gradient(y, x)
d2y_dx2 = tape2.gradient(dy_dx, x)

print(f"\ny = sin(x), dy/dx = cos(x) = {dy_dx.numpy():.4f}")
print(f"d2y/dx2 = -sin(x) = {d2y_dx2.numpy():.4f}")

x = tf.constant([1.0, 2.0])
with tf.GradientTape() as tape2:
    tape2.watch(x)
    with tf.GradientTape() as tape1:
        tape1.watch(x)
        y = tf.reduce_sum(x ** 2)
    grad = tape1.gradient(y, x)
hessian_diag = tape2.gradient(grad, x)

print(f"\ny = sum(x^2), grad = 2x = {grad.numpy()}")
print(f"Hessian diagonal = 2 = {hessian_diag.numpy()}")
