"""
Jacobian and Hessian: tape.jacobian(), tape.batch_jacobian(), Hessian computation.
"""
import tensorflow as tf

x = tf.constant([1.0, 2.0])
with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.stack([x[0] ** 2, x[0] * x[1], x[1] ** 2])

jacobian = tape.jacobian(y, x)
print("Jacobian: dy/dx")
print(f"y = [x0^2, x0*x1, x1^2]")
print(f"Jacobian shape: {jacobian.shape}")
print(f"Jacobian =\n{jacobian.numpy()}")

x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x ** 2

batch_jac = tape.batch_jacobian(y, x)
print(f"\nBatch Jacobian: batch of dy/dx")
print(f"x shape: {x.shape}, y shape: {y.shape}")
print(f"batch_jacobian shape: {batch_jac.shape}")

x = tf.constant(2.0)
with tf.GradientTape() as tape2:
    tape2.watch(x)
    with tf.GradientTape() as tape1:
        tape1.watch(x)
        y = x ** 3 + x ** 2
    grad = tape1.gradient(y, x)
hessian = tape2.gradient(grad, x)

print(f"\nHessian (scalar): y = x^3 + x^2")
print(f"dy/dx = 3x^2 + 2x = {grad.numpy()}")
print(f"d2y/dx2 = 6x + 2 = {hessian.numpy()}")

x = tf.constant([1.0, 2.0])
with tf.GradientTape() as tape2:
    tape2.watch(x)
    with tf.GradientTape() as tape1:
        tape1.watch(x)
        y = tf.reduce_sum(x ** 3)
    grad = tape1.gradient(y, x)
hessian_diag = tape2.gradient(grad, x)

print(f"\nHessian diagonal: y = sum(x^3)")
print(f"grad = 3x^2 = {grad.numpy()}")
print(f"Hessian diag = 6x = {hessian_diag.numpy()}")
