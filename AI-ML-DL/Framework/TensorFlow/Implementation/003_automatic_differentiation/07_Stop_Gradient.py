"""
Stop gradient: tf.stop_gradient for detaching from computation graph.
"""
import tensorflow as tf

x = tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x ** 2
    z = tf.stop_gradient(y) + x

grad = tape.gradient(z, x)
print("Stop gradient - detaching y from backward pass:")
print(f"z = stop_gradient(x^2) + x")
print(f"dz/dx = 1 (y contributes 0) = {grad.numpy()}")

x = tf.constant(2.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x * 2
    z = y + tf.stop_gradient(x)

grad = tape.gradient(z, x)
print(f"\nz = 2x + stop_gradient(x)")
print(f"dz/dx = 2 = {grad.numpy()}")

x = tf.constant([1.0, 2.0, 3.0])
with tf.GradientTape() as tape:
    tape.watch(x)
    h = tf.nn.relu(x)
    h_stopped = tf.stop_gradient(h)
    y = h_stopped * 2 + h

grad = tape.gradient(y, x)
print(f"\nx = {x.numpy()}")
print(f"y = stop_grad(relu(x))*2 + relu(x)")
print(f"dy/dx (only second relu flows) = {grad.numpy()}")

w = tf.Variable(1.0)
x = tf.constant(2.0)
with tf.GradientTape() as tape:
    pred = w * tf.stop_gradient(x)
    loss = (pred - 5.0) ** 2

grad = tape.gradient(loss, w)
print(f"\nPrediction with frozen input: pred = w * stop_grad(x)")
print(f"d(loss)/dw = {grad.numpy()}")
