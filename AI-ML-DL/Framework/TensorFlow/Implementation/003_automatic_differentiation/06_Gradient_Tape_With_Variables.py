"""
GradientTape with Variables: auto-watched trainable variables.
"""
import tensorflow as tf

w = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
b = tf.Variable([0.0, 0.0])
x = tf.constant([[1.0, 1.0], [2.0, 2.0]])

with tf.GradientTape() as tape:
    y = tf.matmul(x, w) + b
    loss = tf.reduce_mean(y ** 2)

grads = tape.gradient(loss, [w, b])
print("Variables auto-watched (trainable=True):")
print(f"w = \n{w.numpy()}")
print(f"d(loss)/dw = \n{grads[0].numpy()}")
print(f"d(loss)/db = {grads[1].numpy()}")

w = tf.Variable(1.0, trainable=False)
x = tf.constant(2.0)
with tf.GradientTape() as tape:
    tape.watch(w)
    y = w * x

grad = tape.gradient(y, w)
print(f"\nNon-trainable variable: must watch manually")
print(f"grad = {grad.numpy()}")

w = tf.Variable([1.0, 2.0, 3.0])
with tf.GradientTape() as tape:
    loss = tf.reduce_sum(w ** 2)

grad = tape.gradient(loss, w)
print(f"\nw = {w.numpy()}")
print(f"loss = sum(w^2) = {loss.numpy()}")
print(f"d(loss)/dw = {grad.numpy()}")

w.assign(tf.constant([0.5, 1.0, 1.5]))
with tf.GradientTape() as tape:
    loss = tf.reduce_sum(w ** 2)

grad = tape.gradient(loss, w)
print(f"\nAfter assign: w = {w.numpy()}")
print(f"d(loss)/dw = {grad.numpy()}")
