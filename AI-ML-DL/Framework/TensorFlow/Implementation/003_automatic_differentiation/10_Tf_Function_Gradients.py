"""
GradientTape inside @tf.function: tracing behavior.
"""
import tensorflow as tf

@tf.function
def compute_gradient(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = x ** 2 + tf.sin(x)
    return tape.gradient(y, x)

x = tf.constant(2.0)
grad = compute_gradient(x)
print("GradientTape inside @tf.function:")
print(f"x = {x.numpy()}, dy/dx = {grad.numpy():.4f}")

@tf.function
def train_step(x, w, optimizer):
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w)
        loss = tf.reduce_mean(y ** 2)
    grads = tape.gradient(loss, w)
    optimizer.apply_gradients(zip(grads, [w]))
    return loss

w = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
x = tf.constant([[1.0, 1.0], [2.0, 2.0]])
optimizer = tf.keras.optimizers.SGD(0.01)

loss = train_step(x, w, optimizer)
print(f"\nTrain step loss: {loss.numpy():.4f}")

@tf.function
def nested_tape_gradient(x):
    with tf.GradientTape() as tape2:
        tape2.watch(x)
        with tf.GradientTape() as tape1:
            tape1.watch(x)
            y = x ** 3
        dy_dx = tape1.gradient(y, x)
    d2y_dx2 = tape2.gradient(dy_dx, x)
    return dy_dx, d2y_dx2

x = tf.constant(2.0)
dy, d2y = nested_tape_gradient(x)
print(f"\nNested tape in tf.function: y = x^3")
print(f"dy/dx = {dy.numpy()}, d2y/dx2 = {d2y.numpy()}")
