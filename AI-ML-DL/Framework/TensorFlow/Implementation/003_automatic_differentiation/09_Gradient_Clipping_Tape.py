"""
Gradient clipping in custom training: clip_by_norm, clip_by_value, clip_by_global_norm.
"""
import tensorflow as tf

w = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
x = tf.constant([[1.0, 1.0], [2.0, 2.0]])

with tf.GradientTape() as tape:
    y = tf.matmul(x, w)
    loss = tf.reduce_mean(y ** 2)

grads = tape.gradient(loss, w)
grads_clipped = tf.clip_by_norm(grads, 1.0)
print("clip_by_norm (max_norm=1.0):")
print(f"Original norm: {tf.norm(grads).numpy():.4f}")
print(f"Clipped norm: {tf.norm(grads_clipped).numpy():.4f}")

grads_clipped_val = tf.clip_by_value(grads, -0.5, 0.5)
print(f"\nclip_by_value (-0.5, 0.5):")
print(f"Original: {grads.numpy()}")
print(f"Clipped: {grads_clipped_val.numpy()}")

w1 = tf.Variable([[1.0, 2.0]])
w2 = tf.Variable([[3.0, 4.0]])
x = tf.constant([[1.0, 1.0]])

with tf.GradientTape() as tape:
    h = tf.matmul(x, w1)
    y = tf.matmul(h, w2)
    loss = tf.reduce_mean(y ** 2)

grads = tape.gradient(loss, [w1, w2])
grads_clipped_global, global_norm = tf.clip_by_global_norm(grads, 2.0)
print(f"\nclip_by_global_norm (max_norm=2.0):")
print(f"Global norm before: {tf.sqrt(sum(tf.reduce_sum(g**2) for g in grads)).numpy():.4f}")
print(f"Global norm after: {global_norm.numpy():.4f}")

optimizer = tf.keras.optimizers.SGD(0.01)
w = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
x = tf.constant([[10.0, 10.0], [20.0, 20.0]])

with tf.GradientTape() as tape:
    y = tf.matmul(x, w)
    loss = tf.reduce_mean(y ** 2)

grads = tape.gradient(loss, w)
grads, _ = tf.clip_by_global_norm([grads], 1.0)
optimizer.apply_gradients(zip(grads, [w]))
print(f"\nTraining step with gradient clipping applied")
