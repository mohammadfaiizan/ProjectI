"""
Analyzing gradient magnitudes per layer.
"""
import tensorflow as tf

w1 = tf.Variable(tf.random.normal([2, 4]) * 0.1)
w2 = tf.Variable(tf.random.normal([4, 2]) * 0.1)
x = tf.constant([[1.0, 2.0], [3.0, 4.0]])

with tf.GradientTape() as tape:
    h = tf.matmul(x, w1)
    h = tf.nn.relu(h)
    y = tf.matmul(h, w2)
    loss = tf.reduce_mean(y ** 2)

grads = tape.gradient(loss, [w1, w2])
grad_norms = [tf.norm(g).numpy() for g in grads]
print("Gradient magnitudes per layer:")
print(f"Layer 1 (w1) grad norm: {grad_norms[0]:.6f}")
print(f"Layer 2 (w2) grad norm: {grad_norms[1]:.6f}")

for i, (g, w) in enumerate(zip(grads, [w1, w2])):
    ratio = tf.norm(g) / (tf.norm(w) + 1e-8)
    print(f"Layer {i+1} grad/param ratio: {ratio.numpy():.6f}")

layers = [w1, w2]
grad_stats = []
for g, w in zip(grads, layers):
    mean_g = tf.reduce_mean(tf.abs(g)).numpy()
    max_g = tf.reduce_max(tf.abs(g)).numpy()
    grad_stats.append((mean_g, max_g))

print(f"\nGradient statistics:")
for i, (mean_g, max_g) in enumerate(grad_stats):
    print(f"Layer {i+1}: mean |grad| = {mean_g:.6f}, max |grad| = {max_g:.6f}")
