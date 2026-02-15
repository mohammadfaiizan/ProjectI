"""
Advanced gradient patterns: gradient penalty, gradient reversal.
"""
import tensorflow as tf

def gradient_penalty(model, real, fake):
    batch_size = tf.shape(real)[0]
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolates = alpha * real + (1 - alpha) * fake
    with tf.GradientTape() as tape:
        tape.watch(interpolates)
        pred = model(interpolates)
    grads = tape.gradient(pred, interpolates)
    grad_norms = tf.sqrt(tf.reduce_sum(grads ** 2, axis=[1, 2, 3]) + 1e-8)
    penalty = tf.reduce_mean((grad_norms - 1.0) ** 2)
    return penalty

class SimpleDiscriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1)
    def call(self, x):
        return self.dense(tf.reshape(x, [tf.shape(x)[0], -1]))

model = SimpleDiscriminator()
real = tf.random.normal([4, 8, 8, 1])
fake = tf.random.normal([4, 8, 8, 1])
penalty = gradient_penalty(model, real, fake)
print("Gradient penalty (WGAN-GP style):")
print(f"Penalty value: {penalty.numpy():.4f}")

@tf.custom_gradient
def gradient_reversal(x, scale=1.0):
    def grad(dy):
        return -scale * dy
    return x, grad

x = tf.constant([1.0, 2.0, 3.0])
w = tf.Variable([1.0, 1.0, 1.0])
with tf.GradientTape() as tape:
    x_rev = gradient_reversal(x)
    y = tf.reduce_sum(x_rev * w)

grad = tape.gradient(y, w)
print(f"\nGradient reversal: y = sum(rev(x) * w)")
print(f"Without reversal grad would be x = {x.numpy()}")
print(f"With reversal grad = -x = {grad.numpy()}")

def gradient_accumulation_penalty(model, x, n_steps=4):
    total_grad_norm = 0.0
    for _ in range(n_steps):
        with tf.GradientTape() as tape:
            y = model(x)
            loss = tf.reduce_mean(y ** 2)
        grads = tape.gradient(loss, model.trainable_variables)
        total_grad_norm += sum(tf.norm(g) for g in grads if g is not None)
    return total_grad_norm / n_steps

model = tf.keras.Sequential([tf.keras.layers.Dense(4, input_shape=(2,))])
x = tf.random.normal([8, 2])
avg_norm = gradient_accumulation_penalty(model, x)
print(f"\nAverage gradient norm over {4} steps: {avg_norm.numpy():.4f}")
