"""
Performance tips for gradient computation.
"""
import tensorflow as tf
import time

@tf.function
def optimized_gradient_computation(x, w):
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w)
        loss = tf.reduce_mean(y ** 2)
    return tape.gradient(loss, w)

x = tf.random.normal([1000, 100])
w = tf.Variable(tf.random.normal([100, 50]))

start = time.perf_counter()
for _ in range(100):
    grad = optimized_gradient_computation(x, w)
elapsed = time.perf_counter() - start
print(f"tf.function + GradientTape (100 runs): {elapsed*1000:.2f} ms")

def eager_gradient(x, w):
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w)
        loss = tf.reduce_mean(y ** 2)
    return tape.gradient(loss, w)

start = time.perf_counter()
for _ in range(100):
    grad = eager_gradient(x, w)
elapsed = time.perf_counter() - start
print(f"Eager GradientTape (100 runs): {elapsed*1000:.2f} ms")

@tf.function
def batch_gradient(x_batch, w):
    with tf.GradientTape() as tape:
        y = tf.matmul(x_batch, w)
        loss = tf.reduce_mean(y ** 2)
    return tape.gradient(loss, w)

x_small = tf.random.normal([10, 100])
x_large = tf.random.normal([1000, 100])

start = time.perf_counter()
for _ in range(100):
    _ = batch_gradient(x_large, w)
large_time = (time.perf_counter() - start) * 1000

start = time.perf_counter()
for _ in range(100):
    _ = batch_gradient(x_small, w)
small_time = (time.perf_counter() - start) * 1000

print(f"\nBatch size impact: large {large_time:.2f} ms vs small {small_time:.2f} ms")
