"""
Debugging NaN gradients and tf.debugging.check_numerics.
"""
import tensorflow as tf

def safe_gradient_check(loss, sources, message="gradient"):
    with tf.GradientTape() as tape:
        tape.watch(sources)
        loss_val = loss
    grads = tape.gradient(loss_val, sources)
    for i, g in enumerate(grads):
        if g is not None:
            checked = tf.debugging.check_numerics(g, f"{message}_{i}")
    return grads

x = tf.constant([1.0, 2.0, 3.0])
with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.reduce_sum(x ** 2)

grads = tape.gradient(y, x)
grads_checked = [tf.debugging.check_numerics(g, "grad_check") for g in grads]
print("check_numerics on valid gradients:")
print(f"grads = {[g.numpy() for g in grads_checked]}")

x = tf.constant(-1.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.math.log(tf.nn.relu(x) + 1e-10)

try:
    grad = tape.gradient(y, x)
    grad_checked = tf.debugging.check_numerics(grad, "log_grad")
    print(f"\nlog(relu(x)) gradient: {grad_checked.numpy()}")
except tf.errors.InvalidArgumentError as e:
    print(f"\nCaught invalid gradient (expected for NaN/Inf): check_numerics works")

x = tf.constant([1.0, 0.0, -1.0])
with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.sqrt(tf.nn.relu(x) + 1e-7)

grads = tape.gradient(y, x)
print(f"\nSafe sqrt gradient: x = {x.numpy()}")
print(f"grads = {grads.numpy()}")

print("\nUse tf.debugging.enable_check_numerics() for global NaN/Inf checks")
