"""
Persistent GradientTape: computing multiple gradients from one tape.
"""
import tensorflow as tf

x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    y = x ** 2
    z = x ** 3

grad_y = tape.gradient(y, x)
grad_z = tape.gradient(z, x)
print("Persistent Tape - Multiple gradients from one tape:")
print(f"dy/dx = {grad_y.numpy()}, dz/dx = {grad_z.numpy()}")

del tape

w = tf.constant([1.0, 2.0, 3.0])
with tf.GradientTape(persistent=True) as tape:
    tape.watch(w)
    a = tf.reduce_sum(w)
    b = tf.reduce_prod(w)
    c = tf.reduce_mean(w ** 2)

grad_a = tape.gradient(a, w)
grad_b = tape.gradient(b, w)
grad_c = tape.gradient(c, w)
print(f"\nw = {w.numpy()}")
print(f"d(sum(w))/dw = {grad_a.numpy()}")
print(f"d(prod(w))/dw = {grad_b.numpy()}")
print(f"d(mean(w^2))/dw = {grad_c.numpy()}")

del tape

x = tf.constant(2.0)
with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    f = x ** 2 + tf.sin(x)

df_dx = tape.gradient(f, x)
d2f_dx2 = tape.gradient(df_dx, x)
print(f"\nf = x^2 + sin(x) at x=2")
print(f"df/dx = {df_dx.numpy():.4f}")
print(f"d2f/dx2 = {d2f_dx2.numpy():.4f}")

del tape
