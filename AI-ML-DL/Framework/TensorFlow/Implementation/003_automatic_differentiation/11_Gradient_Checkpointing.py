"""
Gradient checkpointing: trade compute for memory in long chains.
"""
import tensorflow as tf

def chain_forward(x, n_layers=5):
    w = tf.Variable(tf.random.normal((4, 4)) * 0.1)
    y = x
    for _ in range(n_layers):
        y = tf.matmul(y, w) + 1.0
    return tf.reduce_sum(y)

def main():
    x = tf.constant([[1.0, 2.0, 3.0, 4.0]])
    with tf.GradientTape() as tape:
        tape.watch(x)
        out = chain_forward(x)
    grad = tape.gradient(out, x)
    print(f"Chain output: {out.numpy():.4f}")
    print(f"Gradient shape: {grad.shape}")

    def checkpointed_chain(x, n=4):
        w = tf.Variable(tf.random.normal((4, 4)) * 0.1)
        segments = []
        y = x
        for i in range(n):
            with tf.GradientTape() as t:
                t.watch(y)
                z = tf.matmul(y, w) + 1.0
            segments.append((y, z, t))
            y = z
        return tf.reduce_sum(y), segments

    x2 = tf.constant([[1.0, 2.0, 3.0, 4.0]])
    with tf.GradientTape() as tape:
        tape.watch(x2)
        out2, segs = checkpointed_chain(x2)
    grad2 = tape.gradient(out2, x2)
    print(f"\nCheckpointed chain output: {out2.numpy():.4f}")
    print(f"Segments: {len(segs)}")

    @tf.function
    def train_with_checkpoint(x, w, opt):
        with tf.GradientTape() as tape:
            y = tf.nn.relu(tf.matmul(x, w))
            loss = tf.reduce_mean(y ** 2)
        grads = tape.gradient(loss, w)
        opt.apply_gradients(zip([grads], [w]))
        return loss

    w = tf.Variable(tf.random.normal((4, 4)) * 0.1)
    opt = tf.keras.optimizers.SGD(0.01)
    loss = train_with_checkpoint(x2, w, opt)
    print(f"\nTrain step loss: {loss.numpy():.4f}")
    print("Gradient checkpointing demo complete.")

if __name__ == "__main__":
    main()
