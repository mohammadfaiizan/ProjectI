"""
SimCLR-style, NT-Xent loss, projection head.
"""
import tensorflow as tf
import numpy as np

def main():
    print("=" * 50)
    print("Contrastive Learning - SimCLR Style")
    print("=" * 50)

    encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(16,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8)
    ])
    projection = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(4)
    ])

    def nt_xent_loss(z_i, z_j, temperature=0.5):
        z = tf.concat([z_i, z_j], axis=0)
        z = tf.math.l2_normalize(z, axis=1)
        sim = tf.matmul(z, z, transpose_b=True) / temperature
        n = tf.shape(z_i)[0]
        mask = tf.eye(2 * n)
        sim = sim - 1e9 * mask
        labels = tf.concat([tf.range(n, 2*n), tf.range(n)], axis=0)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, sim, from_logits=True
        )
        return tf.reduce_mean(loss)

    x = tf.random.normal((32, 16))
    aug1 = x + tf.random.normal(tf.shape(x)) * 0.1
    aug2 = x + tf.random.normal(tf.shape(x)) * 0.1

    optimizer = tf.keras.optimizers.Adam(0.001)
    for _ in range(5):
        with tf.GradientTape() as tape:
            h1 = encoder(aug1, training=True)
            h2 = encoder(aug2, training=True)
            z1 = projection(h1, training=True)
            z2 = projection(h2, training=True)
            loss = nt_xent_loss(z1, z2, temperature=0.5)
        vars = encoder.trainable_variables + projection.trainable_variables
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))
    print(f"NT-Xent loss: {loss.numpy():.4f}")

    emb = encoder(x[:8], training=False)
    print(f"Embedding shape: {emb.shape}")
    print("Contrastive learning complete.")

if __name__ == "__main__":
    main()
