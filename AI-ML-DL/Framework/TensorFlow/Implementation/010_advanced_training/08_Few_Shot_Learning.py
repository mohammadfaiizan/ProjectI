"""
Prototypical networks, support/query sets.
"""
import tensorflow as tf
import numpy as np

def main():
    print("=" * 50)
    print("Prototypical Networks - Few-Shot Learning")
    print("=" * 50)

    n_way, k_shot, q_query = 5, 3, 2
    dim = 16
    encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(dim,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8)
    ])

    support = tf.random.normal((n_way * k_shot, dim))
    support_labels = tf.repeat(tf.range(n_way), k_shot)
    query = tf.random.normal((n_way * q_query, dim))
    query_labels = tf.repeat(tf.range(n_way), q_query)

    support_emb = encoder(support, training=True)
    prototypes = []
    for c in range(n_way):
        mask = support_labels == c
        proto = tf.reduce_mean(tf.boolean_mask(support_emb, mask), axis=0)
        prototypes.append(proto)
    prototypes = tf.stack(prototypes)

    query_emb = encoder(query, training=True)
    dists = tf.reduce_sum(
        tf.square(query_emb[:, tf.newaxis, :] - prototypes[tf.newaxis, :, :]),
        axis=2
    )
    logits = -dists
    pred = tf.argmax(logits, axis=1)

    loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(query_labels, logits)
    )
    optimizer = tf.keras.optimizers.Adam(0.001)
    grads = tf.gradients(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(grads, encoder.trainable_variables))
    acc = np.mean(pred.numpy() == query_labels.numpy())
    print(f"Few-shot accuracy: {acc:.2%}, loss: {loss.numpy():.4f}")
    print("Prototypical network complete.")

if __name__ == "__main__":
    main()
