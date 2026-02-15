"""
Classification losses: CategoricalCrossentropy, SparseCategoricalCrossentropy, BinaryCrossentropy, from_logits.
"""
import tensorflow as tf

def main():
    y_true_cat = tf.constant([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=tf.float32)
    y_pred_prob = tf.constant([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.2, 0.2, 0.6]], dtype=tf.float32)
    y_pred_logits = tf.constant([[0.5, 2.0, 0.3], [2.5, -1.0, -0.5], [-0.5, -0.5, 1.0]], dtype=tf.float32)

    cce = tf.keras.losses.CategoricalCrossentropy()
    loss_cce = cce(y_true_cat, y_pred_prob)
    print(f"CategoricalCrossentropy (from probs): {loss_cce.numpy():.4f}")

    cce_logits = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    loss_cce_logits = cce_logits(y_true_cat, y_pred_logits)
    print(f"CategoricalCrossentropy (from_logits): {loss_cce_logits.numpy():.4f}")

    y_sparse = tf.constant([1, 0, 2])
    scce = tf.keras.losses.SparseCategoricalCrossentropy()
    loss_scce = scce(y_sparse, y_pred_prob)
    print(f"SparseCategoricalCrossentropy: {loss_scce.numpy():.4f}")

    scce_logits = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_scce_logits = scce_logits(y_sparse, y_pred_logits)
    print(f"SparseCategoricalCrossentropy (from_logits): {loss_scce_logits.numpy():.4f}")

    y_bin_true = tf.constant([[0, 1], [1, 0], [1, 1]], dtype=tf.float32)
    y_bin_pred = tf.constant([[0.1, 0.9], [0.85, 0.15], [0.7, 0.8]], dtype=tf.float32)
    bce = tf.keras.losses.BinaryCrossentropy()
    loss_bce = bce(y_bin_true, y_bin_pred)
    print(f"BinaryCrossentropy: {loss_bce.numpy():.4f}")

    y_bin_logits = tf.constant([[-2.0, 2.0], [1.5, -1.5], [0.8, 1.2]], dtype=tf.float32)
    bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss_bce_logits = bce_logits(y_bin_true, y_bin_logits)
    print(f"BinaryCrossentropy (from_logits): {loss_bce_logits.numpy():.4f}")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(3, activation='softmax', input_shape=(4,))
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    x = tf.random.normal((3, 4))
    out = model(x)
    loss = cce(y_true_cat, out)
    print(f"Model CCE loss: {loss.numpy():.4f}")
    print("Classification losses verified.")

if __name__ == "__main__":
    main()
