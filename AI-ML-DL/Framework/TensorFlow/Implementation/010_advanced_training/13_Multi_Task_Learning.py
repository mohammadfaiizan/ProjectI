"""
Shared encoder, task-specific heads, loss weighting.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Multi-Task Learning")
    print("=" * 50)

    shared = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
        tf.keras.layers.Dense(32, activation='relu')
    ])
    head_a = tf.keras.layers.Dense(5, activation='softmax', name='task_a')
    head_b = tf.keras.layers.Dense(1, activation='sigmoid', name='task_b')

    def mtl_model(x):
        feat = shared(x, training=True)
        out_a = head_a(feat)
        out_b = head_b(feat)
        return out_a, out_b

    x = tf.random.normal((128, 32))
    y_a = tf.random.uniform((128,), maxval=5, dtype=tf.int32)
    y_b = tf.random.uniform((128,), maxval=2, dtype=tf.float32)
    w_a, w_b = 1.0, 0.5

    optimizer = tf.keras.optimizers.Adam(0.001)
    vars = shared.trainable_variables + head_a.trainable_variables + head_b.trainable_variables
    for _ in range(5):
        with tf.GradientTape() as tape:
            pred_a, pred_b = mtl_model(x)
            loss_a = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y_a, pred_a)
            )
            loss_b = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_b, pred_b))
            loss = w_a * loss_a + w_b * loss_b
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))
    print(f"Task A loss: {loss_a.numpy():.4f}, Task B loss: {loss_b.numpy():.4f}")
    print(f"Total MTL loss: {loss.numpy():.4f}")

    pred_a, pred_b = mtl_model(x[:5])
    print(f"Head A output shape: {pred_a.shape}, Head B: {pred_b.shape}")
    print("Multi-task learning complete.")

if __name__ == "__main__":
    main()
