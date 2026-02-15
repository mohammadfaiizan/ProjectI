"""
MAML-style meta-learning with GradientTape.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("MAML-Style Meta-Learning")
    print("=" * 50)

    def create_model():
        return tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(8,)),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(4, activation='softmax')
        ])

    model = create_model()
    meta_lr = 0.01
    inner_lr = 0.1
    inner_steps = 3

    support_x = tf.random.normal((16, 8))
    support_y = tf.random.uniform((16,), maxval=4, dtype=tf.int32)
    query_x = tf.random.normal((8, 8))
    query_y = tf.random.uniform((8,), maxval=4, dtype=tf.int32)

    theta = [tf.identity(v) for v in model.trainable_variables]
    for step in range(inner_steps):
        with tf.GradientTape() as tape:
            for i, v in enumerate(model.trainable_variables):
                v.assign(theta[i])
            pred = model(support_x, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(support_y, pred)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model.trainable_variables)
        theta = [t - inner_lr * g for t, g in zip(theta, grads)]

    for i, v in enumerate(model.trainable_variables):
        v.assign(theta[i])
    with tf.GradientTape() as meta_tape:
        pred_query = model(query_x, training=True)
        meta_loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(query_y, pred_query)
        )
    meta_grads = meta_tape.gradient(meta_loss, model.trainable_variables)
    optimizer = tf.keras.optimizers.Adam(meta_lr)
    optimizer.apply_gradients(zip(meta_grads, model.trainable_variables))
    print(f"Meta-loss: {meta_loss.numpy():.4f}")

    print("MAML inner/outer loop complete.")

if __name__ == "__main__":
    main()
