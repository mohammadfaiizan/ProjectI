"""
Pretext tasks, rotation prediction.
"""
import tensorflow as tf
import numpy as np

def main():
    print("=" * 50)
    print("Self-Supervised Learning - Rotation Prediction")
    print("=" * 50)

    def rotate_batch(images, k):
        return tf.image.rot90(images, k=k)

    encoder = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(0.001)

    images = tf.random.normal((64, 28, 28, 1))
    rotations = [0, 1, 2, 3]
    batch_rotated = []
    labels = []
    for k in rotations:
        rot_img = rotate_batch(images, k)
        batch_rotated.append(rot_img)
        labels.extend([k] * 64)
    batch_rotated = tf.concat(batch_rotated, axis=0)
    labels = tf.constant(labels, dtype=tf.int32)

    for _ in range(3):
        with tf.GradientTape() as tape:
            pred = encoder(batch_rotated, training=True)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(labels, pred)
            )
        grads = tape.gradient(loss, encoder.trainable_variables)
        optimizer.apply_gradients(zip(grads, encoder.trainable_variables))
    print(f"Rotation prediction loss: {loss.numpy():.4f}")

    acc = np.mean(np.argmax(pred.numpy(), axis=1) == labels.numpy())
    print(f"Rotation accuracy: {acc:.2%}")
    print("Self-supervised pretext task complete.")

if __name__ == "__main__":
    main()
