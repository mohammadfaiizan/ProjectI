"""
Full custom training loop with GradientTape.
"""
import tensorflow as tf
import numpy as np

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    x_train = x_train.reshape(-1, 784)[:500]
    x_test = x_test.reshape(-1, 784)[:100]
    y_train = tf.keras.utils.to_categorical(y_train[:500], 10)
    y_test = tf.keras.utils.to_categorical(y_test[:100], 10)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    epochs = 2
    batch_size = 32

    for epoch in range(epochs):
        epoch_loss = tf.keras.metrics.Mean()
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            with tf.GradientTape() as tape:
                y_pred = model(x_batch, training=True)
                loss = loss_fn(y_batch, y_pred)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss.update_state(loss)

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss.result():.4f}")

    test_loss = loss_fn(y_test, model(x_test, training=False))
    print(f"\nTest loss: {test_loss.numpy():.4f}")
    print("Custom training loop with GradientTape verified.")

if __name__ == "__main__":
    main()
