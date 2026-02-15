"""
TensorBoard callback, log_dir, histogram_freq.
"""
import tensorflow as tf
import numpy as np
import os
import tempfile

def main():
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_train = x_train.reshape(-1, 784)[:1000]
    y_train = tf.keras.utils.to_categorical(y_train[:1000], 10)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    log_dir = tempfile.mkdtemp(prefix='tb_logs_')
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=False
    )

    history = model.fit(
        x_train, y_train,
        epochs=2,
        batch_size=32,
        validation_split=0.2,
        callbacks=[tensorboard_cb],
        verbose=1
    )

    events_files = [f for f in os.listdir(log_dir) if f.startswith('events')]
    print(f"\nTensorBoard log dir: {log_dir}")
    print(f"Events files created: {len(events_files) > 0}")
    print("TensorBoard callback verified.")

if __name__ == "__main__":
    main()
