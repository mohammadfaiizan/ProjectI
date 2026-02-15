"""
Custom callback class (on_epoch_begin/end, on_train_begin/end).
"""
import tensorflow as tf
import numpy as np
import time

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_times = []

    def on_train_begin(self, logs=None):
        print("Training started.")

    def on_train_end(self, logs=None):
        print("Training ended.")

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time()
        print(f"Epoch {epoch + 1} began.")

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self._epoch_start
        self.epoch_times.append(elapsed)
        if logs:
            print(f"Epoch {epoch + 1} end - loss: {logs.get('loss', 'N/A'):.4f}")

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

def main():
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_train = x_train.reshape(-1, 784)[:500]
    y_train = tf.keras.utils.to_categorical(y_train[:500], 10)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    custom_cb = CustomCallback()
    history = model.fit(
        x_train, y_train,
        epochs=2,
        batch_size=32,
        callbacks=[custom_cb],
        verbose=1
    )

    print(f"\nCustom callback epoch times recorded: {len(custom_cb.epoch_times)}")
    print("Custom callback verified.")

if __name__ == "__main__":
    main()
