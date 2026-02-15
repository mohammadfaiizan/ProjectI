"""
Overriding train_step and test_step.
"""
import tensorflow as tf
import numpy as np

class CustomTrainStepModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(y=y, y_pred=y_pred)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = self.compute_loss(y=y, y_pred=y_pred)
        return {"loss": loss}

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    x_train = x_train.reshape(-1, 784)[:500]
    x_test = x_test.reshape(-1, 784)[:100]
    y_train = tf.keras.utils.to_categorical(y_train[:500], 10)
    y_test = tf.keras.utils.to_categorical(y_test[:100], 10)

    model = CustomTrainStepModel()
    model.build(input_shape=(None, 784))
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    history = model.fit(x_train, y_train, epochs=2, batch_size=32, verbose=1)
    print(f"\nCustom train_step loss: {history.history['loss']}")

    results = model.evaluate(x_test, y_test, verbose=1)
    print(f"Custom test_step loss: {results}")
    print("Custom train_step and test_step verified.")

if __name__ == "__main__":
    main()
