"""
Custom model via subclassing tf.keras.Model (__init__, call).
"""
import tensorflow as tf

class CustomModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.dense3 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

def main():
    model = CustomModel(num_classes=10)
    model.build(input_shape=(None, 784))
    print("Subclassed model:")
    model.summary()

    x = tf.random.normal((4, 784))
    out = model(x)
    print(f"\nForward pass output shape: {out.shape}")

    class CNNSubclass(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
            self.pool = tf.keras.layers.GlobalAveragePooling2D()
            self.dense = tf.keras.layers.Dense(10, activation='softmax')

        def call(self, inputs):
            x = self.conv1(inputs)
            x = self.pool(x)
            return self.dense(x)

    cnn = CNNSubclass()
    cnn.build(input_shape=(None, 28, 28, 1))
    print("\nSubclassed CNN model:")
    cnn.summary()
    print("Model subclassing verified.")

if __name__ == "__main__":
    main()
