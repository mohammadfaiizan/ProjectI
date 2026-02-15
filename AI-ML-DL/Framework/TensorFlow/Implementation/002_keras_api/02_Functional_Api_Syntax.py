"""
Functional API: Input, layer calls, Model.
"""
import tensorflow as tf

def main():
    inputs = tf.keras.Input(shape=(784,))
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    print("Functional API model:")
    model.summary()

    inputs2 = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs2)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs2 = tf.keras.layers.Dense(10, activation='softmax')(x)

    model2 = tf.keras.Model(inputs=inputs2, outputs=outputs2)
    print("\nFunctional API CNN model:")
    model2.summary()

    x_test = tf.random.normal((2, 784))
    out = model(x_test)
    print(f"\nForward pass output shape: {out.shape}")
    print("Functional API syntax verified.")

if __name__ == "__main__":
    main()
