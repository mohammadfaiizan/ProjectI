"""
Sequential model creation using tf.keras.Sequential and add layers.
"""
import tensorflow as tf

def main():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    print("Sequential model (add layers):")
    model.summary()

    model2 = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    print("\nSequential model (list constructor):")
    model2.summary()

    model3 = tf.keras.Sequential()
    model3.add(tf.keras.layers.Input(shape=(784,)))
    model3.add(tf.keras.layers.Dense(64, activation='relu'))
    model3.add(tf.keras.layers.Dropout(0.2))
    model3.add(tf.keras.layers.Dense(10, activation='softmax'))
    print("\nSequential model (with Input and Dropout):")
    model3.summary()

    x = tf.random.normal((2, 784))
    out = model(x)
    print(f"\nForward pass output shape: {out.shape}")
    print("Sequential model syntax verified.")

if __name__ == "__main__":
    main()
