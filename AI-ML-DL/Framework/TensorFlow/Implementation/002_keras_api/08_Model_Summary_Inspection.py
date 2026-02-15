"""
model.summary(), model.layers, get_weights, set_weights.
"""
import tensorflow as tf

def main():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    print("model.summary():")
    model.summary()

    print("\nmodel.layers:")
    for i, layer in enumerate(model.layers):
        print(f"  Layer {i}: {layer.name}, output shape: {layer.output_shape}")

    weights = model.get_weights()
    print(f"\nNumber of weight arrays: {len(weights)}")
    for i, w in enumerate(weights):
        print(f"  Weight {i}: shape={w.shape}")

    model2 = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model2.set_weights(weights)
    print("\nWeights copied to model2 via set_weights.")

    x = tf.random.normal((2, 784))
    out1 = model(x)
    out2 = model2(x)
    print(f"Outputs match after set_weights: {tf.reduce_all(tf.abs(out1 - out2) < 1e-5)}")
    print("Model summary and inspection verified.")

if __name__ == "__main__":
    main()
