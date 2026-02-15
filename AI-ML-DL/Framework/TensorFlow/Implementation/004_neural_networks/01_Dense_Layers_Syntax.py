"""
Dense layer syntax: units, activation, kernel/bias access.
"""
import tensorflow as tf

def main():
    layer = tf.keras.layers.Dense(64, activation='relu', input_shape=(32,))
    x = tf.random.normal((2, 32))
    out = layer(x)
    print(f"Dense output shape: {out.shape}")

    layer2 = tf.keras.layers.Dense(10, activation='softmax', use_bias=True)
    layer2.build((None, 64))
    kernel = layer2.kernel
    bias = layer2.bias
    print(f"Kernel shape: {kernel.shape}, Bias shape: {bias.shape}")

    layer3 = tf.keras.layers.Dense(32, use_bias=False)
    layer3.build((None, 16))
    print(f"Weights (no bias): {layer3.weights}")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.build()
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'kernel'):
            print(f"Layer {i} kernel: {layer.kernel.shape}")

    x_test = tf.random.normal((4, 784))
    pred = model(x_test)
    print(f"Model output shape: {pred.shape}")
    print("Dense layers syntax verified.")

if __name__ == "__main__":
    main()
