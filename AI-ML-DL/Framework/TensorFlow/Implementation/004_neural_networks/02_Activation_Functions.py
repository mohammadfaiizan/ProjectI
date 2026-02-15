"""
All activation functions: ReLU, LeakyReLU, PReLU, ELU, SELU, GELU, Swish, Sigmoid, Tanh, Softmax.
"""
import tensorflow as tf

def main():
    x = tf.constant([[-2.0, -1.0, 0.0, 1.0, 2.0]])

    relu = tf.keras.layers.ReLU()
    print(f"ReLU: {relu(x).numpy()}")

    leaky = tf.keras.layers.LeakyReLU(alpha=0.1)
    print(f"LeakyReLU: {leaky(x).numpy()}")

    prelu = tf.keras.layers.PReLU()
    prelu.build(x.shape)
    print(f"PReLU: {prelu(x).numpy()}")

    elu = tf.keras.layers.ELU(alpha=1.0)
    print(f"ELU: {elu(x).numpy()}")

    selu = tf.keras.layers.Activation('selu')
    print(f"SELU: {selu(x).numpy()}")

    gelu = tf.keras.layers.Activation('gelu')
    print(f"GELU: {gelu(x).numpy()}")

    swish = tf.keras.layers.Activation('swish')
    print(f"Swish: {swish(x).numpy()}")

    sigmoid = tf.keras.layers.Activation('sigmoid')
    print(f"Sigmoid: {sigmoid(x).numpy()}")

    tanh = tf.keras.layers.Activation('tanh')
    print(f"Tanh: {tanh(x).numpy()}")

    softmax = tf.keras.layers.Softmax()
    logits = tf.random.normal((2, 5))
    print(f"Softmax (sum=1): {tf.reduce_sum(softmax(logits), axis=1).numpy()}")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, input_shape=(16,)),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Softmax()
    ])
    out = model(tf.random.normal((2, 16)))
    print(f"Model with activations output shape: {out.shape}")
    print("Activation functions verified.")

if __name__ == "__main__":
    main()
