"""
Nesting models, shared layers, reuse.
"""
import tensorflow as tf
import numpy as np

def main():
    shared_dense = tf.keras.layers.Dense(64, activation='relu', name='shared_dense')

    input_a = tf.keras.Input(shape=(32,))
    input_b = tf.keras.Input(shape=(32,))

    out_a = shared_dense(input_a)
    out_b = shared_dense(input_b)

    out_a = tf.keras.layers.Dense(10, activation='softmax')(out_a)
    out_b = tf.keras.layers.Dense(10, activation='softmax')(out_b)

    model_shared = tf.keras.Model(inputs=[input_a, input_b], outputs=[out_a, out_b])
    print("Shared layer model:")
    model_shared.summary()

    encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
        tf.keras.layers.Dense(32, activation='relu')
    ], name='encoder')

    decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
        tf.keras.layers.Dense(32, activation='sigmoid')
    ], name='decoder')

    autoencoder_input = tf.keras.Input(shape=(32,))
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = tf.keras.Model(autoencoder_input, decoded, name='autoencoder')
    print("\nNested model (encoder-decoder):")
    autoencoder.summary()

    x = tf.random.normal((4, 32))
    out = autoencoder(x)
    print(f"\nAutoencoder output shape: {out.shape}")

    sub_out = encoder(x)
    print(f"Encoder submodel output shape: {sub_out.shape}")
    print("Model composition patterns verified.")

if __name__ == "__main__":
    main()
