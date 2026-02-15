"""
Dropout, SpatialDropout1D/2D, AlphaDropout.
"""
import tensorflow as tf

def main():
    tf.random.set_seed(42)
    x = tf.random.normal((4, 64))

    dropout = tf.keras.layers.Dropout(0.5)
    out = dropout(x, training=True)
    zeros = tf.reduce_sum(tf.cast(tf.equal(out, 0), tf.float32))
    print(f"Dropout 0.5 zeros ratio: {zeros.numpy() / x.shape.num_elements():.2f}")

    x1d = tf.random.normal((4, 50, 32))
    spatial1d = tf.keras.layers.SpatialDropout1D(0.3)
    out1d = spatial1d(x1d, training=True)
    print(f"SpatialDropout1D: {x1d.shape} -> {out1d.shape}")

    x2d = tf.random.normal((4, 28, 28, 64))
    spatial2d = tf.keras.layers.SpatialDropout2D(0.3)
    out2d = spatial2d(x2d, training=True)
    print(f"SpatialDropout2D: {x2d.shape} -> {out2d.shape}")

    alpha_dropout = tf.keras.layers.AlphaDropout(0.2)
    out_alpha = alpha_dropout(x, training=True)
    print(f"AlphaDropout output shape: {out_alpha.shape}")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(64,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    out_model = model(x, training=True)
    print(f"Model with dropout output: {out_model.shape}")

    out_infer = dropout(x, training=False)
    print(f"Dropout training=False preserves input: {tf.reduce_all(tf.equal(out_infer, x)).numpy()}")
    print("Dropout layers verified.")

if __name__ == "__main__":
    main()
