"""
BatchNormalization, LayerNormalization, GroupNormalization.
"""
import tensorflow as tf

def main():
    x = tf.random.normal((8, 32, 32, 64))

    bn = tf.keras.layers.BatchNormalization()
    out_bn = bn(x, training=True)
    print(f"BatchNorm output: {out_bn.shape}, mean~0: {tf.reduce_mean(out_bn).numpy():.4f}")

    ln = tf.keras.layers.LayerNormalization(axis=-1)
    out_ln = ln(x)
    print(f"LayerNorm output: {out_ln.shape}")

    gn = tf.keras.layers.GroupNormalization(groups=8)
    out_gn = gn(x)
    print(f"GroupNorm (groups=8) output: {out_gn.shape}")

    ln_axis = tf.keras.layers.LayerNormalization(axis=[1, 2, 3])
    out_ln_axis = ln_axis(x)
    print(f"LayerNorm all spatial: {out_ln_axis.shape}")

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(64, 3),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    out = model(tf.random.normal((4, 28, 28, 1)))
    print(f"Model with norm layers output: {out.shape}")

    bn_train = bn(x, training=True)
    bn_infer = bn(x, training=False)
    print(f"BatchNorm train vs infer different: {not tf.reduce_all(tf.equal(bn_train, bn_infer)).numpy()}")
    print("Normalization layers verified.")

if __name__ == "__main__":
    main()
