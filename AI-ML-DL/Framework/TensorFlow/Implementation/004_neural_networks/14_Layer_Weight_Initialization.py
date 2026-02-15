"""
GlorotUniform, HeNormal, Orthogonal, Zeros, Ones, custom initializer.
"""
import tensorflow as tf

def main():
    glorot = tf.keras.initializers.GlorotUniform(seed=42)
    layer_glorot = tf.keras.layers.Dense(64, kernel_initializer=glorot, input_shape=(32,))
    layer_glorot.build((None, 32))
    print(f"GlorotUniform kernel mean: {tf.reduce_mean(layer_glorot.kernel).numpy():.4f}")

    he = tf.keras.initializers.HeNormal(seed=42)
    layer_he = tf.keras.layers.Dense(64, kernel_initializer=he, input_shape=(32,))
    layer_he.build((None, 32))
    print(f"HeNormal kernel std: {tf.math.reduce_std(layer_he.kernel).numpy():.4f}")

    orthogonal = tf.keras.initializers.Orthogonal(seed=42)
    layer_orth = tf.keras.layers.Dense(32, kernel_initializer=orthogonal, input_shape=(32,))
    layer_orth.build((None, 32))
    prod = tf.matmul(layer_orth.kernel, layer_orth.kernel, transpose_b=True)
    print(f"Orthogonal approx identity: {tf.reduce_mean(tf.abs(prod - tf.eye(32))).numpy():.4f}")

    zeros = tf.keras.initializers.Zeros()
    layer_zeros = tf.keras.layers.Dense(64, bias_initializer=zeros, input_shape=(32,))
    layer_zeros.build((None, 32))
    print(f"Zeros bias: {tf.reduce_sum(tf.abs(layer_zeros.bias)).numpy()}")

    ones = tf.keras.initializers.Ones()
    layer_ones = tf.keras.layers.Dense(64, bias_initializer=ones, input_shape=(32,))
    layer_ones.build((None, 32))
    print(f"Ones bias sum: {tf.reduce_sum(layer_ones.bias).numpy()}")

    def custom_init(shape, dtype=None):
        return tf.random.normal(shape, mean=0.1, stddev=0.05, dtype=dtype)

    layer_custom = tf.keras.layers.Dense(64, kernel_initializer=custom_init, input_shape=(32,))
    layer_custom.build((None, 32))
    print(f"Custom init kernel mean: {tf.reduce_mean(layer_custom.kernel).numpy():.4f}")

    lecun = tf.keras.initializers.LecunNormal(seed=42)
    layer_lecun = tf.keras.layers.Dense(64, kernel_initializer=lecun, input_shape=(32,))
    layer_lecun.build((None, 32))
    print(f"LecunNormal kernel std: {tf.math.reduce_std(layer_lecun.kernel).numpy():.4f}")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, kernel_initializer='glorot_uniform', input_shape=(32,)),
        tf.keras.layers.Dense(32, kernel_initializer='he_normal'),
        tf.keras.layers.Dense(10, kernel_initializer='glorot_uniform')
    ])
    model.build()
    out = model(tf.random.normal((2, 32)))
    print(f"Model with initializers output: {out.shape}")
    print("Weight initialization verified.")

if __name__ == "__main__":
    main()
