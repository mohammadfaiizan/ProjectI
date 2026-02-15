"""
Feature scaling: Min-max, z-score, robust scaling in TF.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Feature Scaling Methods")
    print("=" * 50)

    data = tf.constant([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0], [40.0, 400.0]])

    print("\n--- Min-max scaling ---")
    min_val = tf.reduce_min(data, axis=0)
    max_val = tf.reduce_max(data, axis=0)
    minmax = (data - min_val) / (max_val - min_val + 1e-8)
    print(f"Min-max range: [{tf.reduce_min(minmax).numpy():.4f}, {tf.reduce_max(minmax).numpy():.4f}]")
    print(f"Min-max sample: {minmax[0].numpy()}")

    print("\n--- Z-score (standardization) ---")
    mean = tf.reduce_mean(data, axis=0)
    std = tf.math.reduce_std(data, axis=0)
    zscore = (data - mean) / (std + 1e-8)
    print(f"Z-score mean: {tf.reduce_mean(zscore).numpy():.4f}")
    print(f"Z-score std: {tf.math.reduce_std(zscore).numpy():.4f}")

    print("\n--- Robust scaling (IQR) ---")
    sorted_d = tf.sort(data, axis=0)
    n = tf.shape(data)[0]
    q25_idx = tf.maximum(0, tf.cast(tf.cast(n, tf.float32) * 0.25, tf.int32))
    q75_idx = tf.minimum(n - 1, tf.cast(tf.cast(n, tf.float32) * 0.75, tf.int32))
    q25 = tf.gather(sorted_d, q25_idx, axis=0)
    q75 = tf.gather(sorted_d, q75_idx, axis=0)
    median = tf.gather(sorted_d, n // 2, axis=0)
    iqr = q75 - q25
    robust = (data - median) / (iqr + 1e-8)
    print(f"Robust scaled sample: {robust[0].numpy()}")

    print("\n--- Max-abs scaling ---")
    max_abs = tf.reduce_max(tf.abs(data), axis=0)
    maxabs_scaled = data / (max_abs + 1e-8)
    print(f"Max-abs sample: {maxabs_scaled[0].numpy()}")

    print("\n--- Keras Normalization layer ---")
    norm_layer = tf.keras.layers.Normalization(axis=-1)
    norm_layer.adapt(data)
    keras_norm = norm_layer(data)
    print(f"Keras norm mean: {tf.reduce_mean(keras_norm).numpy():.4f}")
    print(f"Keras norm std: {tf.math.reduce_std(keras_norm).numpy():.4f}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
