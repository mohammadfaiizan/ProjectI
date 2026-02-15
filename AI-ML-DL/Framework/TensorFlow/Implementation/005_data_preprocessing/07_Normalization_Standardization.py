"""
Feature normalization techniques with tf operations.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Normalization and Standardization")
    print("=" * 50)

    print("\n--- Batch normalization (manual) ---")
    x = tf.random.normal((32, 64))
    batch_mean = tf.reduce_mean(x, axis=0)
    batch_var = tf.math.reduce_variance(x, axis=0)
    x_norm = (x - batch_mean) / tf.sqrt(batch_var + 1e-5)
    print(f"Batch norm output mean: {tf.reduce_mean(x_norm).numpy():.4f}")
    print(f"Batch norm output std: {tf.math.reduce_std(x_norm).numpy():.4f}")

    print("\n--- Layer normalization ---")
    layer = tf.keras.layers.LayerNormalization(axis=-1)
    ln_out = layer(x)
    print(f"Layer norm output shape: {ln_out.shape}")
    print(f"Layer norm mean: {tf.reduce_mean(ln_out).numpy():.4f}")

    print("\n--- Feature-wise standardization ---")
    features = tf.constant([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
    f_mean = tf.reduce_mean(features, axis=0)
    f_std = tf.math.reduce_std(features, axis=0)
    feat_std = (features - f_mean) / (f_std + 1e-8)
    print(f"Feature standardized: {feat_std.numpy()}")

    print("\n--- Unit norm (per sample) ---")
    samples = tf.constant([[3.0, 4.0], [1.0, 0.0]])
    unit_norm = tf.math.l2_normalize(samples, axis=-1)
    print(f"Unit norm sample: {unit_norm[0].numpy()}")
    print(f"Norm: {tf.norm(unit_norm[0]).numpy():.4f}")

    print("\n--- Robust scaling (median, IQR) ---")
    data = tf.constant([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
    sorted_data = tf.sort(data, axis=0)
    n = tf.shape(data)[0]
    mid_lo = tf.gather(sorted_data, tf.maximum(0, n//2 - 1), axis=0)
    mid_hi = tf.gather(sorted_data, n//2, axis=0)
    median = (mid_lo + mid_hi) / 2
    q25_idx = tf.maximum(0, tf.cast(tf.cast(n, tf.float32) * 0.25, tf.int32))
    q75_idx = tf.minimum(n - 1, tf.cast(tf.cast(n, tf.float32) * 0.75, tf.int32))
    q25 = tf.gather(sorted_data, q25_idx, axis=0)
    q75 = tf.gather(sorted_data, q75_idx, axis=0)
    iqr = q75 - q25
    robust = (data - median) / (iqr + 1e-8)
    print(f"Robust scaled sample: {robust[0].numpy()}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
