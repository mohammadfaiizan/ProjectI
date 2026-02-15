"""
Statistics computation and batch processing for preprocessing.
"""
import tensorflow as tf
import numpy as np

def main():
    print("=" * 50)
    print("Statistics and Batch Processing")
    print("=" * 50)

    data = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

    print("\n--- Per-feature statistics ---")
    mean = tf.reduce_mean(data, axis=0)
    std = tf.math.reduce_std(data, axis=0)
    var = tf.math.reduce_variance(data, axis=0)
    print(f"Mean: {mean.numpy()}")
    print(f"Std: {std.numpy()}")
    print(f"Variance: {var.numpy()}")

    print("\n--- Min, max, median ---")
    min_val = tf.reduce_min(data, axis=0)
    max_val = tf.reduce_max(data, axis=0)
    sorted_d = tf.sort(data, axis=0)
    n = tf.shape(data)[0]
    mid = n // 2
    median = (tf.gather(sorted_d, mid - 1, axis=0) + tf.gather(sorted_d, mid, axis=0)) / 2.0
    print(f"Min: {min_val.numpy()}")
    print(f"Max: {max_val.numpy()}")
    print(f"Median (approx): {median.numpy()}")

    print("\n--- Batch statistics (moving) ---")
    batch1 = data[:2]
    batch2 = data[2:]
    m1 = tf.reduce_mean(batch1, axis=0)
    m2 = tf.reduce_mean(batch2, axis=0)
    combined_mean = (m1 + m2) / 2.0
    print(f"Batch1 mean: {m1.numpy()}")
    print(f"Batch2 mean: {m2.numpy()}")
    print(f"Combined mean: {combined_mean.numpy()}")

    print("\n--- Keras Normalization adapt (batch) ---")
    norm_layer = tf.keras.layers.Normalization(axis=-1)
    for batch in tf.data.Dataset.from_tensor_slices(data).batch(2):
        norm_layer.adapt(batch)
    normalized = norm_layer(data)
    print(f"Adapted norm output mean: {tf.reduce_mean(normalized).numpy():.4f}")

    print("\n--- Percentiles ---")
    flat = tf.reshape(data, [-1])
    sorted_flat = tf.sort(flat)
    idx_50 = tf.cast(tf.cast(tf.size(flat), tf.float32) * 0.5, tf.int32)
    p50_manual = tf.gather(sorted_flat, tf.minimum(idx_50, tf.size(sorted_flat) - 1))
    print(f"Approx median: {p50_manual.numpy()}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
