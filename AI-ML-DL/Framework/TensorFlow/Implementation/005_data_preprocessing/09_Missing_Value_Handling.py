"""
Missing value handling: NaN detection, masking, imputation with tensors.
"""
import tensorflow as tf
import numpy as np

def main():
    print("=" * 50)
    print("Missing Value Handling")
    print("=" * 50)

    print("\n--- NaN detection ---")
    data = tf.constant([1.0, float('nan'), 3.0, float('nan'), 5.0])
    is_nan = tf.math.is_nan(data)
    print(f"is_nan mask: {is_nan.numpy()}")
    nan_count = tf.reduce_sum(tf.cast(is_nan, tf.int32))
    print(f"NaN count: {nan_count.numpy()}")

    print("\n--- Masking NaN values ---")
    masked = tf.where(is_nan, tf.zeros_like(data), data)
    print(f"Masked (zeros): {masked.numpy()}")

    print("\n--- Imputation (mean) ---")
    valid_mask = ~is_nan
    valid_vals = tf.boolean_mask(data, valid_mask)
    mean_val = tf.reduce_mean(valid_vals)
    imputed = tf.where(is_nan, tf.fill(tf.shape(data), mean_val), data)
    print(f"Mean imputed: {imputed.numpy()}")

    print("\n--- Inf detection ---")
    with_inf = tf.constant([1.0, float('inf'), 3.0, float('-inf')])
    is_finite = tf.math.is_finite(with_inf)
    print(f"is_finite: {is_finite.numpy()}")

    print("\n--- Replace NaN with constant ---")
    repl = tf.where(tf.math.is_nan(data), -1.0, data)
    print(f"Replaced NaN with -1: {repl.numpy()}")

    print("\n--- 2D imputation ---")
    mat = tf.constant([[1.0, float('nan')], [3.0, 4.0], [float('nan'), 6.0]])
    col_means = tf.reduce_mean(tf.where(tf.math.is_nan(mat), tf.zeros_like(mat), mat), axis=0)
    col_counts = tf.reduce_sum(tf.cast(~tf.math.is_nan(mat), tf.float32), axis=0)
    safe_means = col_means / (col_counts + 1e-8)
    imputed_mat = tf.where(tf.math.is_nan(mat), tf.broadcast_to(safe_means, tf.shape(mat)), mat)
    print(f"Column-mean imputed:\n{imputed_mat.numpy()}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
