"""
Data validation: schema checks, range validation, outlier detection.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Data Validation Utilities")
    print("=" * 50)

    data = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    print("\n--- Shape validation ---")
    expected_shape = (None, 2)
    actual = tf.shape(data)
    tf.debugging.assert_equal(actual[1], 2, message="Expected 2 features")
    print(f"Shape validated: {data.shape}")

    print("\n--- Range validation (clipping) ---")
    x = tf.constant([0.5, 1.5, -0.2, 2.5])
    clipped = tf.clip_by_value(x, 0.0, 1.0)
    print(f"Clipped to [0,1]: {clipped.numpy()}")

    print("\n--- Outlier detection (IQR) ---")
    vals = tf.constant([1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 100.0])
    sorted_vals = tf.sort(vals)
    n = tf.size(vals)
    q1_idx = tf.cast(tf.cast(n, tf.float32) * 0.25, tf.int32)
    q3_idx = tf.cast(tf.cast(n, tf.float32) * 0.75, tf.int32)
    q1 = tf.gather(sorted_vals, tf.minimum(q1_idx, n - 1))
    q3 = tf.gather(sorted_vals, tf.minimum(q3_idx, n - 1))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (vals >= lower) & (vals <= upper)
    filtered = tf.boolean_mask(vals, mask)
    print(f"Filtered (IQR): {filtered.numpy()}")

    print("\n--- NaN/Inf check ---")
    clean = tf.constant([1.0, 2.0, 3.0])
    has_nan = tf.reduce_any(tf.math.is_nan(clean))
    has_inf = tf.reduce_any(tf.math.is_inf(clean))
    print(f"Has NaN: {has_nan.numpy()}, Has Inf: {has_inf.numpy()}")

    print("\n--- Dtype validation ---")
    tf.debugging.assert_type(data, tf.float32)
    print("Dtype float32 validated")

    print("\n--- Assert positive ---")
    pos = tf.constant([1.0, 2.0, 3.0])
    tf.debugging.assert_positive(pos)
    print("All values positive")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
