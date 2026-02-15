"""
Validation within data pipeline (filter, assertions).
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Data Validation Pipeline")
    print("=" * 50)

    print("\n--- filter ---")
    ds = tf.data.Dataset.range(10)
    ds_valid = ds.filter(lambda x: x % 2 == 0)
    print(f"Filtered (evens): {list(ds_valid.as_numpy_iterator())}")

    print("\n--- filter with multiple conditions ---")
    ds_pairs = tf.data.Dataset.from_tensor_slices((
        tf.constant([1.0, -1.0, 0.5, 2.0, -0.1]),
        tf.constant([0, 1, 0, 1, 0])
    ))
    ds_clean = ds_pairs.filter(lambda x, y: tf.math.abs(x) <= 1.0)
    for x, y in ds_clean:
        print(f"  x: {x.numpy():.2f}, y: {y.numpy()}")

    print("\n--- tf.debugging.assert (concept) ---")
    def validate_and_scale(x, y):
        tf.debugging.assert_non_negative(x, message="x must be non-negative")
        return x * 2, y

    ds_safe = tf.data.Dataset.from_tensor_slices((
        tf.constant([1.0, 2.0, 3.0]),
        tf.constant([0, 1, 0])
    ))
    ds_validated = ds_safe.map(validate_and_scale)
    for x, y in ds_validated:
        print(f"  Validated: {x.numpy()}, {y.numpy()}")

    print("\n--- filter invalid labels ---")
    ds_labels = tf.data.Dataset.from_tensor_slices((
        tf.constant([[1.0], [2.0], [3.0]]),
        tf.constant([0, 1, -1])
    ))
    ds_label_filter = ds_labels.filter(lambda x, y: y >= 0)
    count = sum(1 for _ in ds_label_filter)
    print(f"Valid samples after label filter: {count}")

    print("\n--- assert_cardinality ---")
    ds_fixed = tf.data.Dataset.range(5)
    ds_asserted = ds_fixed.apply(tf.data.experimental.assert_cardinality(5))
    print(f"Cardinality asserted: {list(ds_asserted.as_numpy_iterator())}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
