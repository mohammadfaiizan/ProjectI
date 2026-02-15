"""
tf.data.Dataset basics: element_spec, cardinality.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("tf.data.Dataset Fundamentals")
    print("=" * 50)

    print("\n--- Basic Dataset Creation ---")
    ds = tf.data.Dataset.range(5)
    print(f"Dataset: {list(ds.as_numpy_iterator())}")

    print("\n--- element_spec ---")
    ds_tensor = tf.data.Dataset.from_tensor_slices(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
    print(f"element_spec: {ds_tensor.element_spec}")
    print(f"element_spec structure: {ds_tensor.element_spec}")

    print("\n--- cardinality ---")
    ds_finite = tf.data.Dataset.range(10)
    print(f"Finite cardinality: {ds_finite.cardinality().numpy()}")
    ds_infinite = tf.data.Dataset.range(5).repeat()
    print(f"Infinite cardinality: {tf.data.INFINITE_CARDINALITY}")

    print("\n--- Nested Structure ---")
    ds_nested = tf.data.Dataset.from_tensor_slices({
        "features": tf.constant([[1.0], [2.0], [3.0]]),
        "labels": tf.constant([0, 1, 0])
    })
    print(f"Nested element_spec: {ds_nested.element_spec}")

    print("\n--- Iteration ---")
    for i, elem in enumerate(ds_finite.take(3)):
        print(f"  Element {i}: {elem.numpy()}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
