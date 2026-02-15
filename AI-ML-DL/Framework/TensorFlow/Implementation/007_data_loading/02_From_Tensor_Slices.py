"""
Dataset.from_tensor_slices with arrays, dicts, tuples.
"""
import tensorflow as tf
import numpy as np

def main():
    print("=" * 50)
    print("from_tensor_slices")
    print("=" * 50)

    print("\n--- From NumPy Array ---")
    arr = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices(arr)
    for i, x in enumerate(ds.take(2)):
        print(f"  Sample {i}: {x.numpy()}")

    print("\n--- From Dict ---")
    data_dict = {
        "features": tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        "labels": tf.constant([0, 1, 0])
    }
    ds_dict = tf.data.Dataset.from_tensor_slices(data_dict)
    for elem in ds_dict.take(2):
        print(f"  features: {elem['features'].numpy()}, label: {elem['labels'].numpy()}")

    print("\n--- From Tuple ---")
    X = tf.constant([[1.0], [2.0], [3.0]])
    y = tf.constant([0, 1, 0])
    ds_tuple = tf.data.Dataset.from_tensor_slices((X, y))
    for feat, label in ds_tuple.take(2):
        print(f"  x: {feat.numpy()}, y: {label.numpy()}")

    print("\n--- Multiple Arrays (aligned) ---")
    a = tf.constant([1, 2, 3])
    b = tf.constant([10, 20, 30])
    ds_multi = tf.data.Dataset.from_tensor_slices((a, b))
    for x, y in ds_multi:
        print(f"  ({x.numpy()}, {y.numpy()})")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
