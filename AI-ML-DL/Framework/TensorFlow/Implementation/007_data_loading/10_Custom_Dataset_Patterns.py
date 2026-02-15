"""
Custom dataset creation patterns.
"""
import tensorflow as tf
import numpy as np

def main():
    print("=" * 50)
    print("Custom Dataset Patterns")
    print("=" * 50)

    print("\n--- Pattern: List of Files ---")
    files = [f"file_{i}.npy" for i in range(3)]
    def load_file(path):
        idx = int(path.split("_")[1].split(".")[0])
        return tf.constant([float(idx), float(idx * 2)])
    ds_files = tf.data.Dataset.from_tensor_slices(files)
    ds_loaded = ds_files.map(
        lambda p: load_file(p),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    for x in ds_loaded.take(2):
        print(f"  Loaded: {x.numpy()}")

    print("\n--- Pattern: Zip Multiple Datasets ---")
    ds_a = tf.data.Dataset.range(3)
    ds_b = tf.data.Dataset.range(10, 13)
    ds_zip = tf.data.Dataset.zip((ds_a, ds_b))
    for a, b in ds_zip:
        print(f"  ({a.numpy()}, {b.numpy()})")

    print("\n--- Pattern: Concatenate ---")
    ds1 = tf.data.Dataset.range(2)
    ds2 = tf.data.Dataset.range(5, 8)
    ds_concat = ds1.concatenate(ds2)
    print(f"Concatenated: {list(ds_concat.as_numpy_iterator())}")

    print("\n--- Pattern: Filter ---")
    ds_filter = tf.data.Dataset.range(10).filter(lambda x: x % 2 == 0)
    print(f"Filtered evens: {list(ds_filter.as_numpy_iterator())}")

    print("\n--- Pattern: Window (for sequences) ---")
    ds_seq = tf.data.Dataset.range(10)
    ds_windows = ds_seq.window(size=3, shift=2, drop_remainder=True)
    for i, w in enumerate(ds_windows.take(2)):
        flat = w.flat_map(lambda x: x)
        print(f"  Window {i}: {list(flat.as_numpy_iterator())}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
