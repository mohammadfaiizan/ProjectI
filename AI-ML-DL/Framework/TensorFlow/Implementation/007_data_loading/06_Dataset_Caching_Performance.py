"""
cache, AUTOTUNE, interleave, parallel map.
"""
import tensorflow as tf
import tempfile
import os

def main():
    print("=" * 50)
    print("Dataset Caching and Performance")
    print("=" * 50)

    print("\n--- cache ---")
    ds = tf.data.Dataset.range(5).map(lambda x: x * 2)
    ds_cached = ds.cache()
    first = list(ds_cached.as_numpy_iterator())
    second = list(ds_cached.as_numpy_iterator())
    print(f"Cached (same both runs): {first} == {second}")

    print("\n--- cache to file ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "cache")
        ds_file_cache = tf.data.Dataset.range(4).map(lambda x: x + 10).cache(cache_path)
        print(f"File-cached: {list(ds_file_cache.as_numpy_iterator())}")

    print("\n--- AUTOTUNE ---")
    ds_auto = (
        tf.data.Dataset.range(20)
        .map(lambda x: x * 2, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(4)
        .prefetch(tf.data.AUTOTUNE)
    )
    for batch in ds_auto.take(2):
        print(f"  Batch: {batch.numpy()}")

    print("\n--- interleave ---")
    files = [tf.data.Dataset.range(i, i + 3) for i in [0, 10, 20]]
    ds_inter = tf.data.Dataset.from_tensor_slices(files).interleave(
        lambda x: x,
        cycle_length=3,
        block_length=1
    )
    print(f"Interleaved: {list(ds_inter.as_numpy_iterator())}")

    print("\n--- parallel map ---")
    ds_par = tf.data.Dataset.range(6).map(
        lambda x: x ** 2,
        num_parallel_calls=4
    )
    print(f"Parallel mapped: {list(ds_par.as_numpy_iterator())}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
