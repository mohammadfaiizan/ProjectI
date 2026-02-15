"""
Debugging data pipelines: take, as_numpy_iterator.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Data Loading Debugging")
    print("=" * 50)

    ds = tf.data.Dataset.range(10).map(lambda x: x * 2).batch(3)

    print("\n--- take ---")
    ds_limited = ds.take(2)
    for batch in ds_limited:
        print(f"  Batch: {batch.numpy()}")

    print("\n--- as_numpy_iterator ---")
    ds_small = tf.data.Dataset.range(5)
    np_list = list(ds_small.as_numpy_iterator())
    print(f"As numpy: {np_list}")

    print("\n--- element_spec inspection ---")
    ds_complex = tf.data.Dataset.from_tensor_slices({
        "a": tf.constant([1.0, 2.0]),
        "b": tf.constant([0, 1])
    })
    print(f"element_spec: {ds_complex.element_spec}")

    print("\n--- reduce for debugging ---")
    ds_count = tf.data.Dataset.range(7)
    count = ds_count.reduce(0, lambda state, x: state + 1)
    print(f"Element count: {count.numpy()}")

    print("\n--- skip ---")
    ds_skip = tf.data.Dataset.range(10).skip(3)
    print(f"After skip(3): {list(ds_skip.take(4).as_numpy_iterator())}")

    print("\n--- Debugging pipeline step by step ---")
    ds_pipe = tf.data.Dataset.range(6).map(lambda x: x + 1).batch(2)
    step1 = list(ds_pipe.take(1).as_numpy_iterator())
    print(f"Pipeline output sample: {step1}")

    print("\n--- tf.data.experimental.sample_from_datasets ---")
    ds_a = tf.data.Dataset.range(3)
    ds_b = tf.data.Dataset.range(10, 13)
    ds_sampled = tf.data.experimental.sample_from_datasets([ds_a, ds_b], weights=[0.5, 0.5], seed=42)
    sampled = list(ds_sampled.as_numpy_iterator())
    print(f"Sampled: {sampled}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
