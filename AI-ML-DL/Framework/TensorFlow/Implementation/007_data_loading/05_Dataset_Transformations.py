"""
map, batch, shuffle, repeat, prefetch chaining.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Dataset Transformations")
    print("=" * 50)

    ds = tf.data.Dataset.range(10)

    print("\n--- map ---")
    ds_mapped = ds.map(lambda x: x * 2)
    print(f"Doubled: {list(ds_mapped.take(5).as_numpy_iterator())}")

    print("\n--- batch ---")
    ds_batched = ds.batch(3)
    for batch in ds_batched:
        print(f"  Batch: {batch.numpy()}")

    print("\n--- shuffle ---")
    ds_shuffled = ds.shuffle(buffer_size=10, seed=42)
    first_run = list(ds_shuffled.take(5).as_numpy_iterator())
    print(f"Shuffled sample: {first_run}")

    print("\n--- repeat ---")
    ds_repeated = ds.take(3).repeat(2)
    print(f"Repeated: {list(ds_repeated.as_numpy_iterator())}")

    print("\n--- Chained Pipeline ---")
    pipeline = (
        tf.data.Dataset.range(12)
        .shuffle(12, seed=1)
        .map(lambda x: x + 1)
        .batch(4)
        .prefetch(tf.data.AUTOTUNE)
    )
    for batch in pipeline.take(2):
        print(f"  Batch: {batch.numpy()}")

    print("\n--- drop_remainder ---")
    ds_drop = tf.data.Dataset.range(10).batch(3, drop_remainder=True)
    for batch in ds_drop:
        print(f"  Batch: {batch.numpy()}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
