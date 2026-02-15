"""
Dataset.from_generator for custom data sources.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("from_generator")
    print("=" * 50)

    print("\n--- Simple Generator ---")
    def gen():
        for i in range(5):
            yield i * 2

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=tf.TensorSpec(shape=(), dtype=tf.int32)
    )
    print(f"Values: {list(ds.as_numpy_iterator())}")

    print("\n--- Generator Yielding Tuples ---")
    def gen_pairs():
        for i in range(3):
            yield (tf.constant([float(i), float(i + 1)]), tf.constant(i % 2))

    ds_pairs = tf.data.Dataset.from_generator(
        gen_pairs,
        output_signature=(
            tf.TensorSpec(shape=(2,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    for x, y in ds_pairs:
        print(f"  x: {x.numpy()}, y: {y.numpy()}")

    print("\n--- Generator with Dict ---")
    def gen_dict():
        for i in range(3):
            yield {"id": i, "value": float(i * 10)}

    ds_dict = tf.data.Dataset.from_generator(
        gen_dict,
        output_signature={
            "id": tf.TensorSpec(shape=(), dtype=tf.int32),
            "value": tf.TensorSpec(shape=(), dtype=tf.float32)
        }
    )
    for elem in ds_dict:
        print(f"  id: {elem['id'].numpy()}, value: {elem['value'].numpy()}")

    print("\n--- Infinite Generator ---")
    def infinite_gen():
        n = 0
        while True:
            yield n
            n += 1

    ds_inf = tf.data.Dataset.from_generator(
        infinite_gen,
        output_signature=tf.TensorSpec(shape=(), dtype=tf.int32)
    )
    limited = list(ds_inf.take(4).as_numpy_iterator())
    print(f"First 4: {limited}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
