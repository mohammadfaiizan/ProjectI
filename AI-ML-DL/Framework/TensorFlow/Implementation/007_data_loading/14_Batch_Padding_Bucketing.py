"""
padded_batch, bucket_by_sequence_length.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Batch Padding and Bucketing")
    print("=" * 50)

    print("\n--- padded_batch ---")
    sequences = [
        tf.constant([1, 2, 3]),
        tf.constant([4, 5]),
        tf.constant([6, 7, 8, 9]),
        tf.constant([10])
    ]
    ds = tf.data.Dataset.from_tensor_slices(sequences)
    ds_padded = ds.padded_batch(
        batch_size=2,
        padded_shapes=[4],
        padding_values=0
    )
    for batch in ds_padded:
        print(f"  Padded batch: {batch.numpy()}")

    print("\n--- padded_batch with nested structure ---")
    ds_nested = tf.data.Dataset.from_tensor_slices({
        "ids": [[1, 2], [3, 4, 5], [6]],
        "labels": [0, 1, 0]
    })
    def to_tensors(x):
        return (
            tf.ragged.constant(x["ids"]),
            tf.constant(x["labels"])
        )
    ds_ragged = ds_nested.map(lambda x: (tf.ragged.constant(x["ids"]), x["labels"]))
    ds_pad_nested = ds_ragged.padded_batch(
        batch_size=2,
        padded_shapes=([None], []),
        padding_values=(0, -1)
    )
    for ids, labels in ds_pad_nested:
        print(f"  ids: {ids.to_tensor().numpy()}")
        print(f"  labels: {labels.numpy()}")

    print("\n--- bucket_by_sequence_length ---")
    ds_seq = tf.data.Dataset.from_tensor_slices((
        tf.ragged.constant([[1, 2], [3, 4, 5, 6], [7, 8], [9, 10, 11]]),
        tf.constant([0, 1, 0, 1])
    ))

    def element_length(x, y):
        return tf.cast(tf.shape(x)[0], tf.int64)

    ds_bucketed = ds_seq.apply(
        tf.data.experimental.bucket_by_sequence_length(
            element_length,
            bucket_boundaries=[2, 4],
            bucket_batch_sizes=[2, 2, 2],
            padded_shapes=([None], []),
            padding_values=(0, -1)
        )
    )
    for batch_ids, batch_labels in ds_bucketed:
        print(f"  Bucket batch ids: {batch_ids.to_tensor().numpy()}")
        print(f"  Bucket batch labels: {batch_labels.numpy()}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
