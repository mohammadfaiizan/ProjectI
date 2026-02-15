"""
tf.io.TFRecordWriter, tf.train.Example, Feature, reading TFRecords.
"""
import tensorflow as tf
import tempfile
import os

def main():
    print("=" * 50)
    print("TFRecord Format")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "data.tfrecord")

        print("\n--- Writing TFRecords ---")
        with tf.io.TFRecordWriter(path) as writer:
            for i in range(3):
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
                            "value": tf.train.Feature(float_list=tf.train.FloatList(value=[float(i * 10)])),
                            "name": tf.train.Feature(bytes_list=tf.train.BytesList(value=[f"item_{i}".encode()]))
                        }
                    )
                )
                writer.write(example.SerializeToString())
        print(f"Wrote 3 records to {path}")

        print("\n--- Reading TFRecords ---")
        raw_ds = tf.data.TFRecordDataset(path)
        def parse_fn(serialized):
            features = {
                "id": tf.io.FixedLenFeature([], tf.int64),
                "value": tf.io.FixedLenFeature([], tf.float32),
                "name": tf.io.FixedLenFeature([], tf.string)
            }
            parsed = tf.io.parse_single_example(serialized, features)
            return parsed

        ds = raw_ds.map(parse_fn)
        for elem in ds:
            print(f"  id: {elem['id'].numpy()}, value: {elem['value'].numpy()}, name: {elem['name'].numpy().decode()}")

        print("\n--- Variable-Length Feature ---")
        path2 = os.path.join(tmpdir, "var.tfrecord")
        with tf.io.TFRecordWriter(path2) as writer:
            ex = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "ids": tf.train.Feature(int64_list=tf.train.Int64List(value=[1, 2, 3]))
                    }
                )
            )
            writer.write(ex.SerializeToString())

        def parse_var(serialized):
            return tf.io.parse_single_example(
                serialized,
                {"ids": tf.io.VarLenFeature(tf.int64)}
            )

        ds_var = tf.data.TFRecordDataset(path2).map(parse_var)
        for elem in ds_var:
            ids = tf.sparse.to_dense(elem["ids"])
            print(f"  ids: {ids.numpy()}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
