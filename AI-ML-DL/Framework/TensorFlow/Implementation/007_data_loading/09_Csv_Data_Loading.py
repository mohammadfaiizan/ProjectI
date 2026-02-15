"""
Loading CSV data with tf.data.experimental.make_csv_dataset.
"""
import tensorflow as tf
import tempfile
import os

def main():
    print("=" * 50)
    print("CSV Data Loading")
    print("=" * 50)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("feature1,feature2,label\n")
        for i in range(6):
            f.write(f"{i * 1.0},{i * 2.0},{i % 2}\n")
        csv_path = f.name

    try:
        print("\n--- make_csv_dataset ---")
        ds = tf.data.experimental.make_csv_dataset(
            csv_path,
            batch_size=2,
            label_name="label",
            num_epochs=1,
            shuffle=True,
            shuffle_seed=42
        )
        for batch in ds.take(2):
            features = batch[0]
            labels = batch[1]
            print(f"  feature1: {features['feature1'].numpy()}")
            print(f"  feature2: {features['feature2'].numpy()}")
            print(f"  labels: {labels.numpy()}")

        print("\n--- column defaults ---")
        ds_defaults = tf.data.experimental.make_csv_dataset(
            csv_path,
            batch_size=2,
            column_defaults=[tf.float32, tf.float32, tf.int32],
            label_name="label",
            header=True
        )
        for batch in ds_defaults.take(1):
            print(f"  Batch keys: {list(batch[0].keys())}")

        print("\n--- select_columns ---")
        ds_select = tf.data.experimental.make_csv_dataset(
            csv_path,
            batch_size=2,
            select_columns=["feature1", "label"],
            label_name="label"
        )
        for batch in ds_select.take(1):
            print(f"  Selected columns: {list(batch[0].keys())}")

    finally:
        os.unlink(csv_path)

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
