"""
tf.keras.utils.text_dataset_from_directory.
"""
import tensorflow as tf
import tempfile
import os

def main():
    print("=" * 50)
    print("Text Data Loading")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        for cls in ["pos", "neg"]:
            d = os.path.join(tmpdir, cls)
            os.makedirs(d)
            texts = ["Great product", "Amazing quality"] if cls == "pos" else ["Bad item", "Poor service"]
            for i, t in enumerate(texts):
                with open(os.path.join(d, f"{i}.txt"), "w") as f:
                    f.write(t)

        print("\n--- text_dataset_from_directory ---")
        ds = tf.keras.utils.text_dataset_from_directory(
            tmpdir,
            labels="inferred",
            label_mode="int",
            batch_size=2,
            shuffle=True,
            seed=42
        )
        print(f"Class names: {ds.class_names}")
        for texts, labels in ds.take(1):
            print(f"  Batch texts: {[t.numpy().decode() for t in texts.numpy()]}")
            print(f"  Batch labels: {labels.numpy()}")

        print("\n--- max_length ---")
        ds_max = tf.keras.utils.text_dataset_from_directory(
            tmpdir,
            labels="inferred",
            batch_size=2,
            max_length=10
        )
        for texts, _ in ds_max.take(1):
            print(f"  Text tensor shape: {texts.shape}")

        print("\n--- validation_split ---")
        ds_train = tf.keras.utils.text_dataset_from_directory(
            tmpdir,
            labels="inferred",
            batch_size=2,
            validation_split=0.5,
            subset="training",
            seed=42
        )
        print(f"Training batches: {len(list(ds_train))}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
