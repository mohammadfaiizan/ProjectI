"""
tf.keras.utils.image_dataset_from_directory.
"""
import tensorflow as tf
import tempfile
import os
import numpy as np

def main():
    print("=" * 50)
    print("Image Data Loading")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        for cls in ["cat", "dog"]:
            d = os.path.join(tmpdir, cls)
            os.makedirs(d)
            for i in range(2):
                img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                tf.keras.utils.save_img(os.path.join(d, f"{i}.png"), img)

        print("\n--- image_dataset_from_directory ---")
        ds = tf.keras.utils.image_dataset_from_directory(
            tmpdir,
            labels="inferred",
            label_mode="int",
            image_size=(32, 32),
            batch_size=2,
            shuffle=True,
            seed=42
        )
        print(f"Class names: {ds.class_names}")
        for images, labels in ds.take(1):
            print(f"  Batch images shape: {images.shape}")
            print(f"  Batch labels: {labels.numpy()}")

        print("\n--- label_mode categorical ---")
        ds_cat = tf.keras.utils.image_dataset_from_directory(
            tmpdir,
            labels="inferred",
            label_mode="categorical",
            image_size=(32, 32),
            batch_size=2
        )
        for _, labels in ds_cat.take(1):
            print(f"  Categorical labels shape: {labels.shape}")

        print("\n--- validation_split ---")
        ds_train = tf.keras.utils.image_dataset_from_directory(
            tmpdir,
            labels="inferred",
            image_size=(32, 32),
            batch_size=2,
            validation_split=0.5,
            subset="training",
            seed=42
        )
        ds_val = tf.keras.utils.image_dataset_from_directory(
            tmpdir,
            labels="inferred",
            image_size=(32, 32),
            batch_size=2,
            validation_split=0.5,
            subset="validation",
            seed=42
        )
        print(f"Training batches: {len(list(ds_train))}")
        print(f"Validation batches: {len(list(ds_val))}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
