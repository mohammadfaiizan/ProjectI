"""
Augmentation within tf.data pipeline using map.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Data Augmentation Pipeline")
    print("=" * 50)

    print("\n--- Augmentation in map ---")
    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((
        tf.random.uniform((4, 32, 32, 3), 0, 1),
        tf.constant([0, 1, 0, 1])
    ))
    ds_aug = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    for img, lbl in ds_aug.take(1):
        print(f"  Augmented shape: {img.shape}, label: {lbl.numpy()}")

    print("\n--- Keras preprocessing in map ---")
    resize = tf.keras.layers.Resizing(64, 64)
    rescale = tf.keras.layers.Rescaling(1.0 / 255.0)
    def preprocess(image, label):
        image = resize(image)
        image = rescale(image)
        return image, label

    ds_img = tf.data.Dataset.from_tensor_slices((
        tf.random.uniform((2, 128, 128, 3), 0, 255, dtype=tf.float32),
        tf.constant([0, 1])
    ))
    ds_prep = ds_img.map(preprocess)
    for img, _ in ds_prep.take(1):
        print(f"  Preprocessed shape: {img.shape}, range: [{tf.reduce_min(img).numpy():.4f}, {tf.reduce_max(img).numpy():.4f}]")

    print("\n--- Conditional augmentation (training vs eval) ---")
    def augment_if_train(image, label, is_training):
        if is_training:
            image = tf.image.random_contrast(image, 0.8, 1.2)
        return image, label

    is_train = tf.constant(True)
    ds_cond = ds.map(lambda x, y: augment_if_train(x, y, is_train))
    for img, _ in ds_cond.take(1):
        print(f"  Conditional aug shape: {img.shape}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
