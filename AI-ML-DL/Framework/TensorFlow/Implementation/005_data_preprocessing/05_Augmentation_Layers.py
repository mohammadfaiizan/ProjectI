"""
Augmentation layers: RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomCrop.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Augmentation Layers")
    print("=" * 50)

    tf.random.set_seed(42)
    img = tf.random.uniform((2, 64, 64, 3), 0, 1)

    print("\n--- RandomFlip ---")
    flip_layer = tf.keras.layers.RandomFlip(mode="horizontal")
    flipped = flip_layer(img)
    print(f"Flipped shape: {flipped.shape}")

    print("\n--- RandomRotation ---")
    rot_layer = tf.keras.layers.RandomRotation(0.2)
    rotated = rot_layer(img)
    print(f"Rotated shape: {rotated.shape}")

    print("\n--- RandomZoom ---")
    zoom_layer = tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2)
    zoomed = zoom_layer(img)
    print(f"Zoomed shape: {zoomed.shape}")

    print("\n--- RandomContrast ---")
    contrast_layer = tf.keras.layers.RandomContrast(0.3)
    contrasted = contrast_layer(img)
    print(f"Contrasted shape: {contrasted.shape}")
    print(f"Contrasted range: [{tf.reduce_min(contrasted).numpy():.4f}, {tf.reduce_max(contrasted).numpy():.4f}]")

    print("\n--- RandomCrop ---")
    crop_layer = tf.keras.layers.RandomCrop(32, 32)
    cropped = crop_layer(img)
    print(f"Cropped shape: {cropped.shape}")

    print("\n--- Combined augmentation ---")
    aug_pipeline = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.2)
    ])
    aug_out = aug_pipeline(img)
    print(f"Augmented output shape: {aug_out.shape}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
