"""
Image preprocessing layers: Resizing, Rescaling, CenterCrop.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Image Preprocessing Layers")
    print("=" * 50)

    print("\n--- Resizing ---")
    img = tf.random.uniform((2, 100, 80, 3), 0, 255)
    resize_layer = tf.keras.layers.Resizing(64, 64)
    resized = resize_layer(img)
    print(f"Input shape: {img.shape}")
    print(f"Resized shape: {resized.shape}")

    print("\n--- Rescaling (image) ---")
    rescale = tf.keras.layers.Rescaling(1.0/255.0)
    scaled = rescale(img)
    print(f"Rescaled range: [{tf.reduce_min(scaled).numpy():.4f}, {tf.reduce_max(scaled).numpy():.4f}]")

    print("\n--- CenterCrop ---")
    center_crop = tf.keras.layers.CenterCrop(50, 40)
    cropped = center_crop(img)
    print(f"Cropped shape: {cropped.shape}")

    print("\n--- Pipeline: Resize -> Rescale -> CenterCrop ---")
    pipeline = tf.keras.Sequential([
        tf.keras.layers.Resizing(128, 128),
        tf.keras.layers.Rescaling(1.0/255.0),
        tf.keras.layers.CenterCrop(64, 64)
    ])
    out = pipeline(img)
    print(f"Pipeline output shape: {out.shape}")
    print(f"Pipeline output dtype: {out.dtype}")

    print("\n--- Resizing with interpolation ---")
    resize_bilinear = tf.keras.layers.Resizing(32, 32, interpolation="bilinear")
    out_bilinear = resize_bilinear(img)
    print(f"Bilinear resized shape: {out_bilinear.shape}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
