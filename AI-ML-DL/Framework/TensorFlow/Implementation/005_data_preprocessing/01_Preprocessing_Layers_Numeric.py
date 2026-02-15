"""
Numeric preprocessing layers: Normalization, Discretization, Rescaling.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Numeric Preprocessing Layers")
    print("=" * 50)

    print("\n--- Normalization ---")
    data = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    norm_layer = tf.keras.layers.Normalization(axis=-1)
    norm_layer.adapt(data)
    normalized = norm_layer(data)
    print(f"Input shape: {data.shape}")
    print(f"Normalized mean: {tf.reduce_mean(normalized).numpy():.4f}")
    print(f"Normalized std: {tf.math.reduce_std(normalized).numpy():.4f}")

    print("\n--- Discretization ---")
    disc_layer = tf.keras.layers.Discretization(bin_boundaries=[0.0, 2.5, 5.0, 7.5])
    disc_out = disc_layer(tf.constant([1.0, 3.0, 6.0, 8.0]))
    print(f"Discretized [1,3,6,8]: {disc_out.numpy()}")

    print("\n--- Rescaling ---")
    rescale_layer = tf.keras.layers.Rescaling(scale=1.0/255.0, offset=0)
    img = tf.constant([[200.0, 150.0], [100.0, 50.0]])
    rescaled = rescale_layer(img)
    print(f"Rescaled (1/255): {rescaled.numpy()}")

    print("\n--- Normalization with adapt ---")
    train_data = tf.random.uniform((100, 5), 0, 100)
    norm_adapt = tf.keras.layers.Normalization()
    norm_adapt.adapt(train_data)
    sample = train_data[:3]
    out = norm_adapt(sample)
    print(f"Adapted norm output shape: {out.shape}")
    print(f"Sample normalized: {out[0].numpy()}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
