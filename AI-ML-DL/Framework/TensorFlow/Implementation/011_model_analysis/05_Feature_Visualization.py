"""
Feature map visualization, filter visualization.
"""
import tensorflow as tf
import numpy as np

def build_cnn():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

def visualize_filters(layer, filter_idx=0):
    weights = layer.get_weights()[0]
    return weights[:, :, :, filter_idx]

def main():
    print("=" * 50)
    print("Feature Visualization - Filter and Feature Maps")
    print("=" * 50)

    model = build_cnn()
    conv_layer = model.layers[0]

    filters = visualize_filters(conv_layer, 0)
    print(f"Filter 0 shape: {filters.shape}")
    print(f"Filter 0 stats: min={filters.min():.4f}, max={filters.max():.4f}")

    intermediate = tf.keras.Model(inputs=model.input, outputs=model.layers[0].output)
    x = tf.random.normal((2, 32, 32, 3))
    feature_maps = intermediate(x)
    print(f"\nFeature maps shape: {feature_maps.shape}")

    fm = feature_maps[0, :, :, :4]
    print(f"First 4 channels spatial mean: {tf.reduce_mean(fm, axis=[0, 1]).numpy()}")

    for i in range(min(4, feature_maps.shape[-1])):
        ch = feature_maps[0, :, :, i].numpy()
        print(f"  Channel {i}: mean={ch.mean():.4f}, std={ch.std():.4f}")

    print("\nFeature visualization demo complete.")

if __name__ == "__main__":
    main()
