"""
Parameter count, FLOPs estimation, layer-by-layer analysis.
"""
import tensorflow as tf

def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

def estimate_conv_flops(layer, input_shape):
    if not isinstance(layer, tf.keras.layers.Conv2D):
        return 0
    _, h, w, c_in = input_shape
    k = layer.kernel_size[0]
    c_out = layer.filters
    return 2 * h * w * k * k * c_in * c_out

def main():
    print("=" * 50)
    print("Model Complexity Analysis")
    print("=" * 50)

    model = build_model()
    total_params = model.count_params()
    print(f"Total parameters: {total_params:,}")

    print("\nLayer-by-layer analysis:")
    current_shape = (None, 32, 32, 3)
    total_flops = 0
    for layer in model.layers:
        params = layer.count_params()
        flops = estimate_conv_flops(layer, current_shape)
        total_flops += flops
        if hasattr(layer, 'output_shape') and layer.output_shape:
            current_shape = layer.output_shape
        print(f"  {layer.name}: params={params:,}, FLOPs~{flops:,}")

    print(f"\nEstimated Conv FLOPs (forward): {total_flops:,}")

    model.summary()
    print("\nModel complexity analysis demo complete.")

if __name__ == "__main__":
    main()
