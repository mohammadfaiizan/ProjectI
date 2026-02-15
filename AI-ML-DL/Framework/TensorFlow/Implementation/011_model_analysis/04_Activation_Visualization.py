"""
Hook-based activation extraction using tf.keras.Model intermediates.
"""
import tensorflow as tf
import numpy as np

def build_model_with_named_layers():
    inp = tf.keras.Input(shape=(28, 28, 1), name='input')
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', name='conv1')(inp)
    x = tf.keras.layers.MaxPooling2D(2, name='pool1')(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', name='conv2')(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)
    out = tf.keras.layers.Dense(10, activation='softmax', name='output')(x)
    return tf.keras.Model(inp, out)

def get_intermediate_outputs(model, layer_names, x):
    outputs = []
    for name in layer_names:
        layer = model.get_layer(name)
        outputs.append(layer.output)
    intermediate_model = tf.keras.Model(inputs=model.input, outputs=outputs)
    return intermediate_model(x)

def main():
    print("=" * 50)
    print("Activation Visualization - Intermediate Outputs")
    print("=" * 50)

    model = build_model_with_named_layers()
    layer_names = ['conv1', 'pool1', 'conv2', 'gap']

    x = tf.random.normal((4, 28, 28, 1))
    activations = get_intermediate_outputs(model, layer_names, x)

    print("Activation shapes per layer:")
    for name, act in zip(layer_names, activations):
        print(f"  {name}: {act.shape}")

    conv1_act = activations[0]
    print(f"\nConv1 activation stats: min={conv1_act.numpy().min():.4f}, max={conv1_act.numpy().max():.4f}")
    print(f"Conv1 activation mean: {tf.reduce_mean(conv1_act).numpy():.4f}")

    channel_means = tf.reduce_mean(conv1_act, axis=[0, 1, 2]).numpy()
    print(f"Conv1 channel-wise means (first 8): {channel_means[:8]}")

    print("\nActivation visualization demo complete.")

if __name__ == "__main__":
    main()
