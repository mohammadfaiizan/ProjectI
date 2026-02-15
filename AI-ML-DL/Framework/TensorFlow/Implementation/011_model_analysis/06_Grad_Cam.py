"""
Grad-CAM implementation for CNN interpretability.
"""
import tensorflow as tf
import numpy as np

def build_cnn_for_gradcam():
    inp = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inp)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(10, activation='softmax')(x)
    return tf.keras.Model(inp, out)

def grad_cam(model, img, layer_name, class_idx):
    conv_layer = model.get_layer(layer_name)
    grad_model = tf.keras.Model(inputs=model.input, outputs=[conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_output)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(weights[:, tf.newaxis, tf.newaxis, :] * conv_output, axis=-1)
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam

def main():
    print("=" * 50)
    print("Grad-CAM - CNN Interpretability")
    print("=" * 50)

    model = build_cnn_for_gradcam()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    x = tf.random.normal((1, 32, 32, 3))
    preds = model(x)
    class_idx = tf.argmax(preds[0]).numpy()
    print(f"Predicted class: {class_idx}")

    layer_name = [l.name for l in model.layers if 'conv' in l.name][-1]
    print(f"Target layer for Grad-CAM: {layer_name}")

    cam = grad_cam(model, x, layer_name, int(class_idx))
    print(f"Grad-CAM shape: {cam.shape}")
    print(f"Grad-CAM range: [{cam.numpy().min():.4f}, {cam.numpy().max():.4f}]")

    heatmap = cam[0].numpy()
    print(f"Heatmap spatial mean: {heatmap.mean():.4f}")

    print("\nGrad-CAM demo complete.")

if __name__ == "__main__":
    main()
