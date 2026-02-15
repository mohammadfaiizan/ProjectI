"""
Semantic segmentation: pixel-wise classification with U-Net style architecture.
"""
import tensorflow as tf

def conv_block(x, filters, name_prefix):
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu', name=f"{name_prefix}_conv1")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu', name=f"{name_prefix}_conv2")(x)
    return x

def build_unet_style_segmentation(input_shape=(128, 128, 3), num_classes=21):
    inp = tf.keras.layers.Input(shape=input_shape)
    c1 = conv_block(inp, 32, "enc1")
    p1 = tf.keras.layers.MaxPooling2D(2)(c1)
    c2 = conv_block(p1, 64, "enc2")
    p2 = tf.keras.layers.MaxPooling2D(2)(c2)
    c3 = conv_block(p2, 128, "enc3")
    p3 = tf.keras.layers.MaxPooling2D(2)(c3)
    b = conv_block(p3, 256, "bottleneck")
    u3 = tf.keras.layers.UpSampling2D(2)(b)
    u3 = tf.keras.layers.Concatenate()([u3, c3])
    u3 = conv_block(u3, 128, "dec3")
    u2 = tf.keras.layers.UpSampling2D(2)(u3)
    u2 = tf.keras.layers.Concatenate()([u2, c2])
    u2 = conv_block(u2, 64, "dec2")
    u1 = tf.keras.layers.UpSampling2D(2)(u2)
    u1 = tf.keras.layers.Concatenate()([u1, c1])
    u1 = conv_block(u1, 32, "dec1")
    out = tf.keras.layers.Conv2D(num_classes, 1, activation='softmax', name="output")(u1)
    model = tf.keras.Model(inp, out)
    return model

def compute_dice(y_true, y_pred, num_classes, smooth=1e-6):
    dice = []
    for c in range(num_classes):
        tc = y_true[..., c]
        pc = y_pred[..., c]
        inter = tf.reduce_sum(tc * pc)
        union = tf.reduce_sum(tc) + tf.reduce_sum(pc)
        dice.append((2 * inter + smooth) / (union + smooth))
    return tf.reduce_mean(dice)

def main():
    model = build_unet_style_segmentation()
    x = tf.random.normal((2, 128, 128, 3))
    out = model(x)
    print(f"Segmentation output shape: {out.shape}")
    print(f"Expected: (2, 128, 128, 21)")
    y_true = tf.random.uniform((2, 128, 128, 21), 0, 1)
    y_pred = out
    dice = compute_dice(y_true, y_pred, 21)
    print(f"Dice score (sample): {dice.numpy():.4f}")
    print("Semantic segmentation model built.")

if __name__ == "__main__":
    main()
