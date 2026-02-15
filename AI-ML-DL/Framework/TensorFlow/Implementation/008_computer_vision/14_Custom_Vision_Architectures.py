"""
Custom architectures (residual blocks, FPN, multi-scale).
"""
import tensorflow as tf

def residual_block(x, filters):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1)(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.Activation('relu')(x + shortcut)

def build_resnet_style(input_shape=(32, 32, 3), num_classes=10):
    inp = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(inp)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inp, out)

def fpn_block(c2, c3, c4, filters=256):
    p4 = tf.keras.layers.Conv2D(filters, 1)(c4)
    p3 = tf.keras.layers.Conv2D(filters, 1)(c3) + tf.keras.layers.UpSampling2D(2)(p4)
    p2 = tf.keras.layers.Conv2D(filters, 1)(c2) + tf.keras.layers.UpSampling2D(2)(p3)
    return p2, p3, p4

def build_multiscale(input_shape=(128, 128, 3)):
    inp = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(inp)
    c2 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(c2)
    c3 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(256, 3, strides=2, padding='same', activation='relu')(c3)
    c4 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    p2, p3, p4 = fpn_block(c2, c3, c4)
    return tf.keras.Model(inp, [p2, p3, p4])

def main():
    resnet = build_resnet_style()
    x = tf.random.normal((2, 32, 32, 3))
    y = resnet(x)
    print(f"ResNet-style output: {y.shape}")
    fpn_model = build_multiscale()
    p2, p3, p4 = fpn_model(tf.random.normal((2, 128, 128, 3)))
    print(f"FPN outputs: p2={p2.shape}, p3={p3.shape}, p4={p4.shape}")
    print(f"Custom architectures params: {resnet.count_params():,}")
    print("Custom vision architectures built.")

if __name__ == "__main__":
    main()
