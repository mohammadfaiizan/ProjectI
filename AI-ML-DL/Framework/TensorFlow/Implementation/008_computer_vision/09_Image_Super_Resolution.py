"""
Super-resolution network (SRCNN-style, sub-pixel conv).
"""
import tensorflow as tf

def build_srcnn_style(scale=2, input_shape=(32, 32, 3)):
    inp = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, 9, padding='same', activation='relu')(inp)
    x = tf.keras.layers.Conv2D(32, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(input_shape[-1] * scale * scale, 5, padding='same')(x)
    x = tf.nn.depth_to_space(x, scale)
    model = tf.keras.Model(inp, x)
    return model

def subpixel_upsample(x, scale=2):
    x = tf.keras.layers.Conv2D(x.shape[-1] * scale * scale, 3, padding='same')(x)
    return tf.nn.depth_to_space(x, scale)

def build_espcn(scale=2, input_shape=(32, 32, 3)):
    inp = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, 5, padding='same', activation='tanh')(inp)
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='tanh')(x)
    x = tf.keras.layers.Conv2D(input_shape[-1] * scale * scale, 3, padding='same')(x)
    x = tf.nn.depth_to_space(x, scale)
    model = tf.keras.Model(inp, x)
    return model

def main():
    model = build_srcnn_style(scale=2)
    x = tf.random.normal((2, 32, 32, 3))
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    espcn = build_espcn(scale=2)
    y_espcn = espcn(x)
    print(f"ESPCN output shape: {y_espcn.shape}")
    print(f"Params: {model.count_params():,}")
    print("Super-resolution models built.")

if __name__ == "__main__":
    main()
