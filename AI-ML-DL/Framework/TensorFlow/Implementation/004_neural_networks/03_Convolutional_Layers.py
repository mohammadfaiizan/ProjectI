"""
Conv1D, Conv2D, Conv3D, padding, strides, dilation_rate, groups.
"""
import tensorflow as tf

def main():
    x1d = tf.random.normal((2, 100, 32))
    conv1d = tf.keras.layers.Conv1D(64, 3, padding='same', strides=1)
    out1d = conv1d(x1d)
    print(f"Conv1D output: {out1d.shape}")

    conv1d_valid = tf.keras.layers.Conv1D(32, 5, padding='valid', strides=2)
    out1d_v = conv1d_valid(x1d)
    print(f"Conv1D valid stride=2: {out1d_v.shape}")

    x2d = tf.random.normal((2, 28, 28, 3))
    conv2d = tf.keras.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1))
    out2d = conv2d(x2d)
    print(f"Conv2D output: {out2d.shape}")

    conv2d_dilated = tf.keras.layers.Conv2D(32, 3, padding='same', dilation_rate=(2, 2))
    out2d_d = conv2d_dilated(x2d)
    print(f"Conv2D dilation_rate=2: {out2d_d.shape}")

    x2d_8ch = tf.random.normal((2, 28, 28, 8))
    conv2d_groups = tf.keras.layers.Conv2D(64, 3, padding='same', groups=4)
    out2d_g = conv2d_groups(x2d_8ch)
    print(f"Conv2D groups=4: {out2d_g.shape}")

    x3d = tf.random.normal((2, 10, 20, 20, 4))
    conv3d = tf.keras.layers.Conv3D(16, (2, 3, 3), padding='valid')
    out3d = conv3d(x3d)
    print(f"Conv3D output: {out3d.shape}")

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    out = model(tf.random.normal((2, 28, 28, 1)))
    print(f"CNN model output: {out.shape}")
    print("Convolutional layers verified.")

if __name__ == "__main__":
    main()
