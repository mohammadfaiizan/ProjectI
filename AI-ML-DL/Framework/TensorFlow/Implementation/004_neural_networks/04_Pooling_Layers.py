"""
MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D.
"""
import tensorflow as tf

def main():
    x = tf.random.normal((2, 28, 28, 64))

    max_pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
    out_max = max_pool(x)
    print(f"MaxPooling2D: {x.shape} -> {out_max.shape}")

    avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2)
    out_avg = avg_pool(x)
    print(f"AveragePooling2D: {x.shape} -> {out_avg.shape}")

    global_max = tf.keras.layers.GlobalMaxPooling2D()
    out_gmax = global_max(x)
    print(f"GlobalMaxPooling2D: {x.shape} -> {out_gmax.shape}")

    global_avg = tf.keras.layers.GlobalAveragePooling2D()
    out_gavg = global_avg(x)
    print(f"GlobalAveragePooling2D: {x.shape} -> {out_gavg.shape}")

    max_pool_3 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')
    out_3 = max_pool_3(x)
    print(f"MaxPooling2D 3x3 same: {out_3.shape}")

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    out = model(tf.random.normal((2, 28, 28, 1)))
    print(f"Model output: {out.shape}")

    max_pool1d = tf.keras.layers.MaxPooling1D(pool_size=2)
    x1d = tf.random.normal((2, 100, 32))
    out1d = max_pool1d(x1d)
    print(f"MaxPooling1D: {out1d.shape}")
    print("Pooling layers verified.")

if __name__ == "__main__":
    main()
