"""
kernel_regularizer, bias_regularizer, activity_regularizer, L1, L2, L1L2.
"""
import tensorflow as tf

def main():
    l1 = tf.keras.regularizers.L1(l1=0.01)
    l2 = tf.keras.regularizers.L2(l2=0.01)
    l1l2 = tf.keras.regularizers.L1L2(l1=0.01, l2=0.01)

    layer_l1 = tf.keras.layers.Dense(64, kernel_regularizer=l1, input_shape=(32,))
    layer_l1.build((None, 32))
    reg_loss = layer_l1.losses
    print(f"L1 kernel regularizer loss: {reg_loss[0].numpy():.4f}")

    layer_l2 = tf.keras.layers.Dense(64, kernel_regularizer=l2, bias_regularizer=l2, input_shape=(32,))
    layer_l2.build((None, 32))
    print(f"L2 kernel+bias regularizer: {len(layer_l2.losses)} losses")

    layer_l1l2 = tf.keras.layers.Dense(64, kernel_regularizer=l1l2, input_shape=(32,))
    layer_l1l2.build((None, 32))
    print(f"L1L2 regularizer: {layer_l1l2.losses[0].numpy():.4f}")

    activity_l2 = tf.keras.regularizers.L2(l2=0.001)
    layer_act = tf.keras.layers.Dense(64, activity_regularizer=activity_l2, input_shape=(32,))
    x = tf.random.normal((2, 32))
    _ = layer_act(x)
    print(f"Activity regularizer: {len(layer_act.losses)} losses")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, kernel_regularizer=l2, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, kernel_regularizer=l1l2, activation='relu'),
        tf.keras.layers.Dense(10, kernel_regularizer=l2, activation='softmax')
    ])
    model.build()
    _ = model(tf.random.normal((2, 784)))
    total_reg = sum(model.losses)
    print(f"Model total regularization: {total_reg.numpy():.4f}")

    conv_reg = tf.keras.layers.Conv2D(32, 3, kernel_regularizer=l2, input_shape=(28, 28, 1))
    conv_reg.build((None, 28, 28, 1))
    print(f"Conv2D with L2: {conv_reg.losses[0].numpy():.4f}")

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    x_train = tf.random.normal((10, 784))
    y_train = tf.random.uniform((10,), maxval=10, dtype=tf.int32)
    hist = model.fit(x_train, y_train, epochs=1, verbose=0)
    print(f"Model fit with regularization: loss={hist.history['loss'][0]:.4f}")
    print("Layer regularization verified.")

if __name__ == "__main__":
    main()
