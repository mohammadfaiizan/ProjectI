"""
TimeDistributed, Wrapper, Lambda layer.
"""
import tensorflow as tf

def main():
    x_seq = tf.random.normal((2, 10, 32))

    dense = tf.keras.layers.Dense(64, activation='relu')
    td = tf.keras.layers.TimeDistributed(dense)
    out_td = td(x_seq)
    print(f"TimeDistributed Dense: {x_seq.shape} -> {out_td.shape}")

    conv = tf.keras.layers.Conv2D(32, 3, activation='relu')
    td_conv = tf.keras.layers.TimeDistributed(conv)
    x_video = tf.random.normal((2, 5, 28, 28, 1))
    out_td_conv = td_conv(x_video)
    print(f"TimeDistributed Conv2D: {x_video.shape} -> {out_td_conv.shape}")

    lambda_layer = tf.keras.layers.Lambda(lambda x: tf.square(x))
    out_lambda = lambda_layer(tf.constant([1.0, 2.0, 3.0]))
    print(f"Lambda square: {out_lambda.numpy()}")

    lambda_norm = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=-1))
    out_norm = lambda_norm(tf.random.normal((2, 5)))
    print(f"Lambda L2 norm shape: {out_norm.shape}")

    class CustomWrapper(tf.keras.layers.Wrapper):
        def __init__(self, layer, **kwargs):
            super().__init__(layer, **kwargs)

        def call(self, inputs, **kwargs):
            return self.layer(inputs, **kwargs)

        def compute_output_shape(self, input_shape):
            return self.layer.compute_output_shape(input_shape)

    wrapped = CustomWrapper(tf.keras.layers.Dense(32))
    wrapped.build((None, 64))
    out_wrapped = wrapped(tf.random.normal((2, 64)))
    print(f"Wrapper output: {out_wrapped.shape}")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10, 32)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation='relu')),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    out_model = model(x_seq)
    print(f"TimeDistributed + LSTM model: {out_model.shape}")

    double_lambda = tf.keras.layers.Lambda(lambda x: x * 2)
    out_double = double_lambda(tf.constant([1.0, 2.0]))
    print(f"Lambda double: {out_double.numpy()}")
    print("Container layers verified.")

if __name__ == "__main__":
    main()
