"""
Custom layer with build() and call(), __init__, trainable weights.
"""
import tensorflow as tf

class DenseCustom(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        x = tf.matmul(inputs, self.kernel) + self.bias
        return self.activation(x)

def main():
    layer = DenseCustom(64, activation='relu')
    x = tf.random.normal((2, 32))
    out = layer(x)
    print(f"Custom Dense output: {out.shape}")

    print(f"Trainable weights: {[w.name for w in layer.trainable_weights]}")

    class ScaleLayer(tf.keras.layers.Layer):
        def __init__(self, scale=1.0, **kwargs):
            super().__init__(**kwargs)
            self.scale = scale

        def build(self, input_shape):
            self.scale_var = self.add_weight(
                name='scale_var',
                shape=(),
                initializer=tf.keras.initializers.Constant(self.scale),
                trainable=True
            )
            super().build(input_shape)

        def call(self, inputs):
            return inputs * self.scale_var

    scale_layer = ScaleLayer(scale=2.0)
    out_scale = scale_layer(x)
    print(f"ScaleLayer output shape: {out_scale.shape}")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32,)),
        DenseCustom(64, activation='relu'),
        DenseCustom(10, activation='softmax')
    ])
    out_model = model(x)
    print(f"Model with custom layers: {out_model.shape}")
    model.summary()
    print("Custom layers basic verified.")

if __name__ == "__main__":
    main()
