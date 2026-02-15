"""
Custom layer: get_config, from_config, serialization, compute_output_shape.
"""
import tensorflow as tf

class DenseSerializable(tf.keras.layers.Layer):
    def __init__(self, units, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation

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
        return tf.keras.activations.get(self.activation)(tf.matmul(inputs, self.kernel) + self.bias)

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units, 'activation': self.activation})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

def main():
    layer = DenseSerializable(64, activation='relu')
    x = tf.random.normal((2, 32))
    out = layer(x)
    print(f"Output shape: {out.shape}")

    config = layer.get_config()
    print(f"Config: {config}")

    restored = DenseSerializable.from_config(config)
    restored.build((None, 32))
    out_restored = restored(x)
    print(f"From config output shape: {out_restored.shape}")

    output_shape = layer.compute_output_shape((None, 32))
    print(f"compute_output_shape: {output_shape}")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32,)),
        DenseSerializable(64, activation='relu'),
        DenseSerializable(10, activation='softmax')
    ])
    model_config = model.get_config()
    model_restored = tf.keras.Sequential.from_config(model_config, custom_objects={'DenseSerializable': DenseSerializable})
    print(f"Model restored from config: {model_restored.layers[1].units}")

    json_str = tf.keras.utils.serialize_keras_object(layer)
    print(f"Serialized: {str(json_str)[:80]}...")
    deserialized = tf.keras.utils.deserialize_keras_object(json_str, custom_objects={'DenseSerializable': DenseSerializable})
    print(f"Deserialized type: {type(deserialized).__name__}")
    print("Custom layers advanced verified.")

if __name__ == "__main__":
    main()
