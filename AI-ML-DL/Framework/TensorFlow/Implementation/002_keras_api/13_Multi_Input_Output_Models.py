"""
Models with multiple inputs and outputs.
"""
import tensorflow as tf
import numpy as np

def main():
    input_a = tf.keras.Input(shape=(32,), name='input_a')
    input_b = tf.keras.Input(shape=(32,), name='input_b')

    branch_a = tf.keras.layers.Dense(16, activation='relu')(input_a)
    branch_b = tf.keras.layers.Dense(16, activation='relu')(input_b)

    combined = tf.keras.layers.concatenate([branch_a, branch_b])
    hidden = tf.keras.layers.Dense(32, activation='relu')(combined)

    output_main = tf.keras.layers.Dense(10, activation='softmax', name='main_output')(hidden)
    output_aux = tf.keras.layers.Dense(1, activation='sigmoid', name='aux_output')(hidden)

    model = tf.keras.Model(inputs=[input_a, input_b], outputs=[output_main, output_aux])
    model.compile(
        optimizer='adam',
        loss={'main_output': 'categorical_crossentropy', 'aux_output': 'binary_crossentropy'},
        loss_weights={'main_output': 1.0, 'aux_output': 0.2}
    )
    print("Multi-input, multi-output model:")
    model.summary()

    x_a = tf.random.normal((4, 32))
    x_b = tf.random.normal((4, 32))
    y_main = tf.keras.utils.to_categorical(np.random.randint(0, 10, 4), 10)
    y_aux = np.random.randint(0, 2, (4, 1)).astype(np.float32)

    model.fit(
        [x_a, x_b],
        [y_main, y_aux],
        epochs=2,
        verbose=1
    )

    pred_main, pred_aux = model.predict([x_a, x_b], verbose=0)
    print(f"\nMain output shape: {pred_main.shape}")
    print(f"Aux output shape: {pred_aux.shape}")
    print("Multi-input/output models verified.")

if __name__ == "__main__":
    main()
