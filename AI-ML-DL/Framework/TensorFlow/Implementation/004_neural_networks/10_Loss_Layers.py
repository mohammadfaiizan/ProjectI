"""
Loss functions as layers and standalone.
"""
import tensorflow as tf

def main():
    y_true = tf.constant([[0, 1, 0], [1, 0, 0]], dtype=tf.float32)
    y_pred = tf.constant([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05]], dtype=tf.float32)

    mse = tf.keras.losses.MeanSquaredError()
    loss_mse = mse(y_true, y_pred)
    print(f"MSE loss: {loss_mse.numpy():.4f}")

    mae = tf.keras.losses.MeanAbsoluteError()
    loss_mae = mae(y_true, y_pred)
    print(f"MAE loss: {loss_mae.numpy():.4f}")

    cce = tf.keras.losses.CategoricalCrossentropy()
    loss_cce = cce(y_true, y_pred)
    print(f"CategoricalCrossentropy: {loss_cce.numpy():.4f}")

    bce = tf.keras.losses.BinaryCrossentropy()
    y_bin_true = tf.constant([[0, 1], [1, 0]], dtype=tf.float32)
    y_bin_pred = tf.constant([[0.1, 0.9], [0.85, 0.15]], dtype=tf.float32)
    loss_bce = bce(y_bin_true, y_bin_pred)
    print(f"BinaryCrossentropy: {loss_bce.numpy():.4f}")

    huber = tf.keras.losses.Huber(delta=1.0)
    loss_huber = huber(y_true, y_pred)
    print(f"Huber loss: {loss_huber.numpy():.4f}")

    sparse_cce = tf.keras.losses.SparseCategoricalCrossentropy()
    y_sparse = tf.constant([1, 0])
    loss_sparse = sparse_cce(y_sparse, y_pred)
    print(f"SparseCategoricalCrossentropy: {loss_sparse.numpy():.4f}")

    class LossLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.loss_fn = tf.keras.losses.CategoricalCrossentropy()

        def call(self, inputs):
            y_true, y_pred = inputs
            loss = self.loss_fn(y_true, y_pred)
            self.add_loss(loss)
            return y_pred

    loss_layer = LossLayer()
    _ = loss_layer([y_true, y_pred])
    print(f"Loss layer losses: {len(loss_layer.losses)}")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='softmax', input_shape=(5,))
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    x = tf.random.normal((4, 5))
    out = model(x)
    loss = cce(y_true[:4], out)
    print(f"Model loss: {loss.numpy():.4f}")
    print("Loss layers verified.")

if __name__ == "__main__":
    main()
