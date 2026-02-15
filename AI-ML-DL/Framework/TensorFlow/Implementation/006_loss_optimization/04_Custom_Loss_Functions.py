"""
Custom loss as function and as tf.keras.losses.Loss subclass.
"""
import tensorflow as tf

def custom_mse_weighted(y_true, y_pred, weight_positive=2.0):
    squared = tf.square(y_true - y_pred)
    weights = tf.where(y_true > 0, weight_positive, 1.0)
    return tf.reduce_mean(weights * squared)

def main():
    y_true = tf.constant([[0.0, 1.0], [1.0, 0.0]], dtype=tf.float32)
    y_pred = tf.constant([[0.1, 0.9], [0.8, 0.2]], dtype=tf.float32)

    loss_fn = custom_mse_weighted(y_true, y_pred, weight_positive=2.0)
    print(f"Custom weighted MSE (func): {loss_fn.numpy():.4f}")

    class WeightedMSELoss(tf.keras.losses.Loss):
        def __init__(self, weight_positive=2.0, name="weighted_mse", **kwargs):
            super().__init__(name=name, **kwargs)
            self.weight_positive = weight_positive

        def call(self, y_true, y_pred):
            squared = tf.square(y_true - y_pred)
            weights = tf.where(y_true > 0, self.weight_positive, 1.0)
            return tf.reduce_mean(weights * squared)

        def get_config(self):
            config = super().get_config()
            config.update({"weight_positive": self.weight_positive})
            return config

    weighted_mse = WeightedMSELoss(weight_positive=2.0)
    loss_class = weighted_mse(y_true, y_pred)
    print(f"Custom weighted MSE (class): {loss_class.numpy():.4f}")

    config = weighted_mse.get_config()
    print(f"Loss config: {config}")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2, input_shape=(4,))
    ])
    model.compile(optimizer='adam', loss=WeightedMSELoss(weight_positive=2.0))
    x = tf.random.normal((2, 4))
    loss_train = model.train_on_batch(x, y_true)
    print(f"Model with custom loss: {loss_train:.4f}")

    class HuberLikeLoss(tf.keras.losses.Loss):
        def __init__(self, delta=1.0, **kwargs):
            super().__init__(**kwargs)
            self.delta = delta

        def call(self, y_true, y_pred):
            error = y_true - y_pred
            abs_error = tf.abs(error)
            quadratic = tf.minimum(abs_error, self.delta)
            linear = abs_error - quadratic
            return tf.reduce_mean(0.5 * tf.square(quadratic) + self.delta * linear)

    huber_like = HuberLikeLoss(delta=1.0)
    loss_huber = huber_like(y_true, y_pred)
    print(f"Custom Huber-like loss: {loss_huber.numpy():.4f}")
    print("Custom loss functions verified.")

if __name__ == "__main__":
    main()
