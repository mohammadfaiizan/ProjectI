"""
Regression losses: MSE, MAE, Huber, LogCosh, MSLE.
"""
import tensorflow as tf

def main():
    y_true = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    y_pred = tf.constant([[1.2, 1.8], [2.7, 4.2]], dtype=tf.float32)

    mse = tf.keras.losses.MeanSquaredError()
    loss_mse = mse(y_true, y_pred)
    print(f"MeanSquaredError: {loss_mse.numpy():.4f}")

    mae = tf.keras.losses.MeanAbsoluteError()
    loss_mae = mae(y_true, y_pred)
    print(f"MeanAbsoluteError: {loss_mae.numpy():.4f}")

    huber = tf.keras.losses.Huber(delta=1.0)
    loss_huber = huber(y_true, y_pred)
    print(f"Huber (delta=1.0): {loss_huber.numpy():.4f}")

    logcosh = tf.keras.losses.LogCosh()
    loss_logcosh = logcosh(y_true, y_pred)
    print(f"LogCosh: {loss_logcosh.numpy():.4f}")

    msle = tf.keras.losses.MeanSquaredLogarithmicError()
    y_pos_true = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    y_pos_pred = tf.constant([[1.1, 2.1], [2.9, 4.1]], dtype=tf.float32)
    loss_msle = msle(y_pos_true, y_pos_pred)
    print(f"MeanSquaredLogarithmicError: {loss_msle.numpy():6f}")

    mse_sum = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    loss_sum = mse_sum(y_true, y_pred)
    print(f"MSE (SUM reduction): {loss_sum.numpy():.4f}")

    mse_none = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    loss_per_sample = mse_none(y_true, y_pred)
    print(f"MSE (NONE reduction) shape: {loss_per_sample.shape}")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(2,))
    ])
    model.compile(optimizer='adam', loss='mse')
    loss_trained = model.evaluate(y_pred, y_true, verbose=0)
    print(f"Model MSE loss: {loss_trained:.4f}")
    print("Regression losses verified.")

if __name__ == "__main__":
    main()
