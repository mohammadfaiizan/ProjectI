"""
Keras Tuner: RandomSearch, Hyperband, BayesianOptimization, HyperModel.
"""
import tensorflow as tf
import os

def main():
    print("=" * 50)
    print("Keras Tuner")
    print("=" * 50)

    try:
        import keras_tuner as kt
        print(f"keras_tuner version: {kt.__version__}")
    except ImportError:
        print("keras_tuner not installed. Install: pip install keras-tuner")
        return

    def build_model(hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
        for i in range(hp.Int("num_layers", 1, 3)):
            model.add(tf.keras.layers.Dense(
                units=hp.Int(f"units_{i}", 32, 256, step=32),
                activation="relu"
            ))
        model.add(tf.keras.layers.Dense(10, activation="softmax"))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(hp.Float("lr", 1e-4, 1e-2, sampling="log")),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0

    print("\nRandomSearch:")
    tuner_rs = kt.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=2,
        executions_per_trial=1,
        directory="/tmp/kt_rs",
        project_name="mnist"
    )
    tuner_rs.search(x_train[:1000], y_train[:1000], epochs=1, validation_split=0.2, verbose=0)
    best = tuner_rs.get_best_hyperparameters(1)[0]
    print(f"  Best val_accuracy trial: {best.values}")

    print("\nHyperband:")
    tuner_hb = kt.Hyperband(
        build_model,
        objective="val_accuracy",
        max_epochs=2,
        factor=2,
        directory="/tmp/kt_hb",
        project_name="mnist"
    )
    tuner_hb.search(x_train[:1000], y_train[:1000], validation_split=0.2, verbose=0)
    print("  Hyperband search completed")

    print("\nBayesianOptimization:")
    tuner_bo = kt.BayesianOptimization(
        build_model,
        objective="val_accuracy",
        max_trials=2,
        directory="/tmp/kt_bo",
        project_name="mnist"
    )
    tuner_bo.search(x_train[:1000], y_train[:1000], epochs=1, validation_split=0.2, verbose=0)
    print("  Bayesian optimization search completed")

    print("\nKeras Tuner demo complete.")

if __name__ == "__main__":
    main()
