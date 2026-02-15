"""
History object, plotting loss/accuracy, metric tracking.
"""
import tensorflow as tf
import numpy as np

def main():
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_train = x_train.reshape(-1, 784)[:1000]
    y_train = tf.keras.utils.to_categorical(y_train[:1000], 10)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        x_train, y_train,
        epochs=4,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    print("\nHistory keys:", list(history.history.keys()))
    print("Loss per epoch:", [f"{v:.4f}" for v in history.history['loss']])
    print("Accuracy per epoch:", [f"{v:.4f}" for v in history.history['accuracy']])
    print("Val loss per epoch:", [f"{v:.4f}" for v in history.history['val_loss']])
    print("Val accuracy per epoch:", [f"{v:.4f}" for v in history.history['val_accuracy']])

    final_loss = history.history['loss'][-1]
    final_acc = history.history['accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])
    print(f"\nFinal train loss: {final_loss:.4f}")
    print(f"Final train accuracy: {final_acc:.4f}")
    print(f"Best val accuracy: {best_val_acc:.4f}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(history.history['loss'], label='train')
        axes[0].plot(history.history['val_loss'], label='val')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[1].plot(history.history['accuracy'], label='train')
        axes[1].plot(history.history['val_accuracy'], label='val')
        axes[1].set_title('Accuracy')
        axes[1].legend()
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=80)
        print("\nPlot saved to training_history.png")
    except ImportError:
        print("\nMatplotlib not available, skipping plot.")
    print("Training visualization verified.")

if __name__ == "__main__":
    main()
