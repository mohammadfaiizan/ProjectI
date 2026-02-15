"""
K-fold cross-validation, metrics computation.
"""
import tensorflow as tf
import numpy as np

def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(16,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

def kfold_cv(x, y, n_splits=3, epochs=2):
    n = len(x)
    indices = np.random.permutation(n)
    fold_size = n // n_splits
    scores = []

    for k in range(n_splits):
        val_start = k * fold_size
        val_end = (k + 1) * fold_size if k < n_splits - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_model()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=epochs, verbose=0)
        _, acc = model.evaluate(x_val, y_val, verbose=0)
        scores.append(acc)
    return scores

def main():
    print("=" * 50)
    print("Model Validation - K-Fold Cross-Validation")
    print("=" * 50)

    np.random.seed(42)
    x = np.random.randn(120, 16).astype(np.float32)
    y = np.random.randint(0, 3, 120)

    scores = kfold_cv(x, y, n_splits=3, epochs=2)
    print(f"Fold accuracies: {[f'{s:.4f}' for s in scores]}")
    print(f"Mean CV accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

    model = build_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    model.fit(x[:80], y[:80], epochs=2, verbose=0)
    results = model.evaluate(x[80:], y[80:], verbose=0)
    print(f"\nTest metrics: loss={results[0]:.4f}, acc={results[1]:.4f}, precision={results[2]:.4f}, recall={results[3]:.4f}")

    print("\nModel validation demo complete.")

if __name__ == "__main__":
    main()
