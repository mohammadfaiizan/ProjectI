"""
Misclassification analysis, per-class performance.
"""
import tensorflow as tf
import numpy as np

def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(16,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

def per_class_metrics(y_true, y_pred, num_classes=10):
    precision = []
    recall = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        pred_c = np.sum(y_pred == c)
        actual_c = np.sum(y_true == c)
        p = tp / pred_c if pred_c > 0 else 0
        r = tp / actual_c if actual_c > 0 else 0
        precision.append(p)
        recall.append(r)
    return precision, recall

def main():
    print("=" * 50)
    print("Error Analysis - Misclassification and Per-Class")
    print("=" * 50)

    model = build_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    x = np.random.randn(200, 16).astype(np.float32)
    y = np.random.randint(0, 10, 200)
    model.fit(x, y, epochs=3, verbose=0)

    y_pred = np.argmax(model.predict(x, verbose=0), axis=1)
    correct = y_pred == y
    misclassified = np.where(~correct)[0]
    print(f"Total samples: {len(y)}, Misclassified: {len(misclassified)}")
    print(f"Overall accuracy: {np.mean(correct):.4f}")

    precision, recall = per_class_metrics(y, y_pred, num_classes=10)
    print("\nPer-class precision (first 5):", [f"{p:.3f}" for p in precision[:5]])
    print("Per-class recall (first 5):", [f"{r:.3f}" for r in recall[:5]])

    worst_class = np.argmin(recall)
    print(f"\nWorst recall class: {worst_class} (recall={recall[worst_class]:.3f})")

    conf_matrix = tf.math.confusion_matrix(y, y_pred)
    print(f"Confusion matrix shape: {conf_matrix.shape}")

    print("\nError analysis demo complete.")

if __name__ == "__main__":
    main()
