"""
SHAP/LIME concepts for TF models.
"""
import tensorflow as tf
import numpy as np

def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

def lime_style_perturbation(x, n_samples=50, sigma=0.1):
    perturbations = np.random.normal(0, sigma, (n_samples,) + x.shape)
    return x + perturbations.astype(np.float32)

def simple_feature_importance(model, x, baseline=None):
    if baseline is None:
        baseline = tf.zeros_like(x)
    with tf.GradientTape() as tape:
        tape.watch(x)
        pred = model(x)
    grads = tape.gradient(pred, x)
    return tf.reduce_mean(tf.abs(grads), axis=0)

def main():
    print("=" * 50)
    print("Model Interpretability - SHAP/LIME Concepts")
    print("=" * 50)

    model = build_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    x = tf.random.normal((1, 10))
    pred = model(x)
    print(f"Prediction shape: {pred.shape}")

    importance = simple_feature_importance(model, x)
    print(f"\nGradient-based feature importance: {importance.numpy()}")

    pert_x = lime_style_perturbation(x.numpy(), n_samples=20)
    pert_preds = model(pert_x)
    print(f"Perturbed predictions mean: {np.mean(pert_preds[:, 0]):.4f}")

    baseline = tf.zeros_like(x)
    imp_baseline = simple_feature_importance(model, x, baseline)
    print(f"Importance vs zero baseline: {imp_baseline.numpy()[:5]}...")

    print("\nModel interpretability demo complete.")

if __name__ == "__main__":
    main()
