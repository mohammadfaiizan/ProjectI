# Visualization and Interpretability

## Table of Contents

1. [Activation Visualization](#1-activation-visualization)
2. [Feature Visualization](#2-feature-visualization)
3. [Grad-CAM](#3-grad-cam)
4. [Model Interpretability (SHAP/LIME)](#4-model-interpretability-shap-lime)

---

## 1. Activation Visualization

**Activation visualization** extracts intermediate layer outputs to understand what representations the model learns at each stage.

### Hook-Based Extraction

Use **tf.keras.Model** with multiple outputs to capture activations from named layers.

```python
layer_names = ['conv1', 'conv2', 'pool1']
outputs = [model.get_layer(name).output for name in layer_names]
intermediate_model = tf.keras.Model(inputs=model.input, outputs=outputs)
activations = intermediate_model(x)
```

### Activation Shapes

Each activation tensor has shape `(batch, height, width, channels)` for conv layers. Inspect shapes to understand feature map dimensions.

```python
for name, act in zip(layer_names, activations):
    print(f"{name}: {act.shape}")
```

### Activation Statistics

Compute min, max, mean per channel to detect dead neurons or saturation.

```python
channel_means = tf.reduce_mean(activation, axis=[0, 1, 2])
```

---

## 2. Feature Visualization

**Feature maps** and **filter visualization** reveal what patterns each layer responds to.

### Filter Weights

Extract and visualize conv layer kernel weights.

```python
weights = layer.get_weights()[0]  # shape: (H, W, C_in, C_out)
filter_0 = weights[:, :, :, 0]   # first filter
```

### Feature Map Extraction

Forward pass through a sub-model ending at a conv layer yields feature maps.

```python
feature_model = tf.keras.Model(inputs=model.input, outputs=conv_layer.output)
feature_maps = feature_model(x)  # shape: (batch, H, W, channels)
```

### Channel Statistics

Per-channel mean and std indicate activation strength.

```python
for i in range(num_channels):
    ch = feature_maps[0, :, :, i]
    print(f"Channel {i}: mean={ch.mean():.4f}, std={ch.std():.4f}")
```

---

## 3. Grad-CAM

**Gradient-weighted Class Activation Mapping (Grad-CAM)** highlights image regions that influence a model's class prediction.

### Algorithm

1. Forward pass to get conv output and final prediction.
2. Compute gradient of target class score w.r.t. conv output.
3. Global average pool gradients to get channel weights.
4. Weighted sum of conv channels + ReLU = class activation map.

```python
def grad_cam(model, img, layer_name, class_idx):
    conv_layer = model.get_layer(layer_name)
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[conv_layer.output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img)
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_output)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(weights[:, tf.newaxis, tf.newaxis, :] * conv_output, axis=-1)
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam
```

### Layer Selection

Use the last conv layer before global pooling for best spatial resolution. Common names: `conv2d`, `block5_conv3`, etc.

### Heatmap Overlay

Resize CAM to input image size and overlay for visualization.

---

## 4. Model Interpretability (SHAP/LIME)

**SHAP** (SHapley Additive exPlanations) and **LIME** (Local Interpretable Model-agnostic Explanations) provide feature-level explanations for model predictions.

### LIME Concept

- **Perturb** the input (e.g., add noise, mask features).
- **Predict** on perturbed samples.
- **Fit** a simple linear model to approximate the complex model locally.
- **Coefficients** = feature importance.

```python
def lime_perturbation(x, n_samples=50, sigma=0.1):
    perturbations = np.random.normal(0, sigma, (n_samples,) + x.shape)
    return x + perturbations
```

### SHAP Concept

- **Shapley values** from game theory: each feature's marginal contribution.
- **Kernel SHAP** and **DeepExplainer** approximate these for neural networks.
- **Gradient-based** methods (e.g., Integrated Gradients) are alternatives.

### Gradient-Based Feature Importance

Simple approximation: gradient magnitude w.r.t. input indicates sensitivity.

```python
with tf.GradientTape() as tape:
    tape.watch(x)
    pred = model(x)
grads = tape.gradient(pred, x)
importance = tf.reduce_mean(tf.abs(grads), axis=0)
```

### Baseline Comparison

Compare importance against a baseline (e.g., zeros, mean) for more meaningful attributions.

```python
importance = simple_feature_importance(model, x, baseline=tf.zeros_like(x))
```

---

## Summary

| Method | Input | Output | Use Case |
|--------|-------|--------|----------|
| Activation viz | Model, layer names | Intermediate tensors | Debug representations |
| Feature viz | Conv layer | Filters, feature maps | Understand filters |
| Grad-CAM | Image, layer, class | Heatmap | CNN interpretability |
| SHAP/LIME | Input, model | Feature importance | Explain predictions |
