# TF Lite and Edge Deployment

## Table of Contents

1. [TensorFlow Lite Overview](#1-tensorflow-lite-overview)
2. [TFLite Conversion](#2-tflite-conversion)
3. [Quantization](#3-quantization)
4. [TensorFlow.js Export](#4-tensorflowjs-export)
5. [Edge Deployment with Coral](#5-edge-deployment-with-coral)
6. [Deployment Targets Comparison](#6-deployment-targets-comparison)

---

## 1. TensorFlow Lite Overview

**TensorFlow Lite (TFLite)** is a lightweight framework for deploying models on mobile, embedded, and edge devices. It provides smaller binary size, lower latency, and reduced memory compared to full TensorFlow.

### Key Characteristics

- **FlatBuffer** format: `.tflite` files are efficient and mmap-able
- **Optimized kernels**: Hand-tuned for ARM, x86, and specialized hardware
- **Delegates**: Offload computation to GPU, DSP, or Edge TPU
- **Interpreter API**: Simple load-and-invoke interface in C++, Java, Swift, Python

### When to Use TFLite

- Mobile apps (Android, iOS)
- Embedded systems (Raspberry Pi, microcontrollers)
- Edge devices with limited compute
- Offline inference requirements

---

## 2. TFLite Conversion

### From Keras Model

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(16,)),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)
```

### From SavedModel

```python
converter = tf.lite.TFLiteConverter.from_saved_model("/path/to/saved_model")
tflite_model = converter.convert()
```

**from_saved_model** is preferred when the model was exported with `tf.saved_model.save` and may have custom signatures or preprocessing.

### From ConcreteFunction

```python
@tf.function(input_signature=[tf.TensorSpec(shape=[1, 32], dtype=tf.float32)])
def serve(x):
    return model(x)

converter = tf.lite.TFLiteConverter.from_concrete_functions([serve.get_concrete_function()])
tflite_model = converter.convert()
```

### Supported Ops

Not all TensorFlow ops are supported in TFLite. Use `converter.target_spec.supported_ops` to control fallback:

- **TFLITE_BUILTINS**: TFLite-native ops only
- **TFLITE_BUILTINS, SELECT_TF_OPS**: Allow some TensorFlow ops (larger runtime)

---

## 3. Quantization

Quantization reduces model size and accelerates inference by using lower-precision weights and activations.

### Dynamic Range Quantization

Weights are quantized to int8; activations remain float32. Calibration is done at runtime.

```python
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

- **Pros**: Easy, no representative dataset, good size reduction
- **Cons**: Activations still float; limited speedup on CPU

### Float16 Quantization

Weights stored as float16; computation may still use float32 on some devices.

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
```

- **Pros**: Smaller model, good for GPU delegates
- **Cons**: Precision loss; not all hardware supports float16

### Full Integer Quantization

Weights and activations are int8. Requires a **representative dataset** for calibration.

```python
def representative_dataset():
    for _ in range(100):
        yield [tf.random.normal((1, 32)).numpy().astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
```

- **Pros**: Best size and speed on integer hardware (e.g., Edge TPU)
- **Cons**: Requires calibration; some models need SELECT_TF_OPS

### Quantization Comparison

| Type | Weights | Activations | Calibration | Use Case |
|------|---------|-------------|-------------|----------|
| None | float32 | float32 | No | Baseline |
| Dynamic | int8 | float32 | No | General mobile |
| Float16 | float16 | float16/32 | No | GPU, smaller size |
| Full Integer | int8 | int8 | Yes | Edge TPU, DSP |

---

## 4. TensorFlow.js Export

**TensorFlow.js** runs TensorFlow models in JavaScript (browser, Node.js). Useful for web-based inference without a backend.

### Conversion Options

1. **Keras to TF.js**: Save Keras model, then convert
2. **SavedModel to TF.js**: Use `tensorflowjs_converter` with `--input_format=tf_saved_model`
3. **Graph Model**: Use `--input_format=tf_graph_model` for frozen graphs

### CLI Conversion

```bash
tensorflowjs_converter --input_format=keras model.h5 output_dir/
tensorflowjs_converter --input_format=tf_saved_model saved_model/ output_dir/
```

### Python API

```python
import tensorflowjs as tfjs
tfjs.converters.convert_tf_saved_model(
    saved_model_dir,
    output_dir,
    quantization_dtype='uint8'  # optional
)
```

### Loading in JavaScript

```javascript
const model = await tf.loadGraphModel('https://example.com/model.json');
const result = model.predict(tf.tensor2d(inputData));
```

### Considerations

- Model size: Quantization reduces download time
- Browser memory: Large models may cause OOM
- Ops support: Not all TF ops are implemented in TF.js

---

## 5. Edge Deployment with Coral

**Coral** devices use the **Edge TPU**, a small ASIC optimized for int8 inference. Models must be compiled for the Edge TPU.

### Requirements

- Model must use **full integer quantization** (int8)
- Supported ops only (no SELECT_TF_OPS for Edge TPU)
- Use `edgetpu_compiler` to generate Edge TPU-compatible `.tflite`

### Conversion Workflow

1. Convert to TFLite with full integer quantization
2. Run `edgetpu_compiler` on the `.tflite` file
3. Deploy the output to Coral device

```bash
edgetpu_compiler model.tflite -o model_edgetpu.tflite
```

### Coral Devices

- **USB Accelerator**: Plugs into host (Linux, Mac, Windows)
- **Dev Board**: Standalone device with Edge TPU
- **System-on-Module**: For custom hardware

### Python Inference with Edge TPU

```python
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

interpreter = Interpreter(
    model_path="model_edgetpu.tflite",
    experimental_delegates=[load_delegate("libedgetpu.so.1")]
)
interpreter.allocate_tensors()
# ... set input, invoke, get output
```

### Op Compatibility

Not all TFLite ops run on the Edge TPU. Unsupported ops fall back to CPU. Check the [Coral compatibility list](https://coral.ai/docs/edgetpu/models-intro/) for supported architectures.

---

## 6. Deployment Targets Comparison

| Target | Format | Quantization | Latency | Use Case |
|--------|--------|--------------|---------|----------|
| TF Serving | SavedModel | No | Low (server) | Cloud, on-prem |
| TFLite CPU | .tflite | Optional | Medium | Mobile, embedded |
| TFLite GPU | .tflite | Float16 | Low | Mobile GPU |
| Edge TPU | .tflite | Full int8 | Very low | Coral devices |
| TF.js | JSON + bin | Optional | Variable | Browser, Node |

---

## Summary

- **TFLite** enables deployment on resource-constrained devices
- Conversion from **Keras** or **SavedModel**; use **input_signature** for fixed shapes
- **Quantization** reduces size and speeds inference; full integer required for Edge TPU
- **TensorFlow.js** targets web; conversion via `tensorflowjs_converter`
- **Coral Edge TPU** requires int8 quantization and `edgetpu_compiler` for optimal performance
