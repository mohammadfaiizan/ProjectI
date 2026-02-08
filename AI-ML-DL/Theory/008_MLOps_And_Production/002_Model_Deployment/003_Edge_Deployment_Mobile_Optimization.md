# Edge Deployment and Mobile Optimization

## Table of Contents

1. [Introduction to Edge Computing](#introduction-to-edge-computing)
2. [Edge Computing Architecture](#edge-computing-architecture)
3. [TensorFlow Lite](#tensorflow-lite)
4. [CoreML for iOS](#coreml-for-ios)
5. [ONNX Runtime Mobile](#onnx-runtime-mobile)
6. [Model Quantization](#model-quantization)
7. [Model Pruning](#model-pruning)
8. [Hardware Acceleration](#hardware-acceleration)
9. [Mobile-Specific Optimizations](#mobile-specific-optimizations)
10. [Key Takeaways](#key-takeaways)

## Introduction to Edge Computing

Edge computing brings ML inference closer to data sources, reducing latency and bandwidth requirements. For mobile and edge devices, this enables:

- **Low Latency**: Inference without network round-trips
- **Privacy**: Data stays on device
- **Offline Capability**: Works without internet connection
- **Bandwidth Savings**: Reduced data transmission
- **Cost Efficiency**: Lower cloud compute costs

### Edge vs Cloud Trade-offs

| Aspect | Edge Deployment | Cloud Deployment |
|--------|----------------|------------------|
| Latency | Very low (<10ms) | Higher (50-500ms) |
| Privacy | High (on-device) | Lower (data sent) |
| Offline | Yes | No |
| Model Size | Limited | Unlimited |
| Updates | Requires app update | Instant |
| Cost | One-time (device) | Per-request |

### Use Cases

**Mobile Applications**:
- Image classification in camera apps
- Voice recognition for assistants
- Augmented reality filters
- Real-time translation

**IoT Devices**:
- Smart cameras for object detection
- Sensors for anomaly detection
- Wearables for health monitoring
- Industrial equipment monitoring

## Edge Computing Architecture

### Architecture Patterns

```
┌─────────────┐
│   Mobile    │
│   Device    │
│             │
│  ┌────────┐│
│  │  Model ││
│  │  (Edge)││
│  └────┬───┘│
│       │    │
│  ┌────▼───┐│
│  │Inference││
│  └────────┘│
└─────────────┘
       │
       │ (Optional sync)
       ▼
┌─────────────┐
│    Cloud    │
│   (Training)│
└─────────────┘
```

### Hybrid Approach

```
┌─────────────┐
│   Device    │
│             │
│  ┌────────┐│     ┌─────────────┐
│  │ Simple ││────▶│   Cloud     │
│  │ Model  ││     │  (Complex)  │
│  └────────┘│     └─────────────┘
│             │
│  ┌────────┐│
│  │Complex ││
│  │ Model  ││
│  └────────┘│
└─────────────┘
```

## TensorFlow Lite

TensorFlow Lite is Google's framework for deploying models on mobile and edge devices.

### Model Conversion

```python
import tensorflow as tf

# Load SavedModel
model = tf.saved_model.load('saved_model')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')

# Optimize
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert
tflite_model = converter.convert()

# Save
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Quantization

```python
# Post-training quantization
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Float16 quantization
converter.target_spec.supported_types = [tf.float16]

# Integer quantization
def representative_dataset():
    for i in range(100):
        yield [np.random.randn(1, 224, 224, 3).astype(np.float32)]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()
```

### Android Integration

```java
import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

// Load model
FileInputStream inputStream = new FileInputStream("model.tflite");
FileChannel fileChannel = inputStream.getChannel();
MappedByteBuffer modelBuffer = fileChannel.map(
    FileChannel.MapMode.READ_ONLY, 0, fileChannel.size()
);

// Create interpreter
Interpreter interpreter = new Interpreter(modelBuffer);

// Prepare input
float[][] input = new float[1][224 * 224 * 3];
// ... populate input ...

// Prepare output
float[][] output = new float[1][1000];

// Run inference
interpreter.run(input, output);

// Get results
float[] predictions = output[0];
```

### iOS Integration

```swift
import TensorFlowLite

// Load model
guard let modelPath = Bundle.main.path(forResource: "model", ofType: "tflite") else {
    fatalError("Model not found")
}

var interpreter: Interpreter
do {
    interpreter = try Interpreter(modelPath: modelPath)
    try interpreter.allocateTensors()
} catch {
    fatalError("Failed to create interpreter: \(error)")
}

// Prepare input
let inputData: Data = // ... prepare input data ...
try interpreter.copy(inputData, toInputAt: 0)

// Run inference
try interpreter.invoke()

// Get output
let outputTensor = try interpreter.output(at: 0)
let results = outputTensor.data
```

### GPU Delegation

```python
# Enable GPU delegate
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False

# Android GPU delegate
import org.tensorflow.lite.gpu.GpuDelegate;

GpuDelegate delegate = new GpuDelegate();
Interpreter.Options options = new Interpreter.Options();
options.addDelegate(delegate);
Interpreter interpreter = new Interpreter(modelBuffer, options);
```

## CoreML for iOS

CoreML is Apple's framework for on-device ML inference.

### Model Conversion

```python
import coremltools as ct

# Convert from TensorFlow
model = ct.convert(
    'model.pb',
    source='tensorflow',
    inputs=[ct.TensorType(name='input', shape=(1, 224, 224, 3))],
    outputs=[ct.TensorType(name='output')]
)

# Add metadata
model.author = 'Your Name'
model.short_description = 'Image Classification Model'
model.version = '1.0'

# Save
model.save('model.mlmodel')
```

### Quantization

```python
# Quantize model
quantized_model = ct.models.neural_network.quantization_utils.quantize_weights(
    model,
    nbits=8
)
quantized_model.save('model_quantized.mlmodel')

# Or use quantization during conversion
model = ct.convert(
    'model.pb',
    source='tensorflow',
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.iOS13
)
```

### Swift Integration

```swift
import CoreML
import Vision

// Load model
guard let modelURL = Bundle.main.url(forResource: "model", withExtension: "mlmodelc") else {
    fatalError("Model not found")
}

guard let model = try? MLModel(contentsOf: modelURL) else {
    fatalError("Failed to load model")
}

// Create prediction request
let request = VNCoreMLRequest(model: try! VNCoreMLModel(for: model)) { request, error in
    guard let results = request.results as? [VNClassificationObservation] else {
        return
    }
    // Process results
}

// Perform request
let handler = VNImageRequestHandler(ciImage: inputImage)
try? handler.perform([request])
```

### CoreML Tools

```python
# Inspect model
model = ct.models.MLModel('model.mlmodel')
print(model.input_description)
print(model.output_description)

# Evaluate model
metrics = model.evaluate(test_data)
print(f"Accuracy: {metrics['accuracy']}")

# Update model
updated_model = ct.models.neural_network.update(
    model,
    new_training_data=train_data,
    new_validation_data=val_data
)
```

## ONNX Runtime Mobile

ONNX Runtime provides cross-platform inference for ONNX models.

### Model Conversion

```python
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Convert to ONNX (from PyTorch)
import torch
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output']
)

# Quantize
quantize_dynamic(
    "model.onnx",
    "model_quantized.onnx",
    weight_type=QuantType.QUInt8
)
```

### Android Integration

```java
import ai.onnxruntime.*;

// Load model
OrtEnvironment env = OrtEnvironment.getEnvironment();
OrtSession session = env.createSession("model.onnx", new OrtSession.SessionOptions());

// Prepare input
OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputData);
Map<String, OnnxTensor> inputs = Collections.singletonMap("input", inputTensor);

// Run inference
OrtSession.Result result = session.run(inputs);

// Get output
OnnxValue output = result.get(0);
float[][] outputData = (float[][]) output.getValue();
```

### iOS Integration

```swift
import ONNXRuntime

// Load model
let ortEnv = try ORTEnv(loggingLevel: .warning)
let ortSession = try ORTSession(env: ortEnv, modelPath: "model.onnx", sessionOptions: nil)

// Prepare input
let inputData = // ... prepare input data ...
let inputTensor = try ORTValue(tensorData: inputData, elementType: .float)

// Run inference
let outputs = try ortSession.run(withInputs: ["input": inputTensor],
                                 outputNames: ["output"],
                                 runOptions: nil)

// Get output
let outputTensor = outputs["output"]
let results = try outputTensor.tensorData() as [Float]
```

## Model Quantization

Quantization reduces model size and improves inference speed by using lower precision.

### Post-Training Quantization

```python
import tensorflow as tf

# Float32 to Float16
converter = tf.lite.TFLiteConverter.from_saved_model('model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

# Float32 to Int8
def representative_dataset():
    for _ in range(100):
        yield [np.random.randn(1, 224, 224, 3).astype(np.float32)]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
```

### Quantization-Aware Training

```python
import tensorflow_model_optimization as tfmot

# Apply quantization to layers
quantize_model = tfmot.quantization.keras.quantize_model

# Quantize entire model
q_aware_model = quantize_model(model)

# Train quantized model
q_aware_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
q_aware_model.fit(train_data, train_labels, epochs=10)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

### Dynamic Quantization

```python
import torch
import torch.quantization

# Prepare model
model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Quantize
torch.quantization.prepare(model, inplace=True)
# ... calibration ...
torch.quantization.convert(model, inplace=True)

# Export
torch.onnx.export(model, dummy_input, "model_quantized.onnx")
```

## Model Pruning

Pruning removes unnecessary connections to reduce model size.

### Magnitude-Based Pruning

```python
import tensorflow_model_optimization as tfmot

# Prune model
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.50,
        final_sparsity=0.90,
        begin_step=0,
        end_step=1000
    )
}

pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
    model,
    **pruning_params
)

# Train pruned model
pruned_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
pruned_model.fit(train_data, train_labels, epochs=10)

# Strip pruning wrappers
final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
```

### Structured Pruning

```python
import torch.nn.utils.prune as prune

# Prune convolutional layers
for module in model.modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.ln_structured(module, name='weight', amount=0.5, n=2, dim=0)
        prune.remove(module, 'weight')

# Export pruned model
torch.onnx.export(model, dummy_input, "model_pruned.onnx")
```

## Hardware Acceleration

### Neural Processing Units (NPUs)

```python
# Qualcomm Snapdragon Neural Processing SDK
import qti.aisw

# Convert and optimize for Snapdragon
converter = qti.aisw.Converter()
converter.convert('model.onnx', 'model.dlc')
converter.optimize('model.dlc', target_chips=['snapdragon_888'])
```

### Apple Neural Engine

```python
import coremltools as ct

# Optimize for Neural Engine
model = ct.convert(
    'model.pb',
    source='tensorflow',
    compute_units=ct.ComputeUnit.ALL,  # Uses Neural Engine
    minimum_deployment_target=ct.target.iOS13
)
```

### Google Edge TPU

```python
import tensorflow as tf

# Compile for Edge TPU
converter = tf.lite.TFLiteConverter.from_saved_model('model')
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.target_edgetpu = True
tflite_model = converter.convert()
```

## Mobile-Specific Optimizations

### Model Size Reduction

```python
# Combine optimizations
converter = tf.lite.TFLiteConverter.from_saved_model('model')

# Quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Pruning (applied during training)
# Quantization-aware training

tflite_model = converter.convert()
```

### Memory Optimization

```python
# Reduce memory footprint
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Use memory mapping
interpreter = tf.lite.Interpreter(
    model_path='model.tflite',
    num_threads=4  # Optimize for device
)
```

### Power Optimization

```python
# Optimize for battery life
converter = tf.lite.TFLiteConverter.from_saved_model('model')
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
```

### Progressive Loading

```swift
// Load model asynchronously
DispatchQueue.global(qos: .userInitiated).async {
    guard let model = try? MLModel(contentsOf: modelURL) else {
        return
    }
    
    DispatchQueue.main.async {
        self.model = model
        self.isModelReady = true
    }
}
```

## Key Takeaways

- Edge computing enables low-latency, privacy-preserving ML inference on devices
- TensorFlow Lite provides optimized inference for Android and iOS devices
- CoreML offers native iOS integration with Apple's Neural Engine
- ONNX Runtime Mobile enables cross-platform deployment
- Model quantization reduces size and improves speed with minimal accuracy loss
- Model pruning removes unnecessary connections to reduce model complexity
- Hardware acceleration (NPUs, Neural Engine, Edge TPU) significantly improves performance
- Mobile-specific optimizations balance model size, memory usage, and power consumption
- Progressive loading and async inference improve user experience
- Edge deployment requires careful consideration of model size, accuracy, and device capabilities
