# SavedModel Format and tf.function

## Table of Contents

1. [SavedModel Format](#1-savedmodel-format)
2. [Saving and Loading Models](#2-saving-and-loading-models)
3. [Signatures and Serving](#3-signatures-and-serving)
4. [tf.function and Graph Execution](#4-tffunction-and-graph-execution)
5. [ConcreteFunction and Tracing](#5-concretefunction-and-tracing)
6. [Input Signatures](#6-input-signatures)

---

## 1. SavedModel Format

The **SavedModel** format is TensorFlow's standard serialization format for complete models. It stores the model's computation graph, weights, and metadata in a directory structure that is portable across platforms and language bindings.

### Directory Structure

A SavedModel directory contains:

- **saved_model.pb**: Protocol buffer with the **MetaGraphDef** (graph structure, signatures, asset files)
- **variables/**: Checkpoint data (weights, optimizer state)
- **assets/**: Optional auxiliary files (vocabulary, lookup tables)

### Why SavedModel

**SavedModel** is preferred over `model.save()` with HDF5 for production because it:

- Preserves the computation graph for optimized inference
- Supports multiple **signatures** (different input/output configurations)
- Works with TF Serving, TF Lite, TF.js without retraining
- Handles custom layers and objects via `get_config`/`from_config`

---

## 2. Saving and Loading Models

### tf.saved_model.save

Use `tf.saved_model.save()` to export a Keras model or a module with `@tf.function`-decorated methods.

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

tf.saved_model.save(model, "/path/to/model")
```

For custom modules, you can specify which functions to expose as signatures:

```python
class MyModule(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 32], dtype=tf.float32)])
    def __call__(self, x):
        return self.model(x)

module = MyModule()
tf.saved_model.save(module, "/path/to/model", signatures={"serving_default": module.__call__})
```

### tf.saved_model.load

Loading returns a **Trackable** object (not necessarily a Keras model). Use `signatures` to run inference.

```python
loaded = tf.saved_model.load("/path/to/model")
infer = loaded.signatures["serving_default"]
result = infer(tf.constant(sample_input))
```

For Keras models, `tf.keras.models.load_model()` restores the full Keras object including custom layers and training configuration.

---

## 3. Signatures and Serving

**Signatures** define named entry points for inference. Each signature specifies input and output names and types.

### Default Signatures

Keras models get a default **serving_default** signature when saved. The input name is typically the first layer's input name; the output name is the last layer's output name.

### Custom Signatures

Define multiple signatures for different use cases (e.g., batch vs single, different preprocessing):

```python
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 32], dtype=tf.float32)
])
def serve(x):
    return model(x)

tf.saved_model.save(
    model,
    "/path/to/model",
    signatures={"serving_default": serve}
)
```

### Signature Structure

Signatures are stored in the MetaGraphDef. Each signature has:

- **inputs**: Map of name -> TensorSpec
- **outputs**: Map of name -> TensorSpec
- **method_name**: Optional method identifier

---

## 4. tf.function and Graph Execution

**tf.function** traces Python functions and compiles them into TensorFlow graphs. Graph execution is faster than eager execution because it avoids Python overhead and enables optimizations.

### Basic Usage

```python
@tf.function
def add(a, b):
    return a + b

result = add(tf.constant(1.0), tf.constant(2.0))
```

### When Tracing Occurs

Tracing happens when:

- The function is called with new **input shapes** or **dtypes**
- The function is called with new **Python values** (non-Tensor arguments)
- `tf.function` is called with `experimental_relaxed_shapes=False` and shapes change

### Retracing and Polymorphism

Each unique combination of argument types/shapes creates a new **ConcreteFunction**. Too many retraces can slow execution. Use **input_signature** to limit polymorphism.

---

## 5. ConcreteFunction and Tracing

A **ConcreteFunction** is a traced, executable graph. It is created when `tf.function` traces a function with specific inputs.

### Obtaining ConcreteFunctions

```python
@tf.function
def compute(x):
    return tf.reduce_sum(x)

concrete = compute.get_concrete_function(tf.TensorSpec(shape=[None, 10], dtype=tf.float32))
```

### input_signature

Providing `input_signature` creates a single ConcreteFunction and prevents retracing for compatible inputs:

```python
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 32], dtype=tf.float32)])
def predict(x):
    return model(x)
```

### Tracing Behavior

- **First call**: Traces the function, builds the graph, caches the ConcreteFunction
- **Subsequent calls**: Reuses the cached graph if inputs match the signature
- **New shapes**: Retraces if input_signature is not set or shapes fall outside the signature

---

## 6. Input Signatures

**Input signatures** constrain the types and shapes of inputs. They improve performance and enable export to formats that require fixed shapes (e.g., TF Lite).

### TensorSpec

```python
tf.TensorSpec(shape=[None, 32], dtype=tf.float32, name="input")
```

- **shape**: Use `None` for variable batch or sequence length
- **dtype**: Must match actual input dtype
- **name**: Optional; used in signature

### Multiple Inputs

```python
@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, 32], dtype=tf.float32),
    tf.TensorSpec(shape=[None], dtype=tf.int32)
])
def train_step(x, y):
    with tf.GradientTape() as tape:
        pred = model(x)
        loss = loss_fn(y, pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

### Signature for SavedModel

When saving, the input_signature becomes part of the exported signature. TF Serving and other runtimes use it to validate requests.

---

## Summary Table

| Concept | Purpose |
|---------|---------|
| SavedModel | Portable model format for production |
| tf.saved_model.save | Export model with graph and weights |
| tf.saved_model.load | Load model; use signatures for inference |
| Signatures | Named input/output contracts for serving |
| tf.function | Compile Python to TensorFlow graph |
| ConcreteFunction | Traced, cached graph for specific inputs |
| input_signature | Limit retracing, enable export |
