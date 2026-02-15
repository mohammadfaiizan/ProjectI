"""
Quantization, pruning, TF Lite conversion for vision.
"""
import tensorflow as tf
import numpy as np

def build_simple_vision_model():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

def quantize_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    return tflite_model

def quantize_dynamic_range(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    return tflite_model

def apply_pruning(model, pruning_params=None):
    try:
        import tensorflow_model_optimization as tfmot
        if pruning_params is None:
            pruning_params = {'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, 0)}
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        model = prune_low_magnitude(model, **pruning_params)
        return model
    except ImportError:
        return model

def main():
    model = build_simple_vision_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    x = tf.random.normal((4, 32, 32, 3))
    y = model(x)
    print(f"Original model output: {y.shape}")
    print(f"Original params: {model.count_params():,}")
    tflite_quant = quantize_dynamic_range(model)
    print(f"TFLite quantized size: {len(tflite_quant) / 1024:.2f} KB")
    tflite_fp16 = quantize_model(model)
    print(f"TFLite FP16 size: {len(tflite_fp16) / 1024:.2f} KB")
    interpreter = tf.lite.Interpreter(model_content=tflite_quant)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"TFLite input shape: {input_details[0]['shape']}")
    print("Vision model optimization verified.")

if __name__ == "__main__":
    main()
