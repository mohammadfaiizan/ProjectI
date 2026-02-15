"""
@tf.function, ConcreteFunction, input_signature, tracing.
"""
import tensorflow as tf

@tf.function(input_signature=[tf.TensorSpec(shape=[None, 32], dtype=tf.float32)])
def predict_fn(x):
    return tf.nn.softmax(tf.reduce_sum(x, axis=1, keepdims=True) * 0.01)

def main():
    print("=" * 50)
    print("tf.function and ConcreteFunction")
    print("=" * 50)

    x = tf.random.normal((4, 32))
    out = predict_fn(x)
    print(f"Output shape: {out.shape}")

    concrete = predict_fn.get_concrete_function()
    print(f"ConcreteFunction: {concrete}")

    traced = concrete(tf.constant([[1.0] * 32]))
    print(f"Traced output shape: {traced.shape}")

    @tf.function
    def dynamic_fn(a, b):
        return a + b

    c1 = dynamic_fn.get_concrete_function(tf.constant(1.0), tf.constant(2.0))
    c2 = dynamic_fn.get_concrete_function(tf.constant([1.0, 2.0]), tf.constant([3.0, 4.0]))
    print(f"Concrete functions created: {c1.output_shape}, {c2.output_shape}")

    result = dynamic_fn(5.0, 10.0)
    print(f"Dynamic result: {result.numpy()}")

    print("tf.function demo complete.")

if __name__ == "__main__":
    main()
