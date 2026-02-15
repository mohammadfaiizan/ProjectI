"""
Custom C++ ops: tf.load_op_library, op registration concepts.
"""
import tensorflow as tf
import os

def main():
    print("=" * 50)
    print("Custom C++ Ops")
    print("=" * 50)

    print("\ntf.load_op_library:")
    print("  Loads a .so (Linux) or .dll (Windows) built from C++ op code")
    print("  Returns module with op functions")

    print("\nCustom op workflow:")
    print("  1. Define op in C++ with REGISTER_OP")
    print("  2. Implement kernel with REGISTER_KERNEL_BUILDER")
    print("  3. Build with bazel: tf_custom_op_library")
    print("  4. Load: op_module = tf.load_op_library('path/to/op.so')")
    print("  5. Call: result = op_module.my_custom_op(input)")

    print("\nExample C++ op registration:")
    cpp_example = """
REGISTER_OP("MyCustomOp")
    .Input("input: float")
    .Output("output: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });
"""
    print(cpp_example)

    print("\nPython usage (when op.so exists):")
    print("  op_module = tf.load_op_library('/path/to/my_op.so')")
    print("  output = op_module.my_custom_op(tf.constant([1.0, 2.0]))")

    print("\nGradient registration:")
    print("  Use @tf.RegisterGradient for custom gradients")
    print("  Or REGISTER_OP_GRADIENT in C++")

    print("\nWhen to use custom ops:")
    print("  - Performance-critical loops")
    print("  - Operations not in TF ops")
    print("  - Fused kernels (e.g., custom activation)")

    print("\nCustom Ops demo complete.")

if __name__ == "__main__":
    main()
