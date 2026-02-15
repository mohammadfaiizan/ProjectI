"""
Data type conversions: tf.cast, numpy interop, Python type conversion.
"""
import tensorflow as tf
import numpy as np

def main():
    print("=" * 50)
    print("Data Type Conversions")
    print("=" * 50)

    print("\n--- tf.cast ---")
    x_int = tf.constant([1, 2, 3], dtype=tf.int32)
    x_float = tf.cast(x_int, tf.float32)
    print(f"int32 -> float32: {x_float.numpy()}")
    x_bool = tf.cast(tf.constant([0, 1, 2]), tf.bool)
    print(f"int -> bool: {x_bool.numpy()}")

    print("\n--- NumPy interop ---")
    np_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    tf_tensor = tf.convert_to_tensor(np_arr)
    print(f"NumPy to Tensor: {tf_tensor.dtype}, shape {tf_tensor.shape}")
    back_np = tf_tensor.numpy()
    print(f"Tensor to NumPy: {type(back_np)}, shape {back_np.shape}")

    print("\n--- Python list conversion ---")
    py_list = [[1, 2], [3, 4]]
    tf_from_list = tf.constant(py_list)
    print(f"List to Tensor: {tf_from_list.numpy()}")

    print("\n--- Dtype handling ---")
    mixed = tf.constant([1.5, 2.5])
    to_int = tf.cast(mixed, tf.int32)
    print(f"float -> int32 (truncate): {to_int.numpy()}")
    to_int64 = tf.cast(x_int, tf.int64)
    print(f"int32 -> int64: {to_int64.numpy()}")

    print("\n--- String conversion ---")
    str_tensor = tf.constant(["1.5", "2.5", "3.0"])
    float_from_str = tf.strings.to_number(str_tensor, out_type=tf.float32)
    print(f"String to float: {float_from_str.numpy()}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
