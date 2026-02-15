"""
tf.debugging (assert_equal, assert_positive, check_numerics, enable_check_numerics).
"""
import tensorflow as tf
import numpy as np

def main():
    print("=" * 50)
    print("TensorFlow Debugging Tools")
    print("=" * 50)

    a = tf.constant([1, 2, 3])
    b = tf.constant([1, 2, 3])
    tf.debugging.assert_equal(a, b)
    print("assert_equal passed")

    x = tf.constant([1.0, 2.0, 3.0])
    tf.debugging.assert_positive(x)
    print("assert_positive passed")

    y = tf.constant([1.0, 2.0, np.nan])
    try:
        tf.debugging.check_numerics(y, "Tensor contains NaN/Inf")
    except tf.errors.InvalidArgumentError as e:
        print(f"check_numerics caught: {type(e).__name__}")

    t = tf.constant([1.0, 2.0, 3.0])
    checked = tf.debugging.check_numerics(t, "Valid tensor")
    print(f"check_numerics on valid tensor: {checked.numpy()}")

    print("\nNote: enable_check_numerics enables NaN/Inf checks for all ops in scope")
    print("TF debugging tools demo complete.")

if __name__ == "__main__":
    main()
