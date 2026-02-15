"""
Eager Execution vs Graph Mode, tf.function intro, numpy() method
"""
import tensorflow as tf
import numpy as np

def main():
    print("=" * 50)
    print("Eager Execution Basics")
    print("=" * 50)
    
    print("\n--- Eager mode (default) ---")
    x = tf.constant(3.0)
    y = tf.constant(4.0)
    z = x * y + 2.0
    print(f"x*y+2 = {z.numpy()}")
    print("Operations execute immediately in eager mode.")
    
    print("\n--- numpy() method ---")
    t = tf.constant([[1, 2], [3, 4]])
    arr = t.numpy()
    print(f"Tensor:\n{t}")
    print(f"As NumPy array:\n{arr}")
    print(f"Type: {type(arr)}")
    
    print("\n--- tf.function (graph mode) ---")
    @tf.function
    def add_and_multiply(a, b):
        s = a + b
        m = a * b
        return s, m
    
    out1, out2 = add_and_multiply(tf.constant(5.0), tf.constant(3.0))
    print(f"add_and_multiply(5, 3): sum={out1.numpy()}, product={out2.numpy()}")
    
    print("\n--- Mixing NumPy and TensorFlow ---")
    np_arr = np.array([1.0, 2.0, 3.0])
    tf_tensor = tf.constant(np_arr)
    result = tf_tensor * 2
    print(f"NumPy -> TF -> *2: {result.numpy()}")
    
    print("\nVerification complete.")

if __name__ == "__main__":
    main()
