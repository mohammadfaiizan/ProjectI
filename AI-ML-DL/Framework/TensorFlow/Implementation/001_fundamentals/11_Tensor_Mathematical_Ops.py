"""
Element-wise ops, tf.math functions: sin, cos, exp, log, sqrt, abs, sign, clip_by_value
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Tensor Mathematical Operations")
    print("=" * 50)
    
    a = tf.constant([1.0, 2.0, 3.0])
    b = tf.constant([4.0, 5.0, 6.0])
    
    print("\n--- Element-wise ops ---")
    print(f"a + b: {(a + b).numpy()}")
    print(f"a - b: {(a - b).numpy()}")
    print(f"a * b: {(a * b).numpy()}")
    print(f"a / b: {(a / b).numpy()}")
    print(f"a ** 2: {(a ** 2).numpy()}")
    
    print("\n--- tf.math functions ---")
    x = tf.constant([0.0, 3.14159 / 2])
    print(f"sin: {tf.math.sin(x).numpy()}")
    print(f"cos: {tf.math.cos(x).numpy()}")
    
    y = tf.constant([1.0, 2.0, 3.0])
    print(f"exp: {tf.math.exp(y).numpy()}")
    print(f"log: {tf.math.log(y).numpy()}")
    print(f"sqrt: {tf.math.sqrt(y).numpy()}")
    
    z = tf.constant([-2.5, 0.0, 3.5])
    print(f"abs: {tf.math.abs(z).numpy()}")
    print(f"sign: {tf.math.sign(z).numpy()}")
    
    print("\n--- tf.clip_by_value ---")
    vals = tf.constant([0.5, 1.5, 2.5, 3.5])
    clipped = tf.clip_by_value(vals, 1.0, 3.0)
    print(f"clip_by_value(1, 3): {vals.numpy()} -> {clipped.numpy()}")
    
    print("\n--- Rounding ---")
    r = tf.constant([1.4, 1.6, 2.5])
    print(f"floor: {tf.math.floor(r).numpy()}")
    print(f"ceil: {tf.math.ceil(r).numpy()}")
    print(f"round: {tf.math.round(r).numpy()}")
    
    print("\nVerification complete.")

if __name__ == "__main__":
    main()
