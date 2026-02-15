"""
Tensor Creation Methods: constant, zeros, ones, fill, range, linspace, eye
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Tensor Creation Methods")
    print("=" * 50)
    
    print("\n--- tf.constant ---")
    c = tf.constant([[1, 2], [3, 4]])
    print(f"constant:\n{c}")
    
    print("\n--- tf.zeros ---")
    z = tf.zeros([2, 3])
    print(f"zeros(2,3):\n{z}")
    
    print("\n--- tf.ones ---")
    o = tf.ones([3, 2])
    print(f"ones(3,2):\n{o}")
    
    print("\n--- tf.fill ---")
    f = tf.fill([2, 4], 7.0)
    print(f"fill([2,4], 7.0):\n{f}")
    
    print("\n--- tf.range ---")
    r1 = tf.range(10)
    r2 = tf.range(2, 10, 2)
    print(f"range(10): {r1.numpy()}")
    print(f"range(2,10,2): {r2.numpy()}")
    
    print("\n--- tf.linspace ---")
    ls = tf.linspace(0.0, 10.0, 5)
    print(f"linspace(0,10,5): {ls.numpy()}")
    
    print("\n--- tf.eye ---")
    e = tf.eye(3)
    print(f"eye(3):\n{e}")
    e2 = tf.eye(3, 4)
    print(f"eye(3,4):\n{e2}")
    
    print("\nVerification complete.")

if __name__ == "__main__":
    main()
