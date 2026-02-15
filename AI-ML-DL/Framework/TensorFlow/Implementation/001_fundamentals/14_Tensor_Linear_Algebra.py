"""
tf.linalg: matmul, det, inv, eigh, svd, solve, norm, trace, cholesky, qr
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Tensor Linear Algebra")
    print("=" * 50)
    
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    
    print("\n--- tf.linalg.matmul ---")
    m = tf.linalg.matmul(a, b)
    print(f"matmul(a, b):\n{m}")
    
    print("\n--- tf.linalg.det ---")
    det = tf.linalg.det(a)
    print(f"det(a): {det.numpy()}")
    
    print("\n--- tf.linalg.inv ---")
    inv = tf.linalg.inv(a)
    print(f"inv(a):\n{inv}")
    print(f"a @ inv(a):\n{tf.linalg.matmul(a, inv)}")
    
    print("\n--- tf.linalg.eigh ---")
    sym = tf.constant([[4.0, 1.0], [1.0, 3.0]])
    eigenvalues, eigenvectors = tf.linalg.eigh(sym)
    print(f"eigenvalues: {eigenvalues.numpy()}")
    print(f"eigenvectors:\n{eigenvectors}")
    
    print("\n--- tf.linalg.svd ---")
    u, s, v = tf.linalg.svd(a)
    print(f"SVD singular values: {s.numpy()}")
    
    print("\n--- tf.linalg.solve ---")
    A = tf.constant([[3.0, 1.0], [1.0, 2.0]])
    rhs = tf.constant([[9.0], [8.0]])
    sol = tf.linalg.solve(A, rhs)
    print(f"solve(A, b):\n{sol}")
    
    print("\n--- tf.linalg.norm ---")
    v = tf.constant([3.0, 4.0])
    n2 = tf.linalg.norm(v)
    print(f"L2 norm: {n2.numpy()}")
    n1 = tf.linalg.norm(v, ord=1)
    print(f"L1 norm: {n1.numpy()}")
    
    print("\n--- tf.linalg.trace ---")
    tr = tf.linalg.trace(a)
    print(f"trace(a): {tr.numpy()}")
    
    print("\n--- tf.linalg.cholesky ---")
    pos_def = tf.constant([[4.0, 2.0], [2.0, 3.0]])
    L = tf.linalg.cholesky(pos_def)
    print(f"cholesky:\n{L}")
    
    print("\n--- tf.linalg.qr ---")
    q, r = tf.linalg.qr(a)
    print(f"Q shape: {q.shape}, R shape: {r.shape}")
    
    print("\nVerification complete.")

if __name__ == "__main__":
    main()
