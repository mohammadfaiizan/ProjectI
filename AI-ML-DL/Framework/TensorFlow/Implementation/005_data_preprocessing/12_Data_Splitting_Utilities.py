"""
Data splitting: Train/val/test with tf.data and manual methods.
"""
import tensorflow as tf
import numpy as np

def main():
    print("=" * 50)
    print("Data Splitting Utilities")
    print("=" * 50)

    n = 100
    X = tf.random.normal((n, 10))
    y = tf.random.uniform((n,), 0, 3, dtype=tf.int32)

    print("\n--- Manual split (70/15/15) ---")
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    print("\n--- tf.data.Dataset split ---")
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(100)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    ds_train = ds.take(train_size)
    ds_val = ds.skip(train_size).take(val_size)
    ds_test = ds.skip(train_size + val_size)
    print(f"Train batches: {len(list(ds_train))}")
    print(f"Val batches: {len(list(ds_val))}")
    print(f"Test batches: {len(list(ds_test))}")

    print("\n--- Dataset.batch ---")
    batched = ds_train.batch(16)
    for batch_x, batch_y in batched.take(2):
        print(f"Batch x shape: {batch_x.shape}, y shape: {batch_y.shape}")

    print("\n--- train_test_split style ---")
    indices = tf.range(n)
    shuffled = tf.random.shuffle(indices)
    split = int(0.8 * n)
    train_idx = shuffled[:split]
    test_idx = shuffled[split:]
    X_tr = tf.gather(X, train_idx)
    X_te = tf.gather(X, test_idx)
    print(f"Shuffled split - Train: {X_tr.shape[0]}, Test: {X_te.shape[0]}")

    print("\n--- Prefetch and cache ---")
    opt_ds = ds_train.batch(16).cache().prefetch(tf.data.AUTOTUNE)
    print(f"Optimized dataset created: {opt_ds}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
