"""
Uncertainty sampling, query strategies, pool-based.
"""
import tensorflow as tf
import numpy as np

def main():
    print("=" * 50)
    print("Active Learning - Uncertainty Sampling")
    print("=" * 50)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(16,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    pool_x = tf.random.normal((200, 16))
    pool_y = tf.random.uniform((200,), maxval=10, dtype=tf.int32)
    labeled_idx = list(range(20))
    unlabeled_idx = [i for i in range(200) if i not in labeled_idx]

    def uncertainty_sampling(model, x, n_query=10):
        pred = model(x, training=False)
        probs = pred.numpy()
        entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
        return np.argsort(entropy)[-n_query:]

    for round in range(3):
        train_x = tf.gather(pool_x, labeled_idx)
        train_y = tf.gather(pool_y, labeled_idx)
        model.fit(train_x, train_y, epochs=2, verbose=0)
        unlabeled_x = tf.gather(pool_x, unlabeled_idx)
        query_idx = uncertainty_sampling(model, unlabeled_x, n_query=10)
        new_labeled = [unlabeled_idx[i] for i in query_idx]
        labeled_idx.extend(new_labeled)
        unlabeled_idx = [i for i in unlabeled_idx if i not in new_labeled]
        print(f"Round {round + 1}: labeled {len(labeled_idx)}, queried {len(new_labeled)}")

    print("Active learning complete.")

if __name__ == "__main__":
    main()
