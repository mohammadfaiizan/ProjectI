"""
EWC, replay buffers, catastrophic forgetting prevention.
"""
import tensorflow as tf
import numpy as np

def main():
    print("=" * 50)
    print("Continual Learning - EWC and Replay")
    print("=" * 50)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(16,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(0.001)

    task1_x = tf.random.normal((100, 16))
    task1_y = tf.random.uniform((100,), maxval=5, dtype=tf.int32)
    for _ in range(3):
        with tf.GradientTape() as tape:
            pred = model(task1_x, training=True)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(task1_y, pred)
            )
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print("Task 1 trained")

    star_params = [tf.identity(v) for v in model.trainable_variables]
    fisher = []
    for _ in range(5):
        with tf.GradientTape() as tape:
            pred = model(task1_x, training=False)
            log_prob = tf.reduce_sum(tf.math.log(pred + 1e-8) * tf.one_hot(task1_y, 5), axis=1)
            loss = -tf.reduce_mean(log_prob)
        grads = tape.gradient(loss, model.trainable_variables)
        fisher.append([tf.square(g) for g in grads])
    fisher_approx = [tf.reduce_mean(tf.stack([f[i] for f in fisher]), axis=0)
                     for i in range(len(model.trainable_variables))]
    print("Fisher diagonal computed")

    replay_buffer_x = task1_x[:20]
    replay_buffer_y = task1_y[:20]
    task2_x = tf.random.normal((80, 16))
    task2_y = tf.random.uniform((80,), maxval=5, dtype=tf.int32)
    ewc_lambda = 100.0

    for _ in range(2):
        with tf.GradientTape() as tape:
            pred_new = model(task2_x, training=True)
            loss_task = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(task2_y, pred_new)
            )
            pred_replay = model(replay_buffer_x, training=True)
            loss_replay = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(replay_buffer_y, pred_replay)
            )
            loss_ewc = 0.5 * ewc_lambda * sum(
                tf.reduce_sum(f * tf.square(p - s))
                for f, p, s in zip(fisher_approx, model.trainable_variables, star_params)
            )
            loss = loss_task + 0.5 * loss_replay + loss_ewc
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print(f"Task 2 + EWC + replay loss: {loss.numpy():.4f}")
    print("Continual learning complete.")

if __name__ == "__main__":
    main()
