"""
Simple policy network with REINFORCE.
"""
import tensorflow as tf
import numpy as np

def main():
    print("=" * 50)
    print("REINFORCE - Policy Gradient")
    print("=" * 50)

    policy = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(0.001)

    def sample_action(state):
        probs = policy(state[np.newaxis], training=False)[0]
        action = np.random.choice(2, p=probs.numpy())
        return action, probs

    states = tf.random.normal((20, 4))
    actions = tf.random.uniform((20,), maxval=2, dtype=tf.int32)
    rewards = tf.random.normal((20,))
    returns = tf.cumsum(rewards[::-1])[::-1]
    returns = (returns - tf.reduce_mean(returns)) / (tf.math.reduce_std(returns) + 1e-8)

    for _ in range(3):
        with tf.GradientTape() as tape:
            probs = policy(states, training=True)
            action_probs = tf.reduce_sum(
                probs * tf.one_hot(actions, 2), axis=1
            )
            log_probs = tf.math.log(action_probs + 1e-8)
            loss = -tf.reduce_mean(log_probs * returns)
        grads = tape.gradient(loss, policy.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy.trainable_variables))
    print(f"REINFORCE loss: {loss.numpy():.4f}")

    state = tf.random.normal((4,))
    action, probs = sample_action(state)
    print(f"Sampled action: {action}, probs: {probs.numpy()}")
    print("REINFORCE complete.")

if __name__ == "__main__":
    main()
