"""
Greedy, top-k, top-p/nucleus, temperature sampling.
"""
import tensorflow as tf
import numpy as np

def greedy_sample(logits):
    return tf.argmax(logits, axis=-1)

def top_k_sample(logits, k=10):
    top_k_logits, top_k_indices = tf.math.top_k(logits, k)
    probs = tf.nn.softmax(top_k_logits)
    idx = tf.random.categorical(tf.math.log(probs + 1e-10), 1)[:, 0]
    return tf.gather(top_k_indices, idx, batch_dims=1)

def top_p_sample(logits, p=0.9):
    probs = tf.nn.softmax(logits, axis=-1)
    sorted_probs = tf.sort(probs, direction="DESCENDING")
    cumsum = tf.cumsum(sorted_probs, axis=-1)
    mask = cumsum <= p
    sorted_probs_masked = tf.where(mask, sorted_probs, tf.zeros_like(probs))
    sorted_indices = tf.argsort(probs, direction="DESCENDING")
    inv = tf.argsort(sorted_indices)
    filtered = tf.gather(sorted_probs_masked, inv, batch_dims=1)
    filtered = filtered / (tf.reduce_sum(filtered, axis=-1, keepdims=True) + 1e-10)
    return tf.random.categorical(tf.math.log(filtered + 1e-10), 1)[:, 0]

def temperature_sample(logits, temperature=1.0):
    scaled = logits / temperature
    return tf.random.categorical(scaled, 1)[:, 0]

def main():
    print("=" * 50)
    print("Text Generation Methods")
    print("=" * 50)

    batch_size, vocab_size = 4, 100
    logits = tf.random.normal((batch_size, vocab_size))

    print("\n--- Greedy ---")
    greedy = greedy_sample(logits)
    print(f"Greedy tokens: {greedy.numpy()}")

    print("\n--- Top-k (k=10) ---")
    topk = top_k_sample(logits, k=10)
    print(f"Top-k tokens: {topk.numpy()}")

    print("\n--- Temperature (T=0.5) ---")
    temp_low = temperature_sample(logits, temperature=0.5)
    print(f"Low temp tokens: {temp_low.numpy()}")

    print("\n--- Temperature (T=2.0) ---")
    temp_high = temperature_sample(logits, temperature=2.0)
    print(f"High temp tokens: {temp_high.numpy()}")

    print("\n--- Top-p (p=0.9) ---")
    topp = top_p_sample(logits, p=0.9)
    print(f"Top-p tokens: {topp.numpy()}")

    print("\n--- Autoregressive generation loop ---")
    seq = tf.constant([[1]])
    model_logits = lambda x: tf.random.normal((1, vocab_size))
    for _ in range(3):
        log = model_logits(seq)
        next_tok = temperature_sample(log, temperature=0.8)
        seq = tf.concat([seq, tf.reshape(next_tok, (1, 1))], axis=1)
    print(f"Generated sequence: {seq.numpy()}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
