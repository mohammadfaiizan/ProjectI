"""
tf.keras pruning API (prune_low_magnitude, PolynomialDecay).
"""
import tensorflow as tf
import tensorflow_model_optimization as tfmot

def main():
    print("=" * 50)
    print("Model Optimization - Pruning")
    print("=" * 50)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=0,
            end_step=100
        )
    }
    pruned_model = prune_low_magnitude(model, **pruning_params)
    pruned_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Pruned model created with PolynomialDecay schedule")

    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    x = tf.random.normal((200, 32))
    y = tf.random.uniform((200,), maxval=10, dtype=tf.int32)
    pruned_model.fit(x, y, epochs=3, callbacks=callbacks, verbose=0)
    print("Training with pruning complete")

    final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    print(f"Stripped model layers: {len(final_model.layers)}")

    for layer in final_model.layers:
        if hasattr(layer, 'kernel'):
            w = layer.kernel.numpy()
            zeros = (w == 0).sum()
            total = w.size
            print(f"  {layer.name}: {zeros}/{total} zeros ({100*zeros/total:.1f}%)")

    print("Pruning demo complete.")

if __name__ == "__main__":
    main()
