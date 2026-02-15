"""
Custom optimizer by subclassing tf.keras.optimizers.Optimizer.
"""
import tensorflow as tf

class SimpleSGD(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.01, name="SimpleSGD", **kwargs):
        super().__init__(name=name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr = self._get_hyper("learning_rate", var_dtype)
        m = self.get_slot(var, "m")
        m.assign(0.9 * m + grad)
        var.assign_sub(lr * m)

    def get_config(self):
        config = super().get_config()
        config.update({"learning_rate": self._serialize_hyperparameter("learning_rate")})
        return config

def main():
    opt = SimpleSGD(learning_rate=0.01)
    print(f"SimpleSGD: lr={opt.learning_rate.numpy()}")

    var = tf.Variable(5.0)
    for _ in range(10):
        with tf.GradientTape() as tape:
            loss = (var - 1.0) ** 2
        grad = tape.gradient(loss, var)
        opt.apply_gradients([(grad, var)])
    print(f"SimpleSGD after 10 steps: var={var.numpy():.4f}")

    class SignSGD(tf.keras.optimizers.Optimizer):
        def __init__(self, learning_rate=0.01, name="SignSGD", **kwargs):
            super().__init__(name=name, **kwargs)
            self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))

        def _resource_apply_dense(self, grad, var, apply_state=None):
            var_dtype = var.dtype.base_dtype
            lr = self._get_hyper("learning_rate", var_dtype)
            sign_grad = tf.sign(grad)
            var.assign_sub(lr * sign_grad)

    sign_opt = SignSGD(0.01)
    var2 = tf.Variable(3.0)
    with tf.GradientTape() as tape:
        loss = (var2 - 0.0) ** 2
    grad = tape.gradient(loss, var2)
    sign_opt.apply_gradients([(grad, var2)])
    print(f"SignSGD step: var={var2.numpy():.4f}")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=SimpleSGD(0.01), loss='mse')
    model.fit(tf.random.normal((16, 4)), tf.random.normal((16, 1)), epochs=2, verbose=0)
    print(f"Model trained with SimpleSGD.")

    config = opt.get_config()
    print(f"SimpleSGD config: {config}")
    print("Custom optimizers verified.")

if __name__ == "__main__":
    main()
